#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_generate_clean_and_ood_splits.py

Official split-generation script for the FAIR-Lab thesis pipeline.

Purpose
-------
Create the clean binary evaluation folds and the OOD evaluation set from the
final human-reviewed dataset artifacts.

Inputs
------
- datasets/final/manifests/manual_selection_final_1500.csv
- datasets/final/manifests/manual_selection_adversarial_subset.csv

Outputs
-------
- datasets/splits/clean/fold_1/{weapon,non_weapon}/
- ...
- datasets/splits/clean/fold_5/{weapon,non_weapon}/
- datasets/splits/ood/ood_eval_set/ood/
- datasets/splits/manifests/clean_folds_manifest.csv
- datasets/splits/manifests/ood_eval_manifest.csv
- datasets/splits/manifests/split_generation_summary.json

Methodological notes
--------------------
- The binary weapon/non_weapon subset is split into five deterministic folds.
- Each fold contains exactly 100 weapon and 100 non_weapon images.
- The assignment is class-stratified and source-aware: source balance is
  approximated while the class-per-fold constraint is enforced strictly.
- OOD samples are not split into folds. They are copied into a single OOD
  evaluation set and tracked through a dedicated manifest.
- Each copied file is hashed with SHA256 and MD5 to support forensic
  traceability and later matching against forensic tool exports.
- This script does not modify the final manual-selection manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.scripts.utils.paths import (
    CLEAN_SPLITS_DIR,
    FINAL_DATASETS_DIR,
    OOD_SPLITS_DIR,
    PREPARED_DATASETS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)


# =============================================================================
# Configuration
# =============================================================================

N_FOLDS = 5
DEFAULT_SEED = 42

FINAL_MANIFEST_PATH = FINAL_DATASETS_DIR / "manifests" / "manual_selection_final_1500.csv"
ADVERSARIAL_SUBSET_PATH = FINAL_DATASETS_DIR / "manifests" / "manual_selection_adversarial_subset.csv"

PREPARED_FINAL_POOL_DIR = PREPARED_DATASETS_DIR / "final_pool"

OOD_EVAL_SET_DIR = OOD_SPLITS_DIR / "ood_eval_set" / "ood"

CLEAN_FOLDS_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
OOD_EVAL_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "ood_eval_manifest.csv"
SUMMARY_JSON_PATH = SPLIT_MANIFESTS_DIR / "split_generation_summary.json"

VALID_BINARY_LABELS = {"weapon", "non_weapon"}
OOD_LABEL = "ood"


# =============================================================================
# Argument parsing and logging
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic clean binary folds and the OOD evaluation set "
            "from the final manual-selection manifests."
        )
    )
    parser.add_argument(
        "--final-manifest",
        type=str,
        default=str(FINAL_MANIFEST_PATH),
        help=f"Final 1500-image manifest (default: {FINAL_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--adversarial-subset",
        type=str,
        default=str(ADVERSARIAL_SUBSET_PATH),
        help=f"Binary adversarial subset manifest (default: {ADVERSARIAL_SUBSET_PATH})",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help=f"Number of clean folds to generate (default: {N_FOLDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic shuffle seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and rebuild existing split directories and manifests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# =============================================================================
# Generic helpers
# =============================================================================

def norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def compute_hashes(file_path: Path) -> tuple[str, str]:
    """
    Compute SHA256 and MD5 hashes for a file.

    SHA256 is the primary integrity hash for the thesis pipeline. MD5 is also
    stored because several forensic tools can expose or use MD5 values in
    reports and hash sets.
    """
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)

    return sha256.hexdigest(), md5.hexdigest()


def repo_relative_string(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def ensure_required_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {name}: {sorted(missing)}")


def resolve_prepared_image_path(relative_path: str) -> Path:
    """
    Resolve an image path stored in the manual-selection manifests.

    The expected value is usually something like:
        images/img_00000001.jpg

    The source file is therefore resolved under:
        datasets/prepared/final_pool/images/img_00000001.jpg
    """
    rel = Path(relative_path)
    candidates = [
        PREPARED_FINAL_POOL_DIR / rel,
        PREPARED_DATASETS_DIR / rel,
        rel,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return candidates[0]


def clear_outputs(force: bool) -> None:
    """
    Prepare output directories.

    Existing split outputs are removed only when --force is supplied. This avoids
    accidental overwrites of already generated split artifacts.
    """
    output_roots = [CLEAN_SPLITS_DIR, OOD_SPLITS_DIR, SPLIT_MANIFESTS_DIR]

    existing = [path for path in output_roots if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Split outputs already exist. Use --force to rebuild them. Existing paths: "
            + ", ".join(str(path) for path in existing)
        )

    if force:
        for path in output_roots:
            if path.exists():
                logging.warning("Removing existing output path: %s", path)
                shutil.rmtree(path)

    CLEAN_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    OOD_EVAL_SET_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Manifest loading and validation
# =============================================================================

def load_manifest(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def validate_input_manifests(final_df: pd.DataFrame, adversarial_df: pd.DataFrame) -> None:
    required = {
        "image_id",
        "relative_path",
        "source_dataset",
        "source_group",
        "sha256",
        "final_label",
    }

    ensure_required_columns(final_df, required, "manual_selection_final_1500.csv")
    ensure_required_columns(adversarial_df, required, "manual_selection_adversarial_subset.csv")

    final_labels = final_df["final_label"].map(norm)
    adversarial_labels = adversarial_df["final_label"].map(norm)

    expected_final_counts = {"weapon": 500, "non_weapon": 500, "ood": 500}
    final_counts = final_labels.value_counts().to_dict()

    for label, expected_count in expected_final_counts.items():
        actual = int(final_counts.get(label, 0))
        if actual != expected_count:
            raise ValueError(
                f"Unexpected final dataset count for label={label}: "
                f"expected={expected_count}, actual={actual}"
            )

    if len(final_df) != 1500:
        raise ValueError(f"Expected final dataset size 1500, found {len(final_df)}")

    if len(adversarial_df) != 1000:
        raise ValueError(f"Expected adversarial subset size 1000, found {len(adversarial_df)}")

    invalid_adv_labels = set(adversarial_labels.unique()) - VALID_BINARY_LABELS
    if invalid_adv_labels:
        raise ValueError(
            f"Adversarial subset contains non-binary labels: {sorted(invalid_adv_labels)}"
        )

    adv_counts = adversarial_labels.value_counts().to_dict()
    for label in sorted(VALID_BINARY_LABELS):
        actual = int(adv_counts.get(label, 0))
        if actual != 500:
            raise ValueError(f"Expected 500 samples for binary label={label}, found {actual}")

    for manifest_name, df in [("final", final_df), ("adversarial", adversarial_df)]:
        duplicated_image_ids = int(df["image_id"].duplicated().sum())
        duplicated_sha256 = int(df["sha256"].duplicated().sum())

        if duplicated_image_ids:
            raise ValueError(
                f"Duplicated image_id values in {manifest_name} manifest: {duplicated_image_ids}"
            )
        if duplicated_sha256:
            raise ValueError(
                f"Duplicated sha256 values in {manifest_name} manifest: {duplicated_sha256}"
            )


def validate_source_files_exist(df: pd.DataFrame, name: str) -> None:
    missing: list[str] = []

    for _, row in df.iterrows():
        source_path = resolve_prepared_image_path(safe_str(row["relative_path"]))
        if not source_path.exists():
            missing.append(f"{row['image_id']} -> {source_path}")

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing source files in {name}: {len(missing)}\nPreview:\n{preview}"
        )


# =============================================================================
# Fold assignment
# =============================================================================

def assign_source_aware_folds(adversarial_df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Assign deterministic, class-stratified, source-aware folds.

    The previous pure round-robin strategy could create small class imbalance
    when source groups were not divisible by the number of folds. This version
    enforces the hard class constraint first:

        each fold = 100 weapon + 100 non_weapon

    Source balance is approximated as a secondary criterion by assigning each
    sample to the fold with the lowest current count for that source within the
    same class, while never exceeding the class-per-fold quota.
    """
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than 1.")

    df = adversarial_df.copy().reset_index(drop=True)
    df["_label_norm"] = df["final_label"].map(norm)
    df["fold"] = ""

    label_counts_total = df["_label_norm"].value_counts().to_dict()
    target_per_label_per_fold: dict[str, int] = {}

    for label in sorted(VALID_BINARY_LABELS):
        total = int(label_counts_total.get(label, 0))
        if total % n_folds != 0:
            raise ValueError(
                f"Label '{label}' has {total} samples, which is not divisible by n_folds={n_folds}."
            )
        target_per_label_per_fold[label] = total // n_folds

    fold_names = [f"fold_{idx}" for idx in range(1, n_folds + 1)]
    label_fold_counts: dict[str, Counter] = {label: Counter() for label in VALID_BINARY_LABELS}
    source_fold_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for label in sorted(VALID_BINARY_LABELS):
        label_df = df[df["_label_norm"] == label].copy()
        sources = sorted(label_df["source_dataset"].astype(str).unique().tolist())

        for source in sources:
            source_indices = label_df[label_df["source_dataset"].astype(str) == source].index.tolist()
            local_rng = random.Random(f"{seed}:{label}:{source}")
            local_rng.shuffle(source_indices)

            for idx in source_indices:
                eligible_folds = [
                    fold
                    for fold in fold_names
                    if label_fold_counts[label][fold] < target_per_label_per_fold[label]
                ]

                if not eligible_folds:
                    raise RuntimeError(
                        f"No eligible fold available while assigning label={label}, source={source}."
                    )

                # Primary objective: keep each source as evenly distributed as possible.
                # Secondary objective: keep the class counts balanced during assignment.
                chosen_fold = min(
                    eligible_folds,
                    key=lambda fold: (
                        source_fold_counts[(label, source)][fold],
                        label_fold_counts[label][fold],
                        fold,
                    ),
                )

                df.at[idx, "fold"] = chosen_fold
                label_fold_counts[label][chosen_fold] += 1
                source_fold_counts[(label, source)][chosen_fold] += 1

    if (df["fold"].astype(str).str.strip() == "").any():
        missing = df[df["fold"].astype(str).str.strip() == ""]["image_id"].tolist()
        raise RuntimeError(f"Some rows were not assigned to any fold: {missing[:20]}")

    df = df.drop(columns=["_label_norm"])
    df = df.sort_values(["fold", "final_label", "source_dataset", "image_id"], kind="stable")
    df = df.reset_index(drop=True)
    return df


def validate_fold_distribution(df: pd.DataFrame, n_folds: int) -> None:
    expected_per_label_per_fold = 500 // n_folds

    if 500 % n_folds != 0:
        raise ValueError(
            "This script expects 500 samples per binary class to be evenly divisible "
            f"by n_folds={n_folds}."
        )

    for fold_idx in range(1, n_folds + 1):
        fold_name = f"fold_{fold_idx}"
        fold_df = df[df["fold"] == fold_name]

        if fold_df.empty:
            raise ValueError(f"No samples assigned to {fold_name}.")

        label_counts = fold_df["final_label"].map(norm).value_counts().to_dict()
        for label in sorted(VALID_BINARY_LABELS):
            actual = int(label_counts.get(label, 0))
            if actual != expected_per_label_per_fold:
                raise ValueError(
                    f"Invalid distribution in {fold_name} for {label}: "
                    f"expected={expected_per_label_per_fold}, actual={actual}"
                )

        expected_fold_total = expected_per_label_per_fold * len(VALID_BINARY_LABELS)
        if len(fold_df) != expected_fold_total:
            raise ValueError(
                f"Invalid total count in {fold_name}: expected={expected_fold_total}, actual={len(fold_df)}"
            )


# =============================================================================
# Copy operations and manifest construction
# =============================================================================

def copy_with_traceability(src_path: Path, dst_path: Path) -> tuple[str, str, int, str]:
    """Copy a file and return SHA256, MD5, size_bytes, and extension of the copy."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)

    sha256, md5 = compute_hashes(dst_path)
    size_bytes = dst_path.stat().st_size
    extension = dst_path.suffix.lower()

    return sha256, md5, size_bytes, extension


def build_clean_folds(folded_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, row in folded_df.iterrows():
        image_id = safe_str(row["image_id"])
        label = norm(row["final_label"])
        fold = safe_str(row["fold"])

        if label not in VALID_BINARY_LABELS:
            raise ValueError(f"Unexpected binary label for {image_id}: {label}")

        src_path = resolve_prepared_image_path(safe_str(row["relative_path"]))
        dst_path = CLEAN_SPLITS_DIR / fold / label / src_path.name

        copied_sha256, copied_md5, copied_size, copied_extension = copy_with_traceability(
            src_path=src_path,
            dst_path=dst_path,
        )

        expected_sha256 = safe_str(row["sha256"])
        if copied_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch after clean split copy for {image_id}: "
                f"expected={expected_sha256}, copied={copied_sha256}"
            )

        rows.append(
            {
                "image_id": image_id,
                "fold": fold,
                "final_label": label,
                "source_dataset": safe_str(row.get("source_dataset", "")),
                "source_group": safe_str(row.get("source_group", "")),
                "source_relative_path": safe_str(row.get("source_relative_path", "")),
                "source_filename": safe_str(row.get("source_filename", "")),
                "prepared_relative_path": safe_str(row.get("relative_path", "")),
                "split_relative_path": repo_relative_string(dst_path),
                "sha256": copied_sha256,
                "md5": copied_md5,
                "size_bytes": copied_size,
                "extension": copied_extension,
                "sample_type": "clean",
                "attack_family": "none",
                "attack_name": "clean",
            }
        )

    return rows


def build_ood_eval_set(final_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ood_df = final_df[final_df["final_label"].map(norm) == OOD_LABEL].copy()
    ood_df = ood_df.sort_values(["source_dataset", "image_id"], kind="stable")

    if len(ood_df) != 500:
        raise ValueError(f"Expected 500 OOD samples, found {len(ood_df)}")

    for _, row in ood_df.iterrows():
        image_id = safe_str(row["image_id"])
        src_path = resolve_prepared_image_path(safe_str(row["relative_path"]))
        dst_path = OOD_EVAL_SET_DIR / src_path.name

        copied_sha256, copied_md5, copied_size, copied_extension = copy_with_traceability(
            src_path=src_path,
            dst_path=dst_path,
        )

        expected_sha256 = safe_str(row["sha256"])
        if copied_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch after OOD copy for {image_id}: "
                f"expected={expected_sha256}, copied={copied_sha256}"
            )

        rows.append(
            {
                "image_id": image_id,
                "fold": "ood_eval_set",
                "final_label": OOD_LABEL,
                "source_dataset": safe_str(row.get("source_dataset", "")),
                "source_group": safe_str(row.get("source_group", "")),
                "source_relative_path": safe_str(row.get("source_relative_path", "")),
                "source_filename": safe_str(row.get("source_filename", "")),
                "prepared_relative_path": safe_str(row.get("relative_path", "")),
                "ood_relative_path": repo_relative_string(dst_path),
                "sha256": copied_sha256,
                "md5": copied_md5,
                "size_bytes": copied_size,
                "extension": copied_extension,
                "sample_type": "ood",
                "attack_family": "none",
                "attack_name": "ood",
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(clean_rows: list[dict[str, Any]], ood_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    clean_label_counts = Counter(row["final_label"] for row in clean_rows)
    clean_fold_counts: dict[str, Counter] = defaultdict(Counter)
    clean_source_counts: dict[str, Counter] = defaultdict(Counter)

    for row in clean_rows:
        clean_fold_counts[row["fold"]][row["final_label"]] += 1
        clean_source_counts[row["source_dataset"]][row["final_label"]] += 1

    ood_source_counts = Counter(row["source_dataset"] for row in ood_rows)

    summary = {
        "script": "datasets/scripts/splits/11_generate_clean_and_ood_splits.py",
        "seed": args.seed,
        "n_folds": args.n_folds,
        "inputs": {
            "final_manifest": str(repo_relative_path(args.final_manifest)),
            "adversarial_subset": str(repo_relative_path(args.adversarial_subset)),
        },
        "outputs": {
            "clean_splits_dir": str(CLEAN_SPLITS_DIR),
            "ood_splits_dir": str(OOD_SPLITS_DIR),
            "clean_folds_manifest": str(CLEAN_FOLDS_MANIFEST_PATH),
            "ood_eval_manifest": str(OOD_EVAL_MANIFEST_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
        },
        "counts": {
            "clean_total": len(clean_rows),
            "ood_total": len(ood_rows),
            "clean_label_counts": dict(clean_label_counts),
            "clean_fold_counts": {
                fold: dict(counter) for fold, counter in sorted(clean_fold_counts.items())
            },
            "clean_source_counts": {
                source: dict(counter) for source, counter in sorted(clean_source_counts.items())
            },
            "ood_source_counts": dict(sorted(ood_source_counts.items())),
        },
        "checks": {
            "clean_total_1000": len(clean_rows) == 1000,
            "ood_total_500": len(ood_rows) == 500,
            "clean_sha256_unique": len({row["sha256"] for row in clean_rows}) == len(clean_rows),
            "ood_sha256_unique": len({row["sha256"] for row in ood_rows}) == len(ood_rows),
            "clean_image_id_unique": len({row["image_id"] for row in clean_rows}) == len(clean_rows),
            "ood_image_id_unique": len({row["image_id"] for row in ood_rows}) == len(ood_rows),
        },
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    final_manifest = repo_relative_path(args.final_manifest)
    adversarial_subset = repo_relative_path(args.adversarial_subset)

    logging.info("Final manifest: %s", final_manifest)
    logging.info("Adversarial subset: %s", adversarial_subset)
    logging.info("Clean splits dir: %s", CLEAN_SPLITS_DIR)
    logging.info("OOD eval dir: %s", OOD_EVAL_SET_DIR)

    final_df = load_manifest(final_manifest, "Final manual-selection manifest")
    adversarial_df = load_manifest(adversarial_subset, "Adversarial subset manifest")

    validate_input_manifests(final_df=final_df, adversarial_df=adversarial_df)
    validate_source_files_exist(adversarial_df, "adversarial subset")
    validate_source_files_exist(final_df[final_df["final_label"].map(norm) == OOD_LABEL], "OOD subset")

    clear_outputs(force=args.force)

    folded_df = assign_source_aware_folds(
        adversarial_df=adversarial_df,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    validate_fold_distribution(folded_df, n_folds=args.n_folds)

    clean_rows = build_clean_folds(folded_df)
    ood_rows = build_ood_eval_set(final_df)

    write_csv(CLEAN_FOLDS_MANIFEST_PATH, clean_rows)
    write_csv(OOD_EVAL_MANIFEST_PATH, ood_rows)
    write_summary(clean_rows=clean_rows, ood_rows=ood_rows, args=args)

    logging.info("Clean folds manifest: %s", CLEAN_FOLDS_MANIFEST_PATH)
    logging.info("OOD eval manifest: %s", OOD_EVAL_MANIFEST_PATH)
    logging.info("Summary JSON: %s", SUMMARY_JSON_PATH)
    logging.info("Split generation completed successfully.")


if __name__ == "__main__":
    main()
