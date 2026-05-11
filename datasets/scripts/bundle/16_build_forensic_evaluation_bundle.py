#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_build_forensic_evaluation_bundle.py

Build the FAIR-Lab forensic evaluation bundle.

The bundle is the operational bridge between the academic pipeline and
commercial forensic-tool testing. It collects clean, OOD, adversarial and
anti-forensic artifacts into a stable, hash-traceable corpus.

Bias-control principle
----------------------
Commercial forensic tools and human analysts must not receive paths or filenames
that reveal the ground-truth class, attack family, attack name, fold, source
model, or OOD status. For this reason, the default bundle contains:

1. metadata/
   Internal manifests, hashes and summaries. These files preserve all labels,
   source information, perturbation metadata and hash mappings. They must not be
   imported into the forensic tools during blind evaluation.

2. blind_tool_input/files/
   A flat anonymized directory intended as the only input folder for Magnet
   AXIOM, X-Ways / Excire, Cellebrite UFED, Oxygen, or any other black-box
   forensic tool. Files are named only as bundle_000001.jpg, bundle_000002.png,
   etc.

3. structured_audit_view/
   Optional structured copy for internal audit/debug only. This view keeps the
   semantic folder hierarchy and must not be used as forensic-tool input.

This script does not run Magnet AXIOM, X-Ways, Cellebrite UFED or Oxygen.
It only prepares the input corpus and traceability manifests required for later
black-box forensic-tool evaluation.

Default outputs
---------------
datasets/forensic_evaluation_bundle/
├── metadata/
│   ├── bundle_manifest.csv
│   ├── bundle_hashes_sha256.csv
│   └── bundle_summary.json
├── blind_tool_input/
│   └── files/
│       ├── bundle_000001.jpg
│       ├── bundle_000002.png
│       └── ...
└── structured_audit_view/
    ├── clean/
    ├── ood/
    ├── adversarial/
    └── anti_forensic/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.utils.paths import (
    ATTACKS_DIR,
    DATASETS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)

SCRIPT_NAME = "datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py"

DEFAULT_CLEAN_MANIFEST = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
DEFAULT_OOD_MANIFEST = SPLIT_MANIFESTS_DIR / "ood_eval_manifest.csv"
DEFAULT_ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"
DEFAULT_BUNDLE_DIR = DATASETS_DIR / "forensic_evaluation_bundle"

SEMANTIC_TOKENS_FORBIDDEN_IN_BLIND_PATHS = {
    "weapon",
    "non_weapon",
    "ood",
    "clean",
    "adversarial",
    "anti_forensic",
    "fgsm",
    "one_pixel",
    "sigma_zero",
    "superdeepfool",
    "color_shift",
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
    "efficientnet_b0",
    "resnet18",
    "clip",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


def repo_relative_string(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def maybe_value(row: pd.Series, *names: str, default: str = "") -> str:
    for name in names:
        if name in row.index:
            value = safe_str(row.get(name, ""))
            if value:
                return value
    return default


def first_existing_column(df: pd.DataFrame, candidates: list[str], manifest_name: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing path column in {manifest_name}. Candidates: {candidates}")


def compute_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FAIR-Lab forensic evaluation bundle.")
    parser.add_argument("--clean-manifest", default=str(DEFAULT_CLEAN_MANIFEST))
    parser.add_argument("--ood-manifest", default=str(DEFAULT_OOD_MANIFEST))
    parser.add_argument("--attack-manifests-dir", default=str(DEFAULT_ATTACK_MANIFESTS_DIR))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument(
        "--layout",
        choices=("blind", "structured", "both"),
        default="both",
        help=(
            "Bundle layout to materialize. 'blind' creates only the anonymized flat "
            "tool input; 'structured' creates only the internal audit view; 'both' "
            "creates both. Default: both."
        ),
    )
    parser.add_argument("--copy-files", dest="copy_files", action="store_true", default=True)
    parser.add_argument("--no-copy-files", dest="copy_files", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


def load_clean_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])
        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "clean",
            "attack_family": "none",
            "attack_name": "clean",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row["split_relative_path"]),
            "original_sha256": safe_str(row.get("sha256", "")),
            "original_md5": safe_str(row.get("md5", "")),
            "input_sha256_manifest": safe_str(row.get("sha256", "")),
            "input_md5_manifest": safe_str(row.get("md5", "")),
        })
    return rows


def load_ood_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])
        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "ood",
            "attack_family": "none",
            "attack_name": "ood",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": "ood_eval_set",
            "final_label": "ood",
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row["ood_relative_path"]),
            "original_sha256": safe_str(row.get("sha256", "")),
            "original_md5": safe_str(row.get("md5", "")),
            "input_sha256_manifest": safe_str(row.get("sha256", "")),
            "input_md5_manifest": safe_str(row.get("md5", "")),
        })
    return rows


def load_attack_rows(path: Path, expected_family: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    path_col = first_existing_column(
        df,
        ["perturbed_relative_path", "adversarial_relative_path", "generated_relative_path", "image_relative_path"],
        path.name,
    )
    rows = []
    for _, row in df.iterrows():
        original_id = maybe_value(row, "original_image_id", "image_id")
        attack_name = maybe_value(row, "attack_name", default=path.stem.replace("_manifest", ""))
        target_model = maybe_value(row, "target_model", "attack_target_model")
        if not target_model:
            target_model = "model_agnostic" if attack_name == "color_shift" else "unknown"
        generated_id = maybe_value(row, "generated_image_id", default=f"{original_id}__{attack_name}")
        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "perturbed",
            "attack_family": maybe_value(row, "attack_family", default=expected_family),
            "attack_name": attack_name,
            "attack_target_model": target_model,
            "original_image_id": original_id,
            "generated_image_id": generated_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row[path_col]),
            "original_sha256": maybe_value(row, "sha256_original", "original_sha256", "sha256"),
            "original_md5": maybe_value(row, "md5_original", "original_md5", "md5"),
            "input_sha256_manifest": maybe_value(row, "sha256_perturbed", "perturbed_sha256", "sha256"),
            "input_md5_manifest": maybe_value(row, "md5_perturbed", "perturbed_md5", "md5"),
        })
    return rows


def discover_adversarial_manifests(manifests_dir: Path) -> list[Path]:
    return sorted(
        p for p in manifests_dir.glob("adversarial_*_manifest.csv")
        if "summary" not in p.name and "evaluation" not in p.name
    )


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    rows.extend(load_clean_rows(repo_relative_path(args.clean_manifest)))
    rows.extend(load_ood_rows(repo_relative_path(args.ood_manifest)))
    manifests_dir = repo_relative_path(args.attack_manifests_dir)
    for manifest in discover_adversarial_manifests(manifests_dir):
        loaded = load_attack_rows(manifest, "adversarial")
        logging.info("Loaded %s: %d", manifest.name, len(loaded))
        rows.extend(loaded)
    anti_manifest = manifests_dir / "anti_forensic_attacks_manifest.csv"
    if anti_manifest.exists():
        loaded = load_attack_rows(anti_manifest, "anti_forensic")
        logging.info("Loaded %s: %d", anti_manifest.name, len(loaded))
        rows.extend(loaded)
    if args.limit > 0:
        rows = rows[: args.limit]
        logging.warning("Limit active: %d rows", len(rows))
    return rows


def blind_subpath(bundle_id: str, src_path: Path) -> Path:
    ext = src_path.suffix.lower() or ".img"
    return Path("blind_tool_input") / "files" / f"{bundle_id}{ext}"


def structured_subpath(row: dict[str, Any], bundle_id: str, src_path: Path) -> Path:
    ext = src_path.suffix.lower() or ".img"
    if row["sample_type"] == "clean":
        subdir = Path("structured_audit_view") / "clean" / row["fold"] / row["final_label"]
    elif row["sample_type"] == "ood":
        subdir = Path("structured_audit_view") / "ood" / "ood_eval_set"
    elif row["attack_family"] == "adversarial":
        subdir = (
            Path("structured_audit_view")
            / "adversarial"
            / row["attack_name"]
            / row["attack_target_model"]
            / row["fold"]
            / row["final_label"]
        )
    elif row["attack_family"] == "anti_forensic":
        subdir = (
            Path("structured_audit_view")
            / "anti_forensic"
            / row["attack_name"]
            / row["fold"]
            / row["final_label"]
        )
    else:
        subdir = Path("structured_audit_view") / "other"
    filename = safe_filename(f"{bundle_id}__{row['generated_image_id']}{ext}")
    return subdir / filename


def should_create_blind(layout: str) -> bool:
    return layout in {"blind", "both"}


def should_create_structured(layout: str) -> bool:
    return layout in {"structured", "both"}


def materialize(
    row: dict[str, Any],
    bundle_dir: Path,
    index: int,
    created_at: str,
    copy_files: bool,
    layout: str,
) -> dict[str, Any]:
    bundle_id = f"bundle_{index:06d}"
    src_path = resolve_repo_path(row["input_relative_path"])
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found for {bundle_id}: {src_path}")

    blind_relative_path = ""
    structured_relative_path = ""
    tool_input_filename = ""
    hash_path = src_path

    if copy_files and should_create_blind(layout):
        blind_path = bundle_dir / blind_subpath(bundle_id, src_path)
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, blind_path)
        blind_relative_path = repo_relative_string(blind_path)
        tool_input_filename = blind_path.name
        hash_path = blind_path

    if copy_files and should_create_structured(layout):
        structured_path = bundle_dir / structured_subpath(row, bundle_id, src_path)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, structured_path)
        structured_relative_path = repo_relative_string(structured_path)
        if not should_create_blind(layout):
            hash_path = structured_path

    sha256, md5 = compute_hashes(hash_path)
    expected = row.get("input_sha256_manifest", "")

    return {
        "bundle_id": bundle_id,
        "tool_input_filename": tool_input_filename,
        "sample_type": row["sample_type"],
        "attack_family": row["attack_family"],
        "attack_name": row["attack_name"],
        "attack_target_model": row["attack_target_model"],
        "fold": row["fold"],
        "final_label": row["final_label"],
        "source_dataset": row["source_dataset"],
        "original_image_id": row["original_image_id"],
        "generated_image_id": row["generated_image_id"],
        "source_manifest": row["source_manifest"],
        "source_relative_path": row["input_relative_path"],
        "blind_relative_path": blind_relative_path,
        "structured_relative_path": structured_relative_path,
        "tool_input_relative_path": blind_relative_path,
        "original_sha256": row.get("original_sha256", ""),
        "original_md5": row.get("original_md5", ""),
        "sha256_manifest": expected,
        "md5_manifest": row.get("input_md5_manifest", ""),
        "sha256_actual": sha256,
        "md5_actual": md5,
        "sha256_matches_manifest": bool(expected) and expected == sha256,
        "size_bytes": hash_path.stat().st_size,
        "extension": hash_path.suffix.lower(),
        "layout": layout,
        "created_at": created_at,
    }


def clear_bundle(bundle_dir: Path, force: bool) -> None:
    if bundle_dir.exists():
        if not force:
            raise FileExistsError(f"Bundle directory already exists. Use --force: {bundle_dir}")
        logging.warning("Removing existing bundle directory: %s", bundle_dir)
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def hash_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "bundle_id": row["bundle_id"],
        "sha256": row["sha256_actual"],
        "md5": row["md5_actual"],
        "tool_input_filename": row["tool_input_filename"],
        "tool_input_relative_path": row["tool_input_relative_path"],
        "blind_relative_path": row["blind_relative_path"],
        "structured_relative_path": row["structured_relative_path"],
        "sample_type": row["sample_type"],
        "attack_family": row["attack_family"],
        "attack_name": row["attack_name"],
        "final_label": row["final_label"],
        "original_image_id": row["original_image_id"],
        "generated_image_id": row["generated_image_id"],
    } for row in rows]


def blind_paths_are_semantically_clean(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        path_value = str(row.get("tool_input_relative_path", "")).lower()
        if not path_value:
            continue
        filename = Path(path_value).name.lower()
        stem = Path(filename).stem.lower()
        if stem != row["bundle_id"].lower():
            return False
        for token in SEMANTIC_TOKENS_FORBIDDEN_IN_BLIND_PATHS:
            if token in filename:
                return False
    return True


def build_summary(rows: list[dict[str, Any]], bundle_dir: Path, created_at: str, layout: str) -> dict[str, Any]:
    metadata_dir = bundle_dir / "metadata"
    blind_dir = bundle_dir / "blind_tool_input" / "files"
    structured_dir = bundle_dir / "structured_audit_view"

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "bundle_dir": repo_relative_string(bundle_dir),
        "layout": layout,
        "outputs": {
            "metadata_dir": repo_relative_string(metadata_dir),
            "bundle_manifest_csv": repo_relative_string(metadata_dir / "bundle_manifest.csv"),
            "bundle_hashes_sha256_csv": repo_relative_string(metadata_dir / "bundle_hashes_sha256.csv"),
            "bundle_summary_json": repo_relative_string(metadata_dir / "bundle_summary.json"),
            "blind_tool_input_dir": repo_relative_string(blind_dir),
            "structured_audit_view_dir": repo_relative_string(structured_dir),
        },
        "counts": {
            "total_bundle_rows": len(rows),
            "by_sample_type": dict(Counter(row["sample_type"] for row in rows)),
            "by_attack_family": dict(Counter(row["attack_family"] for row in rows)),
            "by_attack_name": dict(Counter(row["attack_name"] for row in rows)),
            "by_label": dict(Counter(row["final_label"] for row in rows)),
            "blind_files": sum(1 for row in rows if row["blind_relative_path"]),
            "structured_files": sum(1 for row in rows if row["structured_relative_path"]),
        },
        "checks": {
            "bundle_id_unique": len({row["bundle_id"] for row in rows}) == len(rows),
            "sha256_actual_unique": len({row["sha256_actual"] for row in rows}) == len(rows),
            "all_sha256_match_when_manifest_present": all(row["sha256_matches_manifest"] or not row["sha256_manifest"] for row in rows),
            "blind_paths_semantically_clean": blind_paths_are_semantically_clean(rows),
            "metadata_separated_from_tool_input": True,
        },
        "tool_input_instruction": (
            "For black-box forensic-tool evaluation, import only the directory "
            "datasets/forensic_evaluation_bundle/blind_tool_input/files. Do not import "
            "metadata/ or structured_audit_view/ into forensic tools."
        ),
        "methodological_note": (
            "The blind flat layout is designed to reduce path-induced and analyst-induced bias. "
            "Ground truth labels, perturbation metadata, source information and hash mappings are "
            "preserved only in metadata manifests for post-export normalization."
        ),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    bundle_dir = repo_relative_path(args.bundle_dir)
    clear_bundle(bundle_dir, args.force)
    source_rows = collect_rows(args)
    created_at = utc_now_iso()
    bundle_rows = []
    for index, row in enumerate(source_rows, start=1):
        bundle_rows.append(materialize(row, bundle_dir, index, created_at, args.copy_files, args.layout))
        if index % 500 == 0:
            logging.info("Materialized %d/%d", index, len(source_rows))

    metadata_dir = bundle_dir / "metadata"
    write_csv(metadata_dir / "bundle_manifest.csv", bundle_rows)
    write_csv(metadata_dir / "bundle_hashes_sha256.csv", hash_rows(bundle_rows))
    write_json(metadata_dir / "bundle_summary.json", build_summary(bundle_rows, bundle_dir, created_at, args.layout))
    logging.info("Bundle written: %s", bundle_dir)
    logging.info("Tool input directory: %s", bundle_dir / "blind_tool_input" / "files")
    logging.info("Metadata directory: %s", metadata_dir)


if __name__ == "__main__":
    main()
