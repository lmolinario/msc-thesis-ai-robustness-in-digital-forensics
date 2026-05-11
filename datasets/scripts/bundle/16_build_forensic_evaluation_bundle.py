#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_build_forensic_evaluation_bundle.py

Build the FAIR-Lab forensic evaluation bundle.

The bundle is the operational bridge between the academic pipeline and
commercial forensic-tool testing. It collects clean, OOD, adversarial and
anti-forensic artifacts into a stable, hash-traceable directory structure.

This script does not run Magnet AXIOM, X-Ways, Cellebrite UFED or Oxygen.
It only prepares the input corpus and traceability manifests required for
later black-box forensic-tool evaluation.

Outputs:
- datasets/forensic_evaluation_bundle/bundle_manifest.csv
- datasets/forensic_evaluation_bundle/bundle_hashes_sha256.csv
- datasets/forensic_evaluation_bundle/bundle_summary.json
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
    parser.add_argument("--copy-files", action="store_true", default=True)
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


def bundle_subpath(row: dict[str, Any], bundle_id: str, src_path: Path) -> Path:
    ext = src_path.suffix.lower() or ".img"
    if row["sample_type"] == "clean":
        subdir = Path("clean") / row["fold"] / row["final_label"]
    elif row["sample_type"] == "ood":
        subdir = Path("ood") / "ood_eval_set"
    elif row["attack_family"] == "adversarial":
        subdir = Path("adversarial") / row["attack_name"] / row["attack_target_model"] / row["fold"] / row["final_label"]
    elif row["attack_family"] == "anti_forensic":
        subdir = Path("anti_forensic") / row["attack_name"] / row["fold"] / row["final_label"]
    else:
        subdir = Path("other")
    filename = safe_filename(f"{bundle_id}__{row['generated_image_id']}{ext}")
    return subdir / filename


def materialize(row: dict[str, Any], bundle_dir: Path, index: int, created_at: str, copy_files: bool) -> dict[str, Any]:
    bundle_id = f"bundle_{index:06d}"
    src_path = resolve_repo_path(row["input_relative_path"])
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found for {bundle_id}: {src_path}")
    rel_bundle_path = bundle_subpath(row, bundle_id, src_path)
    dst_path = bundle_dir / rel_bundle_path
    if copy_files:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        hash_path = dst_path
    else:
        hash_path = src_path
    sha256, md5 = compute_hashes(hash_path)
    expected = row.get("input_sha256_manifest", "")
    return {
        "bundle_id": bundle_id,
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
        "bundle_relative_path": repo_relative_string(dst_path) if copy_files else "",
        "original_sha256": row.get("original_sha256", ""),
        "original_md5": row.get("original_md5", ""),
        "sha256_manifest": expected,
        "md5_manifest": row.get("input_md5_manifest", ""),
        "sha256_actual": sha256,
        "md5_actual": md5,
        "sha256_matches_manifest": bool(expected) and expected == sha256,
        "size_bytes": hash_path.stat().st_size,
        "extension": hash_path.suffix.lower(),
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
        "bundle_relative_path": row["bundle_relative_path"],
        "sample_type": row["sample_type"],
        "attack_family": row["attack_family"],
        "attack_name": row["attack_name"],
        "final_label": row["final_label"],
        "original_image_id": row["original_image_id"],
        "generated_image_id": row["generated_image_id"],
    } for row in rows]


def build_summary(rows: list[dict[str, Any]], bundle_dir: Path, created_at: str) -> dict[str, Any]:
    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "bundle_dir": repo_relative_string(bundle_dir),
        "outputs": {
            "bundle_manifest_csv": repo_relative_string(bundle_dir / "bundle_manifest.csv"),
            "bundle_hashes_sha256_csv": repo_relative_string(bundle_dir / "bundle_hashes_sha256.csv"),
            "bundle_summary_json": repo_relative_string(bundle_dir / "bundle_summary.json"),
        },
        "counts": {
            "total_bundle_rows": len(rows),
            "by_sample_type": dict(Counter(row["sample_type"] for row in rows)),
            "by_attack_family": dict(Counter(row["attack_family"] for row in rows)),
            "by_attack_name": dict(Counter(row["attack_name"] for row in rows)),
            "by_label": dict(Counter(row["final_label"] for row in rows)),
        },
        "checks": {
            "bundle_id_unique": len({row["bundle_id"] for row in rows}) == len(rows),
            "sha256_actual_unique": len({row["sha256_actual"] for row in rows}) == len(rows),
            "all_sha256_match_when_manifest_present": all(row["sha256_matches_manifest"] or not row["sha256_manifest"] for row in rows),
        },
        "methodological_note": "The bundle is an operational input corpus for forensic-tool evaluation and does not contain forensic-tool results.",
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
        bundle_rows.append(materialize(row, bundle_dir, index, created_at, args.copy_files))
        if index % 500 == 0:
            logging.info("Materialized %d/%d", index, len(source_rows))
    write_csv(bundle_dir / "bundle_manifest.csv", bundle_rows)
    write_csv(bundle_dir / "bundle_hashes_sha256.csv", hash_rows(bundle_rows))
    write_json(bundle_dir / "bundle_summary.json", build_summary(bundle_rows, bundle_dir, created_at))
    logging.info("Bundle written: %s", bundle_dir)


if __name__ == "__main__":
    main()
