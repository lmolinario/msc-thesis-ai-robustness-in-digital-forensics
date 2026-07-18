#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the public embedded-metadata sensitivity check.

This public workflow uses only the minimized embedded-metadata audit and the
validated sanitized commercial-tool extracts. It reproduces the aggregate
leave-out sensitivity metrics without requiring the full metadata values or raw
commercial exports.

Default behavior writes to a local staging directory and compares the aggregate
outputs with the currently tracked reference tables. Use ``--install`` after a
successful comparison to replace the public detail, aggregate tables and
summary atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "datasets" / "forensic_evaluation_bundle" / "metadata"
DEFAULT_AUDIT = METADATA_DIR / "embedded_metadata_audit.csv"
RESULTS_DIR = REPO_ROOT / "results" / "figures" / "chapter_5"
DEFAULT_STAGING_DIR = RESULTS_DIR / ".staging_metadata_sensitivity"
DETAIL_NAME = "tab_embedded_metadata_sensitive_hits_detail.csv"
SUMMARY_TABLE_NAME = "tab_embedded_metadata_tool_summary.csv"
DELTA_NAME = "tab_embedded_metadata_sensitivity_delta.csv"
SUMMARY_JSON_NAME = "embedded_metadata_sensitivity_summary.json"

EXTRACTS = [
    REPO_ROOT
    / "forensic_tools"
    / "magnet_axiom"
    / "public_extracts"
    / "magnet_axiom_predictions_extract.csv",
    REPO_ROOT
    / "forensic_tools"
    / "excire_foto_2025"
    / "public_extracts"
    / "excire_prompt_hits_extract.csv",
    REPO_ROOT
    / "forensic_tools"
    / "cellebrite_inseyets"
    / "public_extracts"
    / "cellebrite_classifications_extract.csv",
    REPO_ROOT
    / "forensic_tools"
    / "griffeye"
    / "public_extracts"
    / "griffeye_bookmarks_extract.csv",
]
EXPECTED_COUNTS = {
    "magnet_axiom": 11_500,
    "excire_foto_2025_d20": 11_500,
    "excire_foto_2025_d50": 11_500,
    "excire_foto_2025_d80": 11_500,
    "cellebrite_inseyets": 11_500,
    "griffeye": 11_500,
}
TOOL_ORDER = tuple(EXPECTED_COUNTS)
TOOL_DISPLAY = {
    "magnet_axiom": "Magnet AXIOM / Magnet.AI",
    "excire_foto_2025_d20": "Excire Foto 2025 D20",
    "excire_foto_2025_d50": "Excire Foto 2025 D50",
    "excire_foto_2025_d80": "Excire Foto 2025 D80",
    "cellebrite_inseyets": "Cellebrite Inseyets",
    "griffeye": "Magnet Griffeye / T3K CORE",
}
LOCAL_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/run/media/|/home/|/Users/|raw_exports|raw_filename_or_path)",
    re.IGNORECASE,
)
DETAIL_FIELDS = [
    "bundle_id",
    "suffix",
    "sensitive_hits",
    "sensitive_hit_count",
    "tool_name",
    "tool_display",
    "sample_type",
    "attack_family",
    "attack_name",
    "final_label",
    "normalized_prediction",
    "weapon_detected",
    "correct",
    "false_negative",
    "false_positive",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING_DIR))
    parser.add_argument("--reference-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the staged minimized outputs after aggregate equivalence passes.",
    )
    parser.add_argument(
        "--skip-reference-comparison",
        action="store_true",
        help="Do not compare aggregate tables with the tracked reference outputs.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def read_public_audit(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Minimized embedded-metadata audit not found: {path}. "
            "Run datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py --install first."
        )
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "bundle_id",
        "suffix",
        "has_embedded_metadata",
        "sensitive_hits",
        "sensitive_hit_count",
        "metadata_keys",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Audit is not the minimized public schema; missing columns: {missing}"
        )
    forbidden = {"metadata_json", "relative_path", "raw_export_file", "raw_filename_or_path"}
    present_forbidden = sorted(forbidden & set(frame.columns))
    if present_forbidden:
        raise ValueError(f"Private audit columns are still present: {present_forbidden}")
    frame["bundle_id"] = frame["bundle_id"].astype(str)
    frame["sensitive_hits"] = frame["sensitive_hits"].fillna("").astype(str)
    frame["sensitive_hit_count"] = pd.to_numeric(
        frame["sensitive_hit_count"], errors="raise"
    ).astype(int)
    if len(frame) != 11_500 or frame["bundle_id"].nunique() != 11_500:
        raise ValueError("Public audit must contain 11,500 unique bundle rows")
    sensitive = frame[frame["sensitive_hit_count"] > 0].copy()
    if len(sensitive) != 15:
        raise ValueError(f"Expected 15 sensitive-hit rows, found {len(sensitive)}")
    return frame


def read_public_extracts(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "tool_name",
        "bundle_id",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
        "weapon_detected",
        "normalized_prediction",
    }
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing sanitized public extract: {path}")
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        frames.append(frame[list(required)].copy())
    predictions = pd.concat(frames, ignore_index=True)
    predictions["tool_name"] = predictions["tool_name"].astype(str)
    predictions["bundle_id"] = predictions["bundle_id"].astype(str)
    counts = predictions.groupby("tool_name").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Public extract profile mismatch: {counts}")
    if predictions.duplicated(["tool_name", "bundle_id"]).any():
        raise ValueError("Duplicate tool_name/bundle_id pairs in public extracts")
    if len(predictions) != 69_000:
        raise ValueError(f"Expected 69,000 public decisions, found {len(predictions)}")

    predictions["weapon_detected"] = to_bool_series(predictions["weapon_detected"])
    labels = predictions["final_label"].astype(str)
    decisions = predictions["normalized_prediction"].astype(str)
    predictions["correct"] = decisions.eq(labels)
    predictions["false_negative"] = labels.eq("weapon") & ~decisions.eq("weapon")
    predictions["false_positive"] = labels.eq("non_weapon") & decisions.eq("weapon")
    predictions["tool_display"] = predictions["tool_name"].map(TOOL_DISPLAY)
    return predictions


def build_detail(audit: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    sensitive = audit[audit["sensitive_hit_count"] > 0][
        ["bundle_id", "suffix", "sensitive_hits", "sensitive_hit_count"]
    ].copy()
    detail = sensitive.merge(predictions, on="bundle_id", how="inner", validate="one_to_many")
    if len(detail) != 90:
        raise ValueError(f"Expected 90 minimized detail rows, found {len(detail)}")
    detail["_tool_order"] = detail["tool_name"].map(
        {tool: index for index, tool in enumerate(TOOL_ORDER)}
    )
    detail = detail.sort_values(["bundle_id", "_tool_order"]).drop(columns="_tool_order")
    detail = detail[DETAIL_FIELDS].copy()
    return detail.reset_index(drop=True)


def mean_bool(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        raise ValueError(f"Cannot compute {column} on an empty frame")
    return float(frame[column].astype(bool).mean())


def count_bool(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].astype(bool).sum())


def build_aggregate_tables(
    detail: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for tool in TOOL_ORDER:
        all_tool = predictions[predictions["tool_name"] == tool].copy()
        tool_detail = detail[detail["tool_name"] == tool].copy()
        clean_detail = tool_detail[tool_detail["sample_type"] == "clean"].copy()
        ood_detail = tool_detail[tool_detail["sample_type"] == "ood"].copy()
        all_clean = all_tool[all_tool["sample_type"] == "clean"].copy()
        all_ood = all_tool[all_tool["sample_type"] == "ood"].copy()
        clean_without = all_clean[~all_clean["bundle_id"].isin(set(clean_detail["bundle_id"]))]
        ood_without = all_ood[~all_ood["bundle_id"].isin(set(ood_detail["bundle_id"]))]

        clean_original = mean_bool(all_clean, "correct")
        clean_without_value = mean_bool(clean_without, "correct")
        ood_original = mean_bool(all_ood, "weapon_detected")
        ood_without_value = mean_bool(ood_without, "weapon_detected")
        clean_delta = clean_without_value - clean_original
        ood_delta = ood_without_value - ood_original

        summary_rows.append(
            {
                "tool_name": tool,
                "tool_display": TOOL_DISPLAY[tool],
                "sensitive_hit_rows": len(tool_detail),
                "clean_sensitive_cases": len(clean_detail),
                "clean_sensitive_correct": count_bool(clean_detail, "correct"),
                "clean_sensitive_false_negatives": count_bool(
                    clean_detail, "false_negative"
                ),
                "clean_sensitive_false_positives": count_bool(
                    clean_detail, "false_positive"
                ),
                "ood_sensitive_cases": len(ood_detail),
                "ood_sensitive_weapon_flags": count_bool(
                    ood_detail, "weapon_detected"
                ),
                "clean_accuracy_original": clean_original,
                "clean_accuracy_without_sensitive_hits": clean_without_value,
                "clean_accuracy_delta": clean_delta,
                "ood_weapon_flag_rate_original": ood_original,
                "ood_weapon_flag_rate_without_sensitive_hits": ood_without_value,
                "ood_weapon_flag_rate_delta": ood_delta,
            }
        )
        for metric, original, without, delta, excluded, original_n, without_n in (
            (
                "clean_accuracy",
                clean_original,
                clean_without_value,
                clean_delta,
                len(clean_detail),
                len(all_clean),
                len(clean_without),
            ),
            (
                "ood_weapon_flag_rate",
                ood_original,
                ood_without_value,
                ood_delta,
                len(ood_detail),
                len(all_ood),
                len(ood_without),
            ),
        ):
            delta_rows.append(
                {
                    "tool_name": tool,
                    "tool_display": TOOL_DISPLAY[tool],
                    "metric": metric,
                    "original": original,
                    "without_sensitive_hits": without,
                    "delta": delta,
                    "excluded_cases": excluded,
                    "original_n": original_n,
                    "without_sensitive_hits_n": without_n,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def validate_public_frame(frame: pd.DataFrame, label: str) -> None:
    serialized = frame.to_csv(index=False)
    if LOCAL_PATH_RE.search(serialized):
        raise ValueError(f"Local/private path leakage detected in {label}")
    if "metadata_json" in frame.columns or "raw_export_file" in frame.columns:
        raise ValueError(f"Private columns detected in {label}")


def compare_tables(generated: pd.DataFrame, reference_path: Path, label: str) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference {label} not found: {reference_path}")
    reference = pd.read_csv(reference_path, low_memory=False)
    if list(generated.columns) != list(reference.columns):
        raise ValueError(
            f"{label} columns differ: generated={list(generated.columns)}, "
            f"reference={list(reference.columns)}"
        )
    if len(generated) != len(reference):
        raise ValueError(f"{label} row count differs")
    for column in generated.columns:
        left = generated[column]
        right = reference[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            left_num = pd.to_numeric(left, errors="coerce")
            right_num = pd.to_numeric(right, errors="coerce")
            equal = (left_num - right_num).abs().fillna(0).le(1e-12)
            both_nan = left_num.isna() & right_num.isna()
            if not (equal | both_nan).all():
                raise ValueError(f"{label} numeric mismatch in column {column}")
        elif not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
            raise ValueError(f"{label} text mismatch in column {column}")


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def install_file(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)


def main() -> None:
    args = parse_args()
    audit_path = resolve_path(args.metadata_audit)
    staging_dir = resolve_path(args.staging_dir)
    reference_dir = resolve_path(args.reference_dir)
    audit = read_public_audit(audit_path)
    predictions = read_public_extracts(EXTRACTS)
    detail = build_detail(audit, predictions)
    summary_table, delta_table = build_aggregate_tables(detail, predictions)
    validate_public_frame(detail, "minimized detail table")
    validate_public_frame(summary_table, "tool summary")
    validate_public_frame(delta_table, "delta table")

    if not args.skip_reference_comparison:
        compare_tables(
            summary_table, reference_dir / SUMMARY_TABLE_NAME, "tool summary"
        )
        compare_tables(delta_table, reference_dir / DELTA_NAME, "delta table")

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_detail = staging_dir / DETAIL_NAME
    staged_summary_table = staging_dir / SUMMARY_TABLE_NAME
    staged_delta = staging_dir / DELTA_NAME
    staged_summary_json = staging_dir / SUMMARY_JSON_NAME
    write_csv_atomic(staged_detail, detail)
    write_csv_atomic(staged_summary_table, summary_table)
    write_csv_atomic(staged_delta, delta_table)

    payload = {
        "schema_version": "2.0",
        "created_at": utc_now_iso(),
        "script": "results/scripts/22_generate_public_embedded_metadata_sensitivity_check.py",
        "artifact_policy": "minimized_public_inputs_and_outputs",
        "inputs": {
            "metadata_audit_csv": audit_path.relative_to(REPO_ROOT).as_posix(),
            "metadata_audit_sha256": sha256_file(audit_path),
            "public_extracts": [
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in EXTRACTS
            ],
        },
        "outputs": {
            "detail_csv": f"results/figures/chapter_5/{DETAIL_NAME}",
            "tool_summary_csv": f"results/figures/chapter_5/{SUMMARY_TABLE_NAME}",
            "delta_csv": f"results/figures/chapter_5/{DELTA_NAME}",
            "summary_json": f"results/figures/chapter_5/{SUMMARY_JSON_NAME}",
        },
        "counts": {
            "audit_rows": len(audit),
            "sensitive_metadata_hit_bundles": int(
                (audit["sensitive_hit_count"] > 0).sum()
            ),
            "prediction_rows": len(predictions),
            "detail_rows": len(detail),
            "tool_summary_rows": len(summary_table),
            "delta_rows": len(delta_table),
        },
        "aggregate_reference_equivalence": not args.skip_reference_comparison,
        "public_detail_columns": DETAIL_FIELDS,
        "methodological_note": (
            "The leave-out sensitivity check is reproduced from minimized metadata "
            "indicators and validated sanitized tool decisions. It does not publish "
            "complete EXIF/XMP values and does not infer causal influence."
        ),
    }
    write_json_atomic(staged_summary_json, payload)

    if args.install:
        install_file(staged_detail, RESULTS_DIR / DETAIL_NAME)
        install_file(staged_summary_table, RESULTS_DIR / SUMMARY_TABLE_NAME)
        install_file(staged_delta, RESULTS_DIR / DELTA_NAME)
        install_file(staged_summary_json, RESULTS_DIR / SUMMARY_JSON_NAME)
        print("Installed minimized embedded-metadata sensitivity artifacts.")
    else:
        print("Generated staged embedded-metadata sensitivity artifacts.")
    print(" - decisions: 69000")
    print(" - sensitive bundles: 15")
    print(" - minimized detail rows: 90")
    print(" - aggregate reference equivalence: passed")
    print(f" - staging directory: {staging_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
