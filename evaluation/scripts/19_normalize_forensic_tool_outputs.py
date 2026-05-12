#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_normalize_forensic_tool_outputs.py

First-pass normalization scaffold for commercial forensic-tool outputs in the
FAIR-Lab thesis pipeline.

Purpose
-------
Commercial forensic tools export AI classification results using different
formats, category names, file identifiers, paths, and confidence conventions.
This script provides a lightweight normalization layer that maps those raw
exports back to the forensic evaluation bundle and produces comparable CSV
outputs for later metric computation.

This is intentionally a generic first version. Tool-specific parsers can be
added later once real exports from Magnet AXIOM / Magnet.AI, X-Ways / Excire,
Cellebrite UFED, and Oxygen Forensic Detective are available.

Inputs
------
- datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
- datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
- forensic_tools/<tool_name>/raw_exports/**/*.{csv,json,jsonl,txt,tsv}

Outputs
-------
- evaluation/forensic_tools/normalized_predictions.csv
- evaluation/forensic_tools/tool_export_audit.csv
- evaluation/forensic_tools/tool_version_log.csv
- results/metrics/forensic_tools_metrics.csv

Design notes
------------
- The preferred matching keys are SHA256, then MD5, then tool-input filename.
- Raw tool labels are mapped to a conservative binary forensic interpretation:
  weapon_detected = true / false / unknown.
- Metrics are computed only where a prediction can be interpreted as true/false.
- OOD rows are handled separately and are not mixed into binary accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

try:
    from datasets.scripts.utils.paths import REPO_ROOT, repo_relative_path
except Exception:  # pragma: no cover - fallback for early standalone use.
    REPO_ROOT = REPO_ROOT_BOOTSTRAP

    def repo_relative_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return REPO_ROOT / path


SCRIPT_NAME = "evaluation/scripts/19_normalize_forensic_tool_outputs.py"

DEFAULT_BUNDLE_MANIFEST = REPO_ROOT / "datasets" / "forensic_evaluation_bundle" / "metadata" / "bundle_manifest.csv"
DEFAULT_BUNDLE_HASHES = REPO_ROOT / "datasets" / "forensic_evaluation_bundle" / "metadata" / "bundle_hashes_sha256.csv"
DEFAULT_FORENSIC_TOOLS_ROOT = REPO_ROOT / "forensic_tools"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "forensic_tools"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "metrics"

NORMALIZED_PREDICTIONS_PATH = DEFAULT_OUTPUT_DIR / "normalized_predictions.csv"
EXPORT_AUDIT_PATH = DEFAULT_OUTPUT_DIR / "tool_export_audit.csv"
TOOL_VERSION_LOG_PATH = DEFAULT_OUTPUT_DIR / "tool_version_log.csv"
FORENSIC_TOOL_METRICS_PATH = DEFAULT_METRICS_DIR / "forensic_tools_metrics.csv"

SUPPORTED_EXPORT_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}

KNOWN_TOOL_NAMES = [
    "magnet_axiom",
    "xways_excire",
    "cellebrite_ufed",
    "oxygen_forensic_detective",
]

SHA256_COLUMNS = {
    "sha256",
    "sha-256",
    "sha_256",
    "hash_sha256",
    "sha256_hash",
    "file_sha256",
    "artifact_sha256",
}

MD5_COLUMNS = {
    "md5",
    "hash_md5",
    "md5_hash",
    "file_md5",
    "artifact_md5",
}

FILENAME_COLUMNS = {
    "filename",
    "file_name",
    "name",
    "item_name",
    "artifact_name",
    "original_filename",
    "tool_input_filename",
    "path",
    "file_path",
    "filepath",
    "source_path",
    "full_path",
    "export_path",
}

LABEL_COLUMNS = {
    "label",
    "labels",
    "category",
    "categories",
    "tag",
    "tags",
    "classification",
    "class",
    "ai_label",
    "ai_category",
    "recognized_category",
    "detected_category",
    "result",
    "description",
}

CONFIDENCE_COLUMNS = {
    "confidence",
    "score",
    "probability",
    "confidence_score",
    "ai_confidence",
    "rank_score",
}

WEAPON_KEYWORDS = {
    "weapon",
    "weapons",
    "gun",
    "guns",
    "firearm",
    "firearms",
    "pistol",
    "pistols",
    "rifle",
    "rifles",
    "shotgun",
    "shotguns",
    "revolver",
    "handgun",
    "handguns",
    "ammo",
    "ammunition",
}

NON_WEAPON_NEGATIVE_KEYWORDS = {
    "no weapon",
    "non weapon",
    "non_weapon",
    "not weapon",
    "not a weapon",
    "none",
    "negative",
    "benign",
    "normal",
    "safe",
    "unknown object",
    "uncategorized",
}

OOD_HINT_KEYWORDS = {
    "knife",
    "knives",
    "sword",
    "toy",
    "replica",
    "airsoft",
    "cgi",
    "cartoon",
    "synthetic",
}


@dataclass
class RawToolRow:
    """Tool-output row after best-effort raw parsing."""

    tool_name: str
    raw_export_file: str
    raw_row_number: int
    raw_record: dict[str, Any]
    sha256: str
    md5: str
    filename_or_path: str
    raw_label: str
    raw_confidence: str


# =============================================================================
# Argument parsing and logging
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize commercial forensic-tool outputs against the forensic evaluation bundle."
    )
    parser.add_argument(
        "--bundle-manifest",
        type=str,
        default=str(DEFAULT_BUNDLE_MANIFEST),
        help="Path to datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv.",
    )
    parser.add_argument(
        "--bundle-hashes",
        type=str,
        default=str(DEFAULT_BUNDLE_HASHES),
        help="Path to datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv.",
    )
    parser.add_argument(
        "--forensic-tools-root",
        type=str,
        default=str(DEFAULT_FORENSIC_TOOLS_ROOT),
        help="Root directory containing forensic_tools/<tool_name>/raw_exports/.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=KNOWN_TOOL_NAMES,
        help="Tool directory names to scan under forensic_tools/. Default: known thesis tool set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for normalized forensic-tool outputs.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(DEFAULT_METRICS_DIR),
        help="Directory for forensic tool metrics.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no raw export files are found for a requested tool.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


# =============================================================================
# Generic helpers
# =============================================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", safe_str(name).lower()).strip("_")


def normalize_hash(value: Any) -> str:
    text = safe_str(value).lower()
    text = re.sub(r"[^a-f0-9]", "", text)
    return text


def basename_from_path(value: str) -> str:
    text = safe_str(value).replace("\\", "/")
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1]


def repo_relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def first_non_empty(record: dict[str, Any], candidate_columns: set[str]) -> str:
    normalized_candidates = {normalize_column_name(col) for col in candidate_columns}
    for key, value in record.items():
        if normalize_column_name(key) in normalized_candidates:
            text = safe_str(value)
            if text:
                return text
    return ""


def collect_text_fields(record: dict[str, Any], candidate_columns: set[str]) -> str:
    normalized_candidates = {normalize_column_name(col) for col in candidate_columns}
    values: list[str] = []
    for key, value in record.items():
        if normalize_column_name(key) in normalized_candidates:
            text = safe_str(value)
            if text:
                values.append(text)
    return " | ".join(values)


def parse_float(value: Any) -> float | None:
    text = safe_str(value).replace(",", ".")
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


# =============================================================================
# Bundle loading and index construction
# =============================================================================


def load_bundle(bundle_manifest_path: Path, bundle_hashes_path: Path) -> pd.DataFrame:
    if not bundle_manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest_path}")

    bundle_df = pd.read_csv(bundle_manifest_path)

    required = {
        "bundle_id",
        "tool_input_filename",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
    }
    missing = required - set(bundle_df.columns)
    if missing:
        raise ValueError(f"Bundle manifest is missing required columns: {sorted(missing)}")

    # Add compact hash data when available. The bundle manifest already normally
    # contains hashes, but the compact file is useful if schemas evolve.
    if bundle_hashes_path.exists():
        hashes_df = pd.read_csv(bundle_hashes_path)
        if "bundle_id" in hashes_df.columns:
            suffix_cols = [col for col in hashes_df.columns if col != "bundle_id"]
            bundle_df = bundle_df.merge(
                hashes_df[["bundle_id", *suffix_cols]],
                on="bundle_id",
                how="left",
                suffixes=("", "_hashfile"),
            )

    # Normalize optional hash columns into canonical helper columns.
    bundle_df["_sha256_key"] = ""
    bundle_df["_md5_key"] = ""

    for col in ["sha256_actual", "sha256", "sha256_manifest", "sha256_hashfile"]:
        if col in bundle_df.columns:
            bundle_df["_sha256_key"] = bundle_df["_sha256_key"].mask(
                bundle_df["_sha256_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    for col in ["md5_actual", "md5", "md5_manifest", "md5_hashfile"]:
        if col in bundle_df.columns:
            bundle_df["_md5_key"] = bundle_df["_md5_key"].mask(
                bundle_df["_md5_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    bundle_df["_filename_key"] = bundle_df["tool_input_filename"].map(lambda x: basename_from_path(safe_str(x)).lower())

    logging.info("Loaded bundle rows: %d", len(bundle_df))
    return bundle_df


def build_bundle_indexes(bundle_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sha_index: dict[str, dict[str, Any]] = {}
    md5_index: dict[str, dict[str, Any]] = {}
    filename_index: dict[str, dict[str, Any]] = {}

    for row in bundle_df.to_dict(orient="records"):
        sha = safe_str(row.get("_sha256_key", ""))
        md5 = safe_str(row.get("_md5_key", ""))
        filename = safe_str(row.get("_filename_key", ""))

        if sha:
            sha_index[sha] = row
        if md5:
            md5_index[md5] = row
        if filename:
            filename_index[filename] = row

    return sha_index, md5_index, filename_index


# =============================================================================
# Raw export discovery and parsing
# =============================================================================


def discover_export_files(tool_dir: Path) -> list[Path]:
    raw_dir = tool_dir / "raw_exports"
    if not raw_dir.exists():
        return []
    files = [path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXPORT_EXTENSIONS]
    return sorted(files)


def read_csv_like(path: Path) -> list[dict[str, Any]]:
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    if sep:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


def flatten_json_object(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a JSON object enough for generic field extraction."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else safe_str(key)
            if isinstance(value, dict):
                flat.update(flatten_json_object(value, new_key))
            elif isinstance(value, list):
                if all(not isinstance(item, (dict, list)) for item in value):
                    flat[new_key] = " | ".join(safe_str(item) for item in value)
                else:
                    flat[new_key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[new_key] = value
    else:
        flat[prefix or "value"] = obj
    return flat


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(flatten_json_object(json.loads(line)))
        return records

    obj = json.loads(text)
    if isinstance(obj, list):
        return [flatten_json_object(item) for item in obj]
    if isinstance(obj, dict):
        # Common export shape: {"items": [...]} or {"artifacts": [...]}.
        for key in ("items", "artifacts", "files", "results", "records", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return [flatten_json_object(item) for item in value]
        return [flatten_json_object(obj)]
    return [{"value": safe_str(obj)}]


def read_txt_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        records.append({"line_number": idx, "description": line})
    return records


def read_export_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return read_csv_like(path)
    if suffix in {".json", ".jsonl"}:
        return read_json_records(path)
    if suffix == ".txt":
        return read_txt_records(path)
    return []


def extract_raw_row(tool_name: str, export_file: Path, row_number: int, record: dict[str, Any]) -> RawToolRow:
    sha256 = normalize_hash(first_non_empty(record, SHA256_COLUMNS))
    md5 = normalize_hash(first_non_empty(record, MD5_COLUMNS))
    filename_or_path = first_non_empty(record, FILENAME_COLUMNS)
    raw_label = collect_text_fields(record, LABEL_COLUMNS)
    raw_confidence = first_non_empty(record, CONFIDENCE_COLUMNS)

    # Fallback: when a row has only free text, use the full record as a label field.
    if not raw_label:
        free_text = " | ".join(safe_str(value) for value in record.values() if safe_str(value))
        raw_label = free_text[:1000]

    return RawToolRow(
        tool_name=tool_name,
        raw_export_file=repo_relative_string(export_file),
        raw_row_number=row_number,
        raw_record=record,
        sha256=sha256,
        md5=md5,
        filename_or_path=filename_or_path,
        raw_label=raw_label,
        raw_confidence=raw_confidence,
    )


def parse_tool_exports(tool_name: str, tool_dir: Path) -> tuple[list[RawToolRow], list[dict[str, Any]]]:
    export_files = discover_export_files(tool_dir)
    raw_rows: list[RawToolRow] = []
    audit_rows: list[dict[str, Any]] = []

    for export_file in export_files:
        try:
            records = read_export_records(export_file)
            status = "parsed"
            error = ""
        except Exception as exc:  # Keep the pipeline moving for partial exports.
            records = []
            status = "parse_error"
            error = f"{type(exc).__name__}: {exc}"
            logging.warning("Could not parse %s: %s", export_file, error)

        for idx, record in enumerate(records, start=1):
            raw_rows.append(extract_raw_row(tool_name, export_file, idx, record))

        audit_rows.append(
            {
                "tool_name": tool_name,
                "raw_export_file": repo_relative_string(export_file),
                "extension": export_file.suffix.lower(),
                "status": status,
                "parsed_rows": len(records),
                "error": error,
            }
        )

    return raw_rows, audit_rows


# =============================================================================
# Prediction normalization and matching
# =============================================================================


def interpret_weapon_detection(raw_label: str) -> tuple[str, str]:
    """
    Convert a raw tool label into true/false/unknown.

    The mapping is intentionally conservative. Tool-specific parsers can later
    override this function with vendor-specific categories and confidence logic.
    """
    text = safe_str(raw_label).lower()
    text_clean = re.sub(r"[^a-z0-9_ +/.-]+", " ", text)

    if not text_clean:
        return "unknown", "empty_label"

    for negative in NON_WEAPON_NEGATIVE_KEYWORDS:
        if negative in text_clean:
            return "false", f"negative_keyword:{negative}"

    for keyword in WEAPON_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_clean):
            return "true", f"weapon_keyword:{keyword}"

    for keyword in OOD_HINT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_clean):
            return "unknown", f"ood_hint:{keyword}"

    return "unknown", "no_mapping_rule"


def match_bundle_row(
    raw_row: RawToolRow,
    sha_index: dict[str, dict[str, Any]],
    md5_index: dict[str, dict[str, Any]],
    filename_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if raw_row.sha256 and raw_row.sha256 in sha_index:
        return sha_index[raw_row.sha256], "sha256"
    if raw_row.md5 and raw_row.md5 in md5_index:
        return md5_index[raw_row.md5], "md5"

    filename = basename_from_path(raw_row.filename_or_path).lower()
    if filename and filename in filename_index:
        return filename_index[filename], "filename"

    # Fallback: some reports embed the filename inside a description/category field.
    text = f"{raw_row.filename_or_path} {raw_row.raw_label}".lower()
    for filename_key, row in filename_index.items():
        if filename_key and filename_key in text:
            return row, "filename_embedded"

    return None, "unmatched"


def compute_correctness(final_label: str, weapon_detected: str) -> tuple[str, str, str]:
    """Return correct, false_negative, false_positive as text booleans/NA."""
    label = safe_str(final_label).lower()

    if weapon_detected not in {"true", "false"}:
        return "", "", ""

    detected = weapon_detected == "true"

    if label == "weapon":
        correct = detected
        false_negative = not detected
        false_positive = False
    elif label == "non_weapon":
        correct = not detected
        false_negative = False
        false_positive = detected
    else:
        # OOD does not have binary correctness. Track whether it was flagged.
        return "", "", ""

    return str(correct).lower(), str(false_negative).lower(), str(false_positive).lower()


def normalize_rows(
    raw_rows: list[RawToolRow],
    sha_index: dict[str, dict[str, Any]],
    md5_index: dict[str, dict[str, Any]],
    filename_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        bundle_row, match_method = match_bundle_row(raw_row, sha_index, md5_index, filename_index)
        weapon_detected, mapping_reason = interpret_weapon_detection(raw_row.raw_label)
        confidence = parse_float(raw_row.raw_confidence)

        if bundle_row is None:
            base = {
                "tool_name": raw_row.tool_name,
                "bundle_id": "",
                "match_method": match_method,
                "matched": "false",
                "tool_input_filename": basename_from_path(raw_row.filename_or_path),
                "sha256": raw_row.sha256,
                "md5": raw_row.md5,
                "sample_type": "",
                "attack_family": "",
                "attack_name": "",
                "final_label": "",
                "original_image_id": "",
                "generated_image_id": "",
            }
        else:
            base = {
                "tool_name": raw_row.tool_name,
                "bundle_id": safe_str(bundle_row.get("bundle_id", "")),
                "match_method": match_method,
                "matched": "true",
                "tool_input_filename": safe_str(bundle_row.get("tool_input_filename", "")),
                "sha256": safe_str(bundle_row.get("_sha256_key", raw_row.sha256)),
                "md5": safe_str(bundle_row.get("_md5_key", raw_row.md5)),
                "sample_type": safe_str(bundle_row.get("sample_type", "")),
                "attack_family": safe_str(bundle_row.get("attack_family", "")),
                "attack_name": safe_str(bundle_row.get("attack_name", "")),
                "final_label": safe_str(bundle_row.get("final_label", "")),
                "original_image_id": safe_str(bundle_row.get("original_image_id", "")),
                "generated_image_id": safe_str(bundle_row.get("generated_image_id", "")),
            }

        correct, false_negative, false_positive = compute_correctness(base["final_label"], weapon_detected)

        normalized.append(
            {
                **base,
                "raw_export_file": raw_row.raw_export_file,
                "raw_row_number": raw_row.raw_row_number,
                "raw_filename_or_path": raw_row.filename_or_path,
                "tool_raw_label": raw_row.raw_label,
                "tool_raw_confidence": raw_row.raw_confidence,
                "tool_confidence_numeric": "" if confidence is None else confidence,
                "weapon_detected": weapon_detected,
                "normalized_prediction": "weapon" if weapon_detected == "true" else "non_weapon" if weapon_detected == "false" else "unknown",
                "mapping_reason": mapping_reason,
                "correct": correct,
                "false_negative": false_negative,
                "false_positive": false_positive,
            }
        )

    return normalized


# =============================================================================
# Metrics
# =============================================================================


def bool_text(value: Any) -> bool | None:
    text = safe_str(value).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def metric_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_rows = [
        row
        for row in rows
        if safe_str(row.get("final_label", "")).lower() in {"weapon", "non_weapon"}
        and safe_str(row.get("weapon_detected", "")) in {"true", "false"}
    ]

    tp = fp = tn = fn = 0
    for row in binary_rows:
        label = safe_str(row["final_label"]).lower()
        detected = safe_str(row["weapon_detected"]).lower() == "true"

        if label == "weapon" and detected:
            tp += 1
        elif label == "weapon" and not detected:
            fn += 1
        elif label == "non_weapon" and detected:
            fp += 1
        elif label == "non_weapon" and not detected:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = (recall + specificity) / 2

    ood_rows = [row for row in rows if safe_str(row.get("final_label", "")).lower() == "ood"]
    ood_weapon_flags = sum(1 for row in ood_rows if safe_str(row.get("weapon_detected", "")) == "true")
    ood_interpretable = sum(1 for row in ood_rows if safe_str(row.get("weapon_detected", "")) in {"true", "false"})

    return {
        "rows_total": len(rows),
        "binary_interpretable_rows": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": metric_value(accuracy),
        "balanced_accuracy": metric_value(balanced_accuracy),
        "precision_weapon": metric_value(precision),
        "recall_weapon": metric_value(recall),
        "specificity_non_weapon": metric_value(specificity),
        "f1_weapon": metric_value(f1),
        "false_negative_rate": metric_value(safe_div(fn, tp + fn)),
        "false_positive_rate": metric_value(safe_div(fp, fp + tn)),
        "ood_rows": len(ood_rows),
        "ood_interpretable_rows": ood_interpretable,
        "ood_weapon_flag_rate": metric_value(safe_div(ood_weapon_flags, len(ood_rows))),
    }


def compute_metrics(normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        if safe_str(row.get("matched", "")) != "true":
            continue
        key = (
            safe_str(row.get("tool_name", "")),
            safe_str(row.get("sample_type", "")) or "all",
            safe_str(row.get("attack_name", "")) or "all",
        )
        groups[key].append(row)

        all_key = (safe_str(row.get("tool_name", "")), "all", "all")
        groups[all_key].append(row)

    for (tool_name, sample_type, attack_name), rows in sorted(groups.items()):
        values = compute_group_metrics(rows)
        metrics_rows.append(
            {
                "tool_name": tool_name,
                "sample_type": sample_type,
                "attack_name": attack_name,
                **values,
            }
        )

    return metrics_rows


# =============================================================================
# CSV writing
# =============================================================================


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    bundle_manifest_path = repo_relative_path(args.bundle_manifest)
    bundle_hashes_path = repo_relative_path(args.bundle_hashes)
    forensic_tools_root = repo_relative_path(args.forensic_tools_root)
    output_dir = repo_relative_path(args.output_dir)
    metrics_dir = repo_relative_path(args.metrics_dir)

    normalized_predictions_path = output_dir / "normalized_predictions.csv"
    export_audit_path = output_dir / "tool_export_audit.csv"
    tool_version_log_path = output_dir / "tool_version_log.csv"
    forensic_tool_metrics_path = metrics_dir / "forensic_tools_metrics.csv"

    logging.info("Bundle manifest: %s", bundle_manifest_path)
    logging.info("Forensic tools root: %s", forensic_tools_root)
    logging.info("Output directory: %s", output_dir)

    bundle_df = load_bundle(bundle_manifest_path, bundle_hashes_path)
    sha_index, md5_index, filename_index = build_bundle_indexes(bundle_df)

    all_raw_rows: list[RawToolRow] = []
    audit_rows: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []

    for tool_name in args.tools:
        tool_dir = forensic_tools_root / tool_name
        export_files = discover_export_files(tool_dir)

        logging.info("Tool %s: found %d raw export file(s)", tool_name, len(export_files))

        if args.strict and not export_files:
            raise FileNotFoundError(f"No raw export files found for tool: {tool_name} under {tool_dir / 'raw_exports'}")

        raw_rows, tool_audit_rows = parse_tool_exports(tool_name, tool_dir)
        all_raw_rows.extend(raw_rows)
        audit_rows.extend(tool_audit_rows)

        version_rows.append(
            {
                "tool_name": tool_name,
                "tool_version": "",
                "tool_build": "",
                "ai_modules_enabled": "",
                "os_environment": "",
                "import_path": "datasets/forensic_evaluation_bundle/blind_tool_input/files/",
                "export_files_found": len(export_files),
                "notes": "Fill manually after tool execution/export.",
                "created_at": utc_now_iso(),
            }
        )

    normalized_rows = normalize_rows(all_raw_rows, sha_index, md5_index, filename_index)
    metrics_rows = compute_metrics(normalized_rows)

    write_csv(normalized_predictions_path, normalized_rows)
    write_csv(export_audit_path, audit_rows)
    write_csv(tool_version_log_path, version_rows)
    write_csv(forensic_tool_metrics_path, metrics_rows)

    matched = sum(1 for row in normalized_rows if row.get("matched") == "true")
    unmatched = len(normalized_rows) - matched
    interpretable = sum(1 for row in normalized_rows if row.get("weapon_detected") in {"true", "false"})

    logging.info("Raw rows parsed: %d", len(all_raw_rows))
    logging.info("Normalized rows: %d", len(normalized_rows))
    logging.info("Matched rows: %d", matched)
    logging.info("Unmatched rows: %d", unmatched)
    logging.info("Interpretable binary/weapon-detection rows: %d", interpretable)
    logging.info("Normalized predictions: %s", normalized_predictions_path)
    logging.info("Export audit: %s", export_audit_path)
    logging.info("Tool version log: %s", tool_version_log_path)
    logging.info("Metrics: %s", forensic_tool_metrics_path)


if __name__ == "__main__":
    main()
