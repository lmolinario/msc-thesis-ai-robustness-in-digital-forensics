#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_normalize_forensic_tool_outputs.py

Normalize commercial forensic-tool outputs against the FAIR-Lab forensic
evaluation bundle.

Current implemented parser
--------------------------
- Magnet AXIOM / Magnet.AI export:
  - reads only Pictures.csv from forensic_tools/magnet_axiom/raw_exports/**
  - maps Tags = "Possible weapons" to predicted weapon
  - maps empty Tags to not flagged / predicted non_weapon
  - deduplicates AXIOM duplicated rows to one prediction per bundle_id

Inputs
------
- datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
- datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
- forensic_tools/magnet_axiom/raw_exports/**/Pictures.csv

Outputs
-------
- evaluation/forensic_tools/normalized_predictions.csv
- evaluation/forensic_tools/magnet_axiom_normalized_predictions.csv
- evaluation/forensic_tools/tool_export_audit.csv
- evaluation/forensic_tools/tool_version_log.csv
- evaluation/forensic_tools/normalization_summary.json
- results/metrics/forensic_tools_metrics.csv
- results/metrics/magnet_axiom_metrics.csv

Notes
-----
The script is deliberately conservative:
- OOD samples are not mixed into binary accuracy.
- Unknown/non-interpretable predictions are preserved.
- For AXIOM, absence of the "Possible weapons" tag is interpreted as "not flagged",
  not as an unknown state, because AXIOM uses tags as positive detections.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Repository path bootstrap
# =============================================================================

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]

if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

try:
    from datasets.scripts.utils.paths import REPO_ROOT, repo_relative_path
except Exception:  # pragma: no cover
    REPO_ROOT = REPO_ROOT_BOOTSTRAP

    def repo_relative_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return REPO_ROOT / path


SCRIPT_NAME = "evaluation/scripts/19_normalize_forensic_tool_outputs.py"

DEFAULT_BUNDLE_MANIFEST = (
    REPO_ROOT
    / "datasets"
    / "forensic_evaluation_bundle"
    / "metadata"
    / "bundle_manifest.csv"
)

DEFAULT_BUNDLE_HASHES = (
    REPO_ROOT
    / "datasets"
    / "forensic_evaluation_bundle"
    / "metadata"
    / "bundle_hashes_sha256.csv"
)

DEFAULT_FORENSIC_TOOLS_ROOT = REPO_ROOT / "forensic_tools"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "forensic_tools"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "metrics"

SUPPORTED_GENERIC_EXPORT_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}

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
    "md5 hash",
    "md5_hash",
    "hash_md5",
    "file_md5",
    "artifact_md5",
}

FILENAME_COLUMNS = {
    "filename",
    "file name",
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


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RawToolRow:
    """Raw row parsed from a forensic-tool export."""

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
# CLI and logging
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize forensic-tool exports against the FAIR-Lab forensic evaluation bundle."
    )

    parser.add_argument(
        "--bundle-manifest",
        type=str,
        default=str(DEFAULT_BUNDLE_MANIFEST),
        help="Path to bundle_manifest.csv.",
    )

    parser.add_argument(
        "--bundle-hashes",
        type=str,
        default=str(DEFAULT_BUNDLE_HASHES),
        help="Path to bundle_hashes_sha256.csv.",
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
        help="Tool directory names to scan under forensic_tools/.",
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
        help="Directory for forensic-tool metrics.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no raw export files are found for a requested tool.",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable one-prediction-per-bundle_id consolidation.",
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


def basename_from_path(value: Any) -> str:
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


def unique_join(values: list[Any], separator: str = " | ") -> str:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        text = safe_str(value)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            output.append(text)

    return separator.join(output)


# =============================================================================
# Bundle loading and indexing
# =============================================================================

def load_bundle(bundle_manifest_path: Path, bundle_hashes_path: Path) -> pd.DataFrame:
    if not bundle_manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest_path}")

    bundle_df = pd.read_csv(bundle_manifest_path, dtype=str, keep_default_na=False)

    required_columns = {
        "bundle_id",
        "tool_input_filename",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
    }

    missing = required_columns - set(bundle_df.columns)
    if missing:
        raise ValueError(
            f"Bundle manifest is missing required columns: {sorted(missing)}"
        )

    if bundle_hashes_path.exists():
        hashes_df = pd.read_csv(bundle_hashes_path, dtype=str, keep_default_na=False)

        if "bundle_id" in hashes_df.columns:
            hash_columns = [col for col in hashes_df.columns if col != "bundle_id"]
            bundle_df = bundle_df.merge(
                hashes_df[["bundle_id", *hash_columns]],
                on="bundle_id",
                how="left",
                suffixes=("", "_hashfile"),
            )

    bundle_df["_sha256_key"] = ""
    bundle_df["_md5_key"] = ""

    sha_candidate_columns = [
        "sha256_actual",
        "sha256",
        "sha256_manifest",
        "sha256_hashfile",
    ]

    md5_candidate_columns = [
        "md5_actual",
        "md5",
        "md5_manifest",
        "md5_hashfile",
    ]

    for col in sha_candidate_columns:
        if col in bundle_df.columns:
            bundle_df["_sha256_key"] = bundle_df["_sha256_key"].mask(
                bundle_df["_sha256_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    for col in md5_candidate_columns:
        if col in bundle_df.columns:
            bundle_df["_md5_key"] = bundle_df["_md5_key"].mask(
                bundle_df["_md5_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    bundle_df["_filename_key"] = bundle_df["tool_input_filename"].map(
        lambda x: basename_from_path(x).lower()
    )

    logging.info("Loaded bundle rows: %d", len(bundle_df))
    return bundle_df


def build_bundle_indexes(
    bundle_df: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
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

    logging.info("Bundle SHA256 index entries: %d", len(sha_index))
    logging.info("Bundle MD5 index entries: %d", len(md5_index))
    logging.info("Bundle filename index entries: %d", len(filename_index))

    return sha_index, md5_index, filename_index


# =============================================================================
# Raw export discovery and parsing
# =============================================================================

def discover_prediction_export_files(tool_name: str, tool_dir: Path) -> list[Path]:
    """
    Return only files that should be interpreted as prediction exports.

    For Magnet AXIOM, only Pictures.csv is a prediction-bearing export for this
    thesis protocol. ExportSummary.json and other CSVs are audit/configuration
    artifacts, not classification rows.
    """
    raw_dir = tool_dir / "raw_exports"
    if not raw_dir.exists():
        return []

    if tool_name == "magnet_axiom":
        return sorted(
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.name.lower() == "pictures.csv"
        )

    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_GENERIC_EXPORT_EXTENSIONS
    )


def discover_all_export_files(tool_dir: Path) -> list[Path]:
    raw_dir = tool_dir / "raw_exports"
    if not raw_dir.exists():
        return []

    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_GENERIC_EXPORT_EXTENSIONS
    )


def read_csv_like(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            return df.to_dict(orient="records")
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error

    return []


def flatten_json_object(obj: Any, prefix: str = "") -> dict[str, Any]:
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

    elif isinstance(obj, list):
        flat[prefix or "value"] = json.dumps(obj, ensure_ascii=False)

    else:
        flat[prefix or "value"] = obj

    return flat


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
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
        for key in ("items", "artifacts", "files", "results", "records", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return [flatten_json_object(item) for item in value]

        return [flatten_json_object(obj)]

    return [{"value": safe_str(obj)}]


def read_txt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
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


def extract_raw_row(
    tool_name: str,
    export_file: Path,
    row_number: int,
    record: dict[str, Any],
) -> RawToolRow:
    sha256 = normalize_hash(first_non_empty(record, SHA256_COLUMNS))
    md5 = normalize_hash(first_non_empty(record, MD5_COLUMNS))
    filename_or_path = first_non_empty(record, FILENAME_COLUMNS)
    raw_confidence = first_non_empty(record, CONFIDENCE_COLUMNS)

    if tool_name == "magnet_axiom":
        # AXIOM Pictures.csv uses the Tags column as the positive AI category
        # signal. Empty Tags means "not flagged" for this protocol.
        raw_label = first_non_empty(record, {"tags"})
    else:
        raw_label = collect_text_fields(record, LABEL_COLUMNS)

        if not raw_label:
            free_text = " | ".join(
                safe_str(value)
                for value in record.values()
                if safe_str(value)
            )
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


def parse_tool_exports(
    tool_name: str,
    tool_dir: Path,
) -> tuple[list[RawToolRow], list[dict[str, Any]]]:
    prediction_files = discover_prediction_export_files(tool_name, tool_dir)
    all_export_files = discover_all_export_files(tool_dir)

    raw_rows: list[RawToolRow] = []
    audit_rows: list[dict[str, Any]] = []

    prediction_file_set = {path.resolve() for path in prediction_files}

    for export_file in all_export_files:
        is_prediction_file = export_file.resolve() in prediction_file_set

        try:
            records = read_export_records(export_file)
            status = "parsed_prediction_file" if is_prediction_file else "parsed_audit_only"
            error = ""
        except Exception as exc:
            records = []
            status = "parse_error"
            error = f"{type(exc).__name__}: {exc}"
            logging.warning("Could not parse %s: %s", export_file, error)

        if is_prediction_file:
            for idx, record in enumerate(records, start=1):
                raw_rows.append(
                    extract_raw_row(
                        tool_name=tool_name,
                        export_file=export_file,
                        row_number=idx,
                        record=record,
                    )
                )

        audit_rows.append(
            {
                "tool_name": tool_name,
                "raw_export_file": repo_relative_string(export_file),
                "extension": export_file.suffix.lower(),
                "is_prediction_file": str(is_prediction_file).lower(),
                "status": status,
                "parsed_rows": len(records),
                "error": error,
            }
        )

    return raw_rows, audit_rows


# =============================================================================
# Tool version log
# =============================================================================

def extract_version_fields_from_summary(summary_path: Path) -> dict[str, str]:
    try:
        records = read_json_records(summary_path)
    except Exception:
        return {}

    if not records:
        return {}

    flat = records[0]
    normalized_items = {
        normalize_column_name(key): safe_str(value)
        for key, value in flat.items()
    }

    def find_value(patterns: list[str]) -> str:
        for key, value in normalized_items.items():
            if not value:
                continue
            for pattern in patterns:
                if pattern in key:
                    return value
        return ""

    return {
        "tool_version": find_value(["version", "application_version", "axiom_version"]),
        "tool_build": find_value(["build", "build_number"]),
        "case_name": find_value(["case", "case_name"]),
        "export_status": find_value(["status", "export_status"]),
        "export_timestamp": find_value(["timestamp", "created", "exported", "date"]),
    }


def build_tool_version_row(tool_name: str, tool_dir: Path, export_files_found: int) -> dict[str, Any]:
    summary_files = sorted(
        path
        for path in (tool_dir / "raw_exports").rglob("*")
        if path.is_file()
        and path.name.lower() in {"exportsummary.json", "export_summary.json"}
    ) if (tool_dir / "raw_exports").exists() else []

    extracted: dict[str, str] = {}
    summary_file = ""

    if summary_files:
        summary_file = repo_relative_string(summary_files[0])
        extracted = extract_version_fields_from_summary(summary_files[0])

    if tool_name == "magnet_axiom":
        default_notes = (
            "Magnet AXIOM export. Predictions are derived from Pictures.csv Tags; "
            "Tags='Possible weapons' is mapped to weapon_detected=true; empty Tags "
            "is mapped to weapon_detected=false."
        )
    else:
        default_notes = "Fill manually after tool execution/export."

    return {
        "tool_name": tool_name,
        "tool_version": extracted.get("tool_version", ""),
        "tool_build": extracted.get("tool_build", ""),
        "case_name": extracted.get("case_name", ""),
        "export_status": extracted.get("export_status", ""),
        "export_timestamp": extracted.get("export_timestamp", ""),
        "summary_file": summary_file,
        "ai_modules_enabled": "",
        "os_environment": "",
        "import_path": "datasets/forensic_evaluation_bundle/blind_tool_input/files/",
        "export_files_found": export_files_found,
        "notes": default_notes,
        "created_at": utc_now_iso(),
    }


# =============================================================================
# Prediction interpretation and matching
# =============================================================================

def interpret_weapon_detection(tool_name: str, raw_label: str) -> tuple[str, str]:
    """
    Convert a raw label into:
    - "true"
    - "false"
    - "unknown"

    Magnet AXIOM receives a tool-specific rule because empty Tags have a
    meaningful interpretation in this export format.
    """
    label = safe_str(raw_label)
    text = label.lower()
    text_clean = re.sub(r"[^a-z0-9_ +/.-]+", " ", text)

    if tool_name == "magnet_axiom":
        if "possible weapons" in text_clean:
            return "true", "magnet_axiom_tag:possible_weapons"
        if not text_clean:
            return "false", "magnet_axiom_empty_tags:not_flagged"

        # Defensive fallback for future AXIOM category variants.
        if "weapon" in text_clean or "weapons" in text_clean:
            return "true", "magnet_axiom_tag:weapon_keyword"

        return "unknown", "magnet_axiom_unmapped_tag"

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

    # Fallback: sometimes reports embed the filename inside a text field.
    text = f"{raw_row.filename_or_path} {raw_row.raw_label}".lower()
    for filename_key, row in filename_index.items():
        if filename_key and filename_key in text:
            return row, "filename_embedded"

    return None, "unmatched"


def compute_correctness(final_label: str, weapon_detected: str) -> tuple[str, str, str]:
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
        # OOD has no binary correctness.
        return "", "", ""

    return (
        str(correct).lower(),
        str(false_negative).lower(),
        str(false_positive).lower(),
    )


def build_base_match_fields(
    raw_row: RawToolRow,
    bundle_row: dict[str, Any] | None,
    match_method: str,
) -> dict[str, Any]:
    if bundle_row is None:
        return {
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
            "attack_target_model": "",
            "fold": "",
            "final_label": "",
            "source_dataset": "",
            "original_image_id": "",
            "generated_image_id": "",
        }

    return {
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
        "attack_target_model": safe_str(bundle_row.get("attack_target_model", "")),
        "fold": safe_str(bundle_row.get("fold", "")),
        "final_label": safe_str(bundle_row.get("final_label", "")),
        "source_dataset": safe_str(bundle_row.get("source_dataset", "")),
        "original_image_id": safe_str(bundle_row.get("original_image_id", "")),
        "generated_image_id": safe_str(bundle_row.get("generated_image_id", "")),
    }


def normalize_rows(
    raw_rows: list[RawToolRow],
    sha_index: dict[str, dict[str, Any]],
    md5_index: dict[str, dict[str, Any]],
    filename_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        bundle_row, match_method = match_bundle_row(
            raw_row=raw_row,
            sha_index=sha_index,
            md5_index=md5_index,
            filename_index=filename_index,
        )

        weapon_detected, mapping_reason = interpret_weapon_detection(
            tool_name=raw_row.tool_name,
            raw_label=raw_row.raw_label,
        )

        confidence = parse_float(raw_row.raw_confidence)

        base = build_base_match_fields(
            raw_row=raw_row,
            bundle_row=bundle_row,
            match_method=match_method,
        )

        correct, false_negative, false_positive = compute_correctness(
            final_label=base["final_label"],
            weapon_detected=weapon_detected,
        )

        normalized_rows.append(
            {
                **base,
                "raw_export_file": raw_row.raw_export_file,
                "raw_row_number": raw_row.raw_row_number,
                "raw_filename_or_path": raw_row.filename_or_path,
                "tool_raw_label": raw_row.raw_label,
                "tool_raw_confidence": raw_row.raw_confidence,
                "tool_confidence_numeric": "" if confidence is None else confidence,
                "weapon_detected": weapon_detected,
                "normalized_prediction": (
                    "weapon"
                    if weapon_detected == "true"
                    else "non_weapon"
                    if weapon_detected == "false"
                    else "unknown"
                ),
                "mapping_reason": mapping_reason,
                "correct": correct,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "raw_row_count_after_deduplication": 1,
            }
        )

    return normalized_rows


# =============================================================================
# Deduplication / consolidation
# =============================================================================

def choose_consolidated_detection(rows: list[dict[str, Any]]) -> tuple[str, str]:
    detections = [safe_str(row.get("weapon_detected", "")) for row in rows]

    if "true" in detections:
        return "true", "deduplicated:any_positive"

    if "false" in detections:
        return "false", "deduplicated:all_not_flagged_or_no_positive"

    return "unknown", "deduplicated:all_unknown"


def consolidate_matched_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    representative = rows[0].copy()

    weapon_detected, consolidation_reason = choose_consolidated_detection(rows)

    representative["weapon_detected"] = weapon_detected
    representative["normalized_prediction"] = (
        "weapon"
        if weapon_detected == "true"
        else "non_weapon"
        if weapon_detected == "false"
        else "unknown"
    )

    original_reasons = [
        safe_str(row.get("mapping_reason", ""))
        for row in rows
        if safe_str(row.get("mapping_reason", ""))
    ]

    representative["mapping_reason"] = unique_join(
        [consolidation_reason, *original_reasons],
        separator=" | ",
    )

    representative["raw_export_file"] = unique_join(
        [row.get("raw_export_file", "") for row in rows],
        separator=" | ",
    )

    representative["raw_row_number"] = unique_join(
        [row.get("raw_row_number", "") for row in rows],
        separator=" | ",
    )

    representative["tool_raw_label"] = unique_join(
        [row.get("tool_raw_label", "") for row in rows],
        separator=" | ",
    )

    representative["tool_raw_confidence"] = unique_join(
        [row.get("tool_raw_confidence", "") for row in rows],
        separator=" | ",
    )

    numeric_confidences = [
        parse_float(row.get("tool_confidence_numeric", ""))
        for row in rows
        if parse_float(row.get("tool_confidence_numeric", "")) is not None
    ]

    representative["tool_confidence_numeric"] = (
        max(numeric_confidences) if numeric_confidences else ""
    )

    representative["raw_row_count_after_deduplication"] = len(rows)

    correct, false_negative, false_positive = compute_correctness(
        final_label=representative.get("final_label", ""),
        weapon_detected=weapon_detected,
    )

    representative["correct"] = correct
    representative["false_negative"] = false_negative
    representative["false_positive"] = false_positive

    return representative


def deduplicate_predictions(normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Consolidate duplicated matched rows to one prediction per tool_name + bundle_id.

    This is essential for Magnet AXIOM because the current export contains the
    same 11,500 bundle images twice due to two exported evidence sources.
    """
    matched_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unmatched_rows: list[dict[str, Any]] = []

    for row in normalized_rows:
        if safe_str(row.get("matched", "")) == "true" and safe_str(row.get("bundle_id", "")):
            key = (
                safe_str(row.get("tool_name", "")),
                safe_str(row.get("bundle_id", "")),
            )
            matched_groups[key].append(row)
        else:
            unmatched_rows.append(row)

    deduplicated_rows = [
        consolidate_matched_rows(group_rows)
        for _, group_rows in sorted(matched_groups.items())
    ]

    deduplicated_rows.extend(unmatched_rows)

    return deduplicated_rows


# =============================================================================
# Metrics
# =============================================================================

def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def metric_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [
        row
        for row in rows
        if safe_str(row.get("matched", "")) == "true"
    ]

    binary_rows = [
        row
        for row in matched_rows
        if safe_str(row.get("final_label", "")).lower() in {"weapon", "non_weapon"}
    ]

    binary_interpretable_rows = [
        row
        for row in binary_rows
        if safe_str(row.get("weapon_detected", "")) in {"true", "false"}
    ]

    tp = fp = tn = fn = 0

    for row in binary_interpretable_rows:
        label = safe_str(row.get("final_label", "")).lower()
        detected = safe_str(row.get("weapon_detected", "")).lower() == "true"

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

    ood_rows = [
        row
        for row in matched_rows
        if safe_str(row.get("final_label", "")).lower() == "ood"
    ]

    ood_weapon_flags = sum(
        1
        for row in ood_rows
        if safe_str(row.get("weapon_detected", "")) == "true"
    )

    ood_non_weapon_flags = sum(
        1
        for row in ood_rows
        if safe_str(row.get("weapon_detected", "")) == "false"
    )

    ood_unknown = sum(
        1
        for row in ood_rows
        if safe_str(row.get("weapon_detected", "")) == "unknown"
    )

    unknown_rows = sum(
        1
        for row in matched_rows
        if safe_str(row.get("weapon_detected", "")) == "unknown"
    )

    return {
        "rows_total": len(rows),
        "matched_rows": len(matched_rows),
        "binary_rows": len(binary_rows),
        "binary_interpretable_rows": len(binary_interpretable_rows),
        "unknown_rows": unknown_rows,
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
        "ood_weapon_flags": ood_weapon_flags,
        "ood_non_weapon_flags": ood_non_weapon_flags,
        "ood_unknown": ood_unknown,
        "ood_weapon_flag_rate": metric_value(safe_div(ood_weapon_flags, len(ood_rows))),
    }


def add_metric_group(
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    row: dict[str, Any],
    scope: str,
    sample_type: str,
    attack_family: str,
    attack_name: str,
) -> None:
    key = (
        safe_str(row.get("tool_name", "")),
        scope,
        sample_type or "all",
        attack_family or attack_name or "all",
    )
    groups[key].append(row)


def compute_metrics(normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in normalized_rows:
        if safe_str(row.get("matched", "")) != "true":
            continue

        sample_type = safe_str(row.get("sample_type", "")) or "none"
        attack_family = safe_str(row.get("attack_family", "")) or "none"
        attack_name = safe_str(row.get("attack_name", "")) or "none"

        add_metric_group(groups, row, "all", "all", "all", "all")
        add_metric_group(groups, row, "sample_type", sample_type, "all", "all")
        add_metric_group(groups, row, "attack_family", "all", attack_family, "all")
        add_metric_group(groups, row, "attack_name", "all", attack_family, attack_name)
        add_metric_group(groups, row, "sample_type_attack", sample_type, attack_family, attack_name)

    metric_rows: list[dict[str, Any]] = []

    for (tool_name, scope, group_1, group_2), rows in sorted(groups.items()):
        values = compute_group_metrics(rows)
        metric_rows.append(
            {
                "tool_name": tool_name,
                "scope": scope,
                "group_1": group_1,
                "group_2": group_2,
                **values,
            }
        )

    return metric_rows


# =============================================================================
# Output writing
# =============================================================================

def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    return fieldnames


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    normalization_summary_path = output_dir / "normalization_summary.json"
    forensic_tool_metrics_path = metrics_dir / "forensic_tools_metrics.csv"

    logging.info("Script: %s", SCRIPT_NAME)
    logging.info("Repository root: %s", REPO_ROOT)
    logging.info("Bundle manifest: %s", bundle_manifest_path)
    logging.info("Bundle hashes: %s", bundle_hashes_path)
    logging.info("Forensic tools root: %s", forensic_tools_root)
    logging.info("Output directory: %s", output_dir)
    logging.info("Metrics directory: %s", metrics_dir)

    bundle_df = load_bundle(bundle_manifest_path, bundle_hashes_path)
    sha_index, md5_index, filename_index = build_bundle_indexes(bundle_df)

    all_raw_rows: list[RawToolRow] = []
    audit_rows: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []

    per_tool_export_counts: dict[str, int] = {}
    per_tool_raw_row_counts: dict[str, int] = {}

    for tool_name in args.tools:
        tool_dir = forensic_tools_root / tool_name
        prediction_files = discover_prediction_export_files(tool_name, tool_dir)
        all_export_files = discover_all_export_files(tool_dir)

        logging.info(
            "Tool %s: found %d prediction export file(s), %d total export file(s)",
            tool_name,
            len(prediction_files),
            len(all_export_files),
        )

        if args.strict and not prediction_files:
            raise FileNotFoundError(
                f"No prediction export files found for {tool_name} under {tool_dir / 'raw_exports'}"
            )

        raw_rows, tool_audit_rows = parse_tool_exports(tool_name, tool_dir)

        all_raw_rows.extend(raw_rows)
        audit_rows.extend(tool_audit_rows)

        per_tool_export_counts[tool_name] = len(prediction_files)
        per_tool_raw_row_counts[tool_name] = len(raw_rows)

        version_rows.append(
            build_tool_version_row(
                tool_name=tool_name,
                tool_dir=tool_dir,
                export_files_found=len(all_export_files),
            )
        )

    normalized_before_deduplication = normalize_rows(
        raw_rows=all_raw_rows,
        sha_index=sha_index,
        md5_index=md5_index,
        filename_index=filename_index,
    )

    if args.no_deduplicate:
        normalized_rows = normalized_before_deduplication
    else:
        normalized_rows = deduplicate_predictions(normalized_before_deduplication)

    metrics_rows = compute_metrics(normalized_rows)

    write_csv(normalized_predictions_path, normalized_rows)
    write_csv(export_audit_path, audit_rows)
    write_csv(tool_version_log_path, version_rows)
    write_csv(forensic_tool_metrics_path, metrics_rows)

    # Tool-specific outputs for easier thesis inspection.
    for tool_name in args.tools:
        tool_rows = [
            row
            for row in normalized_rows
            if safe_str(row.get("tool_name", "")) == tool_name
        ]
        tool_metrics = [
            row
            for row in metrics_rows
            if safe_str(row.get("tool_name", "")) == tool_name
        ]

        if tool_rows:
            write_csv(
                output_dir / f"{tool_name}_normalized_predictions.csv",
                tool_rows,
            )

        if tool_metrics:
            write_csv(
                metrics_dir / f"{tool_name}_metrics.csv",
                tool_metrics,
            )

    matched_before = sum(
        1
        for row in normalized_before_deduplication
        if row.get("matched") == "true"
    )

    matched_after = sum(
        1
        for row in normalized_rows
        if row.get("matched") == "true"
    )

    unmatched_after = len(normalized_rows) - matched_after

    interpretable_after = sum(
        1
        for row in normalized_rows
        if row.get("weapon_detected") in {"true", "false"}
    )

    possible_weapon_after = sum(
        1
        for row in normalized_rows
        if row.get("weapon_detected") == "true"
    )

    not_flagged_after = sum(
        1
        for row in normalized_rows
        if row.get("weapon_detected") == "false"
    )

    unknown_after = sum(
        1
        for row in normalized_rows
        if row.get("weapon_detected") == "unknown"
    )

    summary = {
        "script": SCRIPT_NAME,
        "created_at": utc_now_iso(),
        "bundle_rows": len(bundle_df),
        "tools_requested": args.tools,
        "per_tool_prediction_export_counts": per_tool_export_counts,
        "per_tool_raw_row_counts": per_tool_raw_row_counts,
        "raw_rows_parsed": len(all_raw_rows),
        "normalized_rows_before_deduplication": len(normalized_before_deduplication),
        "matched_rows_before_deduplication": matched_before,
        "deduplication_enabled": not args.no_deduplicate,
        "normalized_rows_after_deduplication": len(normalized_rows),
        "matched_rows_after_deduplication": matched_after,
        "unmatched_rows_after_deduplication": unmatched_after,
        "interpretable_rows_after_deduplication": interpretable_after,
        "weapon_detected_true_after_deduplication": possible_weapon_after,
        "weapon_detected_false_after_deduplication": not_flagged_after,
        "weapon_detected_unknown_after_deduplication": unknown_after,
        "outputs": {
            "normalized_predictions": repo_relative_string(normalized_predictions_path),
            "tool_export_audit": repo_relative_string(export_audit_path),
            "tool_version_log": repo_relative_string(tool_version_log_path),
            "normalization_summary": repo_relative_string(normalization_summary_path),
            "forensic_tool_metrics": repo_relative_string(forensic_tool_metrics_path),
        },
    }

    write_json(normalization_summary_path, summary)

    logging.info("Raw rows parsed: %d", len(all_raw_rows))
    logging.info(
        "Normalized rows before deduplication: %d",
        len(normalized_before_deduplication),
    )
    logging.info("Matched rows before deduplication: %d", matched_before)
    logging.info("Deduplication enabled: %s", str(not args.no_deduplicate).lower())
    logging.info("Normalized rows after deduplication: %d", len(normalized_rows))
    logging.info("Matched rows after deduplication: %d", matched_after)
    logging.info("Unmatched rows after deduplication: %d", unmatched_after)
    logging.info("Interpretable rows after deduplication: %d", interpretable_after)
    logging.info("weapon_detected=true: %d", possible_weapon_after)
    logging.info("weapon_detected=false: %d", not_flagged_after)
    logging.info("weapon_detected=unknown: %d", unknown_after)
    logging.info("Normalized predictions: %s", normalized_predictions_path)
    logging.info("Export audit: %s", export_audit_path)
    logging.info("Tool version log: %s", tool_version_log_path)
    logging.info("Normalization summary: %s", normalization_summary_path)
    logging.info("Metrics: %s", forensic_tool_metrics_path)


if __name__ == "__main__":
    main()