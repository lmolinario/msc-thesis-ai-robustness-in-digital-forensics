#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
21_generate_embedded_metadata_sensitivity_check.py

Generate the embedded-metadata sensitivity tables used in Chapter 5.

This reporting script connects two already consolidated FAIR-Lab artifacts:

1. datasets/forensic_evaluation_bundle/metadata/embedded_metadata_audit.csv
   Produced by script 16. It identifies blind bundle files whose embedded image
   metadata contain potentially informative or semantically sensitive terms.

2. evaluation/forensic_tools/normalized_predictions.csv
   Produced by script 19. It contains normalized black-box commercial-tool
   predictions for each bundle item.

The script does not rebuild the forensic bundle, does not modify images, and
does not recompute commercial-tool predictions. It only performs a leave-out
sensitivity check: it measures whether excluding the bundle items with sensitive
metadata hits materially changes clean accuracy and OOD weapon flag rate.

Default outputs:
- results/figures/chapter_5/tab_embedded_metadata_sensitive_hits_detail.csv
- results/figures/chapter_5/tab_embedded_metadata_tool_summary.csv
- results/figures/chapter_5/tab_embedded_metadata_sensitivity_delta.csv
- results/figures/chapter_5/embedded_metadata_sensitivity_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.utils.paths import EVALUATION_DIR, REPO_ROOT, RESULTS_DIR


SCRIPT_NAME = "results/scripts/21_generate_embedded_metadata_sensitivity_check.py"

DEFAULT_METADATA_AUDIT_CSV = (
    REPO_ROOT
    / "datasets"
    / "forensic_evaluation_bundle"
    / "metadata"
    / "embedded_metadata_audit.csv"
)
DEFAULT_NORMALIZED_PREDICTIONS_CSV = (
    EVALUATION_DIR / "forensic_tools" / "normalized_predictions.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "figures" / "chapter_5"

DETAIL_CSV_NAME = "tab_embedded_metadata_sensitive_hits_detail.csv"
TOOL_SUMMARY_CSV_NAME = "tab_embedded_metadata_tool_summary.csv"
DELTA_CSV_NAME = "tab_embedded_metadata_sensitivity_delta.csv"
SUMMARY_JSON_NAME = "embedded_metadata_sensitivity_summary.json"

TOOL_ORDER = (
    "magnet_axiom",
    "excire_foto_2025_d20",
    "excire_foto_2025_d50",
    "excire_foto_2025_d80",
    "cellebrite_inseyets",
    "griffeye",
)

TOOL_DISPLAY = {
    "magnet_axiom": "Magnet AXIOM / Magnet.AI",
    "excire_foto_2025": "Excire Foto 2025",
    "excire_foto_2025_d20": "Excire Foto 2025 D20",
    "excire_foto_2025_d50": "Excire Foto 2025 D50",
    "excire_foto_2025_d80": "Excire Foto 2025 D80",
    "cellebrite_inseyets": "Cellebrite Inseyets",
    "griffeye": "Magnet Griffeye / T3K CORE",
}


# =============================================================================
# Generic utilities
# =============================================================================


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_relative_string(path: Path | str) -> str:
    """Return a repository-relative POSIX path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def display_tool(tool_name: Any) -> str:
    """Return a thesis-friendly display name for a tool identifier."""
    return TOOL_DISPLAY.get(str(tool_name), str(tool_name).replace("_", " "))


def ordered_tools(available: set[str]) -> list[str]:
    """Return tools in preferred order, preserving unknown tools at the end."""
    ordered = [tool for tool in TOOL_ORDER if tool in available]
    remaining = sorted(tool for tool in available if tool not in set(ordered))
    return ordered + remaining


def ensure_columns(df: pd.DataFrame, required_columns: list[str], description: str) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {description}: {missing}")


def read_required_csv(path: Path, description: str) -> pd.DataFrame:
    """Read a required CSV file with a clear error on missing/empty input."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {description}: {repo_relative_string(path)}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Required CSV is empty: {repo_relative_string(path)}") from exc


def to_bool(value: Any) -> bool:
    """Convert common CSV boolean encodings to bool."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_text(value: Any) -> str:
    """Return a safe stripped string for possibly missing values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_bundle_id(value: Any) -> str | None:
    """Extract a bundle identifier from a string value."""
    match = re.search(r"(bundle_\d{6})", safe_text(value))
    return match.group(1) if match else None


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric if present."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV file creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s", repo_relative_string(path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %s", repo_relative_string(path))


# =============================================================================
# Metadata audit loading
# =============================================================================


def find_sensitive_column(audit_df: pd.DataFrame) -> str:
    """Return the column containing sensitive metadata hits."""
    candidates = [
        "sensitive_hits",
        "sensitive_terms",
        "metadata_sensitive_hits",
        "hits",
    ]
    for candidate in candidates:
        if candidate in audit_df.columns:
            return candidate

    heuristic = [
        column
        for column in audit_df.columns
        if "sensitive" in column.lower() and "hit" in column.lower()
    ]
    if heuristic:
        return heuristic[0]

    raise ValueError(
        "Could not find a sensitive-hit column in embedded metadata audit CSV."
    )


def add_bundle_id_to_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Add bundle_id to audit rows, extracting it from available path columns."""
    out = audit_df.copy()
    if "bundle_id" in out.columns:
        out["bundle_id"] = out["bundle_id"].astype(str)
        return out

    candidate_columns = [
        "relative_path",
        "tool_input_filename",
        "filename",
        "path",
        "file_path",
    ]
    available = [column for column in candidate_columns if column in out.columns]
    if not available:
        available = list(out.columns)

    out["bundle_id"] = None
    for column in available:
        extracted = out[column].map(extract_bundle_id)
        out["bundle_id"] = out["bundle_id"].where(out["bundle_id"].notna(), extracted)

    return out


def load_sensitive_metadata_hits(audit_csv: Path) -> pd.DataFrame:
    """Load embedded metadata audit rows with non-empty sensitive hits."""
    audit_df = read_required_csv(audit_csv, "embedded metadata audit CSV")
    sensitive_column = find_sensitive_column(audit_df)

    audit_df = add_bundle_id_to_audit(audit_df)
    audit_df[sensitive_column] = audit_df[sensitive_column].fillna("").astype(str)

    sensitive_df = audit_df[audit_df[sensitive_column].str.strip().ne("")].copy()
    sensitive_df = sensitive_df.dropna(subset=["bundle_id"]).copy()
    sensitive_df["bundle_id"] = sensitive_df["bundle_id"].astype(str)

    if sensitive_df.empty:
        raise ValueError("No sensitive metadata hits found in embedded metadata audit CSV.")

    if sensitive_column != "sensitive_hits":
        sensitive_df["sensitive_hits"] = sensitive_df[sensitive_column]

    sensitive_df["sensitive_hit_count"] = sensitive_df["sensitive_hits"].map(
        lambda value: len([item for item in str(value).split(";") if item.strip()])
    )

    # Keep stable, thesis-friendly audit columns first.
    preferred_columns = [
        "bundle_id",
        "relative_path",
        "suffix",
        "has_embedded_metadata",
        "sensitive_hits",
        "sensitive_hit_count",
        "metadata_keys",
        "metadata_json",
    ]
    columns = [column for column in preferred_columns if column in sensitive_df.columns]
    remaining = [column for column in sensitive_df.columns if column not in columns]

    return sensitive_df[columns + remaining].copy()


# =============================================================================
# Prediction loading and metrics
# =============================================================================


def load_predictions(predictions_csv: Path, tools: list[str] | None) -> pd.DataFrame:
    """Load normalized commercial-tool predictions."""
    predictions_df = read_required_csv(predictions_csv, "normalized predictions CSV")
    ensure_columns(
        predictions_df,
        [
            "tool_name",
            "bundle_id",
            "sample_type",
            "final_label",
            "normalized_prediction",
        ],
        "normalized predictions CSV",
    )

    predictions_df = predictions_df.copy()
    predictions_df["tool_name"] = predictions_df["tool_name"].astype(str)
    predictions_df["bundle_id"] = predictions_df["bundle_id"].astype(str)

    if tools:
        requested = set(tools)
        predictions_df = predictions_df[predictions_df["tool_name"].isin(requested)].copy()
        missing_tools = sorted(requested - set(predictions_df["tool_name"]))
        if missing_tools:
            raise ValueError(f"Requested tool(s) not found in predictions CSV: {missing_tools}")

    if predictions_df.empty:
        raise ValueError("No prediction rows available after tool filtering.")

    # Remove accidental duplicate tool/bundle rows while preserving the first
    # normalized decision emitted by script 19.
    predictions_df = predictions_df.drop_duplicates(subset=["tool_name", "bundle_id"]).copy()

    if "weapon_detected" in predictions_df.columns:
        predictions_df["weapon_detected_bool"] = predictions_df["weapon_detected"].map(to_bool)
    else:
        predictions_df["weapon_detected_bool"] = (
            predictions_df["normalized_prediction"].astype(str) == "weapon"
        )

    if "correct" in predictions_df.columns:
        predictions_df["correct_bool"] = predictions_df["correct"].map(to_bool)
    else:
        predictions_df["correct_bool"] = (
            predictions_df["normalized_prediction"].astype(str)
            == predictions_df["final_label"].astype(str)
        )

    if "false_negative" in predictions_df.columns:
        predictions_df["false_negative_bool"] = predictions_df["false_negative"].map(to_bool)
    else:
        predictions_df["false_negative_bool"] = (
            (predictions_df["final_label"].astype(str) == "weapon")
            & (predictions_df["normalized_prediction"].astype(str) != "weapon")
        )

    if "false_positive" in predictions_df.columns:
        predictions_df["false_positive_bool"] = predictions_df["false_positive"].map(to_bool)
    else:
        predictions_df["false_positive_bool"] = (
            (predictions_df["final_label"].astype(str) == "non_weapon")
            & (predictions_df["normalized_prediction"].astype(str) == "weapon")
        )

    predictions_df["tool_display"] = predictions_df["tool_name"].map(display_tool)
    return predictions_df


def build_sensitive_detail_table(
    sensitive_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per sensitive-hit bundle and commercial tool/configuration."""
    detail_df = sensitive_df.merge(
        predictions_df,
        on="bundle_id",
        how="left",
        suffixes=("_audit", "_prediction"),
        indicator=True,
    )

    detail_df["matched_prediction"] = detail_df["_merge"].astype(str) == "both"
    detail_df = detail_df.drop(columns=["_merge"])

    preferred_columns = [
        "bundle_id",
        "relative_path",
        "suffix",
        "has_embedded_metadata",
        "sensitive_hits",
        "sensitive_hit_count",
        "tool_name",
        "tool_display",
        "matched_prediction",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
        "normalized_prediction",
        "weapon_detected",
        "weapon_detected_bool",
        "correct",
        "correct_bool",
        "false_negative",
        "false_negative_bool",
        "false_positive",
        "false_positive_bool",
        "tool_raw_label",
        "tool_raw_confidence",
        "tool_confidence_numeric",
        "mapping_reason",
        "raw_export_file",
        "raw_row_number",
        "raw_filename_or_path",
        "metadata_keys",
        "metadata_json",
    ]
    columns = [column for column in preferred_columns if column in detail_df.columns]
    remaining = [column for column in detail_df.columns if column not in columns]

    detail_df = detail_df[columns + remaining].copy()

    if "tool_name" in detail_df.columns:
        tool_order = {tool: idx for idx, tool in enumerate(ordered_tools(set(detail_df["tool_name"].dropna().astype(str))))}
        detail_df["_tool_order"] = detail_df["tool_name"].map(tool_order).fillna(9999)
        detail_df = detail_df.sort_values(["bundle_id", "_tool_order"]).drop(columns=["_tool_order"])

    return detail_df.reset_index(drop=True)


def metric_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Return a metric mean as float, or None for empty input."""
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].mean()
    if pd.isna(value):
        return None
    return float(value)


def metric_count(frame: pd.DataFrame, column: str) -> int:
    """Return the number of true values for a boolean column."""
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def build_tool_summary_and_delta_tables(
    detail_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-tool embedded-metadata summary and leave-out delta tables."""
    if "tool_name" not in detail_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    available_tools = ordered_tools(set(predictions_df["tool_name"].astype(str)))

    for tool in available_tools:
        tool_predictions = predictions_df[predictions_df["tool_name"].astype(str) == tool].copy()
        tool_detail = detail_df[detail_df["tool_name"].astype(str) == tool].copy()

        clean_detail = tool_detail[tool_detail["sample_type"].astype(str) == "clean"].copy()
        ood_detail = tool_detail[tool_detail["sample_type"].astype(str) == "ood"].copy()

        all_clean = tool_predictions[tool_predictions["sample_type"].astype(str) == "clean"].copy()
        all_ood = tool_predictions[tool_predictions["sample_type"].astype(str) == "ood"].copy()

        sensitive_clean_ids = set(clean_detail["bundle_id"].dropna().astype(str))
        sensitive_ood_ids = set(ood_detail["bundle_id"].dropna().astype(str))

        clean_without = all_clean[~all_clean["bundle_id"].astype(str).isin(sensitive_clean_ids)].copy()
        ood_without = all_ood[~all_ood["bundle_id"].astype(str).isin(sensitive_ood_ids)].copy()

        clean_original = metric_mean(all_clean, "correct_bool")
        clean_without_value = metric_mean(clean_without, "correct_bool")
        ood_original = metric_mean(all_ood, "weapon_detected_bool")
        ood_without_value = metric_mean(ood_without, "weapon_detected_bool")

        clean_delta = (
            None
            if clean_original is None or clean_without_value is None
            else clean_without_value - clean_original
        )
        ood_delta = (
            None
            if ood_original is None or ood_without_value is None
            else ood_without_value - ood_original
        )

        summary_rows.append(
            {
                "tool_name": tool,
                "tool_display": display_tool(tool),
                "sensitive_hit_rows": int(len(tool_detail)),
                "clean_sensitive_cases": int(len(clean_detail)),
                "clean_sensitive_correct": metric_count(clean_detail, "correct_bool"),
                "clean_sensitive_false_negatives": metric_count(clean_detail, "false_negative_bool"),
                "clean_sensitive_false_positives": metric_count(clean_detail, "false_positive_bool"),
                "ood_sensitive_cases": int(len(ood_detail)),
                "ood_sensitive_weapon_flags": metric_count(ood_detail, "weapon_detected_bool"),
                "clean_accuracy_original": clean_original,
                "clean_accuracy_without_sensitive_hits": clean_without_value,
                "clean_accuracy_delta": clean_delta,
                "ood_weapon_flag_rate_original": ood_original,
                "ood_weapon_flag_rate_without_sensitive_hits": ood_without_value,
                "ood_weapon_flag_rate_delta": ood_delta,
            }
        )

        delta_rows.extend(
            [
                {
                    "tool_name": tool,
                    "tool_display": display_tool(tool),
                    "metric": "clean_accuracy",
                    "original": clean_original,
                    "without_sensitive_hits": clean_without_value,
                    "delta": clean_delta,
                    "excluded_cases": int(len(clean_detail)),
                    "original_n": int(len(all_clean)),
                    "without_sensitive_hits_n": int(len(clean_without)),
                },
                {
                    "tool_name": tool,
                    "tool_display": display_tool(tool),
                    "metric": "ood_weapon_flag_rate",
                    "original": ood_original,
                    "without_sensitive_hits": ood_without_value,
                    "delta": ood_delta,
                    "excluded_cases": int(len(ood_detail)),
                    "original_n": int(len(all_ood)),
                    "without_sensitive_hits_n": int(len(ood_without)),
                },
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    delta_df = pd.DataFrame(delta_rows)

    return summary_df, delta_df


# =============================================================================
# Console report
# =============================================================================


def print_console_report(
    sensitive_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    """Print a compact reporting summary to stdout."""
    print("\n" + "#" * 100)
    print("EMBEDDED METADATA SENSITIVITY CHECK")
    print("#" * 100)

    unique_sensitive = sensitive_df["bundle_id"].nunique()
    print(f"Sensitive metadata hit bundles: {unique_sensitive}")
    print(f"Sensitive metadata hit rows:    {len(sensitive_df)}")

    print("\nPer-tool summary:")
    cols = [
        "tool_name",
        "tool_display",
        "clean_sensitive_cases",
        "clean_sensitive_correct",
        "clean_sensitive_false_negatives",
        "clean_sensitive_false_positives",
        "ood_sensitive_cases",
        "ood_sensitive_weapon_flags",
        "clean_accuracy_delta",
        "ood_weapon_flag_rate_delta",
    ]
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        80,
    ):
        print(summary_df[[column for column in cols if column in summary_df.columns]].to_string(index=False))

    print("\nLeave-out sensitivity deltas:")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        80,
    ):
        print(delta_df.to_string(index=False))

    print("\n" + "#" * 100)
    print("END OF EMBEDDED METADATA SENSITIVITY CHECK")
    print("#" * 100 + "\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chapter 5 embedded-metadata sensitivity tables by joining "
            "the bundle metadata audit with normalized commercial-tool predictions."
        )
    )
    parser.add_argument("--metadata-audit", default=str(DEFAULT_METADATA_AUDIT_CSV))
    parser.add_argument("--normalized-predictions", default=str(DEFAULT_NORMALIZED_PREDICTIONS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help=(
            "Optional list of tool_name values to include. Default: all tools "
            "available in the normalized predictions CSV."
        ),
    )
    parser.add_argument(
        "--no-console-report",
        action="store_true",
        help="Do not print the summary tables to stdout.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    metadata_audit_csv = resolve_repo_path(args.metadata_audit)
    normalized_predictions_csv = resolve_repo_path(args.normalized_predictions)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading embedded metadata audit.")
    sensitive_df = load_sensitive_metadata_hits(metadata_audit_csv)

    logging.info("Reading normalized commercial-tool predictions.")
    predictions_df = load_predictions(normalized_predictions_csv, args.tools)

    logging.info("Building embedded metadata sensitivity tables.")
    detail_df = build_sensitive_detail_table(sensitive_df, predictions_df)
    tool_summary_df, delta_df = build_tool_summary_and_delta_tables(detail_df, predictions_df)

    detail_csv = output_dir / DETAIL_CSV_NAME
    tool_summary_csv = output_dir / TOOL_SUMMARY_CSV_NAME
    delta_csv = output_dir / DELTA_CSV_NAME
    summary_json = output_dir / SUMMARY_JSON_NAME

    write_csv(detail_csv, detail_df)
    write_csv(tool_summary_csv, tool_summary_df)
    write_csv(delta_csv, delta_df)

    summary_payload = {
        "script": SCRIPT_NAME,
        "created_at": utc_now_iso(),
        "inputs": {
            "metadata_audit_csv": repo_relative_string(metadata_audit_csv),
            "normalized_predictions_csv": repo_relative_string(normalized_predictions_csv),
        },
        "outputs": {
            "output_dir": repo_relative_string(output_dir),
            "detail_csv": repo_relative_string(detail_csv),
            "tool_summary_csv": repo_relative_string(tool_summary_csv),
            "delta_csv": repo_relative_string(delta_csv),
            "summary_json": repo_relative_string(summary_json),
        },
        "tools": ordered_tools(set(predictions_df["tool_name"].astype(str))),
        "counts": {
            "sensitive_metadata_hit_bundles": int(sensitive_df["bundle_id"].nunique()),
            "sensitive_metadata_hit_rows": int(len(sensitive_df)),
            "prediction_rows": int(len(predictions_df)),
            "detail_rows": int(len(detail_df)),
        },
        "methodological_note": (
            "This script joins metadata-audit hits with already normalized "
            "commercial-tool predictions and computes leave-out sensitivity "
            "deltas. It does not infer causal influence of embedded metadata on "
            "black-box tool decisions."
        ),
    }
    write_json(summary_json, summary_payload)

    if not args.no_console_report:
        print_console_report(sensitive_df, tool_summary_df, delta_df)

    logging.info("Completed embedded metadata sensitivity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
