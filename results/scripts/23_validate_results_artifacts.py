#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the frozen public result layer without regenerating any metric.

The validator checks the canonical commercial-tool prediction table, frozen
commercial metrics, proxy evaluation summary, OOD accounting, Chapter 5 figure
manifest, and reporting provenance. It is read-only unless ``--report`` is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_PREDICTIONS = (
    REPO_ROOT / "evaluation" / "forensic_tools" / "normalized_predictions.csv"
)
CANONICAL_SUMMARY = (
    REPO_ROOT
    / "evaluation"
    / "forensic_tools"
    / "normalized_predictions_public_summary.json"
)
EQUIVALENCE_REPORT = REPO_ROOT / "forensic_tools" / "public_extracts_validation.json"
FORENSIC_METRICS = REPO_ROOT / "results" / "metrics" / "forensic_tools_metrics.csv"
PROXY_SUMMARY = (
    REPO_ROOT / "results" / "metrics" / "proxy_model_evaluation_summary.json"
)
FINAL_OOD = REPO_ROOT / "results" / "metrics" / "final_ood_metrics.csv"
PROXY_OOD = REPO_ROOT / "results" / "metrics" / "proxy_model_ood_metrics.csv"
FIGURE_MANIFEST = (
    REPO_ROOT / "results" / "figures" / "chapter_5" / "chapter5_figures_manifest.csv"
)
FIGURE_SUMMARY = (
    REPO_ROOT / "results" / "figures" / "chapter_5" / "chapter5_figures_summary.json"
)
METADATA_SENSITIVITY_SUMMARY = (
    REPO_ROOT
    / "results"
    / "figures"
    / "chapter_5"
    / "embedded_metadata_sensitivity_summary.json"
)

EXPECTED_TOOL_COUNTS = {
    "magnet_axiom": 11_500,
    "excire_foto_2025_d20": 11_500,
    "excire_foto_2025_d50": 11_500,
    "excire_foto_2025_d80": 11_500,
    "cellebrite_inseyets": 11_500,
    "griffeye": 11_500,
}
EXPECTED_CANONICAL_COLUMNS = {
    "tool_name",
    "bundle_id",
    "matched",
    "match_method",
    "sample_type",
    "attack_family",
    "attack_name",
    "final_label",
    "weapon_detected",
    "normalized_prediction",
}
FORBIDDEN_CANONICAL_COLUMNS = {
    "raw_export_file",
    "raw_row_number",
    "raw_filename_or_path",
    "sha256",
    "md5",
    "metadata_json",
    "tool_input_filename",
}
LOCAL_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/run/media/|/home/|/Users/|raw_exports|blind_tool_input)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional repository-relative or absolute JSON report path.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def resolve_report(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_canonical_predictions() -> dict[str, Any]:
    rows, fields = read_csv(CANONICAL_PREDICTIONS)
    missing = sorted(EXPECTED_CANONICAL_COLUMNS - set(fields))
    forbidden = sorted(FORBIDDEN_CANONICAL_COLUMNS & set(fields))
    if missing:
        raise ValueError(f"Canonical predictions missing columns: {missing}")
    if forbidden:
        raise ValueError(f"Canonical predictions expose forbidden columns: {forbidden}")
    if len(rows) != 69_000:
        raise ValueError(f"Expected 69,000 canonical rows, found {len(rows)}")

    counts = Counter(row["tool_name"].strip() for row in rows)
    if dict(counts) != EXPECTED_TOOL_COUNTS:
        raise ValueError(
            f"Canonical tool profile mismatch: expected {EXPECTED_TOOL_COUNTS}, found {dict(counts)}"
        )

    keys = [(row["tool_name"].strip(), row["bundle_id"].strip()) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate tool_name/bundle_id keys in canonical predictions")

    for index, row in enumerate(rows, start=2):
        if row["matched"].strip().lower() != "true":
            raise ValueError(f"Non-matched canonical row at CSV line {index}")
        if row["weapon_detected"].strip().lower() not in {"true", "false"}:
            raise ValueError(f"Invalid weapon_detected at CSV line {index}")
        if row["normalized_prediction"].strip().lower() not in {
            "weapon",
            "non_weapon",
        }:
            raise ValueError(f"Invalid normalized_prediction at CSV line {index}")
        for value in row.values():
            if LOCAL_PATH_RE.search(str(value or "")):
                raise ValueError(f"Local/raw path leakage at CSV line {index}")

    digest = sha256_file(CANONICAL_PREDICTIONS)
    summary = read_json(CANONICAL_SUMMARY)
    output = summary.get("output", {})
    if output.get("rows") != 69_000:
        raise ValueError("Canonical summary does not report 69,000 rows")
    if output.get("sha256") != digest:
        raise ValueError("Canonical summary SHA256 does not match the committed CSV")
    if summary.get("decision_profile") != EXPECTED_TOOL_COUNTS:
        raise ValueError("Canonical summary decision profile mismatch")
    if summary.get("local_paths_detected") is not False:
        raise ValueError("Canonical summary does not confirm absence of local paths")
    if summary.get("raw_export_fields_included") is not False:
        raise ValueError("Canonical summary does not confirm raw-field exclusion")

    return {"rows": len(rows), "sha256": digest, "tool_profile": dict(counts)}


def validate_commercial_metrics(canonical_sha256: str) -> dict[str, Any]:
    metric_rows, _ = read_csv(FORENSIC_METRICS)
    if len(metric_rows) != 186:
        raise ValueError(f"Expected 186 commercial metric rows, found {len(metric_rows)}")

    report = read_json(EQUIVALENCE_REPORT)
    if report.get("decisions_identical") is not True:
        raise ValueError("Decision equivalence is not confirmed")
    if report.get("metrics_identical") is not True:
        raise ValueError("Metric equivalence is not confirmed")
    if report.get("source_rows") != 69_000:
        raise ValueError("Equivalence report source row count mismatch")
    if report.get("public_extract_rows") != 69_000:
        raise ValueError("Equivalence report public-extract row count mismatch")
    if report.get("metric_rows") != 186:
        raise ValueError("Equivalence report metric row count mismatch")
    if report.get("source_sha256") != canonical_sha256:
        raise ValueError("Equivalence report does not reference the committed canonical CSV")

    return {
        "rows": len(metric_rows),
        "sha256": sha256_file(FORENSIC_METRICS),
        "decisions_identical": True,
        "metrics_identical": True,
    }


def validate_proxy_summary() -> dict[str, Any]:
    payload = read_json(PROXY_SUMMARY)
    counts = payload.get("counts", {})
    expected = {
        "input_samples": 11_500,
        "prediction_rows": 40_500,
        "errors": 0,
        "final_core_metric_rows": 33,
        "final_robustness_metric_rows": 30,
        "final_confusion_matrix_rows": 33,
        "final_ood_metric_rows": 3,
    }
    for key, value in expected.items():
        if counts.get(key) != value:
            raise ValueError(
                f"Proxy summary mismatch for {key}: expected {value}, found {counts.get(key)}"
            )
    return expected


def validate_ood_accounting() -> dict[str, Any]:
    if FINAL_OOD.read_bytes() != PROXY_OOD.read_bytes():
        raise ValueError("final_ood_metrics.csv is not byte-identical to proxy_model_ood_metrics.csv")
    rows, fields = read_csv(FINAL_OOD)
    required = {
        "evaluated_model",
        "sample_type",
        "total",
        "predicted_weapon",
        "predicted_non_weapon",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"OOD metrics missing columns: {missing}")
    if len(rows) != 3:
        raise ValueError(f"Expected three OOD metric rows, found {len(rows)}")
    models = {row["evaluated_model"] for row in rows}
    if models != {"efficientnet_b0", "resnet18", "clip"}:
        raise ValueError(f"Unexpected OOD models: {sorted(models)}")
    for row in rows:
        total = int(row["total"])
        weapon = int(row["predicted_weapon"])
        non_weapon = int(row["predicted_non_weapon"])
        if row["sample_type"] != "ood":
            raise ValueError("Non-OOD row found in final_ood_metrics.csv")
        if total != 2_500:
            raise ValueError(
                f"Expected 2,500 fold-level OOD predictions per architecture, found {total}"
            )
        if weapon + non_weapon != total:
            raise ValueError("OOD predicted counts do not sum to total")
    return {
        "unique_ood_images": 500,
        "folds": 5,
        "predictions_per_architecture": 2_500,
        "architectures": 3,
        "total_ood_prediction_rows": 7_500,
        "interpretation": "500 unique OOD images x 5 fold-specific checkpoints per architecture",
    }


def validate_reporting_assets() -> dict[str, Any]:
    summary = read_json(FIGURE_SUMMARY)
    expected_script = "results/scripts/20_generate_experimental_reporting_assets.py"
    if summary.get("script") != expected_script:
        raise ValueError(
            f"Reporting summary script path mismatch: {summary.get('script')!r}"
        )
    if summary.get("figure_count_unique") != 24:
        raise ValueError("Expected 24 unique Chapter 5 reporting assets")
    if summary.get("file_count") != 41:
        raise ValueError("Expected 41 Chapter 5 manifest rows")

    manifest_rows, fields = read_csv(FIGURE_MANIFEST)
    required = {"figure_id", "output_path", "format", "source_csv"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Figure manifest missing columns: {missing}")
    if len(manifest_rows) != 41:
        raise ValueError(f"Expected 41 figure-manifest rows, found {len(manifest_rows)}")
    unique_ids = {row["figure_id"] for row in manifest_rows}
    if len(unique_ids) != 24:
        raise ValueError(f"Expected 24 unique figure IDs, found {len(unique_ids)}")
    missing_outputs = [
        row["output_path"]
        for row in manifest_rows
        if not (REPO_ROOT / row["output_path"]).is_file()
    ]
    if missing_outputs:
        raise ValueError(f"Missing generated reporting outputs: {missing_outputs[:10]}")

    metadata_summary = read_json(METADATA_SENSITIVITY_SUMMARY)
    counts = metadata_summary.get("counts", {})
    if counts.get("sensitive_metadata_hit_bundles") != 15:
        raise ValueError("Expected 15 sensitive metadata-hit bundles")
    if counts.get("prediction_rows") != 69_000:
        raise ValueError("Metadata sensitivity summary must reference 69,000 predictions")
    if counts.get("detail_rows") != 90:
        raise ValueError("Expected 90 metadata-sensitivity detail rows")

    return {
        "manifest_rows": len(manifest_rows),
        "unique_figure_ids": len(unique_ids),
        "metadata_sensitive_bundles": 15,
        "metadata_detail_rows": 90,
    }


def main() -> int:
    args = parse_args()
    canonical = validate_canonical_predictions()
    commercial = validate_commercial_metrics(canonical["sha256"])
    proxy = validate_proxy_summary()
    ood = validate_ood_accounting()
    reporting = validate_reporting_assets()

    report = {
        "schema_version": "1.0",
        "status": "passed",
        "canonical_commercial_predictions": canonical,
        "commercial_metrics": commercial,
        "proxy_evaluation": proxy,
        "ood_accounting": ood,
        "chapter5_reporting": reporting,
    }

    report_path = resolve_report(args.report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print("Results artifact validation passed.")
    print(" - canonical commercial decisions: 69000")
    print(" - commercial metric rows: 186")
    print(" - proxy prediction rows: 40500")
    print(" - OOD: 500 unique images x 5 folds = 2500 predictions per architecture")
    print(" - Chapter 5 manifest: 41 files, 24 unique asset IDs")
    if report_path is not None:
        print(f" - report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
