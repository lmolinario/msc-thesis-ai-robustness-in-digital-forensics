#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate sanitized commercial-tool extracts against frozen local outputs.

The validator proves two properties before any raw export can be removed:

1. every ``tool_name``/``bundle_id`` decision and condition field is identical
   to the locally regenerated normalized prediction table;
2. recomputing the commercial-tool metrics from the public extracts produces
   exactly the frozen 186-row metric table committed under ``results/metrics``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "evaluation" / "forensic_tools" / "normalized_predictions.csv"
DEFAULT_METRICS = REPO_ROOT / "results" / "metrics" / "forensic_tools_metrics.csv"
DEFAULT_REPORT = REPO_ROOT / "forensic_tools" / "public_extracts_validation.json"

EXTRACT_PATHS = [
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
    "magnet_axiom": 11500,
    "excire_foto_2025_d20": 11500,
    "excire_foto_2025_d50": 11500,
    "excire_foto_2025_d80": 11500,
    "cellebrite_inseyets": 11500,
    "griffeye": 11500,
}

COMPARISON_FIELDS = [
    "sample_type",
    "attack_family",
    "attack_name",
    "final_label",
    "weapon_detected",
    "normalized_prediction",
]

METRIC_KEY_FIELDS = [
    "tool_name",
    "scope",
    "sample_type",
    "attack_family",
    "attack_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def ensure_profile(rows: list[dict[str, str]], label: str) -> None:
    counts = Counter(safe_str(row.get("tool_name")) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"{label} profile mismatch: expected {EXPECTED_COUNTS}, found {dict(counts)}"
        )
    keys = [(safe_str(row.get("tool_name")), safe_str(row.get("bundle_id"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate tool_name/bundle_id pairs")
    if any(not tool or not bundle for tool, bundle in keys):
        raise ValueError(f"{label} contains empty tool or bundle identifiers")


def decision_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (safe_str(row.get("tool_name")), safe_str(row.get("bundle_id"))): {
            field: safe_str(row.get(field)) for field in COMPARISON_FIELDS
        }
        for row in rows
    }


def compare_decisions(
    source_rows: list[dict[str, str]], public_rows: list[dict[str, str]]
) -> None:
    source = decision_map(source_rows)
    public = decision_map(public_rows)
    missing = sorted(set(source) - set(public))
    extra = sorted(set(public) - set(source))
    if missing or extra:
        raise ValueError(
            f"Decision key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    mismatches: list[dict[str, Any]] = []
    for key in sorted(source):
        if source[key] != public[key]:
            mismatches.append(
                {
                    "tool_name": key[0],
                    "bundle_id": key[1],
                    "source": source[key],
                    "public": public[key],
                }
            )
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise ValueError(
            "Sanitized extracts changed one or more decisions or condition fields: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def metric_value(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def compute_group_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    matched_rows = [row for row in rows if safe_str(row.get("matched")) == "true"]
    binary_rows = [
        row
        for row in matched_rows
        if safe_str(row.get("final_label")).lower() in {"weapon", "non_weapon"}
    ]
    binary_interpretable_rows = [
        row
        for row in binary_rows
        if safe_str(row.get("weapon_detected")) in {"true", "false"}
    ]

    tp = fp = tn = fn = 0
    for row in binary_interpretable_rows:
        label = safe_str(row.get("final_label")).lower()
        detected = safe_str(row.get("weapon_detected")).lower() == "true"
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
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )

    ood_rows = [
        row
        for row in matched_rows
        if safe_str(row.get("final_label")).lower() == "ood"
    ]
    ood_weapon_flags = sum(
        1 for row in ood_rows if safe_str(row.get("weapon_detected")) == "true"
    )
    ood_non_weapon_flags = sum(
        1 for row in ood_rows if safe_str(row.get("weapon_detected")) == "false"
    )
    ood_unknown = sum(
        1 for row in ood_rows if safe_str(row.get("weapon_detected")) == "unknown"
    )
    unknown_rows = sum(
        1 for row in matched_rows if safe_str(row.get("weapon_detected")) == "unknown"
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
        "ood_weapon_flag_rate": metric_value(
            safe_div(ood_weapon_flags, len(ood_rows))
        ),
    }


def add_metric_group(
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]],
    row: dict[str, str],
    scope: str,
    sample_type: str,
    attack_family: str,
    attack_name: str,
) -> None:
    key = (
        safe_str(row.get("tool_name")),
        scope,
        sample_type or "all",
        attack_family or "all",
        attack_name or "all",
    )
    groups[key].append(row)


def compute_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in rows:
        if safe_str(row.get("matched")) != "true":
            continue
        sample_type = safe_str(row.get("sample_type")) or "none"
        attack_family = safe_str(row.get("attack_family")) or "none"
        attack_name = safe_str(row.get("attack_name")) or "none"
        add_metric_group(groups, row, "all", "all", "all", "all")
        add_metric_group(groups, row, "sample_type", sample_type, "all", "all")
        add_metric_group(groups, row, "attack_family", "all", attack_family, "all")
        add_metric_group(
            groups, row, "attack_name", "all", attack_family, attack_name
        )
        add_metric_group(
            groups,
            row,
            "sample_type_attack",
            sample_type,
            attack_family,
            attack_name,
        )

    scope_order = {
        "all": 0,
        "sample_type": 1,
        "attack_family": 2,
        "attack_name": 3,
        "sample_type_attack": 4,
    }
    metric_rows: list[dict[str, Any]] = []
    for (tool_name, scope, sample_type, attack_family, attack_name), grouped_rows in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            scope_order.get(item[0][1], 99),
            item[0][2],
            item[0][3],
            item[0][4],
        ),
    ):
        metric_rows.append(
            {
                "tool_name": tool_name,
                "scope": scope,
                "sample_type": sample_type,
                "attack_family": attack_family,
                "attack_name": attack_name,
                **compute_group_metrics(grouped_rows),
            }
        )
    return metric_rows


def normalize_metric_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, str]:
    return {field: safe_str(row.get(field)) for field in fieldnames}


def metric_map(
    rows: list[dict[str, Any]], fieldnames: list[str]
) -> dict[tuple[str, ...], dict[str, str]]:
    normalized = [normalize_metric_row(row, fieldnames) for row in rows]
    return {
        tuple(row[field] for field in METRIC_KEY_FIELDS): row for row in normalized
    }


def compare_metrics(
    computed_rows: list[dict[str, Any]],
    frozen_rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    if len(computed_rows) != 186 or len(frozen_rows) != 186:
        raise ValueError(
            f"Expected 186 metric rows, computed={len(computed_rows)}, frozen={len(frozen_rows)}"
        )
    computed = metric_map(computed_rows, fieldnames)
    frozen = metric_map(frozen_rows, fieldnames)
    missing = sorted(set(frozen) - set(computed))
    extra = sorted(set(computed) - set(frozen))
    if missing or extra:
        raise ValueError(
            f"Metric key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    mismatches: list[dict[str, Any]] = []
    for key in sorted(frozen):
        if frozen[key] != computed[key]:
            changed_fields = {
                field: {
                    "frozen": frozen[key].get(field, ""),
                    "computed": computed[key].get(field, ""),
                }
                for field in fieldnames
                if frozen[key].get(field, "") != computed[key].get(field, "")
            }
            mismatches.append({"key": key, "changed_fields": changed_fields})
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise ValueError(
            "Sanitized extracts do not reproduce frozen metrics: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def main() -> None:
    args = parse_args()
    source_path = resolve_path(args.source)
    metrics_path = resolve_path(args.metrics)
    report_path = resolve_path(args.report)

    source_rows, _ = read_csv(source_path)
    public_rows: list[dict[str, str]] = []
    extract_hashes: dict[str, str] = {}
    for path in EXTRACT_PATHS:
        rows, fields = read_csv(path)
        required = {"tool_name", "bundle_id", *COMPARISON_FIELDS}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"Extract {path} is missing columns: {missing}")
        for row in rows:
            row["matched"] = "true"
        public_rows.extend(rows)
        extract_hashes[repo_relative(path)] = sha256_file(path)

    ensure_profile(source_rows, "source")
    ensure_profile(public_rows, "public extracts")
    compare_decisions(source_rows, public_rows)

    frozen_metrics, metric_fields = read_csv(metrics_path)
    computed_metrics = compute_metrics(public_rows)
    compare_metrics(computed_metrics, frozen_metrics, metric_fields)

    report = {
        "schema_version": "1.0",
        "decisions_identical": True,
        "metrics_identical": True,
        "source": repo_relative(source_path),
        "source_sha256": sha256_file(source_path),
        "source_rows": len(source_rows),
        "public_extract_rows": len(public_rows),
        "decision_profile": EXPECTED_COUNTS,
        "frozen_metrics": repo_relative(metrics_path),
        "frozen_metrics_sha256": sha256_file(metrics_path),
        "metric_rows": len(frozen_metrics),
        "extract_sha256": extract_hashes,
    }

    if report_path.exists() and not args.force:
        raise FileExistsError(
            f"Validation report already exists: {report_path}. Use --force to replace it."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Public extract equivalence validation passed.")
    print(f" - decisions: {len(public_rows)} identical rows")
    print(f" - metrics: {len(frozen_metrics)} identical rows")
    print(f" - report: {repo_relative(report_path)}")


if __name__ == "__main__":
    main()
