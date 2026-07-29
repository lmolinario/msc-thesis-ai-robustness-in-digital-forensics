#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the minimized XAI public artifact used in the results chapter."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = REPO_ROOT / "explainability/manifests/chapter5/thesis_selection.csv"
THESIS_FILE = REPO_ROOT / "docs/LatexThesis/sections/06_results.tex"
EXPECTED_CASES = {
    "xai_case_0001": ("clean_correct_weapon", 1.0, "fig:xai-case1-clean-correct"),
    "xai_case_0006": ("clean_false_negative_weapon", 0.6920745372772217, "fig:xai-case2-clean-false-negative"),
    "xai_case_0009": ("ood_as_weapon", 0.9990302324295044, "fig:xai-case3-ood-as-weapon"),
    "xai_case_0010": ("anti_forensic_failure", 0.8515904545783997, "fig:xai-case4-antiforensic-failure"),
    "xai_case_0015": ("adversarial_high_conf_failure", 1.0, "fig:xai-case5-adversarial-high-confidence-failure"),
}
LOCAL_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|/run/media/|/home/|/Users/|lello|molinario)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--regenerated-manifest", default="")
    parser.add_argument("--strict-thesis-text", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def figure_max_probability(content: str, figure_label: str) -> str:
    pattern = re.compile(
        rf"\\XAIcaseFigureMaskGrid\s*"
        rf"\{{{re.escape(figure_label)}\}}"
        rf"(?:(?!\\XAIcaseFigureMaskGrid).)*?"
        rf"\\textbf\{{(?:confidence|Max-P)\}}\s*:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)"
        rf"\s*\}}",
        re.DOTALL,
    )
    matches = pattern.findall(content)
    if len(matches) != 1:
        fail(f"Expected one Max-P value for figure {figure_label}, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    selection = Path(args.selection)
    if not selection.is_absolute():
        selection = REPO_ROOT / selection

    rows = list(csv.DictReader(selection.open(encoding="utf-8")))
    if len(rows) != 5:
        fail(f"Expected 5 thesis cases, found {len(rows)}")

    ids = {row["case_id"] for row in rows}
    if ids != set(EXPECTED_CASES):
        fail(f"Unexpected case IDs: {sorted(ids)}")

    for row in rows:
        case_id = row["case_id"]
        bucket, expected_probability, _ = EXPECTED_CASES[case_id]
        if row["case_bucket"] != bucket:
            fail(f"Bucket mismatch for {case_id}")
        if abs(float(row["confidence"]) - expected_probability) > 1e-9:
            fail(f"Canonical probability mismatch for {case_id}")

        for column in ("input_asset", "heatmap_asset", "overlay_asset", "top10_mask_asset"):
            value = row[column]
            if LOCAL_PATTERN.search(value):
                fail(f"Local identifier in {column}: {value}")
            if not (REPO_ROOT / value).is_file():
                fail(f"Missing thesis asset: {value}")

    warnings: list[str] = []
    content = THESIS_FILE.read_text(encoding="utf-8")
    for case_id, (_, expected_probability, figure_label) in EXPECTED_CASES.items():
        current = figure_max_probability(content, figure_label)
        expected = f"{expected_probability:.3f}"
        if current != expected:
            message = f"Thesis reports {current} for {case_id}; expected {expected}"
            if args.strict_thesis_text:
                fail(message)
            warnings.append(message)

    if args.regenerated_manifest:
        manifest = Path(args.regenerated_manifest)
        if not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        generated = list(csv.DictReader(manifest.open(encoding="utf-8")))
        by_id = {row["case_id"]: row for row in generated if row.get("case_id") in EXPECTED_CASES}
        if set(by_id) != set(EXPECTED_CASES):
            fail("Regenerated manifest does not contain all five thesis cases")
        for case_id, row in by_id.items():
            if row.get("convergence_status") != "passed":
                fail(f"Adaptive convergence not passed for {case_id}")
            if not row.get("checkpoint_sha256") or not row.get("input_sha256"):
                fail(f"Missing integrity hashes for {case_id}")

    print("Results-chapter XAI public-artifact validation passed.")
    print(" - thesis cases: 5")
    print(" - thesis assets: 20")
    print(" - local path leakage: none")
    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
