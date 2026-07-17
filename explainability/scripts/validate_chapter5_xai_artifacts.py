#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the minimized Chapter 5 XAI public artifact."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = REPO_ROOT / "explainability/manifests/chapter5/thesis_selection.csv"
EXPECTED_CASES = {
    "xai_case_0001": ("clean_correct_weapon", 1.0),
    "xai_case_0006": ("clean_false_negative_weapon", 0.6920745372772217),
    "xai_case_0009": ("ood_as_weapon", 0.9990302324295044),
    "xai_case_0010": ("anti_forensic_failure", 0.8515904545783997),
    "xai_case_0015": ("adversarial_high_conf_failure", 1.0),
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
        bucket, confidence = EXPECTED_CASES[case_id]
        if row["case_bucket"] != bucket:
            fail(f"Bucket mismatch for {case_id}")
        if abs(float(row["confidence"]) - confidence) > 1e-9:
            fail(f"Confidence mismatch for {case_id}")
        for column in ("input_asset", "heatmap_asset", "overlay_asset", "top10_mask_asset"):
            value = row[column]
            if LOCAL_PATTERN.search(value):
                fail(f"Local identifier in {column}: {value}")
            if not (REPO_ROOT / value).is_file():
                fail(f"Missing thesis asset: {value}")

    warnings = []
    english = (REPO_ROOT / "docs/LatexThesis/sections/05_experiments.tex").read_text(encoding="utf-8")
    italian = (REPO_ROOT / "docs/LatexThesis_ITA/sections/05_experiments.tex").read_text(encoding="utf-8")
    for label, content in (("English", english), ("Italian", italian)):
        if "xai\\_case\\_0010" in content and "0.870" in content:
            message = f"{label} thesis still reports legacy rounded confidence 0.870; expected 0.852"
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

    print("Chapter 5 XAI public-artifact validation passed.")
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
