#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synchronize results-chapter XAI Max-P values with the canonical manifest."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION = REPO_ROOT / "explainability/manifests/chapter5/thesis_selection.csv"
TEX_FILE = REPO_ROOT / "docs/LatexThesis/sections/06_results.tex"
FIGURE_LABELS = {
    "xai_case_0001": "fig:xai-case1-clean-correct",
    "xai_case_0006": "fig:xai-case2-clean-false-negative",
    "xai_case_0009": "fig:xai-case3-ood-as-weapon",
    "xai_case_0010": "fig:xai-case4-antiforensic-failure",
    "xai_case_0015": "fig:xai-case5-adversarial-high-confidence-failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Fail when the TeX metadata differs from the manifest.")
    mode.add_argument("--write", action="store_true", help="Update the TeX metadata in place.")
    return parser.parse_args()


def expected_values() -> dict[str, str]:
    rows = list(csv.DictReader(SELECTION.open(encoding="utf-8")))
    values = {row["case_id"]: f"{float(row['confidence']):.3f}" for row in rows}
    missing = set(FIGURE_LABELS) - set(values)
    if missing:
        raise RuntimeError(f"Missing canonical XAI cases: {sorted(missing)}")
    return values


def figure_max_probability_pattern(figure_label: str) -> re.Pattern[str]:
    return re.compile(
        rf"(\\XAIcaseFigureMaskGrid\s*"
        rf"\{{{re.escape(figure_label)}\}}"
        rf"(?:(?!\\XAIcaseFigureMaskGrid).)*?"
        rf"\\textbf\{{(?:confidence|Max-P)\}}\s*:\s*)"
        rf"([0-9]+(?:\.[0-9]+)?)"
        rf"(\s*\}})",
        re.DOTALL,
    )


def synchronize(path: Path, expected: dict[str, str], write: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    updated = text
    changes: list[str] = []

    for case_id, figure_label in FIGURE_LABELS.items():
        pattern = figure_max_probability_pattern(figure_label)
        matches = list(pattern.finditer(updated))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one XAI figure metadata Max-P value for {case_id} "
                f"in {path}, found {len(matches)}"
            )

        current = matches[0].group(2)
        target = expected[case_id]
        if current != target:
            changes.append(f"{case_id}: {current} -> {target}")
            updated = pattern.sub(rf"\g<1>{target}\g<3>", updated, count=1)

    if write and updated != text:
        path.write_text(updated, encoding="utf-8")
    return changes


def main() -> None:
    args = parse_args()
    write = bool(args.write)
    expected = expected_values()
    changes = synchronize(TEX_FILE, expected, write=write)

    if changes:
        for change in changes:
            print(f"{TEX_FILE.relative_to(REPO_ROOT)} | {change}")
        if not write:
            raise SystemExit(1)
        print(f"Updated {len(changes)} results-chapter XAI metadata field(s).")
    else:
        print("Results-chapter XAI metadata is synchronized.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
