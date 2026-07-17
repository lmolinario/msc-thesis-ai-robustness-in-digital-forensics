#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimizing wrapper for commercial forensic-tool normalization.

The original validated implementation is preserved in
`_19_normalize_forensic_ai_tool_predictions_impl.py`. This entry point adds:

- canonical-output overwrite protection;
- minimized public fields for Cellebrite and Excire;
- explicit unmatched-row preservation;
- complete version metadata;
- strict JSON serialization;
- optional, rather than default, per-tool prediction copies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
IMPL_PATH = SCRIPT_DIR / "_19_normalize_forensic_ai_tool_predictions_impl.py"

OFFICIAL_OUTPUT_TOOL_NAMES = {
    "magnet_axiom",
    "excire_foto_2025_d20",
    "excire_foto_2025_d50",
    "excire_foto_2025_d80",
    "cellebrite_inseyets",
    "griffeye",
}


def load_implementation() -> Any:
    spec = importlib.util.spec_from_file_location("fairlab_tool_norm_impl", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load normalization implementation: {IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-per-tool-files", action="store_true")
    return parser.parse_known_args(argv)


def strict_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def single_line(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = text.replace("_x000d_", "\n")
    text = text.replace("\\r", "\n").replace("\\n", "\n")
    text = re.sub(r"[\r\n]+", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return text.strip(" |")


def validate_canonical_predictions(path: Path) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df) != 69000:
        raise ValueError(f"Expected 69,000 canonical predictions, found {len(df)}")
    counts = Counter(df["tool_name"])
    if set(counts) != OFFICIAL_OUTPUT_TOOL_NAMES:
        raise ValueError(f"Unexpected canonical tool set: {sorted(counts)}")
    invalid_counts = {name: count for name, count in counts.items() if count != 11500}
    if invalid_counts:
        raise ValueError(f"Incomplete canonical tool outputs: {invalid_counts}")
    if (df["weapon_detected"] == "unknown").any():
        raise ValueError("Canonical predictions contain unknown values")
    coverage = df.groupby("tool_name")["bundle_id"].nunique()
    if not coverage.eq(11500).all():
        raise ValueError(f"Incomplete bundle coverage: {coverage.to_dict()}")


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    impl = load_implementation()
    args = impl.parse_args()

    output_dir = impl.repo_relative_path(args.output_dir)
    metrics_dir = impl.repo_relative_path(args.metrics_dir)
    canonical = (
        output_dir.resolve() == impl.DEFAULT_OUTPUT_DIR.resolve()
        and metrics_dir.resolve() == impl.DEFAULT_METRICS_DIR.resolve()
    )

    if canonical and args.no_interactive and set(args.tools) != set(impl.KNOWN_TOOL_NAMES):
        raise ValueError(
            "Partial tool selections cannot write to canonical output directories. "
            "Choose alternative --output-dir and --metrics-dir values."
        )

    protected = [
        output_dir / "normalized_predictions.csv",
        output_dir / "unmatched_predictions.csv",
        metrics_dir / "forensic_tools_metrics.csv",
    ]
    if any(path.exists() for path in protected) and not wrapper_args.force:
        raise FileExistsError(
            "Normalization outputs already exist. Use --force only for an intentional regeneration."
        )

    impl.write_json = strict_write_json

    original_extract = impl.extract_raw_row

    def minimized_extract(
        tool_name: str,
        export_file: Path,
        row_number: int,
        record: dict[str, Any],
    ) -> Any:
        if tool_name != "cellebrite_inseyets":
            row = original_extract(tool_name, export_file, row_number, record)
            row.filename_or_path = impl.basename_from_path(row.filename_or_path)
            row.raw_label = single_line(row.raw_label)
            return row

        sha256 = impl.normalize_hash(impl.first_non_empty(record, impl.SHA256_COLUMNS))
        md5 = impl.normalize_hash(impl.first_non_empty(record, impl.MD5_COLUMNS))
        filename = impl.basename_from_path(
            impl.first_non_empty(record, impl.FILENAME_COLUMNS)
        )
        raw_label = impl.first_non_empty(
            record,
            {"Classifications", "Classification", "Classificazioni"},
        )
        raw_confidence = impl.first_non_empty(record, impl.CONFIDENCE_COLUMNS)
        return impl.RawToolRow(
            tool_name=tool_name,
            raw_export_file=impl.repo_relative_string(export_file),
            raw_row_number=row_number,
            raw_record={},
            sha256=sha256,
            md5=md5,
            filename_or_path=filename,
            raw_label=single_line(raw_label),
            raw_confidence=raw_confidence,
        )

    impl.extract_raw_row = minimized_extract

    original_excire = impl.normalize_excire_foto_2025_prompt_exports

    def minimized_excire(*args: Any, **kwargs: Any) -> Any:
        rows, audit_rows, raw_hit_rows = original_excire(*args, **kwargs)
        for row in rows:
            row["raw_filename_or_path"] = row.get("tool_input_filename", "")
        return rows, audit_rows, raw_hit_rows

    impl.normalize_excire_foto_2025_prompt_exports = minimized_excire

    original_version_row = impl.build_tool_version_row

    def complete_version_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_version_row(*args, **kwargs)
        base_name = kwargs.get("tool_name")
        if base_name is None and args:
            base_name = args[0]

        if base_name == "magnet_axiom":
            row.update(
                tool_version="10.1.0.48673",
                tool_build="Magnet AXIOM Process / Magnet.AI",
                case_name="FAIRLAB_AXIOM_RUN_02",
                export_status="completed",
                ai_modules_enabled="Magnet.AI media analysis: Weapons",
            )
        elif base_name == "excire_foto_2025":
            selected = kwargs.get("selected_run_dirs")
            if selected is None and len(args) >= 4:
                selected = args[3]
            row.update(
                tool_version="4.1.5",
                tool_build="Excire Foto 2025",
                case_name=(selected or [Path("FAIRLAB_EXCIRE")])[0].name,
                export_status="completed",
                ai_modules_enabled="semantic text search with fixed firearm prompts",
            )
        elif base_name == "cellebrite_inseyets":
            row["case_name"] = "FAIRLAB_CELLEBRITE_INSEYETS_RUN_01"
        return row

    impl.build_tool_version_row = complete_version_row

    unmatched_rows: list[dict[str, Any]] = []
    original_deduplicate = impl.deduplicate_predictions

    def capturing_deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unmatched_rows.extend(
            row
            for row in rows
            if impl.safe_str(row.get("matched", "")) != "true"
            or not impl.safe_str(row.get("bundle_id", ""))
        )
        return original_deduplicate(rows)

    impl.deduplicate_predictions = capturing_deduplicate

    original_write_csv = impl.write_csv

    def minimized_write_csv(
        path: Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str] | None = None,
    ) -> None:
        if (
            path.name.endswith("_normalized_predictions.csv")
            and path.name != "normalized_predictions.csv"
            and not wrapper_args.write_per_tool_files
        ):
            return
        sanitized: list[dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            if "raw_filename_or_path" in row:
                row["raw_filename_or_path"] = impl.basename_from_path(
                    row.get("raw_filename_or_path", "")
                )
            if "tool_raw_label" in row:
                row["tool_raw_label"] = single_line(row.get("tool_raw_label", ""))
            sanitized.append(row)
        original_write_csv(path, sanitized, fieldnames)

    impl.write_csv = minimized_write_csv

    impl.main()

    unmatched_path = output_dir / "unmatched_predictions.csv"
    original_write_csv(unmatched_path, unmatched_rows)

    summary_path = output_dir / "normalization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_tool = Counter(str(row.get("tool_name", "")) for row in unmatched_rows)
    summary["unmatched_rows_before_deduplication"] = len(unmatched_rows)
    summary["unmatched_rows_by_tool"] = {
        key: value for key, value in sorted(by_tool.items()) if key
    }
    summary.setdefault("outputs", {})["unmatched_predictions"] = (
        impl.repo_relative_string(unmatched_path)
    )
    strict_write_json(summary_path, summary)

    aggregate_path = output_dir / "normalized_predictions.csv"
    if canonical:
        validate_canonical_predictions(aggregate_path)
        if len(unmatched_rows) != 329 or set(by_tool) != {"cellebrite_inseyets"}:
            raise ValueError(
                "Frozen unmatched-row profile mismatch: "
                f"total={len(unmatched_rows)}, by_tool={dict(by_tool)}"
            )

    if not wrapper_args.write_per_tool_files:
        for tool_name in OFFICIAL_OUTPUT_TOOL_NAMES:
            (output_dir / f"{tool_name}_normalized_predictions.csv").unlink(
                missing_ok=True
            )


if __name__ == "__main__":
    main()
