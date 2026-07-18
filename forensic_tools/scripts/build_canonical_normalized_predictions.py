#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the canonical sanitized commercial-tool prediction table.

The four committed public extracts are already validated against the frozen
69,000-row local normalization output and the 186-row metric table. This script
combines those extracts into one repository-friendly prediction-level artifact:

    evaluation/forensic_tools/normalized_predictions.csv

The generated CSV contains no raw export paths, device identifiers, hashes,
file-system metadata, or unrelated commercial-tool fields. It is intended to be
tracked on ``main`` and used by downstream reporting scripts that need one
canonical prediction table.
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
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "forensic_tools" / "normalized_predictions.csv"
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "evaluation"
    / "forensic_tools"
    / "normalized_predictions_public_summary.json"
)

EXTRACT_PATHS = (
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
)

TOOL_ORDER = (
    "magnet_axiom",
    "excire_foto_2025_d20",
    "excire_foto_2025_d50",
    "excire_foto_2025_d80",
    "cellebrite_inseyets",
    "griffeye",
)

EXPECTED_COUNTS = {tool: 11500 for tool in TOOL_ORDER}

COMMON_INPUT_FIELDS = (
    "tool_name",
    "bundle_id",
    "sample_type",
    "attack_family",
    "attack_name",
    "final_label",
    "weapon_detected",
    "normalized_prediction",
)

TOOL_SIGNAL_FIELDS = (
    "tags",
    "classifications",
    "excire_distance_limit",
    "n_prompt_hits",
    "hit_prompts",
    "prompt_firearm_hit",
    "prompt_gun_hit",
    "prompt_pistol_hit",
    "prompt_handgun_hit",
    "prompt_revolver_hit",
    "prompt_rifle_hit",
    "prompt_shotgun_hit",
    "prompt_assault_rifle_hit",
    "firearm_bookmark",
    "secondary_weapon_bookmarks",
)

OUTPUT_FIELDS = (
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
    *TOOL_SIGNAL_FIELDS,
)

LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:home|Users|run/media)/", re.IGNORECASE),
    re.compile(r"\\(?:Users|blind_tool_input|raw_exports)\\", re.IGNORECASE),
    re.compile(r"/(?:blind_tool_input|raw_exports)/", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("_x000d_", "\n")
    text = re.sub(r"[\r\n]+", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return text.strip(" | \t")


def read_extract(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing public extract: {repo_relative(path)}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = sorted(set(COMMON_INPUT_FIELDS) - set(fields))
    if missing:
        raise ValueError(f"Extract {repo_relative(path)} is missing columns: {missing}")
    return rows, fields


def canonical_row(row: dict[str, str]) -> dict[str, str]:
    output = {field: clean_text(row.get(field, "")) for field in OUTPUT_FIELDS}
    output["matched"] = "true"
    output["match_method"] = "validated_public_extract"
    output["weapon_detected"] = output["weapon_detected"].lower()
    output["normalized_prediction"] = output["normalized_prediction"].lower()
    return output


def contains_local_path(value: str) -> bool:
    return any(pattern.search(value) for pattern in LOCAL_PATH_PATTERNS)


def validate_rows(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["tool_name"] for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"Canonical profile mismatch: expected {EXPECTED_COUNTS}, found {dict(counts)}"
        )

    keys = [(row["tool_name"], row["bundle_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate tool_name/bundle_id pairs detected")
    if any(not tool or not bundle for tool, bundle in keys):
        raise ValueError("Empty tool_name or bundle_id detected")

    for index, row in enumerate(rows, start=1):
        if row["matched"] != "true":
            raise ValueError(f"Unexpected matched value at row {index}: {row['matched']!r}")
        if row["weapon_detected"] not in {"true", "false"}:
            raise ValueError(
                f"Unexpected weapon_detected value at row {index}: "
                f"{row['weapon_detected']!r}"
            )
        if row["normalized_prediction"] not in {"weapon", "non_weapon"}:
            raise ValueError(
                f"Unexpected normalized_prediction at row {index}: "
                f"{row['normalized_prediction']!r}"
            )
        if row["final_label"] not in {"weapon", "non_weapon", "ood"}:
            raise ValueError(
                f"Unexpected final_label at row {index}: {row['final_label']!r}"
            )
        for field, value in row.items():
            if contains_local_path(value):
                raise ValueError(
                    f"Local or raw-export path detected at row {index}, "
                    f"field {field}: {value!r}"
                )


def write_csv(path: Path, rows: list[dict[str, str]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Use --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(OUTPUT_FIELDS),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    extract_metadata: list[dict[str, Any]],
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Summary already exists: {path}. Use --force to replace it.")
    payload = {
        "schema_version": "1.0",
        "artifact_status": "canonical_sanitized_prediction_table",
        "output": {
            "path": repo_relative(output_path),
            "rows": len(rows),
            "sha256": sha256_file(output_path),
            "columns": list(OUTPUT_FIELDS),
        },
        "sources": extract_metadata,
        "decision_profile": EXPECTED_COUNTS,
        "local_paths_detected": False,
        "raw_export_fields_included": False,
        "validation_command": (
            "python forensic_tools/scripts/validate_public_extract_equivalence.py "
            "--source evaluation/forensic_tools/normalized_predictions.csv --force"
        ),
        "methodological_note": (
            "The canonical table is reconstructed only from the four committed, "
            "validated sanitized public extracts. It does not contain complete "
            "commercial-tool exports or local acquisition paths."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary)

    all_rows: list[dict[str, str]] = []
    extract_metadata: list[dict[str, Any]] = []
    for extract_path in EXTRACT_PATHS:
        rows, fields = read_extract(extract_path)
        all_rows.extend(canonical_row(row) for row in rows)
        extract_metadata.append(
            {
                "path": repo_relative(extract_path),
                "rows": len(rows),
                "sha256": sha256_file(extract_path),
                "columns": fields,
            }
        )

    order = {tool: index for index, tool in enumerate(TOOL_ORDER)}
    all_rows.sort(key=lambda row: (order[row["tool_name"]], row["bundle_id"]))
    validate_rows(all_rows)
    write_csv(output_path, all_rows, args.force)
    write_summary(summary_path, output_path, all_rows, extract_metadata, args.force)

    print("Canonical sanitized normalized predictions generated.")
    print(f" - rows: {len(all_rows)}")
    print(f" - output: {repo_relative(output_path)}")
    print(f" - sha256: {sha256_file(output_path)}")
    print(f" - summary: {repo_relative(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
