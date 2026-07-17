#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build minimized public extracts from locally regenerated tool predictions.

The source ``evaluation/forensic_tools/normalized_predictions.csv`` is not
tracked on ``main``. Regenerate it locally with step 19 before running this
script. The generated extracts contain only anonymized bundle identifiers,
experimental condition metadata, the observable tool signal needed for audit,
and the normalized binary decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "evaluation" / "forensic_tools" / "normalized_predictions.csv"
DEFAULT_SUMMARY = REPO_ROOT / "forensic_tools" / "public_extracts_summary.json"

EXPECTED_COUNTS = {
    "magnet_axiom": 11500,
    "excire_foto_2025_d20": 11500,
    "excire_foto_2025_d50": 11500,
    "excire_foto_2025_d80": 11500,
    "cellebrite_inseyets": 11500,
    "griffeye": 11500,
}

COMMON_FIELDS = [
    "tool_name",
    "bundle_id",
    "sample_type",
    "attack_family",
    "attack_name",
    "final_label",
    "weapon_detected",
    "normalized_prediction",
]

EXCIRE_PROMPT_FIELDS = [
    "prompt_firearm_hit",
    "prompt_gun_hit",
    "prompt_pistol_hit",
    "prompt_handgun_hit",
    "prompt_revolver_hit",
    "prompt_rifle_hit",
    "prompt_shotgun_hit",
    "prompt_assault_rifle_hit",
]

GRIFFEYE_PRIMARY = "CORE/Violence/Firearm"
GRIFFEYE_SECONDARY = [
    "CORE/Violence/Explosive Weapon",
    "CORE/Violence/Bladed Weapon",
    "CORE/Violence/Archery Weapon",
    "CORE/Military/Military Equipment",
]

OUTPUTS = {
    "magnet": REPO_ROOT
    / "forensic_tools"
    / "magnet_axiom"
    / "public_extracts"
    / "magnet_axiom_predictions_extract.csv",
    "excire": REPO_ROOT
    / "forensic_tools"
    / "excire_foto_2025"
    / "public_extracts"
    / "excire_prompt_hits_extract.csv",
    "cellebrite": REPO_ROOT
    / "forensic_tools"
    / "cellebrite_inseyets"
    / "public_extracts"
    / "cellebrite_classifications_extract.csv",
    "griffeye": REPO_ROOT
    / "forensic_tools"
    / "griffeye"
    / "public_extracts"
    / "griffeye_bookmarks_extract.csv",
}

LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/run/media/", re.IGNORECASE),
    re.compile(r"\\blind_tool_input\\", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("_x000d_", "\n")
    text = re.sub(r"[\r\n]+", " | ", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return text.strip(" | \t")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing normalized prediction source: {path}. Regenerate it locally "
            "with evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
    required = set(COMMON_FIELDS) | {"tool_raw_label"}
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise ValueError(f"Source is missing required columns: {missing}")
    return rows


def common_row(row: dict[str, str]) -> dict[str, str]:
    return {field: clean_text(row.get(field, "")) for field in COMMON_FIELDS}


def ensure_source_profile(rows: list[dict[str, str]]) -> None:
    counts = Counter(clean_text(row.get("tool_name", "")) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"Frozen source profile mismatch: expected {EXPECTED_COUNTS}, found {dict(counts)}"
        )
    keys = [(row.get("tool_name", ""), row.get("bundle_id", "")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Source contains duplicate tool_name/bundle_id pairs")
    if any(not bundle_id for _, bundle_id in keys):
        raise ValueError("Source contains empty bundle identifiers")


def contains_local_path(value: str) -> bool:
    return any(pattern.search(value) for pattern in LOCAL_PATH_PATTERNS)


def validate_minimized_rows(rows: Iterable[dict[str, str]], label: str) -> None:
    for index, row in enumerate(rows, start=1):
        for field, value in row.items():
            if contains_local_path(str(value)):
                raise ValueError(
                    f"Local path detected in {label}, row {index}, field {field}: {value!r}"
                )


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Use --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def build_extracts(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["tool_name"]].append(row)

    magnet = []
    for row in grouped["magnet_axiom"]:
        magnet.append(
            {
                **common_row(row),
                "tags": clean_text(row.get("tool_raw_label", "")),
            }
        )

    excire = []
    for tool_name in (
        "excire_foto_2025_d20",
        "excire_foto_2025_d50",
        "excire_foto_2025_d80",
    ):
        for row in grouped[tool_name]:
            excire.append(
                {
                    **common_row(row),
                    "excire_distance_limit": clean_text(row.get("excire_distance_limit", "")),
                    "n_prompt_hits": clean_text(row.get("n_prompt_hits", "")),
                    "hit_prompts": clean_text(row.get("hit_prompts", "")),
                    **{
                        field: clean_text(row.get(field, ""))
                        for field in EXCIRE_PROMPT_FIELDS
                    },
                }
            )

    cellebrite = []
    for row in grouped["cellebrite_inseyets"]:
        cellebrite.append(
            {
                **common_row(row),
                "classifications": clean_text(row.get("tool_raw_label", "")),
            }
        )

    griffeye = []
    for row in grouped["griffeye"]:
        raw_bookmarks = clean_text(row.get("tool_raw_label", ""))
        secondary = [bookmark for bookmark in GRIFFEYE_SECONDARY if bookmark in raw_bookmarks]
        griffeye.append(
            {
                **common_row(row),
                "firearm_bookmark": "true" if GRIFFEYE_PRIMARY in raw_bookmarks else "false",
                "secondary_weapon_bookmarks": " | ".join(secondary),
            }
        )

    extracts = {
        "magnet": sorted(magnet, key=lambda item: item["bundle_id"]),
        "excire": sorted(excire, key=lambda item: (item["tool_name"], item["bundle_id"])),
        "cellebrite": sorted(cellebrite, key=lambda item: item["bundle_id"]),
        "griffeye": sorted(griffeye, key=lambda item: item["bundle_id"]),
    }
    for label, extract_rows in extracts.items():
        validate_minimized_rows(extract_rows, label)
    return extracts


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    summary_path = resolve_path(args.summary)
    rows = read_rows(source)
    ensure_source_profile(rows)
    extracts = build_extracts(rows)

    fieldnames = {
        "magnet": [*COMMON_FIELDS, "tags"],
        "excire": [
            *COMMON_FIELDS,
            "excire_distance_limit",
            "n_prompt_hits",
            "hit_prompts",
            *EXCIRE_PROMPT_FIELDS,
        ],
        "cellebrite": [*COMMON_FIELDS, "classifications"],
        "griffeye": [
            *COMMON_FIELDS,
            "firearm_bookmark",
            "secondary_weapon_bookmarks",
        ],
    }

    for label, output_path in OUTPUTS.items():
        write_csv(output_path, extracts[label], fieldnames[label], args.force)

    summary = {
        "schema_version": "1.0",
        "source": repo_relative(source),
        "source_sha256": sha256_file(source),
        "source_rows": len(rows),
        "outputs": {
            label: {
                "path": repo_relative(path),
                "rows": len(extracts[label]),
                "sha256": sha256_file(path),
            }
            for label, path in OUTPUTS.items()
        },
        "decision_profile": EXPECTED_COUNTS,
        "local_paths_detected": False,
        "validation_required": "forensic_tools/scripts/validate_public_extract_equivalence.py",
    }
    if summary_path.exists() and not args.force:
        raise FileExistsError(
            f"Summary already exists: {summary_path}. Use --force to replace it."
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Generated minimized public extracts:")
    for label, path in OUTPUTS.items():
        print(f" - {label}: {repo_relative(path)} ({len(extracts[label])} rows)")
    print(f" - summary: {repo_relative(summary_path)}")


if __name__ == "__main__":
    main()
