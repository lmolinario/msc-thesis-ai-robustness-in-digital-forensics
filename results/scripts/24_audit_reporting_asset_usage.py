#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Chapter 5 reporting assets against both LaTeX thesis trees.

The script is read-only by default. It reports which manifest asset identifiers
are referenced by the English and Italian thesis sources, whether thesis-ready
copies exist, and whether those copies are byte-identical to the reporting-layer
files under ``results/figures/chapter_5``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT / "results" / "figures" / "chapter_5" / "chapter5_figures_manifest.csv"
)
THESIS_TREES = {
    "english": REPO_ROOT / "docs" / "LatexThesis",
    "italian": REPO_ROOT / "docs" / "LatexThesis_ITA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional repository-relative or absolute JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing manifest outputs or non-identical existing thesis copies.",
    )
    return parser.parse_args()


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
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


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing figure manifest: {MANIFEST}")
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"figure_id", "output_path", "format", "source_csv"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        return [dict(row) for row in reader]


def load_tex_sources(root: Path) -> tuple[str, list[str]]:
    files = sorted(root.rglob("*.tex"))
    text_parts: list[str] = []
    for path in files:
        text_parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(text_parts), [repo_relative(path) for path in files]


def main() -> int:
    args = parse_args()
    manifest_rows = read_manifest()
    if len(manifest_rows) != 41:
        raise ValueError(f"Expected 41 manifest rows, found {len(manifest_rows)}")

    tex_content: dict[str, str] = {}
    tex_files: dict[str, list[str]] = {}
    for language, root in THESIS_TREES.items():
        content, files = load_tex_sources(root)
        tex_content[language] = content
        tex_files[language] = files

    by_id: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_id.setdefault(row["figure_id"], []).append(row)
    if len(by_id) != 24:
        raise ValueError(f"Expected 24 unique manifest IDs, found {len(by_id)}")

    asset_records: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    mismatched_copies: list[dict[str, str]] = []
    exact_copy_count = 0

    for figure_id, rows in sorted(by_id.items()):
        record: dict[str, Any] = {
            "figure_id": figure_id,
            "referenced": {
                language: figure_id in content
                for language, content in tex_content.items()
            },
            "files": [],
        }
        for row in rows:
            output = (REPO_ROOT / row["output_path"]).resolve()
            file_record: dict[str, Any] = {
                "format": row["format"],
                "reporting_path": repo_relative(output),
                "reporting_exists": output.is_file(),
                "source_csv": row["source_csv"],
                "thesis_copies": {},
            }
            if not output.is_file():
                missing_outputs.append(repo_relative(output))
                record["files"].append(file_record)
                continue

            reporting_hash = sha256_file(output)
            file_record["reporting_sha256"] = reporting_hash
            for language, root in THESIS_TREES.items():
                candidate = root / "images" / output.name
                copy_info: dict[str, Any] = {
                    "path": repo_relative(candidate),
                    "exists": candidate.is_file(),
                }
                if candidate.is_file():
                    candidate_hash = sha256_file(candidate)
                    identical = candidate_hash == reporting_hash
                    copy_info["sha256"] = candidate_hash
                    copy_info["byte_identical"] = identical
                    if identical:
                        exact_copy_count += 1
                    else:
                        mismatched_copies.append(
                            {
                                "figure_id": figure_id,
                                "language": language,
                                "reporting_path": repo_relative(output),
                                "thesis_path": repo_relative(candidate),
                            }
                        )
                file_record["thesis_copies"][language] = copy_info
            record["files"].append(file_record)
        asset_records.append(record)

    unreferenced_both = [
        record["figure_id"]
        for record in asset_records
        if not any(record["referenced"].values())
    ]
    referenced_english = sum(
        1 for record in asset_records if record["referenced"]["english"]
    )
    referenced_italian = sum(
        1 for record in asset_records if record["referenced"]["italian"]
    )

    report = {
        "schema_version": "1.0",
        "manifest": repo_relative(MANIFEST),
        "manifest_rows": len(manifest_rows),
        "unique_asset_ids": len(by_id),
        "tex_files_scanned": tex_files,
        "referenced_asset_ids": {
            "english": referenced_english,
            "italian": referenced_italian,
        },
        "unreferenced_in_both_theses": unreferenced_both,
        "missing_reporting_outputs": missing_outputs,
        "mismatched_existing_thesis_copies": mismatched_copies,
        "byte_identical_thesis_copy_relations": exact_copy_count,
        "assets": asset_records,
    }

    report_path = resolve_path(args.report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print("Chapter 5 reporting-asset audit completed.")
    print(f" - manifest rows: {len(manifest_rows)}")
    print(f" - unique asset IDs: {len(by_id)}")
    print(f" - referenced in English thesis: {referenced_english}")
    print(f" - referenced in Italian thesis: {referenced_italian}")
    print(f" - unreferenced in both theses: {len(unreferenced_both)}")
    print(f" - byte-identical thesis copy relations: {exact_copy_count}")
    print(f" - missing reporting outputs: {len(missing_outputs)}")
    print(f" - mismatched existing thesis copies: {len(mismatched_copies)}")
    if unreferenced_both:
        print(" - unreferenced IDs:")
        for figure_id in unreferenced_both:
            print(f"   - {figure_id}")
    if report_path is not None:
        print(f" - report: {repo_relative(report_path)}")

    if args.strict and (missing_outputs or mismatched_copies):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
