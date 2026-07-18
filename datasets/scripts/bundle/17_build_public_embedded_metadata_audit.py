#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a minimized public embedded-metadata audit.

The full audit produced by step 16 may contain complete EXIF/XMP values and
binary metadata payloads. Those values are useful for local diagnosis but are
not required for the public research artifact. This script derives a public
record that preserves only the fields needed to audit metadata prevalence and
sensitive-term hits.

Default behavior writes to a local staging directory. Use ``--install`` only
after reviewing the generated files; the original full audit is then preserved
locally as an ignored private file and the minimized audit becomes canonical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
METADATA_DIR = REPO_ROOT / "datasets" / "forensic_evaluation_bundle" / "metadata"
DEFAULT_SOURCE = METADATA_DIR / "embedded_metadata_audit.csv"
DEFAULT_STAGING_DIR = METADATA_DIR / ".staging"
CANONICAL_PUBLIC_AUDIT = METADATA_DIR / "embedded_metadata_audit.csv"
PRIVATE_AUDIT = METADATA_DIR / "embedded_metadata_audit.private.csv"
SENSITIVE_INDEX = METADATA_DIR / "embedded_metadata_sensitive_hits.csv"
PUBLIC_SUMMARY = METADATA_DIR / "embedded_metadata_public_summary.json"

EXPECTED_TOTAL = 11_500
EXPECTED_SENSITIVE = 15
BUNDLE_RE = re.compile(r"(bundle_\d{6})", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/run/media/|/home/|/Users/|\\blind_tool_input\\)",
    re.IGNORECASE,
)
PUBLIC_FIELDS = [
    "bundle_id",
    "suffix",
    "has_embedded_metadata",
    "sensitive_hits",
    "sensitive_hit_count",
    "metadata_keys",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING_DIR))
    parser.add_argument(
        "--install",
        action="store_true",
        help=(
            "Preserve the current full audit as an ignored private file and "
            "install the minimized files in the canonical metadata directory."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def bool_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    return "true" if text in {"1", "true", "yes", "y", "t"} else "false"


def split_hits(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    return [item.strip() for item in text.split(";") if item.strip()]


def bundle_id_from_row(row: dict[str, str], row_number: int) -> str:
    for field in ("bundle_id", "relative_path", "filename", "path"):
        match = BUNDLE_RE.search(str(row.get(field, "")))
        if match:
            return match.group(1).lower()
    raise ValueError(f"Cannot determine bundle_id for source row {row_number}")


def read_source(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedded metadata audit not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    required = {"suffix", "has_embedded_metadata", "sensitive_hits", "metadata_keys"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Source audit is missing required columns: {missing}")
    return rows, fields


def minimize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    public_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        hits = split_hits(row.get("sensitive_hits", ""))
        public_rows.append(
            {
                "bundle_id": bundle_id_from_row(row, row_number),
                "suffix": str(row.get("suffix", "")).strip().lower(),
                "has_embedded_metadata": bool_text(row.get("has_embedded_metadata")),
                "sensitive_hits": ";".join(hits),
                "sensitive_hit_count": str(len(hits)),
                "metadata_keys": str(row.get("metadata_keys", "")).strip(),
            }
        )
    return sorted(public_rows, key=lambda row: row["bundle_id"])


def validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} audit rows, found {len(rows)}")
    bundle_ids = [row["bundle_id"] for row in rows]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("Public audit contains duplicate bundle identifiers")
    if any(not BUNDLE_RE.fullmatch(bundle_id) for bundle_id in bundle_ids):
        raise ValueError("Public audit contains an invalid bundle identifier")
    sensitive_count = sum(int(row["sensitive_hit_count"]) > 0 for row in rows)
    if sensitive_count != EXPECTED_SENSITIVE:
        raise ValueError(
            f"Expected {EXPECTED_SENSITIVE} sensitive-hit rows, found {sensitive_count}"
        )
    for row_number, row in enumerate(rows, start=1):
        if set(row) != set(PUBLIC_FIELDS):
            raise ValueError(f"Unexpected public schema at row {row_number}")
        for field, value in row.items():
            if LOCAL_PATH_RE.search(str(value)):
                raise ValueError(
                    f"Local path leakage in row {row_number}, field {field}: {value!r}"
                )
            if "\x00" in str(value):
                raise ValueError(f"NUL byte in row {row_number}, field {field}")


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def output_payload(
    source: Path,
    public_audit: Path,
    sensitive_index: Path,
    rows: list[dict[str, str]],
    source_fields: list[str],
) -> dict[str, Any]:
    sensitive_rows = [row for row in rows if int(row["sensitive_hit_count"]) > 0]
    return {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "script": "datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py",
        "policy": "minimized_public_audit",
        "source": {
            "path": source.relative_to(REPO_ROOT).as_posix()
            if source.is_relative_to(REPO_ROOT)
            else str(source),
            "sha256": sha256_file(source),
            "source_columns": source_fields,
        },
        "outputs": {
            "public_audit": public_audit.relative_to(REPO_ROOT).as_posix(),
            "public_audit_sha256": sha256_file(public_audit),
            "sensitive_hits_index": sensitive_index.relative_to(REPO_ROOT).as_posix(),
            "sensitive_hits_index_sha256": sha256_file(sensitive_index),
        },
        "counts": {
            "audit_rows": len(rows),
            "files_with_embedded_metadata": sum(
                row["has_embedded_metadata"] == "true" for row in rows
            ),
            "files_with_sensitive_hits": len(sensitive_rows),
        },
        "public_columns": PUBLIC_FIELDS,
        "removed_private_columns": [
            field
            for field in source_fields
            if field not in {"relative_path", *PUBLIC_FIELDS}
        ],
        "note": (
            "The public audit records metadata presence, key names and sensitive-term "
            "hits without publishing complete EXIF/XMP values or binary payloads."
        ),
    }


def stage_outputs(
    source: Path,
    staging_dir: Path,
    rows: list[dict[str, str]],
    source_fields: list[str],
) -> tuple[Path, Path, Path]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    public_audit = staging_dir / "embedded_metadata_audit.csv"
    sensitive_index = staging_dir / "embedded_metadata_sensitive_hits.csv"
    summary = staging_dir / "embedded_metadata_public_summary.json"
    sensitive_rows = [row for row in rows if int(row["sensitive_hit_count"]) > 0]
    write_csv_atomic(public_audit, rows)
    write_csv_atomic(sensitive_index, sensitive_rows)
    write_json_atomic(
        summary,
        output_payload(source, public_audit, sensitive_index, rows, source_fields),
    )
    return public_audit, sensitive_index, summary


def install_outputs(
    source: Path,
    staged_audit: Path,
    staged_sensitive: Path,
    staged_summary: Path,
    force: bool,
) -> None:
    if source.resolve() != CANONICAL_PUBLIC_AUDIT.resolve():
        raise ValueError("--install requires the canonical audit as --source")
    if PRIVATE_AUDIT.exists() and not force:
        raise FileExistsError(
            f"Private backup already exists: {PRIVATE_AUDIT}. Use --force only after review."
        )
    if PRIVATE_AUDIT.exists():
        PRIVATE_AUDIT.unlink()
    shutil.copy2(source, PRIVATE_AUDIT)
    os.replace(staged_audit, CANONICAL_PUBLIC_AUDIT)
    os.replace(staged_sensitive, SENSITIVE_INDEX)
    os.replace(staged_summary, PUBLIC_SUMMARY)


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    staging_dir = resolve_path(args.staging_dir)
    source_rows, source_fields = read_source(source)
    public_rows = minimize_rows(source_rows)
    validate_rows(public_rows)
    staged_audit, staged_sensitive, staged_summary = stage_outputs(
        source, staging_dir, public_rows, source_fields
    )
    if args.install:
        install_outputs(
            source,
            staged_audit,
            staged_sensitive,
            staged_summary,
            force=args.force,
        )
        print("Installed minimized public embedded-metadata audit:")
        print(f" - {CANONICAL_PUBLIC_AUDIT.relative_to(REPO_ROOT)}")
        print(f" - {SENSITIVE_INDEX.relative_to(REPO_ROOT)}")
        print(f" - {PUBLIC_SUMMARY.relative_to(REPO_ROOT)}")
        print(f"Private local backup: {PRIVATE_AUDIT.relative_to(REPO_ROOT)}")
    else:
        print("Generated staged minimized embedded-metadata audit:")
        print(f" - {staged_audit.relative_to(REPO_ROOT)}")
        print(f" - {staged_sensitive.relative_to(REPO_ROOT)}")
        print(f" - {staged_summary.relative_to(REPO_ROOT)}")
    print(f"Rows: {len(public_rows)}")
    print(
        "Sensitive-hit rows: "
        f"{sum(int(row['sensitive_hit_count']) > 0 for row in public_rows)}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
