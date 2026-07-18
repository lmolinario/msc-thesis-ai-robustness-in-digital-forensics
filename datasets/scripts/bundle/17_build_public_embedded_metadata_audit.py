#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and optionally install a minimized public embedded-metadata audit.

Step 16 may record complete EXIF/XMP values and binary metadata payloads. Those
values are useful for local diagnosis but are not required by the public
research artifact. This script preserves only bundle identifiers, metadata
presence, metadata key names and sensitive-term hits.

Without ``--install`` the outputs are written under ``metadata/.staging/``.
With ``--install`` the complete source is copied to an ignored local private
file and the minimized outputs become canonical.
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
CANONICAL_AUDIT = METADATA_DIR / "embedded_metadata_audit.csv"
PRIVATE_AUDIT = METADATA_DIR / "embedded_metadata_audit.private.csv"
SENSITIVE_INDEX = METADATA_DIR / "embedded_metadata_sensitive_hits.csv"
PUBLIC_SUMMARY = METADATA_DIR / "embedded_metadata_public_summary.json"
DEFAULT_STAGING = METADATA_DIR / ".staging"

EXPECTED_TOTAL = 11_500
EXPECTED_SENSITIVE = 15
BUNDLE_RE = re.compile(r"bundle_\d{6}", re.IGNORECASE)
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


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(CANONICAL_AUDIT))
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def bool_text(value: Any) -> str:
    return (
        "true"
        if str(value).strip().lower() in {"1", "true", "yes", "y", "t"}
        else "false"
    )


def split_hits(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def extract_bundle_id(row: dict[str, str], row_number: int) -> str:
    for field in ("bundle_id", "relative_path", "filename", "path"):
        match = BUNDLE_RE.search(str(row.get(field, "")))
        if match:
            return match.group(0).lower()
    raise ValueError(f"Cannot determine bundle_id for source row {row_number}")


def load_source(path: Path) -> tuple[list[dict[str, str]], list[str]]:
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


def minimize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        hits = split_hits(row.get("sensitive_hits", ""))
        output.append(
            {
                "bundle_id": extract_bundle_id(row, index),
                "suffix": str(row.get("suffix", "")).strip().lower(),
                "has_embedded_metadata": bool_text(row.get("has_embedded_metadata")),
                "sensitive_hits": ";".join(hits),
                "sensitive_hit_count": str(len(hits)),
                "metadata_keys": str(row.get("metadata_keys", "")).strip(),
            }
        )
    return sorted(output, key=lambda row: row["bundle_id"])


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} rows, found {len(rows)}")
    bundle_ids = [row["bundle_id"] for row in rows]
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("Duplicate bundle identifiers in minimized audit")
    if any(BUNDLE_RE.fullmatch(value) is None for value in bundle_ids):
        raise ValueError("Invalid bundle identifier in minimized audit")
    sensitive = sum(int(row["sensitive_hit_count"]) > 0 for row in rows)
    if sensitive != EXPECTED_SENSITIVE:
        raise ValueError(
            f"Expected {EXPECTED_SENSITIVE} sensitive-hit rows, found {sensitive}"
        )
    for row_number, row in enumerate(rows, start=1):
        if list(row) != PUBLIC_FIELDS:
            raise ValueError(f"Unexpected schema at row {row_number}")
        for field, value in row.items():
            text = str(value)
            if LOCAL_PATH_RE.search(text):
                raise ValueError(
                    f"Local path leakage in row {row_number}, field {field}: {text!r}"
                )
            if "\x00" in text:
                raise ValueError(f"NUL byte in row {row_number}, field {field}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_summary(
    source_path: str,
    source_sha256: str,
    source_fields: list[str],
    audit_path: Path,
    sensitive_path: Path,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "script": "datasets/scripts/bundle/17_build_public_embedded_metadata_audit.py",
        "policy": "minimized_public_audit",
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "source_columns": source_fields,
        },
        "outputs": {
            "public_audit": repo_path(audit_path),
            "public_audit_sha256": sha256_file(audit_path),
            "sensitive_hits_index": repo_path(sensitive_path),
            "sensitive_hits_index_sha256": sha256_file(sensitive_path),
        },
        "counts": {
            "audit_rows": len(rows),
            "files_with_embedded_metadata": sum(
                row["has_embedded_metadata"] == "true" for row in rows
            ),
            "files_with_sensitive_hits": sum(
                int(row["sensitive_hit_count"]) > 0 for row in rows
            ),
        },
        "public_columns": PUBLIC_FIELDS,
        "removed_private_columns": [
            field
            for field in source_fields
            if field not in {"relative_path", *PUBLIC_FIELDS}
        ],
        "note": (
            "The public audit records metadata presence, key names and "
            "sensitive-term hits without publishing complete EXIF/XMP values or "
            "binary payloads."
        ),
    }


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    staging = resolve_path(args.staging_dir)
    source_rows, source_fields = load_source(source)
    source_path = repo_path(source)
    source_sha256 = sha256_file(source)
    public_rows = minimize(source_rows)
    validate(public_rows)
    sensitive_rows = [
        row for row in public_rows if int(row["sensitive_hit_count"]) > 0
    ]

    staged_audit = staging / "embedded_metadata_audit.csv"
    staged_sensitive = staging / "embedded_metadata_sensitive_hits.csv"
    staged_summary = staging / "embedded_metadata_public_summary.json"
    write_csv(staged_audit, public_rows)
    write_csv(staged_sensitive, sensitive_rows)
    write_json(
        staged_summary,
        build_summary(
            source_path,
            source_sha256,
            source_fields,
            staged_audit,
            staged_sensitive,
            public_rows,
        ),
    )

    if args.install:
        if source.resolve() != CANONICAL_AUDIT.resolve():
            raise ValueError("--install requires the canonical audit as --source")
        if PRIVATE_AUDIT.exists() and not args.force:
            raise FileExistsError(
                f"Private backup already exists: {PRIVATE_AUDIT}. "
                "Use --force only after reviewing it."
            )
        if PRIVATE_AUDIT.exists():
            PRIVATE_AUDIT.unlink()
        shutil.copy2(source, PRIVATE_AUDIT)
        os.replace(staged_audit, CANONICAL_AUDIT)
        os.replace(staged_sensitive, SENSITIVE_INDEX)
        write_json(
            PUBLIC_SUMMARY,
            build_summary(
                source_path,
                source_sha256,
                source_fields,
                CANONICAL_AUDIT,
                SENSITIVE_INDEX,
                public_rows,
            ),
        )
        if staged_summary.exists():
            staged_summary.unlink()
        print("Installed minimized public embedded-metadata audit:")
        print(f" - {repo_path(CANONICAL_AUDIT)}")
        print(f" - {repo_path(SENSITIVE_INDEX)}")
        print(f" - {repo_path(PUBLIC_SUMMARY)}")
        print(f"Private local backup: {repo_path(PRIVATE_AUDIT)}")
    else:
        print("Generated staged minimized embedded-metadata audit:")
        print(f" - {repo_path(staged_audit)}")
        print(f" - {repo_path(staged_sensitive)}")
        print(f" - {repo_path(staged_summary)}")

    print(f"Rows: {len(public_rows)}")
    print(f"Sensitive-hit rows: {len(sensitive_rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
