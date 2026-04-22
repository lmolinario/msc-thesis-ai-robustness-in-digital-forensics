#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 09_generate_review_manifest_full.py
===============================================================================

Title:
    Full Review Manifest Generator for the FAIR-Lab Annotation Pipeline

Purpose:
    This script generates the official full review manifest from
    datasets/prepared/final_pool/metadata.csv.

    The resulting manifest is intended to support the downstream workflow,
    including:
        - full-pool automatic prelabeling
        - batch manual review
        - source-aware inspection
        - OOD and exclusion handling
        - review-state tracking

Methodological note:
    This script is the official bridge between:
        1. technical dataset preparation
        2. semantic annotation / manual review

    The prepared metadata is assumed to be technical-only.
    Review-related fields are introduced here for the first time.

Inputs:
    - datasets/prepared/final_pool/metadata.csv

Outputs:
    - datasets/prepared/manifests/review_manifest_full.csv

Dependencies:
    - Python standard library
    - project utility: utils.paths
===============================================================================
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from datasets.scripts.utils.paths import PREPARED_DATASETS_DIR


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

INPUT_METADATA = PREPARED_DATASETS_DIR / "final_pool" / "metadata.csv"
OUTPUT_MANIFEST = PREPARED_DATASETS_DIR / "manifests" / "review_manifest_full.csv"


# -----------------------------------------------------------------------------
# Output Schema
# -----------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    # --- identity / provenance ---
    "image_id",
    "sha256",
    "prepared_filename",
    "relative_path",
    "source_dataset",
    "source_group",
    "source_relative_path",
    "source_filename",

    # --- technical metadata ---
    "width",
    "height",
    "size_bytes",
    "extension",
    "is_valid_image",

    # --- automatic prelabel ---
    "auto_label",
    "auto_confidence",
    "weapon_score",
    "non_weapon_score",
    "score_margin",
    "prelabel_model",
    "prelabel_timestamp",
    "prelabel_status",
    "prelabel_error",

    # --- manual review / final selection ---
    "final_label",
    "review_state",
    "review_notes",
    "reviewer_id",
    "review_timestamp",
    "label_confidence",
    "exclusion_reason",
    "ood_flag",
    "ood_notes",
]


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_input_file(path: Path) -> None:
    """
    Ensure that the input metadata file exists.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input metadata not found: {path}")


def validate_required_columns(fieldnames: list[str] | None) -> None:
    """
    Validate that the input metadata contains the required minimum columns.
    """
    required = {
        "image_id",
        "sha256",
        "prepared_filename",
        "prepared_relative_path",
        "source_dataset",
        "source_group",
        "source_relative_path",
        "source_filename",
        "width",
        "height",
        "size_bytes",
        "extension",
        "is_valid_image",
    }

    available = set(fieldnames or [])
    missing = required - available

    if missing:
        raise ValueError(
            f"Missing required columns in metadata.csv: {sorted(missing)}"
        )


# -----------------------------------------------------------------------------
# Row Construction
# -----------------------------------------------------------------------------

def build_row(metadata_row: dict[str, str]) -> dict[str, str]:
    """
    Build one review manifest row from one prepared metadata row.

    The SHA256 field is propagated directly from metadata.csv and is not
    recomputed here, since it was already established during prepared dataset
    construction.
    """
    return {
        # --- identity / provenance ---
        "image_id": metadata_row.get("image_id", ""),
        "sha256": metadata_row.get("sha256", ""),
        "prepared_filename": metadata_row.get("prepared_filename", ""),
        "relative_path": metadata_row.get("prepared_relative_path", ""),
        "source_dataset": metadata_row.get("source_dataset", ""),
        "source_group": metadata_row.get("source_group", ""),
        "source_relative_path": metadata_row.get("source_relative_path", ""),
        "source_filename": metadata_row.get("source_filename", ""),

        # --- technical metadata ---
        "width": metadata_row.get("width", ""),
        "height": metadata_row.get("height", ""),
        "size_bytes": metadata_row.get("size_bytes", ""),
        "extension": metadata_row.get("extension", ""),
        "is_valid_image": metadata_row.get("is_valid_image", ""),

        # --- automatic prelabel ---
        "auto_label": "",
        "auto_confidence": "",
        "weapon_score": "",
        "non_weapon_score": "",
        "score_margin": "",
        "prelabel_model": "",
        "prelabel_timestamp": "",
        "prelabel_status": "pending",
        "prelabel_error": "",

        # --- manual review / final selection ---
        "final_label": "",
        "review_state": "pending",
        "review_notes": "",
        "reviewer_id": "",
        "review_timestamp": "",
        "label_confidence": "",
        "exclusion_reason": "",
        "ood_flag": "",
        "ood_notes": "",
    }


# -----------------------------------------------------------------------------
# I/O Helpers
# -----------------------------------------------------------------------------

def load_metadata_rows(path: Path) -> list[dict[str, str]]:
    """
    Load and validate input metadata rows.
    """
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_required_columns(reader.fieldnames)

        for metadata_row in reader:
            rows.append(metadata_row)

    if not rows:
        raise ValueError("No rows found in metadata.csv. Manifest not created.")

    return rows


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    """
    Write the full review manifest to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def print_summary(rows: list[dict[str, str]]) -> None:
    """
    Print a compact integrity and status summary for the generated manifest.
    """
    total_rows = len(rows)

    image_ids = [row["image_id"] for row in rows]
    duplicate_image_ids = total_rows - len(set(image_ids))

    review_state_pending = sum(
        1 for row in rows
        if row.get("review_state", "").strip() == "pending"
    )

    prelabel_status_pending = sum(
        1 for row in rows
        if row.get("prelabel_status", "").strip() == "pending"
    )

    final_label_non_empty = sum(
        1 for row in rows
        if row.get("final_label", "").strip() != ""
    )

    source_counter = Counter(row.get("source_dataset", "") for row in rows)

    print(f"Manifest creato: {OUTPUT_MANIFEST}")
    print("\n=== REVIEW MANIFEST SUMMARY ===")
    print(f"Totale righe: {total_rows}")
    print(f"Duplicati image_id: {duplicate_image_ids}")
    print(f"Review state pending: {review_state_pending}")
    print(f"Prelabel status pending: {prelabel_status_pending}")
    print(f"Final label non vuote: {final_label_non_empty}")

    if source_counter:
        print("\nDistribuzione per source_dataset:")
        for source_dataset in sorted(source_counter):
            print(f"{source_dataset}: {source_counter[source_dataset]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point.

    Execution flow:
        1. validate input metadata
        2. load rows
        3. build review manifest rows
        4. write output CSV
        5. print summary
    """
    validate_input_file(INPUT_METADATA)

    metadata_rows = load_metadata_rows(INPUT_METADATA)
    manifest_rows = [build_row(row) for row in metadata_rows]

    write_manifest(manifest_rows, OUTPUT_MANIFEST)
    print_summary(manifest_rows)


if __name__ == "__main__":
    main()