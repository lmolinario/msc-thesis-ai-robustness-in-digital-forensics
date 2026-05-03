#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 01_download_kaggle.py
===============================================================================

Title:
    Automated Download and Indexing of Kaggle Dataset for Forensic AI Pipeline

Purpose:
    This scripts is responsible for downloading a dataset from Kaggle, extracting
    its contents, and generating a structured metadata summary (CSV file).
    It represents the initial stage of the FAIR-Lab pipeline, ensuring
    reproducibility and traceability of raw data sources.

Context (Thesis Contribution):
    Within the experimental framework, this module supports dataset acquisition
    under controlled and reproducible conditions. The generated summary enables:
        - dataset auditing
        - integrity verification
        - traceability of data sources
        - preparation for downstream annotation and evaluation

Inputs:
    - Kaggle dataset identifier (e.g., "snehilsanyal/weapon-detection-test")

Outputs:
    - Extracted dataset directory
    - Structured CSV metadata file containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image (binary flag)
        * sha256

Assumptions:
    - Kaggle API is configured OR manual download is provided
    - Output directory is writable
    - Dataset is provided as a ZIP archive

Dependencies:
    - kaggle API (optional)
    - Python standard libraries (argparse, csv, zipfile, pathlib)

===============================================================================
"""

import argparse
import csv
import logging
import zipfile
from pathlib import Path
import hashlib

# Attempt to import Kaggle API
try:
    import kaggle  # type: ignore
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path

# Default configuration (ensures reproducibility)
DEFAULT_DATASET = "snehilsanyal/weapon-detection-test"
DEFAULT_OUTPUT_DIR = RAW_DATASETS_DIR / "01_kaggle_weapon"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "01_kaggle_weapon" / "download_summary.csv"

# Supported image formats for filtering
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments to configure dataset acquisition.

    This allows flexible integration into automated pipelines while
    maintaining default reproducible settings.
    """
    parser = argparse.ArgumentParser(
        description="Download and extract Kaggle dataset with metadata summary generation."
    )

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-csv", type=str, default=str(DEFAULT_SUMMARY_CSV))

    parser.add_argument("--keep-zip", action="store_true",
                        help="Preserve ZIP archive after extraction.")
    parser.add_argument("--force-extract", action="store_true",
                        help="Force extraction even if files already exist.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed logging output.")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    """
    Configures logging level.

    Verbose mode is useful for debugging and audit trails.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# File Utilities
# -----------------------------------------------------------------------------
def is_image_file(path: Path) -> bool:
    """
    Determines whether a file is a valid image based on extension.

    This classification is later used to separate relevant data
    (images) from auxiliary files (labels, metadata, etc.).
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def has_images_recursive(directory: Path) -> bool:
    """
    Checks if a directory already contains image files.

    Used to avoid redundant downloads and ensure idempotent execution.
    """
    return any(is_image_file(path) for path in directory.rglob("*"))


def find_zip_files(directory: Path) -> list[Path]:
    """
    Identifies ZIP archives within the target directory.

    Enables reuse of previously downloaded datasets.
    """
    return sorted([p for p in directory.glob("*.zip") if p.is_file()])


# -----------------------------------------------------------------------------
# Extraction Logic
# -----------------------------------------------------------------------------
def extract_zip(zip_path: Path, output_dir: Path, keep_zip: bool = False) -> None:
    """
    Extracts dataset archive and optionally removes the ZIP file.

    This step ensures that raw data becomes accessible for processing
    while maintaining optional storage efficiency.
    """
    logging.info("Extracting archive: %s", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        logging.info("Removed ZIP archive: %s", zip_path)


# -----------------------------------------------------------------------------
# Download Logic
# -----------------------------------------------------------------------------
def download_kaggle_dataset(dataset: str, output_dir: Path,
                            keep_zip: bool = False,
                            force_extract: bool = False) -> None:
    """
    Handles dataset acquisition from Kaggle.

    Includes fallback mechanisms to support manual download in case
    of API unavailability or authentication issues.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_zips = find_zip_files(output_dir)
    images_present = has_images_recursive(output_dir)

    # Avoid redundant downloads (important for reproducibility)
    if images_present and not force_extract:
        logging.info("Dataset already present. Skipping download.")
        return

    # Reuse existing ZIP if available
    if existing_zips:
        extract_zip(existing_zips[0], output_dir, keep_zip)
        return

    # Fallback if Kaggle API is unavailable
    if not KAGGLE_AVAILABLE:
        logging.error("Kaggle API not available.")
        return

    try:
        logging.info("Downloading dataset from Kaggle: %s", dataset)

        kaggle.api.dataset_download_files(dataset, path=str(output_dir), unzip=False)

        downloaded_zips = find_zip_files(output_dir)
        if not downloaded_zips:
            raise FileNotFoundError("No ZIP file found after download.")

        zip_path = max(downloaded_zips, key=lambda p: p.stat().st_mtime)
        extract_zip(zip_path, output_dir, keep_zip)

    except Exception as exc:
        logging.error("Kaggle download failed: %s", exc)

def compute_sha256(file_path: Path) -> str:
    """
    Compute the SHA256 digest of a file.

    This supports file-level integrity verification and source traceability.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# -----------------------------------------------------------------------------
# Metadata Generation
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Generates structured metadata for all files in the dataset.

    This is critical for:
        - dataset auditing
        - reproducibility
        - traceability
        - debugging dataset inconsistencies
        - file-level integrity verification
    """

    rows = []

    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue

        rows.append({
            "relative_path": str(file_path.relative_to(output_dir)).replace("\\", "/"),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "is_image": int(is_image_file(file_path)),
            "sha256": compute_sha256(file_path),
        })

    return rows


def write_summary_csv(rows: list[dict[str, str | int]], summary_csv: Path) -> None:
    """
    Writes dataset metadata into a CSV file.

    This file acts as a lightweight dataset index and supports
    downstream validation and analysis tasks.
    """

    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "filename",
                "extension",
                "size_bytes",
                "is_image",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Entry point of the scripts.

    Executes the full pipeline:
        1. Download dataset
        2. Extract files
        3. Generate metadata summary
    """

    args = parse_args()
    setup_logging(args.verbose)

    output_dir = repo_relative_path(args.output_dir)
    summary_csv = repo_relative_path(args.summary_csv)

    download_kaggle_dataset(
        dataset=args.dataset,
        output_dir=output_dir,
        keep_zip=args.keep_zip,
        force_extract=args.force_extract,
    )

    summary_rows = build_summary_rows(output_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Metadata summary generated: %s", summary_csv)
    logging.info("Total files indexed: %d", len(summary_rows))
    logging.info(
        "Total images detected: %d",
        sum(1 for row in summary_rows if row["is_image"] == 1),
    )


if __name__ == "__main__":
    main()