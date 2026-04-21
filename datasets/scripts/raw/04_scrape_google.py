#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 04_scrape_google.py
===============================================================================

Title:
    Google Images-Based Dataset Collector and Metadata Indexer

Purpose:
    This scripts collects images from Google Images using a predefined set of
    weapon-related semantic categories, validates the downloaded image files,
    and generates a structured CSV summary of the resulting dataset.

Research Context:
    In the FAIR-Lab experimental pipeline, this scripts supports the acquisition
    of heterogeneous web images intended to increase visual diversity beyond
    curated benchmark datasets. This is particularly relevant when building
    datasets that better approximate real-world variability, including:

        - unconstrained backgrounds
        - heterogeneous object positions
        - varying acquisition quality
        - visually ambiguous or context-dependent weapon appearances

Methodological Relevance:
    In adversarial robustness and forensic AI research, source diversity is a
    key factor affecting model generalization and operational reliability.
    Web-scraped data can enrich the dataset but also introduces specific risks,
    including noise, duplicates, semantic drift, and inconsistent labeling.

    For this reason, the scripts:
        - uses predefined semantic categories
        - enforces image size constraints
        - validates file integrity after download
        - documents collected resources through a summary CSV
        - records SHA256 digests for file-level traceability and integrity checks

Inputs:
    - output directory
    - summary CSV path
    - maximum total number of valid images
    - maximum number of images per class
    - minimum width and height constraints
    - optional force rebuild flag

Outputs:
    - category-structured downloaded image dataset
    - CSV summary file containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - internet access is available
    - GoogleImageCrawler is correctly installed and functional
    - the selected queries are semantically relevant to the research domain
    - category names are used as weak labels and may require downstream review

Dependencies:
    - Python standard library
    - icrawler
    - Pillow
    - project utility: utils.paths

Important Note:
    Images downloaded from public web search engines should not be treated as
    intrinsically reliable labels. They should instead be considered candidate
    samples subject to later inspection, validation, and possible relabeling.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
from pathlib import Path

from icrawler.builtin import GoogleImageCrawler
from PIL import Image, UnidentifiedImageError

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_BASE_DIR = RAW_DATASETS_DIR / "03_google_scraped" / "dataset_scraping_google"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "03_google_scraped" / "google_scraping_summary.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CATEGORIES = {
    "concealed_weapon": "concealed weapon",
    "gun_on_table": "gun on table",
    "person_with_gun": "person with gun",
    "gun_in_bag": "gun in bag",
    "toy_gun": "toy gun",
    "airsoft_weapon": "airsoft gun realistic",
    "holstered_gun": "holstered pistol",
    "damaged_weapon": "damaged firearm",
    "gun_in_drawer": "gun inside drawer",
    "gun_on_floor": "gun on floor",
    "person_pointing_gun": "person pointing gun",
    "gun_next_to_object": "gun next to phone",
    "gun_in_backpack": "gun inside backpack",
    "3d_printed_gun": "3d printed firearm",
    "gun_in_car": "gun in car seat",
}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    This exposes the main acquisition parameters so that the scraping procedure
    can be reproduced and tuned across experiments.
    """
    parser = argparse.ArgumentParser(
        description="Collect images from Google Images and regenerate the summary CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_BASE_DIR})",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=1000,
        help="Maximum desired total number of valid images.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=125,
        help="Maximum number of images to download per category.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=200,
        help="Minimum image width required by the crawler.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=200,
        help="Minimum image height required by the crawler.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the output directory from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    """
    Configure logging.

    Verbose mode is useful for debugging acquisition failures and documenting
    class-wise scraping behavior.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def is_image_file(path: Path) -> bool:
    """
    Return True if the file is a supported image format.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def compute_sha256(file_path: Path) -> str:
    """
    Compute the SHA256 digest of a file.

    This supports file-level integrity verification and provenance tracking.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def list_image_files(directory: Path) -> list[Path]:
    """
    List all valid image files contained directly in the specified directory.
    """
    return sorted([p for p in directory.iterdir() if is_image_file(p)])


def count_total_valid_images(output_base_dir: Path) -> int:
    """
    Count all image files contained in the output directory tree.
    """
    return sum(1 for p in output_base_dir.rglob("*") if is_image_file(p))


def has_existing_dataset(output_base_dir: Path) -> bool:
    """
    Check whether the output directory already contains any files.
    """
    return output_base_dir.exists() and any(output_base_dir.rglob("*"))


# -----------------------------------------------------------------------------
# Output Directory Preparation
# -----------------------------------------------------------------------------
def prepare_output_dir(output_base_dir: Path, force: bool) -> bool:
    """
    Prepare the output directory before scraping.

    Returns:
        True  -> downloading should be performed
        False -> the dataset already exists and can be reused
    """
    if output_base_dir.exists():
        if force:
            logging.warning("Removing existing output directory: %s", output_base_dir)
            shutil.rmtree(output_base_dir)
            output_base_dir.mkdir(parents=True, exist_ok=True)
            return True

        if has_existing_dataset(output_base_dir):
            logging.info("Google dataset already present in %s. Download skipped.", output_base_dir)
            return False

        output_base_dir.mkdir(parents=True, exist_ok=True)
        return True

    output_base_dir.mkdir(parents=True, exist_ok=True)
    return True


# -----------------------------------------------------------------------------
# Download Logic
# -----------------------------------------------------------------------------
def download_class_images(
    label: str,
    query: str,
    output_base_dir: Path,
    n_images: int,
    min_width: int,
    min_height: int,
) -> None:
    """
    Download images for a single semantic category.

    Parameters:
        label:
            Local class identifier used as directory name.
        query:
            Google Images search query associated with the category.
        output_base_dir:
            Root output directory.
        n_images:
            Maximum number of images to request for this category.
        min_width, min_height:
            Minimum image size constraints enforced by the crawler.

    Methodological note:
        The query acts as a weak supervision mechanism. The downloaded images
        are plausible candidates for the class, but not guaranteed ground truth.
    """
    output_dir = output_base_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading category '%s' using query '%s'", label, query)

    crawler = GoogleImageCrawler(storage={"root_dir": str(output_dir)})
    crawler.crawl(
        keyword=query,
        max_num=n_images,
        min_size=(min_width, min_height),
    )


def remove_invalid_images_from_class(class_dir: Path) -> tuple[int, int]:
    """
    Remove corrupted or unreadable images from a class directory.

    Returns:
        removed:
            Number of invalid files deleted.
        valid_count:
            Number of valid image files remaining after validation.
    """
    removed = 0

    for file_path in class_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            file_path.unlink(missing_ok=True)
            removed += 1

    valid_count = len(list_image_files(class_dir))
    return removed, valid_count


def collect_google_images(
    output_base_dir: Path,
    max_total_images: int,
    max_per_class: int,
    min_width: int,
    min_height: int,
) -> None:
    """
    Collect images across all predefined categories.

    Selection logic:
        - categories are processed sequentially
        - each category receives at most `max_per_class` requested images
        - the overall acquisition stops when `max_total_images` valid images
          have been accumulated
    """
    downloaded_total_valid = 0

    for label, query in CATEGORIES.items():
        if downloaded_total_valid >= max_total_images:
            break

        remaining_quota = max_total_images - downloaded_total_valid
        n_to_download = min(max_per_class, remaining_quota)

        if n_to_download <= 0:
            break

        download_class_images(
            label=label,
            query=query,
            output_base_dir=output_base_dir,
            n_images=n_to_download,
            min_width=min_width,
            min_height=min_height,
        )

        class_dir = output_base_dir / label
        removed_invalid, valid_after_validation = remove_invalid_images_from_class(class_dir)

        downloaded_total_valid += valid_after_validation

        logging.info(
            "Category '%s' -> removed=%d, valid=%d, cumulative_valid=%d",
            label,
            removed_invalid,
            valid_after_validation,
            downloaded_total_valid,
        )

    logging.info("Google scraping completed.")
    logging.info("Total valid images collected: %d", count_total_valid_images(output_base_dir))


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured summary of all files contained in the scraped dataset.

    This summary supports:
        - auditing
        - validation
        - reproducibility checks
        - file-level integrity verification
    """
    rows: list[dict[str, str | int]] = []

    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue

        rows.append(
            {
                "relative_path": str(file_path.relative_to(output_dir)).replace("\\", "/"),
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
                "is_image": int(is_image_file(file_path)),
                "sha256": compute_sha256(file_path),
            }
        )

    return rows


def write_summary_csv(rows: list[dict[str, str | int]], summary_csv: Path) -> None:
    """
    Write the scraping summary to CSV format.
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
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point of the scripts.

    Execution flow:
        1. Parse arguments
        2. Configure logging
        3. Prepare the output directory
        4. Run Google Images collection if needed
        5. Regenerate the summary CSV
    """
    args = parse_args()
    setup_logging(args.verbose)

    output_base_dir = repo_relative_path(args.output_dir)
    summary_csv = repo_relative_path(args.summary_csv)

    should_download = prepare_output_dir(output_base_dir, force=args.force)

    if should_download:
        collect_google_images(
            output_base_dir=output_base_dir,
            max_total_images=args.max_total_images,
            max_per_class=args.max_per_class,
            min_width=args.min_width,
            min_height=args.min_height,
        )

    summary_rows = build_summary_rows(output_base_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Summary CSV regenerated: %s", summary_csv)
    logging.info("Total indexed files: %d", len(summary_rows))
    logging.info(
        "Total indexed valid images: %d",
        sum(1 for row in summary_rows if int(row["is_image"]) == 1),
    )


if __name__ == "__main__":
    main()