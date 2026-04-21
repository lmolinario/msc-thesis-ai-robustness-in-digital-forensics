#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 03_build_subset_deepfirearm.py
===============================================================================

Title:
    Deterministic Subset Builder for the DeepFirearm Dataset

Purpose:
    This scripts creates a deterministic image subset from the DeepFirearm dataset
    by selecting a limited number of images per class under configurable global
    and per-class constraints. It also regenerates a structured CSV summary
    describing the resulting subset contents.

Research Context:
    In the FAIR-Lab experimental workflow, this scripts supports dataset curation
    and controlled reduction of large source collections into a manageable and
    reproducible subset. This is especially relevant when the experimental design
    requires:

        - balanced or semi-balanced sampling across classes
        - controlled dataset size
        - deterministic reproducibility across runs
        - transparent documentation of selected files

Methodological Relevance:
    Subset construction is not a neutral preprocessing step. In forensic AI and
    adversarial robustness experiments, sampling decisions directly affect:

        - class representation
        - downstream model bias
        - robustness estimates
        - comparability across experimental runs

    For this reason, the scripts uses a fixed random seed and explicit class-level
    inclusion thresholds. The output CSV also includes SHA256 digests to support
    file-level traceability and integrity verification of the selected subset.

Inputs:
    - source directory containing one subdirectory per class
    - destination directory for the selected subset
    - CSV output path
    - selection constraints:
        * maximum total number of images
        * maximum number of images per class
        * minimum class size required for inclusion
        * deterministic random seed

Outputs:
    - curated subset directory
    - CSV summary file containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - the source dataset is already organized by class folders
    - each class directory contains image files directly at its root
    - image extensions are sufficient for file-type filtering
    - deterministic sampling is desirable for reproducibility

Dependencies:
    - Python standard library
    - project utilities from datasets.scripts.utils.paths

===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import shutil
from pathlib import Path

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, existing_path_validator, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = RAW_DATASETS_DIR / "02_deepfirearm" / "train"
DEFAULT_DEST_DIR = RAW_DATASETS_DIR / "02_deepfirearm" / "subset_choosen"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "02_deepfirearm" / "subset_summary.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    The scripts is designed to be configurable while preserving deterministic
    default settings suitable for reproducible experimental workflows.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic subset of the DeepFirearm dataset and "
            "regenerate the summary CSV."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Source directory containing DeepFirearm class folders (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=str(DEFAULT_DEST_DIR),
        help=f"Destination directory for the selected subset (default: {DEFAULT_DEST_DIR})",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help=f"Subset summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=1000,
        help="Maximum total number of images to select.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="Maximum number of images to select per class.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=20,
        help="Minimum number of source images required to include a class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the subset from scratch if the destination directory already exists.",
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
    Configure application logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def is_image_file(path: Path) -> bool:
    """
    Return True if the given path corresponds to a supported image file.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def compute_sha256(file_path: Path) -> str:
    """
    Compute the SHA256 digest of a file.

    This supports file-level integrity verification and subset traceability.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def list_class_dirs(root_dir: Path) -> list[Path]:
    """
    List all class directories contained in the source dataset root.
    """
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def list_images(class_dir: Path) -> list[Path]:
    """
    List all supported image files directly contained in a class directory.
    """
    return sorted([p for p in class_dir.iterdir() if is_image_file(p)])


def has_existing_subset(dest_dir: Path) -> bool:
    """
    Determine whether the destination directory already contains files.
    """
    return dest_dir.exists() and any(dest_dir.rglob("*"))


# -----------------------------------------------------------------------------
# Destination Preparation
# -----------------------------------------------------------------------------
def prepare_destination_for_rebuild(dest_dir: Path, force: bool) -> bool:
    """
    Prepare the destination directory before subset creation.

    Returns:
        True  -> the subset must be rebuilt
        False -> the subset already exists and should be reused
    """
    if dest_dir.exists():
        if force:
            logging.warning("Removing existing subset directory: %s", dest_dir)
            shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            return True

        if has_existing_subset(dest_dir):
            logging.info("Subset already available in %s. Rebuild not required.", dest_dir)
            return False

        dest_dir.mkdir(parents=True, exist_ok=True)
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    return True


# -----------------------------------------------------------------------------
# Subset Creation
# -----------------------------------------------------------------------------
def create_subset(
    source_dir: Path,
    dest_dir: Path,
    max_total_images: int,
    max_per_class: int,
    min_per_class: int,
    seed: int,
) -> None:
    """
    Create a deterministic subset from the source dataset.

    Selection policy:
        - classes with fewer than `min_per_class` images are excluded
        - at most `max_per_class` images are selected from each eligible class
        - selection stops once `max_total_images` has been reached
        - random sampling is deterministic thanks to the fixed seed
    """
    rng = random.Random(seed)

    class_dirs = list_class_dirs(source_dir)
    logging.info("Detected available classes: %d", len(class_dirs))

    total_selected = 0

    for class_dir in class_dirs:
        if total_selected >= max_total_images:
            break

        class_name = class_dir.name
        images = list_images(class_dir)
        source_count = len(images)

        if source_count < min_per_class:
            logging.debug(
                "Class excluded due to insufficient images: %s (%d)",
                class_name,
                source_count,
            )
            continue

        remaining_quota = max_total_images - total_selected
        n_select = min(source_count, max_per_class, remaining_quota)

        if n_select <= 0:
            break

        sampled_images = rng.sample(images, n_select)

        dest_class_dir = dest_dir / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sampled_images:
            shutil.copy2(img_path, dest_class_dir / img_path.name)

        total_selected += n_select

        logging.info(
            "Selected %d images for class '%s' out of %d available",
            n_select,
            class_name,
            source_count,
        )

    logging.info("Total selected images: %d", total_selected)
    logging.info("Subset created at: %s", dest_dir)


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured summary of all files contained in the generated subset.

    This metadata artifact supports:
        - subset auditing
        - quick dataset inspection
        - reproducibility checks
        - downstream validation workflows
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
    Write the subset summary to CSV format.
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
        3. Resolve and validate the source path
        4. Prepare the destination directory
        5. Create the subset if required
        6. Regenerate the summary CSV
    """
    args = parse_args()
    setup_logging(args.verbose)

    source_dir = repo_relative_path(args.source_dir)
    dest_dir = repo_relative_path(args.dest_dir)
    summary_csv = repo_relative_path(args.summary_csv)

    validate_source = existing_path_validator(
        "directory",
        lambda p: p.exists() and p.is_dir(),
    )
    source_dir = validate_source(source_dir)

    should_rebuild = prepare_destination_for_rebuild(dest_dir, force=args.force)

    if should_rebuild:
        create_subset(
            source_dir=source_dir,
            dest_dir=dest_dir,
            max_total_images=args.max_total_images,
            max_per_class=args.max_per_class,
            min_per_class=args.min_per_class,
            seed=args.seed,
        )

    summary_rows = build_summary_rows(dest_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Summary CSV regenerated: %s", summary_csv)
    logging.info("Total indexed files: %d", len(summary_rows))
    logging.info(
        "Total indexed images: %d",
        sum(1 for row in summary_rows if row["is_image"] == 1),
    )


if __name__ == "__main__":
    main()