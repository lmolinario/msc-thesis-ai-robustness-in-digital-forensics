#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_build_prepared_dataset.py

Builds the prepared/final_pool dataset from datasets/raw by:
- scanning candidate image files
- validating images technically
- computing SHA256
- removing exact duplicates globally
- copying valid unique images into datasets/prepared/final_pool/images
- generating metadata.csv
- generating reports for invalid images, discarded duplicates, and build summary
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, PREPARED_DATASETS_DIR, repo_relative_path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_SOURCE_DATASETS = [
    "01_kaggle_weapon",
    "02_deepfirearm",
    "03_google_scraped",
    "04_telegram_youtube",
    "05_deepweb",
]

DEFAULT_OUTPUT_DIR = PREPARED_DATASETS_DIR / "final_pool"
DEFAULT_IMAGES_DIRNAME = "images"
DEFAULT_REPORTS_DIRNAME = "reports"
DEFAULT_METADATA_FILENAME = "metadata.csv"
DEFAULT_INVALID_REPORT = "invalid_images.csv"
DEFAULT_DUPLICATES_REPORT = "duplicates_discarded.csv"
DEFAULT_SUMMARY_JSON = "prepared_build_summary.json"


@dataclass
class ValidImageRecord:
    image_id: str
    prepared_filename: str
    prepared_relative_path: str
    source_group: str
    source_dataset: str
    source_relative_path: str
    source_filename: str
    sha256: str
    width: int
    height: int
    size_bytes: int
    extension: str
    is_valid_image: bool
    selected_for_manual_review: bool
    manual_label: str
    review_status: str
    review_notes: str


@dataclass
class InvalidImageRecord:
    source_group: str
    source_dataset: str
    source_relative_path: str
    source_filename: str
    reason: str


@dataclass
class DuplicateDiscardedRecord:
    sha256: str
    kept_image_id: str
    kept_source_dataset: str
    kept_source_relative_path: str
    discarded_source_dataset: str
    discarded_source_relative_path: str
    discarded_source_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prepared/final_pool from raw datasets with validation and global deduplication."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(RAW_DATASETS_DIR),
        help=f"Root directory of raw datasets (default: {RAW_DATASETS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Prepared final pool output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=300,
        help="Minimum image width (default: 300)",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=300,
        help="Minimum image height (default: 300)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the output directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def source_group_from_dataset(source_dataset: str) -> str:
    mapping = {
        "01_kaggle_weapon": "kaggle",
        "02_deepfirearm": "deepfirearm",
        "03_google_scraped": "google",
        "04_telegram_youtube": "telegram_youtube",
        "05_deepweb": "deepweb",
    }
    return mapping.get(source_dataset, source_dataset.lower())


def is_candidate_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def iter_candidate_files(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.rglob("*")):
        if is_candidate_image(path):
            yield path


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_image(file_path: Path, min_width: int, min_height: int) -> tuple[bool, int, int, str]:
    try:
        with Image.open(file_path) as img:
            img.verify()

        with Image.open(file_path) as img:
            width, height = img.size

        if width < min_width or height < min_height:
            return False, width, height, "below_min_size"

        return True, width, height, ""

    except (UnidentifiedImageError, OSError, ValueError):
        return False, 0, 0, "unreadable_image"
    except Exception as exc:
        return False, 0, 0, f"unexpected_error:{type(exc).__name__}"


def ensure_clean_output_dir(output_dir: Path, force: bool) -> tuple[Path, Path, Path]:
    images_dir = output_dir / DEFAULT_IMAGES_DIRNAME
    reports_dir = output_dir / DEFAULT_REPORTS_DIRNAME

    if output_dir.exists():
        if force:
            logging.warning("Removing existing output directory: %s", output_dir)
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to rebuild it."
            )

    images_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, images_dir, reports_dir


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    raw_root = repo_relative_path(args.raw_root)
    output_dir = repo_relative_path(args.output_dir)

    _, images_dir, reports_dir = ensure_clean_output_dir(output_dir, force=args.force)

    metadata_path = output_dir / DEFAULT_METADATA_FILENAME
    invalid_report_path = reports_dir / DEFAULT_INVALID_REPORT
    duplicates_report_path = reports_dir / DEFAULT_DUPLICATES_REPORT
    summary_json_path = reports_dir / DEFAULT_SUMMARY_JSON

    kept_by_sha256: dict[str, ValidImageRecord] = {}
    metadata_rows: list[ValidImageRecord] = []
    invalid_rows: list[InvalidImageRecord] = []
    duplicate_rows: list[DuplicateDiscardedRecord] = []

    total_candidate_files = 0
    valid_unique_images = 0
    invalid_images = 0
    duplicates_discarded = 0

    per_dataset_stats: dict[str, dict[str, int]] = {}

    image_counter = 0

    for source_dataset in DEFAULT_SOURCE_DATASETS:
        dataset_root = raw_root / source_dataset
        if not dataset_root.exists() or not dataset_root.is_dir():
            logging.warning("Skipping missing dataset directory: %s", dataset_root)
            continue

        source_group = source_group_from_dataset(source_dataset)
        per_dataset_stats[source_dataset] = {
            "candidate_files": 0,
            "valid_unique_images": 0,
            "invalid_images": 0,
            "duplicates_discarded": 0,
        }

        logging.info("Processing source dataset: %s", source_dataset)

        for file_path in iter_candidate_files(dataset_root):
            total_candidate_files += 1
            per_dataset_stats[source_dataset]["candidate_files"] += 1

            source_relative_path = str(file_path.relative_to(dataset_root)).replace("\\", "/")
            source_filename = file_path.name

            is_valid, width, height, reason = validate_image(
                file_path=file_path,
                min_width=args.min_width,
                min_height=args.min_height,
            )

            if not is_valid:
                invalid_images += 1
                per_dataset_stats[source_dataset]["invalid_images"] += 1
                invalid_rows.append(
                    InvalidImageRecord(
                        source_group=source_group,
                        source_dataset=source_dataset,
                        source_relative_path=source_relative_path,
                        source_filename=source_filename,
                        reason=reason,
                    )
                )
                continue

            sha256 = compute_sha256(file_path)

            if sha256 in kept_by_sha256:
                duplicates_discarded += 1
                per_dataset_stats[source_dataset]["duplicates_discarded"] += 1
                kept = kept_by_sha256[sha256]
                duplicate_rows.append(
                    DuplicateDiscardedRecord(
                        sha256=sha256,
                        kept_image_id=kept.image_id,
                        kept_source_dataset=kept.source_dataset,
                        kept_source_relative_path=kept.source_relative_path,
                        discarded_source_dataset=source_dataset,
                        discarded_source_relative_path=source_relative_path,
                        discarded_source_filename=source_filename,
                    )
                )
                continue

            image_counter += 1
            image_id = f"img_{image_counter:08d}"
            extension = file_path.suffix.lower()
            prepared_filename = f"{image_id}{extension}"
            prepared_relative_path = f"{DEFAULT_IMAGES_DIRNAME}/{prepared_filename}"
            prepared_file_path = images_dir / prepared_filename

            shutil.copy2(file_path, prepared_file_path)

            copied_sha256 = compute_sha256(prepared_file_path)
            if copied_sha256 != sha256:
                raise RuntimeError(
                    f"SHA256 mismatch after copy: source={file_path} destination={prepared_file_path}"
                )

            record = ValidImageRecord(
                image_id=image_id,
                prepared_filename=prepared_filename,
                prepared_relative_path=prepared_relative_path,
                source_group=source_group,
                source_dataset=source_dataset,
                source_relative_path=source_relative_path,
                source_filename=source_filename,
                sha256=sha256,
                width=width,
                height=height,
                size_bytes=prepared_file_path.stat().st_size,
                extension=extension,
                is_valid_image=True,
                selected_for_manual_review=True,
                manual_label="",
                review_status="pending",
                review_notes="",
            )

            kept_by_sha256[sha256] = record
            metadata_rows.append(record)

            valid_unique_images += 1
            per_dataset_stats[source_dataset]["valid_unique_images"] += 1

    write_csv(
        metadata_path,
        fieldnames=[
            "image_id",
            "prepared_filename",
            "prepared_relative_path",
            "source_group",
            "source_dataset",
            "source_relative_path",
            "source_filename",
            "sha256",
            "width",
            "height",
            "size_bytes",
            "extension",
            "is_valid_image",
            "selected_for_manual_review",
            "manual_label",
            "review_status",
            "review_notes",
        ],
        rows=[row.__dict__ for row in metadata_rows],
    )

    write_csv(
        invalid_report_path,
        fieldnames=[
            "source_group",
            "source_dataset",
            "source_relative_path",
            "source_filename",
            "reason",
        ],
        rows=[row.__dict__ for row in invalid_rows],
    )

    write_csv(
        duplicates_report_path,
        fieldnames=[
            "sha256",
            "kept_image_id",
            "kept_source_dataset",
            "kept_source_relative_path",
            "discarded_source_dataset",
            "discarded_source_relative_path",
            "discarded_source_filename",
        ],
        rows=[row.__dict__ for row in duplicate_rows],
    )

    summary = {
        "total_candidate_files": total_candidate_files,
        "valid_unique_images": valid_unique_images,
        "invalid_images": invalid_images,
        "duplicates_discarded": duplicates_discarded,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "source_datasets_processed": sum(
            1 for dataset in DEFAULT_SOURCE_DATASETS if (raw_root / dataset).exists()
        ),
        "per_dataset_stats": per_dataset_stats,
        "output_dir": str(output_dir),
        "metadata_csv": str(metadata_path),
        "invalid_report_csv": str(invalid_report_path),
        "duplicates_report_csv": str(duplicates_report_path),
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Prepared dataset build completed.")
    logging.info("Valid unique images: %d", valid_unique_images)
    logging.info("Invalid images: %d", invalid_images)
    logging.info("Duplicates discarded: %d", duplicates_discarded)
    logging.info("Metadata CSV: %s", metadata_path)
    logging.info("Invalid images report: %s", invalid_report_path)
    logging.info("Duplicates report: %s", duplicates_report_path)
    logging.info("Summary JSON: %s", summary_json_path)


if __name__ == "__main__":
    main()