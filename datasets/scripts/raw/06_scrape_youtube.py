#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 06_scrape_youtube.py
===============================================================================

Title:
    YouTube Thumbnail Collector with Google Fallback and Metadata Indexer

Purpose:
    This scripts collects image data related to weapon-oriented scenarios by
    retrieving YouTube video thumbnails from query-based searches. If no valid
    thumbnails are found for a given category, the scripts falls back to Google
    Images scraping. It then validates the downloaded files and generates a
    structured CSV summary of the resulting dataset.

Research Context:
    In the FAIR-Lab experimental workflow, this scripts contributes to the
    acquisition of visually heterogeneous images from multimedia-oriented and
    open-source intelligence (OSINT) environments. Compared with static,
    curated datasets, YouTube thumbnails may capture more realistic settings,
    including:

        - surveillance-like scenes
        - news and incident-related imagery
        - public contextual representations of firearms
        - visually noisy or weakly structured operational scenarios

Methodological Relevance:
    In forensic AI and adversarial robustness research, source diversity is
    important for evaluating how image classifiers behave outside clean
    benchmark conditions. Query-based retrieval from YouTube and fallback
    scraping from Google Images can help increase variability, but such data
    should be treated as weakly supervised candidate samples rather than
    verified ground truth.

    For this reason, the scripts:
        - organizes downloads by semantic query category
        - validates all files after download
        - preserves the source acquisition structure
        - produces a metadata summary for auditing and downstream processing
        - records SHA256 digests for file-level integrity verification

Inputs:
    - predefined category-to-query mapping
    - maximum number of YouTube search results per category
    - Google fallback image limit
    - minimum image size for fallback crawling

Outputs:
    - category-structured image dataset
    - CSV summary file containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - internet access is available
    - yt_dlp is correctly installed and functional
    - YouTube thumbnail URLs are accessible
    - Google fallback is acceptable for empty YouTube results
    - downloaded images may require later manual inspection and relabeling

Dependencies:
    - Python standard library
    - yt_dlp
    - icrawler
    - Pillow
    - project utility: utils.paths

Important Note:
    Images collected through query-based retrieval are not guaranteed to be
    semantically correct. They should be considered candidate samples for
    subsequent validation, deduplication, and annotation.
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import logging
import urllib.request
from pathlib import Path

import yt_dlp
from icrawler.builtin import GoogleImageCrawler
from PIL import Image, UnidentifiedImageError

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_BASE_DIR = RAW_DATASETS_DIR / "04_telegram_youtube" / "osint_youtube"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "04_telegram_youtube" / "youtube_download_summary.csv"

MAX_VIDEOS = 100
GOOGLE_FALLBACK_MAX_IMAGES = 100
MIN_SIZE = (200, 200)

CATEGORIES = {
    "cctv_firearm_incident": "firearm incident CCTV",
    "gun_in_bag": "gun in bag",
    "toy_gun_or_airsoft": "airsoft toy gun",
    "training_police_shooting": "police shooting training",
    "person_with_gun": "person with gun",
    "crime_scene_weapon": "crime scene weapon",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def is_image_file(path: Path) -> bool:
    """
    Return True if the given file has a supported image extension.
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


def validate_image_file(file_path: Path) -> bool:
    """
    Validate whether a downloaded file is a readable image.

    This function removes a common source of noise in scraped datasets:
    partially downloaded, malformed, or non-image files.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def has_existing_dataset(output_base_dir: Path) -> bool:
    """
    Check whether the output directory already contains any files.

    This enables idempotent execution, avoiding repeated downloads if the
    dataset is already available locally.
    """
    return output_base_dir.exists() and any(output_base_dir.rglob("*"))


# -----------------------------------------------------------------------------
# Fallback Acquisition
# -----------------------------------------------------------------------------
def fallback_google_images(query: str, output_dir: Path, max_images: int = 100) -> None:
    """
    Download fallback images from Google Images when no valid YouTube thumbnails
    are available for a given query.

    Methodological note:
        This fallback increases acquisition robustness, but it also mixes source
        domains. Therefore, source provenance should ideally be tracked in later
        versions of the pipeline.
    """
    crawler = GoogleImageCrawler(storage={"root_dir": str(output_dir)})
    crawler.crawl(keyword=query, max_num=max_images, min_size=MIN_SIZE)


# -----------------------------------------------------------------------------
# YouTube Thumbnail Acquisition
# -----------------------------------------------------------------------------
def download_youtube_thumbnails(query: str, label: str, output_base_dir: Path) -> int:
    """
    Download YouTube thumbnails for a given semantic query.

    Parameters:
        query:
            Search string submitted to yt_dlp through YouTube search.
        label:
            Local dataset label associated with the query.
        output_base_dir:
            Root output directory for all categories.

    Returns:
        Number of valid YouTube thumbnails retained after validation.

    Methodological note:
        The semantic query acts as a weak supervision mechanism. The resulting
        thumbnails may be visually relevant but not necessarily class-pure.
    """
    output_dir = output_base_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,
        "nocheckcertificate": True,
    }

    valid_count = 0

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch{MAX_VIDEOS}:{query}", download=False)
            entries = result.get("entries", [])

            for i, entry in enumerate(entries):
                thumb_url = entry.get("thumbnail")
                if not thumb_url:
                    continue

                filename = output_dir / f"{label}_{i}.jpg"

                try:
                    if not filename.exists():
                        urllib.request.urlretrieve(thumb_url, filename)

                    if validate_image_file(filename):
                        valid_count += 1
                    else:
                        filename.unlink(missing_ok=True)

                except Exception:
                    filename.unlink(missing_ok=True)

    except Exception as exc:
        logging.warning("yt-dlp error for query '%s': %s", query, exc)

    if valid_count == 0:
        logging.info(
            "No valid YouTube thumbnails found for '%s'. Switching to Google fallback.",
            label,
        )
        fallback_google_images(query, output_dir, max_images=GOOGLE_FALLBACK_MAX_IMAGES)

        for file_path in output_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not validate_image_file(file_path):
                file_path.unlink(missing_ok=True)

    return valid_count


def download_dataset(output_base_dir: Path) -> None:
    """
    Download the full dataset across all predefined categories.

    Each category is processed independently. The acquisition logic first
    attempts YouTube thumbnails and only uses Google Images as a fallback
    when no valid thumbnail is available.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for label, query in CATEGORIES.items():
        logging.info("Processing query: %s", query)
        n_valid = download_youtube_thumbnails(query, label, output_base_dir)
        logging.info("Category '%s': %d valid YouTube thumbnails saved", label, n_valid)
        total_downloaded += n_valid

    logging.info(
        "Initial YouTube download phase completed. Total valid thumbnails saved: %d",
        total_downloaded,
    )


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured summary of all files in the dataset directory.

    This summary provides a compact metadata artifact for:
        - auditing
        - downstream preprocessing
        - provenance tracking
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
    Write the metadata summary to CSV format.
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
    Main entry point of the scripts.

    Execution flow:
        1. Configure logging
        2. Resolve output paths
        3. Download the dataset if not already present
        4. Generate the metadata summary CSV
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_base_dir = repo_relative_path(DEFAULT_OUTPUT_BASE_DIR)
    summary_csv = repo_relative_path(DEFAULT_SUMMARY_CSV)

    if not has_existing_dataset(output_base_dir):
        download_dataset(output_base_dir)

    summary_rows = build_summary_rows(output_base_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Summary CSV regenerated: %s", summary_csv)
    logging.info("Total indexed files: %d", len(summary_rows))
    logging.info(
        "Total indexed images: %d",
        sum(1 for row in summary_rows if int(row["is_image"]) == 1),
    )


if __name__ == "__main__":
    main()