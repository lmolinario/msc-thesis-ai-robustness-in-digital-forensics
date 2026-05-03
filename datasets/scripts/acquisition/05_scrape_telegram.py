#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 05_scrape_telegram.py
===============================================================================

Title:
    Telegram Image Collector and Metadata Indexer for OSINT-Based Dataset Expansion

Purpose:
    This scripts collects image data from a predefined set of public Telegram
    channels and generates a structured CSV summary of the downloaded files.
    It is designed to support OSINT-oriented dataset expansion within a
    reproducible research workflow.

Research Context:
    In the FAIR-Lab pipeline, Telegram scraping contributes to the acquisition of
    images from semi-open and operationally relevant social-media-like sources.
    Compared to benchmark datasets, such sources may better reflect real-world
    variability, including:

        - informal image distribution contexts
        - heterogeneous visual quality
        - non-curated object presentation
        - contextual ambiguity in weapon-related imagery

Methodological Relevance:
    In forensic AI and adversarial robustness research, the source domain of
    images is highly relevant. Telegram channels may expose models to
    distributional conditions that differ significantly from clean academic
    datasets. For this reason, this scripts should be interpreted as a
    dataset-ingestion utility for candidate samples rather than a ground-truth
    labeling system.

    The generated CSV summary includes SHA256 digests to support:
        - file-level integrity verification
        - provenance tracking
        - reproducibility of raw data indexing
        - downstream auditing

Security Note:
    API credentials must never be hardcoded in source code or committed to a
    public repository. They must be supplied through environment variables or a
    secure local configuration mechanism.

Inputs:
    - Telegram API credentials from environment variables
    - a local session name
    - a predefined list of Telegram channels
    - local output directory

Outputs:
    - downloaded image files organized by channel
    - CSV summary containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - the listed Telegram channels are public and accessible
    - the authenticated Telegram account is valid
    - message media of type photo is relevant to the collection process
    - downloaded images are candidate samples and may require later review

Dependencies:
    - Python standard library
    - Telethon
    - project paths utility

===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path

from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Telegram credentials must be provided securely through environment variables.
# Example:
#   export TELEGRAM_API_ID="your_api_id"
#   export TELEGRAM_API_HASH="your_api_hash"
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

# Local session name used by Telethon.
SESSION_NAME = os.getenv("TELEGRAM_SESSION_NAME", "telegram_osint_session")

# Base output directory for the Telegram dataset.
DEFAULT_OUTPUT_DIR = RAW_DATASETS_DIR / "04_telegram_youtube" / "osint_telegram"

# CSV file used to summarize the downloaded files.
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "04_telegram_youtube" / "telegram_download_summary.csv"

# Predefined public Telegram channels used as candidate image sources.
CHANNELS = [
    "weaponsandequipmentarchives",
    "SmithandWessoninc47",
    "UAWeapons",
]

# Supported image file extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

    This supports file-level integrity verification and provenance tracking.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def has_existing_dataset(output_dir: Path) -> bool:
    """
    Check whether the output directory already contains any files.

    This supports idempotent execution by preventing repeated downloads when
    the dataset is already present.
    """
    return output_dir.exists() and any(output_dir.rglob("*"))


def validate_credentials() -> tuple[int, str]:
    """
    Validate and return Telegram API credentials.

    Raises:
        ValueError: if one or more required credentials are missing.
    """
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        raise ValueError(
            "Telegram API credentials are missing. "
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH as environment variables."
        )

    return int(TELEGRAM_API_ID), TELEGRAM_API_HASH


# -----------------------------------------------------------------------------
# Telegram Acquisition Logic
# -----------------------------------------------------------------------------
def download_from_telegram(
    output_dir: Path,
    channels: list[str],
    session_name: str,
    message_limit: int = 200,
) -> None:
    """
    Download image media from a predefined list of Telegram channels.

    Parameters:
        output_dir:
            Root directory where downloaded files will be stored.
        channels:
            List of public Telegram channel usernames.
        session_name:
            Local session name used by Telethon authentication.
        message_limit:
            Maximum number of messages to inspect per channel.

    Methodological note:
        This function collects only photo-type media and stores them under
        channel-specific directories. Channel identity therefore acts as a
        source-domain label, not as a semantic class label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    api_id, api_hash = validate_credentials()
    client = TelegramClient(session_name, api_id, api_hash)

    with client:
        for channel in channels:
            logging.info("Processing Telegram channel: @%s", channel)

            channel_dir = output_dir / channel
            channel_dir.mkdir(parents=True, exist_ok=True)

            for msg in client.iter_messages(channel, limit=message_limit):
                if not isinstance(msg.media, MessageMediaPhoto):
                    continue

                filename = f"{msg.id}.jpg"
                filepath = channel_dir / filename

                # Avoid redownloading files already present in the local archive.
                if not filepath.exists():
                    client.download_media(msg.media, file=str(filepath))


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured summary of all files contained in the Telegram dataset.

    This metadata file is useful for:
        - auditing the acquired dataset
        - tracking source organization by channel
        - supporting downstream preprocessing and validation
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
    Write the Telegram dataset summary to CSV format.
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
        3. Download Telegram images if the dataset is not already present
        4. Regenerate the summary CSV
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir = repo_relative_path(DEFAULT_OUTPUT_DIR)
    summary_csv = repo_relative_path(DEFAULT_SUMMARY_CSV)

    if not has_existing_dataset(output_dir):
        download_from_telegram(
            output_dir=output_dir,
            channels=CHANNELS,
            session_name=SESSION_NAME,
            message_limit=200,
        )

    summary_rows = build_summary_rows(output_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Summary CSV regenerated: %s", summary_csv)
    logging.info("Total indexed files: %d", len(summary_rows))
    logging.info(
        "Total indexed images: %d",
        sum(1 for row in summary_rows if int(row["is_image"]) == 1),
    )


if __name__ == "__main__":
    main()