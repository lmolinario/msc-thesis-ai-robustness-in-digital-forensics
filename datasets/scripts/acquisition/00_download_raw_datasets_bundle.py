#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_raw_datasets_bundle.py

Controlled-access bootstrap script.

No public URL is stored in this repository. Set the environment variable
FAIRLAB_RAW_DATASET_BUNDLE_URL on the local machine before running the script.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import gdown

from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path

ENV_NAME = "FAIRLAB_RAW_DATASET_BUNDLE_URL"
ARCHIVE_DIR = RAW_DATASETS_DIR / "downloaded_raw_archives"
ARCHIVE_PATH = ARCHIVE_DIR / "00_raw_datasets_bundle.zip"
EXTRACT_DIR = ARCHIVE_DIR / "extracted_bundle"


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract the local ZIP archive when the extraction directory is empty."""
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        print(f"[SKIP] Extraction directory already populated: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)
    print(f"[OK] Archive extracted to: {extract_dir}")


def main() -> None:
    """Download and extract the controlled-access archive."""
    file_url = os.getenv(ENV_NAME, "").strip()
    if not file_url:
        raise RuntimeError(
            f"Missing {ENV_NAME}. Set it locally to the controlled-access bundle URL."
        )

    archive_dir = repo_relative_path(ARCHIVE_DIR)
    archive_path = repo_relative_path(ARCHIVE_PATH)
    extract_dir = repo_relative_path(EXTRACT_DIR)
    archive_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"[SKIP] Archive already exists: {archive_path}")
    else:
        print(f"[INFO] Reading bundle URL from environment variable: {ENV_NAME}")
        gdown.download(file_url, str(archive_path), quiet=False, fuzzy=True)
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            raise RuntimeError("Download failed: archive file was not created correctly.")
        print(f"[OK] Download completed: {archive_path}")

    extract_archive(archive_path, extract_dir)
    print("[DONE] Controlled-access bundle bootstrap completed.")


if __name__ == "__main__":
    main()
