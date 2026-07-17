#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_raw_datasets_bundle.py

Restore the externally hosted raw dataset bundle used by the FAIR-Lab thesis
pipeline. Image corpora are intentionally not tracked on the public ``main``
branch.

The bundle is distributed under controlled access. The download URL is supplied
only after authorization and must be provided locally through either:

1. the ``--url`` command-line argument; or
2. the ``FAIRLAB_RAW_DATASET_BUNDLE_URL`` environment variable.

No private Google Drive URL is stored in this repository. The archive is
downloaded under ``datasets/raw/downloaded_raw_archives/`` and extracted
locally. Downloaded and extracted data are ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path

import gdown

from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path

ENV_NAME = "FAIRLAB_RAW_DATASET_BUNDLE_URL"
ACCESS_REQUEST_NOTE = (
    "Access is granted case by case by the thesis author or repository "
    "maintainer. Request authorization before running this script."
)

ARCHIVE_DIR = RAW_DATASETS_DIR / "downloaded_raw_archives"
ARCHIVE_PATH = ARCHIVE_DIR / "00_raw_datasets_bundle.zip"
EXTRACT_DIR = ARCHIVE_DIR / "extracted_bundle"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and safely extract the controlled-access FAIR-Lab raw "
            "dataset bundle."
        )
    )
    parser.add_argument(
        "--url",
        default="",
        help=(
            "Authorized Google Drive or direct bundle URL. When omitted, the "
            f"script reads {ENV_NAME}."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing local archive before downloading.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace an existing extraction directory.",
    )
    return parser


def resolve_bundle_url(cli_url: str) -> tuple[str, str]:
    """Resolve the authorized bundle URL and report its local source."""
    cli_value = cli_url.strip()
    if cli_value:
        return cli_value, "command line"

    env_value = os.getenv(ENV_NAME, "").strip()
    if env_value:
        return env_value, f"environment variable {ENV_NAME}"

    raise RuntimeError(
        f"Missing authorized dataset bundle URL. Provide --url or set {ENV_NAME}. "
        f"{ACCESS_REQUEST_NOTE}"
    )


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_archive(path: Path) -> None:
    """Ensure that the downloaded object is a non-empty ZIP archive."""
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded archive is missing or empty: {path}")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive. Confirm that access "
            "has been granted for the current account and that the supplied URL "
            "points to the authorized raw dataset bundle."
        )


def validate_member_destination(member: zipfile.ZipInfo, extract_dir: Path) -> None:
    """Reject path traversal and symbolic-link entries before extraction."""
    extract_root = extract_dir.resolve()
    destination = (extract_root / member.filename).resolve()
    if destination != extract_root and extract_root not in destination.parents:
        raise RuntimeError(f"Unsafe archive path detected: {member.filename}")

    unix_mode = member.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise RuntimeError(
            f"Symbolic links are not allowed in the bundle: {member.filename}"
        )


def extract_archive(archive_path: Path, extract_dir: Path, force: bool) -> None:
    """Safely extract the bundle, optionally replacing an existing directory."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        if not force:
            print(f"[SKIP] Extraction directory already populated: {extract_dir}")
            return
        print(f"[INFO] Removing existing extraction directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            validate_member_destination(member, extract_dir)
        archive.extractall(extract_dir)
    print(f"[OK] Archive extracted to: {extract_dir}")


def download_archive(url: str, archive_path: Path, force: bool) -> None:
    """Download the authorized bundle with gdown and validate the archive."""
    if archive_path.exists() and archive_path.stat().st_size > 0:
        if not force:
            validate_zip_archive(archive_path)
            print(f"[SKIP] Valid archive already exists: {archive_path}")
            return
        print(f"[INFO] Removing existing archive: {archive_path}")
        archive_path.unlink()

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = gdown.download(
            url=url,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Controlled-access download failed. Confirm that authorization was "
            "granted to the account used for the download and that the supplied "
            f"URL is current. The URL may be provided with --url or {ENV_NAME}."
        ) from exc

    if result is None:
        raise RuntimeError(
            "The controlled-access download returned no file. Confirm the access "
            "authorization and request a current link from the thesis author or "
            "repository maintainer."
        )

    validate_zip_archive(archive_path)
    print(f"[OK] Download completed: {archive_path}")


def main() -> None:
    """Download, validate, hash, and safely extract the authorized bundle."""
    args = build_parser().parse_args()
    bundle_url, url_source = resolve_bundle_url(args.url)

    archive_dir = repo_relative_path(ARCHIVE_DIR)
    archive_path = repo_relative_path(ARCHIVE_PATH)
    extract_dir = repo_relative_path(EXTRACT_DIR)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Authorized bundle URL source: {url_source}")
    download_archive(bundle_url, archive_path, args.force_download)
    print(f"[INFO] Archive SHA256: {sha256_file(archive_path)}")
    extract_archive(archive_path, extract_dir, args.force_extract)
    print("[DONE] Controlled-access raw dataset restoration completed.")


if __name__ == "__main__":
    main()
