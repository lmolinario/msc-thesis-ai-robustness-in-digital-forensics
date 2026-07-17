#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_raw_datasets_bundle.py

Restore the externally hosted raw dataset bundle used by the FAIR-Lab thesis
pipeline. Image corpora are intentionally not tracked on the public ``main``
branch.

The Google Drive page is intentionally retained so that an interested reviewer
can request access. The file itself remains restricted and access is granted
case by case by the thesis author or repository maintainer.

Recommended workflow:

1. run this script with ``--request-access``;
2. sign in to Google Drive and submit the access request;
3. after approval, download the ZIP through the browser;
4. pass the downloaded ZIP to this script with ``--archive``.

An authorized direct-download URL may alternatively be supplied through
``--url`` or ``FAIRLAB_RAW_DATASET_BUNDLE_URL``. Private or temporary URLs must
not be committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import webbrowser
import zipfile
from pathlib import Path

import gdown

from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path

ENV_NAME = "FAIRLAB_RAW_DATASET_BUNDLE_URL"
REQUEST_ACCESS_URL = (
    "https://drive.google.com/file/d/"
    "1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link"
)

ARCHIVE_DIR = RAW_DATASETS_DIR / "downloaded_raw_archives"
ARCHIVE_PATH = ARCHIVE_DIR / "00_raw_datasets_bundle.zip"
EXTRACT_DIR = ARCHIVE_DIR / "extracted_bundle"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Request access to, download, or safely extract the controlled-access "
            "FAIR-Lab raw dataset bundle."
        )
    )
    parser.add_argument(
        "--request-access",
        action="store_true",
        help=(
            "Open the restricted Google Drive page in the default browser so that "
            "the current Google account can submit an access request."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help=(
            "Path to a ZIP downloaded through the browser after access approval. "
            "The archive is validated and extracted without uploading it to Git."
        ),
    )
    parser.add_argument(
        "--url",
        default="",
        help=(
            "Authorized direct-download URL. When omitted, the script reads "
            f"{ENV_NAME}."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing locally downloaded archive.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Replace an existing extraction directory.",
    )
    return parser


def open_access_request_page() -> None:
    """Open the restricted Drive page used to submit an access request."""
    print("[INFO] Opening the controlled-access request page:")
    print(f"       {REQUEST_ACCESS_URL}")
    opened = webbrowser.open(REQUEST_ACCESS_URL, new=2)
    if not opened:
        print("[WARN] The browser could not be opened automatically.")
        print("       Copy the URL above into a browser and select 'Request access'.")


def resolve_bundle_url(cli_url: str) -> tuple[str, str]:
    """Resolve an authorized direct-download URL and report its local source."""
    cli_value = cli_url.strip()
    if cli_value:
        return cli_value, "command line"

    env_value = os.getenv(ENV_NAME, "").strip()
    if env_value:
        return env_value, f"environment variable {ENV_NAME}"

    raise RuntimeError(
        "No authorized direct-download URL was provided. Run this script with "
        "--request-access, wait for approval, download the ZIP through the browser, "
        "and then use --archive <downloaded-zip>. Alternatively, provide an "
        f"authorized URL with --url or {ENV_NAME}."
    )


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_archive(path: Path) -> None:
    """Ensure that the supplied object is a non-empty ZIP archive."""
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Archive is missing or empty: {path}")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            "The supplied file is not a valid ZIP archive. Confirm that it is the "
            "bundle downloaded after access approval."
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
    """Download an authorized bundle URL with gdown and validate the archive."""
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
        )
    except Exception as exc:
        raise RuntimeError(
            "The direct download failed. A restricted Drive page cannot submit an "
            "access request through gdown. Run with --request-access, complete the "
            "approval flow in the browser, download the ZIP, and use --archive."
        ) from exc

    if result is None:
        raise RuntimeError(
            "The direct download returned no file. Complete the controlled-access "
            "flow in the browser and use --archive with the downloaded ZIP."
        )

    validate_zip_archive(archive_path)
    print(f"[OK] Download completed: {archive_path}")


def main() -> None:
    """Request access, download, validate, hash, and safely extract the bundle."""
    args = build_parser().parse_args()

    if args.request_access:
        open_access_request_page()
        if args.archive is None and not args.url.strip() and not os.getenv(ENV_NAME, "").strip():
            print("[DONE] Submit the browser request and rerun after authorization.")
            return

    extract_dir = repo_relative_path(EXTRACT_DIR)

    if args.archive is not None:
        local_archive = args.archive.expanduser().resolve()
        validate_zip_archive(local_archive)
        print(f"[INFO] Local archive: {local_archive}")
        print(f"[INFO] Archive SHA256: {sha256_file(local_archive)}")
        extract_archive(local_archive, extract_dir, args.force_extract)
        print("[DONE] Controlled-access raw dataset restoration completed.")
        return

    bundle_url, url_source = resolve_bundle_url(args.url)
    archive_path = repo_relative_path(ARCHIVE_PATH)

    print(f"[INFO] Authorized direct-download URL source: {url_source}")
    download_archive(bundle_url, archive_path, args.force_download)
    print(f"[INFO] Archive SHA256: {sha256_file(archive_path)}")
    extract_archive(archive_path, extract_dir, args.force_extract)
    print("[DONE] Controlled-access raw dataset restoration completed.")


if __name__ == "__main__":
    main()
