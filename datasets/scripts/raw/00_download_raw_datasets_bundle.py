#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_and_unpack_raw_datasets_bundle.py

Official raw dataset bootstrap script for the public FAIR-Lab thesis repository.

Purpose
-------
This script downloads the archived raw dataset bundle from Google Drive and
unpacks it into the official raw dataset directory structure used by the thesis.

The script is intentionally designed to be:
- reproducible
- defensive
- readable
- suitable for academic release
- safe to re-run multiple times

Main features
-------------
1. Downloads the raw dataset bundle from Google Drive.
2. Skips the download if the archive already exists.
3. Extracts the archive locally.
4. Detects the expected raw dataset folders inside the extracted bundle.
5. Copies each dataset into its official destination under datasets/raw/.
6. Skips extraction/copy when the destination already contains files.

Expected official raw dataset folders
-------------------------------------
- 01_kaggle_weapon
- 02_deepfirearm
- 03_google_scraped
- 04_telegram_youtube
- 05_deepweb

Output structure
----------------
- datasets/raw/downloaded_raw_archives/00_raw_datasets_bundle.zip
- datasets/raw/01_kaggle_weapon/
- datasets/raw/02_deepfirearm/
- datasets/raw/03_google_scraped/
- datasets/raw/04_telegram_youtube/
- datasets/raw/05_deepweb/

Notes
-----
- This script does not overwrite already-populated raw dataset folders.
- It is therefore safe for repeated executions in the public repository.
- The script assumes the downloaded archive is a ZIP file.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from html import unescape
from pathlib import Path

import requests
from tqdm import tqdm

from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path


# =============================================================================
# Configuration
# =============================================================================

FILE_URL = "https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=sharing"

ARCHIVE_DIR = RAW_DATASETS_DIR / "downloaded_raw_archives"
ARCHIVE_PATH = ARCHIVE_DIR / "00_raw_datasets_bundle.zip"
EXTRACT_DIR = ARCHIVE_DIR / "extracted_bundle"

CHUNK_SIZE = 8 * 1024 * 1024

EXPECTED_RAW_DATASET_DIRS = [
    "01_kaggle_weapon",
    "02_deepfirearm",
    "03_google_scraped",
    "04_telegram_youtube",
    "05_deepweb",
]


# =============================================================================
# Google Drive download helpers
# =============================================================================

def extract_file_id(url: str) -> str:
    """
    Extract the Google Drive file ID from a sharing URL.
    """
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Unable to extract file_id from URL: {url}")


def is_html_response(response: requests.Response) -> bool:
    """
    Return True when the HTTP response appears to be HTML instead of binary data.
    """
    content_type = response.headers.get("Content-Type", "").lower()
    return "text/html" in content_type


def save_response_content(response: requests.Response, destination: Path) -> None:
    """
    Stream the HTTP response body to disk with a progress bar.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get("Content-Length", 0))

    with destination.open("wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=destination.name,
    ) as pbar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_confirm_token_from_cookies(response: requests.Response) -> str | None:
    """
    Extract the Google Drive download confirmation token from cookies, if present.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def extract_download_form(html: str) -> tuple[str, str, dict[str, str]] | None:
    """
    Extract method, action, and hidden fields from the Google Drive warning form.
    """
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*method="([^"]+)"[^>]*>(.*?)</form>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not form_match:
        form_match = re.search(
            r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not form_match:
            return None

        action = unescape(form_match.group(1))
        method = "GET"
        form_html = form_match.group(2)
    else:
        action = unescape(form_match.group(1))
        method = form_match.group(2).upper()
        form_html = form_match.group(3)

    inputs = re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
        form_html,
        flags=re.IGNORECASE,
    )

    data = {name: unescape(value) for name, value in inputs}
    return method, action, data


def download_google_drive_file(file_id: str, destination: Path) -> None:
    """
    Download a Google Drive file while handling:
    1. direct binary response
    2. cookie-based confirmation token
    3. virus-scan warning form
    """
    base_url = "https://drive.google.com/uc?export=download"

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        response = session.get(
            base_url,
            params={"id": file_id},
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        if not is_html_response(response):
            save_response_content(response, destination)
            return

        html = response.text

        token = extract_confirm_token_from_cookies(response)
        if token:
            response.close()
            response = session.get(
                base_url,
                params={"id": file_id, "confirm": token},
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()

            if not is_html_response(response):
                save_response_content(response, destination)
                return

            html = response.text

        form = extract_download_form(html)
        if form is not None:
            method, action, data = form
            data.setdefault("id", file_id)

            response.close()

            if method == "POST":
                response = session.post(
                    action,
                    data=data,
                    stream=True,
                    allow_redirects=True,
                )
            else:
                response = session.get(
                    action,
                    params=data,
                    stream=True,
                    allow_redirects=True,
                )

            response.raise_for_status()

            if is_html_response(response):
                snippet = response.text[:1000]
                raise RuntimeError(
                    "Google Drive returned HTML instead of the binary archive.\n"
                    f"Response preview:\n{snippet}"
                )

            save_response_content(response, destination)
            return

        snippet = html[:1000]
        raise RuntimeError(
            "Unable to extract the Google Drive confirmation form.\n"
            f"Response preview:\n{snippet}"
        )


# =============================================================================
# Archive helpers
# =============================================================================

def directory_has_any_files(path: Path) -> bool:
    """
    Return True if the directory exists and contains at least one file anywhere
    in its subtree.
    """
    return path.exists() and any(p.is_file() for p in path.rglob("*"))


def all_expected_raw_dirs_already_present(raw_root: Path) -> bool:
    """
    Return True if all expected official raw dataset folders already exist
    and contain at least one file.
    """
    for name in EXPECTED_RAW_DATASET_DIRS:
        if not directory_has_any_files(raw_root / name):
            return False
    return True


def extract_zip_archive(zip_path: Path, extract_dir: Path) -> None:
    """
    Extract the ZIP archive into the chosen extraction directory.

    If the extraction directory already contains files, extraction is skipped.
    """
    if directory_has_any_files(extract_dir):
        print(f"[SKIP] Extraction already available: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"[OK] Archive extracted to: {extract_dir}")


def find_dataset_dir(extract_root: Path, dataset_name: str) -> Path | None:
    """
    Search recursively for the given dataset directory name inside the extracted bundle.
    """
    candidates = [p for p in extract_root.rglob(dataset_name) if p.is_dir()]
    if not candidates:
        return None

    # Prefer the shortest path (usually the most canonical one)
    candidates = sorted(candidates, key=lambda p: len(p.parts))
    return candidates[0]


def copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy all contents from src_dir into dst_dir.

    This function copies children one by one to avoid nesting src_dir itself
    under the destination.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        target = dst_dir / item.name

        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def sync_expected_raw_datasets(extract_root: Path, raw_root: Path) -> None:
    """
    Populate the official raw dataset directories from the extracted bundle.

    If a destination dataset directory already contains files, it is skipped.
    """
    for dataset_name in EXPECTED_RAW_DATASET_DIRS:
        dst_dir = raw_root / dataset_name

        if directory_has_any_files(dst_dir):
            print(f"[SKIP] Raw dataset already present: {dst_dir}")
            continue

        src_dir = find_dataset_dir(extract_root, dataset_name)
        if src_dir is None:
            print(f"[WARN] Expected dataset directory not found in archive: {dataset_name}")
            continue

        copy_tree_contents(src_dir, dst_dir)
        print(f"[OK] Installed {dataset_name} -> {dst_dir}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    raw_root = repo_relative_path(RAW_DATASETS_DIR)
    archive_dir = repo_relative_path(ARCHIVE_DIR)
    archive_path = repo_relative_path(ARCHIVE_PATH)
    extract_dir = repo_relative_path(EXTRACT_DIR)

    raw_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"RAW_ROOT     : {raw_root.resolve()}")
    print(f"ARCHIVE_PATH : {archive_path.resolve()}")
    print(f"EXTRACT_DIR  : {extract_dir.resolve()}")
    print()

    if all_expected_raw_dirs_already_present(raw_root):
        print("[SKIP] All expected raw dataset directories are already populated.")
        print("[DONE] Nothing to download or unpack.")
        return

    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"[SKIP] Archive already exists: {archive_path}")
    else:
        file_id = extract_file_id(FILE_URL)
        print(f"FILE_URL : {FILE_URL}")
        print(f"FILE_ID  : {file_id}")
        print()
        download_google_drive_file(file_id, archive_path)

        if not archive_path.exists():
            raise RuntimeError("Download failed: archive file was not created.")

        if archive_path.stat().st_size == 0:
            raise RuntimeError("Download failed: archive file is empty.")

        print(f"[OK] Download completed: {archive_path}")

    extract_zip_archive(archive_path, extract_dir)
    sync_expected_raw_datasets(extract_dir, raw_root)

    print()
    print("[DONE] Raw dataset bootstrap completed.")


if __name__ == "__main__":
    main()