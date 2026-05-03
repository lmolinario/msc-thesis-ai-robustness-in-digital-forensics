#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 02_download_github.py
===============================================================================

Title:
    Automated GitHub Repository Downloader and Metadata Indexer for FAIR-Lab

Purpose:
    This scripts downloads a public GitHub repository as a ZIP archive, extracts
    its contents into the local raw datasets directory, and generates a structured
    CSV summary of all indexed files.

Research Context:
    In the FAIR-Lab experimental workflow, this module supports controlled and
    reproducible acquisition of raw sources hosted on public GitHub repositories.
    The scripts improves:
        - provenance tracking
        - repeatability of source acquisition
        - auditability of imported resources
        - consistency across repeated executions

Inputs:
    - GitHub repository identifier in owner/repo format
    - optional branch name
    - local output directory
    - local summary CSV path

Outputs:
    - extracted repository content stored in the raw datasets area
    - CSV summary file containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - the repository is public and reachable over HTTPS
    - the selected branch is available as a ZIP archive
    - the local filesystem is writable
    - repository contents may include both image and non-image files

Dependencies:
    - Python standard library
    - requests
    - project utility: utils.paths

Methodological Relevance:
    In forensic AI and adversarial robustness workflows, controlled source
    acquisition is essential to reduce ambiguity regarding data origin and to
    document exactly which external resources were imported into the experiment.
    The SHA256 digest included in the summary supports file-level integrity
    verification and traceability.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

import requests

from datasets.scripts.utils.paths import RAW_DATASETS_DIR, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_USER_REPO = "jdhao/deep_firearm"
DEFAULT_OUTPUT_DIR = RAW_DATASETS_DIR / "02_deepfirearm"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "02_deepfirearm" / "download_summary.csv"

DEFAULT_BRANCH_CANDIDATES = ("main", "master")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    This enables configurable and reproducible execution of the scripts within
    larger dataset preparation pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Download a public GitHub repository as ZIP and regenerate a summary CSV."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_USER_REPO,
        help=f"GitHub repository in owner/repo format (default: {DEFAULT_USER_REPO})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Specific branch to download. If omitted, the scripts tries main first, then master.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded ZIP archive after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/rebuild even if the target content already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds (default: 60).",
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
    Return True if the given file path corresponds to a supported image file.
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


def has_any_content(directory: Path) -> bool:
    """
    Check whether the target directory already contains any files or subfolders.
    """
    return directory.exists() and any(directory.iterdir())


def sanitize_repo_name(user_repo: str) -> str:
    """
    Convert a GitHub repository identifier into a filesystem-safe directory name.

    Example:
        'owner/repo' -> 'owner_repo'
    """
    return user_repo.replace("/", "_")


def build_github_zip_url(user_repo: str, branch: str) -> str:
    """
    Construct the GitHub ZIP download URL for the selected branch.
    """
    return f"https://github.com/{user_repo}/archive/refs/heads/{branch}.zip"


def find_existing_zip_files(output_dir: Path) -> list[Path]:
    """
    Locate ZIP files already present in the output directory.
    """
    return sorted([p for p in output_dir.glob("*.zip") if p.is_file()])


def find_extracted_root(output_dir: Path, repo_name: str, branch: str) -> Path | None:
    """
    Locate the root directory produced by ZIP extraction.

    GitHub archives usually extract into a directory such as:
        repo-branch
    """
    candidates = [
        output_dir / f"{repo_name}-{branch}",
        output_dir / f"{repo_name.replace('_', '-')}-{branch}",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    matches = sorted(
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith(f"{repo_name}-")
    )
    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# Download / Extraction Logic
# -----------------------------------------------------------------------------
def download_zip(url: str, zip_path: Path, timeout: int) -> None:
    """
    Download a ZIP archive from the given URL using streamed HTTP transfer.
    """
    logging.info("Downloading archive from: %s", url)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """
    Extract the ZIP archive into the specified output directory.
    """
    logging.info("Extracting ZIP archive: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def move_extracted_content(src_dir: Path, target_dir: Path, force: bool = False) -> None:
    """
    Move extracted repository content into the final target directory.
    """
    if target_dir.exists():
        if force:
            logging.warning("Removing existing target directory: %s", target_dir)
            shutil.rmtree(target_dir)
        else:
            logging.info("Target directory already exists: %s", target_dir)
            return

    logging.info("Moving extracted content: %s -> %s", src_dir, target_dir)
    src_dir.rename(target_dir)


def download_github_repo(
    user_repo: str,
    output_dir: Path,
    branch: str | None = None,
    keep_zip: bool = False,
    force: bool = False,
    timeout: int = 60,
) -> None:
    """
    Download and extract a public GitHub repository.

    Behavior:
        - reuses already prepared content if present and force=False
        - reuses existing ZIP if available and force=False
        - tries common default branches if branch is not specified
        - raises a RuntimeError if all download attempts fail
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_leaf_name = user_repo.split("/")[-1]
    target_dir = output_dir / sanitize_repo_name(user_repo)

    if has_any_content(target_dir) and not force:
        logging.info("Repository already present in %s. Download skipped.", target_dir)
        return

    existing_zips = find_existing_zip_files(output_dir)
    if existing_zips and not force:
        zip_path = existing_zips[0]
        logging.info("Existing ZIP archive found: %s", zip_path)
        extract_zip(zip_path, output_dir)

        extracted_root = find_extracted_root(output_dir, repo_leaf_name, branch or "main")
        if extracted_root is not None:
            move_extracted_content(extracted_root, target_dir, force=False)

        if not keep_zip and zip_path.exists():
            zip_path.unlink(missing_ok=True)
            logging.info("Removed ZIP archive: %s", zip_path)

        return

    branches_to_try = [branch] if branch else list(DEFAULT_BRANCH_CANDIDATES)
    last_error: Exception | None = None

    for current_branch in branches_to_try:
        assert current_branch is not None

        url = build_github_zip_url(user_repo, current_branch)
        zip_path = output_dir / f"{sanitize_repo_name(user_repo)}_{current_branch}.zip"

        try:
            download_zip(url, zip_path, timeout=timeout)
            extract_zip(zip_path, output_dir)

            extracted_root = find_extracted_root(output_dir, repo_leaf_name, current_branch)
            if extracted_root is None:
                raise FileNotFoundError(
                    f"Could not locate extracted directory for repo={user_repo}, branch={current_branch}"
                )

            move_extracted_content(extracted_root, target_dir, force=force)

            if not keep_zip and zip_path.exists():
                zip_path.unlink(missing_ok=True)
                logging.info("Removed ZIP archive: %s", zip_path)

            logging.info("Repository successfully prepared at: %s", target_dir)
            return

        except Exception as exc:
            last_error = exc
            logging.warning("Attempt failed for branch '%s': %s", current_branch, exc)

            if zip_path.exists() and not keep_zip:
                zip_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Repository download failed for '{user_repo}' on all attempted branches {branches_to_try}. "
        f"Last error: {last_error}"
    )


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured summary of all files contained in the output directory.

    This summary supports:
        - provenance tracking
        - dataset auditing
        - quick inspection
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
    Write the file summary to CSV format.
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

    Pipeline steps:
        1. Parse arguments
        2. Configure logging
        3. Download and extract repository
        4. Build metadata summary
        5. Save CSV index
    """
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = repo_relative_path(args.output_dir)
    summary_csv = repo_relative_path(args.summary_csv)

    download_github_repo(
        user_repo=args.repo,
        output_dir=output_dir,
        branch=args.branch,
        keep_zip=args.keep_zip,
        force=args.force,
        timeout=args.timeout,
    )

    summary_rows = build_summary_rows(output_dir)
    write_summary_csv(summary_rows, summary_csv)

    logging.info("Summary CSV regenerated: %s", summary_csv)
    logging.info("Total indexed files: %d", len(summary_rows))
    logging.info(
        "Total indexed images: %d",
        sum(1 for row in summary_rows if row["is_image"] == 1),
    )


if __name__ == "__main__":
    main()