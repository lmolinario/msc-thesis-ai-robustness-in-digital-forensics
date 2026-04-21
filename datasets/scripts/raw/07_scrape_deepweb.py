#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Script: 07_scrape_deepweb.py
===============================================================================

Title:
    Deep Web Image Collection Pipeline via Ahmia and Tor

Purpose:
    This scripts collects candidate image files from Ahmia-indexed .onion pages,
    routes requests through the Tor network, validates downloaded images, applies
    basic content-based deduplication, and generates a structured CSV summary of
    the collected files.

Research Context:
    In the FAIR-Lab experimental workflow, this module supports the acquisition
    of visually heterogeneous samples from unconventional online environments.
    Compared with standard public datasets or mainstream web sources, indexed
    .onion pages may expose the collection pipeline to noisier, weaker, and less
    curated visual content.

Methodological Relevance:
    For forensic AI and robustness evaluation, dataset diversity is important.
    However, sources obtained through query-based crawling on indexed .onion
    pages should be regarded as candidate data sources rather than semantically
    verified ground truth. The purpose of this scripts is therefore limited to:

        - controlled source acquisition
        - basic image validation
        - lightweight deduplication
        - reproducible file-level indexing

    It does not perform semantic annotation and does not guarantee that collected
    images are relevant, correctly labeled, or suitable for direct model training
    without subsequent manual review.

Inputs:
    - output directory
    - summary CSV path
    - query list
    - maximum number of links per query
    - maximum number of images per page
    - image size thresholds
    - Tor proxy configuration
    - optional force rebuild flag

Outputs:
    - query-structured image collection
    - CSV summary containing:
        * relative_path
        * filename
        * extension
        * size_bytes
        * is_image
        * sha256

Assumptions:
    - Ahmia is reachable from the surface web
    - Tor proxy is locally available and correctly configured
    - indexed onion links are accessible and return HTML pages
    - downloaded files are candidate samples subject to later inspection

Dependencies:
    - Python standard library
    - requests
    - beautifulsoup4
    - Pillow
    - project utility: utils.paths

Important Note:
    MD5 is used here only as a lightweight content fingerprint for duplicate
    detection and local file naming during the acquisition phase. It is not used
    as a cryptographic integrity mechanism. For file-level integrity tracking in
    the final CSV summary, SHA256 digests are computed separately.

===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

from datasets.scripts.utils.paths  import RAW_DATASETS_DIR, repo_relative_path


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = RAW_DATASETS_DIR / "05_deepweb" / "deepweb_dataset"
DEFAULT_SUMMARY_CSV = RAW_DATASETS_DIR / "05_deepweb" / "deepweb_scraping_summary.csv"

DEFAULT_SEARCH_QUERIES = [
    "gun marketplace",
    "firearm photo",
    "pistol images",
    "smuggled weapons",
    "AK-47 picture",
    "gun trading",
    "assault rifle image",
    "sniper darkweb",
    "hidden firearm photo",
    "weapon",
    "gun",
    "pistol",
    "AK-47",
    "rifle",
    "sniper",
    "firearm",
]

BAN_KEYWORDS = ["logo", "banner", "avatar", "icon", "sprite", "captcha"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    This enables controlled and reproducible execution of the collection pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Collect images from Ahmia-indexed .onion pages and regenerate the summary CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--max-links-per-query",
        type=int,
        default=20,
        help="Maximum number of onion links to visit per query.",
    )
    parser.add_argument(
        "--max-images-per-page",
        type=int,
        default=15,
        help="Maximum number of images to download per onion page.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=200,
        help="Minimum image width.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=200,
        help="Minimum image height.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Pause between visits to onion URLs.",
    )
    parser.add_argument(
        "--tor-host",
        type=str,
        default="127.0.0.1",
        help="Tor SOCKS proxy host.",
    )
    parser.add_argument(
        "--tor-port",
        type=int,
        default=9050,
        help="Tor SOCKS proxy port.",
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
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def has_existing_dataset(output_dir: Path) -> bool:
    """
    Check whether the output directory already contains files.
    """
    return output_dir.exists() and any(output_dir.rglob("*"))


def prepare_output_dir(output_dir: Path, force: bool) -> bool:
    """
    Prepare the output directory before collection.

    Returns:
        True  -> collection should be executed
        False -> existing dataset should be reused
    """
    if output_dir.exists():
        if force:
            logging.warning("Removing existing output directory: %s", output_dir)
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return True

        if has_existing_dataset(output_dir):
            logging.info("Deep web dataset already present in %s. Collection skipped.", output_dir)
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    return True


def create_tor_session(tor_host: str, tor_port: int) -> requests.Session:
    """
    Create a requests session routed through the Tor SOCKS proxy.
    """
    session = requests.Session()
    proxy_url = f"socks5h://{tor_host}:{tor_port}"
    session.proxies.update(
        {
            "http": proxy_url,
            "https": proxy_url,
        }
    )
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (FAIR-Lab DeepWeb Collector)"
        }
    )
    return session


def sanitize_query_to_dirname(query: str) -> str:
    """
    Convert a search query into a filesystem-safe directory name.
    """
    return query.replace(" ", "_").replace("/", "_")


def normalize_content_type(content_type: str) -> str:
    """
    Normalize an HTTP Content-Type header by removing optional parameters.
    """
    return content_type.split(";")[0].strip().lower()


def content_type_to_extension(content_type: str) -> str:
    """
    Map a normalized image content type to a file extension.
    """
    normalized = normalize_content_type(content_type)
    subtype = normalized.split("/")[-1] if "/" in normalized else normalized

    if subtype in {"jpeg", "jpg"}:
        return ".jpg"
    if subtype == "png":
        return ".png"
    if subtype == "webp":
        return ".webp"
    if subtype == "gif":
        return ".gif"
    if subtype == "bmp":
        return ".bmp"
    if subtype == "tiff":
        return ".tiff"

    return ".img"


def compute_md5(content: bytes) -> str:
    """
    Compute an MD5 digest of file content.

    Used only for lightweight duplicate detection and local file naming.
    """
    return hashlib.md5(content).hexdigest()


def compute_sha256_file(file_path: Path) -> str:
    """
    Compute the SHA256 digest of a file.

    This supports file-level integrity verification and provenance tracking in
    the final summary CSV.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_image_file(path: Path) -> bool:
    """
    Return True if the file has a supported image extension.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def validate_image_file(file_path: Path, min_width: int, min_height: int) -> tuple[bool, int, int]:
    """
    Validate a saved image file and check its minimum dimensions.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return False, width, height
            return True, width, height
    except (UnidentifiedImageError, OSError, ValueError):
        return False, 0, 0


def validate_image_content(content: bytes, min_width: int, min_height: int) -> tuple[bool, int, int]:
    """
    Validate image bytes in memory before writing them to disk.
    """
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return False, width, height
            return True, width, height
    except (UnidentifiedImageError, OSError, ValueError):
        return False, 0, 0


# -----------------------------------------------------------------------------
# Ahmia Search
# -----------------------------------------------------------------------------
def search_ahmia(query: str, max_links: int) -> list[str]:
    """
    Query Ahmia and extract candidate .onion redirect targets.

    Returns:
        A list of onion URLs up to the requested maximum.
    """
    search_url = f"https://ahmia.fi/search/?q={quote_plus(query)}"
    links: list[str] = []

    try:
        res = requests.get(search_url, timeout=20)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/redirect?search_term=" not in href:
                continue

            redirect_url = unquote(href.split("redirect_url=")[-1])
            if ".onion" not in redirect_url:
                continue

            if redirect_url not in links:
                links.append(redirect_url)

            if len(links) >= max_links:
                break

        logging.info("Ahmia query '%s' -> %d onion links", query, len(links))
        return links

    except Exception as exc:
        logging.warning("Ahmia error for query '%s': %s", query, exc)
        return []


def iter_candidate_image_urls(page_url: str, html: str):
    """
    Yield candidate image URLs extracted from the HTML of a visited page.

    A simple keyword-based filter is applied to exclude obviously irrelevant
    assets such as icons, banners, and captchas.
    """
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        if any(keyword in src.lower() for keyword in BAN_KEYWORDS):
            continue
        yield urljoin(page_url, src)


# -----------------------------------------------------------------------------
# Download Logic
# -----------------------------------------------------------------------------
def download_images_from_onion(
    url: str,
    session: requests.Session,
    output_subdir: Path,
    max_images: int,
    min_width: int,
    min_height: int,
) -> int:
    """
    Download candidate images from a single onion page.

    The function:
        - requests the HTML page through Tor
        - extracts candidate image URLs
        - downloads image content
        - validates content type and image dimensions
        - deduplicates files using MD5-based local fingerprinting

    Returns:
        Number of newly saved valid images.
    """
    saved_count = 0

    try:
        res = session.get(url, timeout=20)
        res.raise_for_status()

        count = 0
        for img_url in iter_candidate_image_urls(url, res.text):
            if count >= max_images:
                break

            try:
                r = session.get(img_url, timeout=15)
                r.raise_for_status()

                content_type = normalize_content_type(r.headers.get("Content-Type", ""))
                if not content_type.startswith("image/"):
                    continue

                content = r.content
                md5_hash = compute_md5(content)
                extension = content_type_to_extension(content_type)
                file_path = output_subdir / f"{md5_hash}{extension}"

                is_valid, width, height = validate_image_content(
                    content=content,
                    min_width=min_width,
                    min_height=min_height,
                )

                if not is_valid:
                    continue

                if not file_path.exists():
                    with file_path.open("wb") as f:
                        f.write(content)
                    saved_count += 1

                count += 1

            except Exception as exc:
                logging.debug("Image download error for %s: %s", img_url, exc)
                continue

    except Exception as exc:
        logging.warning("Onion page access error for '%s': %s", url, exc)

    return saved_count


def collect_deepweb_images(
    output_dir: Path,
    max_links_per_query: int,
    max_images_per_page: int,
    min_width: int,
    min_height: int,
    sleep_seconds: float,
    tor_host: str,
    tor_port: int,
) -> None:
    """
    Execute the full indexed-onion image collection workflow.

    Workflow:
        1. create a Tor-routed session
        2. query Ahmia for each predefined query
        3. visit candidate onion pages
        4. download valid images
        5. avoid revisiting duplicate onion paths

    Methodological note:
        Query directories act as source-bucket organization only. They do not
        guarantee semantic correctness of the contained images.
    """
    session = create_tor_session(tor_host=tor_host, tor_port=tor_port)
    visited_paths: set[str] = set()
    total_saved = 0

    for query in DEFAULT_SEARCH_QUERIES:
        logging.info("Processing query: %s", query)

        sub_dir = output_dir / sanitize_query_to_dirname(query)
        sub_dir.mkdir(parents=True, exist_ok=True)

        onion_links = search_ahmia(query, max_links=max_links_per_query)

        for onion_url in onion_links:
            parsed = urlparse(onion_url)
            url_path_key = f"{parsed.netloc}{parsed.path}"

            if url_path_key in visited_paths:
                logging.debug("Skipping duplicate onion path: %s", onion_url)
                continue

            visited_paths.add(url_path_key)
            logging.info("Visiting: %s", onion_url)

            n_saved = download_images_from_onion(
                url=onion_url,
                session=session,
                output_subdir=sub_dir,
                max_images=max_images_per_page,
                min_width=min_width,
                min_height=min_height,
            )

            total_saved += n_saved
            logging.info("New files saved from %s: %d", onion_url, n_saved)
            time.sleep(sleep_seconds)

    logging.info("Indexed .onion image collection completed.")
    logging.info("Total new files saved: %d", total_saved)


# -----------------------------------------------------------------------------
# Metadata Summary
# -----------------------------------------------------------------------------
def build_summary_rows(output_dir: Path) -> list[dict[str, str | int]]:
    """
    Build a structured file-level summary of the collected dataset.

    This summary supports:
        - auditing
        - reproducibility
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
                "sha256": compute_sha256_file(file_path),
            }
        )

    return rows


def write_summary_csv(rows: list[dict[str, str | int]], summary_csv: Path) -> None:
    """
    Write the dataset summary to CSV format.
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
        3. Prepare output directory
        4. Run indexed-onion image collection if needed
        5. Regenerate summary CSV
    """
    args = parse_args()
    setup_logging(args.verbose)

    output_dir = repo_relative_path(args.output_dir)
    summary_csv = repo_relative_path(args.summary_csv)

    should_collect = prepare_output_dir(output_dir, force=args.force)

    if should_collect:
        collect_deepweb_images(
            output_dir=output_dir,
            max_links_per_query=args.max_links_per_query,
            max_images_per_page=args.max_images_per_page,
            min_width=args.min_width,
            min_height=args.min_height,
            sleep_seconds=args.sleep_seconds,
            tor_host=args.tor_host,
            tor_port=args.tor_port,
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