#!/usr/bin/env python3
"""
Check BibTeX metadata consistency between local titles and external DOI/URL/arXiv metadata.

Default behavior:
- Reads tesi.bib from the same directory as this script.
- Checks each entry with DOI, arXiv eprint, or URL.
- Compares the local BibTeX title against the external title.
- Writes CSV and Markdown reports.

No external Python dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BIB_FILE = SCRIPT_DIR / "tesi.bib"
DEFAULT_CACHE_FILE = SCRIPT_DIR / "bibliography_metadata_cache.json"
DEFAULT_CSV_REPORT = SCRIPT_DIR / "bibliography_metadata_consistency_report.csv"
DEFAULT_MD_REPORT = SCRIPT_DIR / "bibliography_metadata_consistency_summary.md"

DEFAULT_THRESHOLD = 0.72
REQUEST_DELAY_SECONDS = 0.25
TIMEOUT_SECONDS = 20

USER_AGENT = (
    "msc-thesis-bibliography-checker/1.0 "
    "(mailto:metadata-check@example.local)"
)


# ---------------------------------------------------------------------------
# Manually verified false positives
# ---------------------------------------------------------------------------
# These entries have been manually checked and are considered acceptable.
# They are excluded from the final "requires attention" list even if the
# automatic DOI/URL/title comparison reports a weak match, a generic webpage
# title, a blocked URL, a rate limit, or unavailable external metadata.

ACCEPTED_FALSE_POSITIVES = {
    "casey2011digital",
    "palmer2001road",
    "li2022blip",
    "reith2002examining",
    "farid2016photo",
    "cpp1988",
    "gdpr2016",
    "aiact2024",
    "directive2013cyber",
    "cassazione2020",
    "cassazione2025",
    "Ribeiro2016",
    "swgde_guidelines",
    "faqir2023digital",
    "iso27037",
    "enfsi2022",
    "nowroozi2024verifying",
    "magnet",
    "xways",
    "cellebrite",
    "acpo_guidelines",
    "iso27041",
    "iso27042",
    "iso27043",
    "swgde2014validation",
}

# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

def find_bib_entries(text: str) -> List[Dict[str, str]]:
    """
    Parse BibTeX entries using brace balancing.

    Returns a list of dictionaries with:
    - entry_type
    - key
    - raw
    - fields
    """
    entries: List[Dict[str, str]] = []
    i = 0
    n = len(text)

    while i < n:
        at = text.find("@", i)
        if at == -1:
            break

        line_start = text.rfind("\n", 0, at) + 1
        before_at = text[line_start:at].strip()
        if before_at.startswith("%"):
            i = at + 1
            continue

        brace = text.find("{", at)
        comma = text.find(",", at)

        if brace == -1 or comma == -1 or comma < brace:
            i = at + 1
            continue

        entry_type = text[at + 1:brace].strip().lower()
        if entry_type in {"comment", "preamble", "string"}:
            i = at + 1
            continue

        key = text[brace + 1:comma].strip()

        depth = 0
        end = None
        for j in range(brace, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        if end is None:
            raise ValueError(f"Unbalanced BibTeX entry starting at key={key}")

        raw = text[at:end]
        body = text[comma + 1:end - 1]
        fields = parse_bib_fields(body)

        entries.append(
            {
                "entry_type": entry_type,
                "key": key,
                "raw": raw,
                "fields": fields,
            }
        )

        i = end

    return entries


def parse_bib_fields(body: str) -> Dict[str, str]:
    """
    Extract fields from a BibTeX entry body.

    Handles common values enclosed in {...} or "...".
    """
    fields: Dict[str, str] = {}
    i = 0
    n = len(body)

    while i < n:
        # Skip spaces and commas.
        while i < n and body[i] in " \t\r\n,":
            i += 1

        if i >= n:
            break

        name_start = i
        while i < n and re.match(r"[A-Za-z0-9_\-]", body[i]):
            i += 1

        field_name = body[name_start:i].strip().lower()
        if not field_name:
            i += 1
            continue

        while i < n and body[i].isspace():
            i += 1

        if i >= n or body[i] != "=":
            continue

        i += 1

        while i < n and body[i].isspace():
            i += 1

        if i >= n:
            break

        if body[i] == "{":
            value, i = read_braced_value(body, i)
        elif body[i] == '"':
            value, i = read_quoted_value(body, i)
        else:
            value_start = i
            while i < n and body[i] != ",":
                i += 1
            value = body[value_start:i].strip()

        fields[field_name] = cleanup_latex_text(value)

    return fields


def read_braced_value(text: str, start: int) -> Tuple[str, int]:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


def read_quoted_value(text: str, start: int) -> Tuple[str, int]:
    escaped = False
    for i in range(start + 1, len(text)):
        ch = text[i]
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            return text[start + 1:i], i + 1
        escaped = False
    return text[start + 1:], len(text)


# ---------------------------------------------------------------------------
# Text normalisation and matching
# ---------------------------------------------------------------------------

LATEX_ACCENTS = {
    r"\'a": "a", r"\'e": "e", r"\'i": "i", r"\'o": "o", r"\'u": "u",
    r"\`a": "a", r"\`e": "e", r"\`i": "i", r"\`o": "o", r"\`u": "u",
    r"\"a": "a", r"\"e": "e", r"\"i": "i", r"\"o": "o", r"\"u": "u",
    r"\~n": "n",
}


def cleanup_latex_text(value: str) -> str:
    value = value.strip()

    # Remove common BibTeX escaping.
    value = value.replace("\\&", "&")
    value = value.replace("\\_", "_")
    value = value.replace("\\%", "%")
    value = value.replace("\\#", "#")

    for latex, plain in LATEX_ACCENTS.items():
        value = value.replace(latex, plain)
        value = value.replace("{" + latex + "}", plain)

    # Remove braces used to preserve capitalisation.
    value = value.replace("{", "").replace("}", "")

    # Remove simple LaTeX commands while keeping their argument content when possible.
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)

    return " ".join(value.split())


def normalize_title(title: str) -> str:
    title = html.unescape(title or "")
    title = cleanup_latex_text(title)
    title = title.lower()

    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        ":": " ",
        ";": " ",
        ",": " ",
        ".": " ",
        "-": " ",
        "_": " ",
        "/": " ",
        "\\": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    # Remove very common stop words only if useful.
    return title


def similarity(a: str, b: str) -> float:
    na = normalize_title(a)
    nb = normalize_title(b)

    if not na or not nb:
        return 0.0

    return SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# External metadata fetching
# ---------------------------------------------------------------------------

def http_get(url: str, accept: Optional[str] = None, timeout: int = TIMEOUT_SECONDS) -> Tuple[int, str]:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            data = response.read().decode(charset, errors="replace")
            return response.status, data
    except urllib.error.HTTPError as e:
        try:
            data = e.read().decode("utf-8", errors="replace")
        except Exception:
            data = ""
        return e.code, data
    except Exception as e:
        return 0, str(e)


def fetch_doi_title(doi: str, cache: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], str, str]:
    doi = clean_doi(doi)
    cache_key = f"doi:{doi.lower()}"

    if cache_key in cache:
        item = cache[cache_key]
        return item.get("title"), item.get("source", "cache"), item.get("error", "")

    encoded_doi = urllib.parse.quote(doi, safe="/")
    url = f"https://doi.org/{encoded_doi}"

    status, data = http_get(
        url,
        accept="application/vnd.citationstyles.csl+json"
    )

    title = None
    error = ""

    if status == 200:
        try:
            obj = json.loads(data)
            raw_title = obj.get("title")
            if isinstance(raw_title, str):
                title = cleanup_latex_text(raw_title)
            elif isinstance(raw_title, list) and raw_title:
                title = cleanup_latex_text(str(raw_title[0]))
        except Exception as e:
            error = f"DOI JSON parse error: {e}"
    else:
        error = f"DOI metadata request failed with status={status}"

    cache[cache_key] = {
        "title": title or "",
        "source": "doi.org",
        "error": error,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    return title, "doi.org", error


def fetch_arxiv_title(arxiv_id: str, cache: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], str, str]:
    arxiv_id = clean_arxiv_id(arxiv_id)
    cache_key = f"arxiv:{arxiv_id.lower()}"

    if cache_key in cache:
        item = cache[cache_key]
        return item.get("title"), item.get("source", "cache"), item.get("error", "")

    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"

    status, data = http_get(url)

    title = None
    error = ""

    if status == 200:
        try:
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            title_node = root.find(".//atom:entry/atom:title", ns)
            if title_node is not None and title_node.text:
                title = cleanup_latex_text(title_node.text)
            else:
                error = "No arXiv title found"
        except Exception as e:
            error = f"arXiv XML parse error: {e}"
    else:
        error = f"arXiv request failed with status={status}"

    cache[cache_key] = {
        "title": title or "",
        "source": "arXiv",
        "error": error,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    return title, "arXiv", error


def fetch_url_title(url: str, cache: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], str, str]:
    url = url.strip()
    cache_key = f"url:{url}"

    if cache_key in cache:
        item = cache[cache_key]
        return item.get("title"), item.get("source", "cache"), item.get("error", "")

    arxiv_id = extract_arxiv_id_from_url(url)
    if arxiv_id:
        return fetch_arxiv_title(arxiv_id, cache)

    doi = extract_doi_from_url(url)
    if doi:
        return fetch_doi_title(doi, cache)

    status, data = http_get(url, accept="text/html")

    title = None
    error = ""

    if status == 200:
        title = extract_html_title(data)
        if not title:
            error = "No HTML title found"
    else:
        error = f"URL request failed with status={status}"

    cache[cache_key] = {
        "title": title or "",
        "source": "url-html-title",
        "error": error,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    return title, "url-html-title", error


def extract_html_title(html_text: str) -> Optional[str]:
    patterns = [
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        r'<meta\s+name=["\']citation_title["\']\s+content=["\']([^"\']+)["\']',
        r"<title[^>]*>(.*?)</title>",
    ]

    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()
            title = html.unescape(title)
            return cleanup_latex_text(title)

    return None


# ---------------------------------------------------------------------------
# DOI/arXiv extraction and cleanup
# ---------------------------------------------------------------------------

def clean_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.replace("\\", "")
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip().strip(".")
    return doi


def clean_arxiv_id(value: str) -> str:
    value = value.strip()
    value = value.replace("arXiv:", "")
    value = value.replace("arxiv:", "")
    value = value.replace("https://arxiv.org/abs/", "")
    value = value.replace("http://arxiv.org/abs/", "")
    value = value.replace("https://arxiv.org/pdf/", "")
    value = value.replace("http://arxiv.org/pdf/", "")
    value = value.replace(".pdf", "")
    return value.strip()


def extract_doi_from_url(url: str) -> Optional[str]:
    patterns = [
        r"https?://(?:dx\.)?doi\.org/(10\.\S+)",
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return clean_doi(match.group(1))

    return None


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    patterns = [
        r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv\.org/abs/([a-z\-]+/[0-9]{7}(?:v\d+)?)",
        r"arxiv\.org/pdf/([a-z\-]+/[0-9]{7}(?:v\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url, flags=re.IGNORECASE)
        if match:
            return clean_arxiv_id(match.group(1))

    return None


def detect_arxiv_id(fields: Dict[str, str]) -> Optional[str]:
    eprint = fields.get("eprint", "").strip()
    archive_prefix = fields.get("archiveprefix", "").strip().lower()

    if eprint and ("arxiv" in archive_prefix or re.match(r"^\d{4}\.\d{4,5}", eprint)):
        return clean_arxiv_id(eprint)

    url = fields.get("url", "")
    arxiv_from_url = extract_arxiv_id_from_url(url)
    if arxiv_from_url:
        return arxiv_from_url

    doi = fields.get("doi", "")
    if doi.lower().startswith("10.48550/arxiv."):
        return clean_arxiv_id(doi.split("arXiv.", 1)[-1] if "arXiv." in doi else doi.split("arxiv.", 1)[-1])

    return None


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def audit_entry(entry: Dict[str, str], cache: Dict[str, Dict[str, str]], threshold: float) -> Dict[str, str]:
    fields = entry["fields"]

    key = entry["key"]
    entry_type = entry["entry_type"]
    local_title = fields.get("title", "").strip()
    doi = fields.get("doi", "").strip()
    url = fields.get("url", "").strip()

    external_title = None
    source = ""
    error = ""
    identifier = ""

    arxiv_id = detect_arxiv_id(fields)

    if doi:
        identifier = clean_doi(doi)
        external_title, source, error = fetch_doi_title(identifier, cache)

        # If arXiv DOI metadata fails, try arXiv API.
        if not external_title and arxiv_id:
            external_title, source, error = fetch_arxiv_title(arxiv_id, cache)
            identifier = arxiv_id

    elif arxiv_id:
        identifier = arxiv_id
        external_title, source, error = fetch_arxiv_title(arxiv_id, cache)

    elif url:
        identifier = url
        external_title, source, error = fetch_url_title(url, cache)

    else:
        return {
            "bibkey": key,
            "entry_type": entry_type,
            "status": "SKIPPED_NO_IDENTIFIER",
            "score": "",
            "identifier": "",
            "source": "",
            "local_title": local_title,
            "external_title": "",
            "message": "No DOI, arXiv eprint, or URL available",
        }

    if not local_title:
        return {
            "bibkey": key,
            "entry_type": entry_type,
            "status": "NEEDS_MANUAL_CHECK",
            "score": "",
            "identifier": identifier,
            "source": source,
            "local_title": "",
            "external_title": external_title or "",
            "message": "Local BibTeX title is missing",
        }

    if not external_title:
        return {
            "bibkey": key,
            "entry_type": entry_type,
            "status": "EXTERNAL_METADATA_NOT_FOUND",
            "score": "",
            "identifier": identifier,
            "source": source,
            "local_title": local_title,
            "external_title": "",
            "message": error or "External title not found",
        }

    score = similarity(local_title, external_title)

    if score >= 0.92:
        status = "OK_STRONG_MATCH"
        message = "Local and external titles are strongly consistent"
    elif score >= threshold:
        status = "OK_WEAK_MATCH"
        message = "Titles are similar but should be visually checked"
    else:
        status = "POSSIBLE_INCONGRUENCE"
        message = "Local title differs substantially from external metadata"

    return {
        "bibkey": key,
        "entry_type": entry_type,
        "status": status,
        "score": f"{score:.3f}",
        "identifier": identifier,
        "source": source,
        "local_title": local_title,
        "external_title": external_title,
        "message": message,
    }


def write_csv_report(rows: List[Dict[str, str]], path: Path) -> None:
    fieldnames = [
        "bibkey",
        "entry_type",
        "status",
        "original_status",
        "score",
        "identifier",
        "source",
        "local_title",
        "external_title",
        "message",
    ]

    for row in rows:
        row.setdefault("original_status", "")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_md_summary(rows: List[Dict[str, str]], path: Path) -> None:
    total = len(rows)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    problematic_statuses = {
        "POSSIBLE_INCONGRUENCE",
        "EXTERNAL_METADATA_NOT_FOUND",
        "NEEDS_MANUAL_CHECK",
    }

    problematic = [row for row in rows if row["status"] in problematic_statuses]

    lines = []
    lines.append("# Bibliography metadata consistency summary")
    lines.append("")
    lines.append(f"- Entries checked: **{total}**")
    lines.append("")

    lines.append("## Status counts")
    lines.append("")
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: **{count}**")
    lines.append("")

    lines.append("## Entries requiring attention")
    lines.append("")

    if not problematic:
        lines.append("No problematic entries detected.")
    else:
        lines.append("| Bibkey | Status | Score | Identifier | Local title | External title |")
        lines.append("|---|---:|---:|---|---|---|")
        for row in problematic:
            lines.append(
                "| "
                f"`{row['bibkey']}` | "
                f"{row['status']} | "
                f"{row['score']} | "
                f"{escape_md(row['identifier'])} | "
                f"{escape_md(row['local_title'])} | "
                f"{escape_md(row['external_title'])} |"
            )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_md(text: str) -> str:
    text = text or ""
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check DOI/URL/arXiv metadata consistency against BibTeX titles."
    )
    parser.add_argument(
        "--bib",
        type=Path,
        default=DEFAULT_BIB_FILE,
        help="Path to BibTeX file. Default: tesi.bib in the script directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold for possible incongruence. Default: {DEFAULT_THRESHOLD}.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help="Path to metadata cache JSON.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_REPORT,
        help="Path to output CSV report.",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=DEFAULT_MD_REPORT,
        help="Path to output Markdown summary.",
    )
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Print only problematic entries to terminal.",
    )

    args = parser.parse_args()

    bib_file = args.bib.resolve()

    if not bib_file.exists():
        raise FileNotFoundError(f"BibTeX file not found: {bib_file}")

    print(f"BibTeX file: {bib_file}")

    text = bib_file.read_text(encoding="utf-8", errors="replace")
    entries = find_bib_entries(text)

    cache = load_cache(args.cache)

    rows: List[Dict[str, str]] = []

    for index, entry in enumerate(entries, start=1):
        key = entry["key"]
        print(f"[{index:03d}/{len(entries):03d}] Checking {key}...")
        row = audit_entry(entry, cache, args.threshold)

        if row["bibkey"] in ACCEPTED_FALSE_POSITIVES and row["status"] in {
            "POSSIBLE_INCONGRUENCE",
            "EXTERNAL_METADATA_NOT_FOUND",
            "NEEDS_MANUAL_CHECK",
        }:
            row["original_status"] = row["status"]
            row["status"] = "ACCEPTED_FALSE_POSITIVE"
            row["message"] = (
                "Automatically flagged, but manually verified as acceptable "
                "for this thesis bibliography."
            )

        rows.append(row)

    save_cache(args.cache, cache)
    write_csv_report(rows, args.csv)
    write_md_summary(rows, args.md)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print()
    print("Bibliography metadata consistency audit completed.")
    print(f"Entries audited: {len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print()
    print(f"CSV report: {args.csv}")
    print(f"Markdown summary: {args.md}")
    print(f"Cache file: {args.cache}")

    problematic_statuses = {
        "POSSIBLE_INCONGRUENCE",
        "EXTERNAL_METADATA_NOT_FOUND",
        "NEEDS_MANUAL_CHECK",
    }

    problematic = [row for row in rows if row["status"] in problematic_statuses]

    if problematic:
        print()
        print("Entries requiring attention:")
        for row in problematic:
            print(
                f"- {row['bibkey']} | {row['status']} | "
                f"score={row['score']} | identifier={row['identifier']}"
            )
            if not args.only_problems:
                print(f"  Local:    {row['local_title']}")
                print(f"  External: {row['external_title']}")
                print(f"  Message:  {row['message']}")
    else:
        print()
        print("No problematic entries detected.")


if __name__ == "__main__":
    main()