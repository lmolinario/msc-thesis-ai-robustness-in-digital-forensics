#!/usr/bin/env python3
"""
Audit BibTeX bibliography entries for a thesis.

Features:
- Parses a .bib file without external dependencies.
- Detects duplicate keys, duplicate DOI, duplicate normalized titles.
- Checks required fields by BibTeX type.
- Flags weak URLs, missing identifiers, malformed DOI/ISBN.
- Optional online mode verifies DOI metadata through Crossref/DataCite and arXiv IDs.

Usage:
    # Terminal usage with explicit input
    python audit_bibliography.py tesi.bib --out bibliography_audit.csv
    python audit_bibliography.py tesi.bib --out bibliography_audit.csv --online

    # PyCharm usage without parameters:
    # put tesi.bib in the same directory as this script and press Run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def get_script_dir() -> Path:
    """Return the directory containing this script.

    This makes the script reliable when launched from PyCharm, where the
    working directory may differ from the file location.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


SCRIPT_DIR = get_script_dir()
DEFAULT_BIBFILE = SCRIPT_DIR / "tesi.bib"
DEFAULT_OUTFILE = SCRIPT_DIR / "bibliography_audit.csv"


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ARXIV_RE = re.compile(r"(?:(?:arxiv:)?)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
ISBN_RE = re.compile(r"^(?:97[89])?\d{9}[\dXx]$")

WEAK_DOMAINS = [
    "researchgate.net",
    "academia.edu",
    "sistemapenale.it",
    "terzultimafermata.blog",
    "fynd.academy",
    "digital-detective.net",
    "blog",
]

REQUIRED_BY_TYPE = {
    "article": ["title", "author", "year", "journal"],
    "inproceedings": ["title", "author", "year", "booktitle"],
    "book": ["title", "author", "year", "publisher"],
    "incollection": ["title", "author", "year", "booktitle", "publisher"],
    "phdthesis": ["title", "author", "year", "school"],
    "techreport": ["title", "author", "year", "institution"],
    "misc": ["title", "year"],
}


def parse_bib_entries(text: str) -> list[dict[str, Any]]:
    entries = []
    i = 0
    n = len(text)

    while i < n:
        at = text.find("@", i)
        if at == -1:
            break

        match = re.match(r"@(\w+)\s*[\{\(]\s*([^,\s]+)\s*,", text[at:], flags=re.S)
        if not match:
            i = at + 1
            continue

        entry_type = match.group(1).lower()
        key = match.group(2).strip()

        brace_positions = [p for p in [text.find("{", at), text.find("(", at)] if p != -1]
        if not brace_positions:
            i = at + 1
            continue

        brace_pos = min(brace_positions)
        opener = text[brace_pos]
        closer = "}" if opener == "{" else ")"

        depth = 0
        escape = False
        end = None
        for j in range(brace_pos, n):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        if end is None:
            end = n

        raw = text[at:end]
        entries.append({"type": entry_type, "key": key, "raw": raw})
        i = end

    return entries


def parse_fields(raw: str) -> dict[str, str]:
    comma = raw.find(",")
    body = raw[comma + 1 :].rstrip()
    if body.endswith("}"):
        body = body[:-1]

    fields: dict[str, str] = {}
    i = 0
    n = len(body)

    while i < n:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        if i >= n:
            break

        match = re.match(r"([A-Za-z][A-Za-z0-9_\-]*)\s*=", body[i:])
        if not match:
            i += 1
            continue

        name = match.group(1).lower()
        i += match.end()

        while i < n and body[i].isspace():
            i += 1

        if i >= n:
            fields[name] = ""
            break

        if body[i] == "{":
            i += 1
            depth = 0
            value_chars = []

            while i < n:
                ch = body[i]
                if ch == "\\" and i + 1 < n:
                    value_chars.append(ch)
                    i += 1
                    value_chars.append(body[i])
                    i += 1
                    continue

                if ch == "{":
                    depth += 1
                    value_chars.append(ch)
                    i += 1
                    continue

                if ch == "}":
                    if depth == 0:
                        i += 1
                        break
                    depth -= 1
                    value_chars.append(ch)
                    i += 1
                    continue

                value_chars.append(ch)
                i += 1

            value = "".join(value_chars).strip()

        elif body[i] == '"':
            i += 1
            value_chars = []
            escape = False

            while i < n:
                ch = body[i]
                if escape:
                    value_chars.append(ch)
                    escape = False
                    i += 1
                    continue
                if ch == "\\":
                    value_chars.append(ch)
                    escape = True
                    i += 1
                    continue
                if ch == '"':
                    i += 1
                    break
                value_chars.append(ch)
                i += 1

            value = "".join(value_chars).strip()

        else:
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            value = body[start:i].strip()

        fields[name] = value

    return fields


def clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def normalize_title(title: str) -> str:
    title = clean(title)
    title = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", title)
    title = title.replace("{", "").replace("}", "").replace("$", "")
    title = re.sub(r"[\u2010-\u2015]", "-", title)
    title = re.sub(r"\s+", " ", title.lower()).strip()
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    return title


def title_similarity(a: str, b: str) -> float:
    aa = set(normalize_title(a).split())
    bb = set(normalize_title(b).split())
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def http_json(url: str, timeout: int = 15) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "bibliography-audit/1.0 (mailto:example@example.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def verify_doi_online(doi: str, title: str) -> tuple[str, str, float]:
    doi = doi.strip().rstrip(".")
    if not doi:
        return "", "", 0.0

    quoted = urllib.parse.quote(doi, safe="")
    sources = [
        ("Crossref", f"https://api.crossref.org/works/{quoted}"),
        ("DataCite", f"https://api.datacite.org/dois/{quoted}"),
    ]

    for source_name, url in sources:
        data = http_json(url)
        if not data:
            continue

        found_title = ""
        if source_name == "Crossref":
            msg = data.get("message", {})
            titles = msg.get("title") or []
            found_title = titles[0] if titles else ""
        elif source_name == "DataCite":
            attrs = data.get("data", {}).get("attributes", {})
            titles = attrs.get("titles") or []
            if titles:
                found_title = titles[0].get("title", "")

        if found_title:
            return source_name, found_title, title_similarity(title, found_title)

    return "", "", 0.0


def verify_arxiv_online(arxiv_id: str, title: str) -> tuple[str, str, float]:
    if not arxiv_id:
        return "", "", 0.0

    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "bibliography-audit/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            xml = response.read().decode("utf-8", errors="replace")
        match = re.search(r"<title>(.*?)</title>", xml, flags=re.S)
        if not match:
            return "", "", 0.0
        found_title = re.sub(r"\s+", " ", match.group(1)).strip()
        if found_title.lower() == "arxiv query":
            # take second title if available
            titles = re.findall(r"<title>(.*?)</title>", xml, flags=re.S)
            if len(titles) > 1:
                found_title = re.sub(r"\s+", " ", titles[1]).strip()
        return "arXiv", found_title, title_similarity(title, found_title)
    except Exception:
        return "", "", 0.0


def audit(path: Path, online: bool = False, sleep: float = 0.2) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    entries = parse_bib_entries(text)

    for entry in entries:
        entry["fields"] = parse_fields(entry["raw"])

    keys = Counter(entry["key"] for entry in entries)
    dois = Counter(clean(entry["fields"].get("doi")).lower().rstrip(".") for entry in entries if entry["fields"].get("doi"))
    titles = Counter(normalize_title(entry["fields"].get("title", "")) for entry in entries if entry["fields"].get("title"))
    duplicate_dois = {k for k, v in dois.items() if v > 1}
    duplicate_titles = {k for k, v in titles.items() if v > 1 and len(k) > 20}

    rows = []

    for index, entry in enumerate(entries, start=1):
        fields = entry["fields"]
        key = entry["key"]
        entry_type = entry["type"]
        title = clean(fields.get("title"))
        year = clean(fields.get("year"))
        doi = clean(fields.get("doi"))
        eprint = clean(fields.get("eprint"))
        isbn = clean(fields.get("isbn")).replace("-", "").replace(" ", "")
        url = clean(fields.get("url")) or clean(fields.get("howpublished"))
        journal_or_booktitle = clean(fields.get("journal")) or clean(fields.get("booktitle"))
        publisher_or_institution = clean(fields.get("publisher")) or clean(fields.get("institution")) or clean(fields.get("school"))
        normalized_title = normalize_title(title)
        doi_valid = bool(doi and DOI_RE.fullmatch(doi.rstrip(".")))
        arxiv_match = (
            ARXIV_RE.search(eprint)
            or ARXIV_RE.search(doi)
            or ARXIV_RE.search(url)
            or ("arxiv" in clean(fields.get("archiveprefix")).lower())
            or ("arxiv" in clean(fields.get("eprinttype")).lower())
            or ("arxiv" in clean(fields.get("journal")).lower())
        )
        arxiv_id = ""
        if arxiv_match:
            arxiv_id = arxiv_match.group(1) if hasattr(arxiv_match, "group") else eprint

        source = "doi" if doi else "arxiv" if arxiv_id else "isbn" if isbn else "url" if url else "none"
        issues = []

        for required in REQUIRED_BY_TYPE.get(entry_type, ["title", "year"]):
            if not fields.get(required):
                issues.append(f"missing_{required}")

        if year and not re.fullmatch(r"\d{4}", year):
            issues.append("invalid_year_format")
        if doi and not doi_valid:
            issues.append("doi_format_check")
        if isbn and not ISBN_RE.fullmatch(isbn):
            issues.append("isbn_format_check")
        if keys[key] > 1:
            issues.append("duplicate_key")
        if doi and doi.lower().rstrip(".") in duplicate_dois:
            issues.append("duplicate_doi")
        if normalized_title in duplicate_titles:
            issues.append("duplicate_title")
        if entry_type == "misc" and not (url or doi or arxiv_id or isbn):
            issues.append("misc_without_identifier_url")
        if entry_type == "article" and "ph.d" in clean(fields.get("journal")).lower():
            issues.append("article_type_but_thesis")
        if entry_type == "article" and ("proceedings" in journal_or_booktitle.lower() or "conference" in journal_or_booktitle.lower()) and not fields.get("booktitle"):
            issues.append("article_type_may_be_inproceedings")
        if entry_type == "inproceedings" and "arxiv preprint" in journal_or_booktitle.lower():
            issues.append("inproceedings_type_but_arxiv_preprint")
        if url and any(domain in url.lower() for domain in WEAK_DOMAINS) and not doi and not arxiv_id and not isbn:
            issues.append("weak_url_only")
        if "stratifiedkfold" in url.lower():
            issues.append("url_points_to_api_page_not_paper")

        online_source = ""
        online_title = ""
        title_match = ""

        if online:
            if doi:
                online_source, online_title, similarity = verify_doi_online(doi, title)
                title_match = f"{similarity:.3f}" if online_source else ""
                if online_source and similarity < 0.65:
                    issues.append("online_title_mismatch")
                elif not online_source:
                    issues.append("online_doi_not_found")
                time.sleep(sleep)
            elif arxiv_id:
                online_source, online_title, similarity = verify_arxiv_online(arxiv_id, title)
                title_match = f"{similarity:.3f}" if online_source else ""
                if online_source and similarity < 0.65:
                    issues.append("online_title_mismatch")
                elif not online_source:
                    issues.append("online_arxiv_not_found")
                time.sleep(sleep)

        if "missing_title" in issues or "duplicate_key" in issues:
            status = "REMOVE_OR_REPLACE"
        elif (
            "online_title_mismatch" in issues
            or "duplicate_doi" in issues
            or "duplicate_title" in issues
            or "article_type_but_thesis" in issues
            or "doi_format_check" in issues
            or "url_points_to_api_page_not_paper" in issues
            or "inproceedings_type_but_arxiv_preprint" in issues
            or "article_type_may_be_inproceedings" in issues
            or len(issues) >= 2
        ):
            status = "NEEDS_METADATA_FIX"
        elif source == "none":
            status = "NEEDS_MANUAL_VERIFICATION"
        elif "weak_url_only" in issues:
            status = "WEAK_SOURCE_ONLY"
        elif entry_type == "book" and isbn:
            status = "OK_BOOK"
        elif entry_type == "phdthesis":
            status = "OK_THESIS"
        elif entry_type == "misc" and (
            key.lower().startswith(("iso", "gdpr", "aiact", "cpp", "cp", "budapest", "directive"))
            or "regulation (eu)" in title.lower()
            or "codice" in title.lower()
            or "convention on cybercrime" in title.lower()
            or "directive" in title.lower()
        ) and source != "none":
            status = "OK_STANDARD_OR_LEGAL"
        elif entry_type == "techreport":
            status = "OK_TECHREPORT"
        elif doi:
            status = "OK_PREPRINT" if ("arxiv" in doi.lower() or "techrxiv" in clean(fields.get("journal")).lower() or "preprint" in clean(fields.get("journal")).lower()) else "OK_PUBLISHED"
        elif arxiv_id:
            status = "OK_PREPRINT"
        elif isbn:
            status = "OK_BOOK"
        else:
            status = "NEEDS_MANUAL_VERIFICATION"

        action_map = {
            "OK_PUBLISHED": "tenere",
            "OK_PREPRINT": "tenere ma indicare preprint se non pubblicato",
            "OK_BOOK": "tenere",
            "OK_THESIS": "tenere come phdthesis",
            "OK_TECHREPORT": "tenere",
            "OK_STANDARD_OR_LEGAL": "tenere, preferendo fonte ufficiale",
            "NEEDS_METADATA_FIX": "correggere metadati o deduplicare",
            "WEAK_SOURCE_ONLY": "sostituire con fonte primaria",
            "NEEDS_MANUAL_VERIFICATION": "verificare manualmente",
            "REMOVE_OR_REPLACE": "rimuovere o sostituire",
        }

        rows.append(
            {
                "index": index,
                "bibkey": key,
                "entry_type": entry_type,
                "status": status,
                "recommended_action": action_map[status],
                "title": title,
                "year": year,
                "author": clean(fields.get("author")),
                "journal_or_booktitle": journal_or_booktitle,
                "publisher_or_institution": publisher_or_institution,
                "doi": doi,
                "eprint": eprint,
                "isbn": isbn,
                "url": url,
                "identifier_type": source,
                "issues": "; ".join(issues),
                "online_source": online_source,
                "online_title": online_title,
                "online_title_similarity": title_match,
                "fields_present": ", ".join(sorted(fields.keys())),
            }
        )

    return rows


def resolve_path_from_script_dir(path_value: Path | None, default_path: Path) -> Path:
    """Resolve a user-supplied path, falling back to the script directory.

    If PyCharm uses a different working directory, a relative path such as
    "tesi.bib" is still searched next to this script.
    """
    if path_value is None:
        return default_path

    if path_value.is_absolute():
        return path_value

    if path_value.exists():
        return path_value.resolve()

    candidate = SCRIPT_DIR / path_value
    if candidate.exists():
        return candidate.resolve()

    return path_value.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit BibTeX bibliography entries. If no .bib file is provided, "
            "the script automatically uses tesi.bib located in the same "
            "directory as this script."
        )
    )
    parser.add_argument(
        "bibfile",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the .bib file. Default: tesi.bib next to this script.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: bibliography_audit.csv next to this script "
            "or bibliography_audit_online.csv when --online is used."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    bibfile = resolve_path_from_script_dir(args.bibfile, DEFAULT_BIBFILE)

    if not bibfile.exists():
        print("[ERROR] BibTeX file not found.", file=sys.stderr)
        print(f"        Looked for: {bibfile}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Fix:", file=sys.stderr)
        print(f"  1. Put tesi.bib in: {SCRIPT_DIR}", file=sys.stderr)
        print("  2. Or pass the file explicitly, for example:", file=sys.stderr)
        print("     python audit_bibliography.py path/to/tesi.bib", file=sys.stderr)
        return 2

    if args.out is None:
        outfile_name = "bibliography_audit_online.csv" if args.online else "bibliography_audit.csv"
        outfile = SCRIPT_DIR / outfile_name
    else:
        outfile = resolve_path_from_script_dir(args.out, DEFAULT_OUTFILE)

    json_out = None
    if args.json_out:
        json_out = resolve_path_from_script_dir(args.json_out, SCRIPT_DIR / args.json_out)

    rows = audit(bibfile, online=args.online, sleep=args.sleep)
    fieldnames = [
        "index",
        "bibkey",
        "entry_type",
        "status",
        "recommended_action",
        "title",
        "year",
        "author",
        "journal_or_booktitle",
        "publisher_or_institution",
        "doi",
        "eprint",
        "isbn",
        "url",
        "identifier_type",
        "issues",
        "online_source",
        "online_title",
        "online_title_similarity",
        "fields_present",
    ]

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(row["status"] for row in rows)
    print(f"BibTeX file: {bibfile}")
    print(f"Entries audited: {len(rows)}")
    for status, count in counts.most_common():
        print(f"{status}: {count}")
    print(f"CSV written to: {outfile}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
