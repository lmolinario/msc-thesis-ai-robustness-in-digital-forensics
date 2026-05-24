from pathlib import Path
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


SCRIPT_DIR = Path(__file__).resolve().parent
BIB_FILE = SCRIPT_DIR / "tesi.bib"
BCF_FILE = SCRIPT_DIR / "main.bcf"
OUTPUT_FULL = SCRIPT_DIR / "bibliography_usage_report.csv"
OUTPUT_UNUSED = SCRIPT_DIR / "bibliography_unused_entries.csv"
OUTPUT_MISSING = SCRIPT_DIR / "bibliography_missing_entries.csv"
OUTPUT_SUMMARY = SCRIPT_DIR / "bibliography_usage_summary.md"


def parse_bib_keys(bib_path: Path) -> list[str]:
    """
    Extract BibTeX/BibLaTeX keys from a .bib file.
    Matches entries such as:
        @article{key,
        @misc{key,
        @inproceedings{key,
    """
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
    return sorted(set(pattern.findall(text)))


def parse_cited_keys_from_bcf(bcf_path: Path) -> list[str]:
    """
    Extract citekeys from the Biber .bcf file.
    This is the most reliable source after running pdflatex once.
    """
    tree = ET.parse(bcf_path)
    root = tree.getroot()

    cited = []

    for elem in root.iter():
        # Handles namespaces safely, e.g. {namespace}citekey
        tag = elem.tag.split("}")[-1]
        if tag == "citekey" and elem.text:
            cited.append(elem.text.strip())

    return sorted(set(cited))


def scan_tex_citations(root_dir: Path) -> tuple[Counter, dict[str, set[str]]]:
    """
    Scan .tex files to count citation occurrences and record where each key appears.
    This is useful for locating cited entries in source files.
    """
    cite_counter = Counter()
    cite_locations = defaultdict(set)

    # Covers common BibLaTeX citation commands.
    cite_command = re.compile(
        r"\\(?:"
        r"cite|parencite|textcite|autocite|footcite|smartcite|supercite|"
        r"citeauthor|citeyear|Cite|Parencite|Textcite"
        r")"
        r"(?:\s*\[[^\]]*\]){0,2}"
        r"\s*\{([^}]+)\}"
    )

    tex_files = sorted(root_dir.rglob("*.tex"))

    for tex_file in tex_files:
        # Skip generated or irrelevant files if needed.
        if tex_file.name.startswith("main."):
            continue

        text = tex_file.read_text(encoding="utf-8", errors="replace")

        for match in cite_command.finditer(text):
            raw_keys = match.group(1)
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

            for key in keys:
                cite_counter[key] += 1
                cite_locations[key].add(str(tex_file.relative_to(root_dir)))

    return cite_counter, cite_locations


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not BIB_FILE.exists():
        raise FileNotFoundError(f"BibTeX file not found: {BIB_FILE}")

    if not BCF_FILE.exists():
        raise FileNotFoundError(
            f"Biber control file not found: {BCF_FILE}\n"
            "Run at least: pdflatex main.tex && biber main"
        )

    bib_keys = set(parse_bib_keys(BIB_FILE))
    cited_keys = set(parse_cited_keys_from_bcf(BCF_FILE))
    tex_counts, tex_locations = scan_tex_citations(SCRIPT_DIR)

    used_keys = bib_keys & cited_keys
    unused_keys = bib_keys - cited_keys
    missing_keys = cited_keys - bib_keys

    full_rows = []

    for key in sorted(bib_keys | cited_keys):
        if key in bib_keys and key in cited_keys:
            status = "CITED_AND_PRESENT"
        elif key in bib_keys and key not in cited_keys:
            status = "IN_BIB_NOT_CITED"
        else:
            status = "CITED_BUT_MISSING_IN_BIB"

        full_rows.append(
            {
                "bibkey": key,
                "status": status,
                "in_tesi_bib": "yes" if key in bib_keys else "no",
                "cited_by_biber": "yes" if key in cited_keys else "no",
                "tex_occurrences": tex_counts.get(key, 0),
                "tex_files": "; ".join(sorted(tex_locations.get(key, []))),
            }
        )

    unused_rows = [
        {
            "bibkey": key,
            "status": "IN_BIB_NOT_CITED",
            "tex_occurrences": tex_counts.get(key, 0),
            "tex_files": "; ".join(sorted(tex_locations.get(key, []))),
        }
        for key in sorted(unused_keys)
    ]

    missing_rows = [
        {
            "bibkey": key,
            "status": "CITED_BUT_MISSING_IN_BIB",
            "tex_occurrences": tex_counts.get(key, 0),
            "tex_files": "; ".join(sorted(tex_locations.get(key, []))),
        }
        for key in sorted(missing_keys)
    ]

    write_csv(
        OUTPUT_FULL,
        full_rows,
        [
            "bibkey",
            "status",
            "in_tesi_bib",
            "cited_by_biber",
            "tex_occurrences",
            "tex_files",
        ],
    )

    write_csv(
        OUTPUT_UNUSED,
        unused_rows,
        ["bibkey", "status", "tex_occurrences", "tex_files"],
    )

    write_csv(
        OUTPUT_MISSING,
        missing_rows,
        ["bibkey", "status", "tex_occurrences", "tex_files"],
    )

    summary = f"""# Bibliography Usage Summary

## Input files

- BibTeX file: `{BIB_FILE.name}`
- Biber control file: `{BCF_FILE.name}`

## Counts

| Metric | Count |
|---|---:|
| Entries in `tesi.bib` | {len(bib_keys)} |
| Citekeys used by Biber | {len(cited_keys)} |
| Entries cited and present | {len(used_keys)} |
| Entries in `tesi.bib` but not cited | {len(unused_keys)} |
| Citekeys cited but missing in `tesi.bib` | {len(missing_keys)} |

## Output files

- `{OUTPUT_FULL.name}`
- `{OUTPUT_UNUSED.name}`
- `{OUTPUT_MISSING.name}`
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print("Bibliography usage audit completed.")
    print(f"Entries in tesi.bib: {len(bib_keys)}")
    print(f"Citekeys used by Biber: {len(cited_keys)}")
    print(f"Cited and present: {len(used_keys)}")
    print(f"In tesi.bib but not cited: {len(unused_keys)}")
    print(f"Cited but missing in tesi.bib: {len(missing_keys)}")
    print()
    print(f"Full report: {OUTPUT_FULL}")
    print(f"Unused entries: {OUTPUT_UNUSED}")
    print(f"Missing entries: {OUTPUT_MISSING}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()