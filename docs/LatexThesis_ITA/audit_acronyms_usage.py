from pathlib import Path
import csv
import re
from collections import Counter, defaultdict


SCRIPT_DIR = Path(__file__).resolve().parent
ACRONYMS_FILE = SCRIPT_DIR / "sections" / "000_acronyms.tex"

OUTPUT_FULL = SCRIPT_DIR / "acronym_usage_report.csv"
OUTPUT_UNUSED = SCRIPT_DIR / "acronym_unused_entries.csv"
OUTPUT_MISSING = SCRIPT_DIR / "acronym_missing_entries.csv"
OUTPUT_SUMMARY = SCRIPT_DIR / "acronym_usage_summary.md"


def remove_latex_comments(text: str) -> str:
    """
    Remove LaTeX comments while preserving escaped percent symbols such as \\%.
    This avoids counting commands appearing only in commented-out text.
    """
    cleaned_lines = []

    for line in text.splitlines():
        cleaned_line = re.sub(r"(?<!\\)%.*$", "", line)
        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def parse_defined_entries(acronyms_path: Path) -> dict[str, str]:
    """
    Extract labels defined in the acronym/glossary source file.

    Supported definitions:
        \\newacronym{ai}{AI}{Artificial Intelligence}
        \\newacronym[...]{dfir}{DFIR}{...}
        \\newabbreviation{ai}{AI}{Artificial Intelligence}
        \\newglossaryentry{sigmazero}{...}

    Returns:
        Dictionary mapping each label to its entry type:
            {
                "ai": "acronym",
                "sigmazero": "glossary_entry"
            }
    """
    text = acronyms_path.read_text(encoding="utf-8", errors="replace")
    text = remove_latex_comments(text)

    defined_entries: dict[str, str] = {}

    acronym_pattern = re.compile(
        r"\\new(?:acronym|abbreviation)"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*\{\s*([^{}\s]+)\s*\}",
        re.DOTALL,
    )

    glossary_pattern = re.compile(
        r"\\newglossaryentry"
        r"(?:\s*\[[^\]]*\])?"
        r"\s*\{\s*([^{}\s]+)\s*\}",
        re.DOTALL,
    )

    for match in acronym_pattern.finditer(text):
        label = match.group(1).strip()
        defined_entries[label] = "acronym"

    for match in glossary_pattern.finditer(text):
        label = match.group(1).strip()
        defined_entries[label] = "glossary_entry"

    return dict(sorted(defined_entries.items()))


def scan_tex_entry_usage(
    root_dir: Path,
    definitions_file: Path,
) -> tuple[Counter, dict[str, set[str]], dict[str, set[str]], bool]:
    """
    Scan all .tex files and locate actual acronym/glossary usages.

    Supported usage commands include:
        \\gls{ai}
        \\Gls{ai}
        \\glspl{cnn}
        \\acrshort{dfir}
        \\acrlong{dfir}
        \\acrfull{dfir}
        \\glsentryshort{ai}
        \\glsentrylong{ai}
        \\glslink{sigmazero}{Sigma Zero}
        \\glsadd{ai}

    Returns:
        - Counter of usage occurrences per label.
        - Files in which each label is used.
        - Commands used for each label.
        - Whether \\glsaddall was found.
    """
    usage_counter = Counter()
    usage_locations: dict[str, set[str]] = defaultdict(set)
    usage_commands: dict[str, set[str]] = defaultdict(set)
    glsaddall_found = False

    usage_pattern = re.compile(
        r"""
        \\(?P<command>
            gls|Gls|GLS|
            glspl|Glspl|GLSpl|
            glsfirst|Glsfirst|GLSfirst|
            glstext|Glstext|GLStext|
            acrshort|Acrshort|ACRshort|
            acrlong|Acrlong|ACRlong|
            acrfull|Acrfull|ACRfull|
            glsentryshort|
            glsentrylong|
            glsentryfull|
            glsentrytext|
            glsentryname|
            glslink|
            glsdisp|
            glsadd
        )
        \*?
        (?:\s*\[[^\]]*\]){0,2}
        \s*\{\s*(?P<label>[^{}\s,]+)\s*\}
        """,
        re.VERBOSE,
    )

    tex_files = sorted(root_dir.rglob("*.tex"))

    for tex_file in tex_files:
        if tex_file.resolve() == definitions_file.resolve():
            continue

        relative_path = str(tex_file.relative_to(root_dir))

        text = tex_file.read_text(encoding="utf-8", errors="replace")
        text = remove_latex_comments(text)

        if re.search(r"\\glsaddall\b", text):
            glsaddall_found = True

        for match in usage_pattern.finditer(text):
            command = match.group("command")
            label = match.group("label").strip()

            usage_counter[label] += 1
            usage_locations[label].add(relative_path)
            usage_commands[label].add(command)

    return usage_counter, usage_locations, usage_commands, glsaddall_found


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """
    Write report rows to a UTF-8 CSV file.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ACRONYMS_FILE.exists():
        raise FileNotFoundError(
            f"Acronym definitions file not found: {ACRONYMS_FILE}"
        )

    defined_entries = parse_defined_entries(ACRONYMS_FILE)

    (
        usage_counts,
        usage_locations,
        usage_commands,
        glsaddall_found,
    ) = scan_tex_entry_usage(SCRIPT_DIR, ACRONYMS_FILE)

    defined_keys = set(defined_entries)
    used_keys = set(usage_counts)

    used_and_defined = defined_keys & used_keys
    unused_keys = defined_keys - used_keys
    missing_keys = used_keys - defined_keys

    acronym_defined = {
        key for key, entry_type in defined_entries.items()
        if entry_type == "acronym"
    }

    glossary_defined = {
        key for key, entry_type in defined_entries.items()
        if entry_type == "glossary_entry"
    }

    full_rows = []

    for key in sorted(defined_keys | used_keys):
        if key in defined_keys and key in used_keys:
            status = "USED_AND_DEFINED"
            entry_type = defined_entries[key]
        elif key in defined_keys and key not in used_keys:
            status = "DEFINED_NOT_USED"
            entry_type = defined_entries[key]
        else:
            status = "USED_BUT_MISSING_DEFINITION"
            entry_type = "missing"

        full_rows.append(
            {
                "label": key,
                "entry_type": entry_type,
                "status": status,
                "defined_in_acronyms_file": "yes" if key in defined_keys else "no",
                "used_in_tex_sources": "yes" if key in used_keys else "no",
                "tex_occurrences": usage_counts.get(key, 0),
                "commands": "; ".join(sorted(usage_commands.get(key, []))),
                "tex_files": "; ".join(sorted(usage_locations.get(key, []))),
            }
        )

    unused_rows = [
        {
            "label": key,
            "entry_type": defined_entries[key],
            "status": "DEFINED_NOT_USED",
            "tex_occurrences": usage_counts.get(key, 0),
            "commands": "; ".join(sorted(usage_commands.get(key, []))),
            "tex_files": "; ".join(sorted(usage_locations.get(key, []))),
        }
        for key in sorted(unused_keys)
    ]

    missing_rows = [
        {
            "label": key,
            "entry_type": "missing",
            "status": "USED_BUT_MISSING_DEFINITION",
            "tex_occurrences": usage_counts.get(key, 0),
            "commands": "; ".join(sorted(usage_commands.get(key, []))),
            "tex_files": "; ".join(sorted(usage_locations.get(key, []))),
        }
        for key in sorted(missing_keys)
    ]

    write_csv(
        OUTPUT_FULL,
        full_rows,
        [
            "label",
            "entry_type",
            "status",
            "defined_in_acronyms_file",
            "used_in_tex_sources",
            "tex_occurrences",
            "commands",
            "tex_files",
        ],
    )

    write_csv(
        OUTPUT_UNUSED,
        unused_rows,
        [
            "label",
            "entry_type",
            "status",
            "tex_occurrences",
            "commands",
            "tex_files",
        ],
    )

    write_csv(
        OUTPUT_MISSING,
        missing_rows,
        [
            "label",
            "entry_type",
            "status",
            "tex_occurrences",
            "commands",
            "tex_files",
        ],
    )

    glsaddall_warning = ""

    if glsaddall_found:
        glsaddall_warning = """
## Warning

The command `\\glsaddall` was found in the LaTeX sources. This command forces all
defined glossary entries to be included in the printed glossary, even when they
are not explicitly referenced with commands such as `\\gls{...}`. The unused-entry
report therefore identifies entries not explicitly referenced in the text, not
necessarily entries absent from the compiled glossary.
"""

    summary = f"""# Acronym and Glossary Usage Summary

## Input file

- Acronym and glossary definitions: `{ACRONYMS_FILE.relative_to(SCRIPT_DIR)}`

## Counts

| Metric | Count |
|---|---:|
| Defined acronyms | {len(acronym_defined)} |
| Defined glossary entries | {len(glossary_defined)} |
| Total defined labels | {len(defined_keys)} |
| Labels used in `.tex` sources | {len(used_keys)} |
| Labels used and correctly defined | {len(used_and_defined)} |
| Labels defined but not explicitly used | {len(unused_keys)} |
| Labels used but missing a definition | {len(missing_keys)} |

## Output files

- `{OUTPUT_FULL.name}`
- `{OUTPUT_UNUSED.name}`
- `{OUTPUT_MISSING.name}`

{glsaddall_warning}
"""

    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print("Acronym and glossary usage audit completed.")
    print(f"Defined acronyms: {len(acronym_defined)}")
    print(f"Defined glossary entries: {len(glossary_defined)}")
    print(f"Total defined labels: {len(defined_keys)}")
    print(f"Labels used in .tex sources: {len(used_keys)}")
    print(f"Used and correctly defined: {len(used_and_defined)}")
    print(f"Defined but not explicitly used: {len(unused_keys)}")
    print(f"Used but missing a definition: {len(missing_keys)}")

    if glsaddall_found:
        print()
        print(
            "Warning: \\glsaddall was found. "
            "Unused labels may still appear in the compiled glossary."
        )

    print()
    print(f"Full report: {OUTPUT_FULL}")
    print(f"Unused entries: {OUTPUT_UNUSED}")
    print(f"Missing definitions: {OUTPUT_MISSING}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()