#!/usr/bin/env python3
"""
Check acronym usage in a LaTeX thesis.

The script verifies:
1. Acronyms declared with \\newacronym.
2. Glossary entries declared with \\newglossaryentry.
3. Acronyms declared but never used.
4. Acronym/glossary keys used with \\gls-like commands but not defined.
5. Optional raw textual occurrences of acronym short forms, e.g. "AI" written
   directly instead of using \\gls{ai}.

Typical usage:
    python check_latex_acronyms.py
    python check_latex_acronyms.py --root .
    python check_latex_acronyms.py --root docs/LatexThesis_ITA
    python check_latex_acronyms.py --csv acronym_usage_report.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ACRONYM_COMMAND = r"\newacronym"
GLOSSARY_COMMAND = r"\newglossaryentry"

USAGE_COMMANDS = {
    "gls", "Gls", "GLS",
    "glspl", "Glspl", "GLSpl",
    "glslink", "glsdisp",
    "acrshort", "Acrshort",
    "acrshortpl", "Acrshortpl",
    "acrlong", "Acrlong",
    "acrlongpl", "Acrlongpl",
    "acrfull", "Acrfull",
    "acrfullpl", "Acrfullpl",
    "glsentryshort", "glsentrylong", "glsentryfull",
    "Glsentryshort", "Glsentrylong", "Glsentryfull",
}

IGNORED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "build",
    "out",
    "output",
    "outputs",
    "figures",
    "images",
    "latex-build",
}

IGNORED_TEX_FILES = {
    # Add files here if needed.
}


@dataclass
class Definition:
    key: str
    kind: str
    short: str = ""
    long: str = ""
    file: Path = Path()
    line: int = 0


@dataclass
class Usage:
    key: str
    command: str
    file: Path
    line: int
    context: str


@dataclass
class RawOccurrence:
    acronym_key: str
    short: str
    file: Path
    line: int
    context: str


@dataclass
class Report:
    acronyms: Dict[str, Definition] = field(default_factory=dict)
    glossary_entries: Dict[str, Definition] = field(default_factory=dict)
    usages: List[Usage] = field(default_factory=list)
    raw_occurrences: List[RawOccurrence] = field(default_factory=list)


def strip_latex_comments(text: str) -> str:
    """Remove LaTeX comments while preserving line numbers."""
    cleaned_lines = []

    for line in text.splitlines():
        result = []
        i = 0

        while i < len(line):
            char = line[i]

            if char == "%":
                backslashes = 0
                j = i - 1

                while j >= 0 and line[j] == "\\":
                    backslashes += 1
                    j -= 1

                if backslashes % 2 == 0:
                    break

            result.append(char)
            i += 1

        cleaned_lines.append("".join(result))

    return "\n".join(cleaned_lines)


def line_number_at(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def find_matching_brace(text: str, open_index: int) -> Optional[int]:
    """Return the index of the matching closing brace for text[open_index] == '{'."""
    if open_index >= len(text) or text[open_index] != "{":
        return None

    depth = 0
    escaped = False

    for i in range(open_index, len(text)):
        char = text[i]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i

    return None


def skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def skip_optional_argument(text: str, index: int) -> int:
    """Skip one optional LaTeX argument [ ... ], including nested braces."""
    index = skip_whitespace(text, index)

    if index >= len(text) or text[index] != "[":
        return index

    depth_square = 0
    depth_curly = 0
    escaped = False

    for i in range(index, len(text)):
        char = text[i]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly = max(0, depth_curly - 1)
        elif char == "[" and depth_curly == 0:
            depth_square += 1
        elif char == "]" and depth_curly == 0:
            depth_square -= 1
            if depth_square == 0:
                return i + 1

    return index


def read_mandatory_group(text: str, index: int) -> Tuple[Optional[str], int]:
    index = skip_whitespace(text, index)

    if index >= len(text) or text[index] != "{":
        return None, index

    close_index = find_matching_brace(text, index)

    if close_index is None:
        return None, index

    content = text[index + 1:close_index]
    return content.strip(), close_index + 1


def extract_newacronyms(text: str, file_path: Path) -> Dict[str, Definition]:
    definitions: Dict[str, Definition] = {}
    pattern = re.compile(r"\\newacronym\b")

    for match in pattern.finditer(text):
        index = match.end()
        index = skip_optional_argument(text, index)

        key, index = read_mandatory_group(text, index)
        short, index = read_mandatory_group(text, index)
        long, index = read_mandatory_group(text, index)

        if key:
            definitions[key] = Definition(
                key=key,
                kind="acronym",
                short=short or "",
                long=long or "",
                file=file_path,
                line=line_number_at(text, match.start()),
            )

    return definitions


def extract_glossary_entries(text: str, file_path: Path) -> Dict[str, Definition]:
    definitions: Dict[str, Definition] = {}
    pattern = re.compile(r"\\newglossaryentry\b")

    for match in pattern.finditer(text):
        index = match.end()

        key, index = read_mandatory_group(text, index)

        if key:
            definitions[key] = Definition(
                key=key,
                kind="glossary",
                file=file_path,
                line=line_number_at(text, match.start()),
            )

    return definitions


def extract_usages(text: str, file_path: Path) -> List[Usage]:
    usages: List[Usage] = []

    command_pattern = re.compile(
        r"\\(" + "|".join(re.escape(cmd) for cmd in sorted(USAGE_COMMANDS, key=len, reverse=True)) + r")\b"
    )

    for match in command_pattern.finditer(text):
        command = match.group(1)
        index = match.end()

        # Some glossary commands may accept optional arguments before the key.
        index = skip_optional_argument(text, index)
        index = skip_optional_argument(text, index)

        key, _ = read_mandatory_group(text, index)

        if not key:
            continue

        # For commands such as \glslink{key}{text} and \glsdisp{key}{text},
        # only the first mandatory argument is the acronym/glossary key.
        line = line_number_at(text, match.start())
        context = get_line_context(text, line)

        usages.append(
            Usage(
                key=key,
                command=command,
                file=file_path,
                line=line,
                context=context,
            )
        )

    return usages


def get_line_context(text: str, line_number: int) -> str:
    lines = text.splitlines()

    if 1 <= line_number <= len(lines):
        return lines[line_number - 1].strip()

    return ""


def find_raw_short_occurrences(
    text: str,
    file_path: Path,
    acronyms: Dict[str, Definition],
) -> List[RawOccurrence]:
    """
    Find direct textual occurrences of acronym short forms.

    Example:
        "AI" appears in text, but the preferred form may be "\\gls{ai}".

    This is only a warning mechanism. It may produce false positives in tables,
    captions, bibliography snippets, code listings, or already-expanded contexts.
    """
    occurrences: List[RawOccurrence] = []

    for key, definition in acronyms.items():
        short = definition.short.strip()

        if not short:
            continue

        # Avoid noisy one-letter acronyms.
        if len(short) < 2:
            continue

        pattern = re.compile(rf"(?<![A-Za-z0-9\\]){re.escape(short)}(?![A-Za-z0-9])")

        for match in pattern.finditer(text):
            line = line_number_at(text, match.start())
            context = get_line_context(text, line)

            # Do not flag the declaration itself or proper LaTeX acronym commands.
            if r"\newacronym" in context:
                continue
            if f"\\gls{{{key}}}" in context:
                continue
            if f"\\Gls{{{key}}}" in context:
                continue
            if f"\\acrshort{{{key}}}" in context:
                continue
            if f"\\acrfull{{{key}}}" in context:
                continue
            if f"\\acrlong{{{key}}}" in context:
                continue

            occurrences.append(
                RawOccurrence(
                    acronym_key=key,
                    short=short,
                    file=file_path,
                    line=line,
                    context=context,
                )
            )

    return occurrences


def collect_tex_files(root: Path) -> List[Path]:
    tex_files = []

    for path in root.rglob("*.tex"):
        if path.name in IGNORED_TEX_FILES:
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        tex_files.append(path)

    return sorted(tex_files)


def analyze(root: Path, check_raw_text: bool) -> Report:
    report = Report()
    tex_files = collect_tex_files(root)

    file_texts: Dict[Path, str] = {}

    for tex_file in tex_files:
        raw_text = tex_file.read_text(encoding="utf-8", errors="replace")
        clean_text = strip_latex_comments(raw_text)
        file_texts[tex_file] = clean_text

        report.acronyms.update(extract_newacronyms(clean_text, tex_file))
        report.glossary_entries.update(extract_glossary_entries(clean_text, tex_file))

    for tex_file, clean_text in file_texts.items():
        report.usages.extend(extract_usages(clean_text, tex_file))

    if check_raw_text:
        for tex_file, clean_text in file_texts.items():
            report.raw_occurrences.extend(
                find_raw_short_occurrences(clean_text, tex_file, report.acronyms)
            )

    return report


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_report(report: Report, root: Path, max_raw_warnings: int) -> None:
    all_definitions = {**report.glossary_entries, **report.acronyms}

    used_keys = {usage.key for usage in report.usages}
    acronym_keys = set(report.acronyms.keys())
    defined_keys = set(all_definitions.keys())

    unused_acronyms = sorted(acronym_keys - used_keys)
    undefined_usages = sorted(used_keys - defined_keys)

    usage_count: Dict[str, int] = {}
    for usage in report.usages:
        usage_count[usage.key] = usage_count.get(usage.key, 0) + 1

    print("\n=== LaTeX Acronym Usage Report ===\n")

    print(f"Root: {root}")
    print(f"Declared acronyms: {len(report.acronyms)}")
    print(f"Declared glossary entries: {len(report.glossary_entries)}")
    print(f"Total \\gls-like usages: {len(report.usages)}")
    print(f"Unused acronyms: {len(unused_acronyms)}")
    print(f"Undefined used keys: {len(undefined_usages)}")

    print("\n--- Declared acronyms ---")
    for key in sorted(report.acronyms):
        definition = report.acronyms[key]
        count = usage_count.get(key, 0)
        print(
            f"{key:20s} | {definition.short:15s} | used {count:3d}x | "
            f"{relative(definition.file, root)}:{definition.line}"
        )

    if unused_acronyms:
        print("\n--- Acronyms declared but never used ---")
        for key in unused_acronyms:
            definition = report.acronyms[key]
            print(
                f"{key:20s} | {definition.short:15s} | "
                f"{relative(definition.file, root)}:{definition.line}"
            )

    if undefined_usages:
        print("\n--- Used keys not defined as acronym or glossary entry ---")
        for key in undefined_usages:
            locations = [
                usage for usage in report.usages
                if usage.key == key
            ]

            print(f"\n{key}")
            for usage in locations[:10]:
                print(
                    f"  - \\{usage.command}{{{key}}} at "
                    f"{relative(usage.file, root)}:{usage.line}"
                )
                print(f"    {usage.context}")

            if len(locations) > 10:
                print(f"    ... and {len(locations) - 10} more occurrence(s)")

    if report.raw_occurrences:
        print("\n--- Possible raw acronym text instead of \\gls{...} ---")
        print(
            "These are warnings only. Some may be correct, especially in tables, "
            "captions, code, or already-expanded text.\n"
        )

        for occurrence in report.raw_occurrences[:max_raw_warnings]:
            print(
                f"{occurrence.short:15s} | key={occurrence.acronym_key:15s} | "
                f"{relative(occurrence.file, root)}:{occurrence.line}"
            )
            print(f"  {occurrence.context}")

        if len(report.raw_occurrences) > max_raw_warnings:
            print(
                f"\n... {len(report.raw_occurrences) - max_raw_warnings} "
                f"additional raw-text warning(s) not shown."
            )

    print("\n=== End of report ===\n")


def write_csv(report: Report, root: Path, csv_path: Path) -> None:
    usage_count: Dict[str, int] = {}

    for usage in report.usages:
        usage_count[usage.key] = usage_count.get(usage.key, 0) + 1

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "kind",
            "key",
            "short",
            "long",
            "usage_count",
            "definition_file",
            "definition_line",
            "status",
        ])

        for key in sorted(report.acronyms):
            definition = report.acronyms[key]
            count = usage_count.get(key, 0)

            writer.writerow([
                definition.kind,
                key,
                definition.short,
                definition.long,
                count,
                relative(definition.file, root),
                definition.line,
                "used" if count > 0 else "unused",
            ])

    print(f"CSV report written to: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check acronym usage in a LaTeX thesis."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory of the LaTeX thesis. Default: current directory.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    parser.add_argument(
        "--no-raw-text-check",
        action="store_true",
        help="Disable warnings for raw acronym text such as AI instead of \\gls{ai}.",
    )

    parser.add_argument(
        "--max-raw-warnings",
        type=int,
        default=100,
        help="Maximum number of raw-text warnings printed. Default: 100.",
    )

    args = parser.parse_args()

    root = args.root.resolve()

    if not root.exists():
        print(f"ERROR: root directory does not exist: {root}")
        return 1

    report = analyze(
        root=root,
        check_raw_text=not args.no_raw_text_check,
    )

    print_report(
        report=report,
        root=root,
        max_raw_warnings=args.max_raw_warnings,
    )

    if args.csv:
        write_csv(report=report, root=root, csv_path=args.csv)

    undefined_keys = {
        usage.key
        for usage in report.usages
        if usage.key not in {**report.acronyms, **report.glossary_entries}
    }

    # Exit code:
    # 0 = no critical errors
    # 1 = undefined glossary/acronym keys found
    if undefined_keys:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())