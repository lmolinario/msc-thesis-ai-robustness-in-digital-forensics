#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_normalize_forensic_ai_tool_predictions.py

Interactive and CLI-capable normalization entry point for commercial forensic
-tool outputs in the FAIR-Lab thesis pipeline.

Purpose
-------
Normalize forensic-tool exports against the validated forensic evaluation
bundle and produce comparable CSV metrics for the experimental results and reporting layer.

Implemented normalization logic
------------------------------
- Magnet AXIOM / Magnet.AI:
  - reads Pictures.csv from selected raw export run folders;
  - maps Tags = "Possible weapons" to predicted weapon;
  - maps empty Tags to not flagged / predicted non_weapon;
  - deduplicates duplicated export rows to one prediction per tool + bundle_id.

- X-Ways / Excire Photo AI semantic prompt exports:
  - reads one hit-list CSV per fixed firearm-oriented text prompt;
  - builds the union of all retrieved filenames as predicted weapon;
  - completes all bundle rows not retrieved by any prompt as predicted non_weapon;
  - preserves per-prompt hit flags for auditability.

- Generic forensic AI tool exports:
  - supports CSV, TSV, JSON, JSONL and TXT exports;
  - attempts to infer filename/hash, label/category and confidence columns;
  - maps weapon-related labels to the binary FAIR-Lab task when possible.

Main inputs
-----------
- datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
- datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
- forensic_tools/<tool_name>/raw_exports/<run_id>/...

Main outputs
------------
- evaluation/forensic_tools/normalized_predictions.csv
- evaluation/forensic_tools/<tool_name>_normalized_predictions.csv
- evaluation/forensic_tools/tool_export_audit.csv
- evaluation/forensic_tools/tool_version_log.csv
- evaluation/forensic_tools/normalization_summary.json
- results/metrics/forensic_tools_metrics.csv
- results/metrics/<tool_name>_metrics.csv

Operational note
----------------
By default the script opens an interactive menu, so the analyst can select what
runs/tools must be normalized. Use --no-interactive for fully scripted execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Repository path bootstrap
# =============================================================================

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

try:
    from datasets.scripts.utils.paths import REPO_ROOT, repo_relative_path
except Exception:  # pragma: no cover - fallback for unusual execution contexts
    REPO_ROOT = REPO_ROOT_BOOTSTRAP

    def repo_relative_path(path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (REPO_ROOT / path).resolve()


SCRIPT_NAME = "evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py"

DEFAULT_BUNDLE_MANIFEST = (
    REPO_ROOT
    / "datasets"
    / "forensic_evaluation_bundle"
    / "metadata"
    / "bundle_manifest.csv"
)
DEFAULT_BUNDLE_HASHES = (
    REPO_ROOT
    / "datasets"
    / "forensic_evaluation_bundle"
    / "metadata"
    / "bundle_hashes_sha256.csv"
)
DEFAULT_FORENSIC_TOOLS_ROOT = REPO_ROOT / "forensic_tools"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "forensic_tools"
DEFAULT_METRICS_DIR = REPO_ROOT / "results" / "metrics"

KNOWN_TOOL_NAMES = [
    "magnet_axiom",
    "excire_foto_2025",
    "cellebrite_inseyets",
]
SUPPORTED_GENERIC_EXPORT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".txt",
    ".xlsx",
    ".xls",
}

EXCIRE_FIREARM_PROMPTS = (
    "firearm",
    "gun",
    "pistol",
    "handgun",
    "revolver",
    "rifle",
    "shotgun",
    "assault_rifle",
)

EXCIRE_PROMPT_EXPORT_RE = re.compile(
    r"^excire_(?P<prompt>[a-z0-9_]+)_distance(?P<distance>\d+)\.csv$",
    flags=re.IGNORECASE,
)

SHA256_COLUMNS = {
    "sha256",
    "sha-256",
    "sha_256",
    "hash_sha256",
    "sha256_hash",
    "file_sha256",
    "artifact_sha256",
}

MD5_COLUMNS = {
    "md5",
    "md5 hash",
    "md5_hash",
    "hash_md5",
    "file_md5",
    "artifact_md5",
}

FILENAME_COLUMNS = {
    "filename",
    "file name",
    "file_name",
    "name",
    "nome",
    "item_name",
    "artifact_name",
    "original_filename",
    "tool_input_filename",
    "path",
    "percorso",
    "file_path",
    "filepath",
    "source_path",
    "full_path",
    "export_path",
}

LABEL_COLUMNS = {
    "label",
    "labels",
    "category",
    "categories",
    "tag",
    "tags",
    "classification",
    "Classifications",
    "Classification",
    "Classificazioni",
    "class",
    "ai_label",
    "ai_category",
    "recognized_category",
    "detected_category",
    "result",
    "description",
}

CONFIDENCE_COLUMNS = {
    "confidence",
    "score",
    "probability",
    "confidence_score",
    "ai_confidence",
    "rank_score",
}

WEAPON_KEYWORDS = {
    "weapon",
    "weapons",
    "gun",
    "guns",
    "firearm",
    "firearms",
    "pistol",
    "pistols",
    "rifle",
    "rifles",
    "shotgun",
    "shotguns",
    "revolver",
    "handgun",
    "handguns",
    "ammo",
    "ammunition",
}

NON_WEAPON_NEGATIVE_KEYWORDS = {
    "no weapon",
    "non weapon",
    "non_weapon",
    "not weapon",
    "not a weapon",
    "none",
    "negative",
    "benign",
    "normal",
    "safe",
    "uncategorized",
}

OOD_HINT_KEYWORDS = {
    "knife",
    "knives",
    "sword",
    "toy",
    "replica",
    "airsoft",
    "cgi",
    "cartoon",
    "synthetic",
}

CELLEBRITE_WEAPON_CLASSIFICATIONS_EXTENDED = {
    "armi",
    "pistola",
    "fucile",
}

CELLEBRITE_WEAPON_CLASSIFICATIONS_STRICT = {
    "armi",
}

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RawToolRow:
    """Raw row parsed from a forensic-tool export."""

    tool_name: str
    raw_export_file: str
    raw_row_number: int
    raw_record: dict[str, Any]
    sha256: str
    md5: str
    filename_or_path: str
    raw_label: str
    raw_confidence: str


@dataclass
class InteractiveSelection:
    """Configuration selected from the initial interactive menu."""

    tools: list[str]
    selected_run_dirs_by_tool: dict[str, list[Path]]
    strict: bool
    deduplicate: bool
    output_dir: Path
    metrics_dir: Path


# =============================================================================
# CLI and logging
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize forensic-tool exports against the FAIR-Lab forensic "
            "evaluation bundle. By default, an interactive menu is shown."
        )
    )
    parser.add_argument("--bundle-manifest", default=str(DEFAULT_BUNDLE_MANIFEST))
    parser.add_argument("--bundle-hashes", default=str(DEFAULT_BUNDLE_HASHES))
    parser.add_argument("--forensic-tools-root", default=str(DEFAULT_FORENSIC_TOOLS_ROOT))
    parser.add_argument("--tools", nargs="+", default=KNOWN_TOOL_NAMES)
    parser.add_argument("--selected-run-dir", nargs="*", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", safe_str(name).lower()).strip("_")


def normalize_hash(value: Any) -> str:
    text = safe_str(value).lower()
    return re.sub(r"[^a-f0-9]", "", text)


def basename_from_path(value: Any) -> str:
    text = safe_str(value).replace("\\", "/")
    if not text:
        return ""
    return text.rstrip("/").split("/")[-1]


def repo_relative_string(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate).replace("\\", "/")


def first_non_empty(record: dict[str, Any], candidate_columns: set[str]) -> str:
    normalized_candidates = {normalize_column_name(col) for col in candidate_columns}
    for key, value in record.items():
        if normalize_column_name(key) in normalized_candidates:
            text = safe_str(value)
            if text:
                return text
    return ""


def collect_text_fields(record: dict[str, Any], candidate_columns: set[str]) -> str:
    normalized_candidates = {normalize_column_name(col) for col in candidate_columns}
    values: list[str] = []
    for key, value in record.items():
        if normalize_column_name(key) in normalized_candidates:
            text = safe_str(value)
            if text:
                values.append(text)
    return " | ".join(values)


def parse_float(value: Any) -> float | None:
    text = safe_str(value).replace(",", ".")
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def unique_join(values: list[Any], separator: str = " | ") -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = safe_str(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return separator.join(output)


def ask_text(prompt: str, default: str = "") -> str:
    """Ask for a text value. Return default on empty input or non-interactive EOF."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return value or default


def ask_yes_no(prompt: str, default: bool) -> bool:
    """Ask a yes/no question."""
    default_text = "Y/n" if default else "y/N"
    try:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
    except EOFError:
        return default
    if not value:
        return default
    return value in {"y", "yes", "s", "si", "sì", "true", "1"}


def ask_index(prompt: str, min_value: int, max_value: int, default: int) -> int:
    """Ask for an integer menu choice within a range."""
    while True:
        value = ask_text(prompt, str(default))
        try:
            choice = int(value)
        except ValueError:
            print(f"Invalid value: {value}")
            continue
        if min_value <= choice <= max_value:
            return choice
        print(f"Choose a value between {min_value} and {max_value}.")

def parse_cellebrite_classifications(value: Any) -> set[str]:
    """
    Parse Cellebrite Inseyets / Physical Analyzer image classifications.

    Examples:
    - "Armi (100%)"
    - "Pistola (100%); Armi (100%)"
    - "Fucile (91%) | Oggetto tenuto in mano (78%)"
    """
    text = safe_str(value).lower()
    if not text:
        return set()

    text = text.replace("_x000d_", "\n")
    text = text.replace("\\r", "\n").replace("\\n", "\n")

    parts = re.split(r"[;|,\r\n]+", text)
    labels: set[str] = set()

    for part in parts:
        cleaned = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", "", part)
        cleaned = cleaned.replace("_x000d_", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            labels.add(cleaned)

    return labels

# =============================================================================
# Raw export discovery and interactive menu
# =============================================================================

def prediction_export_name_for_tool(tool_name: str) -> str:
    if tool_name == "magnet_axiom":
        return "Pictures.csv"
    if tool_name == "excire_foto_2025":
        return "Excire prompt hit-list CSVs"
    return "prediction export"


def discover_raw_run_dirs(tool_name: str, forensic_tools_root: Path) -> list[Path]:
    """Discover immediate run directories under forensic_tools/<tool>/raw_exports/."""
    raw_dir = forensic_tools_root / tool_name / "raw_exports"
    if not raw_dir.exists():
        return []

    run_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if run_dirs:
        return run_dirs

    # Fallback for flat exports directly under raw_exports/.
    if any(path.is_file() for path in raw_dir.iterdir()):
        return [raw_dir]

    return []


def run_dir_has_prediction_files(tool_name: str, run_dir: Path) -> bool:
    return bool(discover_prediction_export_files_in_roots(tool_name, [run_dir]))


def format_run_dir_label(run_dir: Path, tool_name: str) -> str:
    prediction_files = discover_prediction_export_files_in_roots(tool_name, [run_dir])
    all_files = discover_export_files_in_roots([run_dir])
    return (
        f"{run_dir.name} "
        f"({len(prediction_files)} prediction file(s), {len(all_files)} export file(s))"
    )


def print_interactive_header() -> None:
    print("\n" + "=" * 78)
    print("FAIR-Lab forensic-tool output normalization")
    print("=" * 78)
    print("This menu selects which forensic-tool exports must be normalized.")
    print("The script does not modify datasets, bundle files, or raw exports.")
    print("It only creates normalized CSV outputs and metrics.\n")


def select_tool_from_menu(forensic_tools_root: Path) -> tuple[list[str], dict[str, list[Path]]]:
    """Interactive selection of tools and run directories."""
    available_tools = []
    for tool_name in KNOWN_TOOL_NAMES:
        run_dirs = discover_raw_run_dirs(tool_name, forensic_tools_root)
        available_tools.append((tool_name, run_dirs))

    print("Available forensic-tool export folders:")
    for index, (tool_name, run_dirs) in enumerate(available_tools, start=1):
        status = f"{len(run_dirs)} run folder(s)" if run_dirs else "no run folders found"
        print(f"  {index}. {tool_name} - {status}")
    print(f"  {len(available_tools) + 1}. All tools with available run folders")
    print(f"  {len(available_tools) + 2}. Manual CLI-style selection")

    choice = ask_index(
        "Select what to analyze",
        min_value=1,
        max_value=len(available_tools) + 2,
        default=1,
    )

    selected_run_dirs_by_tool: dict[str, list[Path]] = {}

    if choice == len(available_tools) + 1:
        tools = [tool_name for tool_name, run_dirs in available_tools if run_dirs]
        for tool_name, run_dirs in available_tools:
            if run_dirs:
                selected_run_dirs_by_tool[tool_name] = choose_run_dirs_for_tool(tool_name, run_dirs)
        return tools, selected_run_dirs_by_tool

    if choice == len(available_tools) + 2:
        tools_text = ask_text("Tool names separated by spaces", "magnet_axiom")
        tools = [item.strip() for item in tools_text.split() if item.strip()]
        return tools, selected_run_dirs_by_tool

    tool_name, run_dirs = available_tools[choice - 1]
    if not run_dirs:
        print(f"No run folders found for {tool_name}. The tool will still be selected.")
        return [tool_name], selected_run_dirs_by_tool

    selected_run_dirs_by_tool[tool_name] = choose_run_dirs_for_tool(tool_name, run_dirs)
    return [tool_name], selected_run_dirs_by_tool


def choose_run_dirs_for_tool(tool_name: str, run_dirs: list[Path]) -> list[Path]:
    """Interactive selection of one/latest/all run directories for a tool."""
    print(f"\nRun folders for {tool_name}:")
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"  {index}. {format_run_dir_label(run_dir, tool_name)}")

    print("\nRun selection mode:")
    print("  1. Use latest run folder only (recommended for thesis reporting)")
    print("  2. Choose one run folder")
    print("  3. Use all run folders and deduplicate by bundle_id")
    print("  4. Choose multiple run folders")

    mode = ask_index("Select run mode", 1, 4, default=1)

    if mode == 1:
        return [run_dirs[-1]]

    if mode == 2:
        selected = ask_index("Select run folder", 1, len(run_dirs), default=len(run_dirs))
        return [run_dirs[selected - 1]]

    if mode == 3:
        return run_dirs

    indexes_text = ask_text(
        "Enter run indexes separated by commas",
        ",".join(str(i) for i in range(1, len(run_dirs) + 1)),
    )
    selected_dirs: list[Path] = []
    for chunk in indexes_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            index = int(chunk)
        except ValueError:
            print(f"Skipping invalid index: {chunk}")
            continue
        if 1 <= index <= len(run_dirs):
            selected_dirs.append(run_dirs[index - 1])
        else:
            print(f"Skipping out-of-range index: {chunk}")

    return selected_dirs or [run_dirs[-1]]


def interactive_menu(args: argparse.Namespace) -> InteractiveSelection:
    """Initial interactive interface used by default."""
    forensic_tools_root = repo_relative_path(args.forensic_tools_root)
    print_interactive_header()

    tools, selected_run_dirs_by_tool = select_tool_from_menu(forensic_tools_root)

    print("\nProcessing options:")
    deduplicate = ask_yes_no("Deduplicate to one prediction per tool + bundle_id", default=not args.no_deduplicate)
    strict = ask_yes_no("Strict mode: fail if selected tool has no prediction exports", default=args.strict)

    output_dir_text = ask_text("Output directory", args.output_dir)
    metrics_dir_text = ask_text("Metrics directory", args.metrics_dir)

    print("\nSelection summary:")
    print(f"  Tools: {', '.join(tools) if tools else 'none'}")
    for tool_name, run_dirs in selected_run_dirs_by_tool.items():
        if run_dirs:
            print(f"  {tool_name} run folder(s):")
            for run_dir in run_dirs:
                print(f"    - {repo_relative_string(run_dir)}")
    print(f"  Deduplicate: {str(deduplicate).lower()}")
    print(f"  Strict: {str(strict).lower()}")
    proceed = ask_yes_no("Proceed with normalization", default=True)
    if not proceed:
        raise SystemExit("Normalization cancelled by user.")

    return InteractiveSelection(
        tools=tools,
        selected_run_dirs_by_tool=selected_run_dirs_by_tool,
        strict=strict,
        deduplicate=deduplicate,
        output_dir=repo_relative_path(output_dir_text),
        metrics_dir=repo_relative_path(metrics_dir_text),
    )


# =============================================================================
# Bundle loading and indexing
# =============================================================================

def load_bundle(bundle_manifest_path: Path, bundle_hashes_path: Path) -> pd.DataFrame:
    if not bundle_manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest_path}")

    bundle_df = pd.read_csv(bundle_manifest_path, dtype=str, keep_default_na=False)
    required_columns = {
        "bundle_id",
        "tool_input_filename",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
    }
    missing = required_columns - set(bundle_df.columns)
    if missing:
        raise ValueError(f"Bundle manifest is missing required columns: {sorted(missing)}")

    if bundle_hashes_path.exists():
        hashes_df = pd.read_csv(bundle_hashes_path, dtype=str, keep_default_na=False)
        if "bundle_id" in hashes_df.columns:
            hash_columns = [col for col in hashes_df.columns if col != "bundle_id"]
            bundle_df = bundle_df.merge(
                hashes_df[["bundle_id", *hash_columns]],
                on="bundle_id",
                how="left",
                suffixes=("", "_hashfile"),
            )

    bundle_df["_sha256_key"] = ""
    bundle_df["_md5_key"] = ""

    for col in ["sha256_actual", "sha256", "sha256_manifest", "sha256_hashfile"]:
        if col in bundle_df.columns:
            bundle_df["_sha256_key"] = bundle_df["_sha256_key"].mask(
                bundle_df["_sha256_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    for col in ["md5_actual", "md5", "md5_manifest", "md5_hashfile"]:
        if col in bundle_df.columns:
            bundle_df["_md5_key"] = bundle_df["_md5_key"].mask(
                bundle_df["_md5_key"].eq(""),
                bundle_df[col].map(normalize_hash),
            )

    bundle_df["_filename_key"] = bundle_df["tool_input_filename"].map(
        lambda x: basename_from_path(x).lower()
    )
    logging.info("Loaded bundle rows: %d", len(bundle_df))
    return bundle_df


def build_bundle_indexes(
    bundle_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sha_index: dict[str, dict[str, Any]] = {}
    md5_index: dict[str, dict[str, Any]] = {}
    filename_index: dict[str, dict[str, Any]] = {}

    for row in bundle_df.to_dict(orient="records"):
        sha = safe_str(row.get("_sha256_key", ""))
        md5 = safe_str(row.get("_md5_key", ""))
        filename = safe_str(row.get("_filename_key", ""))
        if sha:
            sha_index[sha] = row
        if md5:
            md5_index[md5] = row
        if filename:
            filename_index[filename] = row

    logging.info("Bundle SHA256 index entries: %d", len(sha_index))
    logging.info("Bundle MD5 index entries: %d", len(md5_index))
    logging.info("Bundle filename index entries: %d", len(filename_index))
    return sha_index, md5_index, filename_index


# =============================================================================
# Raw export discovery and parsing
# =============================================================================

def discover_prediction_export_files_in_roots(tool_name: str, roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if tool_name == "magnet_axiom":
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.name.lower() == "pictures.csv"
            )
        elif tool_name == "excire_foto_2025":
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and is_excire_prompt_export_file(path)
            )
        else:
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_GENERIC_EXPORT_EXTENSIONS
            )
    return sorted(set(files))


def discover_export_files_in_roots(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_GENERIC_EXPORT_EXTENSIONS
        )
    return sorted(set(files))


def get_scan_roots(tool_name: str, tool_dir: Path, selected_run_dirs: list[Path] | None) -> list[Path]:
    if selected_run_dirs:
        return selected_run_dirs
    return [tool_dir / "raw_exports"]


def discover_prediction_export_files(tool_name: str, tool_dir: Path, selected_run_dirs: list[Path] | None = None) -> list[Path]:
    return discover_prediction_export_files_in_roots(tool_name, get_scan_roots(tool_name, tool_dir, selected_run_dirs))


def discover_all_export_files(tool_dir: Path, selected_run_dirs: list[Path] | None = None) -> list[Path]:
    roots = selected_run_dirs if selected_run_dirs else [tool_dir / "raw_exports"]
    return discover_export_files_in_roots(roots)


def read_csv_like(path: Path) -> list[dict[str, Any]]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=encoding)
            return df.to_dict(orient="records")
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def flatten_json_object(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else safe_str(key)
            if isinstance(value, dict):
                flat.update(flatten_json_object(value, new_key))
            elif isinstance(value, list):
                flat[new_key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[new_key] = value
    elif isinstance(obj, list):
        flat[prefix or "value"] = json.dumps(obj, ensure_ascii=False)
    else:
        flat[prefix or "value"] = obj
    return flat


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [flatten_json_object(json.loads(line)) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return [flatten_json_object(item) for item in obj]
    if isinstance(obj, dict):
        for key in ("items", "artifacts", "files", "results", "records", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return [flatten_json_object(item) for item in value]
        return [flatten_json_object(obj)]
    return [{"value": safe_str(obj)}]


def read_txt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if line:
            records.append({"line_number": idx, "description": line})
    return records


def read_export_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return read_csv_like(path)
    if suffix in {".json", ".jsonl"}:
        return read_json_records(path)
    if suffix == ".txt":
        return read_txt_records(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="Immagini", dtype=str, header=1)
        df.columns = [safe_str(col) for col in df.columns]
        return df.fillna("").to_dict(orient="records")
    return []


def extract_raw_row(tool_name: str, export_file: Path, row_number: int, record: dict[str, Any]) -> RawToolRow:
    sha256 = normalize_hash(first_non_empty(record, SHA256_COLUMNS))
    md5 = normalize_hash(first_non_empty(record, MD5_COLUMNS))
    filename_or_path = first_non_empty(record, FILENAME_COLUMNS)
    raw_confidence = first_non_empty(record, CONFIDENCE_COLUMNS)

    if tool_name == "magnet_axiom":
        raw_label = first_non_empty(record, {"tags"})
    else:
        raw_label = collect_text_fields(record, LABEL_COLUMNS)
        if not raw_label:
            raw_label = " | ".join(safe_str(value) for value in record.values() if safe_str(value))[:1000]

    return RawToolRow(
        tool_name=tool_name,
        raw_export_file=repo_relative_string(export_file),
        raw_row_number=row_number,
        raw_record=record,
        sha256=sha256,
        md5=md5,
        filename_or_path=filename_or_path,
        raw_label=raw_label,
        raw_confidence=raw_confidence,
    )


def parse_tool_exports(
    tool_name: str,
    tool_dir: Path,
    selected_run_dirs: list[Path] | None = None,
) -> tuple[list[RawToolRow], list[dict[str, Any]]]:
    prediction_files = discover_prediction_export_files(tool_name, tool_dir, selected_run_dirs)
    all_export_files = discover_all_export_files(tool_dir, selected_run_dirs)
    prediction_file_set = {path.resolve() for path in prediction_files}

    raw_rows: list[RawToolRow] = []
    audit_rows: list[dict[str, Any]] = []

    for export_file in all_export_files:
        is_prediction_file = export_file.resolve() in prediction_file_set
        try:
            records = read_export_records(export_file)
            status = "parsed_prediction_file" if is_prediction_file else "parsed_audit_only"
            error = ""
        except Exception as exc:
            records = []
            status = "parse_error"
            error = f"{type(exc).__name__}: {exc}"
            logging.warning("Could not parse %s: %s", export_file, error)

        if is_prediction_file:
            for idx, record in enumerate(records, start=1):
                raw_rows.append(extract_raw_row(tool_name, export_file, idx, record))

        audit_rows.append(
            {
                "tool_name": tool_name,
                "raw_export_file": repo_relative_string(export_file),
                "run_dir": infer_run_dir_name(tool_dir, export_file),
                "extension": export_file.suffix.lower(),
                "is_prediction_file": str(is_prediction_file).lower(),
                "status": status,
                "parsed_rows": len(records),
                "error": error,
            }
        )

    return raw_rows, audit_rows


def infer_run_dir_name(tool_dir: Path, export_file: Path) -> str:
    raw_dir = tool_dir / "raw_exports"
    try:
        relative = export_file.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        return ""
    return relative.parts[0] if len(relative.parts) > 1 else "raw_exports"


# =============================================================================
# Tool version log
# =============================================================================

def extract_version_fields_from_summary(summary_path: Path) -> dict[str, str]:
    try:
        records = read_json_records(summary_path)
    except Exception:
        return {}
    if not records:
        return {}

    normalized_items = {normalize_column_name(k): safe_str(v) for k, v in records[0].items()}

    def find_value(patterns: list[str]) -> str:
        for key, value in normalized_items.items():
            if value and any(pattern in key for pattern in patterns):
                return value
        return ""

    return {
        "tool_version": find_value(["version", "application_version", "axiom_version"]),
        "tool_build": find_value(["build", "build_number"]),
        "case_name": find_value(["case", "case_name"]),
        "export_status": find_value(["status", "export_status"]),
        "export_timestamp": find_value(["timestamp", "created", "exported", "date"]),
    }


def build_tool_version_row(
    tool_name: str,
    tool_dir: Path,
    export_files_found: int,
    selected_run_dirs: list[Path] | None = None,
    row_tool_name: str | None = None,
) -> dict[str, Any]:
    roots = selected_run_dirs if selected_run_dirs else [tool_dir / "raw_exports"]
    summary_files: list[Path] = []
    for root in roots:
        if root.exists():
            summary_files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.name.lower() in {"exportsummary.json", "export_summary.json"}
            )
    summary_files = sorted(summary_files)

    extracted: dict[str, str] = {}
    summary_file = ""
    if summary_files:
        summary_file = repo_relative_string(summary_files[0])
        extracted = extract_version_fields_from_summary(summary_files[0])

    if tool_name == "magnet_axiom":
        notes = (
            "Magnet AXIOM / Magnet.AI export. Predictions are derived from Pictures.csv "
            "Tags; Tags='Possible weapons' is mapped to weapon_detected=true; empty Tags "
            "is mapped to weapon_detected=false."
        )

    elif tool_name == "excire_foto_2025":
        notes = (
            "Excire Foto 2025 semantic retrieval export. Predictions are derived from "
            "fixed firearm-oriented prompt hit-list CSVs. An image retrieved by at least "
            "one prompt is mapped to weapon_detected=true; all remaining bundle images "
            "are completed as weapon_detected=false."
        )

    elif tool_name == "cellebrite_inseyets":
        notes = (
            "Cellebrite Inseyets 10.9 / Physical Analyzer image-classification export. "
            "Predictions are derived from the Excel report sheet 'Immagini' and the "
            "observable 'Classifications' column. The extended mapping treats an image "
            "as weapon_detected=true when Classifications contains at least one among "
            "'Armi', 'Pistola', or 'Fucile'. This is an operational recoding of exported "
            "tool output and does not imply access to Cellebrite internal AI model logic."
        )

    else:
        notes = "Unsupported or non-final tool. Fill manually only if intentionally used."

    if tool_name == "cellebrite_inseyets":
        extracted.setdefault("tool_version", "Cellebrite Inseyets 10.9")
        extracted.setdefault("tool_build", "Physical Analyzer 10.9.0.3029 / UFED 10.9.0.284")
        extracted.setdefault("case_name", "CHIAVETTA USB")
        extracted.setdefault("export_status", "completed")
        extracted.setdefault("export_timestamp", "2026-06-05 13:04:59")
        extracted.setdefault("ai_modules_enabled", "media classifications / image classifications")

    return {
        "tool_name": row_tool_name or tool_name,
        "tool_version": extracted.get("tool_version", ""),
        "tool_build": extracted.get("tool_build", ""),
        "case_name": extracted.get("case_name", ""),
        "export_status": extracted.get("export_status", ""),
        "export_timestamp": extracted.get("export_timestamp", ""),
        "summary_file": summary_file,
        "selected_run_dirs": unique_join([repo_relative_string(path) for path in (selected_run_dirs or [])]),
        "ai_modules_enabled": extracted.get("ai_modules_enabled", ""),
        "os_environment": "",
        "import_path": "datasets/forensic_evaluation_bundle/blind_tool_input/files/",
        "export_files_found": export_files_found,
        "notes": notes,
        "created_at": utc_now_iso(),
    }


# =============================================================================
# Prediction interpretation and matching
# =============================================================================

def interpret_weapon_detection(tool_name: str, raw_label: str) -> tuple[str, str]:
    label = safe_str(raw_label)
    text_clean = re.sub(r"[^a-z0-9_ +/.-]+", " ", label.lower())

    if tool_name == "magnet_axiom":
        if "possible weapons" in text_clean:
            return "true", "magnet_axiom_tag:possible_weapons"
        if not text_clean:
            return "false", "magnet_axiom_empty_tags:not_flagged"
        if "weapon" in text_clean or "weapons" in text_clean:
            return "true", "magnet_axiom_tag:weapon_keyword"
        return "unknown", "magnet_axiom_unmapped_tag"

    if tool_name == "cellebrite_inseyets":
        labels = parse_cellebrite_classifications(raw_label)

        weapon_terms = CELLEBRITE_WEAPON_CLASSIFICATIONS_EXTENDED

        if labels.intersection(weapon_terms):
            return "true", "cellebrite_extended_armi_pistola_fucile"

        return "false", "cellebrite_extended_armi_pistola_fucile"

    if not text_clean:
        return "unknown", "empty_label"

    for negative in NON_WEAPON_NEGATIVE_KEYWORDS:
        if negative in text_clean:
            return "false", f"negative_keyword:{negative}"

    for keyword in WEAPON_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_clean):
            return "true", f"weapon_keyword:{keyword}"

    for keyword in OOD_HINT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text_clean):
            return "unknown", f"ood_hint:{keyword}"

    return "unknown", "no_mapping_rule"


def match_bundle_row(
    raw_row: RawToolRow,
    sha_index: dict[str, dict[str, Any]],
    md5_index: dict[str, dict[str, Any]],
    filename_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if raw_row.sha256 and raw_row.sha256 in sha_index:
        return sha_index[raw_row.sha256], "sha256"
    if raw_row.md5 and raw_row.md5 in md5_index:
        return md5_index[raw_row.md5], "md5"

    filename = basename_from_path(raw_row.filename_or_path).lower()
    if filename and filename in filename_index:
        return filename_index[filename], "filename"

    text = f"{raw_row.filename_or_path} {raw_row.raw_label}".lower()
    for filename_key, row in filename_index.items():
        if filename_key and filename_key in text:
            return row, "filename_embedded"

    return None, "unmatched"


def compute_correctness(final_label: str, weapon_detected: str) -> tuple[str, str, str]:
    label = safe_str(final_label).lower()
    if weapon_detected not in {"true", "false"}:
        return "", "", ""

    detected = weapon_detected == "true"
    if label == "weapon":
        return str(detected).lower(), str(not detected).lower(), "false"
    if label == "non_weapon":
        return str(not detected).lower(), "false", str(detected).lower()
    return "", "", ""


def build_base_match_fields(raw_row: RawToolRow, bundle_row: dict[str, Any] | None, match_method: str) -> dict[str, Any]:
    if bundle_row is None:
        return {
            "tool_name": raw_row.tool_name,
            "bundle_id": "",
            "match_method": match_method,
            "matched": "false",
            "tool_input_filename": basename_from_path(raw_row.filename_or_path),
            "sha256": raw_row.sha256,
            "md5": raw_row.md5,
            "sample_type": "",
            "attack_family": "",
            "attack_name": "",
            "attack_target_model": "",
            "fold": "",
            "final_label": "",
            "source_dataset": "",
            "original_image_id": "",
            "generated_image_id": "",
        }

    return {
        "tool_name": raw_row.tool_name,
        "bundle_id": safe_str(bundle_row.get("bundle_id", "")),
        "match_method": match_method,
        "matched": "true",
        "tool_input_filename": safe_str(bundle_row.get("tool_input_filename", "")),
        "sha256": safe_str(bundle_row.get("_sha256_key", raw_row.sha256)),
        "md5": safe_str(bundle_row.get("_md5_key", raw_row.md5)),
        "sample_type": safe_str(bundle_row.get("sample_type", "")),
        "attack_family": safe_str(bundle_row.get("attack_family", "")),
        "attack_name": safe_str(bundle_row.get("attack_name", "")),
        "attack_target_model": safe_str(bundle_row.get("attack_target_model", "")),
        "fold": safe_str(bundle_row.get("fold", "")),
        "final_label": safe_str(bundle_row.get("final_label", "")),
        "source_dataset": safe_str(bundle_row.get("source_dataset", "")),
        "original_image_id": safe_str(bundle_row.get("original_image_id", "")),
        "generated_image_id": safe_str(bundle_row.get("generated_image_id", "")),
    }


# =============================================================================
# X-Ways / Excire Photo AI semantic prompt-hit normalization
# =============================================================================

def is_excire_prompt_export_file(path: Path) -> bool:
    """Return True for fixed Excire Photo AI prompt hit-list CSV exports."""
    match = EXCIRE_PROMPT_EXPORT_RE.match(path.name)
    if not match:
        return False
    return match.group("prompt").lower() in EXCIRE_FIREARM_PROMPTS


def infer_excire_prompt_and_distance(path: Path) -> tuple[str, str] | None:
    """Infer semantic prompt and distance limit from an Excire export filename."""
    match = EXCIRE_PROMPT_EXPORT_RE.match(path.name)
    if not match:
        return None
    prompt = match.group("prompt").lower()
    if prompt not in EXCIRE_FIREARM_PROMPTS:
        return None
    return prompt, match.group("distance")


def read_excire_hit_list(path: Path) -> list[str]:
    """
    Read an Excire Photo AI prompt export.

    Excire exports used in this pipeline are plain hit lists: one image path per
    line, usually without a header. This reader intentionally avoids pandas CSV
    inference because quoted Windows paths and headerless files must be handled
    deterministically.
    """
    hits: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line:
            continue
        if line.lower() in {"path", "file", "filename", "file_path", "filepath"}:
            continue
        hits.append(line)
    return hits


def normalize_xways_excire_prompt_exports(
    tool_name: str,
    tool_dir: Path,
    selected_run_dirs: list[Path] | None,
    bundle_df: pd.DataFrame,
    effective_tool_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Normalize X-Ways / Excire Photo AI prompt hit-list exports.

    The Excire workflow used for this thesis exports only retrieved images for a
    given semantic prompt. Therefore, absence from all fixed prompt hit lists is
    an operational negative prediction. This function completes the prediction
    table to one row per bundle image so that accuracy, FNR/FPR and OOD weapon
    flag rate can be computed consistently with Magnet AXIOM.
    """
    output_tool_name = effective_tool_name or tool_name
    roots = get_scan_roots(tool_name, tool_dir, selected_run_dirs)
    all_export_files = discover_export_files_in_roots(roots)
    prediction_files = [path for path in all_export_files if is_excire_prompt_export_file(path)]

    hits_by_prompt: dict[str, dict[str, dict[str, Any]]] = {prompt: {} for prompt in EXCIRE_FIREARM_PROMPTS}
    distance_values: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    raw_hit_rows = 0

    prediction_file_set = {path.resolve() for path in prediction_files}

    for export_file in all_export_files:
        is_prediction_file = export_file.resolve() in prediction_file_set
        status = "parsed_prediction_file" if is_prediction_file else "parsed_audit_only"
        error = ""
        parsed_rows = 0

        if is_prediction_file:
            inferred = infer_excire_prompt_and_distance(export_file)
            if inferred is None:
                status = "parse_error"
                error = "Could not infer Excire prompt/distance from filename."
            else:
                prompt, distance = inferred
                distance_values.add(distance)
                try:
                    rows = read_excire_hit_list(export_file)
                    parsed_rows = len(rows)
                    raw_hit_rows += parsed_rows
                    for original_path in rows:
                        filename = basename_from_path(original_path)
                        key = filename.lower()
                        if not key:
                            continue
                        hits_by_prompt[prompt][key] = {
                            "filename": filename,
                            "example_original_path": original_path,
                            "raw_export_file": repo_relative_string(export_file),
                        }
                except Exception as exc:  # pragma: no cover - defensive IO branch
                    status = "parse_error"
                    error = f"{type(exc).__name__}: {exc}"
                    logging.warning("Could not parse Excire export %s: %s", export_file, error)
        else:
            try:
                if export_file.suffix.lower() == ".csv":
                    parsed_rows = len(read_excire_hit_list(export_file))
                else:
                    parsed_rows = len(read_export_records(export_file))
            except Exception as exc:  # pragma: no cover - audit-only branch
                status = "parse_error"
                error = f"{type(exc).__name__}: {exc}"

        audit_rows.append(
            {
                "tool_name": output_tool_name,
                "raw_export_file": repo_relative_string(export_file),
                "run_dir": infer_run_dir_name(tool_dir, export_file),
                "extension": export_file.suffix.lower(),
                "is_prediction_file": str(is_prediction_file).lower(),
                "status": status,
                "parsed_rows": parsed_rows,
                "error": error,
            }
        )

    distance_limit = unique_join(sorted(distance_values), separator="|")
    normalized_rows: list[dict[str, Any]] = []

    for bundle_row in bundle_df.to_dict(orient="records"):
        filename_key = safe_str(bundle_row.get("_filename_key", ""))
        prompt_hits = {
            prompt: int(bool(filename_key) and filename_key in hits_by_prompt[prompt])
            for prompt in EXCIRE_FIREARM_PROMPTS
        }
        hit_prompts = [prompt for prompt in EXCIRE_FIREARM_PROMPTS if prompt_hits[prompt] == 1]
        weapon_detected = "true" if hit_prompts else "false"
        normalized_prediction = "weapon" if weapon_detected == "true" else "non_weapon"

        hit_export_files = [
            hits_by_prompt[prompt][filename_key]["raw_export_file"]
            for prompt in hit_prompts
            if filename_key in hits_by_prompt[prompt]
        ]
        hit_original_paths = [
            hits_by_prompt[prompt][filename_key]["example_original_path"]
            for prompt in hit_prompts
            if filename_key in hits_by_prompt[prompt]
        ]

        base = {
            "tool_name": output_tool_name,
            "bundle_id": safe_str(bundle_row.get("bundle_id", "")),
            "match_method": "bundle_manifest_completion",
            "matched": "true",
            "tool_input_filename": safe_str(bundle_row.get("tool_input_filename", "")),
            "sha256": safe_str(bundle_row.get("_sha256_key", "")),
            "md5": safe_str(bundle_row.get("_md5_key", "")),
            "sample_type": safe_str(bundle_row.get("sample_type", "")),
            "attack_family": safe_str(bundle_row.get("attack_family", "")),
            "attack_name": safe_str(bundle_row.get("attack_name", "")),
            "attack_target_model": safe_str(bundle_row.get("attack_target_model", "")),
            "fold": safe_str(bundle_row.get("fold", "")),
            "final_label": safe_str(bundle_row.get("final_label", "")),
            "source_dataset": safe_str(bundle_row.get("source_dataset", "")),
            "original_image_id": safe_str(bundle_row.get("original_image_id", "")),
            "generated_image_id": safe_str(bundle_row.get("generated_image_id", "")),
        }
        correct, false_negative, false_positive = compute_correctness(base["final_label"], weapon_detected)

        normalized_rows.append(
            {
                **base,
                "raw_export_file": unique_join(hit_export_files),
                "raw_row_number": "",
                "raw_filename_or_path": unique_join(hit_original_paths) or base["tool_input_filename"],
                "tool_raw_label": unique_join(hit_prompts),
                "tool_raw_confidence": "",
                "tool_confidence_numeric": "",
                "weapon_detected": weapon_detected,
                "normalized_prediction": normalized_prediction,
                "mapping_reason": (
                    f"xways_excire_semantic_prompt_hit:{unique_join(hit_prompts, separator='|')}"
                    if hit_prompts
                    else "xways_excire_semantic_prompt_no_hit"
                ),
                "correct": correct,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "raw_row_count_after_deduplication": max(len(hit_prompts), 1),
                "excire_distance_limit": distance_limit,
                "excire_promptset": unique_join(list(EXCIRE_FIREARM_PROMPTS), separator="|"),
                "n_prompt_hits": len(hit_prompts),
                "hit_prompts": unique_join(hit_prompts, separator="|"),
                **{f"prompt_{prompt}_hit": prompt_hits[prompt] for prompt in EXCIRE_FIREARM_PROMPTS},
            }
        )

    logging.info(
        "Tool %s: completed %d bundle-level semantic predictions from %d raw prompt hits.",
        tool_name,
        len(normalized_rows),
        raw_hit_rows,
    )
    return normalized_rows, audit_rows, raw_hit_rows


def normalize_rows(
    raw_rows: list[RawToolRow],
    sha_index: dict[str, dict[str, Any]],
    md5_index: dict[str, dict[str, Any]],
    filename_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        bundle_row, match_method = match_bundle_row(raw_row, sha_index, md5_index, filename_index)
        weapon_detected, mapping_reason = interpret_weapon_detection(raw_row.tool_name, raw_row.raw_label)
        confidence = parse_float(raw_row.raw_confidence)
        base = build_base_match_fields(raw_row, bundle_row, match_method)
        correct, false_negative, false_positive = compute_correctness(base["final_label"], weapon_detected)

        normalized_rows.append(
            {
                **base,
                "raw_export_file": raw_row.raw_export_file,
                "raw_row_number": raw_row.raw_row_number,
                "raw_filename_or_path": raw_row.filename_or_path,
                "tool_raw_label": raw_row.raw_label,
                "tool_raw_confidence": raw_row.raw_confidence,
                "tool_confidence_numeric": "" if confidence is None else confidence,
                "weapon_detected": weapon_detected,
                "normalized_prediction": (
                    "weapon" if weapon_detected == "true" else "non_weapon" if weapon_detected == "false" else "unknown"
                ),
                "mapping_reason": mapping_reason,
                "correct": correct,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "raw_row_count_after_deduplication": 1,
            }
        )
    return normalized_rows


# =============================================================================
# Deduplication and metrics
# =============================================================================

def choose_consolidated_detection(rows: list[dict[str, Any]]) -> tuple[str, str]:
    detections = [safe_str(row.get("weapon_detected", "")) for row in rows]
    if "true" in detections:
        return "true", "deduplicated:any_positive"
    if "false" in detections:
        return "false", "deduplicated:all_not_flagged_or_no_positive"
    return "unknown", "deduplicated:all_unknown"


def consolidate_matched_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    representative = rows[0].copy()
    weapon_detected, consolidation_reason = choose_consolidated_detection(rows)
    representative["weapon_detected"] = weapon_detected
    representative["normalized_prediction"] = "weapon" if weapon_detected == "true" else "non_weapon" if weapon_detected == "false" else "unknown"
    representative["mapping_reason"] = unique_join([consolidation_reason, *[row.get("mapping_reason", "") for row in rows]])
    representative["raw_export_file"] = unique_join([row.get("raw_export_file", "") for row in rows])
    representative["raw_row_number"] = unique_join([row.get("raw_row_number", "") for row in rows])
    representative["tool_raw_label"] = unique_join([row.get("tool_raw_label", "") for row in rows])
    representative["tool_raw_confidence"] = unique_join([row.get("tool_raw_confidence", "") for row in rows])

    numeric_confidences = [
        parse_float(row.get("tool_confidence_numeric", ""))
        for row in rows
        if parse_float(row.get("tool_confidence_numeric", "")) is not None
    ]
    representative["tool_confidence_numeric"] = max(numeric_confidences) if numeric_confidences else ""
    representative["raw_row_count_after_deduplication"] = len(rows)

    correct, false_negative, false_positive = compute_correctness(representative.get("final_label", ""), weapon_detected)
    representative["correct"] = correct
    representative["false_negative"] = false_negative
    representative["false_positive"] = false_positive
    return representative


def deduplicate_predictions(normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate normalized predictions to one row per matched tool/bundle item.

    Unmatched rows are intentionally excluded from the normalized prediction
    table because they do not belong to the official forensic evaluation bundle.
    They remain documented through the export audit and raw parsed-row counts.
    """
    matched_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in normalized_rows:
        if safe_str(row.get("matched", "")) == "true" and safe_str(row.get("bundle_id", "")):
            matched_groups[
                (
                    safe_str(row.get("tool_name", "")),
                    safe_str(row.get("bundle_id", "")),
                )
            ].append(row)

    deduplicated_rows = [
        consolidate_matched_rows(group)
        for _, group in sorted(matched_groups.items())
    ]

    return deduplicated_rows


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def metric_value(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [row for row in rows if safe_str(row.get("matched", "")) == "true"]
    binary_rows = [row for row in matched_rows if safe_str(row.get("final_label", "")).lower() in {"weapon", "non_weapon"}]
    binary_interpretable_rows = [row for row in binary_rows if safe_str(row.get("weapon_detected", "")) in {"true", "false"}]

    tp = fp = tn = fn = 0
    for row in binary_interpretable_rows:
        label = safe_str(row.get("final_label", "")).lower()
        detected = safe_str(row.get("weapon_detected", "")).lower() == "true"
        if label == "weapon" and detected:
            tp += 1
        elif label == "weapon" and not detected:
            fn += 1
        elif label == "non_weapon" and detected:
            fp += 1
        elif label == "non_weapon" and not detected:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None

    ood_rows = [row for row in matched_rows if safe_str(row.get("final_label", "")).lower() == "ood"]
    ood_weapon_flags = sum(1 for row in ood_rows if safe_str(row.get("weapon_detected", "")) == "true")
    ood_non_weapon_flags = sum(1 for row in ood_rows if safe_str(row.get("weapon_detected", "")) == "false")
    ood_unknown = sum(1 for row in ood_rows if safe_str(row.get("weapon_detected", "")) == "unknown")
    unknown_rows = sum(1 for row in matched_rows if safe_str(row.get("weapon_detected", "")) == "unknown")

    return {
        "rows_total": len(rows),
        "matched_rows": len(matched_rows),
        "binary_rows": len(binary_rows),
        "binary_interpretable_rows": len(binary_interpretable_rows),
        "unknown_rows": unknown_rows,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": metric_value(accuracy),
        "balanced_accuracy": metric_value(balanced_accuracy),
        "precision_weapon": metric_value(precision),
        "recall_weapon": metric_value(recall),
        "specificity_non_weapon": metric_value(specificity),
        "f1_weapon": metric_value(f1),
        "false_negative_rate": metric_value(safe_div(fn, tp + fn)),
        "false_positive_rate": metric_value(safe_div(fp, fp + tn)),
        "ood_rows": len(ood_rows),
        "ood_weapon_flags": ood_weapon_flags,
        "ood_non_weapon_flags": ood_non_weapon_flags,
        "ood_unknown": ood_unknown,
        "ood_weapon_flag_rate": metric_value(safe_div(ood_weapon_flags, len(ood_rows))),
    }


def add_metric_group(
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]],
    row: dict[str, Any],
    scope: str,
    sample_type: str,
    attack_family: str,
    attack_name: str,
) -> None:
    key = (
        safe_str(row.get("tool_name", "")),
        scope,
        sample_type or "all",
        attack_family or "all",
        attack_name or "all",
    )
    groups[key].append(row)


def compute_metrics(normalized_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        if safe_str(row.get("matched", "")) != "true":
            continue
        sample_type = safe_str(row.get("sample_type", "")) or "none"
        attack_family = safe_str(row.get("attack_family", "")) or "none"
        attack_name = safe_str(row.get("attack_name", "")) or "none"
        add_metric_group(groups, row, "all", "all", "all", "all")
        add_metric_group(groups, row, "sample_type", sample_type, "all", "all")
        add_metric_group(groups, row, "attack_family", "all", attack_family, "all")
        add_metric_group(groups, row, "attack_name", "all", attack_family, attack_name)
        add_metric_group(groups, row, "sample_type_attack", sample_type, attack_family, attack_name)

    scope_order = {"all": 0, "sample_type": 1, "attack_family": 2, "attack_name": 3, "sample_type_attack": 4}
    metric_rows: list[dict[str, Any]] = []
    for (tool_name, scope, sample_type, attack_family, attack_name), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], scope_order.get(item[0][1], 99), item[0][2], item[0][3], item[0][4])
    ):
        metric_rows.append(
            {
                "tool_name": tool_name,
                "scope": scope,
                "sample_type": sample_type,
                "attack_family": attack_family,
                "attack_name": attack_name,
                **compute_group_metrics(rows),
            }
        )
    return metric_rows


# =============================================================================
# Output writing
# =============================================================================

def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def build_selected_run_dirs_from_cli(selected_run_dirs: list[str]) -> dict[str, list[Path]]:
    selected_by_tool: dict[str, list[Path]] = defaultdict(list)
    for path_text in selected_run_dirs:
        run_dir = repo_relative_path(path_text)
        parts = run_dir.parts
        matched_tool = ""
        for tool_name in KNOWN_TOOL_NAMES:
            if tool_name in parts:
                matched_tool = tool_name
                break
        if matched_tool:
            selected_by_tool[matched_tool].append(run_dir)
    return dict(selected_by_tool)


def cleanup_legacy_excire_outputs(output_dir: Path, metrics_dir: Path, output_tool_names: list[str]) -> list[str]:
    """Remove stale legacy Excire aggregate files when D20/D50/D80 variants are produced.

    Earlier versions of the pipeline produced xways_excire_metrics.csv and
    xways_excire_normalized_predictions.csv for a single Excire setting. Once
    the sensitivity-analysis pipeline is used, those files become ambiguous and
    can lead to wrong reporting if left in the repository.
    """
    has_excire_variants = any(name.startswith("xways_excire_d") for name in output_tool_names)
    if not has_excire_variants:
        return []

    stale_paths = [
        output_dir / "xways_excire_normalized_predictions.csv",
        metrics_dir / "xways_excire_metrics.csv",
    ]
    removed: list[str] = []
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()
            removed.append(repo_relative_string(stale_path))
            logging.info("Removed stale legacy Excire output: %s", repo_relative_string(stale_path))
    return removed


def validate_metric_outputs(
    metrics_rows: list[dict[str, Any]],
    metrics_dir: Path,
    forensic_tool_metrics_path: Path,
    output_tool_names: list[str],
) -> dict[str, Any]:
    """Validate that the aggregate metric CSV mirrors the per-tool metric CSVs.

    The aggregate forensic_tools_metrics.csv is the canonical file consumed by
    the reporting script. This check makes the workflow reproducible by ensuring
    that the aggregate file is generated from the same in-memory rows used to
    create the per-tool metrics.
    """
    rows_by_tool: dict[str, int] = defaultdict(int)
    for row in metrics_rows:
        tool_name = safe_str(row.get("tool_name", ""))
        if tool_name:
            rows_by_tool[tool_name] += 1

    per_tool_files: dict[str, dict[str, Any]] = {}
    for tool_name in output_tool_names:
        metric_file = metrics_dir / f"{tool_name}_metrics.csv"
        exists = metric_file.exists()
        row_count = 0
        if exists:
            try:
                row_count = len(pd.read_csv(metric_file))
            except Exception as exc:  # pragma: no cover - defensive branch
                per_tool_files[tool_name] = {
                    "path": repo_relative_string(metric_file),
                    "exists": True,
                    "rows": "",
                    "expected_rows": rows_by_tool.get(tool_name, 0),
                    "matches_expected": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
        per_tool_files[tool_name] = {
            "path": repo_relative_string(metric_file),
            "exists": exists,
            "rows": row_count,
            "expected_rows": rows_by_tool.get(tool_name, 0),
            "matches_expected": bool(exists and row_count == rows_by_tool.get(tool_name, 0)),
            "error": "",
        }

    aggregate_exists = forensic_tool_metrics_path.exists()
    aggregate_rows = 0
    aggregate_tool_counts: dict[str, int] = {}
    aggregate_error = ""
    if aggregate_exists:
        try:
            aggregate_df = pd.read_csv(forensic_tool_metrics_path)
            aggregate_rows = len(aggregate_df)
            if "tool_name" in aggregate_df.columns:
                aggregate_tool_counts = {
                    str(key): int(value)
                    for key, value in aggregate_df["tool_name"].astype(str).value_counts().sort_index().items()
                }
        except Exception as exc:  # pragma: no cover - defensive branch
            aggregate_error = f"{type(exc).__name__}: {exc}"

    expected_tool_counts = {key: int(value) for key, value in sorted(rows_by_tool.items())}
    aggregate_matches_memory = (
        aggregate_exists
        and not aggregate_error
        and aggregate_rows == len(metrics_rows)
        and aggregate_tool_counts == expected_tool_counts
    )
    per_tool_files_match_memory = all(item["matches_expected"] for item in per_tool_files.values())

    validation = {
        "aggregate_path": repo_relative_string(forensic_tool_metrics_path),
        "aggregate_exists": aggregate_exists,
        "aggregate_rows": aggregate_rows,
        "expected_aggregate_rows": len(metrics_rows),
        "aggregate_tool_counts": aggregate_tool_counts,
        "expected_tool_counts": expected_tool_counts,
        "aggregate_matches_memory": aggregate_matches_memory,
        "aggregate_error": aggregate_error,
        "per_tool_files": per_tool_files,
        "per_tool_files_match_memory": per_tool_files_match_memory,
        "all_metric_outputs_consistent": bool(aggregate_matches_memory and per_tool_files_match_memory),
    }

    if not validation["all_metric_outputs_consistent"]:
        logging.warning("Metric output validation did not pass. Check normalization_summary.json.")
    else:
        logging.info("Metric output validation passed: aggregate and per-tool metrics are consistent.")

    return validation


def infer_excire_distance_from_run_dir(run_dir: Path) -> str:
    """Infer the Excire distance value from a run directory name."""
    match = re.search(r"d(\d+)", run_dir.name, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def excire_variant_tool_name(base_tool_name: str, run_dir: Path) -> str:
    """Build a stable tool identifier for one Excire distance setting."""
    distance = infer_excire_distance_from_run_dir(run_dir)
    if distance:
        return f"{base_tool_name}_d{distance}"
    return f"{base_tool_name}_{run_dir.name.lower()}"


def resolve_excire_run_dirs(
    tool_name: str,
    tool_dir: Path,
    selected_run_dirs: list[Path],
) -> list[Path]:
    """
    Return the Excire run directories that must be normalized separately.

    Each EXCIRE_Dxx_FIREARM_PROMPTS folder is a distinct operational setting.
    They must not be merged, because D20, D50 and D80 represent different
    semantic-distance policies.
    """
    if selected_run_dirs:
        return selected_run_dirs
    return discover_raw_run_dirs(tool_name, tool_dir.parent)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    bundle_manifest_path = repo_relative_path(args.bundle_manifest)
    bundle_hashes_path = repo_relative_path(args.bundle_hashes)
    forensic_tools_root = repo_relative_path(args.forensic_tools_root)

    if args.no_interactive:
        tools = args.tools
        selected_run_dirs_by_tool = build_selected_run_dirs_from_cli(args.selected_run_dir)
        output_dir = repo_relative_path(args.output_dir)
        metrics_dir = repo_relative_path(args.metrics_dir)
        strict = args.strict
        deduplicate = not args.no_deduplicate
    else:
        selection = interactive_menu(args)
        tools = selection.tools
        selected_run_dirs_by_tool = selection.selected_run_dirs_by_tool
        output_dir = selection.output_dir
        metrics_dir = selection.metrics_dir
        strict = selection.strict
        deduplicate = selection.deduplicate

    normalized_predictions_path = output_dir / "normalized_predictions.csv"
    export_audit_path = output_dir / "tool_export_audit.csv"
    tool_version_log_path = output_dir / "tool_version_log.csv"
    normalization_summary_path = output_dir / "normalization_summary.json"
    forensic_tool_metrics_path = metrics_dir / "forensic_tools_metrics.csv"

    logging.info("Script: %s", SCRIPT_NAME)
    logging.info("Repository root: %s", REPO_ROOT)
    logging.info("Forensic tools root: %s", forensic_tools_root)
    logging.info("Selected tools: %s", tools)

    bundle_df = load_bundle(bundle_manifest_path, bundle_hashes_path)
    sha_index, md5_index, filename_index = build_bundle_indexes(bundle_df)

    all_raw_rows: list[RawToolRow] = []
    pre_normalized_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []
    per_tool_prediction_export_counts: dict[str, int] = {}
    per_tool_total_export_counts: dict[str, int] = {}
    per_tool_raw_row_counts: dict[str, int] = {}

    for tool_name in tools:
        tool_dir = forensic_tools_root / tool_name
        selected_run_dirs = selected_run_dirs_by_tool.get(tool_name, [])
        prediction_files = discover_prediction_export_files(tool_name, tool_dir, selected_run_dirs or None)
        all_export_files = discover_all_export_files(tool_dir, selected_run_dirs or None)

        logging.info(
            "Tool %s: found %d prediction export file(s), %d total export file(s)",
            tool_name,
            len(prediction_files),
            len(all_export_files),
        )

        if strict and not prediction_files:
            raise FileNotFoundError(
                f"No prediction export files found for {tool_name}. "
                f"Selected run dirs: {[repo_relative_string(path) for path in selected_run_dirs]}"
            )

        if tool_name == "excire_foto_2025":
            excire_run_dirs = resolve_excire_run_dirs(tool_name, tool_dir, selected_run_dirs)
            if strict and not excire_run_dirs:
                raise FileNotFoundError(f"No Excire run folders found under {tool_dir / 'raw_exports'}")

            for run_dir in excire_run_dirs:
                effective_tool_name = excire_variant_tool_name(tool_name, run_dir)
                run_prediction_files = discover_prediction_export_files(tool_name, tool_dir, [run_dir])
                run_all_export_files = discover_all_export_files(tool_dir, [run_dir])

                if strict and not run_prediction_files:
                    raise FileNotFoundError(
                        f"No Excire prediction export files found for {effective_tool_name}. "
                        f"Selected run dir: {repo_relative_string(run_dir)}"
                    )

                tool_normalized_rows, tool_audit_rows, raw_hit_rows = normalize_xways_excire_prompt_exports(
                    tool_name=tool_name,
                    tool_dir=tool_dir,
                    selected_run_dirs=[run_dir],
                    bundle_df=bundle_df,
                    effective_tool_name=effective_tool_name,
                )
                pre_normalized_rows.extend(tool_normalized_rows)
                audit_rows.extend(tool_audit_rows)

                per_tool_prediction_export_counts[effective_tool_name] = len(run_prediction_files)
                per_tool_total_export_counts[effective_tool_name] = len(run_all_export_files)
                per_tool_raw_row_counts[effective_tool_name] = raw_hit_rows

                version_rows.append(
                    build_tool_version_row(
                        tool_name=tool_name,
                        tool_dir=tool_dir,
                        export_files_found=len(run_all_export_files),
                        selected_run_dirs=[run_dir],
                        row_tool_name=effective_tool_name,
                    )
                )
        else:
            raw_rows, tool_audit_rows = parse_tool_exports(tool_name, tool_dir, selected_run_dirs or None)
            all_raw_rows.extend(raw_rows)
            audit_rows.extend(tool_audit_rows)
            per_tool_raw_row_counts[tool_name] = len(raw_rows)
            per_tool_prediction_export_counts[tool_name] = len(prediction_files)
            per_tool_total_export_counts[tool_name] = len(all_export_files)

            version_rows.append(
                build_tool_version_row(
                    tool_name=tool_name,
                    tool_dir=tool_dir,
                    export_files_found=len(all_export_files),
                    selected_run_dirs=selected_run_dirs or None,
                )
            )

    normalized_before_deduplication = [
        *normalize_rows(all_raw_rows, sha_index, md5_index, filename_index),
        *pre_normalized_rows,
    ]
    normalized_rows = deduplicate_predictions(normalized_before_deduplication) if deduplicate else normalized_before_deduplication
    metrics_rows = compute_metrics(normalized_rows)

    write_csv(normalized_predictions_path, normalized_rows)
    write_csv(export_audit_path, audit_rows)
    write_csv(tool_version_log_path, version_rows)
    write_csv(forensic_tool_metrics_path, metrics_rows)

    output_tool_names = sorted(
        {
            safe_str(row.get("tool_name", ""))
            for row in [*normalized_rows, *metrics_rows]
            if safe_str(row.get("tool_name", ""))
        }
    )
    for output_tool_name in output_tool_names:
        tool_rows = [row for row in normalized_rows if safe_str(row.get("tool_name", "")) == output_tool_name]
        tool_metrics = [row for row in metrics_rows if safe_str(row.get("tool_name", "")) == output_tool_name]
        if tool_rows:
            write_csv(output_dir / f"{output_tool_name}_normalized_predictions.csv", tool_rows)
        if tool_metrics:
            write_csv(metrics_dir / f"{output_tool_name}_metrics.csv", tool_metrics)

    stale_outputs_removed = cleanup_legacy_excire_outputs(output_dir, metrics_dir, output_tool_names)
    metric_output_validation = validate_metric_outputs(
        metrics_rows=metrics_rows,
        metrics_dir=metrics_dir,
        forensic_tool_metrics_path=forensic_tool_metrics_path,
        output_tool_names=output_tool_names,
    )

    matched_before = sum(1 for row in normalized_before_deduplication if row.get("matched") == "true")
    matched_after = sum(1 for row in normalized_rows if row.get("matched") == "true")
    unmatched_after = len(normalized_rows) - matched_after
    interpretable_after = sum(1 for row in normalized_rows if row.get("weapon_detected") in {"true", "false"})
    possible_weapon_after = sum(1 for row in normalized_rows if row.get("weapon_detected") == "true")
    not_flagged_after = sum(1 for row in normalized_rows if row.get("weapon_detected") == "false")
    unknown_after = sum(1 for row in normalized_rows if row.get("weapon_detected") == "unknown")

    summary = {
        "script": SCRIPT_NAME,
        "created_at": utc_now_iso(),
        "bundle_rows": len(bundle_df),
        "tools_requested": tools,
        "selected_run_dirs_by_tool": {
            tool_name: [repo_relative_string(path) for path in paths]
            for tool_name, paths in selected_run_dirs_by_tool.items()
        },
        "per_tool_prediction_export_counts": per_tool_prediction_export_counts,
        "per_tool_total_export_counts": per_tool_total_export_counts,
        "per_tool_raw_row_counts": per_tool_raw_row_counts,
        "output_tool_names": output_tool_names,
        "stale_outputs_removed": stale_outputs_removed,
        "metric_output_validation": metric_output_validation,
        "raw_rows_parsed": len(all_raw_rows) + sum(
            count for name, count in per_tool_raw_row_counts.items() if name.startswith("excire_foto_2025")
        ),
        "pre_normalized_rows": len(pre_normalized_rows),
        "normalized_rows_before_deduplication": len(normalized_before_deduplication),
        "matched_rows_before_deduplication": matched_before,
        "deduplication_enabled": deduplicate,
        "normalized_rows_after_deduplication": len(normalized_rows),
        "matched_rows_after_deduplication": matched_after,
        "unmatched_rows_after_deduplication": unmatched_after,
        "interpretable_rows_after_deduplication": interpretable_after,
        "weapon_detected_true_after_deduplication": possible_weapon_after,
        "weapon_detected_false_after_deduplication": not_flagged_after,
        "weapon_detected_unknown_after_deduplication": unknown_after,
        "outputs": {
            "normalized_predictions": repo_relative_string(normalized_predictions_path),
            "tool_export_audit": repo_relative_string(export_audit_path),
            "tool_version_log": repo_relative_string(tool_version_log_path),
            "normalization_summary": repo_relative_string(normalization_summary_path),
            "forensic_tool_metrics": repo_relative_string(forensic_tool_metrics_path),
        },
    }
    write_json(normalization_summary_path, summary)

    logging.info(
        "Raw rows parsed: %d",
        len(all_raw_rows) + sum(
            count for name, count in per_tool_raw_row_counts.items() if name.startswith("excire_foto_2025")
        ),
    )
    logging.info("Pre-normalized rows: %d", len(pre_normalized_rows))
    logging.info("Normalized rows before deduplication: %d", len(normalized_before_deduplication))
    logging.info("Matched rows before deduplication: %d", matched_before)
    logging.info("Deduplication enabled: %s", str(deduplicate).lower())
    logging.info("Normalized rows after deduplication: %d", len(normalized_rows))
    logging.info("Matched rows after deduplication: %d", matched_after)
    logging.info("Unmatched rows after deduplication: %d", unmatched_after)
    logging.info("Interpretable rows after deduplication: %d", interpretable_after)
    logging.info("weapon_detected=true: %d", possible_weapon_after)
    logging.info("weapon_detected=false: %d", not_flagged_after)
    logging.info("weapon_detected=unknown: %d", unknown_after)
    logging.info("Normalized predictions: %s", normalized_predictions_path)
    logging.info("Export audit: %s", export_audit_path)
    logging.info("Tool version log: %s", tool_version_log_path)
    logging.info("Normalization summary: %s", normalization_summary_path)
    logging.info("Metrics: %s", forensic_tool_metrics_path)


if __name__ == "__main__":
    main()
