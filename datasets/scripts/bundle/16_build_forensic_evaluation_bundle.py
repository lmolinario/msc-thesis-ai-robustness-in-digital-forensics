#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_build_forensic_evaluation_bundle.py

Build the FAIRLab forensic evaluation bundle.

The bundle is the operational bridge between the academic pipeline and
commercial forensic-tool testing. It collects clean, OOD, adversarial and
anti-forensic artifacts into a stable, hash-traceable corpus.

Bias-control principle
----------------------
Commercial forensic tools and human analysts must not receive paths or filenames
that reveal the ground-truth class, attack family, attack name, fold, source
model, or OOD status. For this reason, the default bundle contains:

1. metadata/
   Internal manifests, hashes and summaries. These files preserve all labels,
   source information, perturbation metadata and hash mappings. They must not be
   imported into the forensic tools during blind evaluation.

2. blind_tool_input/files/
   A flat anonymized directory intended as the only input folder for Magnet
   AXIOM / Magnet.AI, Excire Foto 2025, Cellebrite Inseyets, or any other
   black-box AI-assisted media analysis tool. Files are named only as
   bundle_000001.jpg, bundle_000002.png, etc.

3. structured_audit_view/
   Optional structured copy for internal audit/debug only. This view keeps the
   semantic folder hierarchy and must not be used as forensic-tool input.

This script does not run Magnet AXIOM, Excire Foto 2025 or Cellebrite Inseyets.
It only prepares the input corpus and traceability manifests required for later
black-box forensic-tool evaluation.

Embedded metadata audit
-----------------------
The script can also audit embedded image metadata in the blind tool input view.
The audit does not modify image files and is written as separate metadata
artifacts:

- metadata/embedded_metadata_audit.csv
- metadata/embedded_metadata_audit_summary.json

The audit can be executed together with bundle generation or separately using:

    python datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py --audit-metadata-only

Default outputs
---------------
datasets/forensic_evaluation_bundle/
├── metadata/
│   ├── bundle_manifest.csv
│   ├── bundle_hashes_sha256.csv
│   ├── bundle_summary.json
│   ├── embedded_metadata_audit.csv
│   └── embedded_metadata_audit_summary.json
├── blind_tool_input/
│   └── files/
│       ├── bundle_000001.jpg
│       ├── bundle_000002.png
│       └── ...
└── structured_audit_view/
    ├── clean/
    ├── ood/
    ├── adversarial/
    └── anti_forensic/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.utils.paths import (
    ATTACKS_DIR,
    DATASETS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)


SCRIPT_NAME = "datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py"

DEFAULT_CLEAN_MANIFEST = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
DEFAULT_OOD_MANIFEST = SPLIT_MANIFESTS_DIR / "ood_eval_manifest.csv"
DEFAULT_ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"
DEFAULT_BUNDLE_DIR = DATASETS_DIR / "forensic_evaluation_bundle"


SEMANTIC_TOKENS_FORBIDDEN_IN_BLIND_PATHS = {
    "weapon",
    "non_weapon",
    "ood",
    "clean",
    "adversarial",
    "anti_forensic",
    "fgsm",
    "one_pixel",
    "sigma_zero",
    "superdeepfool",
    "color_shift",
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
    "efficientnet_b0",
    "resnet18",
    "clip",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
}


EMBEDDED_METADATA_SENSITIVE_TERMS = [
    # Experimental labels and protocol terms.
    "weapon",
    "non_weapon",
    "ood",
    "fgsm",
    "one_pixel",
    "onepixel",
    "sigma_zero",
    "sigmazero",
    "superdeepfool",
    "deepfool",
    "color_shift",
    "colorshift",
    "jpeg_recompression",
    "gaussian_blur",
    "gaussianblur",
    "histogram",
    "contrast",
    "resample",
    "resize",
    "efficientnet",
    "resnet",
    "clip",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",

    # Visual or semantic weapon-related descriptors.
    "gun",
    "firearm",
    "firearms",
    "pistol",
    "rifle",
    "shotgun",
    "handgun",
    "knife",
    "knives",
    "ammunition",
    "bullet",
    "bullets",
    "bazooka",
    "rocket launcher",
    "launcher",

    # Potentially investigation-related descriptors.
    "crime scene",
    "blood",
    "bloody",
    "murder",
]


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    """Convert a possibly missing value into a stripped string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    """Normalize a value to lowercase string form."""
    return safe_str(value).lower()


def repo_relative_string(path: Path | str) -> str:
    """Return a repository-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def maybe_value(row: pd.Series, *names: str, default: str = "") -> str:
    """Return the first non-empty value among candidate column names."""
    for name in names:
        if name in row.index:
            value = safe_str(row.get(name, ""))
            if value:
                return value
    return default


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    manifest_name: str,
) -> str:
    """Return the first existing column among candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing path column in {manifest_name}. Candidates: {candidates}")


def compute_hashes(path: Path) -> tuple[str, str]:
    """Compute SHA-256 and MD5 hashes for a file."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)

    return sha256.hexdigest(), md5.hexdigest()


def safe_filename(value: str) -> str:
    """Create a filesystem-safe filename component."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in value
    )

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the FAIRLab forensic evaluation bundle."
    )

    parser.add_argument("--clean-manifest", default=str(DEFAULT_CLEAN_MANIFEST))
    parser.add_argument("--ood-manifest", default=str(DEFAULT_OOD_MANIFEST))
    parser.add_argument("--attack-manifests-dir", default=str(DEFAULT_ATTACK_MANIFESTS_DIR))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))

    parser.add_argument(
        "--layout",
        choices=("blind", "structured", "both"),
        default="both",
        help=(
            "Bundle layout to materialize. 'blind' creates only the anonymized flat "
            "tool input; 'structured' creates only the internal audit view; 'both' "
            "creates both. Default: both."
        ),
    )

    parser.add_argument("--copy-files", dest="copy_files", action="store_true", default=True)
    parser.add_argument("--no-copy-files", dest="copy_files", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--audit-metadata-only",
        action="store_true",
        help=(
            "Run only the embedded metadata audit on an existing forensic evaluation "
            "bundle, without rebuilding the bundle or modifying bundle_summary.json."
        ),
    )

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def load_clean_rows(path: Path) -> list[dict[str, Any]]:
    """Load clean binary fold rows."""
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])

        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "clean",
            "attack_family": "none",
            "attack_name": "clean",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row["split_relative_path"]),
            "original_sha256": safe_str(row.get("sha256", "")),
            "original_md5": safe_str(row.get("md5", "")),
            "input_sha256_manifest": safe_str(row.get("sha256", "")),
            "input_md5_manifest": safe_str(row.get("md5", "")),
        })

    return rows


def load_ood_rows(path: Path) -> list[dict[str, Any]]:
    """Load OOD evaluation rows."""
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])

        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "ood",
            "attack_family": "none",
            "attack_name": "ood",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": "ood_eval_set",
            "final_label": "ood",
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row["ood_relative_path"]),
            "original_sha256": safe_str(row.get("sha256", "")),
            "original_md5": safe_str(row.get("md5", "")),
            "input_sha256_manifest": safe_str(row.get("sha256", "")),
            "input_md5_manifest": safe_str(row.get("md5", "")),
        })

    return rows


def load_attack_rows(path: Path, expected_family: str) -> list[dict[str, Any]]:
    """Load adversarial or anti-forensic attack rows."""
    df = pd.read_csv(path)

    path_col = first_existing_column(
        df=df,
        candidates=[
            "perturbed_relative_path",
            "adversarial_relative_path",
            "generated_relative_path",
            "image_relative_path",
        ],
        manifest_name=path.name,
    )

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        original_id = maybe_value(row, "original_image_id", "image_id")
        attack_name = maybe_value(
            row,
            "attack_name",
            default=path.stem.replace("_manifest", ""),
        )
        target_model = maybe_value(row, "target_model", "attack_target_model")

        if not target_model:
            target_model = "model_agnostic" if attack_name == "color_shift" else "unknown"

        generated_id = maybe_value(
            row,
            "generated_image_id",
            default=f"{original_id}__{attack_name}",
        )

        rows.append({
            "source_manifest": repo_relative_string(path),
            "sample_type": "perturbed",
            "attack_family": maybe_value(row, "attack_family", default=expected_family),
            "attack_name": attack_name,
            "attack_target_model": target_model,
            "original_image_id": original_id,
            "generated_image_id": generated_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "input_relative_path": safe_str(row[path_col]),
            "original_sha256": maybe_value(
                row,
                "sha256_original",
                "original_sha256",
                "sha256",
            ),
            "original_md5": maybe_value(
                row,
                "md5_original",
                "original_md5",
                "md5",
            ),
            "input_sha256_manifest": maybe_value(
                row,
                "sha256_perturbed",
                "perturbed_sha256",
                "sha256",
            ),
            "input_md5_manifest": maybe_value(
                row,
                "md5_perturbed",
                "perturbed_md5",
                "md5",
            ),
        })

    return rows


def discover_adversarial_manifests(manifests_dir: Path) -> list[Path]:
    """Discover adversarial attack manifests."""
    return sorted(
        p for p in manifests_dir.glob("adversarial_*_manifest.csv")
        if "summary" not in p.name and "evaluation" not in p.name
    )


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect all clean, OOD, adversarial and anti-forensic rows."""
    rows: list[dict[str, Any]] = []

    rows.extend(load_clean_rows(repo_relative_path(args.clean_manifest)))
    rows.extend(load_ood_rows(repo_relative_path(args.ood_manifest)))

    manifests_dir = repo_relative_path(args.attack_manifests_dir)

    for manifest in discover_adversarial_manifests(manifests_dir):
        loaded = load_attack_rows(manifest, "adversarial")
        logging.info("Loaded %s: %d", manifest.name, len(loaded))
        rows.extend(loaded)

    anti_manifest = manifests_dir / "anti_forensic_attacks_manifest.csv"
    if anti_manifest.exists():
        loaded = load_attack_rows(anti_manifest, "anti_forensic")
        logging.info("Loaded %s: %d", anti_manifest.name, len(loaded))
        rows.extend(loaded)

    if args.limit > 0:
        rows = rows[: args.limit]
        logging.warning("Limit active: %d rows", len(rows))

    return rows


def blind_subpath(bundle_id: str, src_path: Path) -> Path:
    """Return the blind tool input relative path for a bundle item."""
    ext = src_path.suffix.lower() or ".img"
    return Path("blind_tool_input") / "files" / f"{bundle_id}{ext}"


def structured_subpath(row: dict[str, Any], bundle_id: str, src_path: Path) -> Path:
    """Return the structured audit view relative path for a bundle item."""
    ext = src_path.suffix.lower() or ".img"

    if row["sample_type"] == "clean":
        subdir = Path("structured_audit_view") / "clean" / row["fold"] / row["final_label"]
    elif row["sample_type"] == "ood":
        subdir = Path("structured_audit_view") / "ood" / "ood_eval_set"
    elif row["attack_family"] == "adversarial":
        subdir = (
            Path("structured_audit_view")
            / "adversarial"
            / row["attack_name"]
            / row["attack_target_model"]
            / row["fold"]
            / row["final_label"]
        )
    elif row["attack_family"] == "anti_forensic":
        subdir = (
            Path("structured_audit_view")
            / "anti_forensic"
            / row["attack_name"]
            / row["fold"]
            / row["final_label"]
        )
    else:
        subdir = Path("structured_audit_view") / "other"

    filename = safe_filename(f"{bundle_id}__{row['generated_image_id']}{ext}")
    return subdir / filename


def should_create_blind(layout: str) -> bool:
    """Return True if the blind view must be created."""
    return layout in {"blind", "both"}


def should_create_structured(layout: str) -> bool:
    """Return True if the structured audit view must be created."""
    return layout in {"structured", "both"}


def materialize(
    row: dict[str, Any],
    bundle_dir: Path,
    index: int,
    created_at: str,
    copy_files: bool,
    layout: str,
) -> dict[str, Any]:
    """Materialize one bundle row and return its manifest entry."""
    bundle_id = f"bundle_{index:06d}"
    src_path = resolve_repo_path(row["input_relative_path"])

    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found for {bundle_id}: {src_path}")

    blind_relative_path = ""
    structured_relative_path = ""
    tool_input_filename = ""
    hash_path = src_path

    if copy_files and should_create_blind(layout):
        blind_path = bundle_dir / blind_subpath(bundle_id, src_path)
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, blind_path)

        blind_relative_path = repo_relative_string(blind_path)
        tool_input_filename = blind_path.name
        hash_path = blind_path

    if copy_files and should_create_structured(layout):
        structured_path = bundle_dir / structured_subpath(row, bundle_id, src_path)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, structured_path)

        structured_relative_path = repo_relative_string(structured_path)

        if not should_create_blind(layout):
            hash_path = structured_path

    sha256, md5 = compute_hashes(hash_path)
    expected = row.get("input_sha256_manifest", "")

    return {
        "bundle_id": bundle_id,
        "tool_input_filename": tool_input_filename,
        "sample_type": row["sample_type"],
        "attack_family": row["attack_family"],
        "attack_name": row["attack_name"],
        "attack_target_model": row["attack_target_model"],
        "fold": row["fold"],
        "final_label": row["final_label"],
        "source_dataset": row["source_dataset"],
        "original_image_id": row["original_image_id"],
        "generated_image_id": row["generated_image_id"],
        "source_manifest": row["source_manifest"],
        "source_relative_path": row["input_relative_path"],
        "blind_relative_path": blind_relative_path,
        "structured_relative_path": structured_relative_path,
        "tool_input_relative_path": blind_relative_path,
        "original_sha256": row.get("original_sha256", ""),
        "original_md5": row.get("original_md5", ""),
        "sha256_manifest": expected,
        "md5_manifest": row.get("input_md5_manifest", ""),
        "sha256_actual": sha256,
        "md5_actual": md5,
        "sha256_matches_manifest": bool(expected) and expected == sha256,
        "size_bytes": hash_path.stat().st_size,
        "extension": hash_path.suffix.lower(),
        "layout": layout,
        "created_at": created_at,
    }


def clear_bundle(bundle_dir: Path, force: bool) -> None:
    """Remove and recreate the bundle directory if requested."""
    if bundle_dir.exists():
        if not force:
            raise FileExistsError(f"Bundle directory already exists. Use --force: {bundle_dir}")

        logging.warning("Removing existing bundle directory: %s", bundle_dir)
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)


def hash_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the hash manifest rows."""
    return [
        {
            "bundle_id": row["bundle_id"],
            "sha256": row["sha256_actual"],
            "md5": row["md5_actual"],
            "tool_input_filename": row["tool_input_filename"],
            "tool_input_relative_path": row["tool_input_relative_path"],
            "blind_relative_path": row["blind_relative_path"],
            "structured_relative_path": row["structured_relative_path"],
            "sample_type": row["sample_type"],
            "attack_family": row["attack_family"],
            "attack_name": row["attack_name"],
            "final_label": row["final_label"],
            "original_image_id": row["original_image_id"],
            "generated_image_id": row["generated_image_id"],
        }
        for row in rows
    ]


def blind_paths_are_semantically_clean(rows: list[dict[str, Any]]) -> bool:
    """Check that blind paths do not expose semantic or experimental tokens."""
    for row in rows:
        path_value = str(row.get("tool_input_relative_path", "")).lower()

        if not path_value:
            continue

        filename = Path(path_value).name.lower()
        stem = Path(filename).stem.lower()

        if stem != row["bundle_id"].lower():
            return False

        for token in SEMANTIC_TOKENS_FORBIDDEN_IN_BLIND_PATHS:
            if token in filename:
                return False

    return True


def build_summary(
    rows: list[dict[str, Any]],
    bundle_dir: Path,
    created_at: str,
    layout: str,
) -> dict[str, Any]:
    """Build the main bundle summary without embedding the metadata audit."""
    metadata_dir = bundle_dir / "metadata"
    blind_dir = bundle_dir / "blind_tool_input" / "files"
    structured_dir = bundle_dir / "structured_audit_view"

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "bundle_dir": repo_relative_string(bundle_dir),
        "layout": layout,
        "outputs": {
            "metadata_dir": repo_relative_string(metadata_dir),
            "bundle_manifest_csv": repo_relative_string(metadata_dir / "bundle_manifest.csv"),
            "bundle_hashes_sha256_csv": repo_relative_string(metadata_dir / "bundle_hashes_sha256.csv"),
            "bundle_summary_json": repo_relative_string(metadata_dir / "bundle_summary.json"),
            "embedded_metadata_audit_csv": repo_relative_string(
                metadata_dir / "embedded_metadata_audit.csv"
            ),
            "embedded_metadata_audit_summary_json": repo_relative_string(
                metadata_dir / "embedded_metadata_audit_summary.json"
            ),
            "blind_tool_input_dir": repo_relative_string(blind_dir),
            "structured_audit_view_dir": repo_relative_string(structured_dir),
        },
        "counts": {
            "total_bundle_rows": len(rows),
            "by_sample_type": dict(Counter(row["sample_type"] for row in rows)),
            "by_attack_family": dict(Counter(row["attack_family"] for row in rows)),
            "by_attack_name": dict(Counter(row["attack_name"] for row in rows)),
            "by_label": dict(Counter(row["final_label"] for row in rows)),
            "blind_files": sum(1 for row in rows if row["blind_relative_path"]),
            "structured_files": sum(1 for row in rows if row["structured_relative_path"]),
        },
        "checks": {
            "bundle_id_unique": len({row["bundle_id"] for row in rows}) == len(rows),
            "sha256_actual_unique": len({row["sha256_actual"] for row in rows}) == len(rows),
            "all_sha256_match_when_manifest_present": all(
                row["sha256_matches_manifest"] or not row["sha256_manifest"]
                for row in rows
            ),
            "blind_paths_semantically_clean": blind_paths_are_semantically_clean(rows),
            "metadata_separated_from_tool_input": True,
        },
        "tool_input_instruction": (
            "For black-box forensic-tool evaluation, import only the directory "
            "datasets/forensic_evaluation_bundle/blind_tool_input/files. Do not import "
            "metadata/ or structured_audit_view/ into forensic tools."
        ),
        "methodological_note": (
            "The blind flat layout is designed to reduce path-induced and analyst-induced bias. "
            "Ground truth labels, perturbation metadata, source information and hash mappings are "
            "preserved only in metadata manifests for post-export normalization. Embedded image "
            "metadata are audited separately in metadata/embedded_metadata_audit.csv and "
            "metadata/embedded_metadata_audit_summary.json."
        ),
    }


def normalize_metadata_value(value: object) -> str:
    """Convert metadata values into a searchable string."""
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode(errors="replace")

    return str(value)


def extract_image_metadata(image_path: Path) -> dict[str, str]:
    """Extract readable embedded metadata without modifying the image file."""
    metadata: dict[str, str] = {}

    try:
        with Image.open(image_path) as img:
            metadata["format"] = normalize_metadata_value(img.format)
            metadata["mode"] = normalize_metadata_value(img.mode)
            metadata["width"] = str(img.width)
            metadata["height"] = str(img.height)

            for key, value in img.info.items():
                metadata[f"info:{key}"] = normalize_metadata_value(value)

            exif = img.getexif()
            if exif:
                for key, value in exif.items():
                    metadata[f"exif:{key}"] = normalize_metadata_value(value)

    except Exception as exc:
        metadata["error"] = repr(exc)

    return metadata


def term_matches_metadata(term: str, text: str) -> bool:
    """
    Return True if a sensitive term is found as a meaningful token or phrase.

    This avoids simple substring matches such as:
    - 'ood' inside 'blood' or 'goods'
    - 'clip' inside 'clipping'
    """
    escaped = re.escape(term.lower())
    pattern = rf"(?<!\w){escaped}(?!\w)"

    return re.search(pattern, text) is not None


def find_sensitive_metadata_hits(metadata: dict[str, str]) -> list[str]:
    """Return sensitive terms found in metadata keys or values."""
    searchable = " ".join(
        [str(k).lower() for k in metadata.keys()]
        + [str(v).lower() for v in metadata.values()]
    )

    hits = [
        term
        for term in EMBEDDED_METADATA_SENSITIVE_TERMS
        if term_matches_metadata(term, searchable)
    ]

    return sorted(set(hits))


def audit_embedded_metadata(
    blind_input_dir: Path,
    metadata_dir: Path,
) -> dict[str, object]:
    """
    Audit embedded image metadata in the blind forensic bundle input.

    The audit does not modify any file. It only checks whether embedded metadata
    may expose semantic labels, attack names, fold identifiers, model names, or
    other potentially informative descriptors.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_csv = metadata_dir / "embedded_metadata_audit.csv"
    output_summary = metadata_dir / "embedded_metadata_audit_summary.json"

    image_paths = sorted(
        p
        for p in blind_input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        }
    )

    rows: list[dict[str, object]] = []
    files_with_metadata = 0
    files_with_sensitive_hits = 0
    files_with_errors = 0

    for image_path in image_paths:
        metadata = extract_image_metadata(image_path)
        hits = find_sensitive_metadata_hits(metadata)

        has_embedded_metadata = any(
            key.startswith("info:") or key.startswith("exif:")
            for key in metadata.keys()
        )

        if has_embedded_metadata:
            files_with_metadata += 1

        if hits:
            files_with_sensitive_hits += 1

        if "error" in metadata:
            files_with_errors += 1

        rows.append({
            "relative_path": image_path.relative_to(blind_input_dir).as_posix(),
            "suffix": image_path.suffix.lower(),
            "has_embedded_metadata": has_embedded_metadata,
            "sensitive_hits": ";".join(hits),
            "metadata_keys": ";".join(sorted(metadata.keys())),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        })

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "suffix",
                "has_embedded_metadata",
                "sensitive_hits",
                "metadata_keys",
                "metadata_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "audit_name": "embedded_metadata_audit",
        "blind_input_dir": repo_relative_string(blind_input_dir),
        "total_files_checked": len(image_paths),
        "files_with_embedded_metadata": files_with_metadata,
        "files_with_sensitive_hits": files_with_sensitive_hits,
        "files_with_errors": files_with_errors,
        "sensitive_terms": EMBEDDED_METADATA_SENSITIVE_TERMS,
        "output_csv": repo_relative_string(output_csv),
        "output_summary": repo_relative_string(output_summary),
        "note": (
            "This audit does not modify image files. It identifies embedded "
            "metadata that may expose semantic or experiment-specific information."
        ),
    }

    output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def run_metadata_audit_only(bundle_dir: Path) -> None:
    """Run only the embedded metadata audit on an existing bundle."""
    metadata_dir = bundle_dir / "metadata"
    blind_tool_input_dir = bundle_dir / "blind_tool_input" / "files"

    if not blind_tool_input_dir.exists():
        raise FileNotFoundError(
            f"Blind tool input directory not found: {blind_tool_input_dir}"
        )

    embedded_metadata_audit_summary = audit_embedded_metadata(
        blind_input_dir=blind_tool_input_dir,
        metadata_dir=metadata_dir,
    )

    logging.info(
        "Embedded metadata audit written: %s",
        metadata_dir / "embedded_metadata_audit_summary.json",
    )

    print("\nEmbedded metadata audit completed:")
    print(json.dumps(embedded_metadata_audit_summary, indent=2, ensure_ascii=False))


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    bundle_dir = repo_relative_path(args.bundle_dir)

    if args.audit_metadata_only:
        run_metadata_audit_only(bundle_dir)
        return

    clear_bundle(bundle_dir, args.force)

    source_rows = collect_rows(args)
    created_at = utc_now_iso()

    bundle_rows: list[dict[str, Any]] = []

    for index, row in enumerate(source_rows, start=1):
        bundle_rows.append(
            materialize(
                row=row,
                bundle_dir=bundle_dir,
                index=index,
                created_at=created_at,
                copy_files=args.copy_files,
                layout=args.layout,
            )
        )

        if index % 500 == 0:
            logging.info("Materialized %d/%d", index, len(source_rows))

    metadata_dir = bundle_dir / "metadata"
    blind_tool_input_dir = bundle_dir / "blind_tool_input" / "files"

    write_csv(metadata_dir / "bundle_manifest.csv", bundle_rows)
    write_csv(metadata_dir / "bundle_hashes_sha256.csv", hash_rows(bundle_rows))
    write_json(
        metadata_dir / "bundle_summary.json",
        build_summary(bundle_rows, bundle_dir, created_at, args.layout),
    )

    embedded_metadata_audit_summary = audit_embedded_metadata(
        blind_input_dir=blind_tool_input_dir,
        metadata_dir=metadata_dir,
    )

    logging.info("Bundle written: %s", bundle_dir)
    logging.info("Tool input directory: %s", blind_tool_input_dir)
    logging.info("Metadata directory: %s", metadata_dir)
    logging.info(
        "Embedded metadata audit written: %s",
        metadata_dir / "embedded_metadata_audit_summary.json",
    )

    print("\nEmbedded metadata audit completed:")
    print(json.dumps(embedded_metadata_audit_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()