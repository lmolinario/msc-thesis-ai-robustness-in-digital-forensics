#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_generate_anti_forensic_attacks.py

Official anti-forensic transformation script for the FAIR-Lab thesis pipeline.

This script supports two equivalent execution modes:

1. Interactive mode, automatically enabled when the script is launched without
   command-line arguments.
2. Command-line mode, used for fully reproducible scripted execution.

Purpose
-------
Generate controlled anti-forensic image transformations from the official clean
binary folds.

Inputs
------
- datasets/splits/manifests/clean_folds_manifest.csv

Outputs
-------
- attacks/anti_forensic/jpeg_recompression/<fold>/<label>/<image_id>__jpeg_recompression.jpg
- attacks/anti_forensic/resample_resize/<fold>/<label>/<image_id>__resample_resize.jpg
- attacks/anti_forensic/gaussian_blur/<fold>/<label>/<image_id>__gaussian_blur.jpg
- attacks/anti_forensic/histogram_modification/<fold>/<label>/<image_id>__histogram_modification.jpg
- attacks/anti_forensic/contrast_stretching/<fold>/<label>/<image_id>__contrast_stretching.jpg
- attacks/manifests/anti_forensic_attacks_manifest.csv
- attacks/manifests/anti_forensic_generation_summary.json

Methodological notes
--------------------
- This script implements anti-forensic transformations as controlled image-processing
  operations, not as model-optimized adversarial examples.
- The output directory keeps an explicit technical naming convention to support
  internal debugging and traceability.
- A later bundle-generation stage should copy these artifacts into a neutral
  forensic evaluation bundle using opaque names such as sample_0000001.jpg.
- The mapping between technical files, labels, folds, transformations, and hashes
  is preserved exclusively through manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from datasets.scripts.utils.paths import (
    ANTI_FORENSIC_DIR,
    ATTACKS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_NAME = "datasets/scripts/attacks/13_generate_anti_forensic_attacks.py"

INPUT_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"

ANTI_FORENSIC_MANIFEST_PATH = ATTACK_MANIFESTS_DIR / "anti_forensic_attacks_manifest.csv"
ANTI_FORENSIC_SUMMARY_PATH = ATTACK_MANIFESTS_DIR / "anti_forensic_generation_summary.json"

VALID_LABELS = {"weapon", "non_weapon"}

ATTACK_NAMES = [
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
]


# =============================================================================
# Interactive helpers
# =============================================================================

def ask_choice(prompt: str, valid_choices: set[str], default: str) -> str:
    """Ask for a constrained textual choice."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip().lower()
        if not value:
            return default
        if value in valid_choices:
            return value
        print(f"Invalid choice: {value}. Valid choices: {', '.join(sorted(valid_choices))}.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question and return a boolean."""
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "s", "si", "sì"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    """Ask for an integer value."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if parsed < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return parsed


def ask_attack_selection() -> list[str]:
    """Ask for one or more anti-forensic transformations."""
    print("\nSelect anti-forensic transformations:")
    for idx, attack_name in enumerate(ATTACK_NAMES, start=1):
        print(f"  {idx}. {attack_name}")
    print("Examples: 1 | 1 3 | 1,2,5 | all")
    value = input("Selection [all]: ").strip().lower()
    if not value or value == "all":
        return ATTACK_NAMES.copy()

    mapping = {str(idx): attack_name for idx, attack_name in enumerate(ATTACK_NAMES, start=1)}
    tokens = value.replace(",", " ").split()
    selected: list[str] = []
    for token in tokens:
        if token in mapping:
            selected.append(mapping[token])
        elif token in ATTACK_NAMES:
            selected.append(token)
        else:
            raise ValueError(f"Invalid anti-forensic attack selection: {token}")
    return list(dict.fromkeys(selected))


def parse_interactive_args() -> argparse.Namespace:
    """Build an argparse namespace through a safe interactive launcher."""
    print("\n" + "=" * 78)
    print("FAIR-Lab anti-forensic transformation generator")
    print("=" * 78)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Input manifest: {INPUT_MANIFEST_PATH}")
    print("\nWhat do you want to generate?")
    print("  1. all anti-forensic transformations [default]")
    print("  2. one or more selected transformations")
    print("  3. smoke test all transformations (--limit 10)")

    selection = ask_choice("Selection", {"1", "2", "3"}, "1")
    selected_attacks = ATTACK_NAMES.copy()
    limit = 0

    if selection == "2":
        selected_attacks = ask_attack_selection()
    elif selection == "3":
        selected_attacks = ATTACK_NAMES.copy()
        limit = 10

    force = ask_yes_no("Overwrite existing selected output directories if present?", default=True)
    verbose = ask_yes_no("Enable verbose logging?", default=False)

    return argparse.Namespace(
        input_manifest=str(INPUT_MANIFEST_PATH),
        attack=selected_attacks,
        force=force,
        jpeg_quality=70,
        resample_scale=0.50,
        blur_radius=1.50,
        contrast_cutoff=1.0,
        limit=limit,
        verbose=verbose,
    )


# =============================================================================
# Argument parsing and logging
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate anti-forensic image transformations from the official clean folds."
    )
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode.")
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=str(INPUT_MANIFEST_PATH),
        help=f"Clean folds manifest (default: {INPUT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--attack",
        nargs="+",
        choices=ATTACK_NAMES,
        default=ATTACK_NAMES,
        help=(
            "One or more anti-forensic transformations to generate. "
            "By default, all anti-forensic transformations are generated."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and rebuild selected attack output directories before generation.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help="JPEG recompression quality for jpeg_recompression (default: 70).",
    )
    parser.add_argument(
        "--resample-scale",
        type=float,
        default=0.50,
        help="Downscale factor for resample_resize before restoring original size (default: 0.50).",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=1.50,
        help="Gaussian blur radius for gaussian_blur (default: 1.50).",
    )
    parser.add_argument(
        "--contrast-cutoff",
        type=float,
        default=1.0,
        help="Autocontrast cutoff percentage for contrast_stretching (default: 1.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of manifest rows to process for smoke tests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1:
        return parse_interactive_args()
    parser = build_parser()
    args = parser.parse_args()
    if args.interactive:
        return parse_interactive_args()
    return args


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def repo_relative_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def compute_hashes(file_path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)

    return sha256.hexdigest(), md5.hexdigest()


def ensure_required_columns(df: pd.DataFrame, required: set[str], manifest_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {manifest_name}: {sorted(missing)}")


def load_manifest(path: Path, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    df = pd.read_csv(path)
    required = {
        "image_id",
        "fold",
        "final_label",
        "source_dataset",
        "split_relative_path",
        "sha256",
        "md5",
    }
    ensure_required_columns(df, required, path.name)

    if df.empty:
        raise ValueError(f"Input manifest is empty: {path}")

    if limit > 0:
        df = df.head(limit).copy()

    invalid_labels = set(df["final_label"].map(norm).unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(f"Invalid labels in input manifest: {sorted(invalid_labels)}")

    duplicated_image_ids = int(df["image_id"].duplicated().sum())
    if duplicated_image_ids:
        raise ValueError(f"Duplicated image_id values in input manifest: {duplicated_image_ids}")

    return df


def resolve_clean_image_path(split_relative_path: str) -> Path:
    path = Path(split_relative_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def validate_source_files(df: pd.DataFrame) -> None:
    missing: list[str] = []
    mismatches: list[str] = []

    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])
        src_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))

        if not src_path.exists() or not src_path.is_file():
            missing.append(f"{image_id} -> {src_path}")
            continue

        computed_sha256, computed_md5 = compute_hashes(src_path)
        expected_sha256 = safe_str(row["sha256"])
        expected_md5 = safe_str(row["md5"])

        if computed_sha256 != expected_sha256:
            mismatches.append(
                f"{image_id} SHA256 expected={expected_sha256} computed={computed_sha256}"
            )
        if expected_md5 and computed_md5 != expected_md5:
            mismatches.append(
                f"{image_id} MD5 expected={expected_md5} computed={computed_md5}"
            )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing clean image files: {len(missing)}\nPreview:\n{preview}"
        )

    if mismatches:
        preview = "\n".join(mismatches[:20])
        raise RuntimeError(
            f"Hash mismatches found in clean inputs: {len(mismatches)}\nPreview:\n{preview}"
        )


def prepare_output_dirs(selected_attacks: list[str], force: bool) -> None:
    ANTI_FORENSIC_DIR.mkdir(parents=True, exist_ok=True)
    ATTACK_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    existing_attack_dirs = [ANTI_FORENSIC_DIR / attack for attack in selected_attacks if (ANTI_FORENSIC_DIR / attack).exists()]

    if existing_attack_dirs and not force:
        raise FileExistsError(
            "Selected attack output directories already exist. Use --force to rebuild them: "
            + ", ".join(str(path) for path in existing_attack_dirs)
        )

    if force:
        for attack in selected_attacks:
            attack_dir = ANTI_FORENSIC_DIR / attack
            if attack_dir.exists():
                logging.warning("Removing existing attack output directory: %s", attack_dir)
                shutil.rmtree(attack_dir)


def open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return img.copy()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {path}") from exc


def save_jpeg(img: Image.Image, output_path: Path, quality: int = 95) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=False,
    )


# =============================================================================
# Anti-forensic transformations
# =============================================================================

def apply_jpeg_recompression(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    params = {
        "quality": args.jpeg_quality,
        "output_format": "JPEG",
    }
    return img, params


def apply_resample_resize(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    original_width, original_height = img.size
    scale = args.resample_scale

    if not (0 < scale < 1):
        raise ValueError("--resample-scale must be between 0 and 1.")

    down_width = max(1, int(round(original_width * scale)))
    down_height = max(1, int(round(original_height * scale)))

    downsampled = img.resize((down_width, down_height), Image.Resampling.BICUBIC)
    restored = downsampled.resize((original_width, original_height), Image.Resampling.BICUBIC)

    params = {
        "scale": scale,
        "downsample_size": f"{down_width}x{down_height}",
        "restored_size": f"{original_width}x{original_height}",
        "resampling": "bicubic",
        "output_format": "JPEG",
        "jpeg_quality": 95,
    }
    return restored, params


def apply_gaussian_blur(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    radius = args.blur_radius
    if radius <= 0:
        raise ValueError("--blur-radius must be greater than 0.")

    transformed = img.filter(ImageFilter.GaussianBlur(radius=radius))
    params = {
        "radius": radius,
        "output_format": "JPEG",
        "jpeg_quality": 95,
    }
    return transformed, params


def apply_histogram_modification(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    transformed = ImageOps.equalize(img)
    params = {
        "method": "histogram_equalization",
        "output_format": "JPEG",
        "jpeg_quality": 95,
    }
    return transformed, params


def apply_contrast_stretching(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    cutoff = args.contrast_cutoff
    if cutoff < 0:
        raise ValueError("--contrast-cutoff must be greater than or equal to 0.")

    transformed = ImageOps.autocontrast(img, cutoff=cutoff)
    params = {
        "method": "autocontrast",
        "cutoff_percent": cutoff,
        "output_format": "JPEG",
        "jpeg_quality": 95,
    }
    return transformed, params


TRANSFORMERS: dict[str, Callable[[Image.Image, argparse.Namespace], tuple[Image.Image, dict[str, Any]]]] = {
    "jpeg_recompression": apply_jpeg_recompression,
    "resample_resize": apply_resample_resize,
    "gaussian_blur": apply_gaussian_blur,
    "histogram_modification": apply_histogram_modification,
    "contrast_stretching": apply_contrast_stretching,
}


def output_quality_for_attack(attack_name: str, args: argparse.Namespace) -> int:
    if attack_name == "jpeg_recompression":
        return args.jpeg_quality
    return 95


# =============================================================================
# Generation logic
# =============================================================================

def build_output_path(row: pd.Series, attack_name: str) -> Path:
    image_id = safe_str(row["image_id"])
    fold = safe_str(row["fold"])
    label = norm(row["final_label"])
    filename = f"{image_id}__{attack_name}.jpg"
    return ANTI_FORENSIC_DIR / attack_name / fold / label / filename


def generate_one(row: pd.Series, attack_name: str, args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    image_id = safe_str(row["image_id"])
    label = norm(row["final_label"])
    fold = safe_str(row["fold"])

    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))
    output_path = build_output_path(row, attack_name)

    img = open_rgb_image(source_path)
    transformer = TRANSFORMERS[attack_name]
    transformed_img, attack_params = transformer(img, args)

    save_jpeg(
        img=transformed_img,
        output_path=output_path,
        quality=output_quality_for_attack(attack_name, args),
    )

    sha256_perturbed, md5_perturbed = compute_hashes(output_path)
    sha256_original, md5_original = compute_hashes(source_path)
    size_bytes = output_path.stat().st_size

    expected_sha256_original = safe_str(row["sha256"])
    if sha256_original != expected_sha256_original:
        raise RuntimeError(
            f"Original SHA256 mismatch for {image_id}: "
            f"manifest={expected_sha256_original}, computed={sha256_original}"
        )

    return {
        "generated_image_id": f"{image_id}__{attack_name}",
        "original_image_id": image_id,
        "fold": fold,
        "final_label": label,
        "source_dataset": safe_str(row.get("source_dataset", "")),
        "source_group": safe_str(row.get("source_group", "")),
        "source_relative_path": safe_str(row.get("source_relative_path", "")),
        "prepared_relative_path": safe_str(row.get("prepared_relative_path", "")),
        "clean_relative_path": safe_str(row["split_relative_path"]),
        "perturbed_relative_path": repo_relative_string(output_path),
        "attack_family": "anti_forensic",
        "attack_name": attack_name,
        "attack_parameters": json.dumps(attack_params, sort_keys=True),
        "sha256_original": sha256_original,
        "md5_original": md5_original,
        "sha256_perturbed": sha256_perturbed,
        "md5_perturbed": md5_perturbed,
        "size_bytes": size_bytes,
        "extension": ".jpg",
        "created_at": created_at,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, Any]],
    input_manifest: Path,
    input_image_count: int,
    selected_attacks: list[str],
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    per_attack_counts = Counter(row["attack_name"] for row in rows)
    per_fold_counts: dict[str, Counter] = defaultdict(Counter)
    per_label_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        per_fold_counts[row["attack_name"]][row["fold"]] += 1
        per_label_counts[row["attack_name"]][row["final_label"]] += 1

    unique_generated_ids = {row["generated_image_id"] for row in rows}
    unique_perturbed_hashes = {row["sha256_perturbed"] for row in rows}
    expected_total = input_image_count * len(selected_attacks)

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "input_manifest": repo_relative_string(input_manifest),
        "output_root": repo_relative_string(ANTI_FORENSIC_DIR),
        "manifest_csv": repo_relative_string(ANTI_FORENSIC_MANIFEST_PATH),
        "summary_json": repo_relative_string(ANTI_FORENSIC_SUMMARY_PATH),
        "selected_attacks": selected_attacks,
        "parameters": {
            "jpeg_quality": args.jpeg_quality,
            "resample_scale": args.resample_scale,
            "blur_radius": args.blur_radius,
            "contrast_cutoff": args.contrast_cutoff,
            "limit": args.limit,
        },
        "counts": {
            "input_images": input_image_count,
            "selected_attack_count": len(selected_attacks),
            "expected_generated_images": expected_total,
            "actual_generated_images": len(rows),
            "per_attack_counts": dict(sorted(per_attack_counts.items())),
            "per_fold_counts": {
                attack: dict(sorted(counter.items()))
                for attack, counter in sorted(per_fold_counts.items())
            },
            "per_label_counts": {
                attack: dict(sorted(counter.items()))
                for attack, counter in sorted(per_label_counts.items())
            },
        },
        "checks": {
            "expected_total_generated": len(rows) == expected_total,
            "generated_image_id_unique": len(unique_generated_ids) == len(rows),
            "perturbed_sha256_unique": len(unique_perturbed_hashes) == len(rows),
            "manifest_written": ANTI_FORENSIC_MANIFEST_PATH.exists(),
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    selected_attacks = list(dict.fromkeys(args.attack))
    input_manifest = repo_relative_path(args.input_manifest)
    created_at = utc_now_iso()

    logging.info("Input manifest: %s", input_manifest)
    logging.info("Selected attacks: %s", ", ".join(selected_attacks))
    logging.info("Output root: %s", ANTI_FORENSIC_DIR)

    df = load_manifest(input_manifest, args.limit)
    validate_source_files(df)
    prepare_output_dirs(selected_attacks=selected_attacks, force=args.force)

    rows: list[dict[str, Any]] = []
    total_expected = len(df) * len(selected_attacks)
    progress_counter = 0

    for attack_name in selected_attacks:
        logging.info("Generating attack: %s", attack_name)
        for _, row in df.iterrows():
            generated_row = generate_one(row=row, attack_name=attack_name, args=args, created_at=created_at)
            rows.append(generated_row)
            progress_counter += 1

            if progress_counter % 250 == 0 or progress_counter == total_expected:
                logging.info("Generated %d/%d images", progress_counter, total_expected)

    write_csv(ANTI_FORENSIC_MANIFEST_PATH, rows)

    summary = build_summary(
        rows=rows,
        input_manifest=input_manifest,
        input_image_count=len(df),
        selected_attacks=selected_attacks,
        args=args,
        created_at=created_at,
    )
    write_summary(ANTI_FORENSIC_SUMMARY_PATH, summary)

    logging.info("Manifest written: %s", ANTI_FORENSIC_MANIFEST_PATH)
    logging.info("Summary written: %s", ANTI_FORENSIC_SUMMARY_PATH)
    logging.info("Anti-forensic generation completed successfully.")


if __name__ == "__main__":
    main()
