#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_generate_adversarial_attacks.py

Official adversarial attack generation entry point for the FAIR-Lab thesis
pipeline.

Purpose
-------
Generate adversarial or adversarial-style perturbations from the official clean
binary folds.

Inputs
------
- datasets/splits/manifests/clean_folds_manifest.csv

Outputs
-------
- attacks/adversarial/<attack_name>/<target_model>/<fold>/<label>/<generated_file>.jpg
- attacks/manifests/adversarial_attacks_manifest.csv
- attacks/manifests/adversarial_generation_summary.json

Methodological notes
--------------------
- This script intentionally separates implemented attacks from planned attacks.
- Model-dependent attacks such as FGSM, Sigma-Zero, One Pixel, and SuperDeepFool
  require a stable target-model interface before being enabled.
- The first implemented attack is Color Shift, treated as a controlled
  model-agnostic color-space perturbation.
- The output directory keeps an explicit technical naming convention to support
  internal debugging and traceability.
- A later bundle-generation stage should copy these artifacts into a neutral
  forensic evaluation bundle using opaque names such as sample_0000001.jpg.
- The mapping between technical files, labels, folds, transformations, model
  targets, and hashes is preserved exclusively through manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from datasets.scripts.utils.paths import (
    ADVERSARIAL_DIR,
    ATTACKS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_NAME = "datasets/scripts/attacks/13_generate_adversarial_attacks.py"

INPUT_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"

ADVERSARIAL_MANIFEST_PATH = ATTACK_MANIFESTS_DIR / "adversarial_attacks_manifest.csv"
ADVERSARIAL_SUMMARY_PATH = ATTACK_MANIFESTS_DIR / "adversarial_generation_summary.json"

VALID_LABELS = {"weapon", "non_weapon"}

PLANNED_ATTACK_NAMES = [
    "fgsm",
    "sigma_zero",
    "one_pixel",
    "color_shift",
    "superdeepfool",
]

IMPLEMENTED_ATTACK_NAMES = [
    "color_shift",
]

MODEL_AGNOSTIC_TARGET = "model_agnostic"


# =============================================================================
# Argument parsing and logging
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adversarial/adversarial-style image perturbations from "
            "the official clean folds."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default=str(INPUT_MANIFEST_PATH),
        help=f"Clean folds manifest (default: {INPUT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--attack",
        nargs="+",
        choices=PLANNED_ATTACK_NAMES,
        default=IMPLEMENTED_ATTACK_NAMES,
        help=(
            "One or more adversarial attacks to generate. "
            "Currently implemented: color_shift. "
            "Planned: fgsm, sigma_zero, one_pixel, superdeepfool."
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
        default=95,
        help="JPEG output quality for generated adversarial images (default: 95).",
    )
    parser.add_argument(
        "--color-red-shift",
        type=int,
        default=12,
        help="Additive red-channel shift for color_shift (default: 12).",
    )
    parser.add_argument(
        "--color-green-shift",
        type=int,
        default=0,
        help="Additive green-channel shift for color_shift (default: 0).",
    )
    parser.add_argument(
        "--color-blue-shift",
        type=int,
        default=-12,
        help="Additive blue-channel shift for color_shift (default: -12).",
    )
    parser.add_argument(
        "--color-saturation-factor",
        type=float,
        default=1.10,
        help="Saturation multiplier for color_shift (default: 1.10).",
    )
    parser.add_argument(
        "--color-contrast-factor",
        type=float,
        default=1.00,
        help="Contrast multiplier for color_shift (default: 1.00).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


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


def validate_selected_attacks(selected_attacks: list[str]) -> None:
    not_implemented = [
        attack for attack in selected_attacks if attack not in IMPLEMENTED_ATTACK_NAMES
    ]

    if not_implemented:
        raise NotImplementedError(
            "The following adversarial attacks are planned but not implemented yet: "
            f"{', '.join(not_implemented)}. "
            "Enable them only after the target-model interface is finalized."
        )


def load_manifest(path: Path) -> pd.DataFrame:
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
    ADVERSARIAL_DIR.mkdir(parents=True, exist_ok=True)
    ATTACK_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    existing_attack_dirs = [
        ADVERSARIAL_DIR / attack for attack in selected_attacks
        if (ADVERSARIAL_DIR / attack).exists()
    ]

    if existing_attack_dirs and not force:
        raise FileExistsError(
            "Selected attack output directories already exist. Use --force to rebuild them: "
            + ", ".join(str(path) for path in existing_attack_dirs)
        )

    if force:
        for attack in selected_attacks:
            attack_dir = ADVERSARIAL_DIR / attack
            if attack_dir.exists():
                logging.warning("Removing existing attack output directory: %s", attack_dir)
                shutil.rmtree(attack_dir)


def open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")
            else:
                img = img.copy()
            return img
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
# Perturbation metrics
# =============================================================================

def compute_perturbation_metrics(
    original_img: Image.Image,
    transformed_img: Image.Image,
) -> dict[str, float | int]:
    original = np.asarray(original_img.convert("RGB"), dtype=np.float32)
    transformed = np.asarray(transformed_img.convert("RGB"), dtype=np.float32)

    if original.shape != transformed.shape:
        raise ValueError(
            "Cannot compute perturbation metrics on images with different shapes: "
            f"original={original.shape}, transformed={transformed.shape}"
        )

    diff = transformed - original
    abs_diff = np.abs(diff)

    return {
        "perturbation_norm_l0": int(np.count_nonzero(abs_diff)),
        "perturbation_norm_l2": float(np.linalg.norm(diff.ravel(), ord=2)),
        "perturbation_norm_linf": float(np.max(abs_diff)),
        "perturbation_mean_abs": float(np.mean(abs_diff)),
    }


# =============================================================================
# Adversarial / adversarial-style transformations
# =============================================================================

def _shift_channel(channel: Image.Image, shift: int) -> Image.Image:
    lookup = [max(0, min(255, value + shift)) for value in range(256)]
    return channel.point(lookup)


def apply_color_shift(
    img: Image.Image,
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, Any], str]:
    if args.color_saturation_factor <= 0:
        raise ValueError("--color-saturation-factor must be greater than 0.")
    if args.color_contrast_factor <= 0:
        raise ValueError("--color-contrast-factor must be greater than 0.")

    red, green, blue = img.convert("RGB").split()
    shifted = Image.merge(
        "RGB",
        (
            _shift_channel(red, args.color_red_shift),
            _shift_channel(green, args.color_green_shift),
            _shift_channel(blue, args.color_blue_shift),
        ),
    )

    if args.color_saturation_factor != 1.0:
        shifted = ImageEnhance.Color(shifted).enhance(args.color_saturation_factor)
    if args.color_contrast_factor != 1.0:
        shifted = ImageEnhance.Contrast(shifted).enhance(args.color_contrast_factor)

    params = {
        "red_shift": args.color_red_shift,
        "green_shift": args.color_green_shift,
        "blue_shift": args.color_blue_shift,
        "saturation_factor": args.color_saturation_factor,
        "contrast_factor": args.color_contrast_factor,
        "output_format": "JPEG",
        "jpeg_quality": args.jpeg_quality,
    }

    return shifted, params, MODEL_AGNOSTIC_TARGET


TRANSFORMERS: dict[
    str,
    Callable[[Image.Image, argparse.Namespace], tuple[Image.Image, dict[str, Any], str]],
] = {
    "color_shift": apply_color_shift,
}


# =============================================================================
# Generation logic
# =============================================================================

def build_output_path(row: pd.Series, attack_name: str, target_model: str) -> Path:
    image_id = safe_str(row["image_id"])
    fold = safe_str(row["fold"])
    label = norm(row["final_label"])
    filename = f"{image_id}__{attack_name}__{target_model}.jpg"
    return ADVERSARIAL_DIR / attack_name / target_model / fold / label / filename


def generate_one(
    row: pd.Series,
    attack_name: str,
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    image_id = safe_str(row["image_id"])
    label = norm(row["final_label"])
    fold = safe_str(row["fold"])

    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))

    img = open_rgb_image(source_path)
    transformer = TRANSFORMERS[attack_name]
    transformed_img, attack_params, target_model = transformer(img, args)

    output_path = build_output_path(row, attack_name, target_model)
    save_jpeg(img=transformed_img, output_path=output_path, quality=args.jpeg_quality)

    sha256_perturbed, md5_perturbed = compute_hashes(output_path)
    sha256_original, md5_original = compute_hashes(source_path)
    size_bytes = output_path.stat().st_size

    expected_sha256_original = safe_str(row["sha256"])
    if sha256_original != expected_sha256_original:
        raise RuntimeError(
            f"Original SHA256 mismatch for {image_id}: "
            f"manifest={expected_sha256_original}, computed={sha256_original}"
        )

    metrics = compute_perturbation_metrics(img, transformed_img)

    return {
        "generated_image_id": f"{image_id}__{attack_name}__{target_model}",
        "original_image_id": image_id,
        "fold": fold,
        "final_label": label,
        "source_dataset": safe_str(row.get("source_dataset", "")),
        "source_group": safe_str(row.get("source_group", "")),
        "source_relative_path": safe_str(row.get("source_relative_path", "")),
        "prepared_relative_path": safe_str(row.get("prepared_relative_path", "")),
        "clean_relative_path": safe_str(row["split_relative_path"]),
        "perturbed_relative_path": repo_relative_string(output_path),
        "attack_family": "adversarial",
        "attack_name": attack_name,
        "attack_parameters": json.dumps(attack_params, sort_keys=True),
        "target_model": target_model,
        "model_dependency": "model_agnostic",
        "attack_success": "not_applicable",
        "original_prediction": "not_computed",
        "adversarial_prediction": "not_computed",
        "original_confidence": "not_computed",
        "adversarial_confidence": "not_computed",
        "sha256_original": sha256_original,
        "md5_original": md5_original,
        "sha256_perturbed": sha256_perturbed,
        "md5_perturbed": md5_perturbed,
        "perturbation_norm_l0": metrics["perturbation_norm_l0"],
        "perturbation_norm_l2": metrics["perturbation_norm_l2"],
        "perturbation_norm_linf": metrics["perturbation_norm_linf"],
        "perturbation_mean_abs": metrics["perturbation_mean_abs"],
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
    selected_attacks: list[str],
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    per_attack_counts = Counter(row["attack_name"] for row in rows)
    per_target_counts = Counter(row["target_model"] for row in rows)
    per_fold_counts: dict[str, Counter] = defaultdict(Counter)
    per_label_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        per_fold_counts[row["attack_name"]][row["fold"]] += 1
        per_label_counts[row["attack_name"]][row["final_label"]] += 1

    unique_generated_ids = {row["generated_image_id"] for row in rows}
    unique_perturbed_hashes = {row["sha256_perturbed"] for row in rows}
    expected_total = len(pd.read_csv(input_manifest)) * len(selected_attacks)

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "input_manifest": repo_relative_string(input_manifest),
        "output_root": repo_relative_string(ADVERSARIAL_DIR),
        "manifest_csv": repo_relative_string(ADVERSARIAL_MANIFEST_PATH),
        "summary_json": repo_relative_string(ADVERSARIAL_SUMMARY_PATH),
        "planned_attacks": PLANNED_ATTACK_NAMES,
        "implemented_attacks": IMPLEMENTED_ATTACK_NAMES,
        "selected_attacks": selected_attacks,
        "parameters": {
            "jpeg_quality": args.jpeg_quality,
            "color_red_shift": args.color_red_shift,
            "color_green_shift": args.color_green_shift,
            "color_blue_shift": args.color_blue_shift,
            "color_saturation_factor": args.color_saturation_factor,
            "color_contrast_factor": args.color_contrast_factor,
        },
        "counts": {
            "input_images": expected_total // len(selected_attacks) if selected_attacks else 0,
            "selected_attack_count": len(selected_attacks),
            "expected_generated_images": expected_total,
            "actual_generated_images": len(rows),
            "per_attack_counts": dict(sorted(per_attack_counts.items())),
            "per_target_model_counts": dict(sorted(per_target_counts.items())),
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
            "manifest_written": ADVERSARIAL_MANIFEST_PATH.exists(),
        },
        "methodological_status": {
            "color_shift": "implemented_as_model_agnostic_color_space_perturbation",
            "fgsm": "planned_requires_target_model_interface",
            "sigma_zero": "planned_requires_reference_implementation_or_stable_adapter",
            "one_pixel": "planned_requires_black_box_or_surrogate_model_interface",
            "superdeepfool": "planned_requires_reference_implementation_or_stable_adapter",
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
    validate_selected_attacks(selected_attacks)

    input_manifest = repo_relative_path(args.input_manifest)
    created_at = utc_now_iso()

    logging.info("Input manifest: %s", input_manifest)
    logging.info("Selected attacks: %s", ", ".join(selected_attacks))
    logging.info("Output root: %s", ADVERSARIAL_DIR)

    df = load_manifest(input_manifest)
    validate_source_files(df)
    prepare_output_dirs(selected_attacks=selected_attacks, force=args.force)

    rows: list[dict[str, Any]] = []
    total_expected = len(df) * len(selected_attacks)
    progress_counter = 0

    for attack_name in selected_attacks:
        logging.info("Generating attack: %s", attack_name)
        for _, row in df.iterrows():
            generated_row = generate_one(
                row=row,
                attack_name=attack_name,
                args=args,
                created_at=created_at,
            )
            rows.append(generated_row)
            progress_counter += 1

            if progress_counter % 250 == 0 or progress_counter == total_expected:
                logging.info("Generated %d/%d images", progress_counter, total_expected)

    write_csv(ADVERSARIAL_MANIFEST_PATH, rows)

    summary = build_summary(
        rows=rows,
        input_manifest=input_manifest,
        selected_attacks=selected_attacks,
        args=args,
        created_at=created_at,
    )
    write_summary(ADVERSARIAL_SUMMARY_PATH, summary)

    logging.info("Manifest written: %s", ADVERSARIAL_MANIFEST_PATH)
    logging.info("Summary written: %s", ADVERSARIAL_SUMMARY_PATH)
    logging.info("Adversarial generation completed successfully.")


if __name__ == "__main__":
    main()
