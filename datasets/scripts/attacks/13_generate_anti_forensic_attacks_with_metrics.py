#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_generate_anti_forensic_attacks.py

Official anti-forensic transformation script for the FAIR-Lab thesis pipeline.

This version preserves the original generation logic and adds an optional
forensic-oriented evaluation stage.

Main execution modes
--------------------
1. Generate anti-forensic transformations only.
2. Generate transformations and evaluate them on a target model.
3. Evaluate an existing anti-forensic manifest without regenerating images.

Forensic evaluation purpose
--------------------------
The evaluation stage measures operational robustness, not AML optimization.
The key metrics are:
- clean accuracy;
- anti-forensic accuracy;
- accuracy drop;
- manipulation-induced error rate on clean-correct images;
- weapon -> non_weapon false negatives;
- non_weapon -> weapon false positives;
- per-attack, per-fold and per-class metrics;
- confidence shift.

Inputs
------
Generation:
- datasets/splits/manifests/clean_folds_manifest.csv

Evaluation:
- attacks/manifests/anti_forensic_attacks_manifest.csv
- trained target-model checkpoint, e.g. EfficientNet-B0

Outputs
-------
Generation:
- attacks/anti_forensic/<attack>/<fold>/<label>/<image_id>__<attack>.jpg
- attacks/manifests/anti_forensic_attacks_manifest.csv
- attacks/manifests/anti_forensic_generation_summary.json

Evaluation:
- attacks/manifests/anti_forensic_<target_model>_evaluation.csv
- attacks/manifests/anti_forensic_<target_model>_evaluation_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

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

CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"

VALID_LABELS = {"weapon", "non_weapon"}
DEFAULT_LABEL_ORDER = ["non_weapon", "weapon"]

ATTACK_NAMES = [
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
]

SUPPORTED_TARGET_MODELS = ["efficientnet_b0"]


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
    print("FAIR-Lab anti-forensic transformation generator/evaluator")
    print("=" * 78)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Input manifest: {INPUT_MANIFEST_PATH}")
    print("\nWhat do you want to do?")
    print("  1. generate all anti-forensic transformations [default]")
    print("  2. generate one or more selected transformations")
    print("  3. smoke test all transformations (--limit 10)")
    print("  4. evaluate existing anti-forensic manifest")

    selection = ask_choice("Selection", {"1", "2", "3", "4"}, "1")
    selected_attacks = ATTACK_NAMES.copy()
    limit = 0
    evaluate_only = selection == "4"

    if selection == "2":
        selected_attacks = ask_attack_selection()
    elif selection == "3":
        selected_attacks = ATTACK_NAMES.copy()
        limit = 10

    if evaluate_only:
        force = False
        evaluate = True
    else:
        force = ask_yes_no("Overwrite existing selected output directories if present?", default=True)
        evaluate = ask_yes_no("Evaluate generated images on a model after generation?", default=False)

    verbose = ask_yes_no("Enable verbose logging?", default=False)

    checkpoint_path = ""
    target_model = "efficientnet_b0"
    if evaluate:
        checkpoint_path = input(
            "Checkpoint path [auto-search models/checkpoints/*efficientnet_b0*]: "
        ).strip()

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
        evaluate=evaluate,
        evaluate_only=evaluate_only,
        target_model=target_model,
        checkpoint_path=checkpoint_path,
        eval_batch_size=32,
        device="auto",
        label_order=DEFAULT_LABEL_ORDER,
    )


# =============================================================================
# Argument parsing and logging
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally evaluate anti-forensic image transformations."
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
        help="One or more anti-forensic transformations to generate.",
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
        help="Optional maximum number of clean manifest rows to process for smoke tests.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Evaluation options.
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate generated anti-forensic images on the selected target model.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip generation and evaluate the existing "
            "attacks/manifests/anti_forensic_attacks_manifest.csv."
        ),
    )
    parser.add_argument(
        "--target-model",
        choices=SUPPORTED_TARGET_MODELS,
        default="efficientnet_b0",
        help="Target model used for evaluation.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to the trained target-model checkpoint. If omitted, an auto-search is attempted.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size used during model inference (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--label-order",
        nargs=2,
        default=DEFAULT_LABEL_ORDER,
        help="Class index order used by the classifier (default: non_weapon weapon).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1:
        return parse_interactive_args()
    parser = build_parser()
    args = parser.parse_args()
    if args.interactive:
        return parse_interactive_args()
    if args.evaluate_only:
        args.evaluate = True
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


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
    return resolve_repo_path(split_relative_path)


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

    existing_attack_dirs = [
        ANTI_FORENSIC_DIR / attack
        for attack in selected_attacks
        if (ANTI_FORENSIC_DIR / attack).exists()
    ]

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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 6)


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


def build_generation_summary(
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


# =============================================================================
# Evaluation logic
# =============================================================================

def evaluation_output_paths(target_model: str) -> tuple[Path, Path]:
    csv_path = ATTACK_MANIFESTS_DIR / f"anti_forensic_{target_model}_evaluation.csv"
    json_path = ATTACK_MANIFESTS_DIR / f"anti_forensic_{target_model}_evaluation_summary.json"
    return csv_path, json_path


def get_torch_device(device_arg: str) -> Any:
    import torch

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_fold_checkpoints(target_model: str, checkpoint_path: str | None = None) -> dict[str, Path]:
    """
    Resolve fold-aware checkpoints for a target model.

    Expected default layout:
        models/checkpoints/<target_model>/fold_1.pt
        models/checkpoints/<target_model>/fold_2.pt
        ...
        models/checkpoints/<target_model>/fold_5.pt

    If checkpoint_path is provided and points to a directory, the function searches
    fold_*.pt inside that directory.
    """
    if checkpoint_path:
        base_path = repo_relative_path(checkpoint_path)
    else:
        base_path = REPO_ROOT / "models" / "checkpoints" / target_model

    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base_path}")

    if base_path.is_file():
        raise ValueError(
            "A single checkpoint file was provided, but fold-aware evaluation requires "
            "a directory containing fold_1.pt ... fold_5.pt."
        )

    fold_checkpoints: dict[str, Path] = {}

    for fold_idx in range(1, 6):
        fold_name = f"fold_{fold_idx}"
        candidates = [
            base_path / f"{fold_name}.pt",
            base_path / f"{fold_name}.pth",
            base_path / f"{fold_name}.ckpt",
        ]

        found = next((path for path in candidates if path.exists() and path.is_file()), None)

        if found is None:
            raise FileNotFoundError(
                f"Missing checkpoint for {fold_name} under {base_path}. "
                f"Expected one of: {', '.join(str(path) for path in candidates)}"
            )

        fold_checkpoints[fold_name] = found

    logging.info("Fold-aware checkpoints resolved:")
    for fold_name, path in sorted(fold_checkpoints.items()):
        logging.info("  %s -> %s", fold_name, path)

    return fold_checkpoints

def find_checkpoint(target_model: str, checkpoint_path: str) -> Path:
    if checkpoint_path:
        path = resolve_repo_path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not CHECKPOINT_ROOT.exists():
        raise FileNotFoundError(
            f"Checkpoint root not found: {CHECKPOINT_ROOT}. Use --checkpoint-path."
        )

    patterns = [f"*{target_model}*.pt", f"*{target_model}*.pth", f"*{target_model}*.ckpt"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(CHECKPOINT_ROOT.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for target_model={target_model} under {CHECKPOINT_ROOT}. "
            "Use --checkpoint-path."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    selected = candidates[0]
    logging.info("Auto-selected checkpoint: %s", selected)
    return selected


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dictionary/state_dict.")

    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return normalize_state_dict_keys(value)

    return normalize_state_dict_keys(checkpoint)


def infer_num_classes_from_state_dict(state_dict: dict[str, Any], default: int = 2) -> int:
    for key in ("classifier.1.weight", "module.classifier.1.weight", "model.classifier.1.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    return default


def load_target_model(target_model: str, checkpoint_path: Path, device: Any) -> Any:
    import torch
    import torch.nn as nn
    from torchvision import models

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    if target_model == "efficientnet_b0":
        num_classes = infer_num_classes_from_state_dict(state_dict, default=2)
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported target model: {target_model}")

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logging.warning("Missing checkpoint keys: %s", load_result.missing_keys[:20])
    if load_result.unexpected_keys:
        logging.warning("Unexpected checkpoint keys: %s", load_result.unexpected_keys[:20])

    model.to(device)
    model.eval()
    return model


def build_preprocess() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def predict_paths(
    paths: list[Path],
    model: Any,
    preprocess: Any,
    device: Any,
    label_order: list[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    predictions: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_paths in batched(paths, batch_size):
            tensors = []
            for path in batch_paths:
                img = open_rgb_image(path)
                tensors.append(preprocess(img))

            batch = torch.stack(tensors).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).detach().cpu()

            for prob in probs:
                pred_idx = int(torch.argmax(prob).item())
                pred_label = label_order[pred_idx] if pred_idx < len(label_order) else str(pred_idx)
                confidence = float(prob[pred_idx].item())
                row = {
                    "prediction_index": pred_idx,
                    "prediction_label": pred_label,
                    "confidence": confidence,
                }
                for idx, label in enumerate(label_order):
                    if idx < len(prob):
                        row[f"prob_{label}"] = float(prob[idx].item())
                predictions.append(row)

    return predictions


def load_anti_forensic_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Anti-forensic manifest not found: {path}")

    df = pd.read_csv(path)
    required = {
        "generated_image_id",
        "original_image_id",
        "fold",
        "final_label",
        "clean_relative_path",
        "perturbed_relative_path",
        "attack_name",
    }
    ensure_required_columns(df, required, path.name)

    if df.empty:
        raise ValueError(f"Anti-forensic manifest is empty: {path}")

    invalid_labels = set(df["final_label"].map(norm).unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(f"Invalid labels in anti-forensic manifest: {sorted(invalid_labels)}")

    return df


def build_metric_block(df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df))
    if total == 0:
        return {
            "total": 0,
            "clean_correct": 0,
            "manipulated_correct": 0,
            "clean_accuracy": None,
            "manipulated_accuracy": None,
            "accuracy_drop": None,
            "induced_error_count": 0,
            "induced_error_rate_clean_correct": None,
            "weapon_to_non_weapon_count": 0,
            "weapon_to_non_weapon_rate_clean_correct_weapon": None,
            "non_weapon_to_weapon_count": 0,
            "non_weapon_to_weapon_rate_clean_correct_non_weapon": None,
        }

    clean_correct = int(df["original_correct"].sum())
    manipulated_correct = int(df["anti_forensic_correct"].sum())
    induced_error_count = int(df["manipulation_induced_error"].sum())

    weapon_clean_correct = df[(df["final_label"] == "weapon") & (df["original_correct"])]
    non_weapon_clean_correct = df[(df["final_label"] == "non_weapon") & (df["original_correct"])]

    weapon_to_non_weapon_count = int(df["weapon_to_non_weapon"].sum())
    non_weapon_to_weapon_count = int(df["non_weapon_to_weapon"].sum())

    clean_accuracy = clean_correct / total if total else None
    manipulated_accuracy = manipulated_correct / total if total else None
    accuracy_drop = None
    if clean_accuracy is not None and manipulated_accuracy is not None:
        accuracy_drop = clean_accuracy - manipulated_accuracy

    induced_error_rate = induced_error_count / clean_correct if clean_correct else None
    weapon_to_non_weapon_rate = (
        weapon_to_non_weapon_count / len(weapon_clean_correct)
        if len(weapon_clean_correct) else None
    )
    non_weapon_to_weapon_rate = (
        non_weapon_to_weapon_count / len(non_weapon_clean_correct)
        if len(non_weapon_clean_correct) else None
    )

    return {
        "total": total,
        "clean_correct": clean_correct,
        "manipulated_correct": manipulated_correct,
        "clean_accuracy": safe_float(clean_accuracy),
        "manipulated_accuracy": safe_float(manipulated_accuracy),
        "accuracy_drop": safe_float(accuracy_drop),
        "induced_error_count": induced_error_count,
        "induced_error_rate_clean_correct": safe_float(induced_error_rate),
        "weapon_to_non_weapon_count": weapon_to_non_weapon_count,
        "weapon_to_non_weapon_rate_clean_correct_weapon": safe_float(weapon_to_non_weapon_rate),
        "non_weapon_to_weapon_count": non_weapon_to_weapon_count,
        "non_weapon_to_weapon_rate_clean_correct_non_weapon": safe_float(non_weapon_to_weapon_rate),
        "clean_confidence_mean": safe_float(df["original_confidence"].mean()),
        "manipulated_confidence_mean": safe_float(df["anti_forensic_confidence"].mean()),
        "confidence_delta_mean": safe_float(df["confidence_delta"].mean()),
    }


def nested_group_metrics(df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value, group in df.groupby(group_col):
        output[str(value)] = build_metric_block(group)
    return output

def build_evaluation_summary(
    eval_df: pd.DataFrame,
    target_model: str,
    fold_checkpoints: dict[str, Path],
    evaluation_csv_path: Path,
    evaluation_summary_path: Path,
    created_at: str,
) -> dict[str, Any]:
    per_attack_per_fold: dict[str, dict[str, Any]] = {}
    for attack_name, attack_group in eval_df.groupby("attack_name"):
        per_attack_per_fold[str(attack_name)] = nested_group_metrics(attack_group, "fold")

    per_attack_per_label: dict[str, dict[str, Any]] = {}
    for attack_name, attack_group in eval_df.groupby("attack_name"):
        per_attack_per_label[str(attack_name)] = nested_group_metrics(attack_group, "final_label")

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "target_model": target_model,
        "checkpoint_mode": "fold_aware",
        "fold_checkpoints": {
            fold: repo_relative_string(path)
            for fold, path in sorted(fold_checkpoints.items())
        },
        "input_manifest": repo_relative_string(ANTI_FORENSIC_MANIFEST_PATH),
        "evaluation_csv": repo_relative_string(evaluation_csv_path),
        "evaluation_summary_json": repo_relative_string(evaluation_summary_path),
        "methodological_note": (
            "Metrics are computed using fold-aware checkpoints. Each image is evaluated "
            "with the model checkpoint corresponding to its fold. The induced error rate "
            "excludes images already misclassified in clean condition."
        ),
        "global_metrics": build_metric_block(eval_df),
        "per_attack_metrics": nested_group_metrics(eval_df, "attack_name"),
        "per_label_metrics": nested_group_metrics(eval_df, "final_label"),
        "per_fold_metrics": nested_group_metrics(eval_df, "fold"),
        "per_attack_per_fold_metrics": per_attack_per_fold,
        "per_attack_per_label_metrics": per_attack_per_label,
    }

def evaluate_anti_forensic_manifest(args: argparse.Namespace, created_at: str) -> None:
    target_model = args.target_model
    label_order = [norm(label) for label in args.label_order]

    if set(label_order) != VALID_LABELS:
        raise ValueError(
            f"--label-order must contain exactly {sorted(VALID_LABELS)}. Got: {label_order}"
        )

    import torch  # noqa: F401 - imported here to fail only when evaluation is requested.

    device = get_torch_device(args.device)

    fold_checkpoints = find_fold_checkpoints(
        target_model=target_model,
        checkpoint_path=args.checkpoint_path,
    )

    logging.info("Evaluating anti-forensic manifest with target model: %s", target_model)
    logging.info("Device: %s", device)
    logging.info("Checkpoint mode: fold-aware")

    preprocess = build_preprocess()
    manifest_df = load_anti_forensic_manifest(ANTI_FORENSIC_MANIFEST_PATH)

    # Validate image paths before inference.
    clean_paths_all = [resolve_repo_path(path) for path in manifest_df["clean_relative_path"].tolist()]
    perturbed_paths_all = [resolve_repo_path(path) for path in manifest_df["perturbed_relative_path"].tolist()]

    for path in clean_paths_all + perturbed_paths_all:
        if not path.exists():
            raise FileNotFoundError(f"Image not found during evaluation: {path}")

    # Model cache: each fold uses its own checkpoint.
    models_by_fold: dict[str, Any] = {}

    def get_model_for_fold(fold: str) -> Any:
        if fold not in fold_checkpoints:
            raise KeyError(f"No checkpoint available for fold={fold}")

        if fold not in models_by_fold:
            checkpoint_for_fold = fold_checkpoints[fold]
            logging.info("Loading model for %s from %s", fold, checkpoint_for_fold)
            models_by_fold[fold] = load_target_model(
                target_model=target_model,
                checkpoint_path=checkpoint_for_fold,
                device=device,
            )

        return models_by_fold[fold]

    # Clean prediction cache.
    # Each clean image belongs to one fold, so it must be evaluated with that fold checkpoint.
    clean_cache: dict[str, dict[str, Any]] = {}

    clean_df = (
        manifest_df[["original_image_id", "fold", "clean_relative_path"]]
        .drop_duplicates("original_image_id")
        .copy()
    )

    logging.info("Predicting clean images fold-aware: %d unique files", len(clean_df))

    for fold, fold_clean_df in clean_df.groupby("fold"):
        fold = safe_str(fold)
        model = get_model_for_fold(fold)

        fold_clean_paths = [
            resolve_repo_path(path)
            for path in fold_clean_df["clean_relative_path"].tolist()
        ]

        fold_clean_predictions = predict_paths(
            paths=fold_clean_paths,
            model=model,
            preprocess=preprocess,
            device=device,
            label_order=label_order,
            batch_size=args.eval_batch_size,
        )

        for (_, clean_row), pred in zip(fold_clean_df.iterrows(), fold_clean_predictions):
            clean_cache[safe_str(clean_row["original_image_id"])] = pred

    # Perturbed predictions.
    logging.info("Predicting anti-forensic images fold-aware: %d files", len(manifest_df))

    evaluation_rows: list[dict[str, Any]] = []

    for fold, fold_manifest_df in manifest_df.groupby("fold"):
        fold = safe_str(fold)
        model = get_model_for_fold(fold)

        fold_perturbed_paths = [
            resolve_repo_path(path)
            for path in fold_manifest_df["perturbed_relative_path"].tolist()
        ]

        fold_perturbed_predictions = predict_paths(
            paths=fold_perturbed_paths,
            model=model,
            preprocess=preprocess,
            device=device,
            label_order=label_order,
            batch_size=args.eval_batch_size,
        )

        for (_, row), pert_pred in zip(fold_manifest_df.iterrows(), fold_perturbed_predictions):
            original_image_id = safe_str(row["original_image_id"])
            final_label = norm(row["final_label"])
            clean_pred = clean_cache[original_image_id]

            original_prediction = clean_pred["prediction_label"]
            anti_forensic_prediction = pert_pred["prediction_label"]

            original_correct = original_prediction == final_label
            anti_forensic_correct = anti_forensic_prediction == final_label

            manipulation_induced_error = original_correct and not anti_forensic_correct
            recovered_by_manipulation = (not original_correct) and anti_forensic_correct

            weapon_to_non_weapon = (
                final_label == "weapon"
                and original_correct
                and anti_forensic_prediction == "non_weapon"
            )

            non_weapon_to_weapon = (
                final_label == "non_weapon"
                and original_correct
                and anti_forensic_prediction == "weapon"
            )

            original_confidence = float(clean_pred["confidence"])
            anti_forensic_confidence = float(pert_pred["confidence"])

            out = dict(row.to_dict())
            out.update(
                {
                    "target_model": target_model,
                    "checkpoint_mode": "fold_aware",
                    "checkpoint_path": repo_relative_string(fold_checkpoints[fold]),
                    "original_prediction": original_prediction,
                    "anti_forensic_prediction": anti_forensic_prediction,
                    "original_confidence": original_confidence,
                    "anti_forensic_confidence": anti_forensic_confidence,
                    "confidence_delta": anti_forensic_confidence - original_confidence,
                    "original_correct": original_correct,
                    "anti_forensic_correct": anti_forensic_correct,
                    "manipulation_induced_error": manipulation_induced_error,
                    "recovered_by_manipulation": recovered_by_manipulation,
                    "weapon_to_non_weapon": weapon_to_non_weapon,
                    "non_weapon_to_weapon": non_weapon_to_weapon,
                    "evaluation_created_at": created_at,
                }
            )

            for label in label_order:
                out[f"original_prob_{label}"] = clean_pred.get(f"prob_{label}", "")
                out[f"anti_forensic_prob_{label}"] = pert_pred.get(f"prob_{label}", "")

            evaluation_rows.append(out)

    evaluation_csv_path, evaluation_summary_path = evaluation_output_paths(target_model)

    eval_df = pd.DataFrame(evaluation_rows)
    evaluation_csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(evaluation_csv_path, index=False, encoding="utf-8")

    summary = build_evaluation_summary(
        eval_df=eval_df,
        target_model=target_model,
        fold_checkpoints=fold_checkpoints,
        evaluation_csv_path=evaluation_csv_path,
        evaluation_summary_path=evaluation_summary_path,
        created_at=created_at,
    )

    write_summary(evaluation_summary_path, summary)

    logging.info("Evaluation CSV written: %s", evaluation_csv_path)
    logging.info("Evaluation summary written: %s", evaluation_summary_path)
    logging.info("Global metrics: %s", json.dumps(summary["global_metrics"], indent=2))

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    created_at = utc_now_iso()

    if args.evaluate_only:
        logging.info("Evaluate-only mode enabled. Generation will be skipped.")
        evaluate_anti_forensic_manifest(args=args, created_at=created_at)
        return

    selected_attacks = list(dict.fromkeys(args.attack))
    input_manifest = repo_relative_path(args.input_manifest)

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

    summary = build_generation_summary(
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

    if args.evaluate:
        evaluate_anti_forensic_manifest(args=args, created_at=created_at)


if __name__ == "__main__":
    main()
