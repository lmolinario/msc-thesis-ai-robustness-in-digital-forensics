#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_generate_adversarial_attacks.py

Fold-aware adversarial generation entry point for the FAIR-Lab thesis pipeline.

This script supports two equivalent execution modes:

1. Interactive mode, automatically enabled when the script is launched without
   command-line arguments.
2. Command-line mode, used for fully reproducible scripted execution.

Operational protocol
--------------------
- EfficientNet-B0 is the default primary proxy target.
- Model-dependent attacks use fold-aware checkpoints from:
  models/checkpoints/<target_model>/<fold>.pt
- FGSM outputs are saved as lossless PNG files.
- Color Shift remains a model-agnostic JPEG perturbation.
- The manifest records checkpoint path and checkpoint SHA256 for reproducibility.

Implemented attacks:
- fgsm
- color_shift
- one_pixel

Planned but intentionally not implemented here:
- sigma_zero
- superdeepfool
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
from typing import Any

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))


import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from datasets.scripts.attacks.adversarial_model_interface import (
    IMPLEMENTED_ATTACK_NAMES,
    MODEL_AGNOSTIC_TARGET,
    PLANNED_ATTACK_NAMES,
    SUPPORTED_TARGET_MODELS,
    TargetModelAdapter,
    TargetModelConfig,
    VALID_BINARY_LABELS,
    is_model_dependent_attack,
    validate_target_model_names,
)
from datasets.scripts.attacks.adversarial_torch_model_adapters import build_target_model_adapter
from datasets.scripts.utils.paths import (
    ADVERSARIAL_DIR,
    ATTACKS_DIR,
    REPO_ROOT,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)


SCRIPT_NAME = "datasets/scripts/attacks/14_generate_adversarial_attacks.py"
INPUT_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"

VALID_LABELS = set(VALID_BINARY_LABELS)
DEFAULT_ATTACKS = ["color_shift"]
DEFAULT_TARGET_MODELS = ["efficientnet_b0"]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"


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


def ask_target_models(default_models: list[str]) -> list[str]:
    """Ask for one or more FGSM target models."""
    print("\nSelect target models for FGSM:")
    print("  1. efficientnet_b0 [recommended primary target]")
    print("  2. resnet18")
    print("  3. clip")
    print("Examples: 1 | 1 2 | 1,2,3 | all")
    value = input("Selection [1]: ").strip().lower()
    if not value:
        return default_models
    if value == "all":
        return ["efficientnet_b0", "resnet18", "clip"]

    mapping = {"1": "efficientnet_b0", "2": "resnet18", "3": "clip"}
    tokens = value.replace(",", " ").split()
    selected: list[str] = []
    for token in tokens:
        if token in mapping:
            selected.append(mapping[token])
        elif token in SUPPORTED_TARGET_MODELS:
            selected.append(token)
        else:
            raise ValueError(f"Invalid target model selection: {token}")
    return list(dict.fromkeys(selected))


def parse_interactive_args() -> argparse.Namespace:
    """Build an argparse namespace through a safe interactive launcher."""
    print("\n" + "=" * 78)
    print("FAIR-Lab adversarial attack generator")
    print("=" * 78)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Input manifest: {INPUT_MANIFEST_PATH}")
    print("\nWhat do you want to generate?")
    print("  1. color_shift only (model-agnostic, no checkpoints required) [default]")
    print("  2. FGSM smoke test on efficientnet_b0 (--limit 10)")
    print("  3. FGSM full generation on efficientnet_b0")
    print("  4. One Pixel smoke test on efficientnet_b0 (--limit 10)")
    print("  5. One Pixel full generation on efficientnet_b0")
    print("  6. custom FGSM target selection")

    selection = ask_choice("Selection", {"1", "2", "3", "4", "5", "6"}, "1")

    attacks = ["color_shift"]
    target_models = DEFAULT_TARGET_MODELS.copy()
    limit = 0

    if selection == "1":
        attacks = ["color_shift"]
    elif selection == "2":
        attacks = ["fgsm"]
        target_models = ["efficientnet_b0"]
        limit = 10
    elif selection == "3":
        attacks = ["fgsm"]
        target_models = ["efficientnet_b0"]
    elif selection == "4":
        attacks = ["one_pixel"]
        target_models = ["efficientnet_b0"]
        limit = 10
    elif selection == "5":
        attacks = ["one_pixel"]
        target_models = ["efficientnet_b0"]
    elif selection == "6":
        attacks = ["fgsm"]
        target_models = ask_target_models(DEFAULT_TARGET_MODELS)
        limit = ask_int("Limit rows for smoke test; use 0 for full generation", 10, minimum=0)

    force = ask_yes_no("Overwrite existing selected output directories if present?", default=False)
    verbose = ask_yes_no("Enable verbose logging?", default=False)

    return argparse.Namespace(
        input_manifest=str(INPUT_MANIFEST_PATH),
        attack=attacks,
        target_model=target_models,
        checkpoint_root=str(DEFAULT_CHECKPOINT_ROOT),
        device="auto",
        input_size=224,
        fgsm_epsilon=8.0 / 255.0,
        force=force,
        jpeg_quality=95,
        limit=limit,
        color_red_shift=12,
        color_green_shift=0,
        color_blue_shift=-12,
        color_saturation_factor=1.10,
        color_contrast_factor=1.00,
        one_pixel_max_iterations=30,
        one_pixel_population_size=8,
        one_pixel_seed=42,
        verbose=verbose,
    )


# =============================================================================
# Argument parsing and logging
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fold-aware adversarial/adversarial-style perturbations."
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
        choices=PLANNED_ATTACK_NAMES,
        default=DEFAULT_ATTACKS,
        help="Attack(s) to generate. Implemented: fgsm, color_shift, one_pixel. Default: color_shift.",
    )
    parser.add_argument(
        "--target-model",
        nargs="+",
        choices=SUPPORTED_TARGET_MODELS,
        default=DEFAULT_TARGET_MODELS,
        help="Target proxy model(s) for model-dependent attacks. Default: efficientnet_b0.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Root directory for fold-aware checkpoints: <root>/<target_model>/<fold>.pt.",
    )

    parser.add_argument(
        "--one-pixel-max-iterations",
        type=int,
        default=30,
        help="Maximum number of Differential Evolution iterations for the One Pixel attack.",
    )
    parser.add_argument(
        "--one-pixel-population-size",
        type=int,
        default=8,
        help=(
            "Differential Evolution population-size multiplier. "
            "The effective population is approximately population_size * 5."
        ),
    )
    parser.add_argument(
        "--one-pixel-seed",
        type=int,
        default=42,
        help="Base random seed for deterministic per-image One Pixel optimization.",
    )


    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--fgsm-epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    parser.add_argument("--color-red-shift", type=int, default=12)
    parser.add_argument("--color-green-shift", type=int, default=0)
    parser.add_argument("--color-blue-shift", type=int, default=-12)
    parser.add_argument("--color-saturation-factor", type=float, default=1.10)
    parser.add_argument("--color-contrast-factor", type=float, default=1.00)
    parser.add_argument("--verbose", action="store_true")
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
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


def repo_relative_string(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def build_run_id(selected_attacks: list[str], selected_target_models: list[str]) -> str:
    """
    Build a stable identifier for the current adversarial generation run.

    The identifier is used to avoid overwriting manifests generated by
    different attacks or target-model configurations.
    """
    attack_part = "_".join(selected_attacks)

    if selected_attacks == ["color_shift"]:
        return "adversarial_color_shift"

    if all(not is_model_dependent_attack(attack) for attack in selected_attacks):
        return f"adversarial_{attack_part}"

    target_part = "_".join(selected_target_models)
    return f"adversarial_{attack_part}_{target_part}"


def build_run_manifest_paths(
    selected_attacks: list[str],
    selected_target_models: list[str],
) -> tuple[Path, Path]:
    """
    Return run-specific manifest and summary paths.

    This prevents a color_shift run from overwriting a previous FGSM manifest,
    and vice versa.
    """
    run_id = build_run_id(selected_attacks, selected_target_models)
    manifest_path = ATTACK_MANIFESTS_DIR / f"{run_id}_manifest.csv"
    summary_path = ATTACK_MANIFESTS_DIR / f"{run_id}_summary.json"
    return manifest_path, summary_path


def validate_manifest_outputs(manifest_path: Path, summary_path: Path, force: bool) -> None:
    """
    Prevent accidental overwrite of existing manifest artifacts unless --force
    is explicitly requested.
    """
    existing = [path for path in (manifest_path, summary_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Manifest output files already exist. Use --force to overwrite them: "
            + ", ".join(str(path) for path in existing)
        )

def compute_hashes(file_path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def validate_selected_attacks(selected_attacks: list[str]) -> None:
    not_implemented = [name for name in selected_attacks if name not in IMPLEMENTED_ATTACK_NAMES]
    if not_implemented:
        raise NotImplementedError(
            "The following adversarial attacks are planned but not implemented yet: "
            f"{', '.join(not_implemented)}."
        )


def validate_attack_parameters(args: argparse.Namespace) -> None:
    if not (0.0 < args.fgsm_epsilon <= 1.0):
        raise ValueError("--fgsm-epsilon must be in the interval (0, 1].")
    if args.input_size <= 0:
        raise ValueError("--input-size must be greater than 0.")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if args.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0.")
    if args.one_pixel_max_iterations <= 0:
        raise ValueError("--one-pixel-max-iterations must be greater than 0.")
    if args.one_pixel_population_size <= 0:
        raise ValueError("--one-pixel-population-size must be greater than 0.")


def load_manifest(path: Path, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"image_id", "fold", "final_label", "source_dataset", "split_relative_path", "sha256", "md5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")
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
            mismatches.append(f"{image_id} SHA256 expected={expected_sha256} computed={computed_sha256}")
        if expected_md5 and computed_md5 != expected_md5:
            mismatches.append(f"{image_id} MD5 expected={expected_md5} computed={computed_md5}")
    if missing:
        raise FileNotFoundError("Missing clean image files:\n" + "\n".join(missing[:20]))
    if mismatches:
        raise RuntimeError("Hash mismatches found in clean inputs:\n" + "\n".join(mismatches[:20]))


def prepare_output_dirs(selected_attacks: list[str], force: bool) -> None:
    ADVERSARIAL_DIR.mkdir(parents=True, exist_ok=True)
    ATTACK_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [ADVERSARIAL_DIR / attack for attack in selected_attacks if (ADVERSARIAL_DIR / attack).exists()]
    if existing and not force:
        raise FileExistsError("Selected attack output directories already exist. Use --force: " + ", ".join(map(str, existing)))
    if force:
        for attack in selected_attacks:
            attack_dir = ADVERSARIAL_DIR / attack
            if attack_dir.exists():
                logging.warning("Removing existing attack output directory: %s", attack_dir)
                shutil.rmtree(attack_dir)


def open_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {path}") from exc


def save_jpeg(img: Image.Image, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=quality, optimize=True, progressive=False)


def save_png(img: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=False)


def resolve_fold_checkpoint(checkpoint_root: Path, target_model: str, fold: str) -> Path:
    checkpoint_path = checkpoint_root / target_model / f"{fold}.pt"
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing fold-aware checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_adapter(target_model: str, fold: str, checkpoint_root: Path, args: argparse.Namespace) -> TargetModelAdapter:
    checkpoint_path = resolve_fold_checkpoint(checkpoint_root, target_model, fold)
    config = TargetModelConfig(
        name=target_model,
        checkpoint_path=checkpoint_path,
        device=args.device,
        input_size=args.input_size,
    )
    adapter = build_target_model_adapter(config)
    adapter.load_model()
    logging.info("Loaded %s %s from %s", target_model, fold, repo_relative_string(checkpoint_path))
    return adapter


def load_fold_aware_adapters(
    df: pd.DataFrame,
    target_models: list[str],
    checkpoint_root: Path,
    args: argparse.Namespace,
) -> dict[tuple[str, str], TargetModelAdapter]:
    adapters: dict[tuple[str, str], TargetModelAdapter] = {}
    folds = sorted({safe_str(value) for value in df["fold"].unique()})
    for model_name in target_models:
        for fold in folds:
            adapters[(model_name, fold)] = load_adapter(model_name, fold, checkpoint_root, args)
    return adapters


def adapter_normalization(adapter: TargetModelAdapter) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if hasattr(adapter, "imagenet_mean") and hasattr(adapter, "imagenet_std"):
        return tuple(adapter.imagenet_mean), tuple(adapter.imagenet_std)
    if hasattr(adapter, "clip_mean") and hasattr(adapter, "clip_std"):
        return tuple(adapter.clip_mean), tuple(adapter.clip_std)
    raise RuntimeError(f"Adapter {adapter.name!r} does not expose normalization statistics.")


def normalization_tensors(model_input: Any, adapter: TargetModelAdapter) -> tuple[Any, Any]:
    mean, std = adapter_normalization(adapter)
    torch_module = __import__("torch")
    mean_tensor = torch_module.tensor(mean, dtype=model_input.dtype, device=model_input.device).view(1, 3, 1, 1)
    std_tensor = torch_module.tensor(std, dtype=model_input.dtype, device=model_input.device).view(1, 3, 1, 1)
    return mean_tensor, std_tensor


def denormalize_tensor(model_input: Any, adapter: TargetModelAdapter) -> Any:
    mean_tensor, std_tensor = normalization_tensors(model_input, adapter)
    return model_input * std_tensor + mean_tensor


def normalize_tensor(pixel_tensor: Any, adapter: TargetModelAdapter) -> Any:
    mean_tensor, std_tensor = normalization_tensors(pixel_tensor, adapter)
    return (pixel_tensor - mean_tensor) / std_tensor


def tensor_to_rgb_image(pixel_tensor: Any) -> Image.Image:
    tensor = pixel_tensor.detach().clamp(0.0, 1.0)[0].cpu()
    array = tensor.permute(1, 2, 0).numpy()
    array_uint8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def tensor_to_pixel_array(pixel_tensor: Any) -> np.ndarray:
    tensor = pixel_tensor.detach().clamp(0.0, 1.0)[0].cpu()
    return tensor.permute(1, 2, 0).numpy().astype(np.float32)


def compute_perturbation_metrics_from_arrays(original: np.ndarray, transformed: np.ndarray) -> dict[str, float | int]:
    if original.shape != transformed.shape:
        raise ValueError(f"Shape mismatch: original={original.shape}, transformed={transformed.shape}")
    diff = transformed.astype(np.float32) - original.astype(np.float32)
    abs_diff = np.abs(diff)
    return {
        "perturbation_norm_l0": int(np.count_nonzero(abs_diff)),
        "perturbation_norm_l2": float(np.linalg.norm(diff.ravel(), ord=2)),
        "perturbation_norm_linf": float(np.max(abs_diff)),
        "perturbation_mean_abs": float(np.mean(abs_diff)),
    }


def compute_perturbation_metrics(original_img: Image.Image, transformed_img: Image.Image) -> dict[str, float | int]:
    original = np.asarray(original_img.convert("RGB"), dtype=np.float32)
    transformed = np.asarray(transformed_img.convert("RGB"), dtype=np.float32)
    return compute_perturbation_metrics_from_arrays(original, transformed)


def confidence_for_prediction(probabilities: dict[str, float], prediction: str) -> float:
    return float(probabilities.get(prediction, 0.0))


def _shift_channel(channel: Image.Image, shift: int) -> Image.Image:
    lookup = [max(0, min(255, value + shift)) for value in range(256)]
    return channel.point(lookup)


def apply_color_shift(img: Image.Image, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any], str]:
    red, green, blue = img.convert("RGB").split()
    shifted = Image.merge(
        "RGB",
        (
            _shift_channel(red, args.color_red_shift),
            _shift_channel(green, args.color_green_shift),
            _shift_channel(blue, args.color_blue_shift),
        ),
    )
    shifted = ImageEnhance.Color(shifted).enhance(args.color_saturation_factor)
    shifted = ImageEnhance.Contrast(shifted).enhance(args.color_contrast_factor)
    params = {
        "red_shift": args.color_red_shift,
        "green_shift": args.color_green_shift,
        "blue_shift": args.color_blue_shift,
        "metric_space": "pixel_[0,255]",
        "saturation_factor": args.color_saturation_factor,
        "contrast_factor": args.color_contrast_factor,
        "output_format": "JPEG",
        "jpeg_quality": args.jpeg_quality,
    }
    return shifted, params, MODEL_AGNOSTIC_TARGET


def apply_fgsm(img: Image.Image, true_label: str, adapter: TargetModelAdapter, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    model_input = adapter.preprocess_image(img)
    original_probabilities = adapter.predict_proba(model_input)
    original_prediction = adapter.predict(model_input)
    original_confidence = confidence_for_prediction(original_probabilities, original_prediction)

    gradient_normalized = adapter.compute_gradient(model_input, true_label)
    original_pixel_tensor = denormalize_tensor(model_input, adapter).clamp(0.0, 1.0)
    _, std_tensor = normalization_tensors(model_input, adapter)

    gradient_pixel_space = gradient_normalized / std_tensor
    adversarial_pixel_tensor = (original_pixel_tensor + args.fgsm_epsilon * gradient_pixel_space.sign()).clamp(0.0, 1.0)
    adversarial_model_input = normalize_tensor(adversarial_pixel_tensor, adapter)

    adversarial_probabilities = adapter.predict_proba(adversarial_model_input)
    adversarial_prediction = adapter.predict(adversarial_model_input)
    adversarial_confidence = confidence_for_prediction(adversarial_probabilities, adversarial_prediction)

    metrics = compute_perturbation_metrics_from_arrays(
        tensor_to_pixel_array(original_pixel_tensor),
        tensor_to_pixel_array(adversarial_pixel_tensor),
    )
    params = {
        "epsilon": args.fgsm_epsilon,
        "epsilon_space": "pixel_[0,1]",
        "metric_space": "pixel_[0,1]",
        "attack_type": "untargeted",
        "target_model": adapter.name,
        "input_size": args.input_size,
        "output_format": "PNG",
        "original_prediction": original_prediction,
        "adversarial_prediction": adversarial_prediction,
        "original_confidence": original_confidence,
        "adversarial_confidence": adversarial_confidence,
        "original_true_label_probability": float(original_probabilities.get(true_label, 0.0)),
        "adversarial_true_label_probability": float(adversarial_probabilities.get(true_label, 0.0)),
        **metrics,
    }
    return tensor_to_rgb_image(adversarial_pixel_tensor), params

def stable_attack_seed(base_seed: int, image_id: str, attack_name: str) -> int:
    """
    Build a deterministic per-image seed for stochastic attacks.

    This avoids using the exact same candidate sequence for every image while
    preserving reproducibility across runs.
    """
    material = f"{base_seed}|{attack_name}|{image_id}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:8], 16)


def count_changed_pixels(original_img: Image.Image, transformed_img: Image.Image) -> int:
    """
    Count pixels whose RGB value changed in at least one channel.
    """
    original = np.asarray(original_img.convert("RGB"), dtype=np.int16)
    transformed = np.asarray(transformed_img.convert("RGB"), dtype=np.int16)
    changed = np.any(original != transformed, axis=2)
    return int(np.count_nonzero(changed))


def apply_one_pixel(
    img: Image.Image,
    true_label: str,
    adapter: TargetModelAdapter,
    args: argparse.Namespace,
    image_id: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Apply an untargeted One Pixel Attack using Differential Evolution.

    The attack optimizes five variables:

    - x coordinate
    - y coordinate
    - red channel
    - green channel
    - blue channel

    The objective minimizes the probability assigned by the target model to the
    true class. The procedure is score-based and uses only model probabilities,
    not gradients. A deterministic per-image seed is used for reproducibility.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for the Differential Evolution-based One Pixel attack. "
            "Install it with: python -m pip install scipy"
        ) from exc

    clean_img = img.convert("RGB")
    width, height = clean_img.size

    original_input = adapter.preprocess_image(clean_img)
    original_probabilities = adapter.predict_proba(original_input)
    original_prediction = adapter.predict(original_input)
    original_confidence = confidence_for_prediction(original_probabilities, original_prediction)
    original_true_probability = float(original_probabilities.get(true_label, 0.0))

    image_seed = stable_attack_seed(
        base_seed=args.one_pixel_seed,
        image_id=image_id,
        attack_name="one_pixel",
    )

    bounds = [
        (0, width - 1),
        (0, height - 1),
        (0, 255),
        (0, 255),
        (0, 255),
    ]

    best_img = clean_img.copy()
    best_prediction = original_prediction
    best_probabilities = original_probabilities
    best_true_probability = original_true_probability
    best_confidence = original_confidence
    best_x = -1
    best_y = -1
    best_rgb = (-1, -1, -1)
    best_candidate = [-1.0, -1.0, -1.0, -1.0, -1.0]

    queries_used = 0
    converged = False

    def decode_candidate(candidate: np.ndarray) -> tuple[int, int, tuple[int, int, int], list[float]]:
        """Convert a Differential Evolution candidate into integer pixel parameters."""
        x = int(np.clip(np.rint(candidate[0]), 0, width - 1))
        y = int(np.clip(np.rint(candidate[1]), 0, height - 1))
        rgb = tuple(int(np.clip(np.rint(value), 0, 255)) for value in candidate[2:5])
        candidate_list = [float(value) for value in candidate]
        return x, y, rgb, candidate_list

    def objective(candidate: np.ndarray) -> float:
        """
        Objective minimized by Differential Evolution.

        Lower values correspond to lower probability assigned to the true label.
        Misclassification is handled through early stopping in the callback.
        """
        nonlocal best_img
        nonlocal best_prediction
        nonlocal best_probabilities
        nonlocal best_true_probability
        nonlocal best_confidence
        nonlocal best_x
        nonlocal best_y
        nonlocal best_rgb
        nonlocal best_candidate
        nonlocal queries_used
        nonlocal converged

        x, y, rgb, candidate_list = decode_candidate(candidate)

        candidate_img = clean_img.copy()
        candidate_img.putpixel((x, y), rgb)

        candidate_input = adapter.preprocess_image(candidate_img)
        candidate_probabilities = adapter.predict_proba(candidate_input)
        candidate_prediction = adapter.predict(candidate_input)
        candidate_true_probability = float(candidate_probabilities.get(true_label, 0.0))
        candidate_confidence = confidence_for_prediction(
            candidate_probabilities,
            candidate_prediction,
        )

        queries_used += 1

        if candidate_true_probability < best_true_probability:
            best_img = candidate_img
            best_prediction = candidate_prediction
            best_probabilities = candidate_probabilities
            best_true_probability = candidate_true_probability
            best_confidence = candidate_confidence
            best_x = x
            best_y = y
            best_rgb = rgb
            best_candidate = candidate_list

        if candidate_prediction != true_label:
            best_img = candidate_img
            best_prediction = candidate_prediction
            best_probabilities = candidate_probabilities
            best_true_probability = candidate_true_probability
            best_confidence = candidate_confidence
            best_x = x
            best_y = y
            best_rgb = rgb
            best_candidate = candidate_list
            converged = True

        return candidate_true_probability

    def stop_callback(_xk: np.ndarray, _convergence: float) -> bool:
        """Stop Differential Evolution after the first successful misclassification."""
        return converged

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.one_pixel_max_iterations,
        popsize=args.one_pixel_population_size,
        seed=image_seed,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.0,
        atol=0.0,
        callback=stop_callback,
    )

    # Ensure that the optimizer's final candidate has been reflected in the
    # tracked best solution.
    objective(np.asarray(result.x, dtype=np.float64))

    metrics = compute_perturbation_metrics(clean_img, best_img)
    changed_pixel_count = count_changed_pixels(clean_img, best_img)

    optimizer_fun = result.fun
    if not np.isfinite(optimizer_fun):
        optimizer_fun = float("nan")

    params = {
        "attack_type": "differential_evolution_one_pixel",
        "target_model": adapter.name,
        "input_size": args.input_size,
        "output_format": "PNG",
        "metric_space": "pixel_[0,255]",
        "optimization_objective": "minimize_true_label_probability",
        "search_space": "x_y_rgb",
        "search_dimension": 5,
        "max_iterations": args.one_pixel_max_iterations,
        "population_size": args.one_pixel_population_size,
        "effective_population_size": args.one_pixel_population_size * 5,
        "polish": False,
        "updating": "immediate",
        "workers": 1,
        "base_seed": args.one_pixel_seed,
        "image_seed": image_seed,
        "queries_used": queries_used,
        "iterations_used": int(getattr(result, "nit", 0)),
        "optimizer_success": bool(getattr(result, "success", False)),
        "optimizer_message": str(getattr(result, "message", "")),
        "optimizer_fun": float(optimizer_fun),
        "converged": converged,
        "changed_pixel_count": changed_pixel_count,
        "best_x": best_x,
        "best_y": best_y,
        "best_rgb": list(best_rgb),
        "best_candidate": best_candidate,
        "original_prediction": original_prediction,
        "adversarial_prediction": best_prediction,
        "original_confidence": original_confidence,
        "adversarial_confidence": best_confidence,
        "original_true_label_probability": original_true_probability,
        "adversarial_true_label_probability": best_true_probability,
        "adversarial_probabilities": best_probabilities,
        **metrics,
    }

    return best_img, params

def output_extension_for_attack(attack_name: str) -> str:
    if attack_name in {"fgsm", "one_pixel", "sigma_zero", "superdeepfool"}:
        return ".png"
    return ".jpg"


def build_output_path(row: pd.Series, attack_name: str, target_model: str) -> Path:
    image_id = safe_str(row["image_id"])
    fold = safe_str(row["fold"])
    label = norm(row["final_label"])
    filename = f"{image_id}__{attack_name}__{target_model}{output_extension_for_attack(attack_name)}"
    return ADVERSARIAL_DIR / attack_name / target_model / fold / label / filename


def build_manifest_row(
    row: pd.Series,
    attack_name: str,
    target_model: str,
    output_path: Path,
    attack_params: dict[str, Any],
    created_at: str,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    image_id = safe_str(row["image_id"])
    label = norm(row["final_label"])
    fold = safe_str(row["fold"])
    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))

    sha256_perturbed, md5_perturbed = compute_hashes(output_path)
    sha256_original, md5_original = compute_hashes(source_path)
    checkpoint_sha256 = ""
    if checkpoint_path is not None:
        checkpoint_sha256, _ = compute_hashes(checkpoint_path)

    original_prediction = attack_params.get("original_prediction", "not_computed")
    adversarial_prediction = attack_params.get("adversarial_prediction", "not_computed")
    original_confidence = attack_params.get("original_confidence", "not_computed")
    adversarial_confidence = attack_params.get("adversarial_confidence", "not_computed")
    original_correct = original_prediction == label if original_prediction != "not_computed" else "not_computed"
    adversarial_correct = adversarial_prediction == label if adversarial_prediction != "not_computed" else "not_computed"

    model_dependency = "model_dependent" if is_model_dependent_attack(attack_name) else "model_agnostic"
    attack_success: bool | str = (
        adversarial_prediction != label
        if is_model_dependent_attack(attack_name)
        else "not_applicable"
    )

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
        "model_dependency": model_dependency,
        "checkpoint_path": repo_relative_string(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "attack_success": attack_success,
        "original_prediction": original_prediction,
        "adversarial_prediction": adversarial_prediction,
        "original_confidence": original_confidence,
        "adversarial_confidence": adversarial_confidence,
        "original_correct": original_correct,
        "adversarial_correct": adversarial_correct,
        "sha256_original": sha256_original,
        "md5_original": md5_original,
        "sha256_perturbed": sha256_perturbed,
        "md5_perturbed": md5_perturbed,
        "perturbation_norm_l0": attack_params.get("perturbation_norm_l0", "not_computed"),
        "perturbation_norm_l2": attack_params.get("perturbation_norm_l2", "not_computed"),
        "perturbation_norm_linf": attack_params.get("perturbation_norm_linf", "not_computed"),
        "perturbation_mean_abs": attack_params.get("perturbation_mean_abs", "not_computed"),
        "size_bytes": output_path.stat().st_size,
        "extension": output_path.suffix.lower(),
        "created_at": created_at,
    }


def generate_color_shift_one(row: pd.Series, args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))
    img = open_rgb_image(source_path)
    transformed_img, attack_params, target_model = apply_color_shift(img, args)
    output_path = build_output_path(row, "color_shift", target_model)
    save_jpeg(transformed_img, output_path, quality=args.jpeg_quality)
    attack_params.update(compute_perturbation_metrics(img, transformed_img))
    return build_manifest_row(row, "color_shift", target_model, output_path, attack_params, created_at)


def generate_fgsm_one(row: pd.Series, adapter: TargetModelAdapter, args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))
    img = open_rgb_image(source_path)
    label = norm(row["final_label"])
    transformed_img, attack_params = apply_fgsm(img, label, adapter, args)
    output_path = build_output_path(row, "fgsm", adapter.name)
    save_png(transformed_img, output_path)
    return build_manifest_row(
        row,
        "fgsm",
        adapter.name,
        output_path,
        attack_params,
        created_at,
        checkpoint_path=adapter.config.checkpoint_path,
    )

def generate_model_dependent_attack_one(
    row: pd.Series,
    attack_name: str,
    adapter: TargetModelAdapter,
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    """
    Generate one model-dependent adversarial artifact for a manifest row.
    """
    source_path = resolve_clean_image_path(safe_str(row["split_relative_path"]))
    img = open_rgb_image(source_path)
    label = norm(row["final_label"])
    image_id = safe_str(row["image_id"])

    if attack_name == "one_pixel":
        transformed_img, attack_params = apply_one_pixel(
            img=img,
            true_label=label,
            adapter=adapter,
            args=args,
            image_id=image_id,
        )
    else:
        raise NotImplementedError(f"Unsupported model-dependent attack: {attack_name}")

    output_path = build_output_path(row, attack_name, adapter.name)
    save_png(transformed_img, output_path)

    return build_manifest_row(
        row,
        attack_name,
        adapter.name,
        output_path,
        attack_params,
        created_at,
        checkpoint_path=adapter.config.checkpoint_path,
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


def build_summary(
    rows: list[dict[str, Any]],
    input_manifest: Path,
    input_image_count: int,
    selected_attacks: list[str],
    selected_target_models: list[str],
    checkpoint_root: Path,
    manifest_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    per_attack_counts = Counter(row["attack_name"] for row in rows)
    per_target_counts = Counter(row["target_model"] for row in rows)
    per_fold_counts: dict[str, Counter] = defaultdict(Counter)
    per_label_counts: dict[str, Counter] = defaultdict(Counter)
    per_attack_success: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_fold_counts[row["attack_name"]][row["fold"]] += 1
        per_label_counts[row["attack_name"]][row["final_label"]] += 1
        per_attack_success[row["attack_name"]][str(row["attack_success"])] += 1

    expected_total = 0
    for attack in selected_attacks:
        expected_total += input_image_count * (len(selected_target_models) if is_model_dependent_attack(attack) else 1)

    return {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "input_manifest": repo_relative_string(input_manifest),
        "output_root": repo_relative_string(ADVERSARIAL_DIR),
        "manifest_csv": repo_relative_string(manifest_path),
        "summary_json": repo_relative_string(summary_path),
        "selected_attacks": selected_attacks,
        "selected_target_models": selected_target_models,
        "checkpoint_root": repo_relative_string(checkpoint_root),
        "parameters": {
            "fgsm_epsilon": args.fgsm_epsilon,
            "fgsm_output_format": "PNG",
            "color_shift_output_format": "JPEG",
            "jpeg_quality": args.jpeg_quality,
            "input_size": args.input_size,
            "device": args.device,
            "limit": args.limit,
            "one_pixel_max_iterations": args.one_pixel_max_iterations,
            "one_pixel_population_size": args.one_pixel_population_size,
            "one_pixel_seed": args.one_pixel_seed,
        },
        "counts": {
            "input_images": input_image_count,
            "expected_generated_images": expected_total,
            "actual_generated_images": len(rows),
            "per_attack_counts": dict(sorted(per_attack_counts.items())),
            "per_target_counts": dict(sorted(per_target_counts.items())),
            "per_fold_counts": {k: dict(sorted(v.items())) for k, v in sorted(per_fold_counts.items())},
            "per_label_counts": {k: dict(sorted(v.items())) for k, v in sorted(per_label_counts.items())},
            "per_attack_success": {k: dict(sorted(v.items())) for k, v in sorted(per_attack_success.items())},
        },
        "checks": {
            "expected_total_generated": len(rows) == expected_total,
            "generated_image_id_unique": len({row["generated_image_id"] for row in rows}) == len(rows),
            "perturbed_sha256_unique": len({row["sha256_perturbed"] for row in rows}) == len(rows),
            "manifest_written": manifest_path.exists(),
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    selected_attacks = list(dict.fromkeys(args.attack))
    selected_target_models = validate_target_model_names(args.target_model)
    input_manifest = repo_relative_path(args.input_manifest)
    checkpoint_root = repo_relative_path(args.checkpoint_root)
    created_at = utc_now_iso()

    manifest_path, summary_path = build_run_manifest_paths(
        selected_attacks=selected_attacks,
        selected_target_models=selected_target_models,
    )

    validate_selected_attacks(selected_attacks)
    validate_attack_parameters(args)

    logging.info("Input manifest: %s", input_manifest)
    logging.info("Selected attacks: %s", ", ".join(selected_attacks))
    logging.info("Selected target models: %s", ", ".join(selected_target_models))
    logging.info("Checkpoint root: %s", checkpoint_root)
    logging.info("Output root: %s", ADVERSARIAL_DIR)

    df = load_manifest(input_manifest, args.limit)
    validate_source_files(df)
    prepare_output_dirs(selected_attacks, args.force)

    validate_manifest_outputs(manifest_path, summary_path, args.force)

    adapters: dict[tuple[str, str], TargetModelAdapter] = {}
    if any(is_model_dependent_attack(attack) for attack in selected_attacks):
        adapters = load_fold_aware_adapters(df, selected_target_models, checkpoint_root, args)

    rows: list[dict[str, Any]] = []
    expected_total = 0
    for attack_name in selected_attacks:
        expected_total += len(df) * (len(selected_target_models) if is_model_dependent_attack(attack_name) else 1)

    progress = 0
    for attack_name in selected_attacks:
        logging.info("Generating attack: %s", attack_name)
        if attack_name == "color_shift":
            for _, row in df.iterrows():
                rows.append(generate_color_shift_one(row, args, created_at))
                progress += 1
                if progress % 250 == 0 or progress == expected_total:
                    logging.info("Generated %d/%d images", progress, expected_total)
            continue

        if attack_name == "fgsm":
            for target_model in selected_target_models:
                for _, row in df.iterrows():
                    fold = safe_str(row["fold"])
                    rows.append(generate_fgsm_one(row, adapters[(target_model, fold)], args, created_at))
                    progress += 1
                    if progress % 250 == 0 or progress == expected_total:
                        logging.info("Generated %d/%d images", progress, expected_total)
            continue

        if attack_name == "one_pixel":
            for target_model in selected_target_models:
                for _, row in df.iterrows():
                    fold = safe_str(row["fold"])
                    rows.append(
                        generate_model_dependent_attack_one(
                            row=row,
                            attack_name=attack_name,
                            adapter=adapters[(target_model, fold)],
                            args=args,
                            created_at=created_at,
                        )
                    )
                    progress += 1
                    if progress % 250 == 0 or progress == expected_total:
                        logging.info("Generated %d/%d images", progress, expected_total)
            continue

        raise NotImplementedError(f"Attack unexpectedly reached generation stage: {attack_name}")

    write_csv(manifest_path, rows)

    write_summary(
        summary_path,
        build_summary(
            rows,
            input_manifest,
            len(df),
            selected_attacks,
            selected_target_models,
            checkpoint_root,
            manifest_path,
            summary_path,
            args,
            created_at,
        ),
    )
    logging.info("Manifest written: %s", manifest_path)
    logging.info("Summary written: %s", summary_path)
    logging.info("Fold-aware adversarial generation completed successfully.")


if __name__ == "__main__":
    main()
