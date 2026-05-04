#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_proxy_models.py

Train fold-aware proxy models for the FAIR-Lab adversarial attack pipeline.

Purpose
-------
This script trains transparent binary proxy models for the official
weapon/non_weapon task. The generated checkpoints are later used to create
model-dependent adversarial perturbations such as FGSM, Sigma Zero,
SuperDeepFool, and One Pixel attacks.

Training protocol
-----------------
For each target fold, the proxy model is trained on all other folds and saved as:

    models/checkpoints/<model_name>/<fold>.pt

This avoids training a proxy model on the same fold that will later be attacked.

Supported models
----------------
- resnet18
- efficientnet_b0
- clip

CLIP is implemented as a frozen visual encoder plus a trained binary head.

Execution modes
---------------
- CLI mode: pass explicit arguments for fully reproducible runs.
- Interactive mode: run the script without arguments, e.g. from PyCharm, and use
  the guided menu. The generated configuration is printed before training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repository root importable when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from datasets.scripts.utils.paths import SPLIT_MANIFESTS_DIR, repo_relative_path


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_NAME = "models/scripts/train_proxy_models.py"
INPUT_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
MODEL_REGISTRY_PATH = REPO_ROOT / "models" / "model_registry.json"
CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"
TRAINING_REPORTS_DIR = REPO_ROOT / "models" / "reports"

VALID_LABELS = ("non_weapon", "weapon")
LABEL_TO_INDEX = {"non_weapon": 0, "weapon": 1}
SUPPORTED_MODELS = ("resnet18", "efficientnet_b0", "clip")
SUPPORTED_FOLD_SELECTION = ("all", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5")


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    fold: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    seed: int
    device: str
    input_size: int
    num_workers: int
    freeze_backbone: bool


# =============================================================================
# Argument parsing and interactive launcher
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fold-aware binary proxy models for FAIR-Lab."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(INPUT_MANIFEST_PATH),
        help=f"Clean folds manifest (default: {INPUT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=SUPPORTED_MODELS,
        required=True,
        help="Proxy model(s) to train.",
    )
    parser.add_argument(
        "--fold",
        nargs="+",
        choices=SUPPORTED_FOLD_SELECTION,
        default=["fold_1"],
        help="Target fold(s) to train for. Use 'all' for all folds.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Internal stratified validation ratio from the training folds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Square input size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze CNN backbone and train only the classifier head.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Invalid answer. Please enter y or n.")


def ask_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{marker}")

    while True:
        answer = input("Selection: ").strip()
        if not answer:
            return options[default_index]
        if answer.isdigit():
            selected = int(answer)
            if 1 <= selected <= len(options):
                return options[selected - 1]
        print(f"Invalid selection. Please enter a number between 1 and {len(options)}.")


def ask_multi_choice(prompt: str, options: list[str], default_all: bool = False) -> list[str]:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")

    default_text = "all" if default_all else ""
    print("\nExamples: 1 | 1 2 | 1,2,3 | all")

    while True:
        answer = input(f"Selection [{default_text}]: ").strip().lower()
        if not answer and default_all:
            return options.copy()
        if answer == "all":
            return options.copy()

        tokens = answer.replace(",", " ").split()
        selected: list[str] = []
        valid = bool(tokens)

        for token in tokens:
            if not token.isdigit():
                valid = False
                break
            index = int(token)
            if not (1 <= index <= len(options)):
                valid = False
                break
            value = options[index - 1]
            if value not in selected:
                selected.append(value)

        if valid and selected:
            return selected

        print(f"Invalid selection. Use numbers between 1 and {len(options)} or 'all'.")


def ask_int(prompt: str, default_value: int) -> int:
    while True:
        answer = input(f"{prompt} [{default_value}]: ").strip()
        if not answer:
            return default_value
        try:
            return int(answer)
        except ValueError:
            print("Invalid value. Please enter an integer.")


def ask_float(prompt: str, default_value: float) -> float:
    while True:
        answer = input(f"{prompt} [{default_value}]: ").strip()
        if not answer:
            return default_value
        try:
            return float(answer)
        except ValueError:
            print("Invalid value. Please enter a number.")


def interactive_args() -> argparse.Namespace:
    print_header("FAIR-Lab proxy model training launcher")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Default manifest: {INPUT_MANIFEST_PATH}")

    scenario = ask_choice(
        prompt="What training scenario do you want to run?",
        options=[
            "Smoke test: resnet18 on fold_1 for 2 epochs",
            "Train ResNet18 on all folds",
            "Train EfficientNet-B0 on all folds",
            "Train CLIP binary head on all folds",
            "Custom selection",
        ],
        default_index=0,
    )

    if scenario.startswith("Smoke test"):
        model = ["resnet18"]
        fold = ["fold_1"]
        epochs = 2
        batch_size = 16
        learning_rate = 1e-4
        freeze_backbone = False
    elif scenario.startswith("Train ResNet18"):
        model = ["resnet18"]
        fold = ["all"]
        epochs = 10
        batch_size = 16
        learning_rate = 1e-4
        freeze_backbone = False
    elif scenario.startswith("Train EfficientNet"):
        model = ["efficientnet_b0"]
        fold = ["all"]
        epochs = 10
        batch_size = 16
        learning_rate = 1e-4
        freeze_backbone = False
    elif scenario.startswith("Train CLIP"):
        model = ["clip"]
        fold = ["all"]
        epochs = 10
        batch_size = 32
        learning_rate = 1e-4
        freeze_backbone = True
    else:
        model = ask_multi_choice(
            prompt="Select proxy model(s) to train:",
            options=list(SUPPORTED_MODELS),
            default_all=False,
        )
        fold = ask_multi_choice(
            prompt="Select target fold(s):",
            options=list(SUPPORTED_FOLD_SELECTION),
            default_all=False,
        )
        epochs = ask_int("Training epochs", 10)
        batch_size = ask_int("Batch size", 16)
        learning_rate = ask_float("Learning rate", 1e-4)
        freeze_backbone = ask_yes_no("Freeze backbone and train only classifier head?", default=False)

    print_header("Training parameters")
    epochs = ask_int("Training epochs", epochs)
    batch_size = ask_int("Batch size", batch_size)
    learning_rate = ask_float("Learning rate", learning_rate)
    weight_decay = ask_float("Weight decay", 1e-4)
    validation_ratio = ask_float("Validation ratio", 0.15)
    seed = ask_int("Random seed", 42)
    input_size = ask_int("Input size", 224)
    num_workers = ask_int("DataLoader workers", 2)
    device = ask_choice("Training device:", ["auto", "cpu", "cuda"], default_index=0)
    force = ask_yes_no("Overwrite existing checkpoints if present?", default=True)
    verbose = ask_yes_no("Enable verbose logging?", default=False)

    command_preview = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        *model,
        "--fold",
        *fold,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--validation-ratio",
        str(validation_ratio),
        "--seed",
        str(seed),
        "--device",
        device,
        "--input-size",
        str(input_size),
        "--num-workers",
        str(num_workers),
    ]
    if freeze_backbone:
        command_preview.append("--freeze-backbone")
    if force:
        command_preview.append("--force")
    if verbose:
        command_preview.append("--verbose")

    print_header("Equivalent reproducible command")
    print(" ".join(shlex.quote(part) for part in command_preview))

    if not ask_yes_no("Start training now?", default=True):
        raise SystemExit("Execution cancelled by user.")

    return argparse.Namespace(
        manifest=str(INPUT_MANIFEST_PATH),
        model=model,
        fold=fold,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_ratio=validation_ratio,
        seed=seed,
        device=device,
        input_size=input_size,
        num_workers=num_workers,
        freeze_backbone=freeze_backbone,
        force=force,
        verbose=verbose,
    )


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


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


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def repo_relative_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def compute_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def select_device(requested_device: str) -> Any:
    import torch

    requested = requested_device.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def expand_folds(selected: list[str], available_folds: list[str]) -> list[str]:
    if "all" in selected:
        return available_folds
    requested = list(dict.fromkeys(selected))
    missing = [fold for fold in requested if fold not in available_folds]
    if missing:
        raise ValueError(f"Requested folds not found in manifest: {missing}")
    return requested


# =============================================================================
# Manifest and dataset handling
# =============================================================================

def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    required = {"image_id", "fold", "final_label", "split_relative_path", "sha256"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in manifest: {sorted(missing)}")

    df = df.copy()
    df["final_label"] = df["final_label"].map(norm)
    invalid_labels = sorted(set(df["final_label"].unique()) - set(VALID_LABELS))
    if invalid_labels:
        raise ValueError(f"Invalid labels in manifest: {invalid_labels}")

    return df


def split_train_validation(
    train_df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < validation_ratio < 0.5):
        raise ValueError("--validation-ratio must be in the interval (0, 0.5).")

    validation_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []

    for label, group in train_df.groupby("final_label", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        validation_count = max(1, int(round(len(group) * validation_ratio)))
        validation_parts.append(group.iloc[:validation_count])
        train_parts.append(group.iloc[validation_count:])

    train_split = pd.concat(train_parts, ignore_index=True)
    validation_split = pd.concat(validation_parts, ignore_index=True)

    train_split = train_split.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_split = validation_split.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_split, validation_split


class ManifestImageDataset:
    """PyTorch-compatible dataset backed by the official fold manifest."""

    def __init__(self, df: pd.DataFrame, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.df.iloc[index]
        image_path = resolve_repo_path(safe_str(row["split_relative_path"]))
        label = LABEL_TO_INDEX[norm(row["final_label"])]

        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                tensor = self.transform(img)
        except UnidentifiedImageError as exc:
            raise ValueError(f"Cannot identify image file: {image_path}") from exc

        return tensor, label


# =============================================================================
# Model builders
# =============================================================================

def build_transforms(model_name: str, input_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    if model_name == "clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_torchvision_binary_model(model_name: str, freeze_backbone: bool) -> Any:
    import torch
    from torchvision import models

    if model_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
        return model

    if model_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported torchvision model: {model_name}")


class ClipBinaryClassifier:
    """Frozen CLIP visual encoder plus trainable binary classification head."""

    def __init__(self, clip_model: Any, feature_dim: int) -> None:
        import torch

        self.torch = torch
        self.clip_model = clip_model
        self.binary_head = torch.nn.Linear(feature_dim, 2)

    def parameters(self) -> Any:
        return self.binary_head.parameters()

    def train(self) -> None:
        self.clip_model.eval()
        self.binary_head.train()

    def eval(self) -> None:
        self.clip_model.eval()
        self.binary_head.eval()

    def to(self, device: Any) -> "ClipBinaryClassifier":
        self.clip_model.to(device)
        self.binary_head.to(device)
        return self

    def __call__(self, x: Any) -> Any:
        features = self.clip_model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.binary_head(features)

    def state_dict(self) -> dict[str, Any]:
        return self.binary_head.state_dict()


def build_clip_binary_model(input_size: int) -> ClipBinaryClassifier:
    import open_clip
    import torch

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    output_dim = getattr(getattr(clip_model, "visual", None), "output_dim", None)
    if not isinstance(output_dim, int) or output_dim <= 0:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = clip_model.encode_image(dummy)
            output_dim = int(features.shape[-1])

    return ClipBinaryClassifier(clip_model=clip_model, feature_dim=output_dim)


def build_model(model_name: str, input_size: int, freeze_backbone: bool) -> Any:
    if model_name in {"resnet18", "efficientnet_b0"}:
        return build_torchvision_binary_model(model_name, freeze_backbone=freeze_backbone)
    if model_name == "clip":
        return build_clip_binary_model(input_size=input_size)
    raise ValueError(f"Unsupported model: {model_name}")


# =============================================================================
# Training and evaluation
# =============================================================================

def build_loaders(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model_name: str,
    config: TrainConfig,
) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader

    train_transform, eval_transform = build_transforms(model_name, config.input_size)
    train_dataset = ManifestImageDataset(train_df, transform=train_transform)
    validation_dataset = ManifestImageDataset(validation_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, validation_loader


def train_one_epoch(model: Any, loader: Any, optimizer: Any, criterion: Any, device: Any) -> dict[str, float]:
    import torch

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * len(labels)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu().item())
        total_samples += int(len(labels))

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def evaluate(model: Any, loader: Any, criterion: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += float(loss.detach().cpu().item()) * len(labels)
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu().item())
            total_samples += int(len(labels))

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def checkpoint_path(model_name: str, fold: str) -> Path:
    return CHECKPOINT_ROOT / model_name / f"{fold}.pt"


def save_checkpoint(
    path: Path,
    model_name: str,
    fold: str,
    model: Any,
    config: TrainConfig,
    metrics_history: list[dict[str, Any]],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)

    if model_name == "clip":
        payload = {
            "model_name": model_name,
            "fold": fold,
            "clip_model_name": "ViT-B-32",
            "clip_pretrained": "openai",
            "binary_head_state_dict": model.state_dict(),
            "label_mapping": LABEL_TO_INDEX,
            "input_size": config.input_size,
            "created_at": utc_now_iso(),
            "training_config": config.__dict__,
            "metrics_history": metrics_history,
        }
    else:
        payload = {
            "model_name": model_name,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "label_mapping": LABEL_TO_INDEX,
            "input_size": config.input_size,
            "created_at": utc_now_iso(),
            "training_config": config.__dict__,
            "metrics_history": metrics_history,
        }

    torch.save(payload, path)


def train_for_fold(
    df: pd.DataFrame,
    model_name: str,
    fold: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    output_path = checkpoint_path(model_name, fold)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Checkpoint already exists. Use --force to overwrite: {output_path}")

    train_source_df = df[df["fold"] != fold].copy()
    holdout_df = df[df["fold"] == fold].copy()
    train_df, validation_df = split_train_validation(
        train_source_df,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    config = TrainConfig(
        model_name=model_name,
        fold=fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        device=args.device,
        input_size=args.input_size,
        num_workers=args.num_workers,
        freeze_backbone=args.freeze_backbone,
    )

    device = select_device(args.device)
    logging.info("Training %s for %s on %s", model_name, fold, device)
    logging.info("Train images: %d | Validation images: %d | Holdout images: %d", len(train_df), len(validation_df), len(holdout_df))

    model = build_model(model_name, input_size=args.input_size, freeze_backbone=args.freeze_backbone).to(device)
    train_loader, validation_loader = build_loaders(train_df, validation_df, model_name, config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_validation_accuracy = -1.0
    best_payload_path = output_path

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_record)

        logging.info(
            "%s %s epoch %d/%d | train_acc=%.4f val_acc=%.4f train_loss=%.4f val_loss=%.4f",
            model_name,
            fold,
            epoch,
            args.epochs,
            train_metrics["accuracy"],
            validation_metrics["accuracy"],
            train_metrics["loss"],
            validation_metrics["loss"],
        )

        if validation_metrics["accuracy"] >= best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            save_checkpoint(best_payload_path, model_name, fold, model, config, history)

    sha256 = compute_sha256(output_path)
    return {
        "model_name": model_name,
        "fold": fold,
        "checkpoint_path": repo_relative_string(output_path),
        "checkpoint_sha256": sha256,
        "train_images": len(train_df),
        "validation_images": len(validation_df),
        "holdout_images": len(holdout_df),
        "best_validation_accuracy": best_validation_accuracy,
        "last_epoch": history[-1] if history else {},
        "created_at": utc_now_iso(),
    }


# =============================================================================
# Reporting
# =============================================================================

def write_training_report(records: list[dict[str, Any]]) -> Path:
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TRAINING_REPORTS_DIR / "proxy_model_training_summary.csv"

    fieldnames = [
        "model_name",
        "fold",
        "checkpoint_path",
        "checkpoint_sha256",
        "train_images",
        "validation_images",
        "holdout_images",
        "best_validation_accuracy",
        "last_epoch",
        "created_at",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["last_epoch"] = json.dumps(row.get("last_epoch", {}), sort_keys=True)
            writer.writerow(row)

    return report_path


def update_registry(records: list[dict[str, Any]]) -> None:
    if MODEL_REGISTRY_PATH.exists():
        registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema_version": "1.0",
            "task": "binary_weapon_classification",
            "label_mapping": LABEL_TO_INDEX,
            "fold_protocol": "leave_one_fold_out_proxy_training",
            "input_manifest": repo_relative_string(INPUT_MANIFEST_PATH),
            "checkpoint_root": repo_relative_string(CHECKPOINT_ROOT),
            "models": {},
        }

    for record in records:
        model_name = record["model_name"]
        fold = record["fold"]
        model_entry = registry.setdefault("models", {}).setdefault(model_name, {})
        sha_map = model_entry.setdefault("sha256", {})
        sha_map[fold] = record["checkpoint_sha256"]
        model_entry["last_trained_at"] = record["created_at"]

    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = interactive_args() if len(sys.argv) == 1 else parse_args()
    setup_logging(args.verbose)
    set_reproducibility(args.seed)

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.input_size <= 0:
        raise ValueError("--input-size must be greater than 0.")

    manifest_path = repo_relative_path(args.manifest)
    df = load_manifest(manifest_path)
    available_folds = sorted(df["fold"].unique())
    selected_folds = expand_folds(args.fold, available_folds)

    logging.info("Manifest: %s", manifest_path)
    logging.info("Models: %s", ", ".join(args.model))
    logging.info("Folds: %s", ", ".join(selected_folds))

    records: list[dict[str, Any]] = []
    for model_name in args.model:
        for fold in selected_folds:
            record = train_for_fold(df=df, model_name=model_name, fold=fold, args=args)
            records.append(record)

    report_path = write_training_report(records)
    update_registry(records)

    logging.info("Training report written: %s", report_path)
    logging.info("Model registry updated: %s", MODEL_REGISTRY_PATH)
    logging.info("Proxy model training completed successfully.")


if __name__ == "__main__":
    main()
