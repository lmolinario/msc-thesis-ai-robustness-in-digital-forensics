#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_train_proxy_models.py

Train the fold-aware proxy models used by the FAIRLab thesis pipeline.

For each held-out fold, the corresponding checkpoint is trained on the other
four folds and stored under:

    models/checkpoints/<model_name>/<fold>.pt

The script preserves the checkpoint payload format consumed by the adversarial
and proxy-evaluation adapters. It validates the frozen split manifest and the
SHA256 of every locally restored image before training.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from datasets.scripts.utils.paths import SPLIT_MANIFESTS_DIR, repo_relative_path

SCRIPT_NAME = "models/scripts/12_train_proxy_models.py"
INPUT_MANIFEST_PATH = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
MODEL_REGISTRY_PATH = REPO_ROOT / "models" / "model_registry.json"
CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"
TRAINING_REPORTS_DIR = REPO_ROOT / "models" / "reports"

VALID_LABELS = ("non_weapon", "weapon")
LABEL_TO_INDEX = {"non_weapon": 0, "weapon": 1}
SUPPORTED_MODELS = ("resnet18", "efficientnet_b0", "clip")
EXPECTED_FOLDS = tuple(f"fold_{index}" for index in range(1, 6))
SUPPORTED_FOLD_SELECTION = ("all", *EXPECTED_FOLDS)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fold-aware binary proxy models for FAIRLab."
    )
    parser.add_argument(
        "--manifest",
        default=str(INPUT_MANIFEST_PATH),
        help=f"Clean-fold manifest (default: {INPUT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=SUPPORTED_MODELS,
        required=True,
        help="Proxy model or models to train.",
    )
    parser.add_argument(
        "--fold",
        nargs="+",
        choices=SUPPORTED_FOLD_SELECTION,
        default=["fold_1"],
        help="Held-out fold or folds. Use 'all' for the complete suite.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Internal stratified validation ratio from the four training folds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze the ResNet18/EfficientNet-B0 backbone and train only the "
            "classifier head. The CLIP visual encoder is always frozen."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint files.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


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
        print("Enter y or n.")


def ask_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{marker}")
    while True:
        answer = input("Selection: ").strip()
        if not answer:
            return options[default_index]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"Enter a number between 1 and {len(options)}.")


def ask_int(prompt: str, default: int) -> int:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return int(answer)
        except ValueError:
            print("Enter an integer.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return float(answer)
        except ValueError:
            print("Enter a number.")


def interactive_args() -> argparse.Namespace:
    scenarios = [
        "Smoke test: ResNet18 / fold_1 / 2 epochs",
        "ResNet18: all folds",
        "EfficientNet-B0: all folds",
        "CLIP binary head: all folds",
        "All proxy models: all folds",
    ]
    scenario = ask_choice("Training scenario", scenarios)

    if scenario.startswith("Smoke"):
        models = ["resnet18"]
        folds = ["fold_1"]
        epochs = 2
        batch_size = 16
    elif scenario.startswith("ResNet18"):
        models = ["resnet18"]
        folds = ["all"]
        epochs = 10
        batch_size = 16
    elif scenario.startswith("EfficientNet"):
        models = ["efficientnet_b0"]
        folds = ["all"]
        epochs = 10
        batch_size = 16
    elif scenario.startswith("CLIP"):
        models = ["clip"]
        folds = ["all"]
        epochs = 10
        batch_size = 32
    else:
        models = list(SUPPORTED_MODELS)
        folds = ["all"]
        epochs = 10
        batch_size = 16

    epochs = ask_int("Epochs", epochs)
    batch_size = ask_int("Batch size", batch_size)
    learning_rate = ask_float("Learning rate", 1e-4)
    weight_decay = ask_float("Weight decay", 1e-4)
    validation_ratio = ask_float("Validation ratio", 0.15)
    seed = ask_int("Random seed", 42)
    input_size = ask_int("Input size", 224)
    num_workers = ask_int("DataLoader workers", 2)
    device = ask_choice("Device", ["auto", "cpu", "cuda"])
    freeze_backbone = ask_yes_no(
        "Freeze CNN backbone? (CLIP is always frozen)", default=False
    )
    force = ask_yes_no("Overwrite existing checkpoints?", default=True)
    verbose = ask_yes_no("Verbose logging?", default=False)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        *models,
        "--fold",
        *folds,
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
        command.append("--freeze-backbone")
    if force:
        command.append("--force")
    if verbose:
        command.append("--verbose")

    print("\nEquivalent command:")
    print(" ".join(shlex.quote(part) for part in command))
    if not ask_yes_no("Start training?", default=True):
        raise SystemExit("Execution cancelled.")

    return argparse.Namespace(
        manifest=str(INPUT_MANIFEST_PATH),
        model=models,
        fold=folds,
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
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def expand_folds(selected: list[str], available: list[str]) -> list[str]:
    if "all" in selected:
        return available
    requested = list(dict.fromkeys(selected))
    missing = [fold for fold in requested if fold not in available]
    if missing:
        raise ValueError(f"Requested folds not found in manifest: {missing}")
    return requested


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    required = {
        "image_id",
        "fold",
        "final_label",
        "split_relative_path",
        "sha256",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    df = df.copy()
    df["image_id"] = df["image_id"].map(safe_str)
    df["fold"] = df["fold"].map(safe_str)
    df["final_label"] = df["final_label"].map(norm)
    df["split_relative_path"] = df["split_relative_path"].map(safe_str)
    df["sha256"] = df["sha256"].map(lambda value: safe_str(value).lower())

    invalid_labels = sorted(set(df["final_label"]) - set(VALID_LABELS))
    if invalid_labels:
        raise ValueError(f"Invalid labels in manifest: {invalid_labels}")
    return df


def validate_manifest_files(df: pd.DataFrame) -> None:
    duplicate_ids = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate image_id values: {duplicate_ids[:10]}")

    actual_folds = sorted(df["fold"].unique().tolist())
    if actual_folds != list(EXPECTED_FOLDS):
        raise ValueError(
            f"Expected folds {list(EXPECTED_FOLDS)}, found {actual_folds}."
        )

    grouped = df.groupby(["fold", "final_label"]).size()
    unexpected_counts: list[str] = []
    for fold in EXPECTED_FOLDS:
        for label in VALID_LABELS:
            count = int(grouped.get((fold, label), 0))
            if count != 100:
                unexpected_counts.append(f"{fold}/{label}={count}")
    if unexpected_counts:
        raise ValueError(
            "The official split requires 100 samples per class and fold. "
            f"Unexpected counts: {unexpected_counts}"
        )

    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    for row in df.itertuples(index=False):
        image_path = resolve_repo_path(row.split_relative_path)
        if not image_path.is_file():
            missing_files.append(repo_relative_string(image_path))
            continue
        actual_sha256 = compute_sha256(image_path).lower()
        if actual_sha256 != row.sha256:
            hash_mismatches.append(
                f"{row.image_id}: expected {row.sha256}, found {actual_sha256}"
            )

    if missing_files:
        raise FileNotFoundError(
            "Split images are missing. Restore the controlled-access data and "
            f"regenerate step 11. First missing paths: {missing_files[:10]}"
        )
    if hash_mismatches:
        raise ValueError(
            f"Split-image SHA256 verification failed: {hash_mismatches[:10]}"
        )


def split_train_validation(
    train_df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be in (0, 0.5).")

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for _, group in train_df.groupby("final_label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
        validation_parts.append(shuffled.iloc[:validation_count])
        train_parts.append(shuffled.iloc[validation_count:])

    train_split = pd.concat(train_parts, ignore_index=True)
    validation_split = pd.concat(validation_parts, ignore_index=True)
    return (
        train_split.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        validation_split.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


class ManifestImageDataset:
    def __init__(self, df: pd.DataFrame, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.df.iloc[index]
        image_path = resolve_repo_path(row["split_relative_path"])
        label = LABEL_TO_INDEX[norm(row["final_label"])]
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                tensor = self.transform(image)
        except UnidentifiedImageError as exc:
            raise ValueError(f"Cannot identify image: {image_path}") from exc
        return tensor, label


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
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, evaluation_transform


def build_torchvision_binary_model(
    model_name: str,
    freeze_backbone: bool,
) -> Any:
    import torch
    from torchvision import models

    if model_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError as exc:
            raise RuntimeError(
                "torchvision does not expose ResNet18_Weights.IMAGENET1K_V1. "
                "Refusing to use random initialization because it would change "
                "the frozen protocol."
            ) from exc
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        except AttributeError as exc:
            raise RuntimeError(
                "torchvision does not expose EfficientNet_B0_Weights.IMAGENET1K_V1. "
                "Refusing to use random initialization because it would change "
                "the frozen protocol."
            ) from exc
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, 2)
        return model

    raise ValueError(f"Unsupported torchvision model: {model_name}")


class ClipBinaryClassifier:
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

    def __call__(self, tensor: Any) -> Any:
        with self.torch.no_grad():
            features = self.clip_model.encode_image(tensor)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return self.binary_head(features)

    def state_dict(self) -> dict[str, Any]:
        return self.binary_head.state_dict()


def build_clip_binary_model(input_size: int) -> ClipBinaryClassifier:
    import open_clip
    import torch

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    output_dim = getattr(getattr(clip_model, "visual", None), "output_dim", None)
    if not isinstance(output_dim, int) or output_dim <= 0:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            output_dim = int(clip_model.encode_image(dummy).shape[-1])
    return ClipBinaryClassifier(clip_model, output_dim)


def build_model(
    model_name: str,
    input_size: int,
    freeze_backbone: bool,
) -> Any:
    if model_name in {"resnet18", "efficientnet_b0"}:
        return build_torchvision_binary_model(model_name, freeze_backbone)
    if model_name == "clip":
        return build_clip_binary_model(input_size)
    raise ValueError(f"Unsupported model: {model_name}")


def build_loaders(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model_name: str,
    config: TrainConfig,
) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader

    train_transform, evaluation_transform = build_transforms(
        model_name,
        config.input_size,
    )
    train_dataset = ManifestImageDataset(train_df, train_transform)
    validation_dataset = ManifestImageDataset(
        validation_df,
        evaluation_transform,
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device != "cpu",
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
    )


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
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

        count = int(len(labels))
        total_loss += float(loss.detach().cpu().item()) * count
        total_correct += int(
            (torch.argmax(logits, dim=1) == labels).sum().detach().cpu().item()
        )
        total_samples += count

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def evaluate(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
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

            count = int(len(labels))
            total_loss += float(loss.detach().cpu().item()) * count
            total_correct += int(
                (torch.argmax(logits, dim=1) == labels)
                .sum()
                .detach()
                .cpu()
                .item()
            )
            total_samples += count

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
    common = {
        "model_name": model_name,
        "fold": fold,
        "label_mapping": LABEL_TO_INDEX,
        "input_size": config.input_size,
        "created_at": utc_now_iso(),
        "training_config": config.__dict__,
        "metrics_history": metrics_history,
    }
    if model_name == "clip":
        payload = {
            **common,
            "clip_model_name": "ViT-B-32",
            "clip_pretrained": "openai",
            "binary_head_state_dict": model.state_dict(),
        }
    else:
        payload = {
            **common,
            "model_state_dict": model.state_dict(),
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
        raise FileExistsError(
            f"Checkpoint exists; use --force to overwrite: {output_path}"
        )

    train_source = df[df["fold"] != fold].copy()
    holdout = df[df["fold"] == fold].copy()
    train_df, validation_df = split_train_validation(
        train_source,
        args.validation_ratio,
        args.seed,
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
        freeze_backbone=True if model_name == "clip" else args.freeze_backbone,
    )

    device = select_device(args.device)
    logging.info(
        "Training %s for %s on %s | train=%d validation=%d holdout=%d",
        model_name,
        fold,
        device,
        len(train_df),
        len(validation_df),
        len(holdout),
    )

    model = build_model(
        model_name,
        input_size=args.input_size,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    train_loader, validation_loader = build_loaders(
        train_df,
        validation_df,
        model_name,
        config,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_validation_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(record)
        logging.info(
            "%s %s epoch %d/%d | train_acc=%.4f val_acc=%.4f "
            "train_loss=%.4f val_loss=%.4f",
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
            save_checkpoint(
                output_path,
                model_name,
                fold,
                model,
                config,
                history,
            )

    return {
        "model_name": model_name,
        "fold": fold,
        "checkpoint_path": repo_relative_string(output_path),
        "checkpoint_sha256": compute_sha256(output_path),
        "train_images": len(train_df),
        "validation_images": len(validation_df),
        "holdout_images": len(holdout),
        "best_validation_accuracy": best_validation_accuracy,
        "last_epoch": history[-1] if history else {},
        "created_at": utc_now_iso(),
    }


REPORT_FIELDS = [
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


def write_training_report(records: list[dict[str, Any]]) -> Path:
    """Upsert model/fold records while retaining untouched frozen records."""
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TRAINING_REPORTS_DIR / "proxy_model_training_summary.csv"

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if report_path.exists():
        with report_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != REPORT_FIELDS:
                raise ValueError(
                    f"Unexpected training-report schema: {reader.fieldnames}"
                )
            for row in reader:
                merged[(row["model_name"], row["fold"])] = dict(row)

    for record in records:
        row = dict(record)
        row["last_epoch"] = json.dumps(
            row.get("last_epoch", {}),
            sort_keys=True,
            allow_nan=False,
        )
        merged[(row["model_name"], row["fold"])] = row

    model_order = {name: index for index, name in enumerate(SUPPORTED_MODELS)}
    fold_order = {fold: index for index, fold in enumerate(EXPECTED_FOLDS)}
    ordered = sorted(
        merged.values(),
        key=lambda row: (
            model_order.get(row["model_name"], len(model_order)),
            fold_order.get(row["fold"], len(fold_order)),
        ),
    )

    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
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
            "training_script": SCRIPT_NAME,
            "training_report": (
                "models/reports/proxy_model_training_summary.csv"
            ),
            "checkpoint_distribution": "git_lfs",
            "models": {},
        }

    for record in records:
        model_name = record["model_name"]
        fold = record["fold"]
        model_entry = registry.setdefault("models", {}).setdefault(
            model_name,
            {},
        )
        model_entry.setdefault("sha256", {})[fold] = record[
            "checkpoint_sha256"
        ]
        model_entry["last_trained_at"] = record["created_at"]

    MODEL_REGISTRY_PATH.write_text(
        json.dumps(
            registry,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.input_size <= 0:
        raise ValueError("--input-size must be greater than zero.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be in (0, 0.5).")


def main() -> None:
    args = interactive_args() if len(sys.argv) == 1 else build_parser().parse_args()
    setup_logging(args.verbose)
    validate_arguments(args)
    set_reproducibility(args.seed)

    manifest_path = repo_relative_path(args.manifest)
    df = load_manifest(manifest_path)
    logging.info("Validating split files and SHA256 hashes")
    validate_manifest_files(df)

    available_folds = sorted(df["fold"].unique().tolist())
    selected_folds = expand_folds(args.fold, available_folds)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Models: %s", ", ".join(args.model))
    logging.info("Held-out folds: %s", ", ".join(selected_folds))

    records: list[dict[str, Any]] = []
    for model_name in args.model:
        for fold in selected_folds:
            records.append(
                train_for_fold(
                    df=df,
                    model_name=model_name,
                    fold=fold,
                    args=args,
                )
            )

    report_path = write_training_report(records)
    update_registry(records)
    logging.info("Training report: %s", report_path)
    logging.info("Model registry: %s", MODEL_REGISTRY_PATH)
    logging.info("Proxy-model training completed.")


if __name__ == "__main__":
    main()
