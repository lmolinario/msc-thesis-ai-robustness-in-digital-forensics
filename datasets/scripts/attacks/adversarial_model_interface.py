#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adversarial_model_interface.py

Stable target-model interface definitions for adversarial attack generation in
the FAIR-Lab thesis pipeline.

Purpose
-------
This module defines the common contract that model-dependent adversarial attacks
must use before they are enabled in the official generation script.

It intentionally does not load PyTorch, torchvision, CLIP, or any other heavy
machine-learning dependency. Concrete adapters are kept in a separate optional
module.

Design principles
-----------------
- Keep 14_generate_adversarial_attacks.py as the official entry point.
- Keep model-dependent attack logic separated from dataset traversal, manifest
  writing, hashing, and output-path construction.
- Make every future attack use the same label mapping, prediction interface,
  gradient interface, and device-selection semantics.
- Avoid placeholder implementations that silently generate fake adversarial
  samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


# =============================================================================
# Official adversarial attack and target-model registry
# =============================================================================

VALID_BINARY_LABELS: Final[tuple[str, str]] = ("non_weapon", "weapon")
LABEL_TO_INDEX: Final[dict[str, int]] = {"non_weapon": 0, "weapon": 1}
INDEX_TO_LABEL: Final[dict[int, str]] = {0: "non_weapon", 1: "weapon"}

MODEL_AGNOSTIC_TARGET: Final[str] = "model_agnostic"

SUPPORTED_TARGET_MODELS: Final[tuple[str, ...]] = (
    "resnet18",
    "efficientnet_b0",
    "clip",
)

MODEL_AGNOSTIC_ATTACK_NAMES: Final[tuple[str, ...]] = (
    "color_shift",
)

MODEL_DEPENDENT_ATTACK_NAMES: Final[tuple[str, ...]] = (
    "fgsm",
    "sigma_zero",
    "one_pixel",
    "superdeepfool",
)

PLANNED_ATTACK_NAMES: Final[tuple[str, ...]] = (
    *MODEL_DEPENDENT_ATTACK_NAMES,
    *MODEL_AGNOSTIC_ATTACK_NAMES,
)

IMPLEMENTED_ATTACK_NAMES: Final[tuple[str, ...]] = (
    "fgsm",
    "color_shift",
)


# =============================================================================
# Target-model configuration
# =============================================================================

@dataclass(frozen=True)
class TargetModelConfig:
    """
    Immutable configuration for a model targeted by adversarial attacks.

    Parameters
    ----------
    name:
        Canonical target-model name. Must belong to SUPPORTED_TARGET_MODELS.

    checkpoint_path:
        Optional path to a trained binary classifier checkpoint.

    device:
        Device selector. The concrete adapter should interpret "auto" as CUDA
        if available, otherwise CPU.

    input_size:
        Expected square input size for image preprocessing.

    labels:
        Ordered binary labels. The order must remain aligned with LABEL_TO_INDEX.
    """

    name: str
    checkpoint_path: Path | None = None
    device: str = "auto"
    input_size: int = 224
    labels: tuple[str, str] = VALID_BINARY_LABELS

    def __post_init__(self) -> None:
        validate_target_model_name(self.name)
        if self.labels != VALID_BINARY_LABELS:
            raise ValueError(
                "TargetModelConfig.labels must remain aligned with the official "
                f"binary label mapping: {VALID_BINARY_LABELS}"
            )
        if self.input_size <= 0:
            raise ValueError("TargetModelConfig.input_size must be greater than 0.")


# =============================================================================
# Abstract model contract
# =============================================================================

class TargetModelAdapter(ABC):
    """
    Abstract contract required by model-dependent adversarial attacks.

    Concrete implementations must expose a consistent binary-classification
    interface for ResNet18, EfficientNet-B0, and CLIP-based proxy models.

    The methods deliberately use `Any` to avoid importing framework-specific
    tensor types in this lightweight interface module. Concrete adapters can use
    PyTorch tensors internally.
    """

    def __init__(self, config: TargetModelConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the effective execution device used by the adapter."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and prepare the model for deterministic inference."""

    @abstractmethod
    def preprocess_image(self, image: Any) -> Any:
        """
        Convert a PIL image or image-like object into the model input tensor.

        The implementation must apply the exact normalization used during
        training/evaluation of the corresponding target model.
        """

    @abstractmethod
    def predict(self, model_input: Any) -> str:
        """Return the predicted label: 'weapon' or 'non_weapon'."""

    @abstractmethod
    def predict_proba(self, model_input: Any) -> dict[str, float]:
        """Return calibrated or softmax-like probabilities for both labels."""

    @abstractmethod
    def compute_loss(self, model_input: Any, true_label: str) -> Any:
        """
        Compute the attack loss for a given input and ground-truth label.

        FGSM and related white-box attacks must use this method rather than
        duplicating loss logic inside each attack implementation.
        """

    @abstractmethod
    def compute_gradient(self, model_input: Any, true_label: str) -> Any:
        """
        Compute the gradient of the attack loss with respect to the input.

        The returned object is framework-specific, but it must be aligned with
        the preprocessed input tensor so that the attack implementation can apply
        the perturbation consistently.
        """


# =============================================================================
# Validation helpers
# =============================================================================

def validate_binary_label(label: str) -> str:
    """Validate and normalize an official binary label."""
    normalized = str(label).strip().lower()
    if normalized not in LABEL_TO_INDEX:
        raise ValueError(
            f"Invalid binary label: {label!r}. Expected one of: {VALID_BINARY_LABELS}"
        )
    return normalized


def label_to_index(label: str) -> int:
    """Map an official binary label to its numeric class index."""
    return LABEL_TO_INDEX[validate_binary_label(label)]


def index_to_label(index: int) -> str:
    """Map a numeric class index back to the official binary label."""
    if index not in INDEX_TO_LABEL:
        raise ValueError(
            f"Invalid class index: {index!r}. Expected one of: {tuple(INDEX_TO_LABEL)}"
        )
    return INDEX_TO_LABEL[index]


def validate_target_model_name(model_name: str) -> str:
    """Validate and normalize an official target-model name."""
    normalized = str(model_name).strip().lower()
    if normalized not in SUPPORTED_TARGET_MODELS:
        raise ValueError(
            f"Invalid target model: {model_name!r}. Expected one of: "
            f"{SUPPORTED_TARGET_MODELS}"
        )
    return normalized


def validate_target_model_names(model_names: list[str] | tuple[str, ...]) -> list[str]:
    """
    Validate a sequence of target-model names while preserving order and removing
    duplicates.
    """
    normalized: list[str] = []
    seen: set[str] = set()

    for model_name in model_names:
        valid_name = validate_target_model_name(model_name)
        if valid_name not in seen:
            normalized.append(valid_name)
            seen.add(valid_name)

    if not normalized:
        raise ValueError("At least one target model must be selected.")

    return normalized


def is_model_agnostic_attack(attack_name: str) -> bool:
    """Return True if the attack is generated once without a target model."""
    return attack_name in MODEL_AGNOSTIC_ATTACK_NAMES


def is_model_dependent_attack(attack_name: str) -> bool:
    """Return True if the attack requires a concrete target-model adapter."""
    return attack_name in MODEL_DEPENDENT_ATTACK_NAMES


def expected_generation_count(
    input_image_count: int,
    selected_attacks: list[str] | tuple[str, ...],
    selected_target_models: list[str] | tuple[str, ...],
) -> int:
    """
    Compute the expected number of generated adversarial images.

    Model-agnostic attacks are generated once per input image. Model-dependent
    attacks are generated once per input image and target model.
    """
    if input_image_count < 0:
        raise ValueError("input_image_count must be greater than or equal to 0.")

    target_models = validate_target_model_names(selected_target_models)
    total = 0

    for attack_name in selected_attacks:
        if is_model_agnostic_attack(attack_name):
            total += input_image_count
        elif is_model_dependent_attack(attack_name):
            total += input_image_count * len(target_models)
        else:
            raise ValueError(f"Unknown adversarial attack: {attack_name!r}")

    return total


def load_model(config: TargetModelConfig) -> TargetModelAdapter:
    """
    Concrete target-model adapters are built by
    datasets.scripts.attacks.adversarial_torch_model_adapters.build_target_model_adapter.
    """
    raise NotImplementedError(
        "Use adversarial_torch_model_adapters.build_target_model_adapter(config) "
        "to construct validated target-model adapters."
    )
