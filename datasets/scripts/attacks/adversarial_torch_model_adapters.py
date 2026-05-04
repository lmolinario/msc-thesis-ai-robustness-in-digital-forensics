#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adversarial_torch_model_adapters.py

Optional PyTorch-based target-model adapters for model-dependent adversarial
attacks in the FAIR-Lab thesis pipeline.

Purpose
-------
This module contains concrete adapters for binary image classifiers used as
white-box/proxy targets during adversarial attack generation.

The module is intentionally optional: importing the main adversarial generation
script must not require PyTorch. Heavy dependencies are imported only when an
adapter is instantiated and loaded.

Supported status
----------------
- resnet18: implemented for a binary torchvision ResNet18 checkpoint.
- efficientnet_b0: implemented for a binary torchvision EfficientNet-B0 checkpoint.
- clip: intentionally blocked until the CLIP strategy is fixed
  (zero-shot prompts vs. trained binary head vs. fine-tuned checkpoint).

Checkpoint convention
---------------------
The ResNet18 and EfficientNet-B0 adapters expect a checkpoint containing either:
- a raw PyTorch state_dict; or
- a dictionary with one of these keys:
  - state_dict
  - model_state_dict
  - model

The loaded architecture is binary and follows the official class mapping:
0 = non_weapon
1 = weapon
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from datasets.scripts.attacks.adversarial_model_interface import (
    TargetModelAdapter,
    TargetModelConfig,
    index_to_label,
    label_to_index,
    validate_target_model_name,
)


# =============================================================================
# Optional dependency handling
# =============================================================================

class MissingTorchDependencyError(RuntimeError):
    """Raised when a PyTorch adapter is requested without ML dependencies."""


def _import_torch_stack() -> tuple[Any, Any, Any]:
    """
    Import PyTorch, torchvision models, and torchvision transforms lazily.

    Returns
    -------
    tuple
        (torch, torchvision.models, torchvision.transforms)
    """
    try:
        import torch  # type: ignore[import-not-found]
        from torchvision import models, transforms  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingTorchDependencyError(
            "PyTorch/torchvision are required for model-dependent adversarial "
            "attacks. Install the optional ML dependencies before loading "
            "target-model adapters."
        ) from exc

    return torch, models, transforms


# =============================================================================
# Shared torchvision binary classifier adapter
# =============================================================================

class TorchVisionBinaryClassifierAdapter(TargetModelAdapter):
    """
    Binary-classification adapter for torchvision image classifiers.

    This adapter is suitable for trained ResNet18 and EfficientNet-B0 checkpoints
    using the official FAIR-Lab binary mapping:
    - class 0: non_weapon
    - class 1: weapon
    """

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(self, config: TargetModelConfig) -> None:
        super().__init__(config)
        self._torch: Any | None = None
        self._models: Any | None = None
        self._transforms: Any | None = None
        self._device: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            return self.config.device
        return str(self._device)

    def load_model(self) -> None:
        if self.config.checkpoint_path is None:
            raise FileNotFoundError(
                f"A checkpoint path is required for target model {self.config.name!r}."
            )

        checkpoint_path = Path(self.config.checkpoint_path).expanduser()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found for target model {self.config.name!r}: "
                f"{checkpoint_path}"
            )

        self._torch, self._models, self._transforms = _import_torch_stack()
        self._device = self._select_device(self.config.device)
        self._model = self._build_binary_model(self.config.name)

        checkpoint = self._torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)

        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._preprocess = self._transforms.Compose(
            [
                self._transforms.Resize((self.config.input_size, self.config.input_size)),
                self._transforms.ToTensor(),
                self._transforms.Normalize(
                    mean=self.imagenet_mean,
                    std=self.imagenet_std,
                ),
            ]
        )

    def preprocess_image(self, image: Image.Image) -> Any:
        self._ensure_loaded()
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)
        return tensor.to(self._device)

    def predict(self, model_input: Any) -> str:
        probabilities = self.predict_proba(model_input)
        best_label = max(probabilities, key=probabilities.get)
        return str(best_label)

    def predict_proba(self, model_input: Any) -> dict[str, float]:
        self._ensure_loaded()
        with self._torch.no_grad():
            logits = self._model(model_input.to(self._device))
            probabilities = self._torch.softmax(logits, dim=1)[0]

        return {
            index_to_label(index): float(probabilities[index].detach().cpu().item())
            for index in range(len(probabilities))
        }

    def compute_loss(self, model_input: Any, true_label: str) -> Any:
        self._ensure_loaded()
        target_index = label_to_index(true_label)
        target = self._torch.tensor([target_index], dtype=self._torch.long, device=self._device)
        logits = self._model(model_input.to(self._device))
        return self._torch.nn.functional.cross_entropy(logits, target)

    def compute_gradient(self, model_input: Any, true_label: str) -> Any:
        self._ensure_loaded()
        attack_input = model_input.detach().clone().to(self._device)
        attack_input.requires_grad_(True)

        self._model.zero_grad(set_to_none=True)
        loss = self.compute_loss(attack_input, true_label)
        loss.backward()

        if attack_input.grad is None:
            raise RuntimeError(
                f"Gradient computation failed for target model {self.config.name!r}."
            )

        return attack_input.grad.detach().clone()

    def _select_device(self, requested_device: str) -> Any:
        requested = str(requested_device).strip().lower()
        if requested == "auto":
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        return self._torch.device(requested)

    def _build_binary_model(self, model_name: str) -> Any:
        name = validate_target_model_name(model_name)

        if name == "resnet18":
            model = self._models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = self._torch.nn.Linear(in_features, 2)
            return model

        if name == "efficientnet_b0":
            model = self._models.efficientnet_b0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = self._torch.nn.Linear(in_features, 2)
            return model

        raise NotImplementedError(
            f"Torchvision binary adapter is not implemented for model {model_name!r}."
        )

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if all(hasattr(value, "shape") for value in checkpoint.values()):
                return checkpoint

        raise ValueError(
            "Unsupported checkpoint format. Expected a raw state_dict or a dict "
            "containing one of: state_dict, model_state_dict, model."
        )

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned[cleaned_key] = value
        return cleaned

    def _ensure_loaded(self) -> None:
        if self._model is None or self._torch is None or self._preprocess is None:
            raise RuntimeError(
                f"Target model {self.config.name!r} is not loaded. "
                "Call load_model() before inference or gradient computation."
            )


# =============================================================================
# CLIP adapter placeholder with explicit methodological block
# =============================================================================

class ClipAdapterNotReady(TargetModelAdapter):
    """
    Explicit block for CLIP until the thesis fixes the CLIP attack strategy.

    CLIP can be attacked in different defensible ways:
    - zero-shot image-text similarity with fixed prompts;
    - a trained binary head on frozen CLIP image embeddings;
    - a fine-tuned CLIP-based binary classifier.

    These options have different preprocessing, logits, losses, and gradients.
    The adapter must therefore remain disabled until the methodological choice is
    made and documented.
    """

    @property
    def device(self) -> str:
        return self.config.device

    def load_model(self) -> None:
        raise NotImplementedError(
            "CLIP target-model adapter is not enabled yet. Fix the CLIP strategy "
            "first: zero-shot prompts, trained binary head, or fine-tuned CLIP."
        )

    def preprocess_image(self, image: Any) -> Any:
        self.load_model()

    def predict(self, model_input: Any) -> str:
        self.load_model()

    def predict_proba(self, model_input: Any) -> dict[str, float]:
        self.load_model()

    def compute_loss(self, model_input: Any, true_label: str) -> Any:
        self.load_model()

    def compute_gradient(self, model_input: Any, true_label: str) -> Any:
        self.load_model()


# =============================================================================
# Adapter factory
# =============================================================================

def build_target_model_adapter(config: TargetModelConfig) -> TargetModelAdapter:
    """
    Build the concrete adapter for a target model.

    The returned adapter still requires an explicit load_model() call. This keeps
    construction cheap and makes error handling clearer in the attack generator.
    """
    model_name = validate_target_model_name(config.name)

    if model_name in {"resnet18", "efficientnet_b0"}:
        return TorchVisionBinaryClassifierAdapter(config)

    if model_name == "clip":
        return ClipAdapterNotReady(config)

    raise NotImplementedError(f"No adapter is available for target model {model_name!r}.")
