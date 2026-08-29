#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adversarial_torch_model_adapters.py

Optional PyTorch-based target-model adapters for model-dependent adversarial
attacks in the FAIRLab thesis pipeline.

Purpose
-------
This module contains concrete adapters for binary image classifiers used as
white-box/proxy targets during adversarial attack generation.

The module is intentionally optional: importing the main adversarial generation
script must not require PyTorch or CLIP dependencies. Heavy dependencies are
imported only when an adapter is instantiated and loaded.

Supported status
----------------
- resnet18: implemented for a binary torchvision ResNet18 checkpoint.
- efficientnet_b0: implemented for a binary torchvision EfficientNet-B0 checkpoint.
- clip: implemented as a CLIP-based binary classifier, i.e., a frozen CLIP visual
  encoder followed by a trained binary classification head.

Checkpoint convention
---------------------
The ResNet18 and EfficientNet-B0 adapters expect a checkpoint containing either:
- a raw PyTorch state_dict; or
- a dictionary with one of these keys:
  - state_dict
  - model_state_dict
  - model

The CLIP-based adapter expects a checkpoint containing either:
- a raw binary-head state_dict; or
- a dictionary with one of these keys:
  - binary_head_state_dict
  - head_state_dict
  - classifier_state_dict
  - state_dict

The loaded architectures follow the official class mapping:
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


class MissingClipDependencyError(RuntimeError):
    """Raised when the CLIP-based adapter is requested without open_clip."""


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


def _import_open_clip() -> Any:
    """Import open_clip lazily only when the CLIP-based adapter is used."""
    try:
        import open_clip  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingClipDependencyError(
            "open_clip_torch is required for the CLIP-based binary classifier. "
            "Install requirements-ml.txt before loading the CLIP adapter."
        ) from exc

    return open_clip


# =============================================================================
# Shared helpers
# =============================================================================

def _select_torch_device(torch_module: Any, requested_device: str) -> Any:
    requested = str(requested_device).strip().lower()
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned[cleaned_key] = value
    return cleaned


def _looks_like_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, dict) and all(
        hasattr(value, "shape") for value in candidate.values()
    )


# =============================================================================
# Shared torchvision binary classifier adapter
# =============================================================================

class TorchVisionBinaryClassifierAdapter(TargetModelAdapter):
    """
    Binary-classification adapter for torchvision image classifiers.

    This adapter is suitable for trained ResNet18 and EfficientNet-B0 checkpoints
    using the official FAIRLab binary mapping:
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
        self._device = _select_torch_device(self._torch, self.config.device)
        self._model = self._build_binary_model(self.config.name)

        checkpoint = self._torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = _strip_module_prefix(state_dict)

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
            if _looks_like_state_dict(checkpoint):
                return checkpoint

        raise ValueError(
            "Unsupported checkpoint format. Expected a raw state_dict or a dict "
            "containing one of: state_dict, model_state_dict, model."
        )

    def _ensure_loaded(self) -> None:
        if self._model is None or self._torch is None or self._preprocess is None:
            raise RuntimeError(
                f"Target model {self.config.name!r} is not loaded. "
                "Call load_model() before inference or gradient computation."
            )


# =============================================================================
# CLIP-based binary classifier adapter
# =============================================================================

class ClipBinaryHeadAdapter(TargetModelAdapter):
    """
    CLIP-based binary classifier adapter.

    Methodological choice
    ---------------------
    This adapter implements the thesis choice of using CLIP as a frozen visual
    representation model and a lightweight trained binary head for the
    weapon/non_weapon task. It is not a zero-shot prompt-based CLIP classifier and
    it is not a fully fine-tuned CLIP model.

    Default backbone
    ----------------
    The default backbone is open_clip ViT-B-32 with OpenAI weights. A checkpoint
    may optionally specify `clip_model_name` and `clip_pretrained`, but all
    generated adversarial samples must document these values in the generation
    summary for reproducibility.
    """

    default_clip_model_name = "ViT-B-32"
    default_clip_pretrained = "openai"
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, config: TargetModelConfig) -> None:
        super().__init__(config)
        self._torch: Any | None = None
        self._transforms: Any | None = None
        self._open_clip: Any | None = None
        self._device: Any | None = None
        self._clip_model: Any | None = None
        self._binary_head: Any | None = None
        self._preprocess: Any | None = None
        self.clip_model_name: str = self.default_clip_model_name
        self.clip_pretrained: str = self.default_clip_pretrained

    @property
    def device(self) -> str:
        if self._device is None:
            return self.config.device
        return str(self._device)

    def load_model(self) -> None:
        if self.config.checkpoint_path is None:
            raise FileNotFoundError(
                "A binary-head checkpoint path is required for the CLIP-based "
                "target model."
            )

        checkpoint_path = Path(self.config.checkpoint_path).expanduser()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CLIP binary-head checkpoint not found: {checkpoint_path}"
            )

        self._torch, _, self._transforms = _import_torch_stack()
        self._open_clip = _import_open_clip()
        self._device = _select_torch_device(self._torch, self.config.device)

        checkpoint = self._torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        self.clip_model_name = self._checkpoint_value(
            checkpoint,
            "clip_model_name",
            self.default_clip_model_name,
        )
        self.clip_pretrained = self._checkpoint_value(
            checkpoint,
            "clip_pretrained",
            self.default_clip_pretrained,
        )

        self._clip_model, _, _ = self._open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
        )
        self._clip_model.to(self._device)
        self._clip_model.eval()

        for parameter in self._clip_model.parameters():
            parameter.requires_grad_(False)

        feature_dim = self._infer_feature_dim()
        self._binary_head = self._torch.nn.Linear(feature_dim, 2)

        head_state_dict = self._extract_binary_head_state_dict(checkpoint)
        head_state_dict = _strip_module_prefix(head_state_dict)
        self._binary_head.load_state_dict(head_state_dict, strict=True)
        self._binary_head.to(self._device)
        self._binary_head.eval()

        self._preprocess = self._transforms.Compose(
            [
                self._transforms.Resize((self.config.input_size, self.config.input_size)),
                self._transforms.ToTensor(),
                self._transforms.Normalize(
                    mean=self.clip_mean,
                    std=self.clip_std,
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
            logits = self._forward_logits(model_input)
            probabilities = self._torch.softmax(logits, dim=1)[0]

        return {
            index_to_label(index): float(probabilities[index].detach().cpu().item())
            for index in range(len(probabilities))
        }

    def compute_loss(self, model_input: Any, true_label: str) -> Any:
        self._ensure_loaded()
        target_index = label_to_index(true_label)
        target = self._torch.tensor([target_index], dtype=self._torch.long, device=self._device)
        logits = self._forward_logits(model_input)
        return self._torch.nn.functional.cross_entropy(logits, target)

    def compute_gradient(self, model_input: Any, true_label: str) -> Any:
        self._ensure_loaded()
        attack_input = model_input.detach().clone().to(self._device)
        attack_input.requires_grad_(True)

        self._clip_model.zero_grad(set_to_none=True)
        self._binary_head.zero_grad(set_to_none=True)
        loss = self.compute_loss(attack_input, true_label)
        loss.backward()

        if attack_input.grad is None:
            raise RuntimeError("Gradient computation failed for CLIP-based target model.")

        return attack_input.grad.detach().clone()

    def _forward_logits(self, model_input: Any) -> Any:
        model_input = model_input.to(self._device)
        features = self._clip_model.encode_image(model_input)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self._binary_head(features)

    def _infer_feature_dim(self) -> int:
        output_dim = getattr(getattr(self._clip_model, "visual", None), "output_dim", None)
        if isinstance(output_dim, int) and output_dim > 0:
            return output_dim

        with self._torch.no_grad():
            dummy = self._torch.zeros(
                1,
                3,
                self.config.input_size,
                self.config.input_size,
                device=self._device,
            )
            features = self._clip_model.encode_image(dummy)
        return int(features.shape[-1])

    @staticmethod
    def _checkpoint_value(checkpoint: Any, key: str, default: str) -> str:
        if isinstance(checkpoint, dict):
            value = checkpoint.get(key, default)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    @staticmethod
    def _extract_binary_head_state_dict(checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in (
                "binary_head_state_dict",
                "head_state_dict",
                "classifier_state_dict",
                "state_dict",
            ):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if _looks_like_state_dict(checkpoint):
                return checkpoint

        raise ValueError(
            "Unsupported CLIP binary-head checkpoint format. Expected a raw "
            "state_dict or a dict containing one of: binary_head_state_dict, "
            "head_state_dict, classifier_state_dict, state_dict."
        )

    def _ensure_loaded(self) -> None:
        if (
            self._clip_model is None
            or self._binary_head is None
            or self._torch is None
            or self._preprocess is None
        ):
            raise RuntimeError(
                "CLIP-based target model is not loaded. Call load_model() before "
                "inference or gradient computation."
            )


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
        return ClipBinaryHeadAdapter(config)

    raise NotImplementedError(f"No adapter is available for target model {model_name!r}.")
