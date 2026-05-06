#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sigma_zero_reference_adapter.py

Reference-integration adapter for Sigma-Zero adversarial generation.

This module intentionally does not reimplement Sigma-Zero. It calls the
reference implementation exposed by `adv_lib.attacks.sigma_zero` and adapts it
to the FAIR-Lab fold-aware proxy-model interface.

Expected dependency
-------------------
Install adversarial-library from the reference repository, preferably pinned to
the inspected commit used during the thesis work:

    python -m pip install git+https://github.com/jeromerony/adversarial-library.git@b14f81a3e1c414a573b969b402c99e65bfe2ca33

The reference function expects:
- a PyTorch model producing logits;
- inputs in pixel space [0, 1];
- integer labels;
- untargeted mode.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any


import numpy as np
from PIL import Image

from datasets.scripts.attacks.adversarial_model_interface import (
    TargetModelAdapter,
    index_to_label,
    label_to_index,
)


def _import_torch() -> Any:
    """Import PyTorch lazily."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Sigma-Zero reference integration."
        ) from exc
    return torch


def _import_adv_lib_sigma_zero() -> Any:
    """
    Import the reference Sigma-Zero implementation without importing
    adv_lib.attacks.__init__.

    The adversarial-library package imports optional visualization utilities
    such as visdom from adv_lib.attacks.__init__. Importing the package-level
    registry would therefore require dependencies that are not needed by the
    FAIR-Lab pipeline. This loader imports only adv_lib/attacks/sigma_zero.py
    directly from the installed package location.
    """
    import importlib.util

    try:
        import adv_lib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "adv_lib is required for the reference Sigma-Zero attack. Install it with:\n"
            "python -m pip install --no-deps "
            "git+https://github.com/jeromerony/adversarial-library.git@b14f81a3e1c414a573b969b402c99e65bfe2ca33"
        ) from exc

    adv_lib_root = Path(adv_lib.__file__).resolve().parent
    sigma_zero_path = adv_lib_root / "attacks" / "sigma_zero.py"

    if not sigma_zero_path.exists():
        raise FileNotFoundError(
            f"Cannot locate adversarial-library Sigma-Zero source file: {sigma_zero_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "fair_lab_adv_lib_sigma_zero_reference",
        sigma_zero_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Sigma-Zero module from: {sigma_zero_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "sigma_zero"):
        raise AttributeError(
            f"The loaded module does not expose a sigma_zero function: {sigma_zero_path}"
        )

    return module.sigma_zero


def _adapter_device(adapter: TargetModelAdapter, torch_module: Any) -> Any:
    """Return the concrete torch.device used by an already-loaded adapter."""
    candidate = getattr(adapter, "_device", None)
    if candidate is not None:
        return candidate
    return torch_module.device(adapter.device)


def _adapter_normalization(adapter: TargetModelAdapter) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the normalization statistics required by the target adapter."""
    if hasattr(adapter, "imagenet_mean") and hasattr(adapter, "imagenet_std"):
        return tuple(adapter.imagenet_mean), tuple(adapter.imagenet_std)
    if hasattr(adapter, "clip_mean") and hasattr(adapter, "clip_std"):
        return tuple(adapter.clip_mean), tuple(adapter.clip_std)
    raise RuntimeError(
        f"Adapter {adapter.name!r} does not expose normalization statistics."
    )


def _normalization_tensors(
    pixel_tensor: Any,
    adapter: TargetModelAdapter,
) -> tuple[Any, Any]:
    """Build mean/std tensors compatible with a pixel-space tensor."""
    torch_module = _import_torch()
    mean, std = _adapter_normalization(adapter)
    mean_tensor = torch_module.tensor(
        mean,
        dtype=pixel_tensor.dtype,
        device=pixel_tensor.device,
    ).view(1, 3, 1, 1)
    std_tensor = torch_module.tensor(
        std,
        dtype=pixel_tensor.dtype,
        device=pixel_tensor.device,
    ).view(1, 3, 1, 1)
    return mean_tensor, std_tensor


def normalize_pixel_tensor(pixel_tensor: Any, adapter: TargetModelAdapter) -> Any:
    """Normalize a pixel-space tensor in [0, 1] for the target adapter."""
    mean_tensor, std_tensor = _normalization_tensors(pixel_tensor, adapter)
    return (pixel_tensor - mean_tensor) / std_tensor


def denormalize_model_input(model_input: Any, adapter: TargetModelAdapter) -> Any:
    """Convert a normalized model input tensor back to pixel space [0, 1]."""
    mean_tensor, std_tensor = _normalization_tensors(model_input, adapter)
    return (model_input * std_tensor + mean_tensor).clamp(0.0, 1.0)


def tensor_to_rgb_image(pixel_tensor: Any) -> Image.Image:
    """Convert a pixel-space BCHW tensor in [0, 1] into a PIL RGB image."""
    tensor = pixel_tensor.detach().clamp(0.0, 1.0)[0].cpu()
    array = tensor.permute(1, 2, 0).numpy()
    array_uint8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def tensor_to_pixel_array(pixel_tensor: Any) -> np.ndarray:
    """Convert a pixel-space BCHW tensor in [0, 1] into an HWC float array."""
    tensor = pixel_tensor.detach().clamp(0.0, 1.0)[0].cpu()
    return tensor.permute(1, 2, 0).numpy().astype(np.float32)


def compute_perturbation_metrics_from_arrays(
    original: np.ndarray,
    transformed: np.ndarray,
) -> dict[str, float | int]:
    """Compute basic perturbation norms in the supplied metric space."""
    if original.shape != transformed.shape:
        raise ValueError(
            f"Shape mismatch: original={original.shape}, transformed={transformed.shape}"
        )
    diff = transformed.astype(np.float32) - original.astype(np.float32)
    abs_diff = np.abs(diff)
    return {
        "perturbation_norm_l0": int(np.count_nonzero(abs_diff)),
        "perturbation_norm_l2": float(np.linalg.norm(diff.ravel(), ord=2)),
        "perturbation_norm_linf": float(np.max(abs_diff)),
        "perturbation_mean_abs": float(np.mean(abs_diff)),
    }


def _probabilities_from_logits(logits: Any) -> dict[str, float]:
    """Convert binary logits into the official FAIR-Lab label-probability dict."""
    torch_module = _import_torch()
    probabilities = torch_module.softmax(logits, dim=1)[0]
    return {
        index_to_label(index): float(probabilities[index].detach().cpu().item())
        for index in range(len(probabilities))
    }


def _prediction_from_probabilities(probabilities: dict[str, float]) -> str:
    """Return the highest-probability official label."""
    return str(max(probabilities, key=probabilities.get))


def _confidence_for_prediction(probabilities: dict[str, float], prediction: str) -> float:
    """Return the confidence associated with a predicted label."""
    return float(probabilities.get(prediction, 0.0))


class PixelSpaceTargetModel:
    """
    Torch-compatible wrapper exposing a pixel-space [0, 1] logits interface.

    The reference Sigma-Zero implementation expects a model that receives
    unnormalized image tensors in [0, 1]. FAIR-Lab target-model adapters normally
    operate on normalized tensors, so this wrapper normalizes the input before
    dispatching to the underlying trained model/head.
    """

    def __init__(self, adapter: TargetModelAdapter) -> None:
        self.adapter = adapter
        self.torch = _import_torch()
        self.device = _adapter_device(adapter, self.torch)

    def __call__(self, pixel_inputs: Any) -> Any:
        return self.forward(pixel_inputs)

    def forward(self, pixel_inputs: Any) -> Any:
        normalized = normalize_pixel_tensor(
            pixel_inputs.to(self.device).clamp(0.0, 1.0),
            self.adapter,
        )

        if hasattr(self.adapter, "_model") and getattr(self.adapter, "_model") is not None:
            return getattr(self.adapter, "_model")(normalized)

        if hasattr(self.adapter, "_forward_logits"):
            return getattr(self.adapter, "_forward_logits")(normalized)

        raise RuntimeError(
            f"Adapter {self.adapter.name!r} cannot expose logits for Sigma-Zero."
        )


def apply_sigma_zero_reference(
    img: Image.Image,
    true_label: str,
    adapter: TargetModelAdapter,
    args: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Apply the reference Sigma-Zero attack through the FAIR-Lab adapter layer.

    The attack is untargeted and model-dependent. The generated tensor is kept in
    pixel space [0, 1] and later saved as PNG by the caller.
    """
    torch_module = _import_torch()
    reference_sigma_zero = _import_adv_lib_sigma_zero()

    model_input = adapter.preprocess_image(img.convert("RGB"))
    device = _adapter_device(adapter, torch_module)

    original_pixel_tensor = denormalize_model_input(model_input, adapter).detach().to(device)
    labels = torch_module.tensor(
        [label_to_index(true_label)],
        dtype=torch_module.long,
        device=device,
    )

    pixel_space_model = PixelSpaceTargetModel(adapter)

    with torch_module.no_grad():
        original_logits = pixel_space_model(original_pixel_tensor)
        original_probabilities = _probabilities_from_logits(original_logits)
        original_prediction = _prediction_from_probabilities(original_probabilities)
        original_confidence = _confidence_for_prediction(
            original_probabilities,
            original_prediction,
        )
        original_true_probability = float(original_probabilities.get(true_label, 0.0))

    adv_pixel_tensor = reference_sigma_zero(
        model=pixel_space_model,
        inputs=original_pixel_tensor.detach(),
        labels=labels,
        num_steps=args.sigma_zero_steps,
        **{
            "η_0": args.sigma_zero_eta,
            "σ": args.sigma_zero_sigma,
            "τ_0": args.sigma_zero_tau,
        },
        τ_factor=args.sigma_zero_tau_factor,
        grad_norm=args.sigma_zero_grad_norm,
        targeted=False,
    ).detach().clamp(0.0, 1.0)

    with torch_module.no_grad():
        adversarial_logits = pixel_space_model(adv_pixel_tensor)
        adversarial_probabilities = _probabilities_from_logits(adversarial_logits)
        adversarial_prediction = _prediction_from_probabilities(adversarial_probabilities)
        adversarial_confidence = _confidence_for_prediction(
            adversarial_probabilities,
            adversarial_prediction,
        )
        adversarial_true_probability = float(
            adversarial_probabilities.get(true_label, 0.0)
        )

    original_array = tensor_to_pixel_array(original_pixel_tensor)
    adversarial_array = tensor_to_pixel_array(adv_pixel_tensor)
    metrics = compute_perturbation_metrics_from_arrays(
        original_array,
        adversarial_array,
    )

    changed_pixel_count = int(
        np.count_nonzero(
            np.any(np.abs(adversarial_array - original_array) > 0.0, axis=2)
        )
    )
    converged = adversarial_prediction != true_label

    params: dict[str, Any] = {
        "attack_type": "sigma_zero_reference_adv_lib",
        "reference_library": "jeromerony/adversarial-library",
        "reference_function": "adv_lib.attacks.sigma_zero",
        "reference_commit": "b14f81a3e1c414a573b969b402c99e65bfe2ca33",
        "target_model": adapter.name,
        "input_size": args.input_size,
        "output_format": "PNG",
        "metric_space": "pixel_[0,1]",
        "optimization_objective": "minimal_l0_untargeted_reference_sigma_zero",
        "steps": args.sigma_zero_steps,
        "eta_0": args.sigma_zero_eta,
        "sigma": args.sigma_zero_sigma,
        "tau_0": args.sigma_zero_tau,
        "tau_factor": args.sigma_zero_tau_factor,
        "grad_norm": args.sigma_zero_grad_norm,
        "targeted": False,
        "converged": converged,
        "changed_pixel_count": changed_pixel_count,
        "original_prediction": original_prediction,
        "adversarial_prediction": adversarial_prediction,
        "original_confidence": original_confidence,
        "adversarial_confidence": adversarial_confidence,
        "original_true_label_probability": original_true_probability,
        "adversarial_true_label_probability": adversarial_true_probability,
        "adversarial_probabilities": adversarial_probabilities,
        **metrics,
    }

    return tensor_to_rgb_image(adv_pixel_tensor), params
