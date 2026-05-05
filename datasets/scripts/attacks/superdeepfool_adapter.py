#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
superdeepfool_adapter.py

Paper-based reference-style SuperDeepFool adapter for the FAIR-Lab thesis
pipeline.

Methodological note
-------------------
This file does not vendor or copy the official SuperDeepFool repository.
It implements an internal, reproducible, paper-based version of the multiclass
SDF(infinity, 1) procedure described in the NeurIPS 2024 SuperDeepFool paper.

The implementation is intended for fold-aware, checkpoint-dependent robustness
evaluation in the thesis pipeline.

Expected conventions
--------------------
- Input images are PIL RGB images.
- The target adapter exposes the same interface used by FGSM:
  preprocess_image(...)
  predict(...)
  predict_proba(...)
- The adapter should expose a PyTorch model through one of:
  model, torch_model, network, net
- The attack operates in normalized model space through the adapter, but
  perturbation metrics are computed in pixel space [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SuperDeepFoolConfig:
    """Configuration for the internal SuperDeepFool reference-style attack."""

    max_outer_iterations: int = 20
    max_deepfool_iterations: int = 50
    projection_steps: int = 1
    candidate_classes: int | None = None
    clip_min: float = 0.0
    clip_max: float = 1.0
    numerical_eps: float = 1e-12
    pixel_change_threshold: float = 1.0 / 255.0


def apply_superdeepfool_reference(
    img: Image.Image,
    true_label: str,
    adapter: Any,
    args: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Apply a paper-based SuperDeepFool SDF(infinity, 1) attack.

    Returns:
        transformed_img:
            Lossless-PNG-ready adversarial PIL image.
        attack_params:
            JSON-serializable metadata dictionary for the FAIR-Lab manifest.
    """

    clean_img = img.convert("RGB")

    model_input = adapter.preprocess_image(clean_img)
    original_pixel_tensor = denormalize_tensor(model_input, adapter).clamp(0.0, 1.0)

    original_probabilities = adapter.predict_proba(model_input)
    original_prediction = adapter.predict(model_input)
    original_confidence = confidence_for_prediction(original_probabilities, original_prediction)
    original_true_probability = float(original_probabilities.get(true_label, 0.0))

    config = SuperDeepFoolConfig(
        max_outer_iterations=args.superdeepfool_max_outer_iterations,
        max_deepfool_iterations=args.superdeepfool_max_deepfool_iterations,
        projection_steps=args.superdeepfool_projection_steps,
        candidate_classes=args.superdeepfool_candidate_classes,
        clip_min=0.0,
        clip_max=1.0,
        numerical_eps=1e-12,
        pixel_change_threshold=1.0 / 255.0,
    )

    attack = SuperDeepFoolReferenceAttack(
        adapter=adapter,
        config=config,
    )

    adversarial_pixel_tensor, runtime_info = attack.generate(
        original_pixel_tensor=original_pixel_tensor,
        true_label=true_label,
    )

    adversarial_model_input = normalize_tensor(adversarial_pixel_tensor, adapter)

    adversarial_probabilities = adapter.predict_proba(adversarial_model_input)
    adversarial_prediction = adapter.predict(adversarial_model_input)
    adversarial_confidence = confidence_for_prediction(
        adversarial_probabilities,
        adversarial_prediction,
    )
    adversarial_true_probability = float(adversarial_probabilities.get(true_label, 0.0))

    metrics = compute_perturbation_metrics_from_tensors(
        original_pixel_tensor=original_pixel_tensor,
        adversarial_pixel_tensor=adversarial_pixel_tensor,
        threshold=config.pixel_change_threshold,
    )

    attack_success = adversarial_prediction != true_label

    attack_params: dict[str, Any] = {
        "attack_type": "superdeepfool_reference_style",
        "attack_variant": "SDF(infinity,1)",
        "reference_basis": "paper_based_internal_implementation",
        "target_model": adapter.name,
        "input_size": args.input_size,
        "output_format": "PNG",
        "metric_space": "pixel_[0,1]",
        "model_dependency": "white_box_checkpoint",
        "attack_family": "adversarial",
        "attack_name": "superdeepfool",
        "attack_success": attack_success,
        "max_outer_iterations": config.max_outer_iterations,
        "max_deepfool_iterations": config.max_deepfool_iterations,
        "projection_steps": config.projection_steps,
        "candidate_classes": config.candidate_classes,
        "numerical_eps": config.numerical_eps,
        "pixel_change_threshold": config.pixel_change_threshold,
        "outer_iterations_used": runtime_info["outer_iterations_used"],
        "deepfool_iterations_used": runtime_info["deepfool_iterations_used"],
        "projection_iterations_used": runtime_info["projection_iterations_used"],
        "converged": runtime_info["converged"],
        "convergence_status": runtime_info["convergence_status"],
        "projection_preserved_adversarial": runtime_info["projection_preserved_adversarial"],
        "original_prediction": original_prediction,
        "adversarial_prediction": adversarial_prediction,
        "original_confidence": original_confidence,
        "adversarial_confidence": adversarial_confidence,
        "original_true_label_probability": original_true_probability,
        "adversarial_true_label_probability": adversarial_true_probability,
        "adversarial_probabilities": adversarial_probabilities,
        **metrics,
    }

    return tensor_to_rgb_image(adversarial_pixel_tensor), attack_params


class SuperDeepFoolReferenceAttack:
    """
    Internal paper-based SuperDeepFool SDF(infinity, 1) implementation.

    The attack:
    1. Finds an adversarial boundary point through a DeepFool-style procedure.
    2. Applies a SuperDeepFool projection step using the gradient difference
       between the boundary class and the reference class.
    3. Returns a valid adversarial point, falling back to the DeepFool boundary
       point if the projection does not preserve adversariality.
    """

    def __init__(self, adapter: Any, config: SuperDeepFoolConfig) -> None:
        self.adapter = adapter
        self.config = config
        self.model = resolve_torch_model(adapter)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate(
        self,
        original_pixel_tensor: torch.Tensor,
        true_label: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate one SuperDeepFool adversarial tensor."""

        x0 = ensure_batch(original_pixel_tensor).detach().clone()
        x0 = x0.to(self.device).float().clamp(
            min=self.config.clip_min,
            max=self.config.clip_max,
        )

        label_names = resolve_label_names(self.adapter, x0)
        label_to_index = {label: index for index, label in enumerate(label_names)}

        if true_label not in label_to_index:
            raise ValueError(
                f"True label {true_label!r} is not present in model label names: {label_names}"
            )

        reference_index = int(label_to_index[true_label])

        current = x0.detach().clone()
        outer_iterations_used = 0
        deepfool_iterations_used = 0
        projection_iterations_used = 0
        projection_preserved_adversarial = False
        convergence_status = "max_outer_iterations_reached"

        for outer_index in range(self.config.max_outer_iterations):
            outer_iterations_used = outer_index + 1

            boundary_point, boundary_info = self._run_deepfool_until_boundary(
                start_pixel_tensor=current,
                reference_index=reference_index,
            )

            deepfool_iterations_used += boundary_info["deepfool_iterations_used"]

            if not boundary_info["boundary_reached"]:
                convergence_status = "deepfool_boundary_not_reached"
                current = boundary_point.detach()
                break

            boundary_index = self._predict_index_from_pixel(boundary_point)

            projected_point = boundary_point.detach().clone()
            for _ in range(self.config.projection_steps):
                projected_point = self._project_boundary_point(
                    original_pixel_tensor=x0,
                    boundary_pixel_tensor=projected_point,
                    reference_index=reference_index,
                    boundary_index=boundary_index,
                )
                projection_iterations_used += 1

            projected_index = self._predict_index_from_pixel(projected_point)

            if projected_index != reference_index:
                current = projected_point.detach()
                projection_preserved_adversarial = True
                convergence_status = "converged_projected"
                break

            current = boundary_point.detach()
            convergence_status = "converged_boundary_projection_not_preserved"
            break

        final_index = self._predict_index_from_pixel(current)
        converged = final_index != reference_index

        if not converged and convergence_status == "max_outer_iterations_reached":
            convergence_status = "max_outer_iterations_reached_not_adversarial"

        runtime_info = {
            "outer_iterations_used": int(outer_iterations_used),
            "deepfool_iterations_used": int(deepfool_iterations_used),
            "projection_iterations_used": int(projection_iterations_used),
            "converged": bool(converged),
            "convergence_status": str(convergence_status),
            "projection_preserved_adversarial": bool(projection_preserved_adversarial),
        }

        return current.detach().cpu(), runtime_info

    @property
    def device(self) -> torch.device:
        """Resolve the active torch device."""

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _run_deepfool_until_boundary(
        self,
        start_pixel_tensor: torch.Tensor,
        reference_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run DeepFool-style iterations until the reference class changes."""

        current = start_pixel_tensor.detach().clone().to(self.device)

        for step_index in range(self.config.max_deepfool_iterations):
            current_index = self._predict_index_from_pixel(current)

            if current_index != reference_index:
                return current.detach(), {
                    "deepfool_iterations_used": int(step_index),
                    "boundary_reached": True,
                }

            step = self._compute_deepfool_step(
                pixel_tensor=current,
                reference_index=reference_index,
            )

            current = torch.clamp(
                current.detach() + step.detach(),
                min=self.config.clip_min,
                max=self.config.clip_max,
            )

        final_index = self._predict_index_from_pixel(current)

        return current.detach(), {
            "deepfool_iterations_used": int(self.config.max_deepfool_iterations),
            "boundary_reached": bool(final_index != reference_index),
        }

    def _compute_deepfool_step(
        self,
        pixel_tensor: torch.Tensor,
        reference_index: int,
    ) -> torch.Tensor:
        """Compute one multiclass DeepFool linearized update step."""

        x = pixel_tensor.detach().clone().to(self.device).requires_grad_(True)
        logits = self._forward_logits_from_pixel(x)

        num_classes = int(logits.shape[1])
        candidate_indices = self._select_candidate_indices(
            logits=logits.detach(),
            reference_index=reference_index,
            num_classes=num_classes,
        )

        best_distance = None
        best_step = None

        for class_index in candidate_indices:
            score_difference = logits[0, class_index] - logits[0, reference_index]

            gradient = torch.autograd.grad(
                outputs=score_difference,
                inputs=x,
                retain_graph=True,
                create_graph=False,
            )[0]

            gradient_flat = gradient.reshape(-1)
            gradient_norm = torch.linalg.vector_norm(gradient_flat, ord=2)
            denominator = gradient_norm.pow(2) + self.config.numerical_eps

            distance = torch.abs(score_difference.detach()) / (
                gradient_norm.detach() + self.config.numerical_eps
            )

            step = (
                torch.abs(score_difference.detach())
                / denominator.detach()
            ) * gradient.detach()

            if best_distance is None or float(distance.item()) < best_distance:
                best_distance = float(distance.item())
                best_step = step

        if best_step is None:
            return torch.zeros_like(pixel_tensor)

        return best_step.detach()

    def _project_boundary_point(
        self,
        original_pixel_tensor: torch.Tensor,
        boundary_pixel_tensor: torch.Tensor,
        reference_index: int,
        boundary_index: int,
    ) -> torch.Tensor:
        """Apply the SuperDeepFool projection step."""

        if boundary_index == reference_index:
            return boundary_pixel_tensor.detach()

        x_boundary = boundary_pixel_tensor.detach().clone().to(self.device).requires_grad_(True)
        x_original = original_pixel_tensor.detach().clone().to(self.device)

        logits = self._forward_logits_from_pixel(x_boundary)
        score_difference = logits[0, boundary_index] - logits[0, reference_index]

        gradient = torch.autograd.grad(
            outputs=score_difference,
            inputs=x_boundary,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        delta = x_boundary.detach() - x_original
        numerator = torch.sum(delta * gradient)
        denominator = torch.sum(gradient * gradient) + self.config.numerical_eps

        projected_delta = (numerator / denominator) * gradient

        projected = torch.clamp(
            x_original + projected_delta,
            min=self.config.clip_min,
            max=self.config.clip_max,
        )

        return projected.detach()

    def _select_candidate_indices(
        self,
        logits: torch.Tensor,
        reference_index: int,
        num_classes: int,
    ) -> list[int]:
        """Select candidate classes for the multiclass DeepFool step."""

        candidate_indices = [
            class_index
            for class_index in range(num_classes)
            if class_index != reference_index
        ]

        if self.config.candidate_classes is None:
            return candidate_indices

        candidate_count = max(
            1,
            min(int(self.config.candidate_classes), num_classes - 1),
        )

        sorted_indices = torch.argsort(logits[0], descending=True).tolist()

        selected = [
            int(class_index)
            for class_index in sorted_indices
            if int(class_index) != reference_index
        ]

        return selected[:candidate_count]

    def _predict_index_from_pixel(self, pixel_tensor: torch.Tensor) -> int:
        """Predict the class index from a pixel-space tensor."""

        with torch.no_grad():
            logits = self._forward_logits_from_pixel(pixel_tensor)
            return int(logits.argmax(dim=1).item())

    def _forward_logits_from_pixel(self, pixel_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass from pixel-space tensor [0, 1] to logits."""

        pixel_tensor = ensure_batch(pixel_tensor).to(self.device).float()
        model_input = normalize_tensor(pixel_tensor, self.adapter).to(self.device)

        if hasattr(self.adapter, "forward_logits"):
            logits = self.adapter.forward_logits(model_input)
        elif hasattr(self.adapter, "predict_logits"):
            logits = self.adapter.predict_logits(model_input)
        elif hasattr(self.adapter, "logits"):
            logits = self.adapter.logits(model_input)
        else:
            logits = self.model(model_input)

        if isinstance(logits, tuple):
            logits = logits[0]

        if isinstance(logits, list):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor):
            raise TypeError("The target model must return torch.Tensor logits.")

        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape (N, C), got {tuple(logits.shape)}.")

        return logits


def resolve_torch_model(adapter: Any) -> Any:
    """Resolve the underlying PyTorch model from the target adapter."""

    for attribute_name in ("model", "torch_model", "network", "net"):
        if hasattr(adapter, attribute_name):
            model = getattr(adapter, attribute_name)
            if model is not None:
                return model

    if callable(adapter):
        return adapter

    raise TypeError(
        "The target adapter does not expose a PyTorch model. "
        "Expected one of: model, torch_model, network, net."
    )


def resolve_label_names(adapter: Any, pixel_tensor: torch.Tensor) -> list[str]:
    """
    Resolve model class names in logit-index order.

    The function first checks common adapter attributes. If they are not present,
    it falls back to the insertion order of predict_proba(...), which is the
    interface already used by the FAIR-Lab attack pipeline.
    """

    for attribute_name in ("class_names", "labels", "classes"):
        if hasattr(adapter, attribute_name):
            value = getattr(adapter, attribute_name)
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]

    for attribute_name in ("id_to_label", "idx_to_label", "index_to_label"):
        if hasattr(adapter, attribute_name):
            value = getattr(adapter, attribute_name)
            if isinstance(value, dict) and value:
                return [str(value[index]) for index in sorted(value)]
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]

    model_input = normalize_tensor(pixel_tensor, adapter)
    probabilities = adapter.predict_proba(model_input)

    if isinstance(probabilities, dict) and probabilities:
        return [str(label) for label in probabilities.keys()]

    raise RuntimeError("Unable to resolve class names from the target adapter.")


def adapter_normalization(adapter: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve normalization statistics from the target adapter."""

    if hasattr(adapter, "imagenet_mean") and hasattr(adapter, "imagenet_std"):
        return tuple(adapter.imagenet_mean), tuple(adapter.imagenet_std)

    if hasattr(adapter, "clip_mean") and hasattr(adapter, "clip_std"):
        return tuple(adapter.clip_mean), tuple(adapter.clip_std)

    if hasattr(adapter, "mean") and hasattr(adapter, "std"):
        return tuple(adapter.mean), tuple(adapter.std)

    raise RuntimeError(
        f"Adapter {getattr(adapter, 'name', '<unknown>')!r} does not expose normalization statistics."
    )


def normalization_tensors(pixel_tensor: torch.Tensor, adapter: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Build mean/std tensors compatible with the input tensor."""

    mean, std = adapter_normalization(adapter)

    mean_tensor = torch.tensor(
        mean,
        dtype=pixel_tensor.dtype,
        device=pixel_tensor.device,
    ).view(1, 3, 1, 1)

    std_tensor = torch.tensor(
        std,
        dtype=pixel_tensor.dtype,
        device=pixel_tensor.device,
    ).view(1, 3, 1, 1)

    return mean_tensor, std_tensor


def denormalize_tensor(model_input: torch.Tensor, adapter: Any) -> torch.Tensor:
    """Convert normalized model input to pixel-space tensor [0, 1]."""

    model_input = ensure_batch(model_input)
    mean_tensor, std_tensor = normalization_tensors(model_input, adapter)
    return model_input * std_tensor + mean_tensor


def normalize_tensor(pixel_tensor: torch.Tensor, adapter: Any) -> torch.Tensor:
    """Convert pixel-space tensor [0, 1] to normalized model input."""

    pixel_tensor = ensure_batch(pixel_tensor)
    mean_tensor, std_tensor = normalization_tensors(pixel_tensor, adapter)
    return (pixel_tensor - mean_tensor) / std_tensor


def ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure that a tensor has a batch dimension."""

    if tensor.ndim == 3:
        return tensor.unsqueeze(0)

    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor with shape (C,H,W) or (1,C,H,W), got {tuple(tensor.shape)}.")

    return tensor


def tensor_to_rgb_image(pixel_tensor: torch.Tensor) -> Image.Image:
    """Convert a pixel-space tensor [0, 1] to PIL RGB image."""

    tensor = ensure_batch(pixel_tensor).detach().clamp(0.0, 1.0)[0].cpu()
    array = tensor.permute(1, 2, 0).numpy()
    array_uint8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def compute_perturbation_metrics_from_tensors(
    original_pixel_tensor: torch.Tensor,
    adversarial_pixel_tensor: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Compute FAIR-Lab perturbation metrics in pixel space [0, 1]."""

    original = ensure_batch(original_pixel_tensor).detach().cpu().float()
    adversarial = ensure_batch(adversarial_pixel_tensor).detach().cpu().float()

    if original.shape != adversarial.shape:
        raise ValueError(
            f"Shape mismatch: original={tuple(original.shape)}, adversarial={tuple(adversarial.shape)}"
        )

    diff = adversarial - original
    abs_diff = torch.abs(diff)

    return {
        "perturbation_norm_l0": int((abs_diff > threshold).sum().item()),
        "perturbation_norm_l2": float(torch.linalg.vector_norm(diff.reshape(-1), ord=2).item()),
        "perturbation_norm_linf": float(abs_diff.max().item()),
        "perturbation_mean_abs": float(abs_diff.mean().item()),
    }


def confidence_for_prediction(probabilities: dict[str, float], prediction: str) -> float:
    """Return the confidence associated with a predicted label."""

    return float(probabilities.get(prediction, 0.0))