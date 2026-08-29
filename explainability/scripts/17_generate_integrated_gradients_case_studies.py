#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity-hardened entry point for FAIRLab Integrated Gradients case studies.

The frozen implementation is preserved in the adjacent implementation module.
This wrapper adds headless-safe execution, checkpoint and input integrity checks,
adaptive convergence diagnostics, pseudonymous reviewer defaults, and atomic
cleanup of the run-specific output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
IMPL_PATH = SCRIPT_PATH.with_name("_17_generate_integrated_gradients_case_studies_impl.py")
DEFAULT_REGISTRY = REPO_ROOT / "models" / "model_registry.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_impl(backend: str) -> types.ModuleType:
    if not IMPL_PATH.exists():
        raise FileNotFoundError(f"Missing implementation module: {IMPL_PATH}")
    source = IMPL_PATH.read_text(encoding="utf-8")
    source = source.replace(
        'matplotlib.use("TkAgg")',
        f'matplotlib.use({backend!r}, force=True)',
        1,
    )
    module = types.ModuleType("fairlab_xai_impl")
    module.__file__ = str(IMPL_PATH)
    module.__package__ = ""
    exec(compile(source, str(IMPL_PATH), "exec"), module.__dict__)
    return module


def wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-n-steps", type=int, default=256)
    parser.add_argument("--convergence-threshold", type=float, default=0.05)
    parser.add_argument("--model-registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--disable-adaptive-convergence", action="store_true")
    return parser.parse_known_args(argv)


def expected_hash_for_row(module: types.ModuleType, row: Any) -> str:
    attack_family = module.norm(row.get("attack_family", ""))
    sample_type = module.norm(row.get("sample_type", ""))
    preferred: list[str] = []
    if attack_family not in {"", "none"} or sample_type in {
        "perturbed", "adversarial", "anti_forensic", "anti-forensic", "transformed"
    }:
        preferred.extend(["sha256_perturbed", "sha256_generated"])
    preferred.extend(["sha256", "sha256_original"])
    for column in preferred:
        value = module.safe_str(row.get(column, "")).lower()
        if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return value
    return ""


def install_hardening(module: types.ModuleType, config: argparse.Namespace) -> None:
    registry_path = Path(config.model_registry)
    if not registry_path.is_absolute():
        registry_path = (REPO_ROOT / registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    original_get = module.AdapterCache.get

    def verified_get(self: Any, model_name: str, fold: str) -> Any:
        checkpoint_path = self.checkpoint_root / model_name / f"{fold}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        expected = registry.get("models", {}).get(model_name, {}).get("sha256", {}).get(fold, "")
        if not expected:
            raise RuntimeError(f"No registry SHA256 for {model_name}/{fold}")
        actual = sha256_file(checkpoint_path)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch for {model_name}/{fold}: expected {expected}, got {actual}"
            )
        if not hasattr(self, "checkpoint_hashes"):
            self.checkpoint_hashes = {}
        self.checkpoint_hashes[(model_name, fold)] = actual
        return original_get(self, model_name, fold)

    module.AdapterCache.get = verified_get

    original_guard = module.ensure_run_outputs_do_not_exist

    def atomic_guard(run_paths: dict[str, Path], force: bool, check_manual_db: bool = False) -> None:
        if not force:
            original_guard(run_paths, force, check_manual_db)
            return
        run_output_dir = run_paths.get("run_output_dir")
        if run_output_dir and run_output_dir.exists():
            shutil.rmtree(run_output_dir)
        for path in run_paths.values():
            if isinstance(path, Path) and path != run_output_dir and path.exists() and path.is_file():
                path.unlink()

    module.ensure_run_outputs_do_not_exist = atomic_guard

    def generate_case(
        row: Any,
        index: int,
        cache: Any,
        n_steps: int,
        created_at: str,
        run_tag: str,
        strategy: str,
        run_output_dir: Path,
        attribution_target_mode: str,
        top_percentile: float,
    ) -> list[dict[str, Any]]:
        torch_module, _, _, IntegratedGradients = module.require_dependencies()
        model_name = module.safe_str(row["evaluated_model"])
        fold = module.safe_str(row["evaluation_fold"])
        image_path = module.resolve_repo_path(module.safe_str(row["image_relative_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        actual_input_sha256 = sha256_file(image_path)
        expected_input_sha256 = expected_hash_for_row(module, row)
        if expected_input_sha256 and actual_input_sha256.lower() != expected_input_sha256.lower():
            raise RuntimeError(
                f"Input SHA256 mismatch for {image_path}: expected {expected_input_sha256}, got {actual_input_sha256}"
            )

        adapter = cache.get(model_name, fold)
        checkpoint_sha256 = cache.checkpoint_hashes[(model_name, fold)]
        model_callable = module.callable_for_captum(adapter)
        image = module.open_rgb_image(image_path)
        input_tensor = adapter.preprocess_image(image)
        input_tensor.requires_grad_(True)
        baseline = torch_module.zeros_like(input_tensor)
        ig = IntegratedGradients(model_callable)
        case_id = f"xai_case_{index:04d}"
        attack_name = module.safe_str(row.get("attack_name", "none")) or "none"
        sample_id = module.safe_str(row.get("sample_id", "")) or image_path.stem
        case_dir = (
            run_output_dir
            / model_name
            / module.sanitize_tag(attack_name)
            / f"{case_id}__{module.sanitize_tag(sample_id)}"
        )

        rows: list[dict[str, Any]] = []
        requested_steps = max(8, int(n_steps))
        maximum_steps = max(requested_steps, int(config.max_n_steps))
        threshold = max(0.0, float(config.convergence_threshold))

        for target_spec in module.build_attribution_targets(row, attribution_target_mode):
            target_role = module.safe_str(target_spec["target_role"])
            target_index = int(target_spec["target_index"])
            target_label = module.safe_str(target_spec["target_label"])

            with torch_module.no_grad():
                input_output = model_callable(input_tensor)[0, target_index]
                baseline_output = model_callable(baseline)[0, target_index]
                output_difference = float((input_output - baseline_output).detach().cpu().item())

            effective_steps = requested_steps
            convergence_status = "threshold_not_met"
            normalized_delta = float("inf")
            delta_value = float("nan")
            attributions = None

            while True:
                attributions, delta = ig.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target_index,
                    n_steps=effective_steps,
                    return_convergence_delta=True,
                )
                delta_value = float(delta.detach().cpu().reshape(-1)[0].item())
                normalized_delta = abs(delta_value) / max(abs(output_difference), 1e-12)
                if normalized_delta <= threshold:
                    convergence_status = "passed"
                    break
                if config.disable_adaptive_convergence or effective_steps >= maximum_steps:
                    break
                effective_steps = min(maximum_steps, effective_steps * 2)

            if attributions is None:
                raise RuntimeError("Integrated Gradients did not produce attributions")

            title = (
                f"{case_id} | {model_name} {fold} | label={row.get('final_label')} "
                f"pred={row.get('prediction')} target={target_role}:{target_label} "
                f"conf={row.get('confidence')}"
            )
            output_paths = module.save_attribution_outputs(
                image=image,
                attributions=attributions,
                case_dir=case_dir,
                case_id=case_id,
                target_role=target_role,
                target_label=target_label,
                title=title,
                top_percentile=top_percentile,
            )
            rows.append(
                {
                    "run_tag": run_tag,
                    "case_id": case_id,
                    "created_at": created_at,
                    "selection_strategy": strategy,
                    "case_bucket": module.safe_str(row.get("case_bucket", "")),
                    "xai_selection_reason": module.safe_str(row.get("xai_selection_reason", "")),
                    "evaluated_model": model_name,
                    "evaluation_fold": fold,
                    "sample_type": module.safe_str(row.get("sample_type", "")),
                    "attack_family": module.safe_str(row.get("attack_family", "")),
                    "attack_name": attack_name,
                    "final_label": module.safe_str(row.get("final_label", "")),
                    "prediction": module.safe_str(row.get("prediction", "")),
                    "confidence": module.safe_str(row.get("confidence", "")),
                    "correct": module.safe_str(row.get("correct", "")),
                    "clean_correct": module.safe_str(row.get("clean_correct", "")),
                    "original_image_id": module.safe_str(row.get("original_image_id", "")),
                    "generated_image_id": module.safe_str(row.get("generated_image_id", "")),
                    "sample_id": sample_id,
                    "input_relative_path": module.safe_str(row.get("image_relative_path", "")),
                    "input_sha256": actual_input_sha256,
                    "expected_input_sha256": expected_input_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "manual_selected": module.safe_str(row.get("manual_selected", "")),
                    "selection_rank": module.safe_str(row.get("selection_rank", "")),
                    "reviewer_id": module.safe_str(row.get("reviewer_id", "")) or "reviewer_01",
                    "review_timestamp": module.safe_str(row.get("review_timestamp", "")),
                    **output_paths,
                    "ig_output_path": output_paths["ig_overlay_path"],
                    "attribution_target_mode": attribution_target_mode,
                    "attribution_target_role": target_role,
                    "attribution_target_index": target_index,
                    "attribution_target_label": target_label,
                    "baseline": "zero_tensor_in_preprocessed_input_space",
                    "output_difference": output_difference,
                    "convergence_delta": delta_value,
                    "abs_convergence_delta": abs(delta_value),
                    "normalized_convergence_delta": normalized_delta,
                    "convergence_threshold": threshold,
                    "convergence_status": convergence_status,
                    "requested_n_steps": requested_steps,
                    "effective_n_steps": effective_steps,
                    "top_percentile": output_paths["top_percentile"],
                    "top_percentile_threshold": output_paths["top_percentile_threshold"],
                    "method": "Integrated Gradients",
                    "n_steps": effective_steps,
                }
            )
        return rows

    module.generate_case = generate_case


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    config, forwarded = wrapper_args(sys.argv[1:])
    manual_review = "--manual-review" in forwarded
    backend = "TkAgg" if manual_review else "Agg"
    os.environ["MPLBACKEND"] = backend

    if "--reviewer-id" not in forwarded:
        forwarded.extend(["--reviewer-id", "reviewer_01"])
    sys.argv = [sys.argv[0], *forwarded]

    module = load_impl(backend)
    install_hardening(module, config)
    parsed = module.parse_args()
    run_tag = module.build_run_tag(parsed)
    run_paths = module.build_run_paths(run_tag)
    module.main()

    summary_path = run_paths["summary_json"]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["integrity_wrapper"] = {
            "entry_point": "explainability/scripts/17_generate_integrated_gradients_case_studies.py",
            "implementation": "explainability/scripts/_17_generate_integrated_gradients_case_studies_impl.py",
            "model_registry": str(Path(config.model_registry)).replace("\\", "/"),
            "checkpoint_sha256_required": True,
            "input_sha256_checked_when_available": True,
            "backend": backend,
            "baseline": "zero_tensor_in_preprocessed_input_space",
            "adaptive_convergence": not config.disable_adaptive_convergence,
            "requested_n_steps": parsed.n_steps,
            "max_n_steps": config.max_n_steps,
            "normalized_convergence_threshold": config.convergence_threshold,
            "versions": {
                "torch": package_version("torch"),
                "captum": package_version("captum"),
                "matplotlib": package_version("matplotlib"),
                "pillow": package_version("Pillow"),
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
