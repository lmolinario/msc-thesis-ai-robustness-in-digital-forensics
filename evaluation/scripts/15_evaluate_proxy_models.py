#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity wrapper for the frozen FAIRLab proxy-model evaluation.

The original evaluated implementation is preserved in
`_15_evaluate_proxy_models_impl.py`. This public entry point adds fail-closed
input validation, fixed manifest discovery, strict JSON serialization and
protection against partial runs overwriting canonical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
IMPL_PATH = SCRIPT_DIR / "_15_evaluate_proxy_models_impl.py"

OFFICIAL_ADVERSARIAL_MANIFEST_NAMES = (
    "adversarial_color_shift_manifest.csv",
    "adversarial_fgsm_efficientnet_b0_manifest.csv",
    "adversarial_one_pixel_efficientnet_b0_manifest.csv",
    "adversarial_sigma_zero_efficientnet_b0_manifest.csv",
    "adversarial_superdeepfool_efficientnet_b0_manifest.csv",
)


def load_implementation() -> Any:
    spec = importlib.util.spec_from_file_location("fairlab_proxy_eval_impl", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluation implementation: {IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--diagnostic-output-dir",
        default="",
        help="Required for partial, limited, or single-fold diagnostic runs.",
    )
    return parser.parse_known_args(argv)


def configure_output_paths(impl: Any, root_value: str) -> None:
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    else:
        root = root.resolve()

    impl.PROXY_EVAL_DIR = root / "proxy_models"
    impl.METRICS_DIR = root / "metrics"
    impl.PREDICTIONS_CSV = impl.PROXY_EVAL_DIR / "proxy_model_predictions.csv"
    impl.SUMMARY_JSON = impl.METRICS_DIR / "proxy_model_evaluation_summary.json"
    impl.FINAL_CORE_METRICS_CSV = impl.METRICS_DIR / "final_core_metrics.csv"
    impl.FINAL_ROBUSTNESS_METRICS_CSV = impl.METRICS_DIR / "final_robustness_metrics.csv"
    impl.FINAL_CONFUSION_MATRICES_CSV = impl.METRICS_DIR / "final_confusion_matrices.csv"
    impl.FINAL_OOD_METRICS_CSV = impl.METRICS_DIR / "final_ood_metrics.csv"


def fixed_manifest_discovery(manifests_dir: Path) -> list[Path]:
    paths = [manifests_dir / name for name in OFFICIAL_ADVERSARIAL_MANIFEST_NAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen adversarial manifests: {missing}")
    return paths


def strict_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_samples(impl: Any, samples: list[dict[str, Any]]) -> None:
    counts = Counter()
    for sample in samples:
        if sample["sample_type"] == "clean":
            counts["clean"] += 1
        if sample["sample_type"] == "ood":
            counts["ood"] += 1
        if sample["attack_family"] == "adversarial":
            counts["adversarial"] += 1
        if sample["attack_family"] == "anti_forensic":
            counts["anti_forensic"] += 1

    expected = Counter(
        clean=1000,
        ood=500,
        adversarial=5000,
        anti_forensic=5000,
    )
    if counts != expected or len(samples) != 11500:
        raise ValueError(
            f"Frozen evaluation profile mismatch: expected {dict(expected)}, "
            f"found {dict(counts)}, total={len(samples)}"
        )

    missing: list[str] = []
    mismatched: list[str] = []
    for sample in samples:
        image_path = impl.resolve_repo_path(sample["image_relative_path"])
        if not image_path.is_file():
            missing.append(impl.repo_relative_string(image_path))
            continue
        expected_sha = impl.safe_str(sample.get("image_sha256_manifest", "")).lower()
        if expected_sha:
            actual_sha = sha256_file(image_path)
            if actual_sha != expected_sha:
                mismatched.append(
                    f"{sample['sample_id']}: expected {expected_sha}, found {actual_sha}"
                )

    if missing:
        raise FileNotFoundError(
            "Controlled image artifacts are missing. Restore the authorized data "
            "and regenerate steps 11, 13, 14 and 16 before step 15. "
            f"First missing paths: {missing[:10]}"
        )
    if mismatched:
        raise ValueError(f"Input SHA256 verification failed: {mismatched[:10]}")


def validate_checkpoints(impl: Any, checkpoint_root: Path) -> None:
    registry_path = REPO_ROOT / "models" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for model_name in impl.DEFAULT_MODELS:
        expected_hashes = registry["models"][model_name]["sha256"]
        for fold in impl.DEFAULT_FOLDS:
            checkpoint = checkpoint_root / model_name / f"{fold}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            actual_sha = sha256_file(checkpoint)
            expected_sha = str(expected_hashes.get(fold, "")).lower()
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Checkpoint SHA256 mismatch for {model_name}/{fold}: "
                    f"expected {expected_sha}, found {actual_sha}"
                )


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    impl = load_implementation()
    impl.discover_adversarial_manifests = fixed_manifest_discovery
    impl.write_json = strict_write_json

    original_attack_samples = impl.attack_samples

    def corrected_attack_samples(path: Path, expected_family: str) -> list[dict[str, Any]]:
        rows = original_attack_samples(path, expected_family)
        for row in rows:
            if expected_family == "anti_forensic" or row.get("attack_name") == "color_shift":
                row["attack_target_model"] = "model_agnostic"
        return rows

    impl.attack_samples = corrected_attack_samples

    original_evaluate_sample = impl.evaluate_sample

    def fail_closed_evaluate_sample(
        sample: dict[str, Any],
        model_name: str,
        fold: str,
        cache: Any,
    ) -> dict[str, Any]:
        result = original_evaluate_sample(sample, model_name, fold, cache)
        if str(result.get("error", "")).strip():
            raise RuntimeError(
                f"Evaluation failed for {result.get('sample_id')} / "
                f"{model_name} / {fold}: {result['error']}"
            )
        return result

    impl.evaluate_sample = fail_closed_evaluate_sample

    args = impl.parse_args()
    official_profile = (
        tuple(args.model) == tuple(impl.DEFAULT_MODELS)
        and args.ood_fold_mode == "all"
        and args.limit == 0
    )
    if not official_profile and not wrapper_args.diagnostic_output_dir:
        raise ValueError(
            "Partial, limited, or single-fold runs cannot overwrite canonical outputs. "
            "Provide --diagnostic-output-dir."
        )
    if wrapper_args.diagnostic_output_dir:
        configure_output_paths(impl, wrapper_args.diagnostic_output_dir)

    samples = impl.load_all_samples(args)
    checkpoint_root = impl.repo_relative_path(args.checkpoint_root)
    validate_samples(impl, samples)
    validate_checkpoints(impl, checkpoint_root)

    impl.main()


if __name__ == "__main__":
    main()
