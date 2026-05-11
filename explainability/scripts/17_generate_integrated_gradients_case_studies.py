#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_generate_integrated_gradients_case_studies.py

Generate Integrated Gradients case studies for FAIR-Lab proxy models.

The script is intended to be executed after:
    evaluation/scripts/15_evaluate_proxy_models.py

It reads:
    evaluation/proxy_models/proxy_model_predictions.csv

and selects representative diagnostic cases such as:
- clean-correct -> perturbed-wrong;
- weapon -> non_weapon failures;
- high-confidence OOD predictions.

Outputs are run-specific and do not overwrite manifests from other strategies.

Example output files:
- explainability/outputs/integrated_gradients/<run_tag>/
- explainability/manifests/integrated_gradients_manifest__<run_tag>.csv
- explainability/manifests/xai_case_studies_manifest__<run_tag>.csv
- explainability/manifests/integrated_gradients_summary__<run_tag>.json

Methodological note:
Integrated Gradients are used as qualitative diagnostic support for transparent
proxy models. They are not generated for commercial black-box forensic tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.attacks.adversarial_model_interface import TargetModelConfig, label_to_index
from datasets.scripts.attacks.adversarial_torch_model_adapters import build_target_model_adapter
from datasets.scripts.utils.paths import EVALUATION_DIR, EXPLAINABILITY_DIR, REPO_ROOT, repo_relative_path

SCRIPT_NAME = "explainability/scripts/17_generate_integrated_gradients_case_studies.py"

SUPPORTED_MODELS = ("resnet18", "efficientnet_b0", "clip")
VALID_LABELS = ("non_weapon", "weapon")

DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"
DEFAULT_PREDICTIONS_CSV = EVALUATION_DIR / "proxy_models" / "proxy_model_predictions.csv"

IG_OUTPUT_ROOT = EXPLAINABILITY_DIR / "outputs" / "integrated_gradients"
CASE_STUDIES_DIR = EXPLAINABILITY_DIR / "outputs" / "case_studies"
MANIFEST_DIR = EXPLAINABILITY_DIR / "manifests"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


def sanitize_tag(value: str) -> str:
    """
    Convert a run descriptor into a filesystem-safe tag.
    """
    value = value.strip().lower()
    value = value.replace(".", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def repo_relative_string(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Integrated Gradients XAI case studies.")

    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))

    parser.add_argument(
        "--model",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=["efficientnet_b0"],
        help="One or more proxy models to explain.",
    )

    parser.add_argument(
        "--strategy",
        choices=("perturbed_failures", "weapon_to_non_weapon", "ood_high_confidence", "all"),
        default="all",
        help="Case-selection strategy.",
    )

    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=224)

    parser.add_argument(
        "--output-tag",
        default="",
        help=(
            "Optional suffix for separating repeated runs with the same model and strategy. "
            "Example: --output-tag thesis_round_1"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite only the manifest/output files for the current run tag.",
    )

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def require_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError(
            "Missing XAI dependencies. Install captum, torch, numpy and matplotlib."
        ) from exc

    return torch, np, plt, IntegratedGradients


def build_run_tag(args: argparse.Namespace) -> str:
    """
    Build a stable run tag from model(s), strategy and relevant parameters.

    Examples:
    - efficientnet_b0__all__thr_0_90
    - efficientnet_b0__weapon_to_non_weapon
    - efficientnet_b0__ood_high_confidence__thr_0_90
    - resnet18_clip__all__thr_0_90
    """
    model_tag = "_".join(args.model)
    parts = [model_tag, args.strategy]

    if args.strategy in {"ood_high_confidence", "all"}:
        threshold_tag = f"thr_{args.high_confidence_threshold:.2f}"
        parts.append(threshold_tag)

    if args.output_tag:
        parts.append(args.output_tag)

    return sanitize_tag("__".join(parts))


def build_run_paths(run_tag: str) -> dict[str, Path]:
    """
    Create run-specific paths so different strategies do not overwrite each other.
    """
    return {
        "run_output_dir": IG_OUTPUT_ROOT / run_tag,
        "ig_manifest_csv": MANIFEST_DIR / f"integrated_gradients_manifest__{run_tag}.csv",
        "case_studies_manifest_csv": MANIFEST_DIR / f"xai_case_studies_manifest__{run_tag}.csv",
        "summary_json": MANIFEST_DIR / f"integrated_gradients_summary__{run_tag}.json",
    }


def ensure_run_outputs_do_not_exist(run_paths: dict[str, Path], force: bool) -> None:
    """
    Prevent accidental overwrites of the current run outputs unless --force is used.
    """
    manifest_paths = [
        run_paths["ig_manifest_csv"],
        run_paths["case_studies_manifest_csv"],
        run_paths["summary_json"],
    ]

    existing = [path for path in manifest_paths if path.exists()]

    if existing and not force:
        existing_list = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "XAI output manifest(s) for this run already exist. "
            "Use --force to overwrite only this run.\n"
            f"{existing_list}"
        )


class AdapterCache:
    def __init__(self, checkpoint_root: Path, device: str, input_size: int) -> None:
        self.checkpoint_root = checkpoint_root
        self.device = device
        self.input_size = input_size
        self.cache: dict[tuple[str, str], Any] = {}

    def get(self, model_name: str, fold: str) -> Any:
        key = (model_name, fold)

        if key in self.cache:
            return self.cache[key]

        checkpoint_path = self.checkpoint_root / model_name / f"{fold}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        config = TargetModelConfig(
            name=model_name,
            checkpoint_path=checkpoint_path,
            device=self.device,
            input_size=self.input_size,
        )

        adapter = build_target_model_adapter(config)
        adapter.load_model()

        self.cache[key] = adapter

        logging.info("Loaded %s %s", model_name, fold)

        return adapter


def callable_for_captum(adapter: Any) -> Any:
    """
    Return a callable compatible with Captum IntegratedGradients.

    Standard PyTorch adapters expose _model.
    CLIP adapter may expose _forward_logits instead.
    """
    if hasattr(adapter, "_model") and adapter._model is not None:
        return adapter._model

    if hasattr(adapter, "_forward_logits"):

        class ClipCallable:
            def __init__(self, wrapped: Any) -> None:
                self.wrapped = wrapped

            def __call__(self, x: Any) -> Any:
                return self.wrapped._forward_logits(x)

            def zero_grad(self, set_to_none: bool = True) -> None:
                if getattr(self.wrapped, "_clip_model", None) is not None:
                    self.wrapped._clip_model.zero_grad(set_to_none=set_to_none)

                if getattr(self.wrapped, "_binary_head", None) is not None:
                    self.wrapped._binary_head.zero_grad(set_to_none=set_to_none)

        return ClipCallable(adapter)

    raise RuntimeError("Unsupported adapter for Integrated Gradients.")


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_predictions(path: Path, models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {
        "evaluated_model",
        "evaluation_fold",
        "sample_type",
        "attack_family",
        "attack_name",
        "final_label",
        "prediction",
        "confidence",
        "image_relative_path",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    available_models = sorted(df["evaluated_model"].dropna().astype(str).unique())

    df = df[df["evaluated_model"].astype(str).isin(models)].copy()

    if df.empty:
        raise RuntimeError(
            "No rows found for the selected model(s). "
            f"Selected models: {models}. "
            f"Available models in CSV: {available_models}"
        )

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()

    if df.empty:
        raise RuntimeError(
            "All rows were removed after filtering the error column. "
            "Check whether the prediction CSV contains actual errors or malformed values."
        )

    logging.info("Prediction rows after filtering: %d", len(df))

    return df


def select_cases(df: pd.DataFrame, strategy: str, max_cases: int, threshold: float) -> pd.DataFrame:
    """
    Select diagnostic XAI cases.

    The script primarily expects:
    - sample_type == perturbed for adversarial/anti-forensic transformed images;
    - sample_type == ood for out-of-distribution samples.

    A few alternative sample_type names are also tolerated to make the script more robust.
    """
    df = df.copy()

    df["sample_type_norm"] = df["sample_type"].map(norm)
    df["final_label_norm"] = df["final_label"].map(norm)
    df["prediction_norm"] = df["prediction"].map(norm)

    df["confidence_numeric"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "correct" in df.columns:
        df["correct_bool"] = bool_series(df["correct"])
    else:
        df["correct_bool"] = False

    if "clean_correct" in df.columns:
        df["clean_correct_bool"] = bool_series(df["clean_correct"])
    else:
        df["clean_correct_bool"] = False

    perturbed_mask = df["sample_type_norm"].isin(
        {
            "perturbed",
            "adversarial",
            "anti_forensic",
            "anti-forensic",
            "transformed",
        }
    )

    ood_mask = df["sample_type_norm"].isin({"ood", "out_of_distribution", "out-of-distribution"})

    parts: list[pd.DataFrame] = []

    if strategy in {"perturbed_failures", "all"}:
        parts.append(
            df[
                perturbed_mask
                & df["clean_correct_bool"]
                & (~df["correct_bool"])
            ].sort_values("confidence_numeric", ascending=False)
        )

    if strategy in {"weapon_to_non_weapon", "all"}:
        parts.append(
            df[
                perturbed_mask
                & (df["final_label_norm"] == "weapon")
                & (df["prediction_norm"] == "non_weapon")
                & df["clean_correct_bool"]
            ].sort_values("confidence_numeric", ascending=False)
        )

    if strategy in {"ood_high_confidence", "all"}:
        parts.append(
            df[
                ood_mask
                & (df["confidence_numeric"] >= threshold)
            ].sort_values("confidence_numeric", ascending=False)
        )

    if not parts:
        return df.head(0)

    selected = pd.concat(parts, ignore_index=True)

    selected = selected.drop_duplicates(
        subset=["evaluated_model", "evaluation_fold", "image_relative_path"],
        keep="first",
    )

    return selected.head(max_cases).copy()


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def tensor_to_numpy_image(torch_module: Any, tensor: Any) -> Any:
    array = tensor.detach().cpu()[0].permute(1, 2, 0).numpy()

    min_value = array.min()
    max_value = array.max()

    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)

    return array


def attribution_target(row: pd.Series) -> int:
    """
    Determine the class for which Integrated Gradients are computed.

    Priority:
    1. final_label, when it is a valid binary task label;
    2. prediction, when final_label is not available or is OOD.
    """
    final_label = norm(row.get("final_label", ""))
    prediction = norm(row.get("prediction", ""))

    if final_label in VALID_LABELS:
        return label_to_index(final_label)

    if prediction in VALID_LABELS:
        return label_to_index(prediction)

    raise ValueError("Cannot determine attribution target.")


def save_figure(input_tensor: Any, attributions: Any, output_path: Path, title: str) -> None:
    torch_module, _, plt, _ = require_dependencies()

    image_array = tensor_to_numpy_image(torch_module, input_tensor)

    heatmap = attributions.detach().cpu()[0].abs().sum(dim=0).numpy()

    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image_array)
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image_array)
    ax2.imshow(heatmap, alpha=0.45, cmap="inferno")
    ax2.set_title("Integrated Gradients")
    ax2.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_case(
    row: pd.Series,
    index: int,
    cache: AdapterCache,
    n_steps: int,
    created_at: str,
    run_tag: str,
    strategy: str,
    run_output_dir: Path,
) -> dict[str, Any]:
    torch_module, _, _, IntegratedGradients = require_dependencies()

    model_name = safe_str(row["evaluated_model"])
    fold = safe_str(row["evaluation_fold"])

    image_path = resolve_repo_path(safe_str(row["image_relative_path"]))

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    adapter = cache.get(model_name, fold)
    model_callable = callable_for_captum(adapter)

    image = open_rgb_image(image_path)
    input_tensor = adapter.preprocess_image(image)
    input_tensor.requires_grad_(True)

    baseline = torch_module.zeros_like(input_tensor)

    target = attribution_target(row)

    ig = IntegratedGradients(model_callable)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
    )

    case_id = f"xai_case_{index:04d}"

    attack_name = safe_str(row.get("attack_name", "none")) or "none"
    sample_id = safe_str(row.get("sample_id", "")) or image_path.stem

    output_path = (
        run_output_dir
        / model_name
        / attack_name
        / f"{case_id}__{sample_id}.png"
    )

    title = (
        f"{case_id} | {model_name} {fold} | "
        f"label={row.get('final_label')} "
        f"pred={row.get('prediction')} "
        f"conf={row.get('confidence')}"
    )

    save_figure(input_tensor, attributions, output_path, title)

    return {
        "run_tag": run_tag,
        "case_id": case_id,
        "created_at": created_at,
        "selection_strategy": strategy,
        "evaluated_model": model_name,
        "evaluation_fold": fold,
        "sample_type": safe_str(row.get("sample_type", "")),
        "attack_family": safe_str(row.get("attack_family", "")),
        "attack_name": attack_name,
        "final_label": safe_str(row.get("final_label", "")),
        "prediction": safe_str(row.get("prediction", "")),
        "confidence": safe_str(row.get("confidence", "")),
        "correct": safe_str(row.get("correct", "")),
        "clean_correct": safe_str(row.get("clean_correct", "")),
        "original_image_id": safe_str(row.get("original_image_id", "")),
        "generated_image_id": safe_str(row.get("generated_image_id", "")),
        "sample_id": sample_id,
        "input_relative_path": safe_str(row.get("image_relative_path", "")),
        "ig_output_path": repo_relative_string(output_path),
        "attribution_target_index": target,
        "attribution_target_label": "non_weapon" if target == 0 else "weapon",
        "method": "Integrated Gradients",
        "n_steps": n_steps,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    require_dependencies()

    run_tag = build_run_tag(args)
    run_paths = build_run_paths(run_tag)

    ensure_run_outputs_do_not_exist(run_paths, args.force)

    predictions_path = repo_relative_path(args.predictions_csv)
    checkpoint_root = repo_relative_path(args.checkpoint_root)

    logging.info("Run tag: %s", run_tag)
    logging.info("Predictions CSV: %s", predictions_path)
    logging.info("Checkpoint root: %s", checkpoint_root)

    df = load_predictions(predictions_path, list(args.model))

    cases = select_cases(
        df=df,
        strategy=args.strategy,
        max_cases=args.max_cases,
        threshold=args.high_confidence_threshold,
    )

    if cases.empty:
        raise RuntimeError(
            "No XAI cases selected. "
            "Check strategy, selected model, sample_type values, confidence threshold, "
            "or prediction CSV contents."
        )

    logging.info("Selected XAI cases: %d", len(cases))

    cache = AdapterCache(
        checkpoint_root=checkpoint_root,
        device=args.device,
        input_size=args.input_size,
    )

    run_paths["run_output_dir"].mkdir(parents=True, exist_ok=True)
    CASE_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    created_at = utc_now_iso()

    rows: list[dict[str, Any]] = []

    for index, (_, row) in enumerate(cases.iterrows(), start=1):
        logging.info("Generating XAI case %d/%d", index, len(cases))

        rows.append(
            generate_case(
                row=row,
                index=index,
                cache=cache,
                n_steps=args.n_steps,
                created_at=created_at,
                run_tag=run_tag,
                strategy=args.strategy,
                run_output_dir=run_paths["run_output_dir"],
            )
        )

    write_csv(run_paths["ig_manifest_csv"], rows)
    write_csv(run_paths["case_studies_manifest_csv"], rows)

    write_json(
        run_paths["summary_json"],
        {
            "script": SCRIPT_NAME,
            "created_at": created_at,
            "run_tag": run_tag,
            "input_predictions_csv": repo_relative_string(predictions_path),
            "checkpoint_root": repo_relative_string(checkpoint_root),
            "models": list(args.model),
            "strategy": args.strategy,
            "max_cases": args.max_cases,
            "generated_cases": len(rows),
            "n_steps": args.n_steps,
            "high_confidence_threshold": args.high_confidence_threshold,
            "device": args.device,
            "input_size": args.input_size,
            "outputs": {
                "integrated_gradients_manifest": repo_relative_string(run_paths["ig_manifest_csv"]),
                "case_studies_manifest": repo_relative_string(run_paths["case_studies_manifest_csv"]),
                "integrated_gradients_output_dir": repo_relative_string(run_paths["run_output_dir"]),
            },
            "methodological_note": (
                "Integrated Gradients are diagnostic support for transparent proxy models, "
                "not black-box forensic tools."
            ),
        },
    )

    logging.info("Integrated Gradients manifest written: %s", run_paths["ig_manifest_csv"])
    logging.info("Case studies manifest written: %s", run_paths["case_studies_manifest_csv"])
    logging.info("Summary written: %s", run_paths["summary_json"])
    logging.info("Output images written under: %s", run_paths["run_output_dir"])


if __name__ == "__main__":
    main()