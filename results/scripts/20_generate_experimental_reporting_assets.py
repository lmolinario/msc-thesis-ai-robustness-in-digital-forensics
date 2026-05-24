#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_generate_experimental_reporting_assets.py

Generate experimental reporting figures, tables and summaries from consolidated FAIR-Lab metric files.

This script belongs to the reporting layer of the thesis repository. It does
not regenerate datasets, attacks, proxy-model predictions, or metrics. It only
reads already consolidated CSV files and produces publication-ready figures for
LaTeX.

Default inputs:
- results/metrics/final_core_metrics.csv
- results/metrics/final_robustness_metrics.csv
- results/metrics/final_confusion_matrices.csv
- results/metrics/final_ood_metrics.csv
- evaluation/proxy_models/proxy_model_predictions.csv  [optional]

Default outputs:
- results/figures/chapter_5/*.pdf
- results/figures/chapter_5/*.png
- results/figures/chapter_5/chapter5_figures_manifest.csv
- results/figures/chapter_5/chapter5_figures_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.utils.paths import EVALUATION_DIR, REPO_ROOT, RESULTS_DIR, repo_relative_path

SCRIPT_NAME = "results/scripts/20_generate_experimental_reporting_assets.py"

METRICS_DIR = RESULTS_DIR / "metrics"
DEFAULT_CORE_METRICS_CSV = METRICS_DIR / "final_core_metrics.csv"
DEFAULT_ROBUSTNESS_METRICS_CSV = METRICS_DIR / "final_robustness_metrics.csv"
DEFAULT_CONFUSION_MATRICES_CSV = METRICS_DIR / "final_confusion_matrices.csv"
DEFAULT_OOD_METRICS_CSV = METRICS_DIR / "final_ood_metrics.csv"
DEFAULT_PREDICTIONS_CSV = EVALUATION_DIR / "proxy_models" / "proxy_model_predictions.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "figures" / "chapter_5"

MODEL_ORDER = ("efficientnet_b0", "resnet18", "clip")
MODEL_DISPLAY = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "clip": "CLIP",
}

ADVERSARIAL_ORDER = (
    "fgsm",
    "superdeepfool",
    "sigma_zero",
    "one_pixel",
    "color_shift",
)

ANTI_FORENSIC_ORDER = (
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
)

ATTACK_DISPLAY = {
    "fgsm": "FGSM",
    "superdeepfool": "SuperDeepFool",
    "sigma_zero": "Sigma Zero",
    "one_pixel": "One Pixel",
    "color_shift": "Color Shift",
    "jpeg_recompression": "JPEG recompression",
    "resample_resize": "Resample + resize",
    "gaussian_blur": "Gaussian blur",
    "histogram_modification": "Histogram modification",
    "contrast_stretching": "Contrast stretching",
}

OOD_METRIC_DISPLAY = {
    "predicted_weapon_rate": "Weapon rate",
    "confidence_mean": "Mean confidence",
    "high_confidence_rate": "High-conf. rate",
}


# =============================================================================
# Generic utilities
# =============================================================================


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_relative_string(path: Path | str) -> str:
    """Return a repository-relative POSIX-style path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def read_required_csv(path: Path, description: str) -> pd.DataFrame:
    """Read a required CSV file and raise a clear error if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {description}: {repo_relative_string(path)}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Required CSV is empty: {repo_relative_string(path)}") from exc


def read_optional_csv(path: Path, description: str, warnings: list[str]) -> pd.DataFrame | None:
    """Read an optional CSV file, returning None if it is missing or empty."""
    if not path.is_file():
        warnings.append(f"Optional {description} not found: {repo_relative_string(path)}")
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        warnings.append(f"Optional {description} is empty: {repo_relative_string(path)}")
        return None


def ensure_columns(df: pd.DataFrame, required_columns: list[str], description: str) -> None:
    """Validate that a DataFrame contains the required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {description}: {missing}")


def ordered_existing(values: tuple[str, ...], available: set[str]) -> list[str]:
    """Return values in preferred order, keeping only those available."""
    return [value for value in values if value in available]


def display_model(model_name: str) -> str:
    """Return a thesis-friendly display name for a model identifier."""
    return MODEL_DISPLAY.get(str(model_name), str(model_name))


def display_attack(attack_name: str) -> str:
    """Return a thesis-friendly display name for an attack/transformation."""
    return ATTACK_DISPLAY.get(str(attack_name), str(attack_name).replace("_", " "))


def save_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the generated-figure manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_id",
        "figure_type",
        "source_csv",
        "output_path",
        "format",
        "created_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with deterministic indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    figure_id: str,
    figure_type: str,
    source_csv: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Save a figure in all requested formats and append rows to the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()

    for fmt in formats:
        output_path = output_dir / f"{figure_id}.{fmt}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        manifest_rows.append(
            {
                "figure_id": figure_id,
                "figure_type": figure_type,
                "source_csv": repo_relative_string(source_csv),
                "output_path": repo_relative_string(output_path),
                "format": fmt,
                "created_at": created_at,
            }
        )
        logging.info("Wrote %s", repo_relative_string(output_path))

    plt.close(fig)


def set_axis_percent(ax: plt.Axes) -> None:
    """Format y-axis values in the [0, 1] range as percentages."""
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


# =============================================================================
# Figure generators
# =============================================================================


def generate_clean_confusion_matrices(
    confusion_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate one clean confusion matrix figure per proxy model."""
    ensure_columns(
        confusion_df,
        ["evaluated_model", "sample_type", "attack_name", "tn", "fp", "fn", "tp"],
        "final_confusion_matrices.csv",
    )

    clean_df = confusion_df[
        (confusion_df["sample_type"].astype(str) == "clean")
        & (confusion_df["attack_name"].astype(str) == "clean")
    ].copy()

    if clean_df.empty:
        raise ValueError("No clean confusion matrix rows found.")

    for model in ordered_existing(MODEL_ORDER, set(clean_df["evaluated_model"].astype(str))):
        row = clean_df[clean_df["evaluated_model"].astype(str) == model].iloc[0]
        matrix = np.array(
            [
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ]
        )

        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred. non_weapon", "Pred. weapon"], rotation=20, ha="right")
        ax.set_yticklabels(["True non_weapon", "True weapon"])
        ax.set_title(f"Clean confusion matrix — {display_model(model)}")

        threshold = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text_color = "white" if matrix[i, j] > threshold else "black"
                ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=text_color, fontsize=12)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.tight_layout()

        figure_id = f"fig_clean_confusion_matrix_{model}"
        save_figure(fig, output_dir, figure_id, "clean_confusion_matrix", source_csv, formats, dpi, manifest_rows)


def generate_ood_summary_figure(
    ood_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate a grouped OOD summary chart from final_ood_metrics.csv."""
    ensure_columns(
        ood_df,
        [
            "evaluated_model",
            "predicted_weapon_rate",
            "confidence_mean",
            "high_confidence_rate",
        ],
        "final_ood_metrics.csv",
    )

    available_models = ordered_existing(MODEL_ORDER, set(ood_df["evaluated_model"].astype(str)))
    if not available_models:
        raise ValueError("No known models found in final_ood_metrics.csv.")

    plot_df = ood_df.set_index("evaluated_model").loc[available_models]
    metrics = ["predicted_weapon_rate", "confidence_mean", "high_confidence_rate"]

    x = np.arange(len(available_models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, metric in enumerate(metrics):
        values = plot_df[metric].astype(float).to_numpy()
        positions = x + (idx - 1) * width
        bars = ax.bar(positions, values, width=width, label=OOD_METRIC_DISPLAY[metric])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in available_models])
    ax.set_ylim(0, 1.08)
    set_axis_percent(ax)
    ax.set_ylabel("Rate / confidence")
    ax.set_title("Out-of-distribution behavior by proxy model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_ood_weapon_rate_and_confidence",
        "ood_summary",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def generate_optional_ood_confidence_distribution(
    predictions_df: pd.DataFrame | None,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Generate an optional OOD confidence boxplot if prediction-level data is available."""
    if predictions_df is None:
        return

    model_column = find_first_column(predictions_df, ["evaluated_model", "model", "model_name"])
    sample_type_column = find_first_column(predictions_df, ["sample_type", "condition"])
    confidence_column = find_first_column(
        predictions_df,
        ["confidence", "prediction_confidence", "predicted_confidence", "max_confidence"],
    )

    if not model_column or not sample_type_column or not confidence_column:
        warnings.append(
            "Skipped optional OOD confidence distribution: prediction CSV does not expose "
            "model/sample_type/confidence columns."
        )
        return

    ood_predictions = predictions_df[predictions_df[sample_type_column].astype(str).str.lower() == "ood"].copy()
    if ood_predictions.empty:
        warnings.append("Skipped optional OOD confidence distribution: no OOD rows found in prediction CSV.")
        return

    available_models = ordered_existing(MODEL_ORDER, set(ood_predictions[model_column].astype(str)))
    data: list[np.ndarray] = []
    labels: list[str] = []
    for model in available_models:
        values = pd.to_numeric(
            ood_predictions.loc[ood_predictions[model_column].astype(str) == model, confidence_column],
            errors="coerce",
        ).dropna()
        if not values.empty:
            data.append(values.to_numpy())
            labels.append(display_model(model))

    if not data:
        warnings.append("Skipped optional OOD confidence distribution: no numeric confidence values found.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylim(0, 1.02)
    set_axis_percent(ax)
    ax.set_ylabel("Prediction confidence")
    ax.set_title("OOD confidence distribution by proxy model")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_ood_confidence_distribution",
        "ood_confidence_distribution",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


def build_drop_matrix(
    robustness_df: pd.DataFrame,
    attack_family: str,
    attack_order: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build attack/transformation x model matrix using accuracy_drop."""
    ensure_columns(
        robustness_df,
        ["evaluated_model", "attack_family", "attack_name", "accuracy_drop"],
        "final_robustness_metrics.csv",
    )

    subset = robustness_df[robustness_df["attack_family"].astype(str) == attack_family].copy()
    if subset.empty:
        raise ValueError(f"No rows found for attack_family={attack_family!r}.")

    available_attacks = ordered_existing(attack_order, set(subset["attack_name"].astype(str)))
    available_models = ordered_existing(MODEL_ORDER, set(subset["evaluated_model"].astype(str)))

    if not available_attacks or not available_models:
        raise ValueError(f"Could not build matrix for attack_family={attack_family!r}.")

    pivot = subset.pivot_table(
        index="attack_name",
        columns="evaluated_model",
        values="accuracy_drop",
        aggfunc="first",
    )
    pivot = pivot.loc[available_attacks, available_models]
    return pivot, available_attacks, available_models


def generate_accuracy_drop_heatmap(
    robustness_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
    attack_family: str,
    attack_order: tuple[str, ...],
    figure_id: str,
    title: str,
    figure_type: str,
) -> None:
    """Generate an accuracy-drop heatmap for adversarial or anti-forensic rows."""
    pivot, attacks, models = build_drop_matrix(robustness_df, attack_family, attack_order)
    values = pivot.astype(float).to_numpy()

    max_abs = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.001)

    fig_height = 1.1 + 0.55 * len(attacks)
    fig, ax = plt.subplots(figsize=(7.6, fig_height))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy drop")

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_xticklabels([display_model(model) for model in models], rotation=25, ha="right")
    ax.set_yticklabels([display_attack(attack) for attack in attacks])
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = "n/a"
                text_color = "black"
            else:
                label = f"{value:+.3f}"
                text_color = "white" if value >= 0.71 else "black"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xlabel("Evaluated model")
    ax.set_ylabel("Perturbation" if attack_family == "adversarial" else "Transformation")
    fig.tight_layout()

    save_figure(fig, output_dir, figure_id, figure_type, source_csv, formats, dpi, manifest_rows)


def generate_max_accuracy_drop_by_model(
    robustness_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate a compact bar chart with maximum observed drop by model/family."""
    ensure_columns(
        robustness_df,
        ["evaluated_model", "attack_family", "accuracy_drop"],
        "final_robustness_metrics.csv",
    )

    subset = robustness_df[robustness_df["attack_family"].isin(["adversarial", "anti_forensic"])].copy()
    subset["accuracy_drop"] = pd.to_numeric(subset["accuracy_drop"], errors="coerce")
    subset = subset.dropna(subset=["accuracy_drop"])

    available_models = ordered_existing(MODEL_ORDER, set(subset["evaluated_model"].astype(str)))
    families = ["adversarial", "anti_forensic"]
    family_display = {"adversarial": "Adversarial", "anti_forensic": "Anti-forensic"}

    values = np.zeros((len(families), len(available_models)))
    for family_idx, family in enumerate(families):
        for model_idx, model in enumerate(available_models):
            rows = subset[(subset["attack_family"] == family) & (subset["evaluated_model"] == model)]
            values[family_idx, model_idx] = rows["accuracy_drop"].max() if not rows.empty else np.nan

    x = np.arange(len(available_models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for idx, family in enumerate(families):
        bars = ax.bar(x + (idx - 0.5) * width, values[idx], width=width, label=family_display[family])
        for bar, value in zip(bars, values[idx]):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([display_model(model) for model in available_models])
    ax.set_ylim(0, max(1.05, float(np.nanmax(values)) + 0.08))
    set_axis_percent(ax)
    ax.set_ylabel("Maximum accuracy drop")
    ax.set_title("Maximum robustness degradation by model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_max_accuracy_drop_by_model",
        "max_accuracy_drop_by_model",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready Chapter 5 figures from consolidated FAIR-Lab metrics."
    )
    parser.add_argument("--core-metrics", default=str(DEFAULT_CORE_METRICS_CSV))
    parser.add_argument("--robustness-metrics", default=str(DEFAULT_ROBUSTNESS_METRICS_CSV))
    parser.add_argument("--confusion-matrices", default=str(DEFAULT_CONFUSION_MATRICES_CSV))
    parser.add_argument("--ood-metrics", default=str(DEFAULT_OOD_METRICS_CSV))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-optional-prediction-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    core_metrics_csv = resolve_repo_path(args.core_metrics)
    robustness_metrics_csv = resolve_repo_path(args.robustness_metrics)
    confusion_matrices_csv = resolve_repo_path(args.confusion_matrices)
    ood_metrics_csv = resolve_repo_path(args.ood_metrics)
    predictions_csv = resolve_repo_path(args.predictions)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    logging.info("Reading consolidated metric files.")
    _core_df = read_required_csv(core_metrics_csv, "core metrics CSV")
    robustness_df = read_required_csv(robustness_metrics_csv, "robustness metrics CSV")
    confusion_df = read_required_csv(confusion_matrices_csv, "confusion matrices CSV")
    ood_df = read_required_csv(ood_metrics_csv, "OOD metrics CSV")

    predictions_df: pd.DataFrame | None = None
    if not args.skip_optional_prediction_figures:
        predictions_df = read_optional_csv(predictions_csv, "prediction-level CSV", warnings)

    logging.info("Generating Chapter 5 figures in %s", repo_relative_string(output_dir))

    generate_clean_confusion_matrices(
        confusion_df,
        confusion_matrices_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
    )

    generate_ood_summary_figure(
        ood_df,
        ood_metrics_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
    )

    generate_optional_ood_confidence_distribution(
        predictions_df,
        predictions_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
        warnings,
    )

    generate_accuracy_drop_heatmap(
        robustness_df,
        robustness_metrics_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
        attack_family="anti_forensic",
        attack_order=ANTI_FORENSIC_ORDER,
        figure_id="fig_anti_forensic_accuracy_drop_heatmap",
        title="Accuracy drop under anti-forensic transformations",
        figure_type="anti_forensic_accuracy_drop_heatmap",
    )

    generate_accuracy_drop_heatmap(
        robustness_df,
        robustness_metrics_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
        attack_family="adversarial",
        attack_order=ADVERSARIAL_ORDER,
        figure_id="fig_adversarial_accuracy_drop_heatmap",
        title="Accuracy drop under adversarial perturbations",
        figure_type="adversarial_accuracy_drop_heatmap",
    )

    generate_max_accuracy_drop_by_model(
        robustness_df,
        robustness_metrics_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
    )

    manifest_csv = output_dir / "chapter5_figures_manifest.csv"
    summary_json = output_dir / "chapter5_figures_summary.json"
    save_manifest_csv(manifest_csv, manifest_rows)

    figure_ids = sorted({row["figure_id"] for row in manifest_rows})
    summary = {
        "script": SCRIPT_NAME,
        "created_at": utc_now_iso(),
        "inputs": {
            "core_metrics_csv": repo_relative_string(core_metrics_csv),
            "robustness_metrics_csv": repo_relative_string(robustness_metrics_csv),
            "confusion_matrices_csv": repo_relative_string(confusion_matrices_csv),
            "ood_metrics_csv": repo_relative_string(ood_metrics_csv),
            "predictions_csv": repo_relative_string(predictions_csv),
        },
        "outputs": {
            "output_dir": repo_relative_string(output_dir),
            "manifest_csv": repo_relative_string(manifest_csv),
            "summary_json": repo_relative_string(summary_json),
        },
        "formats": args.formats,
        "dpi": args.dpi,
        "figure_count_unique": len(figure_ids),
        "file_count": len(manifest_rows),
        "figure_ids": figure_ids,
        "warnings": warnings,
        "methodological_note": (
            "This script only generates thesis figures from consolidated metric outputs. "
            "It does not regenerate datasets, perturbations, model predictions, or metrics."
        ),
    }
    save_json(summary_json, summary)

    logging.info("Wrote %s", repo_relative_string(manifest_csv))
    logging.info("Wrote %s", repo_relative_string(summary_json))

    if warnings:
        logging.warning("Completed with %d warning(s).", len(warnings))
        for warning in warnings:
            logging.warning(warning)
    else:
        logging.info("Completed without warnings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
