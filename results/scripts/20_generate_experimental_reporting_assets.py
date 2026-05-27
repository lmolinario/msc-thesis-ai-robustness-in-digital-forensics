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
- results/metrics/forensic_tools_metrics.csv           [optional]
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

SCRIPT_NAME = "evaluation/scripts/20_generate_experimental_reporting_assets.py"

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

DEFAULT_FORENSIC_TOOLS_METRICS_CSV = METRICS_DIR / "forensic_tools_metrics.csv"

FORENSIC_TOOL_ORDER = (
    "magnet_axiom",
    "xways_excire_d20",
    "xways_excire_d50",
    "xways_excire_d80",
)

FORENSIC_TOOL_DISPLAY = {
    "magnet_axiom": "Magnet AXIOM",
    "xways_excire": "Excire",
    "xways_excire_d20": "Excire D20",
    "xways_excire_d50": "Excire D50",
    "xways_excire_d80": "Excire D80",
}

FORENSIC_FAMILY_DISPLAY = {
    "adversarial": "Adversarial",
    "anti_forensic": "Anti-forensic",
}

FORENSIC_METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "recall_weapon": "Recall weapon",
    "precision_weapon": "Precision weapon",
    "false_negative_rate": "FNR",
    "false_positive_rate": "FPR",
    "ood_weapon_flag_rate": "OOD weapon flag rate",
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


def display_forensic_tool(tool_name: str) -> str:
    """Return a thesis-friendly display name for a forensic tool/configuration."""
    return FORENSIC_TOOL_DISPLAY.get(str(tool_name), str(tool_name).replace("_", " "))


def ordered_forensic_tools(available: set[str]) -> list[str]:
    """Return forensic tool identifiers in the preferred order, preserving unknown tools."""
    ordered = [tool for tool in FORENSIC_TOOL_ORDER if tool in available]
    remaining = sorted(tool for tool in available if tool not in set(ordered))
    return ordered + remaining


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




def save_table_csv(
    df: pd.DataFrame,
    output_dir: Path,
    table_id: str,
    source_csv: Path,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Save a thesis-ready CSV table and append it to the reporting manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_id}.csv"
    df.to_csv(output_path, index=False)
    manifest_rows.append(
        {
            "figure_id": table_id,
            "figure_type": "table_csv",
            "source_csv": repo_relative_string(source_csv),
            "output_path": repo_relative_string(output_path),
            "format": "csv",
            "created_at": utc_now_iso(),
        }
    )
    logging.info("Wrote %s", repo_relative_string(output_path))


def to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric values when they are present."""
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output

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

                if attack_family == "adversarial":
                    text_color = "white" if value >= 0.71 else "black"
                elif attack_family == "anti_forensic":
                    text_color = "white" if value >= 0.02 else "black"
                else:
                    text_color = "black"

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
# Forensic-tool reporting generators
# =============================================================================


def filter_forensic_metrics(
    forensic_df: pd.DataFrame,
    scope: str,
    sample_type: str | None = None,
    attack_family: str | None = None,
    attack_name: str | None = None,
) -> pd.DataFrame:
    """Filter forensic-tool metrics by the grouping columns emitted by script 19."""
    subset = forensic_df[forensic_df["scope"].astype(str) == scope].copy()
    if sample_type is not None:
        subset = subset[subset["sample_type"].astype(str) == sample_type]
    if attack_family is not None:
        subset = subset[subset["attack_family"].astype(str) == attack_family]
    if attack_name is not None:
        subset = subset[subset["attack_name"].astype(str) == attack_name]
    return subset


def prepare_forensic_metric_table(df: pd.DataFrame, include_attack_name: bool = False) -> pd.DataFrame:
    """Return a compact table with the most relevant forensic-tool metrics."""
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_weapon",
        "recall_weapon",
        "false_negative_rate",
        "false_positive_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "ood_rows",
        "ood_weapon_flags",
        "ood_weapon_flag_rate",
    ]
    available = [column for column in metric_columns if column in df.columns]
    base_columns = ["tool_name", "scope", "sample_type", "attack_family"]
    if include_attack_name:
        base_columns.append("attack_name")
    out = df[[column for column in base_columns if column in df.columns] + available].copy()
    out.insert(1, "tool_display", out["tool_name"].map(display_forensic_tool))
    if "attack_name" in out.columns:
        out["attack_display"] = out["attack_name"].map(display_attack)
    return out


def generate_forensic_tables(
    forensic_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate CSV tables for commercial forensic-tool reporting."""
    ensure_columns(
        forensic_df,
        [
            "tool_name",
            "scope",
            "sample_type",
            "attack_family",
            "attack_name",
            "accuracy",
            "recall_weapon",
            "false_negative_rate",
            "false_positive_rate",
            "ood_weapon_flag_rate",
        ],
        "forensic_tools_metrics.csv",
    )

    clean_df = filter_forensic_metrics(forensic_df, "sample_type", sample_type="clean")
    ood_df = filter_forensic_metrics(forensic_df, "sample_type", sample_type="ood")
    family_df = forensic_df[
        (forensic_df["scope"].astype(str) == "attack_family")
        & (forensic_df["attack_family"].astype(str).isin(["adversarial", "anti_forensic"]))
    ].copy()
    attack_df = forensic_df[
        (forensic_df["scope"].astype(str) == "attack_name")
        & (forensic_df["attack_family"].astype(str).isin(["adversarial", "anti_forensic"]))
    ].copy()

    for name, table_df, include_attack in [
        ("tab_forensic_tools_clean_metrics", clean_df, False),
        ("tab_forensic_tools_ood_metrics", ood_df, False),
        ("tab_forensic_tools_attack_family_metrics", family_df, False),
        ("tab_forensic_tools_attack_name_metrics", attack_df, True),
    ]:
        if table_df.empty:
            logging.warning("Skipped %s: no rows available.", name)
            continue
        save_table_csv(
            prepare_forensic_metric_table(table_df, include_attack_name=include_attack),
            output_dir,
            name,
            source_csv,
            manifest_rows,
        )

    sensitivity_rows: list[dict[str, Any]] = []
    available_tools = ordered_forensic_tools(set(forensic_df["tool_name"].astype(str)))
    for tool in available_tools:
        clean_row = clean_df[clean_df["tool_name"].astype(str) == tool]
        ood_row = ood_df[ood_df["tool_name"].astype(str) == tool]
        adv_row = family_df[
            (family_df["tool_name"].astype(str) == tool)
            & (family_df["attack_family"].astype(str) == "adversarial")
        ]
        anti_row = family_df[
            (family_df["tool_name"].astype(str) == tool)
            & (family_df["attack_family"].astype(str) == "anti_forensic")
        ]

        def get_value(frame: pd.DataFrame, column: str) -> Any:
            if frame.empty or column not in frame.columns:
                return ""
            return frame.iloc[0][column]

        sensitivity_rows.append(
            {
                "tool_name": tool,
                "tool_display": display_forensic_tool(tool),
                "clean_accuracy": get_value(clean_row, "accuracy"),
                "clean_recall_weapon": get_value(clean_row, "recall_weapon"),
                "clean_fnr": get_value(clean_row, "false_negative_rate"),
                "clean_fpr": get_value(clean_row, "false_positive_rate"),
                "ood_weapon_flag_rate": get_value(ood_row, "ood_weapon_flag_rate"),
                "adversarial_accuracy": get_value(adv_row, "accuracy"),
                "adversarial_recall_weapon": get_value(adv_row, "recall_weapon"),
                "adversarial_fnr": get_value(adv_row, "false_negative_rate"),
                "adversarial_fpr": get_value(adv_row, "false_positive_rate"),
                "anti_forensic_accuracy": get_value(anti_row, "accuracy"),
                "anti_forensic_recall_weapon": get_value(anti_row, "recall_weapon"),
                "anti_forensic_fnr": get_value(anti_row, "false_negative_rate"),
                "anti_forensic_fpr": get_value(anti_row, "false_positive_rate"),
            }
        )

    save_table_csv(
        pd.DataFrame(sensitivity_rows),
        output_dir,
        "tab_forensic_tools_sensitivity_summary",
        source_csv,
        manifest_rows,
    )


def generate_forensic_clean_comparison_figure(
    forensic_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate a clean-set comparison figure for Magnet and Excire settings."""
    clean_df = filter_forensic_metrics(forensic_df, "sample_type", sample_type="clean")
    if clean_df.empty:
        raise ValueError("No clean rows found in forensic_tools_metrics.csv.")

    metrics = ["accuracy", "recall_weapon", "false_negative_rate", "false_positive_rate"]
    clean_df = to_numeric_columns(clean_df, metrics)

    tools = ordered_forensic_tools(set(clean_df["tool_name"].astype(str)))
    plot_df = clean_df.set_index("tool_name").loc[tools]
    x = np.arange(len(tools))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    for idx, metric in enumerate(metrics):
        values = plot_df[metric].astype(float).to_numpy()
        bars = ax.bar(x + (idx - 1.5) * width, values, width=width, label=FORENSIC_METRIC_DISPLAY[metric])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([display_forensic_tool(tool) for tool in tools], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    set_axis_percent(ax)
    ax.set_ylabel("Metric value")
    ax.set_title("Clean-set comparison of forensic AI tools")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_forensic_tools_clean_metrics_comparison",
        "forensic_tools_clean_metrics_comparison",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


def generate_forensic_ood_weapon_flag_figure(
    forensic_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    """Generate OOD weapon-flag comparison for Magnet and Excire settings."""
    ood_df = filter_forensic_metrics(forensic_df, "sample_type", sample_type="ood")
    if ood_df.empty:
        raise ValueError("No OOD rows found in forensic_tools_metrics.csv.")
    ood_df = to_numeric_columns(ood_df, ["ood_weapon_flag_rate", "ood_weapon_flags", "ood_rows"])

    tools = ordered_forensic_tools(set(ood_df["tool_name"].astype(str)))
    plot_df = ood_df.set_index("tool_name").loc[tools]
    values = plot_df["ood_weapon_flag_rate"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(np.arange(len(tools)), values)
    for bar, value, flags, total in zip(
        bars,
        values,
        plot_df["ood_weapon_flags"].astype(float).to_numpy(),
        plot_df["ood_rows"].astype(float).to_numpy(),
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{value:.3f}\n({int(flags)}/{int(total)})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels([display_forensic_tool(tool) for tool in tools], rotation=20, ha="right")
    ax.set_ylim(0, min(1.0, max(0.1, float(np.nanmax(values)) + 0.18)))
    set_axis_percent(ax)
    ax.set_ylabel("OOD weapon flag rate")
    ax.set_title("OOD weapon flag rate in forensic AI tools")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "fig_forensic_tools_ood_weapon_flag_rate",
        "forensic_tools_ood_weapon_flag_rate",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


def generate_forensic_attack_family_comparison_figure(
    forensic_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
    metric: str,
    figure_id: str,
    title: str,
) -> None:
    """Generate a grouped bar chart comparing adversarial and anti-forensic families."""
    family_df = forensic_df[
        (forensic_df["scope"].astype(str) == "attack_family")
        & (forensic_df["attack_family"].astype(str).isin(["adversarial", "anti_forensic"]))
    ].copy()
    if family_df.empty:
        raise ValueError("No attack-family rows found in forensic_tools_metrics.csv.")

    family_df = to_numeric_columns(family_df, [metric])
    tools = ordered_forensic_tools(set(family_df["tool_name"].astype(str)))
    families = ["anti_forensic", "adversarial"]

    x = np.arange(len(tools))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.4, 5.0))

    for idx, family in enumerate(families):
        values = []
        for tool in tools:
            rows = family_df[
                (family_df["tool_name"].astype(str) == tool)
                & (family_df["attack_family"].astype(str) == family)
            ]
            values.append(float(rows.iloc[0][metric]) if not rows.empty else np.nan)
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=FORENSIC_FAMILY_DISPLAY[family],
        )
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([display_forensic_tool(tool) for tool in tools], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    set_axis_percent(ax)
    ax.set_ylabel(FORENSIC_METRIC_DISPLAY.get(metric, metric))
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    save_figure(fig, output_dir, figure_id, "forensic_tools_attack_family_comparison", source_csv, formats, dpi, manifest_rows)


def build_forensic_accuracy_drop_matrix(
    forensic_df: pd.DataFrame,
    attack_family: str,
    attack_order: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build attack/transformation x tool matrix using clean accuracy minus attack accuracy."""
    clean_df = filter_forensic_metrics(forensic_df, "sample_type", sample_type="clean")
    attack_df = forensic_df[
        (forensic_df["scope"].astype(str) == "attack_name")
        & (forensic_df["attack_family"].astype(str) == attack_family)
    ].copy()
    if clean_df.empty or attack_df.empty:
        raise ValueError(f"Cannot build forensic drop matrix for {attack_family}: missing clean or attack rows.")

    clean_df = to_numeric_columns(clean_df, ["accuracy"])
    attack_df = to_numeric_columns(attack_df, ["accuracy"])

    tools = ordered_forensic_tools(set(attack_df["tool_name"].astype(str)))
    available_attacks = ordered_existing(attack_order, set(attack_df["attack_name"].astype(str)))
    matrix = pd.DataFrame(index=available_attacks, columns=tools, dtype=float)

    clean_accuracy = clean_df.set_index("tool_name")["accuracy"].to_dict()
    for _, row in attack_df.iterrows():
        attack_name = str(row["attack_name"])
        tool_name = str(row["tool_name"])
        if attack_name in matrix.index and tool_name in matrix.columns and tool_name in clean_accuracy:
            matrix.loc[attack_name, tool_name] = float(clean_accuracy[tool_name]) - float(row["accuracy"])

    return matrix, available_attacks, tools


def generate_forensic_accuracy_drop_heatmap(
    forensic_df: pd.DataFrame,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
    attack_family: str,
    attack_order: tuple[str, ...],
    figure_id: str,
    title: str,
) -> None:
    """Generate accuracy-drop heatmap for forensic tools and Excire distance settings."""
    pivot, attacks, tools = build_forensic_accuracy_drop_matrix(forensic_df, attack_family, attack_order)
    values = pivot.astype(float).to_numpy()
    max_value = max(float(np.nanmax(values)), 0.001)

    fig_height = 1.2 + 0.55 * len(attacks)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))
    image = ax.imshow(values, cmap="Reds", vmin=0, vmax=max_value, aspect="auto")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy drop vs clean")

    ax.set_xticks(np.arange(len(tools)))
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_xticklabels([display_forensic_tool(tool) for tool in tools], rotation=25, ha="right")
    ax.set_yticklabels([display_attack(attack) for attack in attacks])
    ax.set_title(title)

    threshold = max_value * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = "n/a"
                text_color = "black"
            else:
                label = f"{value:+.3f}"
                text_color = "white" if value >= threshold else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xlabel("Forensic tool / setting")
    ax.set_ylabel("Perturbation" if attack_family == "adversarial" else "Transformation")
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        figure_id,
        f"forensic_tools_{attack_family}_accuracy_drop_heatmap",
        source_csv,
        formats,
        dpi,
        manifest_rows,
    )


def generate_forensic_reporting_assets(
    forensic_df: pd.DataFrame | None,
    source_csv: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    manifest_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Generate all commercial forensic-tool reporting assets when metrics are available."""
    if forensic_df is None:
        return

    forensic_df = forensic_df.copy()
    required_columns = ["tool_name", "scope", "sample_type", "attack_family", "attack_name"]
    missing = [column for column in required_columns if column not in forensic_df.columns]
    if missing:
        warnings.append(f"Skipped forensic-tool reporting assets: missing columns {missing}.")
        return

    logging.info("Generating forensic-tool reporting assets.")

    numeric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_weapon",
        "recall_weapon",
        "false_negative_rate",
        "false_positive_rate",
        "ood_weapon_flag_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "ood_rows",
        "ood_weapon_flags",
    ]
    forensic_df = to_numeric_columns(forensic_df, numeric_columns)

    generate_forensic_tables(forensic_df, source_csv, output_dir, manifest_rows)
    generate_forensic_clean_comparison_figure(forensic_df, source_csv, output_dir, formats, dpi, manifest_rows)
    generate_forensic_ood_weapon_flag_figure(forensic_df, source_csv, output_dir, formats, dpi, manifest_rows)

    generate_forensic_attack_family_comparison_figure(
        forensic_df,
        source_csv,
        output_dir,
        formats,
        dpi,
        manifest_rows,
        metric="accuracy",
        figure_id="fig_forensic_tools_attack_family_accuracy",
        title="Forensic-tool accuracy under perturbation families",
    )
    generate_forensic_attack_family_comparison_figure(
        forensic_df,
        source_csv,
        output_dir,
        formats,
        dpi,
        manifest_rows,
        metric="recall_weapon",
        figure_id="fig_forensic_tools_attack_family_recall_weapon",
        title="Forensic-tool weapon recall under perturbation families",
    )
    generate_forensic_attack_family_comparison_figure(
        forensic_df,
        source_csv,
        output_dir,
        formats,
        dpi,
        manifest_rows,
        metric="false_positive_rate",
        figure_id="fig_forensic_tools_attack_family_fpr",
        title="Forensic-tool false positive rate under perturbation families",
    )

    generate_forensic_accuracy_drop_heatmap(
        forensic_df,
        source_csv,
        output_dir,
        formats,
        dpi,
        manifest_rows,
        attack_family="anti_forensic",
        attack_order=ANTI_FORENSIC_ORDER,
        figure_id="fig_forensic_tools_anti_forensic_accuracy_drop_heatmap",
        title="Forensic-tool accuracy drop under anti-forensic transformations",
    )

    generate_forensic_accuracy_drop_heatmap(
        forensic_df,
        source_csv,
        output_dir,
        formats,
        dpi,
        manifest_rows,
        attack_family="adversarial",
        attack_order=ADVERSARIAL_ORDER,
        figure_id="fig_forensic_tools_adversarial_accuracy_drop_heatmap",
        title="Forensic-tool accuracy drop under adversarial perturbations",
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
    parser.add_argument("--forensic-tools-metrics", default=str(DEFAULT_FORENSIC_TOOLS_METRICS_CSV))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-optional-prediction-figures", action="store_true")
    parser.add_argument("--skip-forensic-tool-assets", action="store_true")
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
    forensic_tools_metrics_csv = resolve_repo_path(args.forensic_tools_metrics)
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

    forensic_tools_df: pd.DataFrame | None = None
    if not args.skip_forensic_tool_assets:
        forensic_tools_df = read_optional_csv(
            forensic_tools_metrics_csv,
            "forensic-tool metrics CSV",
            warnings,
        )

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

    generate_forensic_reporting_assets(
        forensic_tools_df,
        forensic_tools_metrics_csv,
        output_dir,
        args.formats,
        args.dpi,
        manifest_rows,
        warnings,
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
            "forensic_tools_metrics_csv": repo_relative_string(forensic_tools_metrics_csv),
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
