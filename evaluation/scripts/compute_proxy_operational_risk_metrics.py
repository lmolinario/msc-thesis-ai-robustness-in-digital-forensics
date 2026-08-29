from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Import FAIRLab repository paths
# =============================================================================

def _add_utils_to_path() -> None:
    """
    Make datasets/scripts/utils/paths.py importable regardless of whether the
    script is launched from PyCharm or from the command line.
    """
    current = Path(__file__).resolve()

    for candidate in [current, *current.parents]:
        utils_dir = candidate / "datasets" / "scripts" / "utils"
        if (utils_dir / "paths.py").is_file():
            sys.path.insert(0, str(utils_dir))
            return

    raise RuntimeError(
        "Could not locate datasets/scripts/utils/paths.py from "
        f"{current}"
    )


_add_utils_to_path()

from paths import DOCS_IMAGES_DIR, EVALUATION_DIR, RESULTS_DIR  # noqa: E402


# =============================================================================
# Input and output paths
# =============================================================================

PROXY_PREDICTIONS_PATH = (
    EVALUATION_DIR / "proxy_models" / "proxy_model_predictions.csv"
)

METRICS_DIR = RESULTS_DIR / "metrics"

PROXY_METRICS_OUTPUT = (
    METRICS_DIR / "proxy_operational_risk_metrics.csv"
)

RISK_SUMMARY_CSV_OUTPUT = (
    METRICS_DIR / "operational_risk_summary_data.csv"
)

RISK_SUMMARY_PDF_OUTPUT = (
    DOCS_IMAGES_DIR / "fig_results_operational_risk_summary.pdf"
)

RISK_SUMMARY_PNG_OUTPUT = (
    DOCS_IMAGES_DIR / "fig_results_operational_risk_summary.png"
)


# =============================================================================
# Black-box software metrics from Chapter 5
# =============================================================================
# These values are taken from the consolidated results already reported in
# Chapter 5. They are included here so that this script can regenerate the final
# operational risk summary figure without manual editing.

BLACK_BOX_ROWS = [
    {
        "system": "Magnet AXIOM / Magnet.AI",
        "system_type": "Black-box",
        "clean_fnr": 0.070,
        "overall_fpr": 0.035,
        "ood_weapon_rate": 0.360,
        "adversarial_fnr": 0.116,
        "anti_forensic_fnr": 0.087,
        "adversarial_drop": 0.027,
        "anti_forensic_drop": 0.010,
    },
    {
        "system": "Excire Foto 2025 D50",
        "system_type": "Black-box",
        "clean_fnr": 0.042,
        "overall_fpr": 0.099,
        "ood_weapon_rate": 0.340,
        "adversarial_fnr": 0.062,
        "anti_forensic_fnr": 0.043,
        "adversarial_drop": 0.040,
        "anti_forensic_drop": 0.003,
    },
    {
        "system": "Cellebrite Inseyets",
        "system_type": "Black-box",
        "clean_fnr": 0.032,
        "overall_fpr": 0.042,
        "ood_weapon_rate": 0.292,
        "adversarial_fnr": 0.050,
        "anti_forensic_fnr": 0.037,
        "adversarial_drop": 0.009,
        "anti_forensic_drop": 0.004,
    },
    {
        "system": "Magnet Griffeye / T3K CORE",
        "system_type": "Black-box",
        "clean_fnr": 0.036,
        "overall_fpr": 0.007,
        "ood_weapon_rate": 0.260,
        "adversarial_fnr": 0.055,
        "anti_forensic_fnr": 0.046,
        "adversarial_drop": 0.009,
        "anti_forensic_drop": 0.004,
    },
]


PROXY_DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "clip": "CLIP",
}


SYSTEM_ORDER = [
    "EfficientNet-B0",
    "ResNet18",
    "CLIP",
    "Magnet AXIOM / Magnet.AI",
    "Excire Foto 2025 D50",
    "Cellebrite Inseyets",
    "Magnet Griffeye / T3K CORE",
]


RISK_COLUMNS = [
    "clean_fnr",
    "overall_fpr",
    "ood_weapon_rate",
    "adversarial_fnr",
    "anti_forensic_fnr",
    "adversarial_degradation",
    "anti_forensic_degradation",
]


RISK_COLUMN_LABELS = [
    "Clean\nFNR",
    "Overall\nFPR",
    "OOD\nW-rate",
    "Adv.\nFNR",
    "Anti-for.\nFNR",
    "Adv.\ndegradation",
    "Anti-for.\ndegradation",
]


# =============================================================================
# Metric helpers
# =============================================================================

def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def compute_binary_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    """
    Compute binary metrics for labels weapon/non_weapon.

    Positive class: weapon.
    Negative class: non_weapon.
    """
    tp = ((df["final_label"] == "weapon") & (df["prediction"] == "weapon")).sum()
    fn = ((df["final_label"] == "weapon") & (df["prediction"] == "non_weapon")).sum()
    fp = ((df["final_label"] == "non_weapon") & (df["prediction"] == "weapon")).sum()
    tn = ((df["final_label"] == "non_weapon") & (df["prediction"] == "non_weapon")).sum()

    total = int(tp + tn + fp + fn)

    accuracy = (tp + tn) / total if total else None
    fnr = fn / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None

    return {
        "samples": total,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
    }


def compute_ood_weapon_rate(df: pd.DataFrame) -> dict[str, float | int | None]:
    """
    Compute OOD weapon flag rate for proxy predictions.
    """
    ood = df[df["final_label"] == "ood"].copy()
    total = len(ood)

    if total == 0:
        return {
            "ood_samples": 0,
            "ood_pred_weapon": 0,
            "ood_pred_non_weapon": 0,
            "ood_weapon_rate": None,
        }

    pred_weapon = int((ood["prediction"] == "weapon").sum())
    pred_non_weapon = int((ood["prediction"] == "non_weapon").sum())

    return {
        "ood_samples": int(total),
        "ood_pred_weapon": pred_weapon,
        "ood_pred_non_weapon": pred_non_weapon,
        "ood_weapon_rate": pred_weapon / total,
    }


# =============================================================================
# Proxy metrics
# =============================================================================

def load_proxy_predictions() -> pd.DataFrame:
    if not PROXY_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Proxy prediction file not found: {PROXY_PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PROXY_PREDICTIONS_PATH, low_memory=False)

    required_columns = {
        "evaluated_model",
        "final_label",
        "prediction",
        "attack_family",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in proxy prediction file: "
            f"{sorted(missing)}"
        )

    df["evaluated_model"] = _normalize_string_series(df["evaluated_model"])
    df["final_label"] = _normalize_string_series(df["final_label"])
    df["prediction"] = _normalize_string_series(df["prediction"])
    df["attack_family"] = _normalize_string_series(df["attack_family"])

    return df


def build_proxy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per proxy model with clean, overall, adversarial,
    anti-forensic, and OOD metrics.
    """
    rows: list[dict[str, float | int | str | None]] = []

    for model_name, model_df in df.groupby("evaluated_model"):
        binary = model_df[
            model_df["final_label"].isin(["weapon", "non_weapon"])
        ].copy()

        clean = binary[binary["attack_family"] == "none"]
        adversarial = binary[binary["attack_family"] == "adversarial"]
        anti_forensic = binary[binary["attack_family"] == "anti_forensic"]

        clean_m = compute_binary_metrics(clean)
        overall_m = compute_binary_metrics(binary)
        adv_m = compute_binary_metrics(adversarial)
        anti_m = compute_binary_metrics(anti_forensic)
        ood_m = compute_ood_weapon_rate(model_df)

        clean_accuracy = clean_m["accuracy"]
        adv_accuracy = adv_m["accuracy"]
        anti_accuracy = anti_m["accuracy"]

        adversarial_drop = (
            clean_accuracy - adv_accuracy
            if clean_accuracy is not None and adv_accuracy is not None
            else None
        )

        anti_forensic_drop = (
            clean_accuracy - anti_accuracy
            if clean_accuracy is not None and anti_accuracy is not None
            else None
        )

        rows.append(
            {
                "model": model_name,
                "display_model": PROXY_DISPLAY_NAMES.get(model_name, model_name),

                "clean_samples": clean_m["samples"],
                "clean_accuracy": clean_m["accuracy"],
                "clean_fnr": clean_m["fnr"],
                "clean_fpr": clean_m["fpr"],
                "clean_tp": clean_m["tp"],
                "clean_tn": clean_m["tn"],
                "clean_fp": clean_m["fp"],
                "clean_fn": clean_m["fn"],

                "overall_binary_samples": overall_m["samples"],
                "overall_accuracy": overall_m["accuracy"],
                "overall_fnr": overall_m["fnr"],
                "overall_fpr": overall_m["fpr"],
                "overall_tp": overall_m["tp"],
                "overall_tn": overall_m["tn"],
                "overall_fp": overall_m["fp"],
                "overall_fn": overall_m["fn"],

                "adversarial_samples": adv_m["samples"],
                "adversarial_accuracy": adv_m["accuracy"],
                "adversarial_fnr": adv_m["fnr"],
                "adversarial_fpr": adv_m["fpr"],
                "adversarial_tp": adv_m["tp"],
                "adversarial_tn": adv_m["tn"],
                "adversarial_fp": adv_m["fp"],
                "adversarial_fn": adv_m["fn"],

                "anti_forensic_samples": anti_m["samples"],
                "anti_forensic_accuracy": anti_m["accuracy"],
                "anti_forensic_fnr": anti_m["fnr"],
                "anti_forensic_fpr": anti_m["fpr"],
                "anti_forensic_tp": anti_m["tp"],
                "anti_forensic_tn": anti_m["tn"],
                "anti_forensic_fp": anti_m["fp"],
                "anti_forensic_fn": anti_m["fn"],

                "ood_samples": ood_m["ood_samples"],
                "ood_pred_weapon": ood_m["ood_pred_weapon"],
                "ood_pred_non_weapon": ood_m["ood_pred_non_weapon"],
                "ood_weapon_rate": ood_m["ood_weapon_rate"],

                "adversarial_drop": adversarial_drop,
                "anti_forensic_drop": anti_forensic_drop,
            }
        )

    out = pd.DataFrame(rows)

    order_map = {
        "EfficientNet-B0": 0,
        "ResNet18": 1,
        "CLIP": 2,
    }

    out["sort_order"] = out["display_model"].map(order_map).fillna(999)
    out = out.sort_values("sort_order").drop(columns=["sort_order"])

    return out


def validate_proxy_counts(proxy_metrics: pd.DataFrame) -> None:
    """
    Validate expected sample counts. Raises an error if the expected FAIRLab
    evaluation structure is not found.
    """
    expected = {
        "clean_samples": 1000,
        "adversarial_samples": 5000,
        "anti_forensic_samples": 5000,
        "overall_binary_samples": 11000,
        "ood_samples": 2500,
    }

    for _, row in proxy_metrics.iterrows():
        model = row["display_model"]

        for column, expected_value in expected.items():
            actual = int(row[column])
            if actual != expected_value:
                raise ValueError(
                    f"Unexpected {column} for {model}: "
                    f"expected {expected_value}, got {actual}"
                )


# =============================================================================
# Final risk summary table
# =============================================================================

def build_operational_risk_summary(proxy_metrics: pd.DataFrame) -> pd.DataFrame:
    proxy_rows = []

    for _, row in proxy_metrics.iterrows():
        proxy_rows.append(
            {
                "system": row["display_model"],
                "system_type": "Proxy",
                "clean_fnr": row["clean_fnr"],
                "overall_fpr": row["overall_fpr"],
                "ood_weapon_rate": row["ood_weapon_rate"],
                "adversarial_fnr": row["adversarial_fnr"],
                "anti_forensic_fnr": row["anti_forensic_fnr"],
                "adversarial_drop": row["adversarial_drop"],
                "anti_forensic_drop": row["anti_forensic_drop"],
            }
        )

    summary = pd.DataFrame(proxy_rows + BLACK_BOX_ROWS)

    # For the figure, negative drops are set to zero because the visual summary
    # represents operational degradation. Raw drops remain available in the CSV.
    summary["adversarial_degradation"] = summary["adversarial_drop"].clip(lower=0)
    summary["anti_forensic_degradation"] = summary["anti_forensic_drop"].clip(lower=0)

    order_map = {name: idx for idx, name in enumerate(SYSTEM_ORDER)}
    summary["sort_order"] = summary["system"].map(order_map).fillna(999)
    summary = summary.sort_values("sort_order").drop(columns=["sort_order"])

    return summary


# =============================================================================
# Figure generation
# =============================================================================

def _format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{100 * float(value):.1f}%"


def plot_operational_risk_summary(summary: pd.DataFrame) -> None:
    """
    Generate a compact thesis-ready risk matrix.

    The cell shading represents the metric value on a 0--1 scale. The displayed
    value is the actual percentage. The figure intentionally contains only
    quantitative indicators.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plot_data = summary[RISK_COLUMNS].astype(float).to_numpy()
    systems = summary["system"].tolist()
    system_types = summary["system_type"].tolist()

    fig_width = 12.8
    fig_height = 5.7

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    matrix = ax.imshow(
        plot_data,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="Greys",
    )

    ax.set_xticks(np.arange(len(RISK_COLUMN_LABELS)))
    ax.set_xticklabels(RISK_COLUMN_LABELS, fontsize=9)

    row_labels = [
        f"{system}\n({system_type})"
        for system, system_type in zip(systems, system_types)
    ]

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Annotate each cell with the percentage value.
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            value = plot_data[i, j]
            text_color = "white" if value >= 0.50 else "black"
            ax.text(
                j,
                i,
                _format_percent(value),
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    # Minor grid lines.
    ax.set_xticks(np.arange(-0.5, len(RISK_COLUMN_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Horizontal separator between proxy models and black-box software systems.
    proxy_count = int((summary["system_type"] == "Proxy").sum())
    ax.axhline(proxy_count - 0.5, linewidth=1.5)

    ax.set_title(
        "Operational risk summary across evaluated systems",
        fontsize=12,
        pad=12,
    )

    cbar = fig.colorbar(matrix, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Risk indicator value", rotation=90, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.01,
        0.01,
        (
            "Note: drops below zero are clipped to 0 in the degradation columns "
            "because the figure summarizes operational risk. Raw values are "
            "preserved in operational_risk_summary_data.csv."
        ),
        fontsize=7.5,
        ha="left",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    RISK_SUMMARY_PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RISK_SUMMARY_PNG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(RISK_SUMMARY_PDF_OUTPUT, bbox_inches="tight")
    fig.savefig(RISK_SUMMARY_PNG_OUTPUT, dpi=300, bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading proxy predictions from: {PROXY_PREDICTIONS_PATH}")

    predictions = load_proxy_predictions()
    proxy_metrics = build_proxy_metrics(predictions)

    validate_proxy_counts(proxy_metrics)

    proxy_metrics.to_csv(PROXY_METRICS_OUTPUT, index=False)

    summary = build_operational_risk_summary(proxy_metrics)
    summary.to_csv(RISK_SUMMARY_CSV_OUTPUT, index=False)

    plot_operational_risk_summary(summary)

    print("\nGenerated files:")
    print(f" - {PROXY_METRICS_OUTPUT}")
    print(f" - {RISK_SUMMARY_CSV_OUTPUT}")
    print(f" - {RISK_SUMMARY_PDF_OUTPUT}")
    print(f" - {RISK_SUMMARY_PNG_OUTPUT}")

    print("\nOperational risk summary:")
    columns_to_print = [
        "system",
        "system_type",
        "clean_fnr",
        "overall_fpr",
        "ood_weapon_rate",
        "adversarial_fnr",
        "anti_forensic_fnr",
        "adversarial_drop",
        "anti_forensic_drop",
    ]
    print(summary[columns_to_print].to_string(index=False))


if __name__ == "__main__":
    main()