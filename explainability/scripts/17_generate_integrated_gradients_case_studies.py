#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_generate_integrated_gradients_case_studies.py

Generate Integrated Gradients case studies for FAIR-Lab proxy models.

The script is intended to be executed after:
    evaluation/scripts/15_evaluate_proxy_models.py

It reads:
    evaluation/proxy_models/proxy_model_predictions.csv

and supports two complementary case-selection workflows:

1. Automatic diagnostic selection
   - perturbed_failures
   - weapon_to_non_weapon
   - ood_high_confidence
   - attack_stratified
   - all

2. Manual human-in-the-loop XAI case selection
   - candidates are built according to the selected strategy
   - the reviewer explicitly selects/rejects cases through an image grid
   - reviewer decisions are saved in a run-specific manual selection DB
   - Integrated Gradients are generated only for selected cases

Outputs are run-specific and do not overwrite manifests from other strategies.

Example output files:
- explainability/outputs/integrated_gradients/<run_tag>/
- explainability/manifests/integrated_gradients_manifest__<run_tag>.csv
- explainability/manifests/xai_case_studies_manifest__<run_tag>.csv
- explainability/manifests/integrated_gradients_summary__<run_tag>.json
- explainability/manifests/xai_manual_selection_db__<run_tag>.csv
- explainability/manifests/xai_manual_selection_summary__<run_tag>.json

Methodological note:
Integrated Gradients are used as qualitative diagnostic support for transparent
proxy models. They are not generated for commercial black-box forensic tools.

Recommended thesis-oriented runs
--------------------------------

1. Critical weapon -> non_weapon failures, true-label attribution:

python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --model efficientnet_b0 \
  --strategy weapon_to_non_weapon \
  --max-cases 30 \
  --n-steps 32 \
  --attribution-target true_label \
  --top-percentile 90 \
  --force \
  --verbose

2. Critical weapon -> non_weapon failures, both true/predicted attribution:

python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --model efficientnet_b0 \
  --strategy weapon_to_non_weapon \
  --max-cases 30 \
  --n-steps 32 \
  --attribution-target both \
  --top-percentile 90 \
  --force \
  --verbose

3. High-confidence OOD cases:

python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --model efficientnet_b0 \
  --strategy ood_high_confidence \
  --high-confidence-threshold 0.90 \
  --max-cases 30 \
  --n-steps 32 \
  --attribution-target predicted_label \
  --top-percentile 90 \
  --force \
  --verbose

4. Manual attack-stratified selection, then generation:

python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --model efficientnet_b0 \
  --strategy attack_stratified \
  --cases-per-attack 3 \
  --candidate-limit 250 \
  --manual-review \
  --generate-after-manual \
  --n-steps 32 \
  --attribution-target both \
  --top-percentile 90 \
  --force \
  --verbose

5. Manual review only, without generating IG:

python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --model efficientnet_b0 \
  --strategy attack_stratified \
  --cases-per-attack 3 \
  --candidate-limit 250 \
  --manual-review \
  --manual-only \
  --force \
  --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
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
MANUAL_BACKUP_DIR = MANIFEST_DIR / "manual_selection_backups"

BATCH_SIZE = 10
N_COLS = 5
FIG_W = 18
FIG_H = 9

DEFAULT_ATTACK_ORDER = [
    "fgsm",
    "one_pixel",
    "sigma_zero",
    "superdeepfool",
    "color_shift",
    "jpeg_recompression",
    "resample_resize",
    "gaussian_blur",
    "histogram_modification",
    "contrast_stretching",
]

HELP_TEXT = """
FAIR-LAB XAI MANUAL CASE SELECTION REVIEWER

PURPOSE
Select diagnostic cases for Integrated Gradients case studies.
The reviewer does not assign class labels. It only decides whether a candidate
case should be included in the XAI case-study set.

MOUSE
- Left click   = SELECT case for XAI
- Right click  = REJECT case
- Middle click = PENDING / clear decision

KEYS
- s = SELECT current case
- r / x = REJECT current case
- p / a = PENDING / clear current decision
- u = undo last reviewer action
- g = go to page
- h = help
- t = summary
- q = save + quit
- Enter = zoom current case
- Right / Space = next batch
- Left / Backspace = previous batch
- 1..9 = select case 1..9
- 0 = select case 10

METHODOLOGICAL NOTES
- Use SELECT only for cases that are visually interpretable and diagnostically useful.
- Prefer cases that match the declared strategy.
- Avoid selecting near-duplicates unless they support a specific methodological point.
- For attack-stratified XAI, prefer representative cases across attack_name values.
""".strip()


# =============================================================================
# Generic utilities
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


def sanitize_tag(value: str) -> str:
    """Convert a run descriptor into a filesystem-safe tag."""
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


def safe_write_csv(df: pd.DataFrame, path: Path, make_backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if make_backup and path.exists():
        backup_path = MANUAL_BACKUP_DIR / (
            f"{path.stem}_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{path.suffix}"
        )
        shutil.copy2(path, backup_path)

    tmp_path = path.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False, encoding="utf-8")

    last_exc = None
    for _ in range(10):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.2)

    raise PermissionError(f"Could not replace CSV because it appears locked: {path}") from last_exc


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


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


def require_manual_review_dependencies() -> tuple[Any, Any, Any]:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseButton
    except ImportError as exc:
        raise RuntimeError("Missing manual-review dependencies: matplotlib with Tk backend.") from exc

    return mpimg, plt, MouseButton


# =============================================================================
# CLI
# =============================================================================

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
        choices=(
            "perturbed_failures",
            "weapon_to_non_weapon",
            "ood_high_confidence",
            "attack_stratified",
            "all",
        ),
        default="all",
        help="Case-selection strategy.",
    )

    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument(
        "--cases-per-attack",
        type=int,
        default=3,
        help="Cases per attack_name for --strategy attack_stratified.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=300,
        help="Maximum number of candidates shown in manual-review mode.",
    )
    parser.add_argument(
        "--attack-name",
        nargs="*",
        default=[],
        help="Optional attack_name filter for attack-stratified or manual selection.",
    )

    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=224)

    parser.add_argument(
        "--attribution-target",
        choices=("true_label", "predicted_label", "both"),
        default="true_label",
        help=(
            "Class target used for Integrated Gradients. "
            "'true_label' explains the expected/correct class when available; "
            "'predicted_label' explains the model decision; "
            "'both' generates both maps when the two classes differ."
        ),
    )

    parser.add_argument(
        "--top-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold used to generate the binary top-attribution mask.",
    )

    parser.add_argument(
        "--manual-review",
        action="store_true",
        help="Open a human-in-the-loop reviewer before generating Integrated Gradients.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Run manual selection only and do not generate Integrated Gradients.",
    )
    parser.add_argument(
        "--generate-after-manual",
        action="store_true",
        help="After manual review, generate IG only for selected cases.",
    )
    parser.add_argument(
        "--selection-manifest",
        default="",
        help=(
            "Optional CSV containing manually selected cases. "
            "If provided, the script skips automatic case selection and uses selected rows."
        ),
    )

    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for separating repeated runs with the same model and strategy.",
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


# =============================================================================
# Run paths
# =============================================================================

def build_run_tag(args: argparse.Namespace) -> str:
    model_tag = "_".join(args.model)
    parts = [model_tag, args.strategy]

    if args.strategy in {"ood_high_confidence", "all"}:
        parts.append(f"thr_{args.high_confidence_threshold:.2f}")

    if args.strategy == "attack_stratified":
        parts.append(f"per_attack_{args.cases_per_attack}")

    if args.manual_review:
        parts.append("manual")

    parts.append(f"target_{args.attribution_target}")

    if args.output_tag:
        parts.append(args.output_tag)

    return sanitize_tag("__".join(parts))


def build_run_paths(run_tag: str) -> dict[str, Path]:
    return {
        "run_output_dir": IG_OUTPUT_ROOT / run_tag,
        "ig_manifest_csv": MANIFEST_DIR / f"integrated_gradients_manifest__{run_tag}.csv",
        "case_studies_manifest_csv": MANIFEST_DIR / f"xai_case_studies_manifest__{run_tag}.csv",
        "summary_json": MANIFEST_DIR / f"integrated_gradients_summary__{run_tag}.json",
        "manual_selection_db_csv": MANIFEST_DIR / f"xai_manual_selection_db__{run_tag}.csv",
        "manual_selection_summary_json": MANIFEST_DIR / f"xai_manual_selection_summary__{run_tag}.json",
    }


def ensure_run_outputs_do_not_exist(run_paths: dict[str, Path], force: bool) -> None:
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


def ensure_manual_db_can_be_overwritten(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            "Manual selection DB for this run already exists. "
            "Use --force to overwrite/reuse this run-specific DB:\n"
            f"{path}"
        )


# =============================================================================
# Predictions and case selection
# =============================================================================

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


def enrich_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sample_type_norm"] = df["sample_type"].map(norm)
    df["final_label_norm"] = df["final_label"].map(norm)
    df["prediction_norm"] = df["prediction"].map(norm)
    df["attack_name_norm"] = df["attack_name"].map(norm)
    df["attack_family_norm"] = df["attack_family"].map(norm)
    df["confidence_numeric"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "correct" in df.columns:
        df["correct_bool"] = bool_series(df["correct"])
    else:
        df["correct_bool"] = False

    if "clean_correct" in df.columns:
        df["clean_correct_bool"] = bool_series(df["clean_correct"])
    else:
        df["clean_correct_bool"] = False

    if "true_label_confidence_delta" in df.columns:
        df["true_label_confidence_delta_numeric"] = pd.to_numeric(
            df["true_label_confidence_delta"], errors="coerce"
        )
    else:
        df["true_label_confidence_delta_numeric"] = 0.0

    return df


def perturbed_mask(df: pd.DataFrame) -> pd.Series:
    return df["sample_type_norm"].isin(
        {
            "perturbed",
            "adversarial",
            "anti_forensic",
            "anti-forensic",
            "transformed",
        }
    )


def ood_mask(df: pd.DataFrame) -> pd.Series:
    return df["sample_type_norm"].isin(
        {
            "ood",
            "out_of_distribution",
            "out-of-distribution",
        }
    )


def build_unique_case_key(row: pd.Series) -> str:
    """
    Build a stable key to avoid selecting the same visual case multiple times.

    For OOD samples, the input image path is the best unique key.
    For perturbed samples, original_image_id is preferred so that the same
    original image is not selected multiple times under different folds/attacks.
    """
    sample_type = norm(row.get("sample_type", ""))

    if sample_type in {"ood", "out_of_distribution", "out-of-distribution"}:
        key = safe_str(row.get("image_relative_path", ""))
        if key:
            return key

    original_image_id = safe_str(row.get("original_image_id", ""))
    if original_image_id:
        return original_image_id

    generated_image_id = safe_str(row.get("generated_image_id", ""))
    if generated_image_id:
        return generated_image_id

    image_relative_path = safe_str(row.get("image_relative_path", ""))
    if image_relative_path:
        return image_relative_path

    return safe_str(row.name)


def add_case_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic priority fields used for automatic and manual candidates.

    Lower priority_rank is better.
    """
    df = df.copy()

    is_weapon_to_non_weapon = (
        (df["final_label_norm"] == "weapon")
        & (df["prediction_norm"] == "non_weapon")
        & df["clean_correct_bool"]
    )

    is_clean_correct_perturbed_wrong = df["clean_correct_bool"] & (~df["correct_bool"])
    is_wrong = ~df["correct_bool"]

    df["priority_rank"] = 9
    df.loc[is_wrong, "priority_rank"] = 3
    df.loc[is_clean_correct_perturbed_wrong, "priority_rank"] = 2
    df.loc[is_weapon_to_non_weapon, "priority_rank"] = 1

    # Larger values are more interesting for ranking within the same priority.
    df["priority_score"] = df["confidence_numeric"].fillna(0.0)

    # If available, large negative true-label deltas are diagnostically useful.
    if "true_label_confidence_delta_numeric" in df.columns:
        df["true_label_drop_abs"] = df["true_label_confidence_delta_numeric"].fillna(0.0).abs()
    else:
        df["true_label_drop_abs"] = 0.0

    return df


def apply_attack_filter(df: pd.DataFrame, attack_names: list[str]) -> pd.DataFrame:
    if not attack_names:
        return df

    wanted = {sanitize_tag(x).replace("-", "_") for x in attack_names}

    tmp = df.copy()
    tmp["attack_filter_key"] = tmp["attack_name_norm"].map(lambda x: sanitize_tag(x).replace("-", "_"))

    return tmp[tmp["attack_filter_key"].isin(wanted)].copy()


def select_standard_cases(df: pd.DataFrame, strategy: str, max_cases: int, threshold: float) -> pd.DataFrame:
    df = enrich_prediction_df(df)

    p_mask = perturbed_mask(df)
    o_mask = ood_mask(df)

    parts: list[pd.DataFrame] = []

    if strategy in {"perturbed_failures", "all"}:
        parts.append(
            df[
                p_mask
                & df["clean_correct_bool"]
                & (~df["correct_bool"])
            ].sort_values("confidence_numeric", ascending=False)
        )

    if strategy in {"weapon_to_non_weapon", "all"}:
        parts.append(
            df[
                p_mask
                & (df["final_label_norm"] == "weapon")
                & (df["prediction_norm"] == "non_weapon")
                & df["clean_correct_bool"]
            ].sort_values("confidence_numeric", ascending=False)
        )

    if strategy in {"ood_high_confidence", "all"}:
        parts.append(
            df[
                o_mask
                & (df["confidence_numeric"] >= threshold)
            ].sort_values("confidence_numeric", ascending=False)
        )

    if not parts:
        return df.head(0)

    selected = pd.concat(parts, ignore_index=True)

    selected["unique_case_key"] = selected.apply(build_unique_case_key, axis=1)

    selected = selected.drop_duplicates(
        subset=["evaluated_model", "unique_case_key"],
        keep="first",
    )

    return selected.head(max_cases).copy()


def select_attack_stratified_cases(
    df: pd.DataFrame,
    cases_per_attack: int,
    max_cases: int,
    attack_names: list[str],
    candidate_limit: int,
) -> pd.DataFrame:
    df = enrich_prediction_df(df)
    df = df[perturbed_mask(df)].copy()
    df = apply_attack_filter(df, attack_names)

    if df.empty:
        return df

    df = add_case_priority(df)
    df["unique_case_key"] = df.apply(build_unique_case_key, axis=1)

    attack_order = {name: i for i, name in enumerate(DEFAULT_ATTACK_ORDER)}
    df["attack_order"] = df["attack_name_norm"].map(attack_order).fillna(999).astype(int)

    df = df.sort_values(
        [
            "attack_order",
            "attack_name_norm",
            "priority_rank",
            "true_label_drop_abs",
            "priority_score",
        ],
        ascending=[True, True, True, False, False],
        kind="stable",
    )

    selected_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(["evaluated_model", "attack_name_norm"], sort=False):
        group = group.drop_duplicates(subset=["evaluated_model", "unique_case_key"], keep="first")
        selected_parts.append(group.head(cases_per_attack))

    if not selected_parts:
        return df.head(0)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["attack_order", "attack_name_norm", "priority_rank", "priority_score"],
        ascending=[True, True, True, False],
        kind="stable",
    )

    hard_limit = max_cases if max_cases > 0 else len(selected)
    if candidate_limit > 0:
        hard_limit = min(hard_limit, candidate_limit)

    return selected.head(hard_limit).copy()


def select_cases(
    df: pd.DataFrame,
    strategy: str,
    max_cases: int,
    threshold: float,
    cases_per_attack: int,
    attack_names: list[str],
    candidate_limit: int,
) -> pd.DataFrame:
    if strategy == "attack_stratified":
        return select_attack_stratified_cases(
            df=df,
            cases_per_attack=cases_per_attack,
            max_cases=max_cases,
            attack_names=attack_names,
            candidate_limit=candidate_limit,
        )

    selected = select_standard_cases(
        df=df,
        strategy=strategy,
        max_cases=max_cases,
        threshold=threshold,
    )

    selected = apply_attack_filter(selected, attack_names)

    if candidate_limit > 0:
        selected = selected.head(candidate_limit).copy()

    return selected


# =============================================================================
# Manual XAI case reviewer
# =============================================================================

def build_manual_db(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy().reset_index(drop=True)

    for col in ["manual_xai_decision", "manual_xai_notes", "manual_reviewer_id", "manual_timestamp"]:
        if col not in df.columns:
            df[col] = ""

    if "manual_xai_state" not in df.columns:
        df["manual_xai_state"] = "pending"

    if "case_candidate_id" not in df.columns:
        df.insert(0, "case_candidate_id", [f"candidate_{i:04d}" for i in range(1, len(df) + 1)])

    return df


class ManualXAICaseReviewer:
    def __init__(self, candidates: pd.DataFrame, db_path: Path, summary_path: Path) -> None:
        self.mpimg, self.plt, self.MouseButton = require_manual_review_dependencies()

        self.db_path = db_path
        self.summary_path = summary_path
        self.reviewer_id = input("XAI reviewer ID [default: Lello]: ").strip() or "Lello"

        if self.db_path.exists():
            raw = input(f"Manual DB exists: {self.db_path}\nLoad it? [Y/n]: ").strip().lower()
            if raw in {"", "y", "yes"}:
                self.df = pd.read_csv(self.db_path, low_memory=False)
            else:
                self.df = build_manual_db(candidates)
        else:
            self.df = build_manual_db(candidates)

        self.current_start = 0
        self.selected_pos = 0
        self.batch_indices: list[int] = []
        self.ax_to_df_index: dict[Any, int] = {}
        self.last_action_stack: list[dict[str, Any]] = []

        self.fig = None
        self.axes = []
        self.status_ax = None
        self.help_fig = None
        self.summary_fig = None

        safe_write_csv(self.df, self.db_path, make_backup=False)
        self.save_summary()

    def selected_count(self) -> int:
        return int((self.df["manual_xai_decision"].map(norm) == "selected").sum())

    def rejected_count(self) -> int:
        return int((self.df["manual_xai_decision"].map(norm) == "rejected").sum())

    def pending_count(self) -> int:
        return int((self.df["manual_xai_decision"].map(norm) == "").sum())

    def save_summary(self) -> None:
        summary = {
            "created_at": utc_now_iso(),
            "manual_selection_db": repo_relative_string(self.db_path),
            "reviewer_id": self.reviewer_id,
            "candidate_rows": int(len(self.df)),
            "selected": self.selected_count(),
            "rejected": self.rejected_count(),
            "pending": self.pending_count(),
            "by_attack_name": (
                self.df.groupby(["attack_name", "manual_xai_decision"], dropna=False)
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
                if "attack_name" in self.df.columns
                else []
            ),
            "methodological_note": (
                "Manual XAI case selection chooses diagnostic cases for qualitative analysis. "
                "It does not change model predictions or ground truth labels."
            ),
        }
        write_json(self.summary_path, summary)

    def auto_save(self) -> None:
        safe_write_csv(self.df, self.db_path)
        self.save_summary()

    def get_indices(self) -> list[int]:
        return list(self.df.index)

    def update_batch_indices(self) -> None:
        indices = self.get_indices()
        if not indices:
            self.batch_indices = []
            self.selected_pos = 0
            self.current_start = 0
            return

        if self.current_start >= len(indices):
            self.current_start = max(0, ((len(indices) - 1) // BATCH_SIZE) * BATCH_SIZE)

        self.batch_indices = indices[self.current_start:self.current_start + BATCH_SIZE]

        if self.selected_pos >= len(self.batch_indices):
            self.selected_pos = 0

    def decision_color(self, decision: str) -> str:
        decision = norm(decision)
        if decision == "selected":
            return "green"
        if decision == "rejected":
            return "red"
        return "gray"

    def build_status_text(self) -> str:
        lines = [
            "FAIR-LAB XAI MANUAL CASE SELECTION",
            "",
            f"candidate_rows : {len(self.df)}",
            f"selected       : {self.selected_count()}",
            f"rejected       : {self.rejected_count()}",
            f"pending        : {self.pending_count()}",
            "",
            "Attack distribution:",
        ]

        if "attack_name" in self.df.columns:
            tmp = (
                self.df.groupby(["attack_name", "manual_xai_decision"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            for attack_name, group in tmp.groupby("attack_name"):
                selected = int(group[group["manual_xai_decision"] == "selected"]["count"].sum())
                rejected = int(group[group["manual_xai_decision"] == "rejected"]["count"].sum())
                pending = int(group[group["manual_xai_decision"] == ""]["count"].sum())
                lines.append(f"{safe_str(attack_name):<28} S={selected:>3} R={rejected:>3} P={pending:>3}")

        return "\n".join(lines)

    def _init_main_figure_if_needed(self) -> None:
        if self.fig is not None:
            return

        rows = 2
        self.fig = self.plt.figure(figsize=(FIG_W, FIG_H))
        gs = self.fig.add_gridspec(
            nrows=rows + 1,
            ncols=N_COLS,
            height_ratios=[1] * rows + [1.2],
        )

        self.axes = []
        for r in range(rows):
            for c in range(N_COLS):
                ax = self.fig.add_subplot(gs[r, c])
                self.axes.append(ax)

        self.status_ax = self.fig.add_subplot(gs[rows, :])
        self.status_ax.axis("off")

        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.05, wspace=0.08, hspace=0.35)
        self.fig.canvas.manager.set_window_title("FAIR-Lab XAI Manual Case Reviewer")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        try:
            manager = self.fig.canvas.manager
            window = manager.window
            window.geometry("1700x1000+50+50")
            window.attributes("-topmost", True)
            window.update()
            window.attributes("-topmost", False)
            window.deiconify()
            window.lift()
            window.focus_force()
        except Exception as exc:
            logging.warning("Could not force window geometry/visibility: %s", exc)

    def draw_status_panel(self) -> None:
        self.status_ax.clear()
        self.status_ax.axis("off")
        self.status_ax.text(
            0.01,
            0.98,
            self.build_status_text(),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            transform=self.status_ax.transAxes,
        )

    def resolve_candidate_image_path(self, row: pd.Series) -> Path:
        return resolve_repo_path(safe_str(row["image_relative_path"]))

    def draw_batch(self) -> None:
        self.update_batch_indices()
        self._init_main_figure_if_needed()

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        self.ax_to_df_index.clear()

        if not self.batch_indices:
            self.fig.suptitle("FAIR-Lab XAI Manual Case Reviewer [NO CANDIDATES]", fontsize=12)
            self.draw_status_panel()
            self.fig.canvas.draw_idle()
            return

        for i, df_index in enumerate(self.batch_indices):
            ax = self.axes[i]
            row = self.df.loc[df_index]
            image_path = self.resolve_candidate_image_path(row)

            if image_path.exists():
                try:
                    img = self.mpimg.imread(image_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "ERR IMG", ha="center", va="center", fontsize=10)
            else:
                ax.text(0.5, 0.5, "IMG NOT FOUND", ha="center", va="center", fontsize=10)

            decision = safe_str(row.get("manual_xai_decision", ""))
            title = (
                f"{i + 1}. {safe_str(row.get('case_candidate_id', ''))}\n"
                f"{safe_str(row.get('evaluated_model', ''))} | {safe_str(row.get('evaluation_fold', ''))}\n"
                f"{safe_str(row.get('sample_type', ''))} | {safe_str(row.get('attack_name', ''))}\n"
                f"label={safe_str(row.get('final_label', ''))} pred={safe_str(row.get('prediction', ''))} "
                f"conf={safe_str(row.get('confidence', ''))}\n"
                f"decision={decision or 'pending'}"
            )
            ax.set_title(title, fontsize=7, color=self.decision_color(decision))

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor(self.decision_color(decision))

            if i == self.selected_pos:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor("blue")

            self.ax_to_df_index[ax] = df_index

        total_pages = max(1, (len(self.df) + BATCH_SIZE - 1) // BATCH_SIZE)
        current_page = (self.current_start // BATCH_SIZE) + 1
        self.fig.suptitle(
            f"FAIR-Lab XAI Manual Case Reviewer | page {current_page}/{total_pages} | "
            "Mouse: L=select R=reject M=pending | Keys: s select, r/x reject, p/a pending, u undo, q save+quit",
            fontsize=10,
        )
        self.draw_status_panel()
        self.fig.canvas.draw_idle()

    def set_decision(self, df_index: int, decision: str) -> None:
        if decision not in {"selected", "rejected", ""}:
            return

        previous = safe_str(self.df.loc[df_index, "manual_xai_decision"])
        if previous == decision:
            return

        self.last_action_stack.append(
            {
                "df_index": int(df_index),
                "previous_decision": previous,
                "previous_state": safe_str(self.df.loc[df_index, "manual_xai_state"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "manual_xai_decision"] = decision
        self.df.at[df_index, "manual_xai_state"] = "reviewed" if decision else "pending"
        self.df.at[df_index, "manual_reviewer_id"] = self.reviewer_id if decision else ""
        self.df.at[df_index, "manual_timestamp"] = utc_now_iso() if decision else ""

        self.auto_save()
        self.draw_batch()

    def undo_last_action(self) -> None:
        if not self.last_action_stack:
            print("[INFO] No recent manual action to undo.")
            return

        item = self.last_action_stack.pop()
        df_index = item["df_index"]

        self.df.at[df_index, "manual_xai_decision"] = item["previous_decision"]
        self.df.at[df_index, "manual_xai_state"] = item["previous_state"]
        self.df.at[df_index, "manual_reviewer_id"] = self.reviewer_id if item["previous_decision"] else ""
        self.df.at[df_index, "manual_timestamp"] = utc_now_iso() if item["previous_decision"] else ""

        self.auto_save()
        self.draw_batch()
        print("[OK] Undo executed.")

    def next_batch(self) -> None:
        if self.current_start + BATCH_SIZE < len(self.df):
            self.current_start += BATCH_SIZE
            self.selected_pos = 0
        self.draw_batch()

    def prev_batch(self) -> None:
        self.current_start = max(0, self.current_start - BATCH_SIZE)
        self.selected_pos = 0
        self.draw_batch()

    def go_to_page(self) -> None:
        total_pages = max(1, (len(self.df) + BATCH_SIZE - 1) // BATCH_SIZE)
        current_page = (self.current_start // BATCH_SIZE) + 1
        raw = input(f"Go to page [1-{total_pages}] (current: {current_page}): ").strip()
        if not raw:
            return
        if not raw.isdigit():
            print(f"[WARN] Invalid page value: {raw}")
            return
        page = int(raw)
        if page < 1 or page > total_pages:
            print(f"[WARN] Page out of range: {page}")
            return
        self.current_start = (page - 1) * BATCH_SIZE
        self.selected_pos = 0
        self.draw_batch()

    def open_zoom(self) -> None:
        if not self.batch_indices:
            return
        if self.selected_pos < 0 or self.selected_pos >= len(self.batch_indices):
            return

        df_index = self.batch_indices[self.selected_pos]
        row = self.df.loc[df_index]
        image_path = self.resolve_candidate_image_path(row)
        if not image_path.exists():
            return

        try:
            img = self.mpimg.imread(image_path)
            self.plt.figure(figsize=(10, 10))
            self.plt.imshow(img)
            self.plt.title(
                f"{row.get('case_candidate_id', '')} | {row.get('evaluated_model', '')} | {row.get('attack_name', '')}\n"
                f"label={row.get('final_label', '')} pred={row.get('prediction', '')} conf={row.get('confidence', '')}"
            )
            self.plt.axis("off")
            self.plt.show()
        except Exception as exc:
            print(f"[WARN] Zoom failed: {exc}")

    def open_help_window(self) -> None:
        if self.help_fig is not None:
            try:
                self.plt.figure(self.help_fig.number)
                self.help_fig.canvas.draw_idle()
                return
            except Exception:
                self.help_fig = None

        self.help_fig, ax = self.plt.subplots(figsize=(9, 10))
        self.help_fig.canvas.manager.set_window_title("FAIR-Lab XAI Manual Reviewer Help")
        ax.axis("off")
        ax.text(0.02, 0.98, HELP_TEXT, va="top", ha="left", fontsize=11, family="monospace", wrap=True)
        self.help_fig.tight_layout()

    def open_summary_window(self) -> None:
        if self.summary_fig is None:
            self.summary_fig, ax = self.plt.subplots(figsize=(9, 11))
            self.summary_fig.canvas.manager.set_window_title("FAIR-Lab XAI Manual Selection Summary")
        else:
            ax = self.summary_fig.axes[0]

        ax.clear()
        ax.axis("off")
        ax.text(0.02, 0.98, self.build_status_text(), va="top", ha="left", fontsize=11, family="monospace")
        self.summary_fig.tight_layout()
        self.summary_fig.canvas.draw_idle()

    def on_click(self, event: Any) -> None:
        if event.inaxes not in self.ax_to_df_index:
            return

        df_index = self.ax_to_df_index[event.inaxes]
        if df_index in self.batch_indices:
            self.selected_pos = self.batch_indices.index(df_index)

        if event.button == self.MouseButton.LEFT or event.button == 1:
            self.set_decision(df_index, "selected")
            return
        if event.button == self.MouseButton.RIGHT or event.button == 3:
            self.set_decision(df_index, "rejected")
            return
        if event.button == self.MouseButton.MIDDLE or event.button == 2:
            self.set_decision(df_index, "")
            return

        self.draw_batch()

    def on_key(self, event: Any) -> None:
        key = event.key
        if key is None:
            return
        key = str(key).lower()

        if key in ["right", " "]:
            self.next_batch()
            return
        if key in ["left", "backspace"]:
            self.prev_batch()
            return
        if key == "enter":
            self.open_zoom()
            return
        if key == "q":
            self.auto_save()
            print(f"[OK] Manual selection saved: {self.db_path}")
            self.plt.close(self.fig)
            return
        if key == "h":
            self.open_help_window()
            return
        if key == "t":
            self.open_summary_window()
            return
        if key == "g":
            self.go_to_page()
            return
        if key == "u":
            self.undo_last_action()
            return

        if key.isdigit():
            pos = 9 if key == "0" else int(key) - 1
            if 0 <= pos < len(self.batch_indices):
                self.selected_pos = pos
                self.draw_batch()
            return

        if not self.batch_indices:
            return
        if self.selected_pos < 0 or self.selected_pos >= len(self.batch_indices):
            self.selected_pos = 0

        df_index = self.batch_indices[self.selected_pos]

        if key == "s":
            self.set_decision(df_index, "selected")
            return
        if key in {"r", "x"}:
            self.set_decision(df_index, "rejected")
            return
        if key in {"p", "a"}:
            self.set_decision(df_index, "")
            return

    def run(self) -> pd.DataFrame:
        self.draw_batch()
        self.open_help_window()
        self.plt.show(block=True)

        self.auto_save()

        selected = self.df[self.df["manual_xai_decision"].map(norm) == "selected"].copy()
        selected = selected.reset_index(drop=True)

        print(f"[OK] Manual XAI selection completed. Selected cases: {len(selected)}")
        print(f"[OK] Manual selection DB: {self.db_path}")
        print(f"[OK] Manual selection summary: {self.summary_path}")

        return selected


# =============================================================================
# Model adapter cache
# =============================================================================

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


# =============================================================================
# Integrated Gradients generation
# =============================================================================

def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def label_from_index(index: int) -> str:
    return "non_weapon" if index == 0 else "weapon"


def build_attribution_targets(row: pd.Series, target_mode: str) -> list[dict[str, Any]]:
    """
    Build one or more attribution targets for Integrated Gradients.

    true_label:
        Explain the expected class when final_label is one of the binary labels.
        If final_label is OOD, fall back to the predicted class.

    predicted_label:
        Explain the class actually predicted by the model.

    both:
        Generate both explanations when true and predicted classes are valid
        and different. If final_label is OOD, only predicted_label is generated.
    """
    final_label = norm(row.get("final_label", ""))
    prediction = norm(row.get("prediction", ""))

    targets: list[dict[str, Any]] = []

    true_target: dict[str, Any] | None = None
    predicted_target: dict[str, Any] | None = None

    if final_label in VALID_LABELS:
        true_target = {
            "target_role": "true_label",
            "target_index": label_to_index(final_label),
            "target_label": final_label,
        }

    if prediction in VALID_LABELS:
        predicted_target = {
            "target_role": "predicted_label",
            "target_index": label_to_index(prediction),
            "target_label": prediction,
        }

    if target_mode == "true_label":
        if true_target is not None:
            targets.append(true_target)
        elif predicted_target is not None:
            targets.append({
                **predicted_target,
                "target_role": "predicted_label_fallback_for_ood",
            })

    elif target_mode == "predicted_label":
        if predicted_target is not None:
            targets.append(predicted_target)
        elif true_target is not None:
            targets.append({
                **true_target,
                "target_role": "true_label_fallback",
            })

    elif target_mode == "both":
        if true_target is not None:
            targets.append(true_target)

        if predicted_target is not None:
            already_present = any(
                item["target_index"] == predicted_target["target_index"]
                for item in targets
            )
            if not already_present:
                targets.append(predicted_target)

        if not targets and predicted_target is not None:
            targets.append(predicted_target)

    else:
        raise ValueError(f"Unsupported attribution target mode: {target_mode}")

    if not targets:
        raise ValueError("Cannot determine attribution target.")

    return targets


def attribution_to_heatmap(np_module: Any, attributions: Any) -> Any:
    """Convert RGB attributions into a single normalized 2D heatmap."""
    heatmap = attributions.detach().cpu()[0].abs().sum(dim=0).numpy()

    min_value = float(heatmap.min())
    max_value = float(heatmap.max())

    if max_value > min_value:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    else:
        heatmap = np_module.zeros_like(heatmap)

    return heatmap


def image_to_display_array(np_module: Any, image: Image.Image, height: int, width: int) -> Any:
    resized = image.resize((width, height))
    return np_module.asarray(resized).astype("float32") / 255.0


def save_input_image(image: Image.Image, output_path: Path, height: int, width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height)).save(output_path)


def save_overlay_figure(
    np_module: Any,
    plt_module: Any,
    image: Image.Image,
    heatmap: Any,
    output_path: Path,
    title: str,
) -> None:
    height, width = heatmap.shape
    image_array = image_to_display_array(np_module, image, height, width)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt_module.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image_array)
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image_array)
    ax2.imshow(heatmap, alpha=0.45, cmap="inferno")
    ax2.set_title("Integrated Gradients overlay")
    ax2.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt_module.close(fig)


def save_grayscale_mask(heatmap: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (heatmap * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(mask, mode="L").save(output_path)


def save_top_percentile_mask(
    np_module: Any,
    heatmap: Any,
    output_path: Path,
    top_percentile: float,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold = float(np_module.percentile(heatmap, top_percentile))
    binary_mask = (heatmap >= threshold).astype("uint8") * 255

    Image.fromarray(binary_mask, mode="L").save(output_path)

    return threshold


def save_distribution_plot(
    plt_module: Any,
    heatmap: Any,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt_module.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(heatmap.flatten(), bins=50)
    ax.set_title(title)
    ax.set_xlabel("Normalized attribution value")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt_module.close(fig)


def save_attribution_outputs(
    image: Image.Image,
    attributions: Any,
    case_dir: Path,
    case_id: str,
    target_role: str,
    target_label: str,
    title: str,
    top_percentile: float,
) -> dict[str, Any]:
    _, np_module, plt_module, _ = require_dependencies()

    heatmap = attribution_to_heatmap(np_module, attributions)
    height, width = heatmap.shape

    target_tag = sanitize_tag(f"{target_role}_{target_label}")

    input_path = case_dir / f"{case_id}__input.png"
    overlay_path = case_dir / f"{case_id}__{target_tag}__overlay.png"
    mask_path = case_dir / f"{case_id}__{target_tag}__mask.png"
    top_mask_path = case_dir / f"{case_id}__{target_tag}__top{int(100 - top_percentile)}_mask.png"
    distribution_path = case_dir / f"{case_id}__{target_tag}__distribution.png"

    save_input_image(image, input_path, height, width)
    save_overlay_figure(np_module, plt_module, image, heatmap, overlay_path, title)
    save_grayscale_mask(heatmap, mask_path)

    threshold = save_top_percentile_mask(
        np_module=np_module,
        heatmap=heatmap,
        output_path=top_mask_path,
        top_percentile=top_percentile,
    )

    save_distribution_plot(
        plt_module=plt_module,
        heatmap=heatmap,
        output_path=distribution_path,
        title=f"{case_id} | {target_role}={target_label} | IG distribution",
    )

    return {
        "input_png_path": repo_relative_string(input_path),
        "ig_overlay_path": repo_relative_string(overlay_path),
        "ig_mask_path": repo_relative_string(mask_path),
        "ig_top_percentile_mask_path": repo_relative_string(top_mask_path),
        "ig_distribution_path": repo_relative_string(distribution_path),
        "top_percentile": top_percentile,
        "top_percentile_threshold": threshold,
    }


def generate_case(
    row: pd.Series,
    index: int,
    cache: AdapterCache,
    n_steps: int,
    created_at: str,
    run_tag: str,
    strategy: str,
    run_output_dir: Path,
    attribution_target_mode: str,
    top_percentile: float,
) -> list[dict[str, Any]]:
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

    ig = IntegratedGradients(model_callable)

    case_id = f"xai_case_{index:04d}"

    attack_name = safe_str(row.get("attack_name", "none")) or "none"
    sample_id = safe_str(row.get("sample_id", "")) or image_path.stem

    case_dir = (
        run_output_dir
        / model_name
        / attack_name
        / f"{case_id}__{sample_id}"
    )

    targets = build_attribution_targets(row, attribution_target_mode)

    manifest_rows: list[dict[str, Any]] = []

    for target_spec in targets:
        target_role = safe_str(target_spec["target_role"])
        target_index = int(target_spec["target_index"])
        target_label = safe_str(target_spec["target_label"])

        logging.info(
            "Computing IG for %s | target_role=%s target_label=%s",
            case_id,
            target_role,
            target_label,
        )

        attributions, delta = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_index,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        delta_value = float(delta.detach().cpu().reshape(-1)[0].item())
        abs_delta_value = abs(delta_value)

        title = (
            f"{case_id} | {model_name} {fold} | "
            f"label={row.get('final_label')} "
            f"pred={row.get('prediction')} "
            f"target={target_role}:{target_label} "
            f"conf={row.get('confidence')}"
        )

        output_paths = save_attribution_outputs(
            image=image,
            attributions=attributions,
            case_dir=case_dir,
            case_id=case_id,
            target_role=target_role,
            target_label=target_label,
            title=title,
            top_percentile=top_percentile,
        )

        overlay_path = output_paths["ig_overlay_path"]

        manifest_rows.append(
            {
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
                "manual_xai_decision": safe_str(row.get("manual_xai_decision", "")),
                "manual_reviewer_id": safe_str(row.get("manual_reviewer_id", "")),
                "manual_timestamp": safe_str(row.get("manual_timestamp", "")),
                "ig_output_path": overlay_path,
                "input_png_path": output_paths["input_png_path"],
                "ig_overlay_path": output_paths["ig_overlay_path"],
                "ig_mask_path": output_paths["ig_mask_path"],
                "ig_top_percentile_mask_path": output_paths["ig_top_percentile_mask_path"],
                "ig_distribution_path": output_paths["ig_distribution_path"],
                "attribution_target_mode": attribution_target_mode,
                "attribution_target_role": target_role,
                "attribution_target_index": target_index,
                "attribution_target_label": target_label,
                "convergence_delta": delta_value,
                "abs_convergence_delta": abs_delta_value,
                "top_percentile": output_paths["top_percentile"],
                "top_percentile_threshold": output_paths["top_percentile_threshold"],
                "method": "Integrated Gradients",
                "n_steps": n_steps,
            }
        )

    return manifest_rows


# =============================================================================
# Selection manifest loading
# =============================================================================

def load_selected_cases_from_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if "manual_xai_decision" in df.columns:
        selected = df[df["manual_xai_decision"].map(norm) == "selected"].copy()
    else:
        selected = df.copy()

    if selected.empty:
        raise RuntimeError(f"Selection manifest contains no selected cases: {path}")

    required = {"evaluated_model", "evaluation_fold", "image_relative_path", "final_label", "prediction"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Selection manifest is missing required columns: {sorted(missing)}")

    return selected.reset_index(drop=True)


# =============================================================================
# Main
# =============================================================================

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

    if args.selection_manifest:
        cases = load_selected_cases_from_manifest(repo_relative_path(args.selection_manifest))
        logging.info("Loaded selected XAI cases from manifest: %d", len(cases))
    else:
        df = load_predictions(predictions_path, list(args.model))

        cases = select_cases(
            df=df,
            strategy=args.strategy,
            max_cases=args.max_cases,
            threshold=args.high_confidence_threshold,
            cases_per_attack=args.cases_per_attack,
            attack_names=args.attack_name,
            candidate_limit=args.candidate_limit,
        )

        if cases.empty:
            raise RuntimeError(
                "No XAI cases selected. "
                "Check strategy, selected model, sample_type values, confidence threshold, "
                "attack_name filters, or prediction CSV contents."
            )

        logging.info("Candidate XAI cases: %d", len(cases))

        if args.manual_review:
            ensure_manual_db_can_be_overwritten(run_paths["manual_selection_db_csv"], args.force)

            reviewer = ManualXAICaseReviewer(
                candidates=cases,
                db_path=run_paths["manual_selection_db_csv"],
                summary_path=run_paths["manual_selection_summary_json"],
            )
            cases = reviewer.run()

            if args.manual_only or not args.generate_after_manual:
                logging.info("Manual review completed without IG generation.")
                logging.info("Manual selection DB written: %s", run_paths["manual_selection_db_csv"])
                logging.info("Manual selection summary written: %s", run_paths["manual_selection_summary_json"])
                return

    if cases.empty:
        raise RuntimeError("No selected XAI cases available for Integrated Gradients generation.")

    logging.info("Selected XAI cases for IG generation: %d", len(cases))

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

        case_rows = generate_case(
            row=row,
            index=index,
            cache=cache,
            n_steps=args.n_steps,
            created_at=created_at,
            run_tag=run_tag,
            strategy=args.strategy,
            run_output_dir=run_paths["run_output_dir"],
            attribution_target_mode=args.attribution_target,
            top_percentile=args.top_percentile,
        )

        rows.extend(case_rows)

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
            "cases_per_attack": args.cases_per_attack,
            "attack_name_filter": list(args.attack_name),
            "candidate_limit": args.candidate_limit,
            "manual_review": bool(args.manual_review),
            "manual_selection_db": (
                repo_relative_string(run_paths["manual_selection_db_csv"])
                if run_paths["manual_selection_db_csv"].exists()
                else ""
            ),
            "selected_cases": len(cases),
            "generated_attribution_rows": len(rows),
            "generated_cases": len(rows),
            "n_steps": args.n_steps,
            "high_confidence_threshold": args.high_confidence_threshold,
            "attribution_target": args.attribution_target,
            "top_percentile": args.top_percentile,
            "device": args.device,
            "input_size": args.input_size,
            "outputs": {
                "integrated_gradients_manifest": repo_relative_string(run_paths["ig_manifest_csv"]),
                "case_studies_manifest": repo_relative_string(run_paths["case_studies_manifest_csv"]),
                "integrated_gradients_output_dir": repo_relative_string(run_paths["run_output_dir"]),
            },
            "methodological_note": (
                "Integrated Gradients are diagnostic support for transparent proxy models, "
                "not black-box forensic tools. Manual XAI case selection, when enabled, "
                "selects diagnostic examples only and does not alter ground truth or predictions."
            ),
        },
    )

    logging.info("Integrated Gradients manifest written: %s", run_paths["ig_manifest_csv"])
    logging.info("Case studies manifest written: %s", run_paths["case_studies_manifest_csv"])
    logging.info("Summary written: %s", run_paths["summary_json"])
    logging.info("Output images written under: %s", run_paths["run_output_dir"])


if __name__ == "__main__":
    main()
