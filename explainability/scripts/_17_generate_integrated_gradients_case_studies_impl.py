#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_generate_integrated_gradients_case_studies.py

Generate Integrated Gradients case studies for FAIRLab proxy models.

Supported workflows
-------------------
1. Automatic diagnostic XAI:
   - perturbed_failures
   - weapon_to_non_weapon
   - ood_high_confidence
   - chapter5_core
   - all

2. Manual attack-stratified XAI:
   - attack_stratified
   - optional --attack-name filter
   - sequential manual review, one attack at a time
   - candidate_limit = 0 means show all candidates for the current attack
   - cases_per_attack defines how many cases should be selected, not how many
     candidates should be displayed

3. Manual Chapter 5 thesis-oriented XAI:
   - chapter5_core
   - candidates grouped by case_bucket
   - sequential manual review, one thesis category at a time
   - cases_per_bucket defines how many final cases should be selected per category

4. Generation from an existing manual selection manifest:
   - --selection-manifest explainability/manifests/xai_manual_selection_db__*.csv

Methodological note
-------------------
Integrated Gradients are used as qualitative diagnostic support for transparent
proxy models. They are not generated for commercial black-box forensic tools.
Manual case selection is human-in-the-loop, attack-aware, logged, and exported
as a reproducible manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.attacks.adversarial_model_interface import TargetModelConfig, label_to_index
from datasets.scripts.attacks.adversarial_torch_model_adapters import build_target_model_adapter
from datasets.scripts.utils.paths import EVALUATION_DIR, EXPLAINABILITY_DIR, REPO_ROOT, repo_relative_path


# =============================================================================
# Constants and paths
# =============================================================================

SCRIPT_NAME = "explainability/scripts/17_generate_integrated_gradients_case_studies.py"

SUPPORTED_MODELS = ("resnet18", "efficientnet_b0", "clip")
VALID_LABELS = ("non_weapon", "weapon")

DEFAULT_ATTACKS = [
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

DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"
DEFAULT_PREDICTIONS_CSV = EVALUATION_DIR / "proxy_models" / "proxy_model_predictions.csv"

IG_OUTPUT_ROOT = EXPLAINABILITY_DIR / "outputs" / "integrated_gradients"
MANIFEST_DIR = EXPLAINABILITY_DIR / "manifests"
LOG_DIR = EXPLAINABILITY_DIR / "logs"

BATCH_SIZE = 12
N_COLS = 4
FIG_W = 18
FIG_H = 10

CHAPTER5_BUCKET_GUIDANCE = {
    "clean_correct_weapon": {
        "title": "Caso 1 - Clean corretto: weapon -> weapon",
        "select": "Seleziona 1 immagine weapon pulita, leggibile, correttamente classificata come weapon.",
        "purpose": "Serve come esempio di comportamento atteso del modello.",
        "avoid": "Evita immagini troppo scure, ritagliate male, ambigue o poco comprensibili nel testo della tesi.",
    },
    "clean_false_negative_weapon": {
        "title": "Caso 2 - Falso negativo clean: weapon -> non_weapon",
        "select": "Seleziona 1 immagine weapon pulita, forensicamente rilevante, sbagliata dal modello come non_weapon.",
        "purpose": "Serve a mostrare il rischio operativo del falso negativo su contenuto rilevante.",
        "avoid": "Evita casi in cui anche per un lettore umano l'arma sia quasi invisibile o troppo ambigua.",
    },
    "ood_as_weapon": {
        "title": "Caso 3 - OOD classificato come weapon",
        "select": "Seleziona 1 immagine OOD chiaramente fuori distribuzione ma classificata come weapon.",
        "purpose": "Serve a mostrare il rischio di falso positivo su immagini fuori distribuzione.",
        "avoid": "Evita immagini OOD che sembrano quasi vere armi, se non sono utili a spiegare il caso borderline.",
    },
    "anti_forensic_failure": {
        "title": "Caso 4 - Errore dopo trasformazione anti-forensic",
        "select": "Seleziona 1 immagine manipolata anti-forensic che resti visivamente leggibile ma produca errore.",
        "purpose": "Serve a mostrare che trasformazioni realistiche possono alterare la classificazione operativa.",
        "avoid": "Evita immagini eccessivamente degradate: devono essere ancora interpretabili per il lettore.",
    },
    "adversarial_high_conf_failure": {
        "title": "Caso 5 - Errore adversarial ad alta confidenza",
        "select": "Seleziona 1 immagine adversarial con errore chiaro e confidenza alta, preferibilmente weapon -> non_weapon.",
        "purpose": "Serve a mostrare la fragilita' del modello sotto attacco adversarial.",
        "avoid": "Evita casi visivamente incomprensibili o dove la perturbazione distrugge troppo il contenuto.",
    },
}


def chapter5_bucket_guidance(bucket: str) -> dict[str, str]:
    return CHAPTER5_BUCKET_GUIDANCE.get(
        norm(bucket),
        {
            "title": f"Categoria XAI: {bucket}",
            "select": "Seleziona il caso piu' leggibile e piu' utile per la narrazione del Capitolo 5.",
            "purpose": "La scelta finale resta human-in-the-loop.",
            "avoid": "Evita immagini poco leggibili o non informative.",
        },
    )


def format_chapter5_bucket_guidance(bucket: str, requested_cases: int, candidates_count: int) -> str:
    guide = chapter5_bucket_guidance(bucket)
    return "\n".join(
        [
            "=== CHAPTER 5 XAI SELECTION GUIDE ===",
            guide["title"],
            "",
            f"Cosa devi selezionare : {guide['select']}",
            f"Quante selezionarne   : {requested_cases}",
            f"Candidati mostrati    : {candidates_count}",
            f"Perche' serve         : {guide['purpose']}",
            f"Da evitare            : {guide['avoid']}",
            "",
            "Regola pratica: scegli l'immagine piu' chiara per il lettore, non quella esteticamente piu' bella.",
            "Dopo aver selezionato il numero richiesto di casi, premi q per salvare e passare alla categoria successiva.",
        ]
    )


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
    value = str(value).strip().lower()
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


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    tmp_path = path.with_suffix(".tmp.csv")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def append_log_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "run_tag",
        "reviewer_id",
        "action",
        "candidate_id",
        "attack_name",
        "selected",
        "selection_rank",
        "image_relative_path",
        "notes",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def require_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt_module
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("Missing XAI dependencies. Install captum, torch, numpy and matplotlib.") from exc

    return torch, np, plt_module, IntegratedGradients


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
    "chapter5_core",
    "all",
),
        default="all",
        help="Case-selection strategy.",
    )

    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--cases-per-attack", type=int, default=3)
    parser.add_argument(
        "--cases-per-bucket",
        type=int,
        default=1,
        help="Number of final cases to select for each Chapter 5 XAI bucket in manual review mode.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Maximum candidates shown per attack in manual attack-stratified mode. Use 0 to show all.",
    )
    parser.add_argument("--attack-name", nargs="+", default=[], help="Optional attack_name filter.")
    parser.add_argument("--selection-manifest", default="")
    parser.add_argument("--manual-review", action="store_true")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--generate-after-manual", action="store_true")
    parser.add_argument("--reviewer-id", default="Lello")

    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=224)

    parser.add_argument(
        "--attribution-target",
        choices=("true_label", "predicted_label", "both"),
        default="true_label",
    )
    parser.add_argument("--top-percentile", type=float, default=90.0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def build_run_tag(args: argparse.Namespace) -> str:
    model_tag = "_".join(args.model)

    if args.selection_manifest:
        manifest_tag = sanitize_tag(Path(args.selection_manifest).stem)
        parts = [model_tag, "from_manual_manifest", manifest_tag]
    else:
        parts = [model_tag, args.strategy]
        if args.strategy == "attack_stratified":
            parts.append(f"per_attack_{args.cases_per_attack}")
            if args.attack_name:
                parts.append(args.attack_name[0] if len(args.attack_name) == 1 else "multi_attack")
            if args.manual_review:
                parts.append("manual")
        elif args.strategy == "chapter5_core":
            parts.append(f"candidates_per_bucket_{args.max_cases}")
            if args.manual_review:
                parts.append(f"manual_per_bucket_{args.cases_per_bucket}")
        elif args.strategy in {"ood_high_confidence", "all"}:
            parts.append(f"thr_{args.high_confidence_threshold:.2f}")
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
        "manual_selection_log_csv": LOG_DIR / f"xai_manual_selection_log__{run_tag}.csv",
    }


def ensure_run_outputs_do_not_exist(run_paths: dict[str, Path], force: bool, check_manual_db: bool = False) -> None:
    paths = [run_paths["ig_manifest_csv"], run_paths["case_studies_manifest_csv"], run_paths["summary_json"]]
    if check_manual_db:
        paths += [run_paths["manual_selection_db_csv"], run_paths["manual_selection_summary_json"], run_paths["manual_selection_log_csv"]]
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "Output manifest(s) for this run already exist. Use --force to overwrite this run.\n"
            + "\n".join(f"- {p}" for p in existing)
        )


# =============================================================================
# Prediction loading and candidate selection
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
        raise RuntimeError(f"No rows found for models {models}. Available models: {available_models}")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()
    if df.empty:
        raise RuntimeError("All rows were removed after filtering the error column.")

    logging.info("Prediction rows after filtering: %d", len(df))
    return df


def build_unique_case_key(row: pd.Series) -> str:
    if norm(row.get("sample_type", "")) in {"ood", "out_of_distribution", "out-of-distribution"}:
        key = safe_str(row.get("image_relative_path", ""))
        if key:
            return key
    for col in ["generated_image_id", "original_image_id", "image_relative_path"]:
        key = safe_str(row.get(col, ""))
        if key:
            return key
    return safe_str(row.name)


def add_selection_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_type_norm"] = df["sample_type"].map(norm)
    df["attack_name_norm"] = df["attack_name"].map(norm)
    df["attack_family_norm"] = df["attack_family"].map(norm)
    df["final_label_norm"] = df["final_label"].map(norm)
    df["prediction_norm"] = df["prediction"].map(norm)
    df["confidence_numeric"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["correct_bool"] = bool_series(df["correct"]) if "correct" in df.columns else False
    df["clean_correct_bool"] = bool_series(df["clean_correct"]) if "clean_correct" in df.columns else False
    df["unique_case_key"] = df.apply(build_unique_case_key, axis=1)
    return df


def perturbed_mask(df: pd.DataFrame) -> pd.Series:
    return df["sample_type_norm"].isin({"perturbed", "adversarial", "anti_forensic", "anti-forensic", "transformed"})


def ood_mask(df: pd.DataFrame) -> pd.Series:
    return df["sample_type_norm"].isin({"ood", "out_of_distribution", "out-of-distribution"})


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank manual-review candidates according to forensic relevance.

    Priority order:
    1. weapon -> non_weapon failures, when the clean sample was correct;
    2. non_weapon -> weapon failures, when the clean sample was correct;
    3. any clean-correct -> perturbed-wrong failure;
    4. remaining high-confidence errors;
    5. remaining high-confidence cases.

    This makes the manual reviewer show first the cases that are most relevant
    for the thesis: operational failures induced by adversarial or anti-forensic
    transformations.
    """
    ranked = add_selection_helper_columns(df)

    ranked["is_clean_correct_failure"] = (
        ranked["clean_correct_bool"] & (~ranked["correct_bool"])
    )

    ranked["is_weapon_to_non_weapon"] = (
        ranked["is_clean_correct_failure"]
        & (ranked["final_label_norm"] == "weapon")
        & (ranked["prediction_norm"] == "non_weapon")
    )

    ranked["is_non_weapon_to_weapon"] = (
        ranked["is_clean_correct_failure"]
        & (ranked["final_label_norm"] == "non_weapon")
        & (ranked["prediction_norm"] == "weapon")
    )

    ranked["is_other_failure"] = (
        ranked["is_clean_correct_failure"]
        & (~ranked["is_weapon_to_non_weapon"])
        & (~ranked["is_non_weapon_to_weapon"])
    )

    ranked["is_high_confidence_error"] = (
        (~ranked["correct_bool"])
        & (ranked["confidence_numeric"] >= 0.90)
    )

    def priority_label(row: pd.Series) -> str:
        if row["is_weapon_to_non_weapon"]:
            return "P1_weapon_to_non_weapon"
        if row["is_non_weapon_to_weapon"]:
            return "P2_non_weapon_to_weapon"
        if row["is_other_failure"]:
            return "P3_other_clean_correct_failure"
        if row["is_high_confidence_error"]:
            return "P4_high_confidence_error"
        return "P5_other"

    ranked["xai_priority_label"] = ranked.apply(priority_label, axis=1)

    priority_order = {
        "P1_weapon_to_non_weapon": 1,
        "P2_non_weapon_to_weapon": 2,
        "P3_other_clean_correct_failure": 3,
        "P4_high_confidence_error": 4,
        "P5_other": 5,
    }

    ranked["xai_priority_rank"] = ranked["xai_priority_label"].map(priority_order).fillna(99)

    return ranked.sort_values(
        [
            "xai_priority_rank",
            "confidence_numeric",
            "evaluation_fold",
            "final_label",
            "prediction",
            "image_relative_path",
        ],
        ascending=[True, False, True, True, True, True],
        kind="stable",
    )

def select_chapter5_core_cases(
    df: pd.DataFrame,
    cases_per_bucket: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Select thesis-oriented XAI candidates for Chapter 5.

    Buckets:
    1. clean_correct_weapon
    2. clean_false_negative_weapon
    3. ood_as_weapon
    4. anti_forensic_failure
    5. adversarial_high_conf_failure

    The function returns up to cases_per_bucket candidates for each bucket.
    Final selection remains human-in-the-loop.
    """
    df = add_selection_helper_columns(df)

    clean = df["sample_type_norm"].eq("clean")
    perturbed = perturbed_mask(df)
    ood = ood_mask(df)

    anti_forensic_attacks = {
        "jpeg_recompression",
        "resample_resize",
        "gaussian_blur",
        "histogram_modification",
        "contrast_stretching",
    }

    adversarial_attacks = {
        "fgsm",
        "one_pixel",
        "sigma_zero",
        "superdeepfool",
        "color_shift",
    }

    def prepare_bucket(bucket_df: pd.DataFrame, bucket_name: str, reason: str) -> pd.DataFrame:
        bucket_df = bucket_df.copy()
        if bucket_df.empty:
            return bucket_df

        bucket_df["case_bucket"] = bucket_name
        bucket_df["xai_selection_reason"] = reason

        bucket_df = bucket_df.sort_values(
            [
                "confidence_numeric",
                "evaluation_fold",
                "image_relative_path",
            ],
            ascending=[False, True, True],
            kind="stable",
        )

        bucket_df = bucket_df.drop_duplicates(
            subset=["evaluated_model", "unique_case_key"],
            keep="first",
        )

        return bucket_df.head(cases_per_bucket)

    buckets: list[pd.DataFrame] = []

    # 1. Clean correct weapon -> weapon
    buckets.append(
        prepare_bucket(
            df[
                clean
                & (df["final_label_norm"] == "weapon")
                & (df["prediction_norm"] == "weapon")
                & df["correct_bool"]
            ],
            "clean_correct_weapon",
            "Clean weapon image correctly classified as weapon.",
        )
    )

    # 2. Clean false negative weapon -> non_weapon
    buckets.append(
        prepare_bucket(
            df[
                clean
                & (df["final_label_norm"] == "weapon")
                & (df["prediction_norm"] == "non_weapon")
                & (~df["correct_bool"])
            ],
            "clean_false_negative_weapon",
            "Clean weapon image incorrectly classified as non_weapon.",
        )
    )

    # 3. OOD classified as weapon
    buckets.append(
        prepare_bucket(
            df[
                ood
                & (df["prediction_norm"] == "weapon")
                & (df["confidence_numeric"] >= threshold)
            ],
            "ood_as_weapon",
            "Out-of-distribution image classified as weapon with high confidence.",
        )
    )

    # 4. Anti-forensic failure
    anti_forensic_df = df[
        perturbed
        & (
            df["attack_family_norm"].isin({"anti_forensic", "anti-forensic"})
            | df["attack_name_norm"].isin(anti_forensic_attacks)
        )
        & df["clean_correct_bool"]
        & (~df["correct_bool"])
    ].copy()

    anti_forensic_df = rank_candidates(anti_forensic_df)

    buckets.append(
        prepare_bucket(
            anti_forensic_df,
            "anti_forensic_failure",
            "Anti-forensic transformation caused an operational misclassification.",
        )
    )

    # 5. Adversarial high-confidence failure
    adversarial_df = df[
        perturbed
        & (
            df["attack_family_norm"].isin({"adversarial"})
            | df["attack_name_norm"].isin(adversarial_attacks)
        )
        & df["clean_correct_bool"]
        & (~df["correct_bool"])
        & (df["confidence_numeric"] >= threshold)
    ].copy()

    adversarial_df = rank_candidates(adversarial_df)

    buckets.append(
        prepare_bucket(
            adversarial_df,
            "adversarial_high_conf_failure",
            "Adversarial transformation caused a high-confidence failure.",
        )
    )

    selected = pd.concat([b for b in buckets if not b.empty], ignore_index=True)

    if selected.empty:
        return df.head(0)

    return selected.reset_index(drop=True)

def select_cases(
    df: pd.DataFrame,
    strategy: str,
    max_cases: int,
    threshold: float,
    candidate_limit: int = 0,
    attack_names: list[str] | None = None,
) -> pd.DataFrame:
    df = add_selection_helper_columns(df)

    if strategy == "chapter5_core":
        return select_chapter5_core_cases(
            df=df,
            cases_per_bucket=max_cases,
            threshold=threshold,
        )

    if strategy == "attack_stratified":
        selected = df[perturbed_mask(df)].copy()
        if attack_names:
            attack_set = {norm(x) for x in attack_names}
            selected = selected[selected["attack_name_norm"].isin(attack_set)].copy()
        selected = rank_candidates(selected)
        selected = selected.drop_duplicates(
            subset=["evaluated_model", "attack_name", "unique_case_key"],
            keep="first",
        )
        if candidate_limit and candidate_limit > 0:
            selected = selected.groupby(["evaluated_model", "attack_name"], group_keys=False).head(candidate_limit)
        return selected.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    if strategy in {"perturbed_failures", "all"}:
        parts.append(df[perturbed_mask(df) & df["clean_correct_bool"] & (~df["correct_bool"])].sort_values("confidence_numeric", ascending=False))
    if strategy in {"weapon_to_non_weapon", "all"}:
        parts.append(
            df[
                perturbed_mask(df)
                & (df["final_label_norm"] == "weapon")
                & (df["prediction_norm"] == "non_weapon")
                & df["clean_correct_bool"]
            ].sort_values("confidence_numeric", ascending=False)
        )
    if strategy in {"ood_high_confidence", "all"}:
        parts.append(df[ood_mask(df) & (df["confidence_numeric"] >= threshold)].sort_values("confidence_numeric", ascending=False))

    if not parts:
        return df.head(0)

    selected = pd.concat(parts, ignore_index=True)
    selected["unique_case_key"] = selected.apply(build_unique_case_key, axis=1)
    selected = selected.drop_duplicates(subset=["evaluated_model", "unique_case_key"], keep="first")
    return selected.head(max_cases).reset_index(drop=True)


def available_attacks_from_candidates(candidates: pd.DataFrame, requested_attack_names: list[str]) -> list[str]:
    available = [x for x in candidates["attack_name"].dropna().astype(str).unique().tolist() if x.strip()]
    available_norm_map = {norm(x): x for x in available}
    if requested_attack_names:
        result = []
        for requested in requested_attack_names:
            key = norm(requested)
            if key in available_norm_map:
                result.append(available_norm_map[key])
            else:
                logging.warning("Requested attack_name not found in candidates: %s", requested)
        return result
    ordered = [a for a in DEFAULT_ATTACKS if a in available]
    ordered += [a for a in sorted(available) if a not in ordered]
    return ordered


def available_case_buckets_from_candidates(candidates: pd.DataFrame) -> list[str]:
    """Return Chapter 5 XAI buckets in the narrative order used in the thesis."""
    if "case_bucket" not in candidates.columns:
        return []

    preferred_order = [
        "clean_correct_weapon",
        "clean_false_negative_weapon",
        "ood_as_weapon",
        "anti_forensic_failure",
        "adversarial_high_conf_failure",
    ]

    available = [
        x for x in candidates["case_bucket"].dropna().astype(str).unique().tolist()
        if x.strip()
    ]
    available_norm_map = {norm(x): x for x in available}

    ordered = [
        available_norm_map[norm(bucket)]
        for bucket in preferred_order
        if norm(bucket) in available_norm_map
    ]
    ordered += [x for x in sorted(available) if x not in ordered]
    return ordered


# =============================================================================
# Manual XAI reviewer
# =============================================================================

HELP_TEXT = """
FAIR-LAB XAI MANUAL CASE REVIEWER

MOUSE
- Left click   = select case
- Right click  = unselect case
- Middle click = zoom selected case

KEYS
- s = save
- q = save + close current group review
- u = undo last select/unselect action
- g = go to page
- t = summary
- h = help
- Enter = zoom selected image
- Right / Space = next batch
- Left / Backspace = previous batch
- 1..9 = select image 1..9
- 0 = select image 10

NOTES
- The requested number of cases is advisory and visible.
- Every action is logged.
- In multi-group workflows, closing one group continues with the next group.
""".strip()


class XAIManualCaseReviewer:
    """
    Interactive visual reviewer for XAI case selection.

    The same reviewer supports two grouping modes:
    - attack-stratified review: group_field="attack_name"
    - Chapter 5 thesis review: group_field="case_bucket"

    The final selection is stored in a CSV manifest and can be used directly for
    Integrated Gradients generation.
    """

    def __init__(
        self,
        candidates_df: pd.DataFrame,
        group_name: str,
        group_field: str,
        requested_cases: int,
        run_tag: str,
        reviewer_id: str,
        db_path: Path,
        summary_path: Path,
        log_path: Path,
        created_at: str,
        group_label: str = "group",
    ) -> None:
        self.candidates_df = candidates_df.copy().reset_index(drop=True)
        self.group_name = group_name
        self.group_field = group_field
        self.group_label = group_label
        self.requested_cases = requested_cases
        self.run_tag = run_tag
        self.reviewer_id = reviewer_id
        self.db_path = db_path
        self.summary_path = summary_path
        self.log_path = log_path
        self.created_at = created_at

        # Backward-compatible alias used only in some messages.
        self.attack_name = group_name if group_field == "attack_name" else ""

        self.fig = None
        self.axes = []
        self.status_ax = None
        self.help_fig = None
        self.summary_fig = None

        self.current_start = 0
        self.selected_pos = 0
        self.batch_indices: list[int] = []
        self.ax_to_local_index: dict[Any, int] = {}
        self.last_action_stack: list[dict[str, Any]] = []

        self.db_df = self.load_or_create_db()
        self.ensure_current_candidates_in_db()

    def total_pages(self) -> int:
        return max(1, math.ceil(len(self.candidates_df) / BATCH_SIZE))

    def current_page(self) -> int:
        return (self.current_start // BATCH_SIZE) + 1

    def candidate_id_from_row(self, row: pd.Series) -> str:
        model = sanitize_tag(safe_str(row.get("evaluated_model", "")))
        fold = sanitize_tag(safe_str(row.get("evaluation_fold", "")))
        bucket = sanitize_tag(safe_str(row.get("case_bucket", "")))
        attack = sanitize_tag(safe_str(row.get("attack_name", "")))
        generated = sanitize_tag(safe_str(row.get("generated_image_id", "")))
        original = sanitize_tag(safe_str(row.get("original_image_id", "")))
        sample = sanitize_tag(safe_str(row.get("sample_id", "")))
        path_stem = sanitize_tag(Path(safe_str(row.get("image_relative_path", "unknown"))).stem)
        identifier = generated or original or sample or path_stem
        parts = [model, fold]
        if bucket:
            parts.append(bucket)
        if attack:
            parts.append(attack)
        parts.append(identifier)
        return "__".join([p for p in parts if p])

    def load_or_create_db(self) -> pd.DataFrame:
        if self.db_path.exists():
            df = pd.read_csv(self.db_path, low_memory=False)
        else:
            df = pd.DataFrame()

        required_columns = [
            "run_tag",
            "created_at",
            "candidate_id",
            "manual_selected",
            "selection_rank",
            "reviewer_id",
            "review_timestamp",
            "selection_action",
            "selection_notes",
            "group_field",
            "group_name",
            "requested_cases_for_group",
            "requested_cases_for_attack",
            "case_bucket",
            "xai_selection_reason",
            "xai_priority_label",
            "evaluated_model",
            "evaluation_fold",
            "sample_type",
            "attack_family",
            "attack_name",
            "final_label",
            "prediction",
            "confidence",
            "correct",
            "clean_correct",
            "original_image_id",
            "generated_image_id",
            "sample_id",
            "image_relative_path",
            "clean_relative_path",
            "perturbed_relative_path",
            "sha256_original",
            "sha256_perturbed",
            "md5_original",
            "md5_perturbed",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        df = df[required_columns].copy()
        df = df.astype("object")
        df = df.fillna("")
        return df

    def row_to_db_row(self, row: pd.Series) -> dict[str, Any]:
        return {
            "run_tag": self.run_tag,
            "created_at": self.created_at,
            "candidate_id": self.candidate_id_from_row(row),
            "manual_selected": "false",
            "selection_rank": "",
            "reviewer_id": "",
            "review_timestamp": "",
            "selection_action": "",
            "selection_notes": "",
            "group_field": self.group_field,
            "group_name": self.group_name,
            "requested_cases_for_group": str(self.requested_cases),
            "requested_cases_for_attack": str(self.requested_cases) if self.group_field == "attack_name" else "",
            "case_bucket": safe_str(row.get("case_bucket", "")),
            "xai_selection_reason": safe_str(row.get("xai_selection_reason", "")),
            "xai_priority_label": safe_str(row.get("xai_priority_label", "")),
            "evaluated_model": safe_str(row.get("evaluated_model", "")),
            "evaluation_fold": safe_str(row.get("evaluation_fold", "")),
            "sample_type": safe_str(row.get("sample_type", "")),
            "attack_family": safe_str(row.get("attack_family", "")),
            "attack_name": safe_str(row.get("attack_name", "")),
            "final_label": safe_str(row.get("final_label", "")),
            "prediction": safe_str(row.get("prediction", "")),
            "confidence": safe_str(row.get("confidence", "")),
            "correct": safe_str(row.get("correct", "")),
            "clean_correct": safe_str(row.get("clean_correct", "")),
            "original_image_id": safe_str(row.get("original_image_id", "")),
            "generated_image_id": safe_str(row.get("generated_image_id", "")),
            "sample_id": safe_str(row.get("sample_id", "")) or Path(safe_str(row.get("image_relative_path", ""))).stem,
            "image_relative_path": safe_str(row.get("image_relative_path", "")),
            "clean_relative_path": safe_str(row.get("clean_relative_path", "")),
            "perturbed_relative_path": safe_str(row.get("perturbed_relative_path", "")),
            "sha256_original": safe_str(row.get("sha256_original", "")),
            "sha256_perturbed": safe_str(row.get("sha256_perturbed", "")),
            "md5_original": safe_str(row.get("md5_original", "")),
            "md5_perturbed": safe_str(row.get("md5_perturbed", "")),
        }

    def ensure_current_candidates_in_db(self) -> None:
        existing = set(self.db_df["candidate_id"].astype(str).tolist())
        rows = []
        for _, row in self.candidates_df.iterrows():
            candidate_id = self.candidate_id_from_row(row)
            if candidate_id not in existing:
                rows.append(self.row_to_db_row(row))
        if rows:
            self.db_df = pd.concat([self.db_df, pd.DataFrame(rows)], ignore_index=True)
            self.save_db_and_summary("initialize_candidates")

    def db_row_index(self, candidate_id: str) -> int | None:
        matches = self.db_df.index[self.db_df["candidate_id"].astype(str) == candidate_id].tolist()
        return int(matches[0]) if matches else None

    def is_selected(self, candidate_id: str) -> bool:
        idx = self.db_row_index(candidate_id)
        if idx is None:
            return False
        return norm(self.db_df.loc[idx, "manual_selected"]) in {"true", "1", "yes"}

    def selected_count_for_group(self) -> int:
        if self.group_field not in self.db_df.columns:
            return 0
        tmp = self.db_df[
            (self.db_df[self.group_field].map(norm) == norm(self.group_name))
            & (self.db_df["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"}))
        ]
        return int(len(tmp))

    def selected_count_for_attack(self) -> int:
        # Kept for backward compatibility with the previous attack-oriented reviewer.
        return self.selected_count_for_group()

    def next_selection_rank(self) -> int:
        ranks = pd.to_numeric(self.db_df["selection_rank"], errors="coerce").dropna()
        return 1 if ranks.empty else int(ranks.max()) + 1

    def log_action(
        self,
        candidate_id: str,
        action: str,
        selected: bool,
        selection_rank: Any = "",
        notes: str = "",
    ) -> None:
        row = self.db_df[self.db_df["candidate_id"].astype(str) == candidate_id]
        image_relative_path = safe_str(row.iloc[0].get("image_relative_path", "")) if not row.empty else ""
        append_log_csv(
            self.log_path,
            {
                "timestamp": utc_now_iso(),
                "run_tag": self.run_tag,
                "reviewer_id": self.reviewer_id,
                "action": action,
                "candidate_id": candidate_id,
                "group_field": self.group_field,
                "group_name": self.group_name,
                "attack_name": safe_str(row.iloc[0].get("attack_name", "")) if not row.empty else "",
                "case_bucket": safe_str(row.iloc[0].get("case_bucket", "")) if not row.empty else "",
                "selected": selected,
                "selection_rank": selection_rank,
                "image_relative_path": image_relative_path,
                "notes": notes,
            },
        )

    def save_db_and_summary(self, action: str = "") -> None:
        write_dataframe_csv(self.db_path, self.db_df)

        group_summary = []
        if self.group_field in self.db_df.columns:
            for group_name, group in self.db_df.groupby(self.group_field, dropna=False):
                selected = group["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"})
                requested = pd.to_numeric(group["requested_cases_for_group"], errors="coerce").fillna(0)
                group_summary.append(
                    {
                        "group_field": self.group_field,
                        "group_name": group_name,
                        "candidate_rows": int(len(group)),
                        "selected_cases": int(selected.sum()),
                        "requested_cases": int(requested.max()) if not requested.empty else 0,
                    }
                )

        attack_summary = []
        if "attack_name" in self.db_df.columns:
            for attack, group in self.db_df.groupby("attack_name", dropna=False):
                selected = group["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"})
                attack_summary.append(
                    {
                        "attack_name": attack,
                        "candidate_rows": int(len(group)),
                        "selected_cases": int(selected.sum()),
                    }
                )

        bucket_summary = []
        if "case_bucket" in self.db_df.columns:
            for bucket, group in self.db_df.groupby("case_bucket", dropna=False):
                selected = group["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"})
                bucket_summary.append(
                    {
                        "case_bucket": bucket,
                        "candidate_rows": int(len(group)),
                        "selected_cases": int(selected.sum()),
                    }
                )

        write_json(
            self.summary_path,
            {
                "script": SCRIPT_NAME,
                "run_tag": self.run_tag,
                "updated_at": utc_now_iso(),
                "reviewer_id": self.reviewer_id,
                "last_action": action,
                "manual_selection_db": repo_relative_string(self.db_path),
                "manual_selection_log": repo_relative_string(self.log_path),
                "total_candidate_rows": int(len(self.db_df)),
                "total_selected_cases": int(self.db_df["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()),
                "active_group": {
                    "group_label": self.group_label,
                    "group_field": self.group_field,
                    "group_name": self.group_name,
                    "selected_cases": self.selected_count_for_group(),
                    "requested_cases": self.requested_cases,
                },
                "group_summary": group_summary,
                "attack_summary": attack_summary,
                "case_bucket_summary": bucket_summary,
            },
        )

    def select_case(self, local_index: int) -> None:
        row = self.candidates_df.iloc[local_index]
        candidate_id = self.candidate_id_from_row(row)
        idx = self.db_row_index(candidate_id)
        if idx is None or self.is_selected(candidate_id):
            return

        previous_rank = safe_str(self.db_df.loc[idx, "selection_rank"])
        self.last_action_stack.append(
            {"candidate_id": candidate_id, "previous_selected": False, "previous_rank": previous_rank}
        )

        rank = self.next_selection_rank()
        self.db_df.at[idx, "manual_selected"] = "true"
        self.db_df.at[idx, "selection_rank"] = str(rank)
        self.db_df.at[idx, "reviewer_id"] = self.reviewer_id
        self.db_df.at[idx, "review_timestamp"] = utc_now_iso()
        self.db_df.at[idx, "selection_action"] = "select"

        self.log_action(candidate_id, "select", True, rank)
        self.save_db_and_summary("select")
        self.draw_batch(preserve_candidate_id=candidate_id)

        if self.selected_count_for_group() >= self.requested_cases:
            print(
                f"[INFO] Target reached for {self.group_label} {self.group_name}: "
                f"{self.selected_count_for_group()}/{self.requested_cases}. "
                "Press q to save and continue."
            )

    def unselect_case(self, local_index: int) -> None:
        row = self.candidates_df.iloc[local_index]
        candidate_id = self.candidate_id_from_row(row)
        idx = self.db_row_index(candidate_id)
        if idx is None or not self.is_selected(candidate_id):
            return

        previous_rank = safe_str(self.db_df.loc[idx, "selection_rank"])
        self.last_action_stack.append(
            {"candidate_id": candidate_id, "previous_selected": True, "previous_rank": previous_rank}
        )

        self.db_df.at[idx, "manual_selected"] = "false"
        self.db_df.at[idx, "selection_rank"] = ""
        self.db_df.at[idx, "reviewer_id"] = self.reviewer_id
        self.db_df.at[idx, "review_timestamp"] = utc_now_iso()
        self.db_df.at[idx, "selection_action"] = "unselect"

        self.log_action(candidate_id, "unselect", False, "")
        self.save_db_and_summary("unselect")
        self.draw_batch(preserve_candidate_id=candidate_id)

    def undo_last_action(self) -> None:
        if not self.last_action_stack:
            print("[INFO] No recent action to undo.")
            return

        item = self.last_action_stack.pop()
        candidate_id = item["candidate_id"]
        idx = self.db_row_index(candidate_id)
        if idx is None:
            return

        selected = "true" if bool(item["previous_selected"]) else "false"
        previous_rank = safe_str(item["previous_rank"])
        self.db_df.at[idx, "manual_selected"] = selected
        self.db_df.at[idx, "selection_rank"] = previous_rank
        self.db_df.at[idx, "reviewer_id"] = self.reviewer_id
        self.db_df.at[idx, "review_timestamp"] = utc_now_iso()
        self.db_df.at[idx, "selection_action"] = "undo"

        self.log_action(candidate_id, "undo", selected, previous_rank)
        self.save_db_and_summary("undo")
        self.draw_batch(preserve_candidate_id=candidate_id)

    def get_local_indices(self) -> list[int]:
        return list(range(len(self.candidates_df)))

    def update_batch_indices(self, preserve_candidate_id: str | None = None) -> None:
        indices = self.get_local_indices()

        if not indices:
            self.batch_indices = []
            self.selected_pos = 0
            self.current_start = 0
            return

        if preserve_candidate_id:
            for pos, local_index in enumerate(indices):
                if self.candidate_id_from_row(self.candidates_df.iloc[local_index]) == preserve_candidate_id:
                    self.current_start = (pos // BATCH_SIZE) * BATCH_SIZE
                    self.batch_indices = indices[self.current_start:self.current_start + BATCH_SIZE]
                    self.selected_pos = pos % BATCH_SIZE
                    return

        if self.current_start >= len(indices):
            self.current_start = max(0, ((len(indices) - 1) // BATCH_SIZE) * BATCH_SIZE)

        self.batch_indices = indices[self.current_start:self.current_start + BATCH_SIZE]

        if self.selected_pos >= len(self.batch_indices):
            self.selected_pos = 0

    def build_status_text(self) -> str:
        lines = [
            "=== FAIR-LAB XAI MANUAL CASE REVIEWER ===",
            f"run_tag        : {self.run_tag}",
            f"{self.group_label:<15}: {self.group_name}",
            f"group_field    : {self.group_field}",
            f"selected       : {self.selected_count_for_group()}/{self.requested_cases}",
            f"candidates     : {len(self.candidates_df)}",
            f"page           : {self.current_page()}/{self.total_pages()}",
        ]

        if self.group_field == "case_bucket":
            guide = chapter5_bucket_guidance(self.group_name)
            lines.extend(
                [
                    "",
                    f"TO SELECT      : {guide['select']}",
                    f"HOW MANY       : {self.requested_cases}",
                    f"PURPOSE        : {guide['purpose']}",
                    f"AVOID          : {guide['avoid']}",
                ]
            )

        lines.extend(
            [
                "",
                "Mouse: left=select | right=unselect | middle=zoom",
                "Keys : s=save | q=save+close current group | u=undo | g=page | t=summary | h=help",
            ]
        )
        return "\n".join(lines)

    def init_figure_if_needed(self) -> None:
        if self.fig is not None:
            return

        rows = math.ceil(BATCH_SIZE / N_COLS)
        self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = self.fig.add_gridspec(nrows=rows + 1, ncols=N_COLS, height_ratios=[1] * rows + [1.1])
        self.axes = [self.fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(N_COLS)]
        self.status_ax = self.fig.add_subplot(gs[rows, :])
        self.status_ax.axis("off")
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.05, wspace=0.08, hspace=0.35)
        self.fig.canvas.manager.set_window_title("FAIRLab XAI Manual Case Reviewer")
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
        except Exception:
            pass

    def draw_status_panel(self) -> None:
        if self.status_ax is None:
            return
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

    def draw_batch(self, preserve_candidate_id: str | None = None) -> None:
        self.update_batch_indices(preserve_candidate_id=preserve_candidate_id)
        self.init_figure_if_needed()

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        if not self.batch_indices:
            self.fig.suptitle(
                f"FAIRLab XAI Reviewer | {self.group_label}={self.group_name} | NO CANDIDATES",
                fontsize=12,
            )
            self.draw_status_panel()
            self.fig.canvas.draw_idle()
            return

        self.ax_to_local_index.clear()

        for i, local_index in enumerate(self.batch_indices):
            ax = self.axes[i]
            row = self.candidates_df.iloc[local_index]
            candidate_id = self.candidate_id_from_row(row)
            selected = self.is_selected(candidate_id)
            image_path = resolve_repo_path(safe_str(row.get("image_relative_path", "")))

            if image_path.exists():
                try:
                    ax.imshow(mpimg.imread(image_path))
                except Exception:
                    ax.text(0.5, 0.5, "ERR IMG", ha="center", va="center", fontsize=10)
            else:
                ax.text(0.5, 0.5, "IMG NOT FOUND", ha="center", va="center", fontsize=10)

            priority_label = safe_str(row.get("xai_priority_label", ""))
            priority_short = priority_label.replace("P1_", "P1 ").replace("P2_", "P2 ").replace(
                "P3_", "P3 "
            ).replace("P4_", "P4 ").replace("P5_", "P5 ")

            bucket = safe_str(row.get("case_bucket", ""))
            reason = safe_str(row.get("xai_selection_reason", ""))
            attack = safe_str(row.get("attack_name", "")) or "none"

            title = (
                f"{i + 1}. {candidate_id[:42]}\n"
                f"bucket={bucket or '-'} | attack={attack}\n"
                f"{priority_short}\n"
                f"fold={safe_str(row.get('evaluation_fold', ''))} | "
                f"label={safe_str(row.get('final_label', ''))} -> pred={safe_str(row.get('prediction', ''))}\n"
                f"conf={safe_str(row.get('confidence', ''))} | selected={selected}"
            )
            if reason:
                title += f"\n{reason[:70]}"

            ax.set_title(title, fontsize=8, color="green" if selected else "black")

            if selected:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("green")

            if i == self.selected_pos:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor("red")

            self.ax_to_local_index[ax] = local_index

        self.fig.suptitle(
            f"FAIRLab XAI Reviewer | {self.group_label}={self.group_name} | "
            f"selected={self.selected_count_for_group()}/{self.requested_cases} | "
            f"page={self.current_page()}/{self.total_pages()}",
            fontsize=11,
        )
        self.draw_status_panel()
        self.fig.canvas.draw_idle()

    def next_batch(self) -> None:
        indices = self.get_local_indices()

        if not indices:
            print("[INFO] No candidates available.")
            return

        max_start = max(0, ((len(indices) - 1) // BATCH_SIZE) * BATCH_SIZE)

        if self.current_start >= max_start:
            print(
                f"[INFO] Already at last page "
                f"({self.current_page()}/{self.total_pages()}) "
                f"for {self.group_label} {self.group_name}."
            )
            return

        self.current_start = min(self.current_start + BATCH_SIZE, max_start)
        self.selected_pos = 0
        self.draw_batch()

    def prev_batch(self) -> None:
        indices = self.get_local_indices()

        if not indices:
            print("[INFO] No candidates available.")
            return

        if self.current_start <= 0:
            print(
                f"[INFO] Already at first page "
                f"({self.current_page()}/{self.total_pages()}) "
                f"for {self.group_label} {self.group_name}."
            )
            return

        self.current_start = max(0, self.current_start - BATCH_SIZE)
        self.selected_pos = 0
        self.draw_batch()

    def go_to_page(self) -> None:
        total_pages = max(1, math.ceil(len(self.candidates_df) / BATCH_SIZE))
        current_page = (self.current_start // BATCH_SIZE) + 1
        raw = input(f"Go to page [1-{total_pages}] (current: {current_page}): ").strip()

        if not raw or not raw.isdigit():
            return

        page = int(raw)
        if 1 <= page <= total_pages:
            self.current_start = (page - 1) * BATCH_SIZE
            self.selected_pos = 0
            self.draw_batch()

    def open_help_window(self) -> None:
        if self.help_fig is not None:
            try:
                plt.figure(self.help_fig.number)
                self.help_fig.canvas.draw_idle()
                return
            except Exception:
                self.help_fig = None

        self.help_fig, ax = plt.subplots(figsize=(8, 10))
        self.help_fig.canvas.manager.set_window_title("XAI Manual Reviewer Help")
        ax.axis("off")
        ax.text(0.02, 0.98, HELP_TEXT, va="top", ha="left", fontsize=11, family="monospace", wrap=True)
        self.help_fig.tight_layout()

    def open_summary_window(self) -> None:
        lines = [
            "FAIR-LAB XAI MANUAL SELECTION SUMMARY",
            "",
            f"run_tag      : {self.run_tag}",
            f"{self.group_label:<13}: {self.group_name}",
            f"group_field  : {self.group_field}",
            f"selected     : {self.selected_count_for_group()} / {self.requested_cases}",
            f"candidates   : {len(self.candidates_df)}",
            "",
            "SELECTED CASES",
            "",
        ]

        selected_df = self.db_df[
            (self.db_df[self.group_field].map(norm) == norm(self.group_name))
            & (self.db_df["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"}))
        ].copy()

        selected_df["rank_num"] = pd.to_numeric(selected_df["selection_rank"], errors="coerce")
        selected_df = selected_df.sort_values("rank_num", kind="stable")

        for _, row in selected_df.iterrows():
            lines.append(
                f"{safe_str(row.get('selection_rank', ''))}. {safe_str(row.get('candidate_id', ''))} | "
                f"{safe_str(row.get('final_label', ''))}->{safe_str(row.get('prediction', ''))} | "
                f"conf={safe_str(row.get('confidence', ''))}"
            )

        if self.summary_fig is None:
            self.summary_fig, ax = plt.subplots(figsize=(10, 11))
            self.summary_fig.canvas.manager.set_window_title("XAI Manual Selection Summary")
        else:
            ax = self.summary_fig.axes[0]

        ax.clear()
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        self.summary_fig.tight_layout()
        self.summary_fig.canvas.draw_idle()

    def open_zoom(self) -> None:
        if not self.batch_indices or self.selected_pos >= len(self.batch_indices):
            return

        row = self.candidates_df.iloc[self.batch_indices[self.selected_pos]]
        image_path = resolve_repo_path(safe_str(row.get("image_relative_path", "")))

        if not image_path.exists():
            return

        try:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mpimg.imread(image_path))
            ax.set_title(
                f"{self.group_label}={self.group_name} | {safe_str(row.get('evaluation_fold', ''))}\n"
                f"bucket={safe_str(row.get('case_bucket', ''))} | attack={safe_str(row.get('attack_name', ''))}\n"
                f"label={safe_str(row.get('final_label', ''))} | "
                f"prediction={safe_str(row.get('prediction', ''))} | "
                f"confidence={safe_str(row.get('confidence', ''))}\n"
                f"{safe_str(row.get('image_relative_path', ''))}"
            )
            ax.axis("off")
            fig.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"[WARN] Zoom failed: {exc}")

    def on_click(self, event) -> None:
        if event.inaxes not in self.ax_to_local_index:
            return

        local_index = self.ax_to_local_index[event.inaxes]

        if local_index in self.batch_indices:
            self.selected_pos = self.batch_indices.index(local_index)

        if event.button == MouseButton.LEFT or event.button == 1:
            self.select_case(local_index)
        elif event.button == MouseButton.RIGHT or event.button == 3:
            self.unselect_case(local_index)
        elif event.button == MouseButton.MIDDLE or event.button == 2:
            self.open_zoom()

    def on_key(self, event) -> None:
        key = str(event.key).lower() if event.key is not None else ""

        if key in ["right", " ", "space", "pagedown", "n"]:
            self.next_batch()
        elif key in ["left", "backspace", "pageup", "p"]:
            self.prev_batch()
        elif key == "enter":
            self.open_zoom()
        elif key == "s":
            self.save_db_and_summary("manual_save")
            print("[OK] Saved.")
        elif key == "q":
            self.save_db_and_summary("close_group_review")
            print(f"[OK] Saved review for {self.group_label}: {self.group_name}")
            if self.fig is not None:
                plt.close(self.fig)
        elif key == "h":
            self.open_help_window()
        elif key == "g":
            self.go_to_page()
        elif key == "t":
            self.open_summary_window()
        elif key == "u":
            self.undo_last_action()
        elif key.isdigit():
            pos = 9 if key == "0" else int(key) - 1
            if 0 <= pos < len(self.batch_indices):
                self.selected_pos = pos
                self.draw_batch()

    def run(self) -> None:
        self.draw_batch()
        self.open_help_window()
        plt.show(block=True)
        self.save_db_and_summary("reviewer_closed")


def run_manual_review(
    candidates: pd.DataFrame,
    attacks_to_review: list[str],
    cases_per_attack: int,
    run_tag: str,
    reviewer_id: str,
    run_paths: dict[str, Path],
    created_at: str,
) -> pd.DataFrame:
    if candidates.empty:
        raise RuntimeError("No manual review candidates available.")

    for attack in attacks_to_review:
        attack_df = candidates[candidates["attack_name"].map(norm) == norm(attack)].copy()
        attack_df = rank_candidates(attack_df).reset_index(drop=True)

        if attack_df.empty:
            logging.warning("No candidates available for attack: %s", attack)
            continue

        print("\n" + "=" * 80)
        print(f"Manual XAI review for attack: {attack}")
        print(f"Candidates shown: {len(attack_df)}")
        print(f"Requested selected cases: {cases_per_attack}")
        print("=" * 80)

        reviewer = XAIManualCaseReviewer(
            candidates_df=attack_df,
            group_name=attack,
            group_field="attack_name",
            group_label="attack_name",
            requested_cases=cases_per_attack,
            run_tag=run_tag,
            reviewer_id=reviewer_id,
            db_path=run_paths["manual_selection_db_csv"],
            summary_path=run_paths["manual_selection_summary_json"],
            log_path=run_paths["manual_selection_log_csv"],
            created_at=created_at,
        )
        reviewer.run()

        print(
            f"[INFO] Completed attack {attack}: "
            f"selected={reviewer.selected_count_for_group()}/{cases_per_attack}"
        )

    if not run_paths["manual_selection_db_csv"].exists():
        raise RuntimeError("Manual selection DB was not created.")

    return pd.read_csv(run_paths["manual_selection_db_csv"], low_memory=False)


def run_manual_review_for_chapter5(
    candidates: pd.DataFrame,
    cases_per_bucket: int,
    run_tag: str,
    reviewer_id: str,
    run_paths: dict[str, Path],
    created_at: str,
) -> pd.DataFrame:
    """
    Review Chapter 5 XAI candidates one thesis category at a time.

    Expected buckets:
    - clean_correct_weapon
    - clean_false_negative_weapon
    - ood_as_weapon
    - anti_forensic_failure
    - adversarial_high_conf_failure
    """
    if candidates.empty:
        raise RuntimeError("No Chapter 5 manual review candidates available.")

    if "case_bucket" not in candidates.columns:
        raise RuntimeError("chapter5_core manual review requires a 'case_bucket' column.")

    buckets_to_review = available_case_buckets_from_candidates(candidates)
    if not buckets_to_review:
        raise RuntimeError("No case_bucket values available for Chapter 5 manual review.")

    for bucket in buckets_to_review:
        bucket_df = candidates[candidates["case_bucket"].map(norm) == norm(bucket)].copy()
        bucket_df = bucket_df.sort_values(
            ["confidence_numeric", "evaluation_fold", "image_relative_path"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)

        if bucket_df.empty:
            logging.warning("No candidates available for case_bucket: %s", bucket)
            continue

        print("\n" + "=" * 80)
        print(format_chapter5_bucket_guidance(bucket, cases_per_bucket, len(bucket_df)))
        print("=" * 80)

        reviewer = XAIManualCaseReviewer(
            candidates_df=bucket_df,
            group_name=bucket,
            group_field="case_bucket",
            group_label="case_bucket",
            requested_cases=cases_per_bucket,
            run_tag=run_tag,
            reviewer_id=reviewer_id,
            db_path=run_paths["manual_selection_db_csv"],
            summary_path=run_paths["manual_selection_summary_json"],
            log_path=run_paths["manual_selection_log_csv"],
            created_at=created_at,
        )
        reviewer.run()

        print(
            f"[INFO] Completed Chapter 5 bucket {bucket}: "
            f"selected={reviewer.selected_count_for_group()}/{cases_per_bucket}"
        )

    if not run_paths["manual_selection_db_csv"].exists():
        raise RuntimeError("Manual selection DB was not created.")

    return pd.read_csv(run_paths["manual_selection_db_csv"], low_memory=False)


def load_selected_rows_from_manual_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manual selection manifest not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "manual_selected" not in df.columns:
        raise ValueError("Manual selection manifest must contain 'manual_selected' column.")
    selected = df[df["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    if selected.empty:
        raise RuntimeError("No selected rows found in manual selection manifest.")
    if "image_relative_path" not in selected.columns and "input_relative_path" in selected.columns:
        selected["image_relative_path"] = selected["input_relative_path"]
    return selected.reset_index(drop=True)


# =============================================================================
# Model adapters and Integrated Gradients generation
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
        config = TargetModelConfig(name=model_name, checkpoint_path=checkpoint_path, device=self.device, input_size=self.input_size)
        adapter = build_target_model_adapter(config)
        adapter.load_model()
        self.cache[key] = adapter
        logging.info("Loaded %s %s", model_name, fold)
        return adapter


def callable_for_captum(adapter: Any) -> Any:
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


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def build_attribution_targets(row: pd.Series, target_mode: str) -> list[dict[str, Any]]:
    final_label = norm(row.get("final_label", ""))
    prediction = norm(row.get("prediction", ""))
    true_target = None
    predicted_target = None
    if final_label in VALID_LABELS:
        true_target = {"target_role": "true_label", "target_index": label_to_index(final_label), "target_label": final_label}
    if prediction in VALID_LABELS:
        predicted_target = {"target_role": "predicted_label", "target_index": label_to_index(prediction), "target_label": prediction}
    targets: list[dict[str, Any]] = []
    if target_mode == "true_label":
        if true_target is not None:
            targets.append(true_target)
        elif predicted_target is not None:
            targets.append({**predicted_target, "target_role": "predicted_label_fallback_for_ood"})
    elif target_mode == "predicted_label":
        if predicted_target is not None:
            targets.append(predicted_target)
        elif true_target is not None:
            targets.append({**true_target, "target_role": "true_label_fallback"})
    elif target_mode == "both":
        if true_target is not None:
            targets.append(true_target)
        if predicted_target is not None and not any(t["target_index"] == predicted_target["target_index"] for t in targets):
            targets.append(predicted_target)
        if not targets and predicted_target is not None:
            targets.append(predicted_target)
    if not targets:
        raise ValueError("Cannot determine attribution target.")
    return targets


def attribution_to_heatmap(np_module: Any, attributions: Any) -> Any:
    heatmap = attributions.detach().cpu()[0].abs().sum(dim=0).numpy()
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > min_value:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    else:
        heatmap = np_module.zeros_like(heatmap)
    return heatmap


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
    """
    Save XAI visual assets as separate files.

    Output policy
    -------------
    - input_png_path: resized input image only.
    - ig_overlay_path: input image with IG heatmap overlaid, no side-by-side layout.
    - ig_heatmap_path: colorized IG heatmap only.
    - ig_comparison_path: optional side-by-side diagnostic figure for quick review.
    - ig_mask_path: normalized grayscale attribution mask.
    - ig_top_percentile_mask_path: binary mask for the highest-attribution pixels.
    - ig_distribution_path: attribution value distribution.

    This keeps thesis-ready assets separate for LaTeX while preserving a
    diagnostic comparison image for manual inspection.
    """
    _, np_module, plt_module, _ = require_dependencies()

    heatmap = attribution_to_heatmap(np_module, attributions)
    height, width = heatmap.shape

    resized_image = image.resize((width, height))
    image_array = np_module.asarray(resized_image).astype("float32") / 255.0

    target_tag = sanitize_tag(f"{target_role}_{target_label}")

    input_path = case_dir / f"{case_id}__input.png"
    overlay_path = case_dir / f"{case_id}__{target_tag}__overlay.png"
    heatmap_path = case_dir / f"{case_id}__{target_tag}__heatmap.png"
    comparison_path = case_dir / f"{case_id}__{target_tag}__comparison.png"
    mask_path = case_dir / f"{case_id}__{target_tag}__mask.png"
    top_mask_path = case_dir / f"{case_id}__{target_tag}__top{int(100 - top_percentile)}_mask.png"
    distribution_path = case_dir / f"{case_id}__{target_tag}__distribution.png"

    case_dir.mkdir(parents=True, exist_ok=True)

    # 1) Input image only.
    resized_image.save(input_path)

    # 2) Colorized IG heatmap only.
    cmap = plt_module.get_cmap("inferno")
    heatmap_rgb = cmap(heatmap)[..., :3].astype("float32")
    Image.fromarray(
        (heatmap_rgb * 255.0).clip(0, 255).astype("uint8"),
        mode="RGB",
    ).save(heatmap_path)

    # 3) True overlay only: no axes, no titles, no side-by-side layout.
    overlay_alpha = 0.45
    overlay_rgb = ((1.0 - overlay_alpha) * image_array) + (overlay_alpha * heatmap_rgb)
    Image.fromarray(
        (overlay_rgb * 255.0).clip(0, 255).astype("uint8"),
        mode="RGB",
    ).save(overlay_path)

    # 4) Diagnostic comparison figure, useful for manual review but not required
    #    for the thesis layout.
    fig = plt_module.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image_array)
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay_rgb)
    ax2.set_title("Integrated Gradients overlay")
    ax2.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=180)
    plt_module.close(fig)

    # 5) Numeric masks.
    Image.fromarray(
        (heatmap * 255.0).clip(0, 255).astype("uint8"),
        mode="L",
    ).save(mask_path)

    threshold = float(np_module.percentile(heatmap, top_percentile))
    Image.fromarray(
        ((heatmap >= threshold).astype("uint8") * 255),
        mode="L",
    ).save(top_mask_path)

    # 6) Attribution distribution.
    fig = plt_module.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(heatmap.flatten(), bins=50)
    ax.set_title(f"{case_id} | {target_role}={target_label} | IG distribution")
    ax.set_xlabel("Normalized attribution value")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt_module.close(fig)

    return {
        "input_png_path": repo_relative_string(input_path),
        "ig_overlay_path": repo_relative_string(overlay_path),
        "ig_heatmap_path": repo_relative_string(heatmap_path),
        "ig_comparison_path": repo_relative_string(comparison_path),
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
    case_dir = run_output_dir / model_name / sanitize_tag(attack_name) / f"{case_id}__{sanitize_tag(sample_id)}"
    manifest_rows: list[dict[str, Any]] = []
    for target_spec in build_attribution_targets(row, attribution_target_mode):
        target_role = safe_str(target_spec["target_role"])
        target_index = int(target_spec["target_index"])
        target_label = safe_str(target_spec["target_label"])
        logging.info("Computing IG for %s | target_role=%s target_label=%s", case_id, target_role, target_label)
        attributions, delta = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_index,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        delta_value = float(delta.detach().cpu().reshape(-1)[0].item())
        title = (
            f"{case_id} | {model_name} {fold} | "
            f"label={row.get('final_label')} pred={row.get('prediction')} "
            f"target={target_role}:{target_label} conf={row.get('confidence')}"
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
        manifest_rows.append(
            {
                "run_tag": run_tag,
                "case_id": case_id,
                "created_at": created_at,
                "selection_strategy": strategy,
                "case_bucket": safe_str(row.get("case_bucket", "")),
                "xai_selection_reason": safe_str(row.get("xai_selection_reason", "")),
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
                "manual_selected": safe_str(row.get("manual_selected", "")),
                "selection_rank": safe_str(row.get("selection_rank", "")),
                "reviewer_id": safe_str(row.get("reviewer_id", "")),
                "review_timestamp": safe_str(row.get("review_timestamp", "")),
                "ig_output_path": output_paths["ig_overlay_path"],
                "input_png_path": output_paths["input_png_path"],
                "ig_overlay_path": output_paths["ig_overlay_path"],
                "ig_heatmap_path": output_paths["ig_heatmap_path"],
                "ig_comparison_path": output_paths["ig_comparison_path"],
                "ig_mask_path": output_paths["ig_mask_path"],
                "ig_top_percentile_mask_path": output_paths["ig_top_percentile_mask_path"],
                "ig_distribution_path": output_paths["ig_distribution_path"],
                "attribution_target_mode": attribution_target_mode,
                "attribution_target_role": target_role,
                "attribution_target_index": target_index,
                "attribution_target_label": target_label,
                "convergence_delta": delta_value,
                "abs_convergence_delta": abs(delta_value),
                "top_percentile": output_paths["top_percentile"],
                "top_percentile_threshold": output_paths["top_percentile_threshold"],
                "method": "Integrated Gradients",
                "n_steps": n_steps,

            }
        )
    return manifest_rows


def generate_integrated_gradients_for_rows(
    rows_df: pd.DataFrame,
    args: argparse.Namespace,
    run_tag: str,
    run_paths: dict[str, Path],
    created_at: str,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    if rows_df.empty:
        raise RuntimeError("No rows available for Integrated Gradients generation.")
    cache = AdapterCache(checkpoint_root=checkpoint_root, device=args.device, input_size=args.input_size)
    run_paths["run_output_dir"].mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(rows_df.iterrows(), start=1):
        logging.info("Generating XAI case %d/%d", index, len(rows_df))
        rows.extend(
            generate_case(
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
        )
    write_csv(run_paths["ig_manifest_csv"], rows)
    write_csv(run_paths["case_studies_manifest_csv"], rows)
    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    require_dependencies()

    run_tag = build_run_tag(args)
    run_paths = build_run_paths(run_tag)
    created_at = utc_now_iso()

    predictions_path = repo_relative_path(args.predictions_csv)
    checkpoint_root = repo_relative_path(args.checkpoint_root)

    logging.info("Run tag: %s", run_tag)
    logging.info("Predictions CSV: %s", predictions_path)
    logging.info("Checkpoint root: %s", checkpoint_root)

    if args.selection_manifest:
        ensure_run_outputs_do_not_exist(run_paths, args.force)
        selected_rows = load_selected_rows_from_manual_manifest(repo_relative_path(args.selection_manifest))
        ig_rows = generate_integrated_gradients_for_rows(
            rows_df=selected_rows,
            args=args,
            run_tag=run_tag,
            run_paths=run_paths,
            created_at=created_at,
            checkpoint_root=checkpoint_root,
        )
        write_json(
            run_paths["summary_json"],
            {
                "script": SCRIPT_NAME,
                "created_at": created_at,
                "run_tag": run_tag,
                "selection_manifest": repo_relative_string(args.selection_manifest),
                "selected_rows": int(len(selected_rows)),
                "generated_attribution_rows": int(len(ig_rows)),
                "outputs": {
                    "integrated_gradients_manifest": repo_relative_string(run_paths["ig_manifest_csv"]),
                    "case_studies_manifest": repo_relative_string(run_paths["case_studies_manifest_csv"]),
                    "integrated_gradients_output_dir": repo_relative_string(run_paths["run_output_dir"]),
                },
            },
        )
        return

    ensure_run_outputs_do_not_exist(run_paths, args.force, check_manual_db=args.manual_review)

    df = load_predictions(predictions_path, list(args.model))
    candidates = select_cases(
        df=df,
        strategy=args.strategy,
        max_cases=args.max_cases,
        threshold=args.high_confidence_threshold,
        candidate_limit=args.candidate_limit,
        attack_names=args.attack_name,
    )
    if candidates.empty:
        raise RuntimeError("No XAI cases selected. Check strategy, model, attack_name filter, or prediction CSV contents.")

    logging.info("Candidate rows selected: %d", len(candidates))

    if args.manual_review:
        if args.strategy == "attack_stratified":
            attacks_to_review = available_attacks_from_candidates(candidates, args.attack_name)
            if not attacks_to_review:
                raise RuntimeError("No attack_name values available for manual review.")
            logging.info("Attacks to review sequentially: %s", attacks_to_review)
            manual_db = run_manual_review(
                candidates=candidates,
                attacks_to_review=attacks_to_review,
                cases_per_attack=args.cases_per_attack,
                run_tag=run_tag,
                reviewer_id=args.reviewer_id,
                run_paths=run_paths,
                created_at=created_at,
            )
        elif args.strategy == "chapter5_core":
            buckets_to_review = available_case_buckets_from_candidates(candidates)
            if not buckets_to_review:
                raise RuntimeError("No case_bucket values available for Chapter 5 manual review.")
            logging.info("Chapter 5 buckets to review sequentially: %s", buckets_to_review)
            manual_db = run_manual_review_for_chapter5(
                candidates=candidates,
                cases_per_bucket=args.cases_per_bucket,
                run_tag=run_tag,
                reviewer_id=args.reviewer_id,
                run_paths=run_paths,
                created_at=created_at,
            )
        else:
            raise RuntimeError(
                "--manual-review is supported only with "
                "--strategy attack_stratified or --strategy chapter5_core."
            )

        selected_rows_for_ig = manual_db[
            manual_db["manual_selected"].astype(str).str.lower().isin({"true", "1", "yes"})
        ].copy()

        if selected_rows_for_ig.empty:
            raise RuntimeError("Manual review completed, but no selected cases were found.")

        if args.manual_only:
            write_json(
                run_paths["summary_json"],
                {
                    "script": SCRIPT_NAME,
                    "created_at": created_at,
                    "run_tag": run_tag,
                    "strategy": args.strategy,
                    "manual_review": True,
                    "manual_only": True,
                    "manual_selection_db": repo_relative_string(run_paths["manual_selection_db_csv"]),
                    "selected_cases": int(len(selected_rows_for_ig)),
                    "generated_attribution_rows": 0,
                },
            )
            return

        if not args.generate_after_manual:
            logging.info("Manual review completed. --generate-after-manual not set, so IG generation is skipped.")
            return
    else:
        if args.strategy == "chapter5_core":
            selected_rows_for_ig = candidates.copy()
        else:
            selected_rows_for_ig = candidates.head(args.max_cases).copy()

    ig_rows = generate_integrated_gradients_for_rows(
        rows_df=selected_rows_for_ig,
        args=args,
        run_tag=run_tag,
        run_paths=run_paths,
        created_at=created_at,
        checkpoint_root=checkpoint_root,
    )

    write_json(
        run_paths["summary_json"],
        {
            "script": SCRIPT_NAME,
            "created_at": created_at,
            "run_tag": run_tag,
            "strategy": args.strategy,
            "manual_review": bool(args.manual_review),
            "manual_only": bool(args.manual_only),
            "generate_after_manual": bool(args.generate_after_manual),
            "max_cases": args.max_cases,
            "cases_per_attack": args.cases_per_attack,
            "cases_per_bucket": args.cases_per_bucket,
            "candidate_limit": args.candidate_limit,
            "attack_name_filter": args.attack_name,
            "candidate_rows": int(len(candidates)),
            "selected_cases": int(len(selected_rows_for_ig)),
            "generated_attribution_rows": int(len(ig_rows)),
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
                "manual_selection_db": repo_relative_string(run_paths["manual_selection_db_csv"]) if run_paths["manual_selection_db_csv"].exists() else "",
                "manual_selection_summary": repo_relative_string(run_paths["manual_selection_summary_json"]) if run_paths["manual_selection_summary_json"].exists() else "",
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
