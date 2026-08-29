#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15_evaluate_proxy_models.py

Fold-aware evaluation entry point for FAIRLab proxy models.

The script evaluates transparent proxy models on clean, adversarial,
anti-forensic and OOD artifacts. It is intentionally separate from the
commercial forensic-tool evaluation layer.

Outputs:
- evaluation/proxy_models/proxy_model_predictions.csv
- results/metrics/proxy_model_clean_metrics.csv
- results/metrics/proxy_model_adversarial_metrics.csv
- results/metrics/proxy_model_anti_forensic_metrics.csv
- results/metrics/proxy_model_ood_metrics.csv
- results/metrics/proxy_model_comparative_metrics.csv
- results/metrics/final_core_metrics.csv
- results/metrics/final_robustness_metrics.csv
- results/metrics/final_confusion_matrices.csv
- results/metrics/final_ood_metrics.csv
- results/metrics/proxy_model_evaluation_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

# Make the repository root importable when the script is executed directly.
REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from datasets.scripts.attacks.adversarial_model_interface import TargetModelConfig
from datasets.scripts.attacks.adversarial_torch_model_adapters import build_target_model_adapter
from datasets.scripts.utils.paths import (
    ATTACKS_DIR,
    EVALUATION_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SPLIT_MANIFESTS_DIR,
    repo_relative_path,
)

SCRIPT_NAME = "evaluation/scripts/15_evaluate_proxy_models.py"
VALID_LABELS = ("non_weapon", "weapon")
SUPPORTED_MODELS = ("resnet18", "efficientnet_b0", "clip")
DEFAULT_MODELS = ("efficientnet_b0", "resnet18", "clip")
DEFAULT_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")

DEFAULT_CLEAN_MANIFEST = SPLIT_MANIFESTS_DIR / "clean_folds_manifest.csv"
DEFAULT_OOD_MANIFEST = SPLIT_MANIFESTS_DIR / "ood_eval_manifest.csv"
DEFAULT_ATTACK_MANIFESTS_DIR = ATTACKS_DIR / "manifests"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "models" / "checkpoints"

PROXY_EVAL_DIR = EVALUATION_DIR / "proxy_models"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_CSV = PROXY_EVAL_DIR / "proxy_model_predictions.csv"
SUMMARY_JSON = METRICS_DIR / "proxy_model_evaluation_summary.json"

FINAL_CORE_METRICS_CSV = METRICS_DIR / "final_core_metrics.csv"
FINAL_ROBUSTNESS_METRICS_CSV = METRICS_DIR / "final_robustness_metrics.csv"
FINAL_CONFUSION_MATRICES_CSV = METRICS_DIR / "final_confusion_matrices.csv"
FINAL_OOD_METRICS_CSV = METRICS_DIR / "final_ood_metrics.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return safe_str(value).lower()


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


def first_existing_column(df: pd.DataFrame, candidates: list[str], manifest_name: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find any path column in {manifest_name}: {candidates}")


def maybe_value(row: pd.Series, *names: str, default: str = "") -> str:
    for name in names:
        if name in row.index:
            value = safe_str(row.get(name, ""))
            if value:
                return value
    return default


def compute_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


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

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FAIRLab proxy models.")
    parser.add_argument("--clean-manifest", default=str(DEFAULT_CLEAN_MANIFEST))
    parser.add_argument("--ood-manifest", default=str(DEFAULT_OOD_MANIFEST))
    parser.add_argument("--attack-manifests-dir", default=str(DEFAULT_ATTACK_MANIFESTS_DIR))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--model", nargs="+", choices=SUPPORTED_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--ood-fold-mode", choices=("all", "fold_1"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


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


def clean_samples(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])
        rows.append({
            "sample_id": image_id,
            "sample_type": "clean",
            "attack_family": "none",
            "attack_name": "clean",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "image_relative_path": safe_str(row["split_relative_path"]),
            "image_sha256_manifest": safe_str(row.get("sha256", "")),
            "image_md5_manifest": safe_str(row.get("md5", "")),
            "manifest_source": repo_relative_string(path),
        })
    return rows


def ood_samples(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        image_id = safe_str(row["image_id"])
        rows.append({
            "sample_id": image_id,
            "sample_type": "ood",
            "attack_family": "none",
            "attack_name": "ood",
            "attack_target_model": "none",
            "original_image_id": image_id,
            "generated_image_id": image_id,
            "fold": "ood_eval_set",
            "final_label": "ood",
            "source_dataset": maybe_value(row, "source_dataset"),
            "image_relative_path": safe_str(row["ood_relative_path"]),
            "image_sha256_manifest": safe_str(row.get("sha256", "")),
            "image_md5_manifest": safe_str(row.get("md5", "")),
            "manifest_source": repo_relative_string(path),
        })
    return rows


def attack_samples(path: Path, expected_family: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    path_col = first_existing_column(
        df,
        ["perturbed_relative_path", "adversarial_relative_path", "generated_relative_path", "image_relative_path"],
        path.name,
    )
    rows = []
    for _, row in df.iterrows():
        original_id = maybe_value(row, "original_image_id", "image_id")
        attack_name = maybe_value(row, "attack_name", default=path.stem.replace("_manifest", ""))
        target_model = maybe_value(row, "target_model", "attack_target_model")
        if not target_model:
            target_model = "model_agnostic" if attack_name == "color_shift" else "unknown"
        generated_id = maybe_value(row, "generated_image_id", default=f"{original_id}__{attack_name}")
        rows.append({
            "sample_id": generated_id,
            "sample_type": "perturbed",
            "attack_family": maybe_value(row, "attack_family", default=expected_family),
            "attack_name": attack_name,
            "attack_target_model": target_model,
            "original_image_id": original_id,
            "generated_image_id": generated_id,
            "fold": safe_str(row["fold"]),
            "final_label": norm(row["final_label"]),
            "source_dataset": maybe_value(row, "source_dataset"),
            "image_relative_path": safe_str(row[path_col]),
            "image_sha256_manifest": maybe_value(row, "sha256_perturbed", "perturbed_sha256", "sha256"),
            "image_md5_manifest": maybe_value(row, "md5_perturbed", "perturbed_md5", "md5"),
            "manifest_source": repo_relative_string(path),
        })
    return rows


def discover_adversarial_manifests(manifests_dir: Path) -> list[Path]:
    return sorted(
        p for p in manifests_dir.glob("adversarial_*_manifest.csv")
        if "summary" not in p.name and "evaluation" not in p.name
    )


def load_all_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifests_dir = repo_relative_path(args.attack_manifests_dir)
    samples = clean_samples(repo_relative_path(args.clean_manifest))
    samples.extend(ood_samples(repo_relative_path(args.ood_manifest)))
    for manifest in discover_adversarial_manifests(manifests_dir):
        loaded = attack_samples(manifest, "adversarial")
        logging.info("Loaded %s: %d", manifest.name, len(loaded))
        samples.extend(loaded)
    anti_manifest = manifests_dir / "anti_forensic_attacks_manifest.csv"
    if anti_manifest.exists():
        loaded = attack_samples(anti_manifest, "anti_forensic")
        logging.info("Loaded %s: %d", anti_manifest.name, len(loaded))
        samples.extend(loaded)
    if args.limit > 0:
        samples = samples[: args.limit]
        logging.warning("Limit active: %d samples", len(samples))
    return samples


def folds_for_sample(sample: dict[str, Any], ood_fold_mode: str) -> list[str]:
    if sample["sample_type"] == "ood":
        return list(DEFAULT_FOLDS) if ood_fold_mode == "all" else ["fold_1"]
    return [sample["fold"]]


def open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image: {path}") from exc


def evaluate_sample(sample: dict[str, Any], model_name: str, fold: str, cache: AdapterCache) -> dict[str, Any]:
    row = dict(sample)
    row["evaluated_model"] = model_name
    row["evaluation_fold"] = fold
    row["model_checkpoint_path"] = repo_relative_string(cache.checkpoint_root / model_name / f"{fold}.pt")
    image_path = resolve_repo_path(sample["image_relative_path"])
    try:
        image = open_image(image_path)
        sha256, md5 = compute_hashes(image_path)
        adapter = cache.get(model_name, fold)
        probs = adapter.predict_proba(adapter.preprocess_image(image))
        pred = max(probs, key=probs.get)
        true_label = sample["final_label"]
        row.update({
            "image_exists": True,
            "image_sha256_actual": sha256,
            "image_md5_actual": md5,
            "prediction": pred,
            "confidence": float(probs[pred]),
            "prob_non_weapon": float(probs.get("non_weapon", 0.0)),
            "prob_weapon": float(probs.get("weapon", 0.0)),
            "true_label_confidence": float(probs.get(true_label, 0.0)) if true_label in VALID_LABELS else "",
            "correct": pred == true_label if true_label in VALID_LABELS else "",
            "error": "",
        })
    except Exception as exc:
        row.update({
            "image_exists": image_path.exists(),
            "image_sha256_actual": "",
            "image_md5_actual": "",
            "prediction": "",
            "confidence": "",
            "prob_non_weapon": "",
            "prob_weapon": "",
            "true_label_confidence": "",
            "correct": "",
            "error": f"{type(exc).__name__}: {exc}",
        })
    return row


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def f1_from_precision_recall(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_defined(*values: float | None) -> float | None:
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def binary_metrics(df: pd.DataFrame, group: dict[str, Any]) -> dict[str, Any]:
    valid = df[df["final_label"].isin(VALID_LABELS) & df["prediction"].isin(VALID_LABELS)]
    total = len(valid)
    if total == 0:
        return {
            **group,
            "total": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision_weapon": None,
            "recall_weapon": None,
            "f1_weapon": None,
            "precision_non_weapon": None,
            "recall_non_weapon": None,
            "f1_non_weapon": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "misclassification_rate": None,
            "weapon_to_non_weapon": 0,
            "non_weapon_to_weapon": 0,
            "confidence_mean": None,
        }

    y_true = valid["final_label"]
    y_pred = valid["prediction"]

    tp = int(((y_true == "weapon") & (y_pred == "weapon")).sum())
    tn = int(((y_true == "non_weapon") & (y_pred == "non_weapon")).sum())
    fp = int(((y_true == "non_weapon") & (y_pred == "weapon")).sum())
    fn = int(((y_true == "weapon") & (y_pred == "non_weapon")).sum())

    acc = safe_div(tp + tn, total)
    rec_w = safe_div(tp, tp + fn)
    rec_n = safe_div(tn, tn + fp)
    prec_w = safe_div(tp, tp + fp)
    prec_n = safe_div(tn, tn + fn)
    f1_w = f1_from_precision_recall(prec_w, rec_w)
    f1_n = f1_from_precision_recall(prec_n, rec_n)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    misclassification_rate = safe_div(fp + fn, total)

    return {
        **group,
        "total": total,
        "accuracy": acc,
        "balanced_accuracy": mean_defined(rec_w, rec_n),
        "precision_weapon": prec_w,
        "recall_weapon": rec_w,
        "f1_weapon": f1_w,
        "precision_non_weapon": prec_n,
        "recall_non_weapon": rec_n,
        "f1_non_weapon": f1_n,
        "macro_precision": mean_defined(prec_w, prec_n),
        "macro_recall": mean_defined(rec_w, rec_n),
        "macro_f1": mean_defined(f1_w, f1_n),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "misclassification_rate": misclassification_rate,
        "weapon_to_non_weapon": fn,
        "non_weapon_to_weapon": fp,
        "confidence_mean": pd.to_numeric(valid["confidence"], errors="coerce").mean(),
    }


def grouped_metrics(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for keys, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(binary_metrics(group_df, dict(zip(group_cols, keys))))
    return rows


def add_clean_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    clean = out[(out["sample_type"] == "clean") & (out["error"].astype(str) == "")]
    lookup = {
        (r["evaluated_model"], r["evaluation_fold"], r["original_image_id"]): r
        for _, r in clean.iterrows()
    }
    clean_correct = []
    clean_prediction = []
    clean_conf = []
    clean_prob_non_weapon = []
    clean_prob_weapon = []
    delta = []

    for _, r in out.iterrows():
        base = lookup.get((r["evaluated_model"], r["evaluation_fold"], r["original_image_id"]))
        if base is None:
            clean_correct.append("")
            clean_prediction.append("")
            clean_conf.append("")
            clean_prob_non_weapon.append("")
            clean_prob_weapon.append("")
            delta.append("")
            continue
        clean_correct.append(base["correct"])
        clean_prediction.append(base["prediction"])
        clean_conf.append(base["confidence"])
        clean_prob_non_weapon.append(base["prob_non_weapon"])
        clean_prob_weapon.append(base["prob_weapon"])
        try:
            delta.append(float(r["confidence"]) - float(base["confidence"]))
        except Exception:
            delta.append("")

    out["clean_correct"] = clean_correct
    out["clean_prediction"] = clean_prediction
    out["clean_confidence"] = clean_conf
    out["clean_prob_non_weapon"] = clean_prob_non_weapon
    out["clean_prob_weapon"] = clean_prob_weapon
    out["confidence_delta_vs_clean"] = delta
    return out


def comparative_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    pert = df[(df["sample_type"] == "perturbed") & (df["final_label"].isin(VALID_LABELS)) & (df["error"].astype(str) == "")]
    rows = []
    for keys, group in pert.groupby(["evaluated_model", "attack_family", "attack_name", "attack_target_model"], dropna=False):
        evaluated_model, family, attack_name, target_model = keys
        clean_ok = group["clean_correct"].astype(str).str.lower() == "true"
        pert_ok = group["correct"].astype(str).str.lower() == "true"
        induced = clean_ok & (~pert_ok)
        clean_weapon = clean_ok & (group["final_label"] == "weapon")
        clean_non_weapon = clean_ok & (group["final_label"] == "non_weapon")
        w_to_n = clean_weapon & (group["prediction"] == "non_weapon")
        n_to_w = clean_non_weapon & (group["prediction"] == "weapon")
        total = len(group)

        clean_like = group.copy()
        clean_like["prediction"] = clean_like["clean_prediction"]
        clean_like["confidence"] = clean_like["clean_confidence"]
        clean_binary = binary_metrics(clean_like, {})
        pert_binary = binary_metrics(group, {})

        clean_acc = safe_div(int(clean_ok.sum()), total)
        pert_acc = safe_div(int(pert_ok.sum()), total)
        clean_macro_f1 = clean_binary.get("macro_f1")
        perturbed_macro_f1 = pert_binary.get("macro_f1")
        f1_drop = None
        if clean_macro_f1 is not None and perturbed_macro_f1 is not None:
            f1_drop = clean_macro_f1 - perturbed_macro_f1

        attack_success_rate = safe_div(int(induced.sum()), int(clean_ok.sum()))
        confidence_shift = pd.to_numeric(group["confidence_delta_vs_clean"], errors="coerce").mean()

        rows.append({
            "evaluated_model": evaluated_model,
            "attack_family": family,
            "attack_name": attack_name,
            "attack_target_model": target_model,
            "total": total,
            "clean_accuracy": clean_acc,
            "perturbed_accuracy": pert_acc,
            "robust_accuracy": pert_acc,
            "accuracy_drop": None if clean_acc is None or pert_acc is None else clean_acc - pert_acc,
            "clean_macro_f1": clean_macro_f1,
            "perturbed_macro_f1": perturbed_macro_f1,
            "f1_drop": f1_drop,
            "attack_success_rate": attack_success_rate,
            "misclassification_rate": None if pert_acc is None else 1.0 - pert_acc,
            "induced_error_count": int(induced.sum()),
            "induced_error_rate_clean_correct": attack_success_rate,
            "weapon_to_non_weapon_count": int(w_to_n.sum()),
            "weapon_to_non_weapon_rate_clean_correct_weapon": safe_div(int(w_to_n.sum()), int(clean_weapon.sum())),
            "non_weapon_to_weapon_count": int(n_to_w.sum()),
            "non_weapon_to_weapon_rate_clean_correct_non_weapon": safe_div(int(n_to_w.sum()), int(clean_non_weapon.sum())),
            "confidence_shift": confidence_shift,
            "confidence_delta_mean": confidence_shift,
        })
    return rows


def ood_metrics(df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    ood = df[(df["sample_type"] == "ood") & (df["error"].astype(str) == "")]
    rows = []
    for model, group in ood.groupby("evaluated_model", dropna=False):
        conf = pd.to_numeric(group["confidence"], errors="coerce")
        rows.append({
            "evaluated_model": model,
            "sample_type": "ood",
            "total": len(group),
            "predicted_weapon": int((group["prediction"] == "weapon").sum()),
            "predicted_non_weapon": int((group["prediction"] == "non_weapon").sum()),
            "predicted_weapon_rate": safe_div(int((group["prediction"] == "weapon").sum()), len(group)),
            "confidence_mean": conf.mean(),
            "high_confidence_threshold": threshold,
            "high_confidence_count": int((conf >= threshold).sum()),
            "high_confidence_rate": safe_div(int((conf >= threshold).sum()), len(group)),
        })
    return rows


def standardize_core_row(row: dict[str, Any], sample_type: str, attack_family: str, attack_name: str, attack_target_model: str) -> dict[str, Any]:
    return {
        "evaluated_model": row.get("evaluated_model", ""),
        "sample_type": sample_type,
        "attack_family": attack_family,
        "attack_name": attack_name,
        "attack_target_model": attack_target_model,
        "total": row.get("total"),
        "accuracy": row.get("accuracy"),
        "balanced_accuracy": row.get("balanced_accuracy"),
        "precision_weapon": row.get("precision_weapon"),
        "recall_weapon": row.get("recall_weapon"),
        "f1_weapon": row.get("f1_weapon"),
        "precision_non_weapon": row.get("precision_non_weapon"),
        "recall_non_weapon": row.get("recall_non_weapon"),
        "f1_non_weapon": row.get("f1_non_weapon"),
        "macro_precision": row.get("macro_precision"),
        "macro_recall": row.get("macro_recall"),
        "macro_f1": row.get("macro_f1"),
        "tp": row.get("tp"),
        "tn": row.get("tn"),
        "fp": row.get("fp"),
        "fn": row.get("fn"),
        "false_positive_rate": row.get("false_positive_rate"),
        "false_negative_rate": row.get("false_negative_rate"),
        "misclassification_rate": row.get("misclassification_rate"),
        "weapon_to_non_weapon": row.get("weapon_to_non_weapon"),
        "non_weapon_to_weapon": row.get("non_weapon_to_weapon"),
        "confidence_mean": row.get("confidence_mean"),
    }


def final_core_metrics(
    clean_rows: list[dict[str, Any]],
    adversarial_rows: list[dict[str, Any]],
    anti_forensic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []

    for row in clean_rows:
        final_rows.append(standardize_core_row(row, "clean", "none", "clean", "none"))

    for row in adversarial_rows:
        final_rows.append(standardize_core_row(
            row,
            "perturbed",
            "adversarial",
            safe_str(row.get("attack_name", "")),
            safe_str(row.get("attack_target_model", "")),
        ))

    for row in anti_forensic_rows:
        final_rows.append(standardize_core_row(
            row,
            "perturbed",
            "anti_forensic",
            safe_str(row.get("attack_name", "")),
            safe_str(row.get("attack_target_model", "unknown")) or "unknown",
        ))

    return final_rows


def final_confusion_matrices(core_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "evaluated_model": row.get("evaluated_model", ""),
        "sample_type": row.get("sample_type", ""),
        "attack_family": row.get("attack_family", ""),
        "attack_name": row.get("attack_name", ""),
        "attack_target_model": row.get("attack_target_model", ""),
        "tn": row.get("tn"),
        "fp": row.get("fp"),
        "fn": row.get("fn"),
        "tp": row.get("tp"),
        "confusion_matrix_layout": "[[tn, fp], [fn, tp]]",
    } for row in core_rows]


def final_robustness_metrics(comparative_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "evaluated_model",
        "attack_family",
        "attack_name",
        "attack_target_model",
        "total",
        "clean_accuracy",
        "perturbed_accuracy",
        "robust_accuracy",
        "accuracy_drop",
        "clean_macro_f1",
        "perturbed_macro_f1",
        "f1_drop",
        "attack_success_rate",
        "induced_error_count",
        "misclassification_rate",
        "confidence_shift",
        "weapon_to_non_weapon_count",
        "weapon_to_non_weapon_rate_clean_correct_weapon",
        "non_weapon_to_weapon_count",
        "non_weapon_to_weapon_rate_clean_correct_non_weapon",
    ]
    return [{field: row.get(field, "") for field in fields} for row in comparative_rows]


def final_ood_metrics(ood_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return ood_rows


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if PREDICTIONS_CSV.exists() and not args.force:
        raise FileExistsError(f"Output already exists. Use --force: {PREDICTIONS_CSV}")

    samples = load_all_samples(args)
    cache = AdapterCache(repo_relative_path(args.checkpoint_root), args.device, args.input_size)
    rows = []
    created_at = utc_now_iso()
    for index, sample in enumerate(samples, start=1):
        for model_name in args.model:
            for fold in folds_for_sample(sample, args.ood_fold_mode):
                result = evaluate_sample(sample, model_name, fold, cache)
                result["created_at"] = created_at
                rows.append(result)
        if index % 100 == 0:
            logging.info("Evaluated %d/%d samples", index, len(samples))

    df = add_clean_comparison(pd.DataFrame(rows))
    PROXY_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_CSV, index=False)

    clean_df = df[df["sample_type"] == "clean"]
    adv_df = df[(df["sample_type"] == "perturbed") & (df["attack_family"] == "adversarial")]
    anti_df = df[(df["sample_type"] == "perturbed") & (df["attack_family"] == "anti_forensic")]

    clean_metrics = grouped_metrics(clean_df, ["evaluated_model"])
    adversarial_metrics = grouped_metrics(adv_df, ["evaluated_model", "attack_name", "attack_target_model"])
    anti_forensic_metrics = grouped_metrics(anti_df, ["evaluated_model", "attack_name"])
    ood_metric_rows = ood_metrics(df, args.high_confidence_threshold)
    comparative_metric_rows = comparative_metrics(df)

    final_core_rows = final_core_metrics(clean_metrics, adversarial_metrics, anti_forensic_metrics)
    final_confusion_rows = final_confusion_matrices(final_core_rows)
    final_robustness_rows = final_robustness_metrics(comparative_metric_rows)
    final_ood_rows = final_ood_metrics(ood_metric_rows)

    write_csv(METRICS_DIR / "proxy_model_clean_metrics.csv", clean_metrics)
    write_csv(METRICS_DIR / "proxy_model_adversarial_metrics.csv", adversarial_metrics)
    write_csv(METRICS_DIR / "proxy_model_anti_forensic_metrics.csv", anti_forensic_metrics)
    write_csv(METRICS_DIR / "proxy_model_ood_metrics.csv", ood_metric_rows)
    write_csv(METRICS_DIR / "proxy_model_comparative_metrics.csv", comparative_metric_rows)

    write_csv(FINAL_CORE_METRICS_CSV, final_core_rows)
    write_csv(FINAL_ROBUSTNESS_METRICS_CSV, final_robustness_rows)
    write_csv(FINAL_CONFUSION_MATRICES_CSV, final_confusion_rows)
    write_csv(FINAL_OOD_METRICS_CSV, final_ood_rows)

    summary = {
        "script": SCRIPT_NAME,
        "created_at": created_at,
        "models": list(args.model),
        "outputs": {
            "predictions_csv": repo_relative_string(PREDICTIONS_CSV),
            "metrics_dir": repo_relative_string(METRICS_DIR),
            "proxy_clean_metrics_csv": repo_relative_string(METRICS_DIR / "proxy_model_clean_metrics.csv"),
            "proxy_adversarial_metrics_csv": repo_relative_string(METRICS_DIR / "proxy_model_adversarial_metrics.csv"),
            "proxy_anti_forensic_metrics_csv": repo_relative_string(METRICS_DIR / "proxy_model_anti_forensic_metrics.csv"),
            "proxy_ood_metrics_csv": repo_relative_string(METRICS_DIR / "proxy_model_ood_metrics.csv"),
            "proxy_comparative_metrics_csv": repo_relative_string(METRICS_DIR / "proxy_model_comparative_metrics.csv"),
            "final_core_metrics_csv": repo_relative_string(FINAL_CORE_METRICS_CSV),
            "final_robustness_metrics_csv": repo_relative_string(FINAL_ROBUSTNESS_METRICS_CSV),
            "final_confusion_matrices_csv": repo_relative_string(FINAL_CONFUSION_MATRICES_CSV),
            "final_ood_metrics_csv": repo_relative_string(FINAL_OOD_METRICS_CSV),
            "summary_json": repo_relative_string(SUMMARY_JSON),
        },
        "counts": {
            "input_samples": len(samples),
            "prediction_rows": len(df),
            "errors": int((df["error"].astype(str) != "").sum()),
            "by_sample_type": dict(Counter(df["sample_type"])),
            "by_attack_family": dict(Counter(df["attack_family"])),
            "final_core_metric_rows": len(final_core_rows),
            "final_robustness_metric_rows": len(final_robustness_rows),
            "final_confusion_matrix_rows": len(final_confusion_rows),
            "final_ood_metric_rows": len(final_ood_rows),
        },
        "metric_schema_notes": {
            "positive_class": "weapon",
            "negative_class": "non_weapon",
            "confusion_matrix_layout": "[[tn, fp], [fn, tp]]",
            "false_positive_rate": "fp / (fp + tn)",
            "false_negative_rate": "fn / (fn + tp)",
            "robust_accuracy": "perturbed_accuracy",
            "attack_success_rate": "induced_error_count / clean_correct_count",
            "confidence_shift": "mean(perturbed_confidence - clean_confidence)",
            "misclassification_rate": "1 - accuracy for core metrics; 1 - perturbed_accuracy for robustness metrics",
        },
        "methodological_notes": [
            "Binary samples are evaluated with the checkpoint corresponding to their fold.",
            "OOD samples are evaluated separately and are not included in binary accuracy.",
            "Comparative metrics match perturbed predictions to clean predictions through model, fold and original_image_id.",
            "Final metric tables expose canonical metric names for thesis reporting and forensic-tool comparison.",
        ],
    }
    write_json(SUMMARY_JSON, summary)
    logging.info("Evaluation completed: %s", PREDICTIONS_CSV)
    logging.info("Final metric tables written under: %s", METRICS_DIR)


if __name__ == "__main__":
    main()
