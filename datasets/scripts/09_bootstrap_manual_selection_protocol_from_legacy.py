#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_bootstrap_manual_selection_protocol_from_legacy.py

Official bootstrap utility for the public FAIR-Lab thesis repository.

Purpose
-------
This script initializes the new official manual selection protocol database
from a clean prepared review manifest and previously validated legacy final
outputs from the historical working repository.

The goal is to preserve the already validated manual selection decisions while
migrating them into the new public, documented, and academically defendable
pipeline.

Inputs
------
1. Prepared full review manifest:
   - datasets/prepared/manifests/review_manifest_full.csv

2. Legacy validated final outputs:
   - legacy frozen dataset CSV
   - legacy removed dataset CSV

Outputs
-------
Working DB:
- datasets/final/manifests/manual_selection_protocol_db.csv

Derived exports:
- datasets/final/manifests/manual_selection_final_1500.csv
- datasets/final/manifests/manual_selection_removed.csv
- datasets/final/manifests/manual_selection_adversarial_subset.csv

Reports:
- datasets/final/reports/manual_selection_log.csv
- datasets/final/reports/manual_selection_state.json
- datasets/final/reports/manual_selection_summary.json

Methodological rationale
------------------------
This script does not re-run the historical manual selection process.
Instead, it imports already validated legacy final decisions into the new
official protocol structure, allowing the public repository to:

- preserve the validated final dataset
- expose a cleaner official reviewer pipeline
- resume or inspect manual decisions using the new reviewer
- keep all outputs consistent with the thesis release
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.scripts.utils.paths import PREPARED_DATASETS_DIR, repo_relative_path


# =============================================================================
# Paths
# =============================================================================

PREPARED_MANIFEST_PATH = PREPARED_DATASETS_DIR / "manifests" / "review_manifest_full.csv"

FINAL_DIR = PREPARED_DATASETS_DIR.parent / "final"
FINAL_MANIFESTS_DIR = FINAL_DIR / "manifests"
FINAL_REPORTS_DIR = FINAL_DIR / "reports"

WORK_DB_PATH = FINAL_MANIFESTS_DIR / "manual_selection_protocol_db.csv"

OUT_FINAL_1500_CSV = FINAL_MANIFESTS_DIR / "manual_selection_final_1500.csv"
OUT_REMOVED_CSV = FINAL_MANIFESTS_DIR / "manual_selection_removed.csv"
OUT_ADVERSARIAL_SUBSET_CSV = FINAL_MANIFESTS_DIR / "manual_selection_adversarial_subset.csv"

LOG_PATH = FINAL_REPORTS_DIR / "manual_selection_log.csv"
STATE_PATH = FINAL_REPORTS_DIR / "manual_selection_state.json"
SUMMARY_PATH = FINAL_REPORTS_DIR / "manual_selection_summary.json"

BACKUP_DIR = FINAL_REPORTS_DIR / "backups"


# These defaults are convenient when the script is executed inside the old
# working repository. In the new public repository, explicit CLI paths can be
# passed if preferred.
DEFAULT_LEGACY_FROZEN = PREPARED_DATASETS_DIR / "final_pool" / "33_final_frozen_dataset.csv"
DEFAULT_LEGACY_REMOVED = PREPARED_DATASETS_DIR / "final_pool" / "33_final_removed_dataset.csv"


# =============================================================================
# Targets
# =============================================================================

TARGET_WEAPON = 500
TARGET_NON_WEAPON = 500
TARGET_OOD = 500


# =============================================================================
# Utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_dirs() -> None:
    FINAL_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: Path, make_backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if make_backup and path.exists():
        backup_path = BACKUP_DIR / f"{path.stem}_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{path.suffix}"
        shutil.copy2(path, backup_path)

    tmp_path = path.with_suffix(".tmp.csv")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    tmp_path.replace(path)


def safe_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def ensure_protocol_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the protocol DB contains the full official schema.
    """
    defaults = {
        "image_id": "",
        "relative_path": "",
        "source_dataset": "",
        "source_group": "",
        "sha256": "",
        "auto_label": "",
        "auto_confidence": "",
        "score_margin": "",
        "weapon_score": "",
        "non_weapon_score": "",
        "selection_label": "",
        "selection_status": "pending",
        "selected_for_final": "",
        "selection_notes": "",
        "selection_reviewer_id": "",
        "selection_timestamp": "",
        "selection_source_priority": "",
        "legacy_bootstrap_status": "",
        "legacy_assigned_label": "",
        "bootstrap_source": "",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def source_priority(dataset_name: str) -> int:
    priority_map = {
        "04_telegram_youtube": 1,
        "05_deepweb": 2,
        "03_google_scraped": 3,
        "01_kaggle_weapon": 4,
        "02_deepfirearm": 5,
    }
    return priority_map.get(dataset_name, 999)


def validate_required_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def build_initial_protocol_db(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the clean protocol DB starting from the prepared full review manifest.
    """
    df = manifest_df.copy()
    df = ensure_protocol_columns(df)

    validate_required_columns(df, ["image_id", "relative_path", "source_dataset"], "prepared_manifest")

    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    df["selection_source_priority"] = df["source_dataset"].map(source_priority).fillna(999).astype(int)

    protocol_columns = [
        "image_id",
        "relative_path",
        "source_dataset",
        "source_group",
        "sha256",
        "auto_label",
        "auto_confidence",
        "score_margin",
        "weapon_score",
        "non_weapon_score",
        "selection_label",
        "selection_status",
        "selected_for_final",
        "selection_notes",
        "selection_reviewer_id",
        "selection_timestamp",
        "selection_source_priority",
        "legacy_bootstrap_status",
        "legacy_assigned_label",
        "bootstrap_source",
    ]

    return df[protocol_columns].copy()


def read_legacy_csv(path: Path, df_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{df_name} not found: {path}")

    df = pd.read_csv(path)
    validate_required_columns(df, ["image_id"], df_name)

    df["image_id"] = df["image_id"].astype(str)

    if "assigned_label" not in df.columns:
        df["assigned_label"] = ""

    return df


def append_bootstrap_log(
    rows: list[dict[str, Any]],
    log_path: Path,
) -> None:
    """
    Create a fresh bootstrap log containing one event per imported legacy decision.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "reviewer_id",
                "image_id",
                "source_dataset",
                "relative_path",
                "action",
                "previous_label",
                "new_label",
                "auto_label",
                "auto_confidence",
                "score_margin",
                "selected_for_final",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_protocol_outputs(protocol_df: pd.DataFrame) -> dict[str, Any]:
    """
    Export final 1500, removed set, adversarial subset, and summary.
    """
    labels = protocol_df["selection_label"].map(norm)

    final_df = protocol_df[labels.isin({"weapon", "non_weapon", "ood"})].copy()
    final_df = final_df.sort_values(
        ["selection_label", "source_dataset", "image_id"],
        kind="stable",
    ).reset_index(drop=True)

    removed_df = protocol_df[labels == "exclude"].copy()
    removed_df = removed_df.sort_values(
        ["source_dataset", "image_id"],
        kind="stable",
    ).reset_index(drop=True)

    adversarial_df = final_df[final_df["selection_label"].map(norm).isin({"weapon", "non_weapon"})].copy()
    adversarial_df = adversarial_df.sort_values(
        ["selection_label", "source_dataset", "image_id"],
        kind="stable",
    ).reset_index(drop=True)

    safe_write_csv(final_df, OUT_FINAL_1500_CSV)
    safe_write_csv(removed_df, OUT_REMOVED_CSV)
    safe_write_csv(adversarial_df, OUT_ADVERSARIAL_SUBSET_CSV)

    counts = {
        "weapon": int((final_df["selection_label"].map(norm) == "weapon").sum()),
        "non_weapon": int((final_df["selection_label"].map(norm) == "non_weapon").sum()),
        "ood": int((final_df["selection_label"].map(norm) == "ood").sum()),
        "exclude": int((protocol_df["selection_label"].map(norm) == "exclude").sum()),
        "pending": int((protocol_df["selection_status"].map(norm) != "reviewed").sum()),
        "reviewed": int((protocol_df["selection_status"].map(norm) == "reviewed").sum()),
    }

    summary = {
        "timestamp": now_iso(),
        "input_manifest": str(PREPARED_MANIFEST_PATH),
        "work_db": str(WORK_DB_PATH),
        "outputs": {
            "final_1500_csv": str(OUT_FINAL_1500_CSV),
            "removed_csv": str(OUT_REMOVED_CSV),
            "adversarial_subset_csv": str(OUT_ADVERSARIAL_SUBSET_CSV),
            "log_csv": str(LOG_PATH),
            "state_json": str(STATE_PATH),
            "summary_json": str(SUMMARY_PATH),
        },
        "counts": {
            "weapon": counts["weapon"],
            "non_weapon": counts["non_weapon"],
            "ood": counts["ood"],
            "exclude": counts["exclude"],
            "pending": counts["pending"],
            "reviewed": counts["reviewed"],
            "final_total": counts["weapon"] + counts["non_weapon"] + counts["ood"],
            "adversarial_subset_total": counts["weapon"] + counts["non_weapon"],
        },
        "checks": {
            "final_1500_exact": (
                counts["weapon"] == TARGET_WEAPON
                and counts["non_weapon"] == TARGET_NON_WEAPON
                and counts["ood"] == TARGET_OOD
            ),
            "adversarial_subset_1000_exact": (
                counts["weapon"] + counts["non_weapon"] == 1000
            ),
        },
        "bootstrap_distribution": (
            protocol_df.groupby(["legacy_bootstrap_status", "source_dataset"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["legacy_bootstrap_status", "count", "source_dataset"], ascending=[True, False, True], kind="stable")
            .to_dict(orient="records")
        ),
    }

    safe_write_json(summary, SUMMARY_PATH)
    return summary


# =============================================================================
# Bootstrap logic
# =============================================================================

def bootstrap_protocol_db(
    manifest_df: pd.DataFrame,
    legacy_frozen_df: pd.DataFrame,
    legacy_removed_df: pd.DataFrame,
    reviewer_id: str,
) -> pd.DataFrame:
    """
    Merge legacy validated final outputs into the new official protocol DB.

    Rules
    -----
    - If image_id is in legacy frozen:
        selection_label   = legacy assigned_label
        selection_status  = reviewed
        selected_for_final= yes

    - If image_id is in legacy removed:
        selection_label   = exclude
        selection_status  = reviewed
        selected_for_final= no

    - Otherwise:
        selection_label   = ""
        selection_status  = pending
    """
    protocol_df = build_initial_protocol_db(manifest_df)

    frozen_map = {
        str(row["image_id"]): row
        for _, row in legacy_frozen_df.iterrows()
    }
    removed_map = {
        str(row["image_id"]): row
        for _, row in legacy_removed_df.iterrows()
    }

    timestamp = now_iso()

    for idx, row in protocol_df.iterrows():
        image_id = str(row["image_id"])

        if image_id in frozen_map:
            legacy_row = frozen_map[image_id]
            assigned_label = norm(legacy_row.get("assigned_label", ""))

            if assigned_label not in {"weapon", "non_weapon", "ood"}:
                continue

            protocol_df.at[idx, "selection_label"] = assigned_label
            protocol_df.at[idx, "selection_status"] = "reviewed"
            protocol_df.at[idx, "selected_for_final"] = "yes"
            protocol_df.at[idx, "selection_reviewer_id"] = reviewer_id
            protocol_df.at[idx, "selection_timestamp"] = timestamp
            protocol_df.at[idx, "legacy_bootstrap_status"] = "frozen"
            protocol_df.at[idx, "legacy_assigned_label"] = assigned_label
            protocol_df.at[idx, "bootstrap_source"] = "legacy_validated_outputs"

        elif image_id in removed_map:
            legacy_row = removed_map[image_id]
            legacy_assigned_label = norm(legacy_row.get("assigned_label", ""))

            protocol_df.at[idx, "selection_label"] = "exclude"
            protocol_df.at[idx, "selection_status"] = "reviewed"
            protocol_df.at[idx, "selected_for_final"] = "no"
            protocol_df.at[idx, "selection_reviewer_id"] = reviewer_id
            protocol_df.at[idx, "selection_timestamp"] = timestamp
            protocol_df.at[idx, "legacy_bootstrap_status"] = "removed"
            protocol_df.at[idx, "legacy_assigned_label"] = legacy_assigned_label
            protocol_df.at[idx, "bootstrap_source"] = "legacy_validated_outputs"

        else:
            protocol_df.at[idx, "legacy_bootstrap_status"] = "unseen"
            protocol_df.at[idx, "bootstrap_source"] = "prepared_manifest_only"

    return protocol_df


def build_bootstrap_log_rows(protocol_df: pd.DataFrame, reviewer_id: str) -> list[dict[str, Any]]:
    """
    Build a clean log file describing the imported bootstrap decisions.
    """
    rows: list[dict[str, Any]] = []

    for _, row in protocol_df.iterrows():
        bootstrap_status = norm(row.get("legacy_bootstrap_status", ""))
        selection_label = norm(row.get("selection_label", ""))

        if bootstrap_status not in {"frozen", "removed"}:
            continue

        rows.append(
            {
                "timestamp": safe_str(row.get("selection_timestamp", "")) or now_iso(),
                "reviewer_id": reviewer_id,
                "image_id": safe_str(row.get("image_id", "")),
                "source_dataset": safe_str(row.get("source_dataset", "")),
                "relative_path": safe_str(row.get("relative_path", "")),
                "action": "bootstrap_import",
                "previous_label": "",
                "new_label": selection_label,
                "auto_label": safe_str(row.get("auto_label", "")),
                "auto_confidence": safe_str(row.get("auto_confidence", "")),
                "score_margin": safe_str(row.get("score_margin", "")),
                "selected_for_final": safe_str(row.get("selected_for_final", "")),
            }
        )

    return rows


def build_initial_state(reviewer_id: str) -> dict[str, Any]:
    """
    Create an initial reviewer state file for the new public protocol.
    """
    return {
        "timestamp": now_iso(),
        "reviewer_id": reviewer_id,
        "current_start": 0,
        "selected_df_index": None,
        "last_action": "bootstrap_initialized_from_legacy_outputs",
        "view_mode": "pending",
        "last_action_stack": [],
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the new official manual selection protocol DB from legacy validated final outputs."
    )
    parser.add_argument(
        "--prepared-manifest",
        type=str,
        default=str(PREPARED_MANIFEST_PATH),
        help="Path to datasets/prepared/manifests/review_manifest_full.csv",
    )
    parser.add_argument(
        "--legacy-frozen",
        type=str,
        default=str(DEFAULT_LEGACY_FROZEN),
        help="Path to legacy 33_final_frozen_dataset.csv",
    )
    parser.add_argument(
        "--legacy-removed",
        type=str,
        default=str(DEFAULT_LEGACY_REMOVED),
        help="Path to legacy 33_final_removed_dataset.csv",
    )
    parser.add_argument(
        "--reviewer-id",
        type=str,
        default="Lello",
        help="Reviewer ID stored in the bootstrap metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manual_selection_protocol_db.csv and related outputs.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    ensure_dirs()

    prepared_manifest_path = repo_relative_path(args.prepared_manifest)
    legacy_frozen_path = repo_relative_path(args.legacy_frozen)
    legacy_removed_path = repo_relative_path(args.legacy_removed)

    print(f"PREPARED_MANIFEST : {prepared_manifest_path}")
    print(f"LEGACY_FROZEN     : {legacy_frozen_path}")
    print(f"LEGACY_REMOVED    : {legacy_removed_path}")
    print(f"WORK_DB           : {WORK_DB_PATH}")
    print()

    if WORK_DB_PATH.exists() and not args.force:
        raise FileExistsError(
            f"{WORK_DB_PATH} already exists. Use --force to rebuild the bootstrap outputs."
        )

    if not prepared_manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {prepared_manifest_path}")

    manifest_df = pd.read_csv(prepared_manifest_path)
    legacy_frozen_df = read_legacy_csv(legacy_frozen_path, "legacy_frozen")
    legacy_removed_df = read_legacy_csv(legacy_removed_path, "legacy_removed")

    protocol_df = bootstrap_protocol_db(
        manifest_df=manifest_df,
        legacy_frozen_df=legacy_frozen_df,
        legacy_removed_df=legacy_removed_df,
        reviewer_id=args.reviewer_id,
    )

    safe_write_csv(protocol_df, WORK_DB_PATH, make_backup=args.force)

    bootstrap_log_rows = build_bootstrap_log_rows(protocol_df, args.reviewer_id)
    append_bootstrap_log(bootstrap_log_rows, LOG_PATH)

    initial_state = build_initial_state(args.reviewer_id)
    safe_write_json(initial_state, STATE_PATH)

    summary = export_protocol_outputs(protocol_df)

    print("=== BOOTSTRAP COMPLETED ===")
    print(f"Protocol DB                  : {WORK_DB_PATH}")
    print(f"Bootstrap log                : {LOG_PATH}")
    print(f"Reviewer state               : {STATE_PATH}")
    print(f"Summary JSON                 : {SUMMARY_PATH}")
    print()
    print("=== COUNTS ===")
    print(f"weapon                       : {summary['counts']['weapon']}")
    print(f"non_weapon                   : {summary['counts']['non_weapon']}")
    print(f"ood                          : {summary['counts']['ood']}")
    print(f"exclude                      : {summary['counts']['exclude']}")
    print(f"pending                      : {summary['counts']['pending']}")
    print(f"reviewed                     : {summary['counts']['reviewed']}")
    print(f"final_total                  : {summary['counts']['final_total']}")
    print(f"adversarial_subset_total     : {summary['counts']['adversarial_subset_total']}")
    print()
    print("=== CHECKS ===")
    print(f"final_1500_exact             : {summary['checks']['final_1500_exact']}")
    print(f"adversarial_subset_1000_exact: {summary['checks']['adversarial_subset_1000_exact']}")
    print()
    print("=== NEXT STEP ===")
    print("Run the official reviewer to inspect or continue the selection process:")
    print("  python datasets/scripts/final/10_manual_selection_protocol_reviewer.py")


if __name__ == "__main__":
    main()