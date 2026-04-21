#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_manual_selection_protocol_reviewer.py

Official manual selection protocol reviewer for the FAIR-Lab public thesis pipeline.

Purpose
-------
This script provides a single, documented, human-in-the-loop reviewer used to
manually select the final 1500 images from the full prepared review manifest
according to the thesis methodology.

Input
-----
- datasets/prepared/manifests/review_manifest_full.csv

Outputs
-------
Working DB:
- datasets/final/manifests/manual_selection_protocol_db.csv

Final exports:
- datasets/final/manifests/manual_selection_final_1500.csv
- datasets/final/manifests/manual_selection_removed.csv
- datasets/final/manifests/manual_selection_adversarial_subset.csv

Reports:
- datasets/final/reports/manual_selection_log.csv
- datasets/final/reports/manual_selection_state.json
- datasets/final/reports/manual_selection_summary.json

Methodological notes
--------------------
- The process is manual, not fully automatic.
- The script documents every decision with logs and timestamps.
- Global targets are enforced:
    * 500 weapon
    * 500 non_weapon
    * 500 ood
- Excluded items are tracked explicitly.
- The adversarial subset is derived automatically from the final 1500 by keeping
  only weapon and non_weapon samples.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import MouseButton

from datasets.scripts.utils.paths import PREPARED_DATASETS_DIR, repo_relative_path


# =============================================================================
# Paths
# =============================================================================

PREPARED_MANIFEST_PATH = PREPARED_DATASETS_DIR / "manifests" / "review_manifest_full.csv"

FINAL_DIR = PREPARED_DATASETS_DIR.parent / "final"
FINAL_MANIFESTS_DIR = FINAL_DIR / "manifests"
FINAL_REPORTS_DIR = FINAL_DIR / "reports"
FINAL_EXPORTS_DIR = FINAL_DIR / "exports"

WORK_DB_PATH = FINAL_MANIFESTS_DIR / "manual_selection_protocol_db.csv"

OUT_FINAL_1500_CSV = FINAL_MANIFESTS_DIR / "manual_selection_final_1500.csv"
OUT_REMOVED_CSV = FINAL_MANIFESTS_DIR / "manual_selection_removed.csv"
OUT_ADVERSARIAL_SUBSET_CSV = FINAL_MANIFESTS_DIR / "manual_selection_adversarial_subset.csv"

LOG_PATH = FINAL_REPORTS_DIR / "manual_selection_log.csv"
STATE_PATH = FINAL_REPORTS_DIR / "manual_selection_state.json"
SUMMARY_PATH = FINAL_REPORTS_DIR / "manual_selection_summary.json"

BACKUP_DIR = FINAL_REPORTS_DIR / "backups"


# =============================================================================
# Targets and UI
# =============================================================================

TARGET_WEAPON = 500
TARGET_NON_WEAPON = 500
TARGET_OOD = 500

VALID_FINAL_LABELS = {"weapon", "non_weapon", "ood", "exclude"}

BATCH_SIZE = 12
N_COLS = 4
FIG_W = 16
FIG_H = 10

VIEW_MODES = {"pending", "reviewed", "selected", "excluded"}

LABEL_COLORS = {
    "": "gray",
    "weapon": "green",
    "non_weapon": "blue",
    "ood": "orange",
    "exclude": "gray",
}

SOURCE_PRIORITY = {
    "04_telegram_youtube": 1,
    "05_deepweb": 2,
    "03_google_scraped": 3,
    "01_kaggle_weapon": 4,
    "02_deepfirearm": 5,
}

HELP_TEXT = """
OFFICIAL MANUAL SELECTION PROTOCOL REVIEWER

GOAL
- Build the final 1500-image dataset manually, according to the thesis methodology.

FINAL TARGETS
- weapon      : 500
- non_weapon  : 500
- ood         : 500

LABELS
- WEAPON
  Real individual firearm, clearly visible, coherent with the main task.
- NON_WEAPON
  Clean realistic negative sample, no weapon present.
- OOD
  Out-of-distribution / borderline / semantically ambiguous sample.
- EXCLUDE
  Unusable, unreliable, or intentionally excluded from the final release.

MOUSE
- Left click   = WEAPON
- Right click  = NON_WEAPON
- Middle click = OOD

KEYS
- w = WEAPON
- n = NON_WEAPON
- o = OOD
- e / x = EXCLUDE
- a = clear current decision (back to pending)
- u = undo last action
- r = toggle view mode
- s = save
- q = save + quit
- h = help
- t = summary
- Enter = zoom selected image
- Right / Space = next batch
- Left / Backspace = previous batch
- 1..9 = select image in current batch

NOTES
- Global class targets are hard-enforced.
- Source distribution is monitored, but not hard-blocked.
- All decisions are logged.
- Final exports are regenerated automatically after each save.
""".strip()


# =============================================================================
# Utility functions
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
    FINAL_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
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


def append_session_log(row: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
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
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_image_path(relative_path_value: str) -> Path | None:
    rel = Path(str(relative_path_value).strip())

    candidates = [
        PREPARED_DATASETS_DIR / "final_pool" / rel,
        PREPARED_DATASETS_DIR / rel,
        rel,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def to_float(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# =============================================================================
# DB initialization
# =============================================================================

def ensure_protocol_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def build_initial_protocol_db(manifest_df: pd.DataFrame) -> pd.DataFrame:
    df = manifest_df.copy()
    df = ensure_protocol_columns(df)

    if "source_dataset" in df.columns:
        df["selection_source_priority"] = df["source_dataset"].map(SOURCE_PRIORITY).fillna(999).astype(int)
    else:
        df["selection_source_priority"] = 999

    if "image_id" not in df.columns:
        raise ValueError("The manifest must contain 'image_id'.")

    df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    protocol_cols = [
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
    ]

    for col in protocol_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[protocol_cols].copy()
    return df


# =============================================================================
# Reviewer class
# =============================================================================

class ManualSelectionProtocolReviewer:
    def __init__(self) -> None:
        ensure_dirs()

        self.reviewer_id = input("Reviewer ID [default: Lello]: ").strip() or "Lello"

        if WORK_DB_PATH.exists():
            self.df = pd.read_csv(WORK_DB_PATH)
        else:
            if not PREPARED_MANIFEST_PATH.exists():
                raise FileNotFoundError(f"Prepared manifest not found: {PREPARED_MANIFEST_PATH}")
            manifest_df = pd.read_csv(PREPARED_MANIFEST_PATH)
            self.df = build_initial_protocol_db(manifest_df)
            safe_write_csv(self.df, WORK_DB_PATH, make_backup=False)

        self.df = ensure_protocol_columns(self.df)
        self.df["image_id"] = self.df["image_id"].astype(str)

        self.fig = None
        self.axes = []
        self.help_fig = None
        self.summary_fig = None

        self.current_start = 0
        self.selected_pos = 0
        self.batch_indices: list[int] = []
        self.ax_to_df_index: dict[Any, int] = {}

        self.last_action_stack: list[dict[str, Any]] = []
        self.view_mode = "pending"

        self.load_state()

    # -------------------------------------------------------------------------
    # Counts and summaries
    # -------------------------------------------------------------------------
    def count_label(self, label: str) -> int:
        return int((self.df["selection_label"].map(norm) == label).sum())

    def count_pending(self) -> int:
        return int((self.df["selection_status"].map(norm) != "reviewed").sum())

    def count_reviewed(self) -> int:
        return int((self.df["selection_status"].map(norm) == "reviewed").sum())

    def final_counts(self) -> dict[str, int]:
        return {
            "weapon": self.count_label("weapon"),
            "non_weapon": self.count_label("non_weapon"),
            "ood": self.count_label("ood"),
            "exclude": self.count_label("exclude"),
        }

    def remaining_targets(self) -> dict[str, int]:
        counts = self.final_counts()
        return {
            "weapon": max(0, TARGET_WEAPON - counts["weapon"]),
            "non_weapon": max(0, TARGET_NON_WEAPON - counts["non_weapon"]),
            "ood": max(0, TARGET_OOD - counts["ood"]),
        }

    def all_targets_reached(self) -> bool:
        rem = self.remaining_targets()
        return rem["weapon"] <= 0 and rem["non_weapon"] <= 0 and rem["ood"] <= 0

    def source_status(self) -> pd.DataFrame:
        tmp = self.df.copy()
        tmp["_label"] = tmp["selection_label"].map(norm)
        tmp["_status"] = tmp["selection_status"].map(norm)

        rows = []
        for src, g in tmp.groupby("source_dataset", dropna=False):
            rows.append({
                "source_dataset": src,
                "rows": int(len(g)),
                "weapon": int((g["_label"] == "weapon").sum()),
                "non_weapon": int((g["_label"] == "non_weapon").sum()),
                "ood": int((g["_label"] == "ood").sum()),
                "exclude": int((g["_label"] == "exclude").sum()),
                "pending": int((g["_status"] != "reviewed").sum()),
                "reviewed": int((g["_status"] == "reviewed").sum()),
            })
        if not rows:
            return pd.DataFrame(columns=["source_dataset", "rows", "weapon", "non_weapon", "ood", "exclude", "pending", "reviewed"])
        return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)

    def print_status(self) -> None:
        counts = self.final_counts()
        rem = self.remaining_targets()

        print("\n=== OFFICIAL MANUAL SELECTION STATUS ===")
        print(
            f"weapon={counts['weapon']}/{TARGET_WEAPON} (remaining={rem['weapon']}) | "
            f"non_weapon={counts['non_weapon']}/{TARGET_NON_WEAPON} (remaining={rem['non_weapon']}) | "
            f"ood={counts['ood']}/{TARGET_OOD} (remaining={rem['ood']}) | "
            f"exclude={counts['exclude']} | "
            f"pending={self.count_pending()} | "
            f"reviewed={self.count_reviewed()}"
        )

        src_df = self.source_status()
        if not src_df.empty:
            print("\n=== CURRENT SOURCE STATUS ===")
            for _, r in src_df.iterrows():
                print(
                    f"{str(r['source_dataset']):<24} "
                    f"rows={int(r['rows']):>5} | "
                    f"W={int(r['weapon']):>4} | "
                    f"NW={int(r['non_weapon']):>4} | "
                    f"OOD={int(r['ood']):>4} | "
                    f"EX={int(r['exclude']):>4} | "
                    f"P={int(r['pending']):>5} | "
                    f"R={int(r['reviewed']):>5}"
                )

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------
    def save_state(self, last_action: str = "") -> None:
        selected_df_index = None
        if self.batch_indices and 0 <= self.selected_pos < len(self.batch_indices):
            selected_df_index = int(self.batch_indices[self.selected_pos])

        state = {
            "timestamp": now_iso(),
            "reviewer_id": self.reviewer_id,
            "current_start": self.current_start,
            "selected_df_index": selected_df_index,
            "last_action": last_action,
            "view_mode": self.view_mode,
            "last_action_stack": self.last_action_stack[-20:],
        }
        safe_write_json(state, STATE_PATH)

    def load_state(self) -> None:
        if not STATE_PATH.exists():
            self.print_status()
            return

        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            print("\nLast saved state:")
            print(json.dumps(state, indent=2, ensure_ascii=False))

            self.current_start = int(state.get("current_start", 0) or 0)
            self.selected_pos = 0
            self.view_mode = state.get("view_mode", "pending") or "pending"

            raw_stack = state.get("last_action_stack", [])
            if isinstance(raw_stack, list):
                self.last_action_stack = raw_stack
        except Exception as exc:
            print(f"[WARN] Could not load previous state: {exc}")

        self.print_status()

    # -------------------------------------------------------------------------
    # Priority and batch construction
    # -------------------------------------------------------------------------
    def infer_preferred_class(self, row: pd.Series) -> str:
        remaining = self.remaining_targets()

        auto_label = norm(row.get("auto_label", ""))
        auto_conf = to_float(row.get("auto_confidence", ""), -1.0)
        score_margin = to_float(row.get("score_margin", ""), 999.0)
        source = norm(row.get("source_dataset", ""))

        class_score = {
            "weapon": 500,
            "non_weapon": 500,
            "ood": 500,
        }

        if remaining["weapon"] > 0:
            score = 100
            if auto_label == "weapon":
                score = 0
            elif source in {"01_kaggle_weapon", "02_deepfirearm"}:
                score = 10
            elif auto_conf >= 0.80:
                score = 15
            class_score["weapon"] = score

        if remaining["non_weapon"] > 0:
            score = 100
            if auto_label == "non_weapon":
                score = 0
            elif source in {"05_deepweb", "04_telegram_youtube", "03_google_scraped"}:
                score = 10
            elif auto_conf >= 0.75:
                score = 15
            class_score["non_weapon"] = score

        if remaining["ood"] > 0:
            score = 100
            if auto_conf < 0.60:
                score = 0
            elif score_margin < 0.10:
                score = 5
            elif source in {"05_deepweb", "04_telegram_youtube"}:
                score = 10
            class_score["ood"] = score

        preferred_class = min(class_score, key=lambda c: (class_score[c], -remaining[c]))
        return preferred_class

    def pending_priority_score(self, row: pd.Series) -> tuple:
        preferred_class = self.infer_preferred_class(row)

        pr = row.get("selection_source_priority", 999)
        try:
            pr = int(pr)
        except Exception:
            pr = 999

        auto_conf = to_float(row.get("auto_confidence", ""), -1.0)
        score_margin = to_float(row.get("score_margin", ""), 999.0)
        image_id = safe_str(row.get("image_id", ""))

        preference_rank = {
            "weapon": 0,
            "non_weapon": 1,
            "ood": 2,
        }

        return (
            preference_rank.get(preferred_class, 9),
            pr,
            score_margin,
            -auto_conf,
            image_id,
        )

    def get_indices_for_current_view(self) -> list[int]:
        status = self.df["selection_status"].map(norm)
        labels = self.df["selection_label"].map(norm)

        if self.view_mode == "reviewed":
            view_df = self.df[status == "reviewed"].copy()
            return list(view_df.index)

        if self.view_mode == "selected":
            view_df = self.df[labels.isin({"weapon", "non_weapon", "ood"})].copy()
            return list(view_df.index)

        if self.view_mode == "excluded":
            view_df = self.df[labels == "exclude"].copy()
            return list(view_df.index)

        view_df = self.df[status != "reviewed"].copy()
        if view_df.empty:
            return []

        view_df["_priority"] = view_df.apply(self.pending_priority_score, axis=1)
        view_df = view_df.sort_values("_priority", kind="stable")
        return list(view_df.index)

    def update_batch_indices(self, preserve_image_id: str | None = None) -> None:
        indices = self.get_indices_for_current_view()

        if not indices:
            self.batch_indices = []
            self.selected_pos = 0
            self.current_start = 0
            return

        if preserve_image_id:
            try:
                pos_in_view = next(
                    i for i, idx in enumerate(indices)
                    if str(self.df.loc[idx, "image_id"]) == preserve_image_id
                )
                self.current_start = (pos_in_view // BATCH_SIZE) * BATCH_SIZE
                self.batch_indices = indices[self.current_start:self.current_start + BATCH_SIZE]
                self.selected_pos = pos_in_view % BATCH_SIZE
                if self.selected_pos >= len(self.batch_indices):
                    self.selected_pos = max(0, len(self.batch_indices) - 1)
                return
            except StopIteration:
                pass

        if self.current_start >= len(indices):
            self.current_start = max(0, ((len(indices) - 1) // BATCH_SIZE) * BATCH_SIZE)

        self.batch_indices = indices[self.current_start:self.current_start + BATCH_SIZE]

        if self.selected_pos >= len(self.batch_indices):
            self.selected_pos = 0

    # -------------------------------------------------------------------------
    # Export outputs
    # -------------------------------------------------------------------------
    def export_outputs(self) -> None:
        labels = self.df["selection_label"].map(norm)

        final_df = self.df[labels.isin({"weapon", "non_weapon", "ood"})].copy()
        final_df = final_df.sort_values(["selection_label", "source_dataset", "image_id"], kind="stable").reset_index(drop=True)

        removed_df = self.df[labels == "exclude"].copy()
        removed_df = removed_df.sort_values(["source_dataset", "image_id"], kind="stable").reset_index(drop=True)

        adversarial_df = final_df[final_df["selection_label"].map(norm).isin({"weapon", "non_weapon"})].copy()
        adversarial_df = adversarial_df.sort_values(["selection_label", "source_dataset", "image_id"], kind="stable").reset_index(drop=True)

        safe_write_csv(final_df, OUT_FINAL_1500_CSV)
        safe_write_csv(removed_df, OUT_REMOVED_CSV)
        safe_write_csv(adversarial_df, OUT_ADVERSARIAL_SUBSET_CSV)

        counts = self.final_counts()
        rem = self.remaining_targets()

        summary = {
            "timestamp": now_iso(),
            "reviewer_id": self.reviewer_id,
            "input_manifest": str(PREPARED_MANIFEST_PATH),
            "work_db": str(WORK_DB_PATH),
            "outputs": {
                "final_1500_csv": str(OUT_FINAL_1500_CSV),
                "removed_csv": str(OUT_REMOVED_CSV),
                "adversarial_subset_csv": str(OUT_ADVERSARIAL_SUBSET_CSV),
                "log_csv": str(LOG_PATH),
                "state_json": str(STATE_PATH),
            },
            "counts": {
                "weapon": counts["weapon"],
                "non_weapon": counts["non_weapon"],
                "ood": counts["ood"],
                "exclude": counts["exclude"],
                "pending": self.count_pending(),
                "reviewed": self.count_reviewed(),
                "final_total": counts["weapon"] + counts["non_weapon"] + counts["ood"],
                "adversarial_subset_total": counts["weapon"] + counts["non_weapon"],
            },
            "remaining_targets": rem,
            "checks": {
                "final_1500_exact": counts["weapon"] == TARGET_WEAPON and counts["non_weapon"] == TARGET_NON_WEAPON and counts["ood"] == TARGET_OOD,
                "adversarial_subset_1000_exact": (counts["weapon"] + counts["non_weapon"]) == 1000,
            },
            "source_distribution": self.source_status().to_dict(orient="records"),
        }

        safe_write_json(summary, SUMMARY_PATH)

    def auto_save(self, last_action: str = "") -> None:
        safe_write_csv(self.df, WORK_DB_PATH)
        self.save_state(last_action=last_action)
        self.export_outputs()

    # -------------------------------------------------------------------------
    # Decision application
    # -------------------------------------------------------------------------
    def can_assign_label(self, label: str) -> tuple[bool, str]:
        rem = self.remaining_targets()

        if label == "weapon" and rem["weapon"] <= 0:
            return False, f"Target weapon già raggiunto ({TARGET_WEAPON})."
        if label == "non_weapon" and rem["non_weapon"] <= 0:
            return False, f"Target non_weapon già raggiunto ({TARGET_NON_WEAPON})."
        if label == "ood" and rem["ood"] <= 0:
            return False, f"Target ood già raggiunto ({TARGET_OOD})."

        return True, ""

    def set_label(self, df_index: int, label: str) -> None:
        if label not in VALID_FINAL_LABELS:
            return

        allowed, reason = self.can_assign_label(label) if label in {"weapon", "non_weapon", "ood"} else (True, "")
        if not allowed:
            print(f"[BLOCKED] {reason}")
            return

        ts = now_iso()

        prev_label = safe_str(self.df.loc[df_index, "selection_label"])
        image_id = safe_str(self.df.loc[df_index, "image_id"])

        self.last_action_stack.append(
            {
                "df_index": int(df_index),
                "image_id": image_id,
                "previous_label": prev_label,
                "previous_status": safe_str(self.df.loc[df_index, "selection_status"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "selection_label"] = label
        self.df.at[df_index, "selection_status"] = "reviewed"
        self.df.at[df_index, "selection_reviewer_id"] = self.reviewer_id
        self.df.at[df_index, "selection_timestamp"] = ts
        self.df.at[df_index, "selected_for_final"] = "yes" if label in {"weapon", "non_weapon", "ood"} else "no"

        append_session_log(
            {
                "timestamp": ts,
                "reviewer_id": self.reviewer_id,
                "image_id": image_id,
                "source_dataset": safe_str(self.df.loc[df_index, "source_dataset"]),
                "relative_path": safe_str(self.df.loc[df_index, "relative_path"]),
                "action": "assign_label",
                "previous_label": prev_label,
                "new_label": label,
                "auto_label": safe_str(self.df.loc[df_index, "auto_label"]),
                "auto_confidence": safe_str(self.df.loc[df_index, "auto_confidence"]),
                "score_margin": safe_str(self.df.loc[df_index, "score_margin"]),
                "selected_for_final": self.df.at[df_index, "selected_for_final"],
            }
        )

        self.auto_save(last_action=f"assign_{label}")
        self.draw_batch(preserve_image_id=image_id)

    def clear_label(self, df_index: int) -> None:
        ts = now_iso()
        prev_label = safe_str(self.df.loc[df_index, "selection_label"])
        image_id = safe_str(self.df.loc[df_index, "image_id"])

        self.last_action_stack.append(
            {
                "df_index": int(df_index),
                "image_id": image_id,
                "previous_label": prev_label,
                "previous_status": safe_str(self.df.loc[df_index, "selection_status"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "selection_label"] = ""
        self.df.at[df_index, "selection_status"] = "pending"
        self.df.at[df_index, "selection_reviewer_id"] = ""
        self.df.at[df_index, "selection_timestamp"] = ""
        self.df.at[df_index, "selected_for_final"] = ""

        append_session_log(
            {
                "timestamp": ts,
                "reviewer_id": self.reviewer_id,
                "image_id": image_id,
                "source_dataset": safe_str(self.df.loc[df_index, "source_dataset"]),
                "relative_path": safe_str(self.df.loc[df_index, "relative_path"]),
                "action": "clear_label",
                "previous_label": prev_label,
                "new_label": "",
                "auto_label": safe_str(self.df.loc[df_index, "auto_label"]),
                "auto_confidence": safe_str(self.df.loc[df_index, "auto_confidence"]),
                "score_margin": safe_str(self.df.loc[df_index, "score_margin"]),
                "selected_for_final": "",
            }
        )

        self.auto_save(last_action="clear_label")
        self.draw_batch(preserve_image_id=image_id)

    def undo_last_action(self) -> None:
        if not self.last_action_stack:
            print("[INFO] Nessuna azione recente da annullare.")
            return

        item = self.last_action_stack.pop()
        df_index = item["df_index"]
        image_id = item["image_id"]

        prev_label = item["previous_label"]
        prev_status = item["previous_status"]

        self.df.at[df_index, "selection_label"] = prev_label
        self.df.at[df_index, "selection_status"] = prev_status if prev_status else "pending"
        self.df.at[df_index, "selection_reviewer_id"] = self.reviewer_id if prev_label else ""
        self.df.at[df_index, "selection_timestamp"] = now_iso() if prev_label else ""
        self.df.at[df_index, "selected_for_final"] = "yes" if prev_label in {"weapon", "non_weapon", "ood"} else ("no" if prev_label == "exclude" else "")

        append_session_log(
            {
                "timestamp": now_iso(),
                "reviewer_id": self.reviewer_id,
                "image_id": image_id,
                "source_dataset": safe_str(self.df.loc[df_index, "source_dataset"]),
                "relative_path": safe_str(self.df.loc[df_index, "relative_path"]),
                "action": "undo_last_action",
                "previous_label": safe_str(self.df.loc[df_index, "selection_label"]),
                "new_label": prev_label,
                "auto_label": safe_str(self.df.loc[df_index, "auto_label"]),
                "auto_confidence": safe_str(self.df.loc[df_index, "auto_confidence"]),
                "score_margin": safe_str(self.df.loc[df_index, "score_margin"]),
                "selected_for_final": self.df.at[df_index, "selected_for_final"],
            }
        )

        self.auto_save(last_action="undo_last_action")
        self.draw_batch(preserve_image_id=image_id)
        print(f"[OK] Undo eseguito su {image_id}.")

    # -------------------------------------------------------------------------
    # Figure and display
    # -------------------------------------------------------------------------
    def title_color(self, label: str) -> str:
        return LABEL_COLORS.get(norm(label), "black")

    def open_help_window(self) -> None:
        if self.help_fig is not None:
            try:
                plt.figure(self.help_fig.number)
                self.help_fig.canvas.draw_idle()
                return
            except Exception:
                self.help_fig = None

        self.help_fig, ax = plt.subplots(figsize=(8, 10))
        self.help_fig.canvas.manager.set_window_title("Manual Selection Protocol Help")
        ax.axis("off")
        ax.text(0.02, 0.98, HELP_TEXT, va="top", ha="left", fontsize=11, family="monospace", wrap=True)
        self.help_fig.tight_layout()

    def open_summary_window(self) -> None:
        counts = self.final_counts()
        rem = self.remaining_targets()

        lines = [
            "OFFICIAL MANUAL SELECTION SUMMARY",
            "",
            f"weapon        : {counts['weapon']} / {TARGET_WEAPON}    remaining: {rem['weapon']}",
            f"non_weapon    : {counts['non_weapon']} / {TARGET_NON_WEAPON}    remaining: {rem['non_weapon']}",
            f"ood           : {counts['ood']} / {TARGET_OOD}    remaining: {rem['ood']}",
            f"exclude       : {counts['exclude']}",
            "",
            f"pending       : {self.count_pending()}",
            f"reviewed      : {self.count_reviewed()}",
            f"view_mode     : {self.view_mode}",
            "",
            "SOURCE STATUS",
            "",
        ]

        src_df = self.source_status()
        if not src_df.empty:
            for _, r in src_df.iterrows():
                lines.append(
                    f"{str(r['source_dataset']):<24} "
                    f"W={int(r['weapon']):>3} "
                    f"NW={int(r['non_weapon']):>3} "
                    f"OOD={int(r['ood']):>3} "
                    f"EX={int(r['exclude']):>3} "
                    f"P={int(r['pending']):>4} "
                    f"R={int(r['reviewed']):>4}"
                )

        text = "\n".join(lines)

        if self.summary_fig is None:
            self.summary_fig, ax = plt.subplots(figsize=(8, 10))
            self.summary_fig.canvas.manager.set_window_title("Manual Selection Summary")
        else:
            ax = self.summary_fig.axes[0]

        ax.clear()
        ax.axis("off")
        ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, family="monospace")
        self.summary_fig.tight_layout()
        self.summary_fig.canvas.draw_idle()

    def _init_main_figure_if_needed(self) -> None:
        if self.fig is None:
            self.fig, self.axes = plt.subplots(
                math.ceil(BATCH_SIZE / N_COLS),
                N_COLS,
                figsize=(FIG_W, FIG_H),
            )
            self.fig.canvas.manager.set_window_title("Official Manual Selection Protocol Reviewer")
            self.axes = self.axes.flatten()
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def draw_batch(self, preserve_image_id: str | None = None) -> None:
        self.update_batch_indices(preserve_image_id=preserve_image_id)
        self._init_main_figure_if_needed()

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        if not self.batch_indices:
            self.fig.suptitle("Official Manual Selection Protocol Reviewer [NO IMAGES IN CURRENT VIEW]", fontsize=12)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            self.fig.canvas.draw_idle()
            self.open_summary_window()
            return

        self.ax_to_df_index.clear()

        for i, df_index in enumerate(self.batch_indices):
            ax = self.axes[i]
            row = self.df.loc[df_index]

            image_path = resolve_image_path(row["relative_path"])
            if image_path is not None and image_path.exists():
                try:
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "ERR IMG", ha="center", va="center", fontsize=10)
            else:
                ax.text(0.5, 0.5, "IMG NOT FOUND", ha="center", va="center", fontsize=10)

            label = safe_str(row.get("selection_label", ""))
            src = safe_str(row.get("source_dataset", ""))
            img_id = safe_str(row.get("image_id", ""))
            auto_label = safe_str(row.get("auto_label", ""))
            auto_conf = safe_str(row.get("auto_confidence", ""))
            margin = safe_str(row.get("score_margin", ""))
            preferred_class = self.infer_preferred_class(row)
            status = safe_str(row.get("selection_status", ""))

            title = (
                f"{i+1}. {img_id}\n"
                f"{src}\n"
                f"auto={auto_label} | conf={auto_conf} | m={margin}\n"
                f"pref={preferred_class} | final={label or '-'} | {status}"
            )
            ax.set_title(title, fontsize=8, color=self.title_color(label))

            if i == self.selected_pos:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3.0)
                    spine.set_edgecolor("red")

            if label:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(max(spine.get_linewidth(), 1.5))
                    spine.set_edgecolor(self.title_color(label))

            self.ax_to_df_index[ax] = df_index

        indices = self.get_indices_for_current_view()
        total_pages = max(1, math.ceil(max(1, len(indices)) / BATCH_SIZE))
        current_page = (self.current_start // BATCH_SIZE) + 1

        self.fig.suptitle(
            f"Official Manual Selection Protocol Reviewer [{self.view_mode.upper()}] | page {current_page}/{total_pages}\n"
            f"Mouse: left=WEAPON right=NON_WEAPON middle=OOD | Keys: w n o e/x a u r h t s q enter zoom →/space next ←/backspace prev",
            fontsize=10,
        )
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        self.fig.canvas.draw_idle()
        self.open_summary_window()

    # -------------------------------------------------------------------------
    # Navigation and interaction
    # -------------------------------------------------------------------------
    def next_batch(self) -> None:
        indices = self.get_indices_for_current_view()
        if not indices:
            return
        self.current_start = min(
            self.current_start + BATCH_SIZE,
            max(0, ((len(indices) - 1) // BATCH_SIZE) * BATCH_SIZE),
        )
        self.selected_pos = 0
        self.save_state(last_action="next_batch")
        self.draw_batch()

    def prev_batch(self) -> None:
        self.current_start = max(0, self.current_start - BATCH_SIZE)
        self.selected_pos = 0
        self.save_state(last_action="prev_batch")
        self.draw_batch()

    def toggle_view_mode(self) -> None:
        current_order = ["pending", "reviewed", "selected", "excluded"]
        try:
            idx = current_order.index(self.view_mode)
        except ValueError:
            idx = 0
        self.view_mode = current_order[(idx + 1) % len(current_order)]
        self.current_start = 0
        self.selected_pos = 0
        self.save_state(last_action=f"toggle_view_{self.view_mode}")
        self.draw_batch()
        print(f"[OK] View mode: {self.view_mode}")

    def open_zoom(self) -> None:
        if not self.batch_indices:
            return
        if self.selected_pos < 0 or self.selected_pos >= len(self.batch_indices):
            return

        df_index = self.batch_indices[self.selected_pos]
        row = self.df.loc[df_index]
        image_path = resolve_image_path(row["relative_path"])
        if image_path is None or not image_path.exists():
            return

        try:
            img = mpimg.imread(image_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(
                f"{row.get('image_id', '')} | {row.get('source_dataset', '')}\n"
                f"auto={row.get('auto_label', '')} | conf={row.get('auto_confidence', '')} | "
                f"margin={row.get('score_margin', '')} | final={row.get('selection_label', '') or '-'}"
            )
            plt.axis("off")
            plt.show()
        except Exception as exc:
            print(f"[WARN] Zoom failed: {exc}")

    def on_click(self, event) -> None:
        if event.inaxes not in self.ax_to_df_index:
            return

        df_index = self.ax_to_df_index[event.inaxes]
        if df_index in self.batch_indices:
            self.selected_pos = self.batch_indices.index(df_index)

        button = event.button
        if button == MouseButton.LEFT or button == 1:
            self.set_label(df_index, "weapon")
            return
        if button == MouseButton.RIGHT or button == 3:
            self.set_label(df_index, "non_weapon")
            return
        if button == MouseButton.MIDDLE or button == 2:
            self.set_label(df_index, "ood")
            return

        self.save_state(last_action="select_click")
        self.draw_batch()

    def on_key(self, event) -> None:
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
        if key == "s":
            self.auto_save(last_action="manual_save")
            print("[OK] Saved.")
            return
        if key == "q":
            self.auto_save(last_action="quit")
            print("[OK] Saved. Exiting.")
            plt.close(self.fig)
            return
        if key == "r":
            self.toggle_view_mode()
            return
        if key == "h":
            self.open_help_window()
            return
        if key == "t":
            self.open_summary_window()
            return
        if key == "u":
            self.undo_last_action()
            return

        if key.isdigit():
            pos = int(key) - 1
            if 0 <= pos < len(self.batch_indices):
                self.selected_pos = pos
                self.save_state(last_action=f"select_{key}")
                self.draw_batch()
            return

        if not self.batch_indices:
            return

        if self.selected_pos < 0 or self.selected_pos >= len(self.batch_indices):
            self.selected_pos = 0

        df_index = self.batch_indices[self.selected_pos]

        if key == "w":
            self.set_label(df_index, "weapon")
            return
        if key == "n":
            self.set_label(df_index, "non_weapon")
            return
        if key == "o":
            self.set_label(df_index, "ood")
            return
        if key in {"e", "x"}:
            self.set_label(df_index, "exclude")
            return
        if key == "a":
            self.clear_label(df_index)
            return

    def run(self) -> None:
        self.draw_batch()
        self.open_help_window()
        self.open_summary_window()
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    reviewer = ManualSelectionProtocolReviewer()
    reviewer.run()


if __name__ == "__main__":
    main()