#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_manual_selection_protocol_reviewer.py

Official manual selection protocol reviewer for the FAIR-Lab public thesis pipeline.

Purpose
-------
This script provides a documented, human-in-the-loop reviewer used to inspect,
refine, and extend the final manual image selection process according to the
thesis methodology.

The reviewer supports two explicit operating modes:

1. review_selection
   - inspect already assigned images from one selected class
   - remove the current class assignment and send the image back to pending

2. review_pending
   - inspect pending images from one selected source dataset
   - assign weapon / non_weapon / ood / exclude manually

Input
-----
- datasets/prepared/manifests/review_manifest_full.csv
- datasets/final/manifests/manual_selection_protocol_db.csv

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
- All reviewer actions are logged.
- The script can be re-opened and resumed.
- Global targets are enforced for weapon / non_weapon / ood.
- Existing assigned classes can be explicitly reviewed and cleared.
- Pending images can be reviewed source by source.
- No automatic scoring or automatic class suggestion is used in the reviewer UI.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import MouseButton

from datasets.scripts.utils.paths import PREPARED_DATASETS_DIR


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
VALID_SELECTION_CLASSES = {"weapon", "non_weapon", "ood"}
VALID_SESSION_MODES = {"review_pending", "review_selection"}

BATCH_SIZE = 10
N_COLS = 5
FIG_W = 18
FIG_H = 9

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

SESSION MODES
- review_selection
  Inspect already assigned images from one selected class.
  Main action: remove current assignment and return the image to pending.

- review_pending
  Inspect pending images from one selected source dataset.
  Main actions: assign weapon / non_weapon / ood / exclude.

GLOBAL TARGETS
- weapon      : 500
- non_weapon  : 500
- ood         : 500

PENDING MODE MOUSE
- Left click   = WEAPON
- Right click  = NON_WEAPON
- Middle click = OOD

KEYS
- w = WEAPON               [pending mode only]
- n = NON_WEAPON           [pending mode only]
- o = OOD                  [pending mode only]
- e / x = EXCLUDE          [pending mode only]
- a = clear pending decision / remove current assignment
- u = undo last action
- m = change session mode / filter
- s = save
- q = save + quit
- h = help
- t = summary
- Enter = zoom selected image
- Right / Space = next batch
- Left / Backspace = previous batch
- 1..9 = select image 1..9
- 0 = select image 10

NOTES
- review_selection shows only one already assigned class at a time.
- review_pending shows only pending images from one selected source dataset.
- All decisions are logged and exports are regenerated automatically.
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
        backup_path = BACKUP_DIR / (
            f"{path.stem}_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{path.suffix}"
        )
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
                "session_mode",
                "review_class",
                "review_source",
                "image_id",
                "source_dataset",
                "relative_path",
                "action",
                "previous_label",
                "new_label",
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

    return df[protocol_cols].copy()


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

        self.session_mode = "review_pending"
        self.review_class = ""
        self.review_source = ""

        self.load_state()

    # -------------------------------------------------------------------------
    # Session mode and filters
    # -------------------------------------------------------------------------
    def ask_review_class(self, default: str = "weapon") -> str:
        raw = input(
            f"Class to review [weapon/non_weapon/ood] (default: {default}): "
        ).strip().lower() or default
        if raw not in VALID_SELECTION_CLASSES:
            print(f"[WARN] Invalid class: {raw}. Using default: {default}")
            return default
        return raw

    def available_sources(self) -> list[str]:
        if "source_dataset" not in self.df.columns:
            return []
        return sorted(
            [s for s in self.df["source_dataset"].dropna().astype(str).unique().tolist() if s.strip()]
        )

    def ask_review_source(self, default: str | None = None) -> str:
        sources = self.available_sources()
        if not sources:
            raise ValueError("No source_dataset values available in the protocol DB.")

        print("\nAvailable source datasets:")
        for i, src in enumerate(sources, start=1):
            print(f"{i}. {src}")

        prompt_default = default or sources[0]
        raw = input(
            f"Source to review [name or index] (default: {prompt_default}): "
        ).strip()

        if raw == "":
            return prompt_default

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(sources):
                return sources[idx]

        if raw in sources:
            return raw

        print(f"[WARN] Invalid source: {raw}. Using default: {prompt_default}")
        return prompt_default

    def choose_review_session(self) -> None:
        self.print_status()

        raw = input(
            "\nChoose workflow [review_pending/review_selection] "
            "(default: review_pending): "
        ).strip().lower() or "review_pending"

        if raw not in {"review_pending", "review_selection"}:
            print(f"[WARN] Invalid workflow: {raw}. Using review_pending.")
            raw = "review_pending"

        if raw == "review_pending":
            self.session_mode = "review_pending"
            self.review_class = ""
            self.review_source = self.ask_review_source(default=self.review_source or None)
        else:
            self.session_mode = "review_selection"
            self.review_source = ""
            self.review_class = self.ask_review_class(default=self.review_class or "weapon")

        self.current_start = 0
        self.selected_pos = 0

        print(
            f"\n[OK] session_mode={self.session_mode} | "
            f"review_class={self.review_class or '-'} | "
            f"review_source={self.review_source or '-'}"
        )

    def change_review_class(self) -> None:
        """
        Change only the currently reviewed assigned class inside review_selection mode.
        """
        if self.session_mode != "review_selection":
            print("[INFO] change_review_class() is available only in review_selection mode.")
            return

        new_class = self.ask_review_class(default=self.review_class or "weapon")
        self.review_class = new_class
        self.current_start = 0
        self.selected_pos = 0
        self.save_state(last_action=f"change_review_class_{new_class}")
        self.draw_batch()
        print(f"[OK] review_class={self.review_class}")


    def change_session_mode(self) -> None:
        self.choose_review_session()
        self.save_state(last_action=f"change_session_mode_{self.session_mode}")
        self.draw_batch()

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
            return pd.DataFrame(
                columns=["source_dataset", "rows", "weapon", "non_weapon", "ood", "exclude", "pending", "reviewed"]
            )
        return pd.DataFrame(rows).sort_values("source_dataset").reset_index(drop=True)

    def current_subset_status(self) -> dict[str, Any]:
        if self.session_mode == "review_selection":
            tmp = self.df[self.df["selection_label"].map(norm) == self.review_class].copy()
            return {
                "session_mode": self.session_mode,
                "review_class": self.review_class,
                "rows": int(len(tmp)),
            }

        tmp = self.df[
            (self.df["selection_status"].map(norm) != "reviewed")
            & (self.df["source_dataset"].astype(str) == self.review_source)
        ].copy()
        return {
            "session_mode": self.session_mode,
            "review_source": self.review_source,
            "rows": int(len(tmp)),
        }

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
            "last_action_stack": self.last_action_stack[-20:],
            "session_mode": self.session_mode,
            "review_class": self.review_class,
            "review_source": self.review_source,
        }
        safe_write_json(state, STATE_PATH)

    def load_state(self) -> None:
        if not STATE_PATH.exists():
            return

        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            print("\nLast saved state:")
            print(json.dumps(state, indent=2, ensure_ascii=False))

            self.current_start = int(state.get("current_start", 0) or 0)
            self.selected_pos = 0
            self.session_mode = state.get("session_mode", "review_pending") or "review_pending"
            self.review_class = state.get("review_class", "") or ""
            self.review_source = state.get("review_source", "") or ""

            raw_stack = state.get("last_action_stack", [])
            if isinstance(raw_stack, list):
                self.last_action_stack = raw_stack
        except Exception as exc:
            print(f"[WARN] Could not load previous state: {exc}")

    # -------------------------------------------------------------------------
    # Subset construction
    # -------------------------------------------------------------------------
    def get_indices_for_current_view(self) -> list[int]:
        if self.session_mode == "review_selection":
            tmp = self.df[self.df["selection_label"].map(norm) == self.review_class].copy()
            if tmp.empty:
                return []

            tmp["_source_priority"] = pd.to_numeric(
                tmp["selection_source_priority"], errors="coerce"
            ).fillna(999).astype(int)

            tmp = tmp.sort_values(
                ["_source_priority", "source_dataset", "image_id"],
                kind="stable",
            )
            return list(tmp.index)

        tmp = self.df[
            (self.df["selection_status"].map(norm) != "reviewed")
            & (self.df["source_dataset"].astype(str) == self.review_source)
        ].copy()

        if tmp.empty:
            return []

        tmp["_source_priority"] = pd.to_numeric(
            tmp["selection_source_priority"], errors="coerce"
        ).fillna(999).astype(int)

        tmp = tmp.sort_values(
            ["_source_priority", "source_dataset", "image_id"],
            kind="stable",
        )
        return list(tmp.index)

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
        final_df = final_df.sort_values(
            ["selection_label", "source_dataset", "image_id"], kind="stable"
        ).reset_index(drop=True)

        removed_df = self.df[labels == "exclude"].copy()
        removed_df = removed_df.sort_values(
            ["source_dataset", "image_id"], kind="stable"
        ).reset_index(drop=True)

        adversarial_df = final_df[
            final_df["selection_label"].map(norm).isin({"weapon", "non_weapon"})
        ].copy()
        adversarial_df = adversarial_df.sort_values(
            ["selection_label", "source_dataset", "image_id"], kind="stable"
        ).reset_index(drop=True)

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
            "session_mode": self.session_mode,
            "review_class": self.review_class,
            "review_source": self.review_source,
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
                "final_1500_exact": (
                    counts["weapon"] == TARGET_WEAPON
                    and counts["non_weapon"] == TARGET_NON_WEAPON
                    and counts["ood"] == TARGET_OOD
                ),
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
            return False, f"Weapon target already reached ({TARGET_WEAPON})."
        if label == "non_weapon" and rem["non_weapon"] <= 0:
            return False, f"Non-weapon target already reached ({TARGET_NON_WEAPON})."
        if label == "ood" and rem["ood"] <= 0:
            return False, f"OOD target already reached ({TARGET_OOD})."

        return True, ""

    def log_action(self, df_index: int, action: str, previous_label: str, new_label: str) -> None:
        append_session_log(
            {
                "timestamp": now_iso(),
                "reviewer_id": self.reviewer_id,
                "session_mode": self.session_mode,
                "review_class": self.review_class,
                "review_source": self.review_source,
                "image_id": safe_str(self.df.loc[df_index, "image_id"]),
                "source_dataset": safe_str(self.df.loc[df_index, "source_dataset"]),
                "relative_path": safe_str(self.df.loc[df_index, "relative_path"]),
                "action": action,
                "previous_label": previous_label,
                "new_label": new_label,
                "selected_for_final": safe_str(self.df.loc[df_index, "selected_for_final"]),
            }
        )

    def set_label(self, df_index: int, label: str) -> None:
        if self.session_mode != "review_pending":
            print("[INFO] Label assignment is available only in review_pending mode.")
            return

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
                "previous_selected_for_final": safe_str(self.df.loc[df_index, "selected_for_final"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "selection_label"] = label
        self.df.at[df_index, "selection_status"] = "reviewed"
        self.df.at[df_index, "selection_reviewer_id"] = self.reviewer_id
        self.df.at[df_index, "selection_timestamp"] = ts
        self.df.at[df_index, "selected_for_final"] = "yes" if label in {"weapon", "non_weapon", "ood"} else "no"

        self.log_action(df_index, "assign_label", prev_label, label)
        self.auto_save(last_action=f"assign_{label}")
        self.draw_batch(preserve_image_id=image_id)

    def clear_pending_decision(self, df_index: int) -> None:
        if self.session_mode != "review_pending":
            print("[INFO] This action is available only in review_pending mode.")
            return

        prev_label = safe_str(self.df.loc[df_index, "selection_label"])
        image_id = safe_str(self.df.loc[df_index, "image_id"])

        self.last_action_stack.append(
            {
                "df_index": int(df_index),
                "image_id": image_id,
                "previous_label": prev_label,
                "previous_status": safe_str(self.df.loc[df_index, "selection_status"]),
                "previous_selected_for_final": safe_str(self.df.loc[df_index, "selected_for_final"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "selection_label"] = ""
        self.df.at[df_index, "selection_status"] = "pending"
        self.df.at[df_index, "selection_reviewer_id"] = ""
        self.df.at[df_index, "selection_timestamp"] = ""
        self.df.at[df_index, "selected_for_final"] = ""

        self.log_action(df_index, "clear_pending_decision", prev_label, "")
        self.auto_save(last_action="clear_pending_decision")
        self.draw_batch(preserve_image_id=image_id)

    def remove_existing_assignment(self, df_index: int) -> None:
        if self.session_mode != "review_selection":
            print("[INFO] This action is available only in review_selection mode.")
            return

        prev_label = safe_str(self.df.loc[df_index, "selection_label"])
        image_id = safe_str(self.df.loc[df_index, "image_id"])

        if norm(prev_label) not in VALID_SELECTION_CLASSES:
            print("[INFO] The selected image does not currently belong to a reviewable class.")
            return

        self.last_action_stack.append(
            {
                "df_index": int(df_index),
                "image_id": image_id,
                "previous_label": prev_label,
                "previous_status": safe_str(self.df.loc[df_index, "selection_status"]),
                "previous_selected_for_final": safe_str(self.df.loc[df_index, "selected_for_final"]),
            }
        )
        self.last_action_stack = self.last_action_stack[-100:]

        self.df.at[df_index, "selection_label"] = ""
        self.df.at[df_index, "selection_status"] = "pending"
        self.df.at[df_index, "selection_reviewer_id"] = ""
        self.df.at[df_index, "selection_timestamp"] = ""
        self.df.at[df_index, "selected_for_final"] = ""

        self.log_action(df_index, "remove_existing_assignment", prev_label, "")
        self.auto_save(last_action="remove_existing_assignment")
        self.draw_batch(preserve_image_id=image_id)

    def undo_last_action(self) -> None:
        if not self.last_action_stack:
            print("[INFO] No recent action to undo.")
            return

        item = self.last_action_stack.pop()
        df_index = item["df_index"]
        image_id = item["image_id"]

        current_label = safe_str(self.df.loc[df_index, "selection_label"])
        previous_label = item["previous_label"]
        previous_status = item["previous_status"]
        previous_selected_for_final = item["previous_selected_for_final"]

        self.df.at[df_index, "selection_label"] = previous_label
        self.df.at[df_index, "selection_status"] = previous_status if previous_status else "pending"
        self.df.at[df_index, "selected_for_final"] = previous_selected_for_final

        if previous_label:
            self.df.at[df_index, "selection_reviewer_id"] = self.reviewer_id
            self.df.at[df_index, "selection_timestamp"] = now_iso()
        else:
            self.df.at[df_index, "selection_reviewer_id"] = ""
            self.df.at[df_index, "selection_timestamp"] = ""

        self.log_action(df_index, "undo_last_action", current_label, previous_label)
        self.auto_save(last_action="undo_last_action")
        self.draw_batch(preserve_image_id=image_id)
        print(f"[OK] Undo executed on {image_id}.")

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
        subset = self.current_subset_status()

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
            "",
            f"session_mode  : {self.session_mode}",
            f"review_class  : {self.review_class or '-'}",
            f"review_source : {self.review_source or '-'}",
            f"subset_rows   : {subset['rows']}",
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

            try:
                manager = self.fig.canvas.manager
                window = manager.window
                window.geometry("1600x950+50+50")
                window.attributes("-topmost", True)
                window.update()
                window.attributes("-topmost", False)
                window.deiconify()
                window.lift()
                window.focus_force()
            except Exception as exc:
                print(f"[WARN] Could not force window geometry/visibility: {exc}", flush=True)

            self.axes = self.axes.flatten()
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

            try:
                self.fig.canvas.draw()
            except Exception as exc:
                print(f"[WARN] Could not draw main figure: {exc}", flush=True)

    def draw_batch(self, preserve_image_id: str | None = None) -> None:
        self.update_batch_indices(preserve_image_id=preserve_image_id)
        self._init_main_figure_if_needed()

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        if not self.batch_indices:
            mode_title = (
                f"review_selection:{self.review_class}"
                if self.session_mode == "review_selection"
                else f"review_pending:{self.review_source}"
            )
            self.fig.suptitle(
                f"Official Manual Selection Protocol Reviewer [{mode_title}] [NO IMAGES IN CURRENT VIEW]",
                fontsize=12,
            )
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            self.fig.canvas.draw_idle()
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

            if self.session_mode == "review_pending":
                title = (
                    f"{i+1}. {img_id}\n"
                    f"{src}\n"
                    f"final={label or '-'} | status=pending"
                )
            else:
                title = (
                    f"{i + 1}. {img_id}\n"
                    f"{src}\n"
                    f"assigned={label or '-'}\n"
                    f"mouse: L=remove R=undo M=change-class"
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

        if self.session_mode == "review_selection":
            mode_txt = f"review_selection | class={self.review_class}"
            actions_txt = "Mouse: L=remove R=undo M=change-class | Keys: a=remove u=undo c=change-class m=change-workflow"
        else:
            mode_txt = f"review_pending | source={self.review_source}"
            actions_txt = "Mouse: left=weapon right=non_weapon middle=ood | Keys: w n o e/x a=clear u=undo m=change-mode"

        self.fig.suptitle(
            f"Official Manual Selection Protocol Reviewer [{mode_txt}] | page {current_page}/{total_pages}\n"
            f"{actions_txt} | 1..9 select | 0=10th | enter=zoom →/space next ←/backspace prev",
            fontsize=10,
        )

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        self.fig.canvas.draw_idle()

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
                f"final={row.get('selection_label', '') or '-'}"
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

        if self.session_mode == "review_pending":
            if button == MouseButton.LEFT or button == 1:
                self.set_label(df_index, "weapon")
                return
            if button == MouseButton.RIGHT or button == 3:
                self.set_label(df_index, "non_weapon")
                return
            if button == MouseButton.MIDDLE or button == 2:
                self.set_label(df_index, "ood")
                return

        if self.session_mode == "review_selection":
            if button == MouseButton.LEFT or button == 1:
                self.remove_existing_assignment(df_index)
                return
            if button == MouseButton.RIGHT or button == 3:
                self.undo_last_action()
                return
            if button == MouseButton.MIDDLE or button == 2:
                self.change_review_class()
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
        if key == "h":
            self.open_help_window()
            return
        if key == "t":
            self.open_summary_window()
            return
        if key == "u":
            self.undo_last_action()
            return
        if key == "m":
            self.change_session_mode()
            return

        if key.isdigit():
            if key == "0":
                pos = 9
            else:
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

        if self.session_mode == "review_pending":
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
                self.clear_pending_decision(df_index)
                return

        if self.session_mode == "review_selection":
            if key == "a":
                self.remove_existing_assignment(df_index)
                return
            if key == "c":
                self.change_review_class()
                return

    def run(self) -> None:
        self.choose_review_session()
        self.draw_batch()
        plt.show(block=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    reviewer = ManualSelectionProtocolReviewer()
    reviewer.run()


if __name__ == "__main__":
    main()