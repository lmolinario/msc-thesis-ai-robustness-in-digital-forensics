#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanitized interactive launcher for the FAIR-Lab XAI entry point."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINT = "explainability/scripts/17_generate_integrated_gradients_case_studies.py"
LOG_PATH = REPO_ROOT / "explainability/logs/xai_interactive_launcher_commands.jsonl"

PRESETS = {
    "1": {
        "name": "Chapter 5 manual selection and adaptive IG regeneration",
        "args": [
            "--model", "efficientnet_b0", "--strategy", "chapter5_core",
            "--max-cases", "500", "--cases-per-bucket", "3",
            "--manual-review", "--generate-after-manual",
            "--n-steps", "64", "--max-n-steps", "256",
            "--convergence-threshold", "0.05",
            "--attribution-target", "predicted_label",
            "--top-percentile", "90", "--device", "auto",
            "--input-size", "224", "--output-tag", "chapter5_validated",
            "--force", "--verbose",
        ],
    },
    "2": {"name": "Regenerate from a manual selection manifest", "manifest": True},
    "3": {
        "name": "Automatic weapon-to-non-weapon failures",
        "args": [
            "--model", "efficientnet_b0", "--strategy", "weapon_to_non_weapon",
            "--max-cases", "30", "--n-steps", "64", "--max-n-steps", "256",
            "--convergence-threshold", "0.05", "--attribution-target", "predicted_label",
            "--device", "auto", "--force",
        ],
    },
    "4": {
        "name": "Automatic high-confidence OOD cases",
        "args": [
            "--model", "efficientnet_b0", "--strategy", "ood_high_confidence",
            "--max-cases", "30", "--n-steps", "64", "--max-n-steps", "256",
            "--convergence-threshold", "0.05", "--attribution-target", "predicted_label",
            "--device", "auto", "--force",
        ],
    },
}


def log_command(preset: str, args: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": preset,
        "script": ENTRY_POINT,
        "arguments": args,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    print("FAIR-Lab XAI launcher")
    for key, preset in PRESETS.items():
        print(f"{key}. {preset['name']}")
    print("0. Exit")
    choice = input("Select workflow: ").strip()
    if choice == "0":
        return
    if choice not in PRESETS:
        raise SystemExit(f"Invalid workflow: {choice}")
    preset = PRESETS[choice]
    if preset.get("manifest"):
        manifest = input("Manual selection manifest path: ").strip()
        if not manifest:
            raise SystemExit("A manifest path is required")
        args = [
            "--selection-manifest", manifest,
            "--model", "efficientnet_b0", "--strategy", "chapter5_core",
            "--n-steps", "64", "--max-n-steps", "256",
            "--convergence-threshold", "0.05",
            "--attribution-target", "predicted_label", "--device", "auto", "--force",
        ]
    else:
        args = list(preset["args"])
    command = [sys.executable, str(REPO_ROOT / ENTRY_POINT), *args]
    print("Command:")
    print(" ".join(shlex.quote(part) for part in [sys.executable, ENTRY_POINT, *args]))
    log_command(preset["name"], args)
    if input("Execute now? [Y/n]: ").strip().lower() in {"n", "no"}:
        return
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
