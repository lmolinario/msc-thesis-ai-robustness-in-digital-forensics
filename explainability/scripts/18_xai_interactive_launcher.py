#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18_xai_interactive_launcher.py

Interactive terminal launcher for FAIR-Lab Integrated Gradients case studies.

Workflow summary
----------------
1. Auto XAI: weapon -> non_weapon | true-label attribution.
2. Auto XAI: weapon -> non_weapon | true + predicted attribution.
3. Auto XAI: high-confidence OOD | predicted-label attribution.
4. Manual XAI: one attack -> select N cases -> generate IG.
5. Manual XAI: all attacks -> select N cases per attack -> generate IG.
6. Manual XAI: all attacks -> select N cases per attack -> save manifest only.
7. Generate IG from existing manual selection manifest.
8. Custom guided run.

Notes
-----
- candidate_limit = 0 means show all candidates for each reviewed attack.
- cases_per_attack defines how many cases should be selected, not how many
  candidates should be displayed.
- The launcher only builds and executes commands for the official XAI script.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
XAI_SCRIPT = REPO_ROOT / "explainability" / "scripts" / "17_generate_integrated_gradients_case_studies.py"
LOG_DIR = REPO_ROOT / "explainability" / "logs"
COMMAND_LOG_PATH = LOG_DIR / "xai_interactive_launcher_commands.jsonl"

SUPPORTED_MODELS = ["efficientnet_b0", "resnet18", "clip"]
SUPPORTED_STRATEGIES = [
    "weapon_to_non_weapon",
    "ood_high_confidence",
    "perturbed_failures",
    "attack_stratified",
    "all",
]
SUPPORTED_ATTRIBUTION_TARGETS = ["true_label", "predicted_label", "both"]
SUPPORTED_DEVICES = ["auto", "cpu", "cuda"]

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

DEFAULT_SELECTION_MANIFEST = (
    "explainability/manifests/"
    "xai_manual_selection_db__efficientnet_b0_attack_stratified_per_attack_3_manual_target_both.csv"
)

PRESETS: dict[str, dict[str, Any]] = {
    "1": {
        "name": "Auto XAI: weapon -> non_weapon | true-label attribution",
        "description": "Automatic selection of critical weapon -> non_weapon failures.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "weapon_to_non_weapon",
            "max_cases": 30,
            "n_steps": 32,
            "attribution_target": "true_label",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "force": True,
            "verbose": True,
        },
    },
    "2": {
        "name": "Auto XAI: weapon -> non_weapon | true + predicted attribution",
        "description": "Automatic selection of critical failures with both attribution targets.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "weapon_to_non_weapon",
            "max_cases": 30,
            "n_steps": 32,
            "attribution_target": "both",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "force": True,
            "verbose": True,
        },
    },
    "3": {
        "name": "Auto XAI: high-confidence OOD | predicted-label attribution",
        "description": "Automatic selection of high-confidence OOD cases.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "ood_high_confidence",
            "max_cases": 30,
            "n_steps": 32,
            "attribution_target": "predicted_label",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "force": True,
            "verbose": True,
        },
    },
    "4": {
        "name": "Manual XAI: one attack -> select N cases -> generate IG",
        "description": "Review one selected attack_name and generate IG for manually selected cases.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "attack_stratified",
            "max_cases": 3,
            "cases_per_attack": 3,
            "candidate_limit": 0,
            "n_steps": 32,
            "attribution_target": "both",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "manual_review": True,
            "generate_after_manual": True,
            "force": True,
            "verbose": True,
        },
    },
    "5": {
        "name": "Manual XAI: all attacks -> select N cases per attack -> generate IG",
        "description": "Review all attacks sequentially and generate IG at the end.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "attack_stratified",
            "max_cases": 30,
            "cases_per_attack": 3,
            "candidate_limit": 0,
            "attack_name": DEFAULT_ATTACKS.copy(),
            "n_steps": 32,
            "attribution_target": "both",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "manual_review": True,
            "generate_after_manual": True,
            "force": True,
            "verbose": True,
        },
    },
    "6": {
        "name": "Manual XAI: all attacks -> select N cases per attack -> save manifest only",
        "description": "Review all attacks sequentially and save only the manual selection manifest.",
        "args": {
            "model": ["efficientnet_b0"],
            "strategy": "attack_stratified",
            "max_cases": 30,
            "cases_per_attack": 3,
            "candidate_limit": 0,
            "attack_name": DEFAULT_ATTACKS.copy(),
            "n_steps": 32,
            "attribution_target": "both",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "manual_review": True,
            "manual_only": True,
            "force": True,
            "verbose": True,
        },
    },
    "7": {
        "name": "Generate IG from existing manual selection manifest",
        "description": "Generate IG only for rows already marked as selected in a saved manifest.",
        "args": {
            "selection_manifest": "ASK",
            "model": ["efficientnet_b0"],
            "strategy": "attack_stratified",
            "n_steps": 32,
            "attribution_target": "both",
            "top_percentile": 90,
            "high_confidence_threshold": 0.90,
            "device": "auto",
            "input_size": 224,
            "force": True,
            "verbose": True,
        },
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ask_string(prompt: str, default: str = "") -> str:
    if default:
        raw = input(f"{prompt} [default: {default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [default: {default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        print(f"[WARN] Invalid integer value: {raw}. Using default: {default}.")
        return default


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [default: {default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid float value: {raw}. Using default: {default}.")
        return default


def ask_bool(prompt: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{default_text}]: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes", "s", "si", "sì", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False
    print(f"[WARN] Invalid boolean value: {raw}. Using default: {default}.")
    return default


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, start=1):
        marker = "*" if choice == default else " "
        print(f"{i}. {choice} {marker}")
    raw = input(f"Select [default: {default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(choices):
            return choices[index]
    if raw in choices:
        return raw
    print(f"[WARN] Invalid choice: {raw}. Using default: {default}.")
    return default


def ask_multi_choice(prompt: str, choices: list[str], default: list[str]) -> list[str]:
    print(f"\n{prompt}")
    print("Insert comma-separated indexes or names. Leave empty for default.")
    print("Use 'all' for all values, 'none' for no filter.")
    default_set = set(default)
    for i, choice in enumerate(choices, start=1):
        marker = "*" if choice in default_set else " "
        print(f"{i}. {choice} {marker}")
    raw = input("Selection: ").strip()
    if not raw:
        return default
    if raw.lower() == "all":
        return choices.copy()
    if raw.lower() == "none":
        return []
    selected: list[str] = []
    for token in [x.strip() for x in raw.split(",") if x.strip()]:
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(choices):
                selected.append(choices[index])
            else:
                print(f"[WARN] Ignoring invalid index: {token}")
        elif token in choices:
            selected.append(token)
        else:
            print(f"[WARN] Ignoring invalid value: {token}")
    unique_selected = []
    for item in selected:
        if item not in unique_selected:
            unique_selected.append(item)
    return unique_selected


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in command)


def repo_relative_or_raw(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        return path_value.replace("\\", "/")
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def log_command(command: list[str], preset_name: str, args_payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now_iso(),
        "preset": preset_name,
        "repo_root": str(REPO_ROOT),
        "command": command,
        "command_string": command_to_string(command),
        "args": args_payload,
    }
    with COMMAND_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_header() -> None:
    print("\n" * 3)
    print("=" * 80)
    print("FAIR-LAB XAI INTERACTIVE LAUNCHER")
    print("=" * 80)
    print(f"Repository root : {REPO_ROOT}")
    print(f"XAI script      : {XAI_SCRIPT}")
    print(f"Command log     : {COMMAND_LOG_PATH}")
    print("=" * 80)


def ensure_xai_script_exists() -> None:
    if not XAI_SCRIPT.exists():
        raise FileNotFoundError(f"Official XAI script not found: {XAI_SCRIPT}")


def apply_mandatory_preset_questions(args: dict[str, Any], preset_key: str) -> dict[str, Any]:
    updated = dict(args)
    if preset_key == "4":
        attack_name = ask_choice("Select the attack_name to review", DEFAULT_ATTACKS, "sigma_zero")
        requested_cases = ask_int(f"How many cases do you want to select for {attack_name}?", int(updated.get("cases_per_attack", 3)))
        updated["attack_name"] = [attack_name]
        updated["cases_per_attack"] = requested_cases
        updated["max_cases"] = requested_cases
        updated["candidate_limit"] = ask_int("Candidate limit for this attack (0 = show all candidates)", int(updated.get("candidate_limit", 0)))
        updated["output_tag"] = updated.get("output_tag", "") or f"manual_{attack_name}_{requested_cases}_cases"
    elif preset_key in {"5", "6"}:
        requested_cases = ask_int("How many cases do you want to select per attack?", int(updated.get("cases_per_attack", 3)))
        updated["cases_per_attack"] = requested_cases
        updated["max_cases"] = requested_cases * len(DEFAULT_ATTACKS)
        updated["candidate_limit"] = ask_int("Candidate limit per attack (0 = show all candidates)", int(updated.get("candidate_limit", 0)))
        updated["attack_name"] = DEFAULT_ATTACKS.copy()
        mode_tag = "generate_ig" if preset_key == "5" else "manifest_only"
        updated["output_tag"] = updated.get("output_tag", "") or f"manual_all_attacks_{requested_cases}_cases_{mode_tag}"
    return updated


def build_command(args: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(XAI_SCRIPT)]
    selection_manifest = args.get("selection_manifest", "")
    if selection_manifest:
        command.extend(["--selection-manifest", repo_relative_or_raw(selection_manifest)])
    else:
        command.extend(["--model", *args.get("model", ["efficientnet_b0"])])
        command.extend(["--strategy", args.get("strategy", "all")])
        if "max_cases" in args:
            command.extend(["--max-cases", str(args["max_cases"])])
        if "cases_per_attack" in args:
            command.extend(["--cases-per-attack", str(args["cases_per_attack"])])
        if "candidate_limit" in args:
            command.extend(["--candidate-limit", str(args["candidate_limit"])])
        if args.get("attack_name", []):
            command.extend(["--attack-name", *args["attack_name"]])
        if args.get("manual_review", False):
            command.append("--manual-review")
        if args.get("manual_only", False):
            command.append("--manual-only")
        if args.get("generate_after_manual", False):
            command.append("--generate-after-manual")
    command.extend(["--n-steps", str(args.get("n_steps", 32))])
    command.extend(["--high-confidence-threshold", str(args.get("high_confidence_threshold", 0.90))])
    command.extend(["--attribution-target", args.get("attribution_target", "true_label")])
    command.extend(["--top-percentile", str(args.get("top_percentile", 90))])
    command.extend(["--device", args.get("device", "auto")])
    command.extend(["--input-size", str(args.get("input_size", 224))])
    if args.get("output_tag", ""):
        command.extend(["--output-tag", args["output_tag"]])
    if args.get("force", True):
        command.append("--force")
    if args.get("verbose", True):
        command.append("--verbose")
    return command


def maybe_customize_preset(args: dict[str, Any]) -> dict[str, Any]:
    print("\nCurrent preset configuration:")
    print(json.dumps(args, indent=2, ensure_ascii=False))
    if not ask_bool("Customize advanced options before execution?", default=False):
        return args
    updated = dict(args)
    if not updated.get("selection_manifest"):
        updated["model"] = ask_multi_choice("Select model(s)", SUPPORTED_MODELS, updated.get("model", ["efficientnet_b0"]))
        updated["strategy"] = ask_choice("Select strategy", SUPPORTED_STRATEGIES, updated.get("strategy", "attack_stratified"))
        if updated["strategy"] == "attack_stratified":
            updated["candidate_limit"] = ask_int("Candidate limit per attack (0 = show all candidates)", int(updated.get("candidate_limit", 0)))
        else:
            updated["max_cases"] = ask_int("Max cases", int(updated.get("max_cases", 30)))
        updated["manual_review"] = ask_bool("Enable manual review", bool(updated.get("manual_review", False)))
        if updated["manual_review"]:
            updated["manual_only"] = ask_bool("Manual selection only, without IG generation", bool(updated.get("manual_only", False)))
            updated["generate_after_manual"] = False if updated["manual_only"] else ask_bool("Generate IG after manual review", bool(updated.get("generate_after_manual", True)))
    updated["n_steps"] = ask_int("Integrated Gradients steps", int(updated.get("n_steps", 32)))
    updated["high_confidence_threshold"] = ask_float("High-confidence threshold", float(updated.get("high_confidence_threshold", 0.90)))
    updated["attribution_target"] = ask_choice("Attribution target", SUPPORTED_ATTRIBUTION_TARGETS, updated.get("attribution_target", "true_label"))
    updated["top_percentile"] = ask_float("Top percentile", float(updated.get("top_percentile", 90)))
    updated["device"] = ask_choice("Device", SUPPORTED_DEVICES, updated.get("device", "auto"))
    updated["input_size"] = ask_int("Input size", int(updated.get("input_size", 224)))
    updated["output_tag"] = ask_string("Optional output tag", updated.get("output_tag", ""))
    updated["force"] = ask_bool("Force overwrite current run outputs", bool(updated.get("force", True)))
    updated["verbose"] = ask_bool("Verbose logging", bool(updated.get("verbose", True)))
    return updated


def build_custom_run() -> dict[str, Any]:
    print("\nCUSTOM GUIDED RUN")
    print("-" * 80)
    args: dict[str, Any] = {}
    use_existing = ask_bool("Use existing manual selection manifest?", default=False)
    if use_existing:
        args["selection_manifest"] = ask_string("Selection manifest path", DEFAULT_SELECTION_MANIFEST)
        args["model"] = ["efficientnet_b0"]
        args["strategy"] = "attack_stratified"
    else:
        args["model"] = ask_multi_choice("Select model(s)", SUPPORTED_MODELS, ["efficientnet_b0"])
        args["strategy"] = ask_choice("Select strategy", SUPPORTED_STRATEGIES, "attack_stratified")
        if args["strategy"] == "attack_stratified":
            scope = ask_choice("Manual review scope", ["one_attack", "all_attacks", "custom_attack_filter"], "one_attack")
            if scope == "one_attack":
                attack_name = ask_choice("Select attack_name", DEFAULT_ATTACKS, "sigma_zero")
                args["attack_name"] = [attack_name]
                args["cases_per_attack"] = ask_int(f"How many cases do you want to select for {attack_name}?", 3)
                args["max_cases"] = args["cases_per_attack"]
            elif scope == "all_attacks":
                args["attack_name"] = DEFAULT_ATTACKS.copy()
                args["cases_per_attack"] = ask_int("How many cases per attack?", 3)
                args["max_cases"] = args["cases_per_attack"] * len(DEFAULT_ATTACKS)
            else:
                args["attack_name"] = ask_multi_choice("Select attack_name filter", DEFAULT_ATTACKS, DEFAULT_ATTACKS)
                args["cases_per_attack"] = ask_int("How many cases per selected attack?", 3)
                args["max_cases"] = args["cases_per_attack"] * max(1, len(args["attack_name"]))
            args["candidate_limit"] = ask_int("Candidate limit per attack (0 = show all candidates)", 0)
            args["manual_review"] = ask_bool("Enable manual review", default=True)
            if args["manual_review"]:
                args["manual_only"] = ask_bool("Manual selection only, without IG generation", default=False)
                args["generate_after_manual"] = False if args["manual_only"] else ask_bool("Generate IG after manual review", default=True)
        else:
            args["max_cases"] = ask_int("Max cases", 30)
            args["manual_review"] = ask_bool("Enable manual review", default=False)
    args["n_steps"] = ask_int("Integrated Gradients steps", 32)
    args["high_confidence_threshold"] = ask_float("High-confidence threshold", 0.90)
    args["attribution_target"] = ask_choice("Attribution target", SUPPORTED_ATTRIBUTION_TARGETS, "both")
    args["top_percentile"] = ask_float("Top percentile", 90)
    args["device"] = ask_choice("Device", SUPPORTED_DEVICES, "auto")
    args["input_size"] = ask_int("Input size", 224)
    args["output_tag"] = ask_string("Optional output tag", "")
    args["force"] = ask_bool("Force overwrite current run outputs", True)
    args["verbose"] = ask_bool("Verbose logging", True)
    return args


def execute_command(command: list[str], preset_name: str, args_payload: dict[str, Any]) -> int:
    print("\nCommand to execute:")
    print(command_to_string(command))
    log_command(command, preset_name, args_payload)
    if not ask_bool("Execute now?", default=True):
        print("[INFO] Execution cancelled. Command was saved to launcher log.")
        return 0
    print("\n" + "=" * 80)
    print("RUNNING XAI SCRIPT")
    print("=" * 80)
    process = subprocess.Popen(command, cwd=str(REPO_ROOT))
    return_code = process.wait()
    print("\n" + "=" * 80)
    print(f"Process finished with exit code {return_code}")
    print("=" * 80)
    return int(return_code)


def print_menu() -> None:
    print("\nAvailable workflows:")
    for key in sorted(PRESETS.keys(), key=int):
        print(f"{key}. {PRESETS[key]['name']}")
    print("8. Custom guided run")
    print("0. Exit")


def show_preset_details(key: str) -> None:
    preset = PRESETS[key]
    print("\n" + "-" * 80)
    print(preset["name"])
    print("-" * 80)
    print(preset["description"])
    print("-" * 80)


def main() -> None:
    ensure_xai_script_exists()
    while True:
        print_header()
        print_menu()
        choice = input("\nSelect workflow: ").strip()
        if choice == "0":
            print("[OK] Exit.")
            return
        if choice in PRESETS:
            show_preset_details(choice)
            args = dict(PRESETS[choice]["args"])
            if args.get("selection_manifest") == "ASK":
                args["selection_manifest"] = ask_string("Selection manifest path", DEFAULT_SELECTION_MANIFEST)
            else:
                args = apply_mandatory_preset_questions(args, choice)
            args = maybe_customize_preset(args)
            command = build_command(args)
            return_code = execute_command(command, PRESETS[choice]["name"], args)
        elif choice == "8":
            args = build_custom_run()
            command = build_command(args)
            return_code = execute_command(command, "Custom guided run", args)
        else:
            print(f"[WARN] Invalid workflow: {choice}")
            input("Press Enter to continue...")
            continue
        if return_code != 0:
            print("[WARN] The XAI script ended with a non-zero exit code.")
        if not ask_bool("Run another workflow?", default=False):
            return


if __name__ == "__main__":
    main()
