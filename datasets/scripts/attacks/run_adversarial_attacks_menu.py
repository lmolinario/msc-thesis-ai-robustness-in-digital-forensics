#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_adversarial_attacks_menu.py

Interactive local launcher for adversarial attack generation in the FAIR-Lab
thesis pipeline.

Purpose
-------
This script provides a guided menu for running the official adversarial attack
generator without manually typing all CLI arguments.

Important methodological note
-----------------------------
The official, reproducible entry point remains:

    datasets/scripts/attacks/13_generate_adversarial_attacks.py

This launcher is only an operational convenience for local execution, PyCharm,
or terminal use. It builds and executes an explicit command line, so each run can
still be copied, inspected, and reproduced.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_SCRIPT = REPO_ROOT / "datasets" / "scripts" / "attacks" / "13_generate_adversarial_attacks.py"

DEFAULT_CHECKPOINTS = {
    "resnet18": REPO_ROOT / "models" / "resnet18_binary.pt",
    "efficientnet_b0": REPO_ROOT / "models" / "efficientnet_b0_binary.pt",
    "clip": REPO_ROOT / "models" / "clip_binary_head.pt",
}

SUPPORTED_ATTACKS = ["color_shift", "fgsm"]
SUPPORTED_TARGET_MODELS = ["resnet18", "efficientnet_b0", "clip"]


# =============================================================================
# Console helpers
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "s", "si", "sì"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Risposta non valida. Inserisci y/n.")


def ask_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{marker}")

    while True:
        answer = input("Selezione: ").strip()
        if not answer:
            return options[default_index]
        if answer.isdigit():
            selected = int(answer)
            if 1 <= selected <= len(options):
                return options[selected - 1]
        print(f"Selezione non valida. Inserisci un numero tra 1 e {len(options)}.")


def ask_multi_choice(
    prompt: str,
    options: list[str],
    default_all: bool = True,
) -> list[str]:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")

    default_text = "all" if default_all else ""
    print("\nEsempi: 1 | 1 2 | 1,2,3 | all")

    while True:
        answer = input(f"Selezione [{default_text}]: ").strip().lower()
        if not answer and default_all:
            return options.copy()
        if answer == "all":
            return options.copy()

        normalized = answer.replace(",", " ").split()
        if not normalized:
            print("Selezione vuota non valida.")
            continue

        selected: list[str] = []
        valid = True
        for item in normalized:
            if not item.isdigit():
                valid = False
                break
            index = int(item)
            if not (1 <= index <= len(options)):
                valid = False
                break
            value = options[index - 1]
            if value not in selected:
                selected.append(value)

        if valid and selected:
            return selected

        print(f"Selezione non valida. Usa numeri tra 1 e {len(options)} oppure 'all'.")


def ask_path(prompt: str, default_path: Path) -> Path:
    answer = input(f"{prompt}\nDefault: {default_path}\nPercorso: ").strip()
    if not answer:
        return default_path
    path = Path(answer).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def ask_float(prompt: str, default_value: float) -> float:
    while True:
        answer = input(f"{prompt} [{default_value}]: ").strip()
        if not answer:
            return default_value
        try:
            return float(answer)
        except ValueError:
            print("Valore non valido. Inserisci un numero.")


def ask_int(prompt: str, default_value: int) -> int:
    while True:
        answer = input(f"{prompt} [{default_value}]: ").strip()
        if not answer:
            return default_value
        try:
            return int(answer)
        except ValueError:
            print("Valore non valido. Inserisci un intero.")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


# =============================================================================
# Menu logic
# =============================================================================

def choose_attacks() -> list[str]:
    mode = ask_choice(
        prompt="Cosa vuoi generare?",
        options=[
            "Solo color_shift (model-agnostic, non richiede checkpoint)",
            "Solo fgsm (model-dependent, richiede checkpoint)",
            "color_shift + fgsm",
        ],
        default_index=0,
    )

    if mode.startswith("Solo color_shift"):
        return ["color_shift"]
    if mode.startswith("Solo fgsm"):
        return ["fgsm"]
    return ["color_shift", "fgsm"]


def choose_target_models(attacks: list[str]) -> list[str]:
    if "fgsm" not in attacks:
        return SUPPORTED_TARGET_MODELS.copy()

    return ask_multi_choice(
        prompt="Seleziona i target model per FGSM:",
        options=SUPPORTED_TARGET_MODELS,
        default_all=True,
    )


def collect_checkpoint_args(target_models: list[str], attacks: list[str]) -> list[str]:
    if "fgsm" not in attacks:
        return []

    args: list[str] = []
    print_header("Checkpoint per FGSM")

    for model_name in target_models:
        checkpoint_path = ask_path(
            prompt=f"Checkpoint per {model_name}",
            default_path=DEFAULT_CHECKPOINTS[model_name],
        )

        if not checkpoint_path.exists():
            print(f"\nATTENZIONE: checkpoint non trovato: {checkpoint_path}")
            proceed = ask_yes_no(
                "Vuoi continuare comunque? Lo script ufficiale si fermerà se il file manca.",
                default=False,
            )
            if not proceed:
                raise SystemExit("Esecuzione annullata: checkpoint mancante.")

        if model_name == "resnet18":
            args.extend(["--checkpoint-resnet18", relative_or_absolute(checkpoint_path)])
        elif model_name == "efficientnet_b0":
            args.extend(["--checkpoint-efficientnet-b0", relative_or_absolute(checkpoint_path)])
        elif model_name == "clip":
            args.extend(["--checkpoint-clip", relative_or_absolute(checkpoint_path)])

    return args


def build_command() -> list[str]:
    print_header("FAIR-Lab adversarial attack launcher")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Generator script: {GENERATOR_SCRIPT}")

    attacks = choose_attacks()
    target_models = choose_target_models(attacks)

    command = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--attack",
        *attacks,
    ]

    if target_models:
        command.extend(["--target-model", *target_models])

    command.extend(collect_checkpoint_args(target_models, attacks))

    if "fgsm" in attacks:
        epsilon = ask_float("FGSM epsilon in pixel space [0,1]", 8.0 / 255.0)
        command.extend(["--fgsm-epsilon", str(epsilon)])

        device = ask_choice(
            prompt="Device per model-dependent attacks:",
            options=["auto", "cpu", "cuda"],
            default_index=0,
        )
        command.extend(["--device", device])

        input_size = ask_int("Input size quadrato per gli adapter", 224)
        command.extend(["--input-size", str(input_size)])

    jpeg_quality = ask_int("JPEG quality output", 95)
    command.extend(["--jpeg-quality", str(jpeg_quality)])

    if "color_shift" in attacks:
        print_header("Parametri color_shift")
        red_shift = ask_int("Red channel shift", 12)
        green_shift = ask_int("Green channel shift", 0)
        blue_shift = ask_int("Blue channel shift", -12)
        saturation = ask_float("Saturation factor", 1.10)
        contrast = ask_float("Contrast factor", 1.00)

        command.extend(
            [
                "--color-red-shift",
                str(red_shift),
                "--color-green-shift",
                str(green_shift),
                "--color-blue-shift",
                str(blue_shift),
                "--color-saturation-factor",
                str(saturation),
                "--color-contrast-factor",
                str(contrast),
            ]
        )

    if ask_yes_no("Vuoi rigenerare sovrascrivendo le cartelle degli attacchi selezionati?", default=True):
        command.append("--force")

    if ask_yes_no("Vuoi logging verbose?", default=False):
        command.append("--verbose")

    return command


def run_command(command: list[str]) -> int:
    print_header("Comando generato")
    print(" ".join(shlex.quote(part) for part in command))

    if not ask_yes_no("Eseguire ora questo comando?", default=True):
        print("Esecuzione annullata. Puoi copiare il comando qui sopra e lanciarlo manualmente.")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Official generator script not found: {GENERATOR_SCRIPT}")

    command = build_command()
    exit_code = run_command(command)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
