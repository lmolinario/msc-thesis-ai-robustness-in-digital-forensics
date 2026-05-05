#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_generate_adversarial_attacks.py

Official numbered entry point for fold-aware adversarial generation.

This wrapper preserves the global pipeline numbering while delegating execution
to the existing implementation kept in:

    datasets/scripts/attacks/13_generate_adversarial_attacks.py

Execution modes are inherited from the underlying script:
- direct interactive execution
- command-line execution
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("13_generate_adversarial_attacks.py")


def main() -> None:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Adversarial implementation not found: {SCRIPT_PATH}")
    sys.argv[0] = str(SCRIPT_PATH)
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
