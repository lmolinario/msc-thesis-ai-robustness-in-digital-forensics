#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_train_proxy_models.py

Official numbered entry point for fold-aware proxy model training.

This wrapper preserves the pipeline numbering while delegating execution to the
rubric-ready training implementation kept in:

    models/scripts/train_proxy_models.py

Execution modes are inherited from the underlying script:
- direct interactive execution: python models/scripts/12_train_proxy_models.py
- command-line execution: python models/scripts/12_train_proxy_models.py --model ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("train_proxy_models.py")


def main() -> None:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Training implementation not found: {SCRIPT_PATH}")
    sys.argv[0] = str(SCRIPT_PATH)
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
