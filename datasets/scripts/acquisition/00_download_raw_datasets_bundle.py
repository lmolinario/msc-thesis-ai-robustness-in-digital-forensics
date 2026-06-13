#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_raw_datasets_bundle.py

Public repository notice
------------------------
The archived raw dataset bundle is not distributed through a public URL in this
repository. Access is available only on request or through controlled sharing,
subject to legal, ethical, source-specific and institutional constraints.

This placeholder is intentionally kept in the public repository to document the
bootstrap entry point without exposing a direct download link.
"""

from __future__ import annotations

import os

RAW_DATASET_BUNDLE_URL_ENV = "FAIRLAB_RAW_DATASET_BUNDLE_URL"


def main() -> None:
    """Explain the controlled-access policy for the raw dataset bundle."""
    bundle_url = os.getenv(RAW_DATASET_BUNDLE_URL_ENV, "").strip()

    if bundle_url:
        print(
            "A controlled-access raw dataset bundle URL was provided through "
            f"{RAW_DATASET_BUNDLE_URL_ENV}. The public repository does not "
            "print or persist this URL."
        )
        print(
            "Use the private working copy of this script or the documented "
            "controlled-access procedure to restore the raw dataset locally."
        )
        return

    raise RuntimeError(
        "The raw dataset bundle is not publicly distributed from this repository. "
        f"Set {RAW_DATASET_BUNDLE_URL_ENV} only when controlled access has been "
        "explicitly granted."
    )


if __name__ == "__main__":
    main()
