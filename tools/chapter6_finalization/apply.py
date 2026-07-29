#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "docs/LatexThesis/sections/06_results.tex"
SYNC = REPO_ROOT / "explainability/scripts/sync_chapter5_xai_metadata.py"
VALIDATOR = REPO_ROOT / "explainability/scripts/validate_chapter5_xai_artifacts.py"

EXPECTED_SHA256 = {
    "metadata": "f1d58d6473828e405cf7d61e09a0caea9eeeb34ce5384ee27a436cb4b2ec6c9f",
    "comparative": "28a11cd8f54f08ca75921a72661d4ff221a51134ba059a01517d490e5bc01454",
    "xai": "560382ba14aea49f9e3638cc24743861ea1ef9d355d2e5ec20824a5964f75397",
    "limitations": "86efeeb2cc861a0429e9949aef2e911bbc60b3c86faeb76cc7a729bbbe693002",
}


def verify(name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256[name]:
        raise RuntimeError(
            f"Replacement checksum mismatch for {name}: "
            f"expected {EXPECTED_SHA256[name]}, got {digest}"
        )
    return text


def read_plain(name: str) -> str:
    return verify(name, (STAGING / f"{name}.tex").read_text(encoding="utf-8"))


def read_compressed(name: str) -> str:
    parts = sorted(STAGING.glob(f"{name}_*.b64"))
    if not parts:
        raise RuntimeError(f"No compressed replacement chunks found for {name}")
    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    decoded = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")
    return verify(name, decoded)


def replace_block(text: str, start: str, end: str | None, replacement: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        raise RuntimeError(f"Start marker not found: {start}")
    if end is None:
        end_idx = len(text)
    else:
        end_idx = text.find(end, start_idx + len(start))
        if end_idx < 0:
            raise RuntimeError(f"End marker not found after {start}: {end}")
    return text[:start_idx] + replacement.rstrip() + "\n" + text[end_idx:]


def patch_results() -> None:
    text = RESULTS.read_text(encoding="utf-8")

    text = replace_block(
        text,
        r"\subsection{Embedded Metadata Sensitivity Check}",
        r"\section{Comparative Operational Robustness Analysis}",
        read_plain("metadata"),
    )
    text = replace_block(
        text,
        r"\section{Comparative Operational Robustness Analysis}",
        r"\section{Explainability Analysis and Representative Failure Cases}",
        read_compressed("comparative"),
    )
    text = replace_block(
        text,
        r"\section{Explainability Analysis and Representative Failure Cases}",
        r"\section{Operational Interpretation and Experimental Limitations}",
        read_compressed("xai"),
    )
    text = replace_block(
        text,
        r"\section{Operational Interpretation and Experimental Limitations}",
        None,
        read_plain("limitations"),
    )

    text = text.replace(
        "the four model-dependent adversarial attacks: \\gls{fgsm}, One\n"
        "Pixel, Sigma-Zero, and SuperDeepFool.",
        "the four model-dependent adversarial attacks: \\gls{fgsm},\n"
        "\\gls{onepixel}, \\gls{sigmazero}, and \\gls{superdeepfool}.",
    )
    text = text.replace(
        "cross-system confidence\nscores.",
        "cross-system probability or uncertainty\nscores.",
    )
    text = text.replace(
        "canonical public population contains no \\texttt{unknown} outputs.",
        "canonical normalized result population contains no \\texttt{unknown} outputs.",
    )

    required = (
        r"\label{tab:results-embedded-metadata-leaveout}",
        r"\label{tab:results-comparative-adversarial}",
        r"\textbf{Max-P}: 1.000",
        r"\item \textbf{Derived-sample dependence.}",
    )
    for marker in required:
        if marker not in text:
            raise RuntimeError(f"Expected Chapter 6 marker missing after patch: {marker}")
    if r"\textbf{confidence}" in text:
        raise RuntimeError("Legacy XAI confidence metadata remains in Chapter 6")

    RESULTS.write_text(text, encoding="utf-8")


def patch_sync_script() -> None:
    text = SYNC.read_text(encoding="utf-8")
    text = text.replace(
        '"""Synchronize displayed Chapter 5 XAI confidence values with the canonical manifest."""',
        '"""Synchronize results-chapter XAI Max-P values with the canonical manifest."""',
    )
    text = text.replace(
        'TEX_FILE = REPO_ROOT / "docs/LatexThesis/sections/05_experiments.tex"',
        'TEX_FILE = REPO_ROOT / "docs/LatexThesis/sections/06_results.tex"',
    )
    text = text.replace(
        r'rf"\\textbf\{{confidence\}}\s*:\s*"',
        r'rf"\\textbf\{{(?:confidence|Max-P)\}}\s*:\s*"',
    )
    text = text.replace(
        "XAI figure metadata confidence",
        "XAI figure metadata Max-P value",
    )
    if "05_experiments.tex" in text:
        raise RuntimeError("Obsolete XAI synchronization target remains")
    if "(?:confidence|Max-P)" not in text:
        raise RuntimeError("Max-P-compatible pattern missing from synchronization script")
    SYNC.write_text(text, encoding="utf-8")


def patch_validator() -> None:
    text = VALIDATOR.read_text(encoding="utf-8")
    text = text.replace(
        r'rf"\\textbf\{{confidence\}}\s*:\s*"',
        r'rf"\\textbf\{{(?:confidence|Max-P)\}}\s*:\s*"',
    )
    text = text.replace(
        "Expected one confidence value for figure",
        "Expected one Max-P value for figure",
    )
    if "(?:confidence|Max-P)" not in text:
        raise RuntimeError("Max-P-compatible pattern missing from XAI validator")
    VALIDATOR.write_text(text, encoding="utf-8")


def main() -> None:
    patch_results()
    patch_sync_script()
    patch_validator()
    print("Chapter 6 final revisions applied and verified.")


if __name__ == "__main__":
    main()
