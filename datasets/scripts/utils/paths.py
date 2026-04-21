from __future__ import annotations

from pathlib import Path
from typing import Callable


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until the real repository root is found.
    Current criterion: the directory contains both 'datasets'
    and 'datasets/scripts', matching the current project layout.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "datasets").is_dir() and (candidate / "datasets" / "scripts").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not determine repository root starting from: {start}"
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve())

DATASETS_DIR = REPO_ROOT / "datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PREPARED_DATASETS_DIR = DATASETS_DIR / "prepared"
SPLITS_DIR = DATASETS_DIR / "splits" / "Clean_Dataset"
METADATA_DIR = DATASETS_DIR / "metadata"

ATTACKS_DIR = REPO_ROOT / "attacks"
ADVERSARIAL_DIR = ATTACKS_DIR / "adversarial"
ANTI_FORENSIC_DIR = ATTACKS_DIR / "anti_forensic"

EVALUATION_DIR = REPO_ROOT / "evaluation"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_DIR = REPO_ROOT / "models"
EXPLAINABILITY_DIR = REPO_ROOT / "explainability"
FORENSIC_TOOLS_DIR = REPO_ROOT / "forensic_tools"

DOCS_DIR = REPO_ROOT / "docs" / "LatexThesis"
DOCS_IMAGES_DIR = DOCS_DIR / "images"

DEFAULT_PATHS = {
    "repo_root": REPO_ROOT,
    "datasets": DATASETS_DIR,
    "raw": RAW_DATASETS_DIR,
    "prepared": PREPARED_DATASETS_DIR,
    "splits": SPLITS_DIR,
    "metadata": METADATA_DIR,
    "results": RESULTS_DIR,
    "figures": FIGURES_DIR,
    "tables": TABLES_DIR,
    "plots": PLOTS_DIR,
    "docs_images": DOCS_IMAGES_DIR,
}


def repo_relative_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def existing_path_validator(
    description: str, predicate: Callable[[Path], bool]
) -> Callable[[Path], Path]:
    def _validator(path: Path) -> Path:
        if not predicate(path):
            raise FileNotFoundError(
                f"Expected {description} at '{path}', but it was not found."
            )
        return path

    return _validator