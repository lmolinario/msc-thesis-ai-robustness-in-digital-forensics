from __future__ import annotations

from pathlib import Path
from typing import Callable


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until the repository root is found.

    The repository root is identified as the first parent directory containing both:
    - datasets/
    - datasets/scripts/

    This criterion matches the current FAIR-Lab thesis repository layout.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "datasets").is_dir() and (candidate / "datasets" / "scripts").is_dir():
            return candidate

    raise RuntimeError(
        f"Could not determine repository root starting from: {start}"
    )


# =============================================================================
# Repository root
# =============================================================================

REPO_ROOT = _find_repo_root(Path(__file__).resolve())


# =============================================================================
# Dataset directories
# =============================================================================

DATASETS_DIR = REPO_ROOT / "datasets"

RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PREPARED_DATASETS_DIR = DATASETS_DIR / "prepared"
FINAL_DATASETS_DIR = DATASETS_DIR / "final"


# =============================================================================
# Dataset script directories
# =============================================================================

DATASET_SCRIPTS_DIR = DATASETS_DIR / "scripts"

ACQUISITION_SCRIPTS_DIR = DATASET_SCRIPTS_DIR / "acquisition"
PREPARED_SCRIPTS_DIR = DATASET_SCRIPTS_DIR / "prepared"
FINAL_SCRIPTS_DIR = DATASET_SCRIPTS_DIR / "final"
SPLIT_SCRIPTS_DIR = DATASET_SCRIPTS_DIR / "splits"
UTILS_SCRIPTS_DIR = DATASET_SCRIPTS_DIR / "utils"


# =============================================================================
# Split directories
# =============================================================================

SPLITS_DIR = DATASETS_DIR / "splits"

CLEAN_SPLITS_DIR = SPLITS_DIR / "clean"
OOD_SPLITS_DIR = SPLITS_DIR / "ood"
SPLIT_MANIFESTS_DIR = SPLITS_DIR / "manifests"


# =============================================================================
# Metadata directories
# =============================================================================

METADATA_DIR = DATASETS_DIR / "metadata"


# =============================================================================
# Attack directories
# =============================================================================

ATTACKS_DIR = REPO_ROOT / "attacks"

ADVERSARIAL_DIR = ATTACKS_DIR / "adversarial"
ANTI_FORENSIC_DIR = ATTACKS_DIR / "anti_forensic"


# =============================================================================
# Evaluation and result directories
# =============================================================================

EVALUATION_DIR = REPO_ROOT / "evaluation"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"


# =============================================================================
# Model, explainability, and forensic tool directories
# =============================================================================

MODELS_DIR = REPO_ROOT / "models"
EXPLAINABILITY_DIR = REPO_ROOT / "explainability"
FORENSIC_TOOLS_DIR = REPO_ROOT / "forensic_tools"


# =============================================================================
# Documentation directories
# =============================================================================

DOCS_DIR = REPO_ROOT / "docs" / "LatexThesis"
DOCS_IMAGES_DIR = DOCS_DIR / "images"


# =============================================================================
# Default path registry
# =============================================================================

DEFAULT_PATHS = {
    "repo_root": REPO_ROOT,

    # Dataset roots
    "datasets": DATASETS_DIR,
    "raw": RAW_DATASETS_DIR,
    "prepared": PREPARED_DATASETS_DIR,
    "final": FINAL_DATASETS_DIR,

    # Dataset scripts
    "dataset_scripts": DATASET_SCRIPTS_DIR,
    "acquisition_scripts": ACQUISITION_SCRIPTS_DIR,
    "prepared_scripts": PREPARED_SCRIPTS_DIR,
    "final_scripts": FINAL_SCRIPTS_DIR,
    "split_scripts": SPLIT_SCRIPTS_DIR,
    "utils_scripts": UTILS_SCRIPTS_DIR,

    # Splits
    "splits": SPLITS_DIR,
    "clean_splits": CLEAN_SPLITS_DIR,
    "ood_splits": OOD_SPLITS_DIR,
    "split_manifests": SPLIT_MANIFESTS_DIR,

    # Metadata
    "metadata": METADATA_DIR,

    # Attacks
    "attacks": ATTACKS_DIR,
    "adversarial": ADVERSARIAL_DIR,
    "anti_forensic": ANTI_FORENSIC_DIR,

    # Evaluation and results
    "evaluation": EVALUATION_DIR,
    "results": RESULTS_DIR,
    "figures": FIGURES_DIR,
    "tables": TABLES_DIR,
    "plots": PLOTS_DIR,

    # Models, explainability, forensic tools
    "models": MODELS_DIR,
    "explainability": EXPLAINABILITY_DIR,
    "forensic_tools": FORENSIC_TOOLS_DIR,

    # Documentation
    "docs": DOCS_DIR,
    "docs_images": DOCS_IMAGES_DIR,
}


# =============================================================================
# Helper functions
# =============================================================================

def repo_relative_path(path_str: str | Path) -> Path:
    """
    Resolve a path relative to the repository root.

    If the provided path is already absolute, it is returned as an absolute,
    expanded Path. If it is relative, it is resolved against REPO_ROOT.

    Parameters
    ----------
    path_str:
        Relative or absolute path expressed as a string or Path object.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    candidate = Path(path_str).expanduser()

    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()

    return candidate


def existing_path_validator(
    description: str,
    predicate: Callable[[Path], bool],
) -> Callable[[Path], Path]:
    """
    Build a reusable path validator.

    Parameters
    ----------
    description:
        Human-readable description of the expected path type.
        Example: "directory", "metadata CSV", "image file".

    predicate:
        Function that receives a Path and returns True when the path is valid.

    Returns
    -------
    Callable[[Path], Path]
        Validator function that returns the path if valid, otherwise raises
        FileNotFoundError.
    """

    def _validator(path: Path) -> Path:
        if not predicate(path):
            raise FileNotFoundError(
                f"Expected {description} at '{path}', but it was not found."
            )
        return path

    return _validator