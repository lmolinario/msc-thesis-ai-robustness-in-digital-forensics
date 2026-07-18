# Repository Map

This document maps the curated public repository to the MSc thesis research workflow.

## Final Top-Level Structure

| Path | Role |
|---|---|
| `.github/` | Security policy and lightweight repository audit workflow |
| `attacks/` | Adversarial and anti-forensic manifests plus local-output boundaries |
| `datasets/` | Dataset acquisition/preparation scripts, frozen manifests, splits, and bundle metadata |
| `docs/` | Artifact documentation, maintenance records, repository assets, and the authoritative LaTeX thesis source |
| `evaluation/` | Proxy predictions and canonical commercial-tool normalization outputs |
| `explainability/` | Integrated Gradients scripts, logs, and canonical thesis-selection manifest |
| `forensic_tools/` | Tool registry, sanitized extracts, validation reports, and tool documentation |
| `models/` | Proxy checkpoints, training script, model card, and registry |
| `results/` | Frozen metrics, Chapter 5 reporting outputs, and validators |
| `tools/` | Local PowerShell and LaTeX audit helpers |

The root intentionally contains only core repository metadata:

```text
.env.example
.gitattributes
.gitignore
CHANGELOG.md
CITATION.cff
LICENSE
README.md
requirements.txt
```

## `datasets/`

Purpose:

- controlled source acquisition;
- technical validation and deduplication;
- human-in-the-loop review;
- final dataset freezing;
- clean/OOD split generation;
- forensic evaluation bundle construction.

Canonical files:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Image corpora and generated image directories are restored or regenerated locally and are excluded from current `main`.

## `attacks/`

Final perturbation families:

```text
adversarial:
  fgsm
  superdeepfool
  sigma_zero
  one_pixel
  color_shift

anti_forensic:
  jpeg_recompression
  resample_resize
  gaussian_blur
  histogram_modification
  contrast_stretching
```

Public `main` retains scripts, manifests, and reports. Generated perturbation images remain local or controlled.

## `models/`

Final proxy architectures:

```text
efficientnet_b0
resnet18
clip
```

Key files:

```text
models/MODEL_CARD.md
models/model_registry.json
models/scripts/12_train_proxy_models.py
models/checkpoints/
models/reports/
```

The registry records the 15 fold-aware checkpoints and their SHA256 digests.

## `evaluation/`

### Proxy layer

```text
evaluation/proxy_models/proxy_model_predictions.csv
evaluation/scripts/15_evaluate_proxy_models.py
```

### Commercial-tool layer

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

The canonical commercial table contains 69,000 sanitized decisions and excludes complete raw-export paths and unrelated proprietary metadata.

## `forensic_tools/`

Purpose:

- document the six evaluated configurations;
- preserve tool versions and run identifiers;
- retain tool-specific sanitized prediction extracts;
- reconstruct the canonical combined prediction table;
- validate exact decision and metric equivalence.

Key files:

```text
forensic_tools/run_registry.json
forensic_tools/public_extracts_summary.json
forensic_tools/public_extracts_validation.json
forensic_tools/scripts/build_public_tool_extracts.py
forensic_tools/scripts/build_canonical_normalized_predictions.py
forensic_tools/scripts/validate_public_extract_equivalence.py
forensic_tools/*/public_extracts/
```

Complete raw commercial exports are not distributed on current `main`.

## `explainability/`

Purpose:

- generate Integrated Gradients case studies for transparent proxies;
- document human-reviewed candidate selection;
- preserve the five cases and twenty image assets used by Chapter 5.

Key files:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
explainability/scripts/validate_chapter5_xai_artifacts.py
explainability/manifests/chapter5/thesis_selection.csv
```

The complete historical XAI output tree is not distributed on current `main`. Thesis-ready assets are stored under `docs/LatexThesis/images/`.

## `results/`

Canonical metric sources:

```text
results/metrics/final_core_metrics.csv
results/metrics/final_robustness_metrics.csv
results/metrics/final_confusion_matrices.csv
results/metrics/final_ood_metrics.csv
results/metrics/forensic_tools_metrics.csv
```

Reporting and validation:

```text
results/figures/chapter_5/
results/scripts/20_generate_experimental_reporting_assets.py
results/scripts/21_generate_embedded_metadata_sensitivity_check.py
results/scripts/23_validate_results_artifacts.py
results/scripts/24_audit_reporting_asset_usage.py
```

OOD accounting uses 500 unique images evaluated by five fold-specific checkpoints, yielding 2,500 predictions per architecture.

## `docs/`

```text
docs/artifact/       research-artifact governance and reproducibility documents
docs/maintenance/    audit and release-maintenance records
docs/assets/         repository-facing graphics
docs/LatexThesis/    authoritative thesis source
```

Authoritative thesis entry point:

```text
docs/LatexThesis/main.tex
```

## `tools/`

```text
tools/tasks.ps1
tools/latex/audit_latex_images_used.py
```

These tools perform non-destructive local checks and are not part of the numbered experimental pipeline.

## Source-of-Truth Principle

The following areas define what was actually produced and reported:

```text
datasets/final/manifests/
datasets/splits/manifests/
attacks/manifests/
models/model_registry.json
evaluation/proxy_models/
evaluation/forensic_tools/normalized_predictions.csv
forensic_tools/public_extracts_validation.json
results/metrics/
explainability/manifests/chapter5/thesis_selection.csv
datasets/forensic_evaluation_bundle/metadata/
docs/LatexThesis/
```

Historical development records are preserved outside the curated branch through the protected snapshot documented in `docs/artifact/ARCHIVE_SNAPSHOT.md`.
