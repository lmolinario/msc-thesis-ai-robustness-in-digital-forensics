# Repository Map

This document maps the public repository structure to the MSc thesis research artifact. It is intended for reviewers, supervisors, committee members, and researchers who need to locate the implementation, manifests, evaluation outputs, and thesis sources.

---

## Top-Level Structure

| Path | Role | Main artifact type |
|---|---|---|
| `datasets/` | Dataset acquisition, preparation, human review, frozen manifests, splits, and forensic evaluation bundle | Scripts, manifests, metadata |
| `attacks/` | Adversarial and anti-forensic perturbation artifacts | Generated inputs, manifests |
| `models/` | Transparent proxy-model training and registry | Scripts, checkpoints, registry |
| `evaluation/` | Proxy-model evaluation and commercial-tool output normalization | Predictions, normalized outputs, scripts |
| `explainability/` | Integrated Gradients case-study workflow | Scripts, manifests, XAI outputs |
| `forensic_tools/` | Commercial-tool export organization | Raw export areas and tool-specific documentation |
| `results/` | Final metrics, figures, and reporting assets | CSV/JSON metrics, figures, scripts |
| `docs/` | English and Italian LaTeX thesis sources and repository documentation | Thesis source, bibliography, figures |
| `.github/` | Automated lightweight repository audit | GitHub Actions workflow |

Historical development notes and milestone records are preserved in the archival branch:

```text
archive/pre-commission-cleanup-2026-07-16
```

They are intentionally excluded from the public-facing `main` branch because they are not canonical sources for the final experimental state.

---

## `datasets/`

Purpose:

- controlled raw source acquisition;
- technical validation and deduplication;
- human-in-the-loop semantic review;
- final dataset freezing;
- clean/OOD split generation;
- forensic evaluation bundle construction.

Key areas:

```text
datasets/final/manifests/
datasets/splits/manifests/
datasets/forensic_evaluation_bundle/metadata/
datasets/scripts/
```

Canonical files:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

---

## `attacks/`

Purpose:

- store adversarial perturbation outputs;
- store anti-forensic transformation outputs;
- preserve generation and evaluation manifests.

Key areas:

```text
attacks/adversarial/
attacks/anti_forensic/
attacks/manifests/
```

Final perturbation families:

```text
adversarial: fgsm, superdeepfool, sigma_zero, one_pixel, color_shift
anti-forensic: jpeg_recompression, resample_resize, gaussian_blur,
               histogram_modification, contrast_stretching
```

---

## `models/`

Purpose:

- train transparent proxy models;
- preserve fold-aware checkpoints;
- document model configurations and hashes.

Final proxy models:

```text
efficientnet_b0
resnet18
clip
```

Key areas:

```text
models/scripts/
models/checkpoints/
models/reports/
models/model_registry.json
```

---

## `evaluation/`

Purpose:

- evaluate transparent proxy models;
- normalize commercial black-box forensic-tool exports;
- generate prediction-level and metric-level outputs.

Key areas:

```text
evaluation/proxy_models/
evaluation/forensic_tools/
evaluation/scripts/
```

Canonical files:

```text
evaluation/proxy_models/proxy_model_predictions.csv
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/scripts/15_evaluate_proxy_models.py
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

---

## `explainability/`

Purpose:

- generate qualitative Integrated Gradients case studies;
- preserve case-selection manifests;
- support the interpretation presented in Chapter 5.

Key areas:

```text
explainability/scripts/
explainability/manifests/
explainability/outputs/integrated_gradients/
```

---

## `forensic_tools/`

Purpose:

- organize tool-specific raw export areas;
- document versions and export context;
- preserve the boundary between proprietary exports and normalized outputs.

Final tool areas:

```text
forensic_tools/magnet_axiom/
forensic_tools/excire_foto_2025/
forensic_tools/cellebrite_inseyets/
forensic_tools/griffeye/
```

---

## `results/`

Purpose:

- collect final metrics;
- generate thesis-ready figures and reporting assets;
- preserve the quantitative source files used by Chapter 5.

Key areas:

```text
results/metrics/
results/figures/
results/scripts/
```

Official reporting scripts:

```text
results/scripts/20_generate_experimental_reporting_assets.py
results/scripts/21_generate_embedded_metadata_sensitivity_check.py
```

---

## `docs/`

Purpose:

- store the final English LaTeX thesis source;
- retain the Italian source as a separate archival language version;
- store thesis figures, bibliography, acronym definitions, and repository assets.

Key areas:

```text
docs/LatexThesis/
docs/LatexThesis_ITA/
docs/assets/
```

The canonical thesis source for the submitted artifact is:

```text
docs/LatexThesis/main.tex
```

---

## Source-of-Truth Principle

Repository documentation explains the workflow, but the following areas define what was actually produced and reported:

```text
datasets/final/manifests/
datasets/splits/manifests/
attacks/manifests/
models/model_registry.json
evaluation/proxy_models/
evaluation/forensic_tools/
results/metrics/
datasets/forensic_evaluation_bundle/metadata/
docs/LatexThesis/
```

Numerical values should be derived from the canonical manifests, normalized predictions, metrics, and final thesis source rather than from historical development notes.
