# Repository Map

This document maps the repository structure to the MSc thesis artifact. It is intended for reviewers, supervisors, committee members, and researchers who need to understand where each component of the experimental workflow is stored.

---

## Top-Level Structure

| Path | Role | Artifact type | Thesis role |
|---|---|---|---|
| `datasets/` | Dataset acquisition, preparation, selection, splits, and forensic bundle metadata | Code, manifests, metadata | Dataset construction and source consolidation |
| `attacks/` | Generated adversarial and anti-forensic perturbation artifacts and manifests | Outputs, manifests, documentation | Robustness stressors |
| `models/` | Transparent proxy-model configuration, checkpoints, and training scripts | Code, model artifacts, registry | Reproducible proxy baseline |
| `evaluation/` | Proxy-model evaluation and commercial-tool normalization | Code, predictions, normalized outputs | Quantitative evaluation layer |
| `explainability/` | Integrated Gradients case-study workflow | Code, manifests, XAI outputs | Qualitative diagnostic interpretation |
| `forensic_tools/` | Commercial-tool export organization and documentation | Raw export areas, notes, controlled outputs | Black-box forensic-tool evaluation |
| `results/` | Thesis-oriented metrics, figures, tables, and reporting assets | Metrics, figures, scripts | Results chapter material |
| `docs/` | LaTeX thesis source and supporting documentation | Thesis source, figures, bibliography | Official thesis text |
| `progress/` | Milestones, operational notes, and working audit trail | Progress documentation | Historical workflow audit |

---

## `datasets/`

Purpose:

- controlled raw source acquisition;
- technical preparation and deduplication;
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

Key files:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Artifact role:

```text
frozen manifests + controlled split definitions + forensic bundle metadata
```

---

## `attacks/`

Purpose:

- store adversarial perturbation outputs;
- store anti-forensic transformation outputs;
- preserve attack/transformation manifests;
- document perturbation generation protocol.

Key areas:

```text
attacks/adversarial/
attacks/anti_forensic/
attacks/manifests/
```

Main perturbation families:

```text
adversarial: fgsm, superdeepfool, sigma_zero, one_pixel, color_shift
anti-forensic: jpeg_recompression, resample_resize, gaussian_blur, histogram_modification, contrast_stretching
```

Artifact role:

```text
stress-test inputs for proxy models and commercial black-box tools
```

---

## `models/`

Purpose:

- train transparent proxy models;
- preserve fold-aware checkpoints;
- document model registry and training configuration.

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
models/model_registry.json
```

Artifact role:

```text
transparent and reproducible proxy baseline for robustness testing
```

---

## `evaluation/`

Purpose:

- evaluate transparent proxy models;
- normalize commercial black-box forensic-tool exports;
- create prediction-level and metric-level outputs.

Key areas:

```text
evaluation/proxy_models/
evaluation/forensic_tools/
evaluation/scripts/
```

Key files:

```text
evaluation/proxy_models/proxy_model_predictions.csv
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/scripts/15_evaluate_proxy_models.py
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Artifact role:

```text
common evaluation layer for transparent proxy outputs and black-box commercial outputs
```

---

## `explainability/`

Purpose:

- generate qualitative Integrated Gradients case studies;
- document selected representative cases;
- support Chapter 5 interpretation.

Key areas:

```text
explainability/scripts/
explainability/manifests/
explainability/outputs/integrated_gradients/
```

Key scripts:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
```

Artifact role:

```text
qualitative diagnostic layer for transparent proxy models only
```

---

## `forensic_tools/`

Purpose:

- organize commercial-tool raw export areas;
- document tool versions and export context;
- preserve the boundary between raw commercial exports and normalized evaluation outputs.

Final tool areas:

```text
forensic_tools/magnet_axiom/raw_exports/
forensic_tools/excire_foto_2025/raw_exports/
forensic_tools/cellebrite_inseyets/raw_exports/
forensic_tools/griffeye/raw_exports/
```

Artifact role:

```text
commercial black-box export organization and audit context
```

---

## `results/`

Purpose:

- collect final metrics;
- store thesis-oriented figures and reporting assets;
- generate final reporting material for Chapter 5.

Key areas:

```text
results/metrics/
results/figures/
results/scripts/
```

Key scripts:

```text
results/scripts/20_generate_experimental_reporting_assets.py
results/scripts/21_generate_embedded_metadata_sensitivity_check.py
```

Artifact role:

```text
final quantitative and reporting layer
```

---

## `docs/`

Purpose:

- store the final LaTeX thesis source;
- store thesis figures, bibliography, acronym definitions, and supporting documentation.

Official thesis source:

```text
docs/LatexThesis/
```

Key files:

```text
docs/LatexThesis/main.tex
docs/LatexThesis/sections/
docs/LatexThesis/tesi.bib
docs/LatexThesis/sections/000_acronyms.tex
```

Artifact role:

```text
official frozen thesis source
```

---

## `progress/`

Purpose:

- preserve the operational history of the thesis workflow;
- document milestone status;
- record decisions and audit notes without mixing them with source code or thesis text.

Key areas:

```text
progress/milestones/
progress/logs/
progress/notes/
```

Artifact role:

```text
historical workflow audit, not numerical source of truth
```

---

## Source-of-Truth Principle

Progress notes explain how and why the workflow evolved. The following areas define what was actually produced and reported:

```text
datasets/final/manifests/
datasets/splits/manifests/
attacks/manifests/
evaluation/proxy_models/
evaluation/forensic_tools/
results/metrics/
docs/LatexThesis/
```
