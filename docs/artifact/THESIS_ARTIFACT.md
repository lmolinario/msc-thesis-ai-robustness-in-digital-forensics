# Thesis Research Artifact

## Title

**Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

## Academic Context

| Field | Value |
|---|---|
| Degree | MSc / Master's Degree |
| Programme | Computer Engineering, Cybersecurity and Artificial Intelligence |
| Institution | University of Cagliari |
| Author | Lello Molinario |
| Supervisor | Davide Maiorca |
| Repository role | Final frozen research artifact supporting the MSc thesis |

## Artifact Purpose

This repository preserves the experimental and documentary artifact supporting a study of the operational robustness of AI-based image-classification and media-triage systems in Digital/Computer Forensics under:

- clean in-distribution inputs;
- out-of-distribution images;
- adversarial perturbations;
- anti-forensic transformations;
- observable black-box outputs from commercial tools.

The artifact supports auditability, traceability, and controlled reproducibility. It is not a general-purpose classifier, unrestricted benchmark release, or redistributable dataset mirror.

## Main Contents

The curated `main` branch contains:

- numbered dataset, training, perturbation, evaluation, XAI, and reporting scripts;
- frozen dataset, split, attack, and bundle manifests;
- 15 transparent proxy-model checkpoints and their registry;
- proxy prediction and metric outputs;
- a canonical sanitized commercial-tool prediction table containing 69,000 decisions;
- four tool-specific sanitized extracts and an exact equivalence report;
- 186 frozen commercial metric rows;
- five thesis-selected Integrated Gradients cases and 20 thesis-ready XAI assets;
- the authoritative LaTeX thesis source;
- access, security, reproducibility, audit, and release documentation.

## Deliberately Excluded from `main`

The repository does not intentionally redistribute:

- raw third-party image corpora;
- prepared, split, perturbed, or blind-input image directories;
- complete commercial-tool raw exports;
- licensed forensic software or proprietary databases;
- commercial case files, evidence material, or operational investigative data;
- secrets, credentials, tokens, temporary signed URLs, or private download links.

Full end-to-end reruns therefore require controlled-access images and, for commercial-tool processing, licensed software environments.

## Official Sources of Truth

### Thesis

```text
docs/LatexThesis/
```

### Frozen dataset and splits

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
```

### Forensic evaluation bundle

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

### Proxy predictions and results

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/
results/figures/chapter_5/
```

### Commercial-tool predictions and validation

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
forensic_tools/public_extracts_validation.json
results/metrics/forensic_tools_metrics.csv
```

### XAI selection

```text
explainability/manifests/chapter5/thesis_selection.csv
```

## Final Commercial-Tool Perimeter

| Tool | Version / module | Evaluated signal |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | exported `Possible weapons` tag |
| Excire Foto 2025 | 4.1.5, D20/D50/D80 | fixed firearm-oriented semantic prompt membership |
| Cellebrite Inseyets | 10.9 / Physical Analyzer 10.9.0.3029 | exported weapon classifications |
| Magnet Griffeye / T3K CORE | Griffeye 26.2.108 / T3K CORE 1.18.0 | exported firearm bookmark |

The internal architectures, thresholds, weights, training data, calibrated probabilities, and undocumented decision logic are not inspected.

## Reproducibility Level

| Component | Publicly auditable | Publicly rerunnable | Boundary |
|---|---:|---:|---|
| Code and repository structure | Yes | Yes | Lightweight scripts and documentation are tracked |
| Frozen manifests and metrics | Yes | Yes | CSV/JSON artifacts are committed |
| Canonical commercial decisions | Yes | Yes | Rebuilt from committed sanitized extracts |
| Reporting and result validation | Yes | Yes | Read-only validators are committed |
| Raw image pipeline | Partially | Controlled | Requires approved data access |
| Full proxy training/evaluation | Partially | Controlled | Requires images and compatible compute |
| Commercial-tool execution | Through outputs | Licensed / controlled | Requires licensed software and blind input files |
| Proprietary commercial AI internals | No | No | Outside the black-box protocol |

## Governance Documents

```text
docs/artifact/ARTIFACT_EVALUATION.md
docs/artifact/REPOSITORY_MAP.md
docs/artifact/DATA_DICTIONARY.md
docs/artifact/ENVIRONMENT.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/DATA_ACCESS.md
.github/SECURITY.md
docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md
docs/maintenance/RELEASE_CHECKLIST.md
```

## Historical Preservation

The complete pre-cleanup state is preserved for provenance through:

```text
branch: archive/pre-commission-cleanup-2026-07-16
tag:    snapshot/pre-commission-cleanup-2026-07-16
commit: 309a4580537ebc3bb7950f29c090bb2729fc603b
```

The current `main` branch remains authoritative. Historical preservation does not grant redistribution permission for third-party data or proprietary exports.

## Artifact Boundary

The correct interpretation is:

```text
public research artifact + controlled-access data + black-box commercial evaluation
```

not:

```text
fully open raw dataset + unrestricted commercial-tool reproduction package
```
