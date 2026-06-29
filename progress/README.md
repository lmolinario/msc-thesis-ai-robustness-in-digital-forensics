# Progress Tracking

This directory documents operational progress, decisions, milestones, and working notes for the thesis repository.

It is intended to make the research workflow auditable over time without mixing temporary notes with code, datasets, or thesis text.

---

## Directory Structure

```text
progress/
├── README.md
├── milestones/
├── logs/
└── notes/
```

---

## Milestones

Recommended milestone files:

```text
progress/milestones/01_dataset_acquisition.md
progress/milestones/02_prepared_dataset.md
progress/milestones/03_manual_selection.md
progress/milestones/04_split_generation.md
progress/milestones/05_attack_generation.md
progress/milestones/06_proxy_model_training.md
progress/milestones/07_proxy_model_evaluation.md
progress/milestones/08_forensic_evaluation_bundle.md
progress/milestones/09_commercial_forensic_tools_evaluation.md
progress/milestones/10_xai_case_studies.md
```

---

## Current Milestone State

| Milestone | Status |
|---|---|
| Dataset acquisition | Completed |
| Prepared dataset | Completed |
| Manual selection / frozen dataset | Completed |
| Clean and OOD split generation | Completed |
| Attack generation | Completed |
| Proxy model training | Completed |
| Proxy model evaluation | Completed |
| Forensic evaluation bundle | Generated and validated |
| Magnet AXIOM / Magnet.AI evaluation | Completed and normalized |
| Excire Foto 2025 evaluation | Completed and normalized |
| Cellebrite Inseyets evaluation | Completed and normalized |
| Magnet Griffeye / T3K CORE evaluation | Completed and normalized |
| XAI case studies | Completed and integrated in Chapter 5 |
| Thesis reporting | Completed and frozen |

---

## Commercial Forensic-Tool Status

The final black-box commercial-tool evaluation perimeter is:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673
- Excire Foto 2025, version 4.1.5
- Cellebrite Inseyets, version 10.9
- Magnet Griffeye x64, version 26.2.108, with T3K CORE v1.18.0
```

Commercial-tool exports have been normalized and mapped back to the forensic evaluation bundle. The resulting normalized outputs and metrics are available in:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
results/metrics/magnet_axiom_metrics.csv
results/metrics/excire_foto_2025_d20_metrics.csv
results/metrics/excire_foto_2025_d50_metrics.csv
results/metrics/excire_foto_2025_d80_metrics.csv
results/metrics/cellebrite_inseyets_metrics.csv
results/metrics/griffeye_metrics.csv
```

Excire Foto 2025 is evaluated as a standalone AI-assisted semantic retrieval tool. Cellebrite Inseyets is evaluated through observable image classifications exported by the Cellebrite Physical Analyzer report. Magnet Griffeye / T3K CORE is evaluated through automatically generated semantic bookmarks.

Commercial tools are treated as operational black boxes. Their outputs are interpreted as observable operational signals, not as direct evidence of internal model behavior.

---

## XAI Status

Integrated Gradients XAI case studies are completed and integrated in Chapter 5.

Selected representative cases:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

The XAI layer remains qualitative and diagnostic. It is used to interpret transparent proxy-model behavior and is not a primary robustness metric.

---

## Logging Principles

Use progress logs for:

- major pipeline decisions;
- deviations from the expected workflow;
- manual selection decisions;
- forensic-tool import/export notes;
- normalization issues;
- XAI case-study selection notes;
- thesis reporting decisions.

Avoid storing sensitive case material, personal data, proprietary tool case files, or unnecessary screenshots in this directory.

---

## Source of Truth

Progress notes are not the source of truth for numerical results.

Use these files instead:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/
attacks/manifests/
evaluation/proxy_models/
results/metrics/
datasets/forensic_evaluation_bundle/metadata/
evaluation/forensic_tools/
docs/LatexThesis/sections/05_experiments.tex
```

Progress notes explain why and when decisions were made; manifests, metrics, normalized exports, and thesis text define what was actually produced and reported.
