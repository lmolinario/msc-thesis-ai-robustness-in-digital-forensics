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
| Commercial forensic-tool evaluation | In progress |
| XAI case studies | In progress for Chapter 5 |
| Thesis reporting | In progress |

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
```

Progress notes explain why and when decisions were made; manifests and metrics define what was actually produced.
