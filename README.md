# MSc Thesis – AI Robustness in Digital Forensics

This repository contains the working research pipeline for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the operational robustness of AI-based image classification systems in digital forensic scenarios. The workflow compares local AI models and forensic AI tools under clean inputs, perturbed inputs, and out-of-distribution samples.

At this stage, this is a **complete working repository**. Datasets, generated outputs, manifests, reports, and intermediate artifacts may be versioned to support continuity across multiple workstations. A cleaned public release can be derived later.

---

## Current Status

Completed stages:

1. dataset acquisition structure;
2. prepared dataset construction;
3. full review manifest generation;
4. manual human-in-the-loop selection;
5. final frozen dataset construction;
6. clean fold generation;
7. OOD evaluation set generation;
8. proxy model training setup;
9. adversarial and anti-forensic generation setup.

Current focus:

1. run perturbation smoke tests;
2. generate adversarial and anti-forensic perturbations;
3. build the forensic evaluation bundle;
4. define the evaluation schema for local models and forensic tools.

Planned next stages:

1. evaluate local AI models and forensic AI tools under a shared metric protocol;
2. perform explainability-based case studies;
3. consolidate results and thesis chapters.

---

## Official Dataset Artifacts

The official frozen dataset is:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

The official binary subset used for clean folds and attack generation is:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

The previous `33_final_frozen_dataset.csv` naming convention is no longer used.

---

## Pipeline Overview

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
    ↓
datasets/scripts/acquisition/01_download_kaggle.py ... 07_scrape_deepweb.py
    ↓
datasets/scripts/prepared/08_build_prepared_dataset.py
    ↓
datasets/prepared/final_pool/
    ↓
datasets/scripts/prepared/09_generate_review_manifest_full.py
    ↓
datasets/prepared/manifests/review_manifest_full.csv
    ↓
datasets/scripts/final/10_manual_selection_protocol_reviewer.py
    ↓
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
    ↓
datasets/scripts/splits/11_generate_clean_and_ood_splits.py
    ↓
datasets/splits/clean/
datasets/splits/ood/
    ↓
models/scripts/12_train_proxy_models.py
    ↓
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
    ↓
attacks/
    ↓
datasets/forensic_evaluation_bundle/
    ↓
models/
forensic_tools/
evaluation/
explainability/
results/
```

`datasets/forensic_evaluation_bundle/` is a planned operational bundle. It will be generated after clean, adversarial, anti-forensic, and OOD artifacts are ready to be tested with forensic AI tools.

---

## Repository Structure

```text
msc-thesis-ai-robustness-in-digital-forensics/
├── datasets/
│   ├── README.md
│   ├── raw/
│   ├── prepared/
│   ├── final/
│   ├── splits/
│   │   ├── clean/
│   │   ├── ood/
│   │   └── manifests/
│   ├── forensic_evaluation_bundle/
│   └── scripts/
│       ├── utils/
│       ├── acquisition/
│       ├── prepared/
│       ├── final/
│       ├── splits/
│       ├── attacks/
│       └── bundle/
├── attacks/
│   ├── adversarial/
│   └── anti_forensic/
├── models/
│   ├── scripts/
│   ├── checkpoints/
│   └── reports/
├── evaluation/
├── explainability/
├── forensic_tools/
├── results/
├── docs/
└── progress/
    ├── milestones/
    ├── logs/
    └── notes/
```

---

## Official Script Sequence

The operational pipeline uses numbered scripts as the official entry points:

```text
00_download_raw_datasets_bundle.py
01_download_kaggle.py
02_download_github.py
03_build_subset_deepfirearm.py
04_scrape_google.py
05_scrape_telegram.py
06_scrape_youtube.py
07_scrape_deepweb.py
08_build_prepared_dataset.py
09_generate_review_manifest_full.py
10_manual_selection_protocol_reviewer.py
11_generate_clean_and_ood_splits.py
12_train_proxy_models.py
13_generate_anti_forensic_attacks.py
14_generate_adversarial_attacks.py
```

Compatibility implementation files may be retained internally, but numbered scripts are the official scripts to cite and run in reproducible experiments.

---

## Dataset Design

The final dataset is organized into three semantic groups:

- target-class images;
- negative-class images;
- out-of-distribution samples.

Final frozen distribution:

| Group | Count |
|---|---:|
| target class | 500 |
| negative class | 500 |
| OOD | 500 |

The binary subset is divided into five clean folds. Each fold contains 200 samples, equally balanced between the two binary classes.

OOD samples are evaluated separately as a single OOD evaluation set and are not used as direct targets for perturbation generation.

---

## Split Strategy

Clean folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

OOD evaluation set:

```text
datasets/splits/ood/ood_eval_set/ood/
```

Split manifests:

```text
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/splits/manifests/split_generation_summary.json
```

The repository uses `fold_1`, `fold_2`, etc., rather than `test_set_1`, because this is more consistent with experimental machine learning terminology.

---

## Reproducibility and Traceability

The repository is designed around traceable artifacts:

- file-level SHA256 hashing;
- MD5 hashing for forensic tool compatibility;
- deterministic preparation and split generation where possible;
- CSV manifests for each major stage;
- manual review logs;
- split manifests;
- attack manifests;
- future forensic bundle manifest;
- normalized evaluation outputs.

The hash-based mapping is especially important because forensic tools may rename files, alter export structures, or provide different reporting formats.

---

## Progress Tracking

The `progress/` directory documents operational progress and decisions.

Current structure:

```text
progress/
├── milestones/
│   ├── 01_dataset_acquisition.md
│   ├── 02_prepared_dataset.md
│   ├── 03_manual_selection.md
│   ├── 04_split_generation.md
│   └── 05_anti_forensic_generation.md
├── logs/
│   └── README.md
└── notes/
    ├── methodological_decisions.md
    ├── open_questions.md
    └── operational_pipeline.md
```

---

## Dataset Availability

This repository is currently maintained as a working research repository. During development, data and generated outputs may be versioned to allow continuation across multiple machines.

For a future public release, raw images and sensitive generated artifacts may be removed or replaced with manifests, hashes, sample images where permissible, documentation, reproducible scripts, and aggregated metrics.

---

## Research Context

This work is developed within the MSc program in:

> Computer Engineering, Cybersecurity and Artificial Intelligence

and is aligned with research areas including Digital Forensics, AI-based forensic analysis, Adversarial Machine Learning, Anti-forensics, Robustness Evaluation, and Explainable AI.

---

## Citation

Citation details will be added upon thesis completion.

---

## License

To be defined.

Recommended separation:

- code: MIT License or Apache-2.0;
- documentation: CC BY 4.0;
- datasets/images: restricted according to legal, ethical, and source-specific constraints.
