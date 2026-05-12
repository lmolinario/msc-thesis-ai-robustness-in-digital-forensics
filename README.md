# MSc Thesis – AI Robustness in Digital Forensics

This repository contains the working research pipeline for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the operational robustness of AI-based image classification systems in digital forensic scenarios. The workflow compares local proxy AI models and forensic AI tools under clean inputs, adversarial perturbations, anti-forensic transformations, and out-of-distribution samples.

At this stage, this is a **complete working research repository**. Datasets, generated outputs, manifests, reports, and intermediate artifacts may be versioned to support continuity across multiple workstations. A cleaned public release can be derived later.

---

## Current Status

Completed stages:

1. dataset acquisition structure;
2. prepared dataset construction;
3. full review manifest generation;
4. manual human-in-the-loop final selection;
5. final frozen dataset construction;
6. clean fold generation;
7. OOD evaluation set generation;
8. proxy model training;
9. anti-forensic transformation generation;
10. adversarial attack generation;
11. local proxy model evaluation under clean, OOD, adversarial, and anti-forensic conditions;
12. initial forensic evaluation bundle construction.

Current focus:

1. validate forensic evaluation bundle completeness and traceability;
2. run forensic AI tools on the blind bundle;
3. normalize forensic tool outputs;
4. compare local proxy models and forensic tools under a shared metric protocol;
5. select representative failure cases for explainability and thesis discussion.

Next thesis stage:

1. consolidate Chapter 5 results for proxy models;
2. complete forensic tool evaluation;
3. integrate comparative results and operational implications;
4. finalize explainability case studies.

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
evaluation/scripts/15_evaluate_proxy_models.py
    ↓
evaluation/proxy_models/
results/metrics/
    ↓
datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py
    ↓
datasets/forensic_evaluation_bundle/
    ↓
forensic_tools/
evaluation/forensic_tools/
results/
explainability/
```

`datasets/forensic_evaluation_bundle/` is the operational bridge between the local experimental pipeline and the forensic AI tool evaluation phase. It is intended to provide blind tool inputs while preserving internal traceability through metadata and hashes.

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
│   ├── anti_forensic/
│   └── manifests/
├── models/
│   ├── scripts/
│   ├── checkpoints/
│   └── reports/
├── evaluation/
│   ├── scripts/
│   └── proxy_models/
├── explainability/
├── forensic_tools/
├── results/
│   └── metrics/
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
15_evaluate_proxy_models.py
16_build_forensic_evaluation_bundle.py
```

---

## Dataset Design

The final dataset is organized into three semantic groups:

- `weapon`;
- `non_weapon`;
- `ood`.

Final frozen distribution:

| Group | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |
| OOD | 500 |

The binary subset is divided into five clean folds. Each fold contains 200 samples, equally balanced between `weapon` and `non_weapon`.

OOD samples are evaluated separately as a single OOD evaluation set and are not used as direct targets for perturbation generation.

---

## Attack and Transformation Set

Adversarial attacks generated for the thesis pipeline:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

Anti-forensic transformations generated for the thesis pipeline:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Adversarial attacks are generated primarily against the EfficientNet-B0 proxy target where model dependency is required. Color Shift is treated as model-agnostic. Anti-forensic transformations are model-agnostic image-processing transformations.

---

## Evaluation Status

Proxy models evaluated:

```text
efficientnet_b0
resnet18
clip
```

Main evaluation entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Main output areas:

```text
evaluation/proxy_models/
results/metrics/
```

The proxy evaluation covers:

- clean binary folds;
- OOD samples;
- adversarial perturbations;
- anti-forensic transformations;
- comparative clean-vs-perturbed metrics.

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
- proxy model evaluation outputs;
- forensic bundle metadata;
- normalized forensic tool outputs.

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

The next documentation alignment step is to add milestones for adversarial generation, proxy model evaluation, forensic bundle construction, forensic tool evaluation, and explainability case studies.

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
