# MSc Thesis – AI Robustness in Digital Forensics

This repository contains the working research pipeline for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis investigates the operational robustness of AI-based image classification systems in digital forensic scenarios. The experimental workflow evaluates how local AI models and forensic AI tools behave when exposed to clean images, adversarial perturbations, anti-forensic transformations, and out-of-distribution samples.

At this stage, the repository is a **complete working repository**: datasets, generated outputs, manifests, reports, and intermediate artifacts may be versioned to support continuity across multiple workstations. A cleaned public release can be derived later.

---

## Research Objective

The objective of the thesis is to assess whether AI-based image classification systems can remain reliable when forensic image inputs are manipulated with adversarial or anti-forensic intent.

The main research questions are:

- How do AI-based image classifiers behave under adversarial perturbations?
- How do realistic anti-forensic image transformations affect model and tool predictions?
- Do forensic AI tools provide stable and trustworthy results under manipulated inputs?
- How can local AI models and forensic tools be compared under a shared evaluation protocol?
- Can explainability help interpret model failures and robustness degradation?

---

## Methodological Position

The repository follows a **human-in-the-loop** experimental design.

The dataset is not treated as a fully automatic artifact. Instead, it is built through a documented workflow involving:

1. heterogeneous source acquisition;
2. technical validation and deduplication;
3. full-pool review manifest generation;
4. manual semantic selection;
5. frozen dataset construction;
6. reproducible split generation;
7. attack generation;
8. model and forensic tool evaluation;
9. metric aggregation and explainability analysis.

The official frozen dataset is:

```text
 datasets/final/manifests/manual_selection_final_1500.csv
```

The official adversarial/anti-forensic subset is:

```text
 datasets/final/manifests/manual_selection_adversarial_subset.csv
```

The previous `33_final_frozen_dataset.csv` naming convention is no longer used.

---

## Pipeline Overview

The current pipeline is organized as follows:

```text
datasets/scripts/acquisition/
    ↓
datasets/raw/
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

---

## Repository Structure

```text
msc-thesis-ai-robustness-in-digital-forensics/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── LICENSE
│
├── datasets/
│   ├── README.md
│   │
│   ├── raw/
│   │   ├── downloaded_raw_archives/
│   │   ├── 01_kaggle_weapon/
│   │   ├── 02_deepfirearm/
│   │   ├── 03_google_scraped/
│   │   ├── 04_telegram_youtube/
│   │   └── 05_deepweb/
│   │
│   ├── prepared/
│   │   ├── final_pool/
│   │   │   ├── images/
│   │   │   ├── metadata.csv
│   │   │   └── reports/
│   │   │       ├── prepared_build_summary.json
│   │   │       ├── invalid_images.csv
│   │   │       └── duplicates_discarded.csv
│   │   └── manifests/
│   │       └── review_manifest_full.csv
│   │
│   ├── final/
│   │   ├── manifests/
│   │   │   ├── manual_selection_protocol_db.csv
│   │   │   ├── manual_selection_final_1500.csv
│   │   │   ├── manual_selection_adversarial_subset.csv
│   │   │   └── manual_selection_removed.csv
│   │   └── reports/
│   │       ├── manual_selection_summary.json
│   │       ├── manual_selection_log.csv
│   │       ├── manual_selection_state.json
│   │       └── backups/
│   │
│   ├── splits/
│   │   ├── clean/
│   │   │   ├── fold_1/
│   │   │   │   ├── weapon/
│   │   │   │   └── non_weapon/
│   │   │   ├── fold_2/
│   │   │   ├── fold_3/
│   │   │   ├── fold_4/
│   │   │   └── fold_5/
│   │   ├── ood/
│   │   │   └── ood_eval_set/
│   │   │       └── ood/
│   │   └── manifests/
│   │       ├── clean_folds_manifest.csv
│   │       └── ood_eval_manifest.csv
│   │
│   ├── forensic_evaluation_bundle/
│   │   ├── README.md
│   │   ├── bundle_manifest.csv
│   │   ├── bundle_hashes_sha256.csv
│   │   ├── bundle_summary.json
│   │   ├── clean/
│   │   ├── adversarial/
│   │   ├── anti_forensic/
│   │   └── ood/
│   │
│   └── scripts/
│       ├── utils/
│       │   ├── __init__.py
│       │   └── paths.py
│       ├── acquisition/
│       │   ├── 00_download_raw_datasets_bundle.py
│       │   ├── 01_download_kaggle.py
│       │   ├── 02_download_github.py
│       │   ├── 03_build_subset_deepfirearm.py
│       │   ├── 04_scrape_google.py
│       │   ├── 05_scrape_telegram.py
│       │   ├── 06_scrape_youtube.py
│       │   └── 07_scrape_deepweb.py
│       ├── prepared/
│       │   ├── 08_build_prepared_dataset.py
│       │   └── 09_generate_review_manifest_full.py
│       ├── final/
│       │   └── 10_manual_selection_protocol_reviewer.py
│       ├── splits/
│       │   └── 11_generate_clean_and_ood_splits.py
│       └── bundle/
│           └── 12_build_forensic_evaluation_bundle.py
│
├── attacks/
│   ├── README.md
│   ├── adversarial/
│   │   ├── fgsm/
│   │   ├── superdeepfool/
│   │   ├── sigma_zero/
│   │   ├── one_pixel/
│   │   └── color_shift/
│   └── anti_forensic/
│       ├── jpeg_recompression/
│       ├── resample_resize/
│       ├── gaussian_blur/
│       ├── histogram_modification/
│       └── contrast_stretching/
│
├── models/
│   ├── README.md
│   ├── clip/
│   ├── blip/
│   ├── resnet18/
│   ├── efficientnet_b0/
│   └── svm_baseline/
│
├── evaluation/
│   ├── README.md
│   ├── clean/
│   ├── adversarial/
│   ├── anti_forensic/
│   ├── ood/
│   └── forensic_tools/
│
├── explainability/
│   ├── README.md
│   ├── scripts/
│   ├── configs/
│   ├── outputs/
│   │   ├── integrated_gradients/
│   │   └── case_studies/
│   └── manifests/
│       ├── integrated_gradients_manifest.csv
│       └── xai_case_studies_manifest.csv
│
├── forensic_tools/
│   ├── README.md
│   ├── magnet_axiom/
│   ├── xways/
│   ├── cellebrite_ufed/
│   └── oxygen/
│
├── results/
│   ├── README.md
│   ├── metrics/
│   ├── tables/
│   ├── plots/
│   ├── figures/
│   └── reports/
│
├── docs/
│   ├── methodology/
│   ├── dataset_protocol/
│   ├── experimental_protocol/
│   └── LatexThesis/
│
└── progress/
    ├── milestones/
    ├── logs/
    └── notes/
```

---

## Main Directory Roles

### `datasets/`

Contains all dataset-related artifacts, from acquisition to final splits and forensic export bundles.

Important subdirectories:

- `datasets/raw/`: raw acquired data. This directory must remain distinct from `datasets/scripts/acquisition/`.
- `datasets/scripts/acquisition/`: scripts used to acquire or reconstruct the raw sources.
- `datasets/prepared/`: technically validated, deduplicated, and indexed image pool.
- `datasets/final/`: manually selected frozen dataset and audit outputs.
- `datasets/splits/`: clean stratified folds and OOD evaluation set.
- `datasets/forensic_evaluation_bundle/`: transferable bundle for testing with forensic AI tools.

### `attacks/`

Contains adversarial and anti-forensic perturbation logic and outputs.

Adversarial attacks:

- FGSM
- SuperDeepFool
- Sigma Zero
- One Pixel Attack
- Color Shift

Anti-forensic transformations:

- JPEG recompression
- Resample and resize
- Gaussian blur
- Histogram modification
- Contrast stretching

### `models/`

Contains local AI model evaluation code, configurations, predictions, checkpoints, and reports.

Planned model families:

- CLIP
- BLIP
- ResNet18
- EfficientNet-B0
- SVM baseline

### `evaluation/`

Contains scripts, normalized outputs, and reports for evaluating:

- clean images;
- adversarial perturbations;
- anti-forensic transformations;
- OOD images;
- forensic AI tool outputs.

The goal is to normalize local model predictions and forensic tool outputs into a shared evaluation schema.

### `forensic_tools/`

Contains tool-specific protocols, exports, parsers, and reports for:

- Magnet AXIOM;
- X-Ways;
- Cellebrite UFED;
- Oxygen Forensic Detective.

Tool exports should be matched back to the original dataset primarily through SHA256, then MD5, and finally filename/image_id as a fallback.

### `explainability/`

Contains post-evaluation explainability analysis.

Explainability is not the primary metric layer. It is used for qualitative case studies, including:

- clean vs perturbed attribution maps;
- successful adversarial or anti-forensic failure cases;
- OOD false positives;
- disagreement cases between local models and forensic tools.

The expected main method is Integrated Gradients for local models where feasible.

### `results/`

Contains aggregated experimental outputs, metrics, tables, plots, figures, and final reports.

Expected metric files include:

- `clean_baseline_metrics.csv`
- `adversarial_robustness_metrics.csv`
- `anti_forensic_robustness_metrics.csv`
- `ood_metrics.csv`
- `forensic_tools_metrics.csv`
- `comparative_metrics.csv`

### `docs/`

Contains stable methodological documentation, including:

- dataset protocol;
- experimental protocol;
- threat model;
- forensic tool protocol;
- explainability protocol;
- LaTeX thesis assets.

### `progress/`

Contains operational progress tracking.

This directory is used to document what has been completed, with which scripts, using which inputs, and producing which outputs. It is intentionally separate from `docs/` and `results/`.

Suggested structure:

```text
progress/
├── milestones/
│   ├── 01_dataset_acquisition.md
│   ├── 02_prepared_dataset.md
│   ├── 03_manual_selection.md
│   ├── 04_split_generation.md
│   ├── 05_attack_generation.md
│   ├── 06_forensic_bundle.md
│   ├── 07_model_evaluation.md
│   ├── 08_forensic_tool_evaluation.md
│   └── 09_explainability.md
├── logs/
└── notes/
```

---

## Dataset Design

The final dataset is organized into three semantic groups:

- `weapon`: real, visually recognizable firearms relevant to the classification task;
- `non_weapon`: realistic negative samples with no weapon present;
- `ood`: out-of-distribution, borderline, synthetic, anomalous, or semantically non-standard samples.

The final frozen dataset contains:

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |
| ood | 500 |

The adversarial and anti-forensic experiments are performed only on the binary subset:

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |

OOD samples are evaluated separately as a single OOD evaluation set, not as adversarial attack targets.

---

## Split Strategy

The binary `weapon` / `non_weapon` subset is divided into five stratified clean folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

The repository uses the term `fold_1`, `fold_2`, etc., rather than `test_set_1`, because this naming is more consistent with experimental machine learning and cross-validation terminology.

OOD images are stored separately:

```text
datasets/splits/ood/ood_eval_set/ood/
```

The split manifests are:

```text
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
```

---

## Forensic Evaluation Bundle

The forensic evaluation bundle is designed to be copied to a USB drive, external disk, or another workstation for testing with forensic AI tools.

It contains:

- clean folds;
- adversarial perturbation outputs;
- anti-forensic transformation outputs;
- the OOD evaluation set;
- a global manifest;
- SHA256 hashes;
- a summary JSON.

Expected files:

```text
datasets/forensic_evaluation_bundle/bundle_manifest.csv
datasets/forensic_evaluation_bundle/bundle_hashes_sha256.csv
datasets/forensic_evaluation_bundle/bundle_summary.json
```

Every file in the bundle must be traceable through:

- `image_id`;
- `final_label`;
- `fold`;
- `sample_type`;
- `attack_family`;
- `attack_name`;
- `relative_path`;
- `sha256`;
- optionally `md5` for compatibility with forensic tool exports.

---

## Evaluation Metrics

The same core metrics should be used for local AI models and forensic AI tools whenever possible.

For binary clean and perturbed evaluation:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- false positive rate;
- false negative rate;
- confusion matrix.

For robustness evaluation:

- accuracy drop;
- F1 drop;
- robust accuracy;
- attack success rate.

For OOD evaluation:

- OOD weapon false positive rate;
- OOD detection-as-weapon rate;
- OOD unknown/rejected rate, if available;
- OOD category distribution, if applicable.

---

## Reproducibility and Traceability

The repository is designed around traceable artifacts:

- file-level SHA256 hashing;
- deterministic dataset preparation where possible;
- CSV manifests for each major stage;
- manual review logs;
- split manifests;
- attack manifests;
- forensic bundle manifest;
- normalized evaluation outputs.

The hash-based mapping is especially important because forensic tools may rename files, alter export structures, or provide different reporting formats.

---

## Dataset Availability

This repository is currently maintained as a working research repository. During development, data and generated outputs may be versioned to allow continuation across multiple machines.

For a future public release, raw images and sensitive generated artifacts may be removed or replaced with:

- manifests;
- hashes;
- sample images where permissible;
- documentation;
- reproducible scripts;
- aggregated metrics.

---

## Research Context

This work is developed within the MSc program in:

> Computer Engineering, Cybersecurity and Artificial Intelligence

and is aligned with research areas including:

- Digital Forensics;
- AI-based forensic analysis;
- Adversarial Machine Learning;
- Anti-forensics;
- Robustness evaluation;
- Explainable AI.

---

## Status

Work in progress.

Current focus:

1. finalize repository structure;
2. generate clean folds and OOD evaluation set;
3. generate adversarial and anti-forensic outputs;
4. build the forensic evaluation bundle;
5. evaluate local AI models and forensic AI tools under a shared metric protocol.

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
