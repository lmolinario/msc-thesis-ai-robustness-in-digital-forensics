# MSc Thesis – AI Robustness in Digital Forensics

This repository contains the working research pipeline for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the **operational robustness of AI-based image classification systems in digital forensic scenarios**. The workflow compares local proxy AI models and commercial forensic AI tools under clean inputs, adversarial perturbations, anti-forensic transformations, and out-of-distribution samples.

The focus of the project is **Digital/Computer Forensics**, not Adversarial Machine Learning as an isolated optimization problem. Adversarial and anti-forensic manipulations are used as experimental stressors to assess reliability, traceability, robustness, and the operational risk of AI-assisted triage in forensic workflows.

At this stage, this is a **complete working research repository**. Datasets, generated outputs, manifests, reports, and intermediate artifacts may be versioned to support continuity across multiple workstations. A cleaned public release can be derived later.

---

## Current Status

The repository is aligned with the following operational state.

| Stage | Status | Main artifacts |
|---|---|---|
| Dataset acquisition | Completed | `datasets/scripts/acquisition/` |
| Prepared dataset construction | Completed | `datasets/prepared/` |
| Human-in-the-loop final selection | Completed | `datasets/final/manifests/manual_selection_final_1500.csv` |
| Frozen dataset | Completed | 1500 images: 500 `weapon`, 500 `non_weapon`, 500 `ood` |
| Binary subset | Completed | `datasets/final/manifests/manual_selection_adversarial_subset.csv` |
| Clean/OOD split generation | Completed | `datasets/splits/manifests/clean_folds_manifest.csv`, `datasets/splits/manifests/ood_eval_manifest.csv` |
| Proxy model training | Completed | `efficientnet_b0`, `resnet18`, `clip` |
| Adversarial attack generation | Completed | `fgsm`, `superdeepfool`, `sigma_zero`, `one_pixel`, `color_shift` |
| Anti-forensic transformation generation | Completed | `jpeg_recompression`, `resample_resize`, `gaussian_blur`, `histogram_modification`, `contrast_stretching` |
| Proxy model evaluation | Completed | `evaluation/proxy_models/proxy_model_predictions.csv`, `results/metrics/` |
| Forensic evaluation bundle | Generated and validated | `datasets/forensic_evaluation_bundle/` |
| Commercial forensic tool evaluation | Pending | `forensic_tools/`, `evaluation/forensic_tools/` |
| Explainability case studies | Prepared, not yet produced | `explainability/scripts/17_generate_integrated_gradients_case_studies.py` |

Proxy evaluation summary:

```text
models          = efficientnet_b0, resnet18, clip
input_samples   = 11500
prediction_rows = 40500
errors          = 0
```

Forensic evaluation bundle summary:

```text
clean          = 1000
ood            = 500
adversarial    = 5000
anti_forensic  = 5000
total          = 11500
```

Bundle validation checks are positive:

```text
bundle_id_unique                         = true
sha256_actual_unique                     = true
all_sha256_match_when_manifest_present   = true
blind_paths_semantically_clean           = true
metadata_separated_from_tool_input       = true
```

---

## Immediate Operational Focus

The next work block is the **commercial forensic-tool evaluation phase**.

For black-box forensic-tool evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do **not** import the following directories into forensic tools:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

Those directories contain ground-truth labels, perturbation metadata, source information, and hash mappings. They are reserved for post-export normalization and audit.

Planned forensic tools:

```text
Magnet AXIOM / Magnet.AI
X-Ways Forensics / Excire
Cellebrite UFED
Oxygen Forensic Detective
```

Expected next implementation step:

```text
evaluation/scripts/19_normalize_forensic_tool_outputs.py
```

The number `19` is intentionally reserved for forensic-tool normalization because `18_xai_interactive_launcher.py` already exists under `explainability/scripts/`.

---

## Official Dataset Artifacts

The official frozen dataset is:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

Distribution:

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| `ood` | 500 |
| **Total** | **1500** |

The official binary subset used for clean folds and perturbation generation is:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Distribution:

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

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
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
    ↓
models/scripts/12_train_proxy_models.py
    ↓
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
    ↓
attacks/
attacks/manifests/
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
commercial forensic tools
    ↓
evaluation/scripts/19_normalize_forensic_tool_outputs.py
    ↓
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
    ↓
explainability/scripts/17_generate_integrated_gradients_case_studies.py
```

`datasets/forensic_evaluation_bundle/` is the operational bridge between the local experimental pipeline and the forensic AI tool evaluation phase. It provides blind tool inputs while preserving internal traceability through metadata and hashes.

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
│   └── scripts/
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
17_generate_integrated_gradients_case_studies.py
18_xai_interactive_launcher.py
```

Planned next numbered script:

```text
19_normalize_forensic_tool_outputs.py
```

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

Model-dependent adversarial attacks are generated against the EfficientNet-B0 proxy target. Color Shift is treated as model-agnostic. Anti-forensic transformations are model-agnostic image-processing transformations.

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
- comparative clean-vs-perturbed metrics;
- final metric tables prepared for thesis reporting and later forensic-tool comparison.

---

## Forensic Evaluation Bundle

The forensic evaluation bundle is located at:

```text
datasets/forensic_evaluation_bundle/
```

Structure:

```text
datasets/forensic_evaluation_bundle/
├── metadata/
│   ├── bundle_manifest.csv
│   ├── bundle_hashes_sha256.csv
│   └── bundle_summary.json
├── blind_tool_input/
│   └── files/
└── structured_audit_view/
```

Purpose:

- `blind_tool_input/files/`: flat, semantically neutral file input for commercial forensic tools;
- `metadata/`: ground truth, hash mappings, perturbation metadata, and audit metadata;
- `structured_audit_view/`: human-readable audit organization, not intended for tool import.

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
- future normalized forensic tool outputs.

The hash-based mapping is especially important because forensic tools may rename files, alter export structures, or provide different reporting formats.

---

## Progress Tracking

The `progress/` directory documents operational progress and decisions.

Current milestone structure:

```text
progress/milestones/
├── 01_dataset_acquisition.md
├── 02_prepared_dataset.md
├── 03_manual_selection.md
├── 04_split_generation.md
├── 05_attack_generation.md
├── 06_proxy_model_training.md
├── 07_proxy_model_evaluation.md
├── 08_forensic_evaluation_bundle.md
├── 09_commercial_forensic_tools_evaluation.md
└── 10_xai_case_studies.md
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
