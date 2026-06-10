<p align="center">
  <img src="docs/assets/repository_header.png" alt="Evaluating the Robustness of AI-based Forensic Tools" width="100%">
</p>

# MSc Thesis – AI Robustness in Digital Forensics

## Evaluating the Robustness of AI-based Forensic Tools under Adversarial and Anti-Forensic Attacks

This repository contains the working research pipeline for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the **operational robustness of AI-based image classification and media-triage systems in digital forensic scenarios**. The workflow compares transparent local proxy models and commercial black-box forensic / AI-assisted tools under clean inputs, out-of-distribution samples, adversarial perturbations, and anti-forensic image transformations.

The focus of the project is **Digital/Computer Forensics**, not Adversarial Machine Learning as an isolated optimization problem. Adversarial and anti-forensic manipulations are used as controlled experimental stressors to assess reliability, traceability, robustness, and operational risk in AI-assisted forensic triage.

---

## Current Operational Status

The repository is aligned with the following consolidated state.

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
| Commercial forensic-tool evaluation | Completed and normalized | Magnet AXIOM / Magnet.AI, Excire Foto 2025, Cellebrite Inseyets, Magnet Griffeye / T3K CORE |
| Explainability / XAI | Completed and integrated in Chapter 5 | Five representative Integrated Gradients case studies selected and included in the thesis text |
| Thesis reporting | In progress | `docs/LatexThesis_ITA/` |

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

Commercial forensic-tool normalization summary:

```text
bundle_rows                         = 11500
tools_requested                     = magnet_axiom, excire_foto_2025, cellebrite_inseyets, griffeye
normalized_rows_after_deduplication = 69000
matched_rows_after_deduplication    = 69000
unmatched_rows_after_deduplication  = 0
interpretable_rows_after_dedup      = 69000
weapon_detected=unknown             = 0
metric_outputs_consistent           = true
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

The reproducible proxy-model layer, the commercial black-box tool layer, the forensic evaluation bundle, and the Integrated Gradients/XAI case studies are consolidated.

The remaining operational focus is:

```text
revise Chapter 4 to include Griffeye / T3K CORE in the commercial-tool methodology
revise Chapter 5 to include Griffeye in the commercial-tool results and comparison
revise Chapter 6 / final discussion around tool-dependent operational robustness
perform final thesis-wide consistency review
prepare the future English version
```

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

The consolidated commercial / black-box evaluation perimeter is:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673
- Excire Foto 2025, version 4.1.5
- Cellebrite Inseyets, version 10.9
- Magnet Griffeye x64, version 26.2.108, with T3K CORE v1.18.0

Excluded from the final experimental perimeter:
- Oxygen Forensic Detective
- Autopsy
```

Official forensic-tool prediction normalization entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

The number `19` is intentionally reserved for forensic-tool output normalization because `17` and `18` are already used by the Integrated Gradients/XAI workflow.

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

OOD samples are not used to train proxy models or generate adversarial attacks. They are evaluated separately as an operational robustness risk.

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
Commercial black-box tools:
- Magnet AXIOM / Magnet.AI
- Excire Foto 2025
- Cellebrite Inseyets
- Magnet Griffeye / T3K CORE
    ↓
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
    ↓
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
    ↓
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
    ↓
Chapter 5 representative XAI case studies
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
│   ├── forensic_evaluation_bundle/
│   └── scripts/
├── attacks/
│   ├── README.md
│   ├── adversarial/
│   ├── anti_forensic/
│   └── manifests/
├── models/
│   ├── README.md
│   ├── scripts/
│   ├── checkpoints/
│   └── reports/
├── evaluation/
│   ├── README.md
│   ├── scripts/
│   ├── proxy_models/
│   └── forensic_tools/
├── explainability/
│   ├── README.md
│   ├── scripts/
│   ├── manifests/
│   ├── logs/
│   └── outputs/
├── forensic_tools/
│   ├── README.md
│   ├── magnet_axiom/
│   ├── excire_foto_2025/
│   ├── cellebrite_inseyets/
│   └── griffeye/
├── results/
│   ├── README.md
│   └── metrics/
├── docs/
│   ├── README.md
│   └── LatexThesis_ITA/
└── progress/
    ├── README.md
    ├── milestones/
    ├── logs/
    └── notes/
```

---

## Official Script Sequence

The operational pipeline uses numbered scripts as official entry points:

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
19_normalize_forensic_ai_tool_predictions.py
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

Main proxy evaluation entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Main proxy output areas:

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
- thesis-ready metric tables for proxy models.

Commercial tool evaluation is consolidated through normalized outputs in:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

The commercial / black-box tools are evaluated as observable operational systems. The normalization layer does not assume access to internal models, proprietary training data, undocumented thresholds, or calibrated confidence scores.

### Commercial tool global binary comparison

The following table reports the global binary `weapon` / `non_weapon` metrics on the 11,000 binary bundle items. OOD behavior is reported separately as `OOD flag rate` over the 500 OOD samples.

| Tool / configuration | Accuracy | Recall weapon | FNR | FPR | OOD flag rate |
|---|---:|---:|---:|---:|---:|
| Magnet Griffeye / T3K CORE | 0.971727 | 0.950727 | 0.049273 | 0.007273 | 0.260000 |
| Cellebrite Inseyets 10.9 | 0.958091 | 0.957818 | 0.042182 | 0.041636 | 0.292000 |
| Magnet AXIOM / Magnet.AI | 0.933364 | 0.901455 | 0.098545 | 0.034727 | 0.360000 |
| Excire Foto 2025 D20 | 0.910727 | 0.857091 | 0.142909 | 0.035636 | 0.238000 |
| Excire Foto 2025 D50 | 0.924545 | 0.948545 | 0.051455 | 0.099455 | 0.340000 |
| Excire Foto 2025 D80 | 0.887091 | 0.981273 | 0.018727 | 0.207091 | 0.522000 |

Operational interpretation: commercial-tool robustness is **tool-dependent**. Griffeye achieves the highest global accuracy and the lowest false-positive rate under the primary firearm-only mapping, while Cellebrite provides slightly higher weapon recall. Excire D80 maximizes recall but at the cost of many more false positives. Magnet AXIOM / Magnet.AI remains operationally useful but produces more false negatives on the weapon class.

---

## Griffeye / T3K CORE Normalization

Magnet Griffeye is evaluated as a commercial black-box forensic media-triage tool through automatic semantic bookmarks generated by T3K CORE.

```text
Tool          = Magnet Griffeye x64
Version       = 26.2.108
AI module     = T3K CORE
Module version= 1.18.0
Run folder    = forensic_tools/griffeye/raw_exports/FAIRLAB_GRIFFEYE_T3_RUN_01
```

The primary thesis mapping is firearm-oriented:

```text
weapon_detected = true  if Bookmarks contains CORE/Violence/Firearm
weapon_detected = false otherwise
```

The following bookmarks are intentionally excluded from the primary metric and retained only as secondary semantic indicators:

```text
CORE/Violence/Explosive Weapon
CORE/Violence/Bladed Weapon
CORE/Violence/Archery Weapon
CORE/Military/Military Equipment
```

Official Griffeye outputs:

```text
evaluation/forensic_tools/griffeye_normalized_predictions.csv
results/metrics/griffeye_metrics.csv
```

Griffeye normalization checks:

```text
rows in normalized prediction file = 11501 (1 header + 11500 predictions)
matched_rows                       = 11500
unmatched_rows                     = 0
unknown_rows                       = 0
positive firearm bookmarks         = 5399
negative / no-firearm rows         = 6101
```

---

## Explainability Status

The explainability workflow uses Integrated Gradients on transparent proxy models. It is intended as qualitative diagnostic support for Chapter 5, not as a primary robustness metric.

Official XAI entry points:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
```

The Chapter 5 XAI selection has been completed and integrated into the thesis text. The selected representative cases are:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

Integrated Gradients are generated only for transparent proxy models and are not claimed as explanations of commercial black-box forensic tools.

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
- normalized forensic-tool outputs;
- XAI case-study manifests and thesis figure references where applicable.

The hash-based mapping is especially important because forensic tools may rename files, alter export structures, or provide different reporting formats.

---

## Dataset Availability

This repository is currently maintained as a working research repository. During development, data and generated outputs may be versioned to allow continuation across multiple machines.

For a future public release, raw images, proprietary exports, installer files, license files, sensitive generated artifacts, and heavy tool outputs may be removed or replaced with manifests, hashes, sample images where permissible, documentation, reproducible scripts, and aggregated metrics.

---

## Research Context

This work is developed within the MSc program in:

> Computer Engineering, Cybersecurity and Artificial Intelligence

and is aligned with research areas including Digital Forensics, AI-based forensic analysis, Adversarial Machine Learning, Anti-forensics, Robustness Evaluation, and Explainable AI.

---

## Citation

This repository contains the experimental artifacts developed for the following forthcoming Master's thesis:

> Molinario, L. (2026). *Evaluating the Robustness of AI-based Forensic Tools under Adversarial and Anti-Forensic Attacks*. Master's thesis, University of Cagliari, Department of Engineering, Master's Degree in Computer Engineering, Cybersecurity and Artificial Intelligence. Academic Year 2025/2026. Thesis in preparation.

### BibTeX

```bibtex
@mastersthesis{molinario2026robustnessforensictools,
  author  = {Molinario, Lello},
  title   = {Evaluating the Robustness of {AI}-based Forensic Tools under Adversarial and Anti-Forensic Attacks},
  school  = {University of Cagliari, Department of Engineering},
  type    = {Master's thesis},
  address = {Cagliari, Italy},
  year    = {2026},
  note    = {Master's Degree in Computer Engineering, Cybersecurity and Artificial Intelligence. Academic Year 2025/2026. Thesis in preparation.}
}
```

---

## License

Code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

Thesis text, figures, documentation, datasets, images, forensic-tool exports, model checkpoints and generated artifacts are not automatically covered by the code license unless explicitly stated.

Datasets, images and forensic artifacts may be subject to third-party terms, ethical restrictions, legal constraints, source-specific limitations, or institutional handling requirements. They should not be redistributed unless their redistribution status has been verified.
