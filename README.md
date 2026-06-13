<p align="center">
  <img src="docs/assets/repository_header.png" alt="Evaluating the Robustness of AI-based Forensic Tools" width="100%">
</p>

# MSc Thesis – AI Robustness in Digital Forensics

## Evaluating the Robustness of AI-based Forensic Tools under Adversarial and Anti-Forensic Attacks

This repository contains the research pipeline, documentation, manifests, normalized outputs, and LaTeX thesis source for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the **operational robustness of AI-based image-classification and media-triage systems in Digital/Computer Forensics**. The workflow compares transparent local proxy models and commercial black-box forensic / AI-assisted tools under clean inputs, out-of-distribution samples, adversarial perturbations, and anti-forensic image transformations.

The focus is **Digital/Computer Forensics**, not Adversarial Machine Learning as an isolated optimization problem. Adversarial and anti-forensic manipulations are used as controlled experimental stressors to assess reliability, traceability, robustness, and operational risk in AI-assisted forensic triage.

---

## Research artifact status

This repository is organized as a controlled academic research artifact rather than a generic code dump. It includes:

- numbered execution scripts;
- frozen dataset manifests;
- human-in-the-loop selection records;
- hash-based traceability artifacts;
- proxy-model evaluation outputs;
- adversarial and anti-forensic perturbation workflows;
- blind forensic evaluation bundle construction;
- commercial black-box tool normalization;
- final metric tables;
- Integrated Gradients explainability workflow;
- LaTeX thesis source files;
- citation, data-access, security, and reproducibility documentation.

---

## Current operational status

| Stage | Status | Main artifacts |
|---|---|---|
| Dataset acquisition | Completed | `datasets/scripts/acquisition/` |
| Prepared dataset construction | Completed | `datasets/prepared/` |
| Human-in-the-loop final selection | Completed | `datasets/final/manifests/manual_selection_final_1500.csv` |
| Frozen dataset | Completed | 1500 images: 500 `weapon`, 500 `non_weapon`, 500 `ood` |
| Binary subset | Completed | `datasets/final/manifests/manual_selection_adversarial_subset.csv` |
| Clean/OOD split generation | Completed | `datasets/splits/manifests/` |
| Proxy model training | Completed | `efficientnet_b0`, `resnet18`, `clip` |
| Adversarial attack generation | Completed | `fgsm`, `superdeepfool`, `sigma_zero`, `one_pixel`, `color_shift` |
| Anti-forensic transformation generation | Completed | `jpeg_recompression`, `resample_resize`, `gaussian_blur`, `histogram_modification`, `contrast_stretching` |
| Proxy model evaluation | Completed | `evaluation/proxy_models/`, `results/metrics/` |
| Forensic evaluation bundle | Generated and validated | `datasets/forensic_evaluation_bundle/` |
| Commercial forensic-tool evaluation | Completed and normalized | Magnet AXIOM, Excire Foto 2025, Cellebrite Inseyets, Magnet Griffeye / T3K CORE |
| Explainability / XAI | Completed | Integrated Gradients case studies for Chapter 5 |
| Thesis reporting | In progress | `docs/LatexThesis_ITA/` |

---

## Official dataset artifacts

Official frozen dataset:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| `ood` | 500 |
| **Total** | **1500** |

Official binary subset:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

OOD samples are not used to train proxy models or generate adversarial attacks. They are evaluated separately as an operational robustness risk.

---

## Forensic evaluation bundle

The forensic evaluation bundle contains 11,500 files:

| Condition | Files |
|---|---:|
| Clean | 1000 |
| OOD | 500 |
| Adversarial | 5000 |
| Anti-forensic | 5000 |
| **Total** | **11500** |

For black-box forensic-tool evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

Those directories contain ground-truth labels, perturbation metadata, source information, and hash mappings. They are reserved for post-export normalization and audit.

---

## Final commercial-tool perimeter

The final commercial / black-box evaluation perimeter is:

| Tool | Version / module | Role |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Commercial forensic AI categorization |
| Excire Foto 2025 | 4.1.5 | Standalone AI-assisted semantic image retrieval |
| Cellebrite Inseyets | 10.9 | Commercial black-box AI-assisted media analysis |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE 1.18.0 | Commercial forensic media triage and semantic bookmarking |

Excluded from the final experimental perimeter:

```text
Oxygen Forensic Detective
Autopsy
X-Ways Forensics
```

Official commercial-tool normalization entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

---

## Official script sequence

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

## Repository structure

```text
msc-thesis-ai-robustness-in-digital-forensics/
├── datasets/
├── attacks/
├── models/
├── evaluation/
├── explainability/
├── forensic_tools/
├── results/
├── docs/
├── progress/
├── CITATION.cff
├── DATA_ACCESS.md
├── SECURITY.md
├── REPRODUCIBILITY.md
├── ACADEMIC_REPOSITORY_AUDIT.md
├── .env.example
├── requirements.txt
└── LICENSE
```

---

## Data access and reproducibility

The repository does **not** expose public raw dataset download links. Raw data access is controlled because the dataset includes heterogeneous source material that may be subject to legal, ethical, platform, or source-specific restrictions.

Controlled restoration uses:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

See:

- `DATA_ACCESS.md` for controlled raw dataset access;
- `.env.example` for safe local environment variable names;
- `REPRODUCIBILITY.md` for the reproducibility workflow;
- `SECURITY.md` for secret and data-exposure handling.

---

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

If this repository is used for academic review, thesis verification, or related research, cite it using the metadata in `CITATION.cff`.

---

## License

The repository code is distributed under the MIT License. Raw datasets, third-party images, commercial forensic-tool exports, and controlled-access bundles are not covered by unrestricted redistribution unless explicitly stated.
