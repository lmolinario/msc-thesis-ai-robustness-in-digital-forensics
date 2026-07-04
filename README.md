<p align="center">
  <img src="docs/assets/repository_header.png" alt="Scientific workflow for AI-based forensic image-classification robustness evaluation" width="100%">
</p>

# MSc Thesis – AI Robustness in Digital Forensics

## Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks

![MSc Thesis](https://img.shields.io/badge/MSc%20Thesis-Frozen-blue)
![Research Artifact](https://img.shields.io/badge/research%20artifact-final-brightgreen)
![Data](https://img.shields.io/badge/data-controlled%20access-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Audit](https://github.com/lmolinario/msc-thesis-ai-robustness-in-digital-forensics/actions/workflows/repository-audit.yml/badge.svg)

This repository contains the frozen research artifact for an MSc thesis in **Computer Engineering, Cybersecurity and Artificial Intelligence**.

The thesis evaluates the **operational robustness of AI-based image-classification and media-triage systems in Digital/Computer Forensics**. The study compares transparent proxy models and commercial black-box forensic tools under clean inputs, out-of-distribution samples, adversarial perturbations, and anti-forensic transformations.

The repository is intended as a **research artifact**, not as a general-purpose forensic classifier or an operational investigative tool.

---

## At a glance

| Component          | Description                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Research focus     | Robustness of AI-based forensic image-classification tools                                  |
| Evaluation setting | Controlled forensic-style media triage                                                      |
| Input conditions   | Clean, OOD, adversarial, anti-forensic                                                      |
| Proxy models       | ResNet18, EfficientNet-B0, CLIP-based visual proxy                                          |
| Commercial tools   | Magnet AXIOM / Magnet.AI, Excire Foto 2025, Cellebrite Inseyets, Griffeye / T3K CORE        |
| Main outputs       | Normalized predictions, robustness metrics, XAI case studies, thesis-ready reporting assets |
| Data policy        | Controlled access; raw source data are not publicly redistributed                           |
| Repository status  | Frozen MSc thesis research artifact                                                         |

---

## Research objective

The repository supports an experimental study designed to answer the following question:

> How robust are AI-based forensic image-classification and media-triage systems when exposed to realistic adversarial and anti-forensic input manipulations?

The evaluation is organized around three complementary perspectives:

1. **Transparent proxy evaluation**
   Controlled experiments on proxy models with known architecture, fold-aware training, adversarial attacks, anti-forensic transformations, confidence analysis, and XAI inspection.

2. **Commercial black-box tool evaluation**
   Blind-bundle testing of forensic and AI-assisted media-triage tools, followed by post-export normalization and comparison against the hidden ground truth.

3. **Operational robustness analysis**
   Measurement of failure modes that matter in forensic triage, including missed weapon-related content, false alarms, unknown classifications, OOD mislabeling, confidence shifts, and robustness degradation under perturbation.

---

## Experimental workflow

The experimental protocol follows the FAIR-Lab methodology described in the thesis. It separates dataset construction, controlled proxy-model evaluation, black-box forensic-tool evaluation, output normalization, XAI inspection, and traceability controls.

```mermaid
flowchart TD
    A[Source image pool]
    B[Human-in-the-loop review]
    C[Final frozen dataset<br/>500 weapon / 500 non-weapon / 500 OOD]

    D[Binary evaluation subset<br/>1000 clean images]
    E[Clean OOD evaluation set<br/>500 clean images]

    F[Perturbation generation<br/>adversarial + anti-forensic<br/>binary subset only]

    G[Forensic evaluation bundle<br/>clean binary + clean OOD + perturbed binary samples]

    H[Controlled proxy model evaluation<br/>EfficientNet-B0 / ResNet18 / CLIP]
    I[Black-box software evaluation<br/>Magnet.AI / Excire Foto / Inseyets / Griffeye-T3K]

    J[Metrics, normalization, XAI and traceability]

    A --> B --> C
    C --> D
    C --> E
    D --> F
    D --> G
    E --> G
    F --> G
    G --> H
    G --> I
    H --> J
    I --> J
```

---

## Frozen status

| Area                                | Status                  | Main artifacts                                            |
| ----------------------------------- | ----------------------- | --------------------------------------------------------- |
| Dataset construction                | Completed               | `datasets/`                                               |
| Proxy model training and evaluation | Completed               | `models/`, `evaluation/proxy_models/`, `results/metrics/` |
| Perturbation generation             | Completed               | `attacks/`                                                |
| Forensic evaluation bundle          | Generated and validated | `datasets/forensic_evaluation_bundle/`                    |
| Commercial-tool normalization       | Completed               | `evaluation/forensic_tools/`, `forensic_tools/`           |
| XAI case studies                    | Completed               | `explainability/`                                         |
| Thesis source                       | Completed and frozen    | `docs/LatexThesis/`                                       |

---

## Quick navigation

| Document                       | Purpose                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `THESIS_ARTIFACT.md`           | Official declaration of the thesis research artifact, academic context, boundaries, and source-of-truth areas |
| `REPOSITORY_MAP.md`            | Directory-level map linking repository areas to the thesis workflow                                           |
| `ARTIFACT_EVALUATION.md`       | Evaluation statement defining what can be audited, reproduced, or only reproduced under controlled access     |
| `DATA_DICTIONARY.md`           | Interpretation guide for CSV/JSON manifests, predictions, normalized outputs, and metrics                     |
| `ENVIRONMENT.md`               | Execution-environment notes, dependency expectations, and reproducibility boundaries                          |
| `REPRODUCIBILITY.md`           | Controlled reproducibility workflow                                                                           |
| `DATA_ACCESS.md`               | Raw-data and controlled-access policy                                                                         |
| `SECURITY.md`                  | Secret, proprietary-data, and exposure-handling policy                                                        |
| `ACADEMIC_REPOSITORY_AUDIT.md` | Academic repository audit record                                                                              |
| `RELEASE_CHECKLIST.md`         | Checklist for final GitHub release, release assets, and DOI archival                                          |
| `CHANGELOG.md`                 | Repository-level changelog for thesis-artifact release management                                             |

---

## Official dataset artifacts

Official frozen dataset manifest:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

Official binary subset used for proxy training, adversarial perturbation generation, and binary robustness evaluation:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Out-of-distribution samples remain separate from proxy training and adversarial generation. They are evaluated as an operational robustness risk rather than as an additional supervised training class.

---

## Forensic evaluation bundle

For black-box forensic-tool evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do **not** import:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

Those directories contain ground-truth labels, perturbation metadata, source information, and hash mappings. They are reserved for post-export normalization and audit.

---

## Final commercial-tool perimeter

| Tool                       | Version / module                       | Role                                                      |
| -------------------------- | -------------------------------------- | --------------------------------------------------------- |
| Magnet AXIOM / Magnet.AI   | 10.1.0.48673                           | Commercial forensic AI categorization                     |
| Excire Foto 2025           | 4.1.5                                  | Standalone AI-assisted semantic image retrieval           |
| Cellebrite Inseyets        | 10.9                                   | Commercial black-box AI-assisted media analysis           |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE 1.18.0 | Commercial forensic media triage and semantic bookmarking |

Official normalization entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

---

## Main artifact areas

```text
msc-thesis-ai-robustness-in-digital-forensics/
├── datasets/          # Dataset manifests, splits, bundle construction, controlled metadata
├── attacks/           # Generated adversarial and anti-forensic perturbation artifacts
├── models/            # Proxy model training scripts, checkpoints, and registries
├── evaluation/        # Proxy and commercial-tool evaluation / normalization scripts
├── explainability/    # XAI case studies and inspection utilities
├── forensic_tools/    # Commercial-tool export normalization support
├── results/           # Metrics, tables, figures, and thesis-ready reporting assets
├── docs/              # Thesis source, repository assets, and documentation
├── progress/          # Internal progress and audit records
├── .github/           # Lightweight repository-audit workflow
└── tasks.ps1          # Local audit helper
```

---

## Reproducibility boundary

This repository supports **controlled reproducibility**.

The following components can be audited or reproduced from the repository structure and scripts:

* dataset manifest construction logic;
* split generation;
* proxy model training workflow;
* perturbation generation workflow;
* proxy-model evaluation;
* commercial-tool output normalization logic;
* metric generation;
* XAI reporting assets;
* thesis figures and tables generated from frozen outputs.

The following components require controlled access or external software:

* raw source image datasets;
* commercial forensic-tool execution;
* proprietary commercial-tool exports where redistribution is not permitted;
* local environments requiring licensed forensic software.

The repository does **not** expose public raw dataset download links. Controlled restoration uses:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

See:

* `DATA_ACCESS.md` for controlled raw dataset access;
* `.env.example` for safe local environment variable names;
* `REPRODUCIBILITY.md` for the reproducibility workflow;
* `SECURITY.md` for secret and data-exposure handling.

---

## Local audit helper

A lightweight PowerShell helper is available for non-destructive repository checks:

```powershell
.\tasks.ps1 status
.\tasks.ps1 check-json
.\tasks.ps1 check-python-syntax
.\tasks.ps1 check-text-guards
.\tasks.ps1 check-thesis-log
.\tasks.ps1 audit-all
```

The GitHub Actions workflow under:

```text
.github/workflows/repository-audit.yml
```

runs lightweight JSON, Python syntax, and stale-pattern checks on push and pull request events.

---

## Official script sequence

The official experimental sequence is listed below for auditability. Detailed execution notes are provided in `REPRODUCIBILITY.md` and `REPOSITORY_MAP.md`.

<details>
<summary>Show official script sequence</summary>

| Step | Script                                               | Path                                                                      | Purpose                                                |
| ---: | ---------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------ |
|   00 | `00_download_raw_datasets_bundle.py`                 | `datasets/scripts/acquisition/00_download_raw_datasets_bundle.py`         | Controlled restoration of the raw dataset bundle       |
|   01 | `01_download_kaggle.py`                              | `datasets/scripts/acquisition/01_download_kaggle.py`                      | Kaggle source acquisition                              |
|   02 | `02_download_github.py`                              | `datasets/scripts/acquisition/02_download_github.py`                      | GitHub-based source acquisition                        |
|   03 | `03_build_subset_deepfirearm.py`                     | `datasets/scripts/acquisition/03_build_subset_deepfirearm.py`             | DeepFirearm subset preparation                         |
|   04 | `04_scrape_google.py`                                | `datasets/scripts/acquisition/04_scrape_google.py`                        | Controlled Google-derived source collection            |
|   05 | `05_scrape_telegram.py`                              | `datasets/scripts/acquisition/05_scrape_telegram.py`                      | Controlled Telegram-derived source collection          |
|   06 | `06_scrape_youtube.py`                               | `datasets/scripts/acquisition/06_scrape_youtube.py`                       | Controlled YouTube-derived source collection           |
|   07 | `07_scrape_deepweb.py`                               | `datasets/scripts/acquisition/07_scrape_deepweb.py`                       | Controlled non-indexed web-source collection           |
|   08 | `08_build_prepared_dataset.py`                       | `datasets/scripts/prepared/08_build_prepared_dataset.py`                  | Technical preparation of the candidate image pool      |
|   09 | `09_generate_review_manifest_full.py`                | `datasets/scripts/prepared/09_generate_review_manifest_full.py`           | Full review manifest generation                        |
|   10 | `10_manual_selection_protocol_reviewer.py`           | `datasets/scripts/final/10_manual_selection_protocol_reviewer.py`         | Manual selection and freezing protocol                 |
|   11 | `11_generate_clean_and_ood_splits.py`                | `datasets/scripts/splits/11_generate_clean_and_ood_splits.py`             | Clean binary folds and OOD evaluation split generation |
|   12 | `12_train_proxy_models.py`                           | `models/scripts/12_train_proxy_models.py`                                 | Fold-aware proxy model training                        |
|   13 | `13_generate_anti_forensic_attacks.py`               | `datasets/scripts/attacks/13_generate_anti_forensic_attacks.py`           | Anti-forensic transformation generation                |
|   14 | `14_generate_adversarial_attacks.py`                 | `datasets/scripts/attacks/14_generate_adversarial_attacks.py`             | Adversarial perturbation generation                    |
|   15 | `15_evaluate_proxy_models.py`                        | `evaluation/scripts/15_evaluate_proxy_models.py`                          | Transparent proxy-model evaluation                     |
|   16 | `16_build_forensic_evaluation_bundle.py`             | `datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py`          | Blind forensic evaluation bundle construction          |
|   17 | `17_generate_integrated_gradients_case_studies.py`   | `explainability/scripts/17_generate_integrated_gradients_case_studies.py` | Integrated Gradients case-study generation             |
|   18 | `18_xai_interactive_launcher.py`                     | `explainability/scripts/18_xai_interactive_launcher.py`                   | Interactive XAI inspection launcher                    |
|   19 | `19_normalize_forensic_ai_tool_predictions.py`       | `evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py`         | Commercial forensic-tool output normalization          |
|   20 | `20_generate_experimental_reporting_assets.py`       | `results/scripts/20_generate_experimental_reporting_assets.py`            | Thesis-ready experimental reporting assets             |
|   21 | `21_generate_embedded_metadata_sensitivity_check.py` | `results/scripts/21_generate_embedded_metadata_sensitivity_check.py`      | Embedded-metadata sensitivity tables for Chapter 5     |

</details>

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
├── .github/
├── THESIS_ARTIFACT.md
├── REPOSITORY_MAP.md
├── ARTIFACT_EVALUATION.md
├── DATA_DICTIONARY.md
├── ENVIRONMENT.md
├── RELEASE_CHECKLIST.md
├── CHANGELOG.md
├── CITATION.cff
├── DATA_ACCESS.md
├── SECURITY.md
├── REPRODUCIBILITY.md
├── ACADEMIC_REPOSITORY_AUDIT.md
├── tasks.ps1
├── .env.example
├── requirements.txt
└── LICENSE
```

---

## Citation

Citation metadata are provided in:

```text
CITATION.cff
```

A DOI badge and release citation should be added after the final GitHub release is archived through Zenodo or another institutional repository.

---

## License and data notice

The repository code is distributed under the MIT License.

Raw datasets, third-party images, controlled-access bundles, commercial forensic-tool outputs, and proprietary software artifacts are not covered by unrestricted redistribution unless explicitly stated.

This repository is released for academic transparency, auditability, and controlled reproducibility of the MSc thesis artifact.
