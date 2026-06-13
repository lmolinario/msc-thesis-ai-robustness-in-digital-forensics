# Reproducibility Guide

This document describes how to reproduce, audit, and extend the experimental pipeline used in this repository.

The repository supports the MSc thesis:

```text
Evaluating the Robustness of AI-based Forensic Tools under Adversarial and Anti-Forensic Attacks
```

The project evaluates the operational robustness of AI-based image-classification and media-triage systems in Digital/Computer Forensics. The objective is not to provide an unrestricted raw-image benchmark, but to document a traceable and controlled forensic AI evaluation workflow.

---

## 1. Repository scope

The repository includes:

- acquisition and dataset-preparation scripts;
- frozen manifests and audit metadata;
- human-in-the-loop final selection records;
- proxy-model training and evaluation scripts;
- adversarial and anti-forensic perturbation scripts;
- blind forensic evaluation bundle construction;
- commercial black-box tool normalization scripts;
- metric outputs and thesis-ready result tables;
- Integrated Gradients explainability workflow;
- LaTeX thesis source files and documentation.

The repository does not publicly redistribute the full raw dataset bundle. Raw data access is governed by `DATA_ACCESS.md`.

---

## 2. Environment setup

Clone the repository:

```bash
git clone https://github.com/lmolinario/msc-thesis-ai-robustness-in-digital-forensics.git
cd msc-thesis-ai-robustness-in-digital-forensics
```

Create and activate a Python virtual environment.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled systems, install `torch` and `torchvision` according to the official PyTorch selector before running GPU-dependent stages.

---

## 3. Controlled raw data access

The raw dataset bundle is restored through:

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The script does not contain a hardcoded download URL. After controlled access has been granted, configure the URL locally through:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

Windows PowerShell:

```powershell
$env:FAIRLAB_RAW_DATASET_BUNDLE_URL="<controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Linux/macOS:

```bash
export FAIRLAB_RAW_DATASET_BUNDLE_URL="<controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Do not commit private URLs, `.env` files, API keys, or session files.

---

## 4. Official pipeline sequence

The repository follows a numbered research pipeline:

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

## 5. Final dataset and evaluation bundle

Official frozen dataset:

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

Official binary subset:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Distribution:

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

Forensic evaluation bundle:

```text
datasets/forensic_evaluation_bundle/
```

Bundle composition:

| Condition | Files |
|---|---:|
| Clean | 1000 |
| OOD | 500 |
| Adversarial | 5000 |
| Anti-forensic | 5000 |
| **Total** | **11500** |

For commercial tool processing, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

These directories contain ground-truth labels, source metadata, perturbation metadata, and hash mappings.

---

## 6. Proxy models

Transparent proxy models used in the thesis:

```text
efficientnet_b0
resnet18
clip
```

Training entry point:

```bash
python models/scripts/12_train_proxy_models.py \
  --model resnet18 efficientnet_b0 clip \
  --fold all \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --validation-ratio 0.15 \
  --seed 42 \
  --device auto \
  --input-size 224 \
  --num-workers 2
```

Evaluation entry point:

```bash
python evaluation/scripts/15_evaluate_proxy_models.py \
  --model efficientnet_b0 resnet18 clip \
  --device auto
```

Main outputs:

```text
evaluation/proxy_models/
results/metrics/
```

---

## 7. Perturbation generation

Adversarial attacks:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

Anti-forensic transformations:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Entry points:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
python datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

Attack outputs are stored under:

```text
attacks/adversarial/
attacks/anti_forensic/
attacks/manifests/
```

---

## 8. Final commercial black-box tool perimeter

The final experimental perimeter is restricted to the following tools:

| Tool | Version / module | Role in the thesis |
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

X-Ways, Oxygen, Autopsy, and earlier UFED-oriented wording may appear only as historical or non-final references. They must not be described as final evaluated tools.

Commercial-tool outputs are normalized through:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Main outputs:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

The commercial tools are evaluated as black boxes. The pipeline does not assume access to internal models, proprietary thresholds, training data, or calibrated confidence scores.

---

## 9. Explainability workflow

Explainability uses Integrated Gradients on transparent proxy models only.

Entry points:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
```

The XAI layer is qualitative and diagnostic. It is not a primary robustness metric and is not used to explain proprietary black-box commercial forensic tools.

---

## 10. Traceability and auditability

The pipeline preserves traceability through:

- stable image identifiers;
- SHA256 hashes;
- MD5 hashes for forensic-tool compatibility;
- source metadata;
- fold identifiers;
- attack and transformation manifests;
- bundle manifests;
- commercial-tool normalization logs;
- metric CSV/JSON files;
- thesis table and figure references.

SHA256 is the primary integrity hash used in the thesis pipeline.

---

## 11. Reproducibility limitations

Full reproduction requires access to controlled raw data and, for the black-box evaluation layer, access to proprietary commercial forensic tools. Therefore, the repository supports controlled reproducibility rather than unrestricted end-to-end public reruns.

Where raw images or proprietary tool outputs cannot be redistributed, auditability is preserved through code, manifests, hashes, normalized outputs, metrics, and documentation.

---

## 12. Related policy files

- `DATA_ACCESS.md` describes controlled raw dataset access.
- `SECURITY.md` describes handling of exposed secrets or private data links.
- `.env.example` lists safe environment variable names without secrets.
- `CITATION.cff` provides repository citation metadata.
- `ACADEMIC_REPOSITORY_AUDIT.md` records repository-level academic readiness checks.
