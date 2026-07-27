# Reproducibility Guide

This document describes how to audit and reproduce the experimental pipeline
supporting:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

The artifact supports controlled reproducibility. It does not provide
unrestricted public redistribution of the full image corpus or licensed
commercial software.

## 1. Repository Scope

Public `main` includes:

- numbered acquisition, preparation, training, perturbation, evaluation, and reporting scripts;
- frozen dataset, split, attack, and bundle manifests;
- proxy checkpoints and prediction outputs;
- 69,000 sanitized commercial-tool decisions;
- 186 commercial metric rows;
- proxy robustness and OOD metric tables;
- canonical XAI selection and thesis-ready assets;
- the authoritative LaTeX thesis source;
- validation and audit utilities;
- authoritative complete-ZIP checksums for the controlled image artifacts.

Public `main` excludes image corpora and complete commercial raw exports.

## 2. Environment Setup on Kali/Linux

```bash
git clone https://github.com/lmolinario/msc-thesis-ai-robustness-in-digital-forensics.git
cd msc-thesis-ai-robustness-in-digital-forensics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For GPU-dependent stages, install a PyTorch build compatible with the local CUDA
environment.

## 3. Controlled Data Access

Access conditions are governed by:

```text
docs/artifact/DATA_ACCESS.md
```

Authoritative archive-level digests are stored in:

```text
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

Two controlled artifacts serve different purposes:

| Artifact | Purpose |
|---|---|
| `00_raw_datasets_bundle.zip` | Restore the heterogeneous source corpora and regenerate the numbered pipeline |
| `16_frozen_forensic_evaluation_bundle.zip` | Restore the exact 11,500 files used for commercial black-box processing |

### Raw source restoration

Request access:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --request-access
```

After approval, validate and extract the downloaded archive locally:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

### Exact frozen-bundle restoration

After the stable request page has been configured, request access with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact frozen \
  --request-access
```

Restore the exact black-box input with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact frozen \
  --archive "/path/to/16_frozen_forensic_evaluation_bundle.zip"
```

The script automatically verifies the complete archive against the authoritative
repository checksum. Frozen restoration additionally verifies all 11,500 blind
files against:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
```

Authorized direct URLs may be supplied through `--url` or the corresponding
local environment variables. Never commit private, signed, or temporary values.

## 4. Numbered Experimental Pipeline

| Step | Entry point | Purpose |
|---:|---|---|
| 00 | `datasets/scripts/acquisition/00_download_raw_datasets_bundle.py` | Controlled raw or exact frozen-bundle restoration |
| 01 | `datasets/scripts/acquisition/01_download_kaggle.py` | Kaggle-source acquisition |
| 02 | `datasets/scripts/acquisition/02_download_github.py` | GitHub-source acquisition |
| 03 | `datasets/scripts/acquisition/03_build_subset_deepfirearm.py` | DeepFirearm subset preparation |
| 04 | `datasets/scripts/acquisition/04_scrape_google.py` | Controlled web-source acquisition |
| 05 | `datasets/scripts/acquisition/05_scrape_telegram.py` | Controlled Telegram acquisition |
| 06 | `datasets/scripts/acquisition/06_scrape_youtube.py` | Controlled YouTube acquisition |
| 07 | `datasets/scripts/acquisition/07_scrape_deepweb.py` | Controlled non-indexed-source acquisition |
| 08 | `datasets/scripts/prepared/08_build_prepared_dataset.py` | Technical preparation and deduplication |
| 09 | `datasets/scripts/prepared/09_generate_review_manifest_full.py` | Human-review manifest generation |
| 10 | `datasets/scripts/final/10_manual_selection_protocol_reviewer.py` | Human-in-the-loop final freezing |
| 11 | `datasets/scripts/splits/11_generate_clean_and_ood_splits.py` | Clean folds and OOD split |
| 12 | `models/scripts/12_train_proxy_models.py` | Fold-aware proxy training |
| 13 | `datasets/scripts/attacks/13_generate_anti_forensic_attacks.py` | Anti-forensic generation |
| 14 | `datasets/scripts/attacks/14_generate_adversarial_attacks.py` | Adversarial generation |
| 15 | `evaluation/scripts/15_evaluate_proxy_models.py` | Transparent proxy evaluation |
| 16 | `datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py` | 11,500-item bundle construction |
| 17 | `explainability/scripts/17_generate_integrated_gradients_case_studies.py` | Integrated Gradients generation |
| 18 | `explainability/scripts/18_xai_interactive_launcher.py` | Human XAI review |
| 19 | `evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py` | Commercial export normalization |
| 20 | `results/scripts/20_generate_experimental_reporting_assets.py` | Thesis reporting assets for the experimental-results chapter |
| 21 | `results/scripts/21_generate_embedded_metadata_sensitivity_check.py` | Metadata-sensitivity analysis |

Public-artifact support utilities:

| Utility | Entry point | Purpose |
|---:|---|---|
| 22 | `results/scripts/22_generate_public_embedded_metadata_sensitivity_check.py` | Optional privacy-reduced analysis |
| 23 | `results/scripts/23_validate_results_artifacts.py` | Frozen-result validation |
| 24 | `results/scripts/24_audit_reporting_asset_usage.py` | Reporting/LaTeX asset audit |

## 5. Frozen Dataset and Bundle

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
```

Frozen dataset:

```text
500 weapon + 500 non-weapon + 500 OOD = 1,500 images
```

Forensic evaluation bundle:

```text
1,000 clean binary
  500 clean OOD
5,000 adversarial
5,000 anti-forensic
-------------------
11,500 files
```

For commercial processing, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import metadata or structured audit views.

## 6. Proxy Models

Architectures:

```text
efficientnet_b0
resnet18
clip
```

Registry and checkpoints:

```text
models/model_registry.json
models/checkpoints/
```

Example evaluation:

```bash
python evaluation/scripts/15_evaluate_proxy_models.py \
  --model efficientnet_b0 resnet18 clip \
  --device auto
```

Partial or diagnostic runs must use a separate output directory and must not
replace canonical frozen outputs.

## 7. Perturbations

Adversarial families:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

Anti-forensic families:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Generated image directories are local. Public manifests preserve provenance,
parameters, source identifiers, and integrity digests.

## 8. Commercial Black-Box Evaluation

Frozen perimeter:

| Configuration | Version |
|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 |
| Excire Foto D20/D50/D80 | 4.1.5 |
| Cellebrite Inseyets | 10.9 / Physical Analyzer 10.9.0.3029 |
| Griffeye / T3K CORE | 26.2.108 / 1.18.0 |

Public canonical table:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

Rebuild from committed public extracts:

```bash
python forensic_tools/scripts/build_canonical_normalized_predictions.py --force
```

Validate exact equivalence:

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --report forensic_tools/public_extracts_validation.json \
  --force
```

Expected:

```text
69,000 identical decision rows
186 identical metric rows
```

## 9. Explainability

Integrated Gradients is applied only to transparent proxy models.

Final five-case manifest:

```text
explainability/manifests/chapter5/thesis_selection.csv
```

Validate:

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py \
  --strict-thesis-text
```

## 10. Result and Reporting Validation

```bash
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

The result validator checks 69,000 commercial decisions, 186 commercial metrics,
40,500 proxy predictions, the `500 OOD images × 5 folds = 2,500 predictions per
architecture` accounting, Chapter 5 manifest counts, and metadata-sensitivity
counts.

## 11. LaTeX Audit and Compilation

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex

cd docs/LatexThesis
latexmk -pdf main.tex
cd ../..
```

Generated auxiliary files and `main.pdf` are ignored.

## 12. Kali/Linux Audit Helper

```bash
bash tools/tasks.sh status
bash tools/tasks.sh check-json
bash tools/tasks.sh check-python-syntax
bash tools/tasks.sh check-text-guards
bash tools/tasks.sh check-xai
bash tools/tasks.sh check-results
bash tools/tasks.sh check-assets
bash tools/tasks.sh check-latex-images
bash tools/tasks.sh check-thesis-log
bash tools/tasks.sh audit-all
```

## 13. Traceability

The pipeline preserves traceability through stable identifiers, SHA-256 digests,
bundle IDs, fold assignments, attack manifests, checkpoint hashes, normalized
decisions, metric tables, XAI manifests, and figure-generation records.

SHA-256 is the primary integrity digest. MD5 is retained only where required for
compatibility with commercial-tool matching workflows.

Archive-level and per-file integrity are distinct:

- `CONTROLLED_ARTIFACT_CHECKSUMS.sha256` authenticates the complete distributed ZIP;
- `bundle_hashes_sha256.csv` authenticates each restored blind input.

## 14. Reproducibility Boundary

Publicly reproducible:

- repository and code audit;
- canonical commercial-table reconstruction;
- commercial metric recomputation from sanitized decisions;
- result and reporting validation;
- LaTeX source and figure audit.

Controlled:

- raw image restoration;
- complete image-pipeline rerun;
- exact restoration of the frozen 11,500-file black-box input;
- proxy retraining and attack regeneration.

Licensed:

- commercial-tool reprocessing.

Exact frozen-bundle restoration makes the original commercial inputs available
under controlled access, but it does not remove the need for compatible licensed
software or guarantee access to proprietary model internals.

## 15. Related Documents

```text
docs/artifact/DATA_ACCESS.md
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
docs/artifact/ENVIRONMENT.md
docs/artifact/DATA_DICTIONARY.md
datasets/forensic_evaluation_bundle/README.md
.github/SECURITY.md
docs/maintenance/RELEASE_CHECKLIST.md
```

The authoritative thesis source is `docs/LatexThesis/`. Substantive changes
require a new versioned release rather than silent replacement of the frozen
artifact.
