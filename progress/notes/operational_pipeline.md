# Operational Pipeline

This document summarizes the operational workflow used by the repository for the MSc thesis on AI robustness in digital forensics.

It is intended as a working methodological note. The canonical repository entry point remains `README.md`; this file only expands the execution logic behind the pipeline.

---

## 1. Experimental input definition

The first objective is to define what enters the experimental workflow.

Official dataset artifacts:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

The final frozen dataset contains three semantic groups:

| Group | Count | Role |
|---|---:|---|
| `weapon` | 500 | Positive class for the nominal forensic image classification task |
| `non_weapon` | 500 | Negative class for the nominal forensic image classification task |
| `ood` | 500 | Out-of-distribution and borderline evaluation set |

The adversarial and anti-forensic experiments are performed on the binary subset only:

```text
weapon = 500
non_weapon = 500
```

OOD samples are evaluated separately and are not used as attack targets.

---

## 2. Data trust layer

The first layer of the pipeline builds confidence in the dataset before any model or forensic tool is evaluated.

### 2.1 Raw acquisition

Input sources:

```text
datasets/raw/01_kaggle_weapon/
datasets/raw/02_deepfirearm/
datasets/raw/03_google_scraped/
datasets/raw/04_telegram_youtube/
datasets/raw/05_deepweb/
```

Expected activities:

- collect or reconstruct raw image sources;
- preserve source-level provenance;
- avoid mixing raw sources with generated or curated images.

### 2.2 Technical preparation

Main script:

```text
datasets/scripts/prepared/08_build_prepared_dataset.py
```

Expected outputs:

```text
datasets/prepared/final_pool/images/
datasets/prepared/final_pool/metadata.csv
datasets/prepared/final_pool/reports/prepared_build_summary.json
datasets/prepared/final_pool/reports/invalid_images.csv
datasets/prepared/final_pool/reports/duplicates_discarded.csv
```

Main checks:

- recursive image scanning;
- image validity;
- metadata reconstruction;
- SHA256 hashing;
- global duplicate removal;
- stable copy into the prepared image pool.

### 2.3 Review manifest generation

Main script:

```text
datasets/scripts/prepared/09_generate_review_manifest_full.py
```

Expected output:

```text
datasets/prepared/manifests/review_manifest_full.csv
```

This manifest bridges technical preparation and human-in-the-loop semantic selection.

### 2.4 Manual semantic selection

Main script:

```text
datasets/scripts/final/10_manual_selection_protocol_reviewer.py
```

Expected outputs:

```text
datasets/final/manifests/manual_selection_protocol_db.csv
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/final/manifests/manual_selection_removed.csv
datasets/final/reports/manual_selection_summary.json
datasets/final/reports/manual_selection_log.csv
datasets/final/reports/manual_selection_state.json
```

The manual reviewer is part of the method, not a temporary workaround. It documents the human-in-the-loop labeling and selection protocol used to freeze the benchmark.

### 2.5 Clean and OOD split generation

Main script:

```text
datasets/scripts/splits/11_generate_clean_and_ood_splits.py
```

Expected outputs:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
datasets/splits/ood/ood_eval_set/
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
```

The repository uses the naming `fold_1` ... `fold_5`, not `test_set_1` ... `test_set_5`.

---

## 3. Measurement trust layer

The second layer builds confidence in the measurement process.

### 3.1 Clean baseline evaluation

Planned model families:

- ResNet18;
- EfficientNet-B0;
- CLIP;
- BLIP;
- SVM baseline.

Expected outputs:

```text
evaluation/clean/
results/metrics/clean_baseline_metrics.csv
```

Core metrics:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- false positive rate;
- false negative rate.

### 3.2 Perturbation generation

Adversarial attacks:

- FGSM;
- SuperDeepFool;
- Sigma Zero;
- One Pixel Attack;
- Color Shift.

Anti-forensic transformations:

- JPEG recompression;
- Resample and resize;
- Gaussian blur;
- Histogram modification;
- Contrast stretching.

Expected structure:

```text
attacks/adversarial/
attacks/anti_forensic/
```

Each attack output should be traceable to the clean source image through `image_id`, fold, label, relative path, and hash.

### 3.3 Evaluation under perturbation

Expected outputs:

```text
evaluation/adversarial/
evaluation/anti_forensic/
results/metrics/adversarial_robustness_metrics.csv
results/metrics/anti_forensic_robustness_metrics.csv
```

Robustness metrics:

- clean accuracy vs perturbed accuracy;
- F1 drop;
- robust accuracy;
- attack success rate;
- misclassification rate;
- confidence shift.

### 3.4 Explainability

Expected method:

- Integrated Gradients where feasible.

Expected outputs:

```text
explainability/outputs/integrated_gradients/
explainability/outputs/case_studies/
explainability/manifests/integrated_gradients_manifest.csv
explainability/manifests/xai_case_studies_manifest.csv
```

The explainability layer is qualitative and diagnostic. It is used to interpret representative failures, not to replace quantitative robustness metrics.

---

## 4. Forensic operational layer

The third layer evaluates professional forensic tools under the same data conditions.

Target tools:

- X-Ways;
- Magnet AXIOM;
- Cellebrite UFED;
- Oxygen Forensic Detective.

### 4.1 Forensic evaluation bundle

Main script:

```text
datasets/scripts/bundle/12_build_forensic_evaluation_bundle.py
```

Expected outputs:

```text
datasets/forensic_evaluation_bundle/
datasets/forensic_evaluation_bundle/bundle_manifest.csv
datasets/forensic_evaluation_bundle/bundle_hashes_sha256.csv
datasets/forensic_evaluation_bundle/bundle_summary.json
```

The bundle should include:

- clean folds;
- adversarial outputs;
- anti-forensic outputs;
- OOD evaluation samples;
- global manifest;
- SHA256 hashes.

### 4.2 Tool execution

Each tool should be tested on comparable inputs:

- clean images;
- adversarially perturbed images;
- anti-forensically transformed images;
- OOD samples.

Expected tool-specific areas:

```text
forensic_tools/magnet_axiom/
forensic_tools/xways/
forensic_tools/cellebrite_ufed/
forensic_tools/oxygen/
```

### 4.3 Forensic output normalization

Expected output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

Tool exports should be linked back to the dataset primarily through:

1. SHA256;
2. MD5, if available;
3. filename or `image_id` as fallback.

---

## 5. Final comparative analysis

The final analysis compares:

- clean vs perturbed behavior;
- model vs model;
- local AI models vs forensic AI tools;
- adversarial attacks vs anti-forensic transformations;
- nominal binary classification vs OOD behavior.

Expected final outputs:

```text
results/metrics/comparative_metrics.csv
results/tables/
results/plots/
results/figures/
results/reports/
```

---

## Fundamental rule

The workflow is organized around three levels of trust:

1. **Trust in the data**
   - inventory;
   - validation;
   - deduplication;
   - manual selection;
   - frozen manifests;
   - reproducible splits.

2. **Trust in the measurement**
   - stable metrics;
   - consistent fold structure;
   - clean/perturbed comparability;
   - attack manifests;
   - confidence and robustness analysis.

3. **Trust in the forensic interpretation**
   - forensic tool reports;
   - hash-based traceability;
   - normalized outputs;
   - comparative interpretation;
   - discussion of operational and evidentiary implications.
