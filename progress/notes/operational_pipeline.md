# Operational Pipeline

This document summarizes the operational workflow used by the repository for the MSc thesis on AI robustness in digital forensics.

It is intended as a working methodological note. The canonical repository entry point remains `README.md`; this file expands the execution logic behind the pipeline and reflects the current repository state.

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

### 3.1 Proxy model training

Main script:

```text
models/scripts/12_train_proxy_models.py
```

Proxy models currently used in the local evaluation layer:

- EfficientNet-B0;
- ResNet18;
- CLIP.

BLIP may be used as a semantic/caption-based evaluator, but it is not a primary adversarial generation target.

Expected checkpoint area:

```text
models/checkpoints/
```

Fold-aware checkpoints are used for binary evaluation and model-dependent attack generation.

### 3.2 Clean and OOD evaluation

Main evaluation script:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Current output areas:

```text
evaluation/proxy_models/
results/metrics/
```

Core metrics:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- false positive rate;
- false negative rate;
- confidence statistics where available.

OOD samples are evaluated separately and are not included in binary accuracy.

### 3.3 Perturbation generation

Adversarial attacks generated:

- FGSM;
- SuperDeepFool;
- Sigma Zero;
- One Pixel Attack;
- Color Shift.

Anti-forensic transformations generated:

- JPEG recompression;
- Resample and resize;
- Gaussian blur;
- Histogram modification;
- Contrast stretching.

Official scripts:

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

Output structure:

```text
attacks/adversarial/
attacks/anti_forensic/
attacks/manifests/
```

Each attack output is traceable to the clean source image through `image_id`, fold, label, relative path, and hash.

### 3.4 Evaluation under perturbation

Main evaluation script:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Current output files include:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

Robustness metrics:

- clean accuracy vs perturbed accuracy;
- F1 drop;
- robust accuracy;
- attack success rate;
- misclassification rate;
- confidence shift;
- comparative clean-vs-perturbed behavior by attack.

Comparative metrics match perturbed predictions to clean predictions through model, fold, and original image identifier.

### 3.5 Explainability

Method:

- Integrated Gradients.

Status:

```text
completed and integrated in Chapter 5
```

Selected representative cases:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

Output areas:

```text
explainability/outputs/integrated_gradients/
explainability/manifests/
explainability/logs/
docs/LatexThesis_ITA/images/fig_xai_case*_*.png
```

The explainability layer is qualitative and diagnostic. It is used to interpret representative proxy-model failures and successes, not to replace quantitative robustness metrics or explain proprietary black-box forensic tools.

---

## 4. Forensic operational layer

The third layer evaluates professional forensic tools under the same data conditions.

Final tool perimeter:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673

Completed / analyzed:
- X-Ways Forensics / Excire Foto 2025, version 4.1.5

Pending / to be consolidated:
- Cellebrite Inseyets, version 10.9

Excluded:
- Oxygen Forensic Detective
- Autopsy
```

### 4.1 Forensic evaluation bundle

Main script:

```text
datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py
```

Expected output area:

```text
datasets/forensic_evaluation_bundle/
```

Current logical bundle areas:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

The bundle includes:

- clean folds;
- adversarial outputs;
- anti-forensic outputs;
- OOD evaluation samples;
- global metadata;
- SHA256/MD5 hashes where available;
- blind filenames for tool-facing input;
- internal mapping from blind files to original image identifiers, labels, attacks, and sample types.

### 4.2 Bundle validation

Before running forensic tools, the bundle must be checked for:

- expected number of files by sample type;
- expected number of files by attack;
- expected number of files by fold;
- presence of clean/adversarial/anti-forensic/OOD samples;
- absence of label leakage in blind filenames;
- valid SHA256/MD5 mappings;
- stable relation between blind filename, original image identifier, and generated artifact.

### 4.3 Tool execution

Eligible tools should be tested on comparable inputs:

- clean images;
- adversarially perturbed images;
- anti-forensically transformed images;
- OOD samples.

Current tool-specific areas:

```text
forensic_tools/magnet_axiom/
forensic_tools/excire_foto_2025/
forensic_tools/cellebrite_inseyets/
```

Large proprietary case files, installers, licensed databases, and heavy exports should not be committed unless strictly necessary and legally/ethically appropriate.

### 4.4 Forensic output normalization

Expected output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

Tool exports should be linked back to the dataset primarily through:

1. SHA256;
2. MD5, if available;
3. filename or `image_id` as fallback.

The normalized schema should make commercial-tool outputs comparable with local proxy-model predictions when such comparison is methodologically justified.

---

## 5. Final comparative analysis

The final analysis compares:

- clean vs perturbed behavior;
- model vs model;
- local proxy models vs eligible forensic AI tools;
- adversarial attacks vs anti-forensic transformations;
- nominal binary classification vs OOD behavior;
- quantitative robustness metrics vs qualitative explainability findings.

Expected final outputs:

```text
results/metrics/comparative_metrics.csv
results/tables/
results/plots/
results/figures/
results/reports/
```

---

## 6. Current operational state

Completed:

```text
raw/prepared/final dataset pipeline
clean and OOD split generation
proxy model training
anti-forensic transformation generation
adversarial attack generation
proxy model evaluation
forensic evaluation bundle construction and validation
Magnet AXIOM / Magnet.AI normalization
Excire Foto 2025 semantic retrieval analysis
Integrated Gradients Chapter 5 case-study integration
```

Current focus:

```text
consolidate Cellebrite Inseyets if feasible
finalize Chapter 5 and Chapter 6
perform final thesis-wide revision
prepare future English version
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
   - proxy model predictions;
   - confidence and robustness analysis;
   - qualitative XAI case-study traceability.

3. **Trust in the forensic interpretation**
   - forensic tool reports;
   - hash-based traceability;
   - normalized outputs;
   - comparative interpretation;
   - discussion of operational and evidentiary implications.
