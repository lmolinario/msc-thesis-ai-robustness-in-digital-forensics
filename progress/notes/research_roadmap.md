# Research Roadmap

This document provides a high-level roadmap for the MSc thesis repository.

The repository investigates the operational robustness of AI-based image classification systems in digital forensic scenarios, with a focus on manipulated image inputs and forensic tool behavior.

---

## 1. Research context and objective

The thesis evaluates whether AI-based image classification systems remain reliable when forensic image inputs are manipulated with adversarial or anti-forensic intent.

Main goals:

- evaluate robustness degradation under adversarial perturbations;
- evaluate robustness degradation under realistic anti-forensic image transformations;
- compare local transparent proxy models with forensic AI tools;
- document how failures affect forensic reliability and evidentiary interpretation;
- support the analysis with traceable experimental artifacts and qualitative explainability.

---

## 2. Dataset preparation

The dataset is built from heterogeneous sources:

```text
01_kaggle_weapon
02_deepfirearm
03_google_scraped
04_telegram_youtube
05_deepweb
```

The benchmark is based on a human-in-the-loop selection protocol.

Official frozen dataset:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

Composition:

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| `ood` | 500 |

Official adversarial and anti-forensic subset:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

Composition:

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |

OOD samples are retained as a separate evaluation set and are not attacked.

Split structure:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
datasets/splits/ood/ood_eval_set/
```

---

## 3. Academic AI pipeline

Consolidated proxy models:

- EfficientNet-B0;
- ResNet18;
- CLIP.

BLIP may be discussed as a semantic/caption-based evaluator where relevant, but it is not a primary adversarial generation target. SVM is retained only as historical/baseline context if needed and is not central to the final Chapter 5 reporting.

Main evaluation stages:

1. clean baseline inference;
2. OOD behavior evaluation;
3. anti-forensic inference;
4. adversarial inference;
5. robustness comparison across models.

Core metrics:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- false positive rate;
- false negative rate;
- confusion matrix.

Robustness metrics:

- accuracy drop;
- F1 drop;
- robust accuracy;
- attack success rate;
- confidence shift;
- misclassification rate.

---

## 4. Perturbation generation

### 4.1 Adversarial attacks

Generated attacks:

- FGSM;
- SuperDeepFool;
- Sigma Zero Attack;
- One Pixel Attack;
- Color Shift.

### 4.2 Anti-forensic transformations

Generated transformations:

- JPEG recompression;
- Resample and resize;
- Gaussian blur;
- Histogram modification;
- Contrast stretching.

The perturbation stage preserves traceability between clean and manipulated samples through manifests and hashes.

---

## 5. Explainability

Explainability is used as a qualitative diagnostic layer.

Main method:

- Integrated Gradients.

The Chapter 5 XAI selection is completed and integrated into the thesis text. Selected representative cases:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

XAI is used to interpret transparent proxy-model behavior only. It is not treated as a primary robustness metric and is not used to explain proprietary black-box forensic tools.

---

## 6. Forensic tool pipeline

Final forensic-tool perimeter:

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

Main workflow:

1. build a forensic evaluation bundle;
2. import clean, adversarial, anti-forensic, and OOD samples into eligible tools;
3. export reports, logs, and classification or retrieval outputs;
4. normalize the outputs into a shared schema when possible;
5. compare tool behavior against the local AI models when outputs are comparable.

Expected tool-side metrics:

- detection / retrieval rate;
- false positive rate;
- false negative rate;
- clean vs perturbed detection drop;
- OOD-as-weapon rate;
- report and hash traceability.

---

## 7. Comparative analysis

The final analysis compares:

- local models against each other;
- clean results against perturbed results;
- adversarial attacks against anti-forensic transformations;
- local AI models against eligible forensic AI tools;
- nominal binary behavior against OOD behavior;
- quantitative metrics against qualitative Integrated Gradients case studies.

Expected final outputs:

```text
results/metrics/comparative_metrics.csv
results/tables/
results/plots/
results/figures/
results/reports/
```

---

## 8. Legal and forensic discussion

The discussion should connect technical findings with forensic implications, including:

- risk of false negatives;
- risk of false positives;
- reliability of AI-based forensic classification;
- integrity, authenticity, and repeatability of digital evidence;
- limits of black-box forensic AI tools;
- need for documented validation and human oversight;
- implications of AI regulation and evidentiary standards.

---

## 9. Current operational priorities

Completed:

1. frozen dataset and adversarial subset;
2. clean folds and OOD evaluation set;
3. adversarial and anti-forensic outputs;
4. forensic evaluation bundle;
5. local proxy-model evaluation;
6. Magnet AXIOM / Magnet.AI normalization;
7. Excire Foto 2025 semantic retrieval analysis;
8. Integrated Gradients representative case studies;
9. Chapter 5 XAI integration.

Remaining priorities:

1. consolidate Cellebrite Inseyets 10.9 only if comparable exports become available;
2. finalize Chapter 5 and Chapter 6;
3. perform final thesis-wide revision;
4. prepare the future English version.
