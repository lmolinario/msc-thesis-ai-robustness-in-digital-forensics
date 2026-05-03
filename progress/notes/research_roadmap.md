# Research Roadmap

This document provides a high-level roadmap for the MSc thesis repository.

The repository investigates the operational robustness of AI-based image classification systems in digital forensic scenarios, with a focus on manipulated image inputs and forensic tool behavior.

---

## 1. Research context and objective

The thesis evaluates whether AI-based image classification systems remain reliable when forensic image inputs are manipulated with adversarial or anti-forensic intent.

Main goals:

- evaluate robustness degradation under adversarial perturbations;
- evaluate robustness degradation under realistic anti-forensic image transformations;
- compare local AI models with forensic AI tools;
- document how failures affect forensic reliability and evidentiary interpretation;
- support the analysis with explainability and traceable experimental artifacts.

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

The current benchmark is based on a human-in-the-loop selection protocol.

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

Expected split structure:

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

Planned model families:

- ResNet18;
- EfficientNet-B0;
- CLIP;
- BLIP;
- SVM baseline.

Main evaluation stages:

1. clean baseline inference;
2. adversarial inference;
3. anti-forensic inference;
4. OOD behavior evaluation;
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

Planned attacks:

- FGSM;
- SuperDeepFool;
- Sigma Zero Attack;
- One Pixel Attack;
- Color Shift.

### 4.2 Anti-forensic transformations

Planned transformations:

- JPEG recompression;
- Resample and resize;
- Gaussian blur;
- Histogram modification;
- Contrast stretching.

The perturbation stage must preserve traceability between clean and manipulated samples through manifests and hashes.

---

## 5. Explainability

Explainability is used as a qualitative diagnostic layer.

Main method:

- Integrated Gradients, where feasible.

Expected use cases:

- compare clean vs perturbed attribution maps;
- inspect successful attacks;
- inspect false positives and false negatives;
- analyze disagreement between local models and forensic tools;
- support the final discussion with representative visual evidence.

---

## 6. Forensic tool pipeline

Target forensic tools:

- X-Ways;
- Magnet AXIOM;
- Cellebrite UFED;
- Oxygen Forensic Detective.

Main workflow:

1. build a forensic evaluation bundle;
2. import clean, adversarial, anti-forensic, and OOD samples into the tools;
3. export reports, logs, and classification outputs;
4. normalize the outputs into a shared schema;
5. compare tool behavior against the local AI models.

Expected tool-side metrics:

- detection rate;
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
- local AI models against forensic AI tools;
- nominal binary behavior against OOD behavior.

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

Current next stages:

1. keep the frozen dataset and adversarial subset as the official starting point;
2. generate clean folds and the OOD evaluation set;
3. generate adversarial and anti-forensic outputs;
4. build the forensic evaluation bundle;
5. evaluate local AI models;
6. evaluate forensic tools;
7. normalize outputs and aggregate comparative metrics;
8. produce explainability case studies;
9. write the experimental and discussion chapters.
