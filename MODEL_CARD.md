# Model Card

## Model Suite Name

**FAIR-Lab Proxy Models for Forensic Image Robustness Evaluation**

This model card describes the transparent proxy models used in the MSc thesis repository:

```text
msc-thesis-ai-robustness-in-digital-forensics
```

The proxy models are used to evaluate the operational robustness of AI-based image classification systems in a Digital/Computer Forensics scenario. They are not commercial forensic tools and are not intended for operational deployment.

---

## Model Purpose

The proxy models serve three methodological purposes:

1. provide transparent and reproducible baselines for binary image classification;
2. generate model-dependent adversarial perturbations in a controlled fold-aware setting;
3. support diagnostic explainability analysis through white-box access.

The models are used as experimental instruments for robustness evaluation. Their outputs are not intended to constitute evidence, automated decisions, or standalone forensic conclusions.

---

## Task Definition

The official proxy-model task is binary image classification:

```text
0 = non_weapon
1 = weapon
```

The label mapping is stored in:

```text
models/model_registry.json
```

OOD images are not used for proxy-model training and are not used as adversarial attack targets. They are evaluated separately as an operational robustness and false-positive risk condition.

---

## Supported Models

The current proxy model suite includes:

| Model name | Architecture | Role |
|---|---|---|
| `resnet18` | `torchvision.resnet18` | Supervised CNN binary classifier. |
| `efficientnet_b0` | `torchvision.efficientnet_b0` | Supervised CNN binary classifier and primary adversarial target. |
| `clip` | `open_clip.ViT-B-32` | Frozen CLIP visual encoder with trained binary head. |

The models are intentionally heterogeneous to compare different visual representation strategies while keeping the task and split protocol fixed.

---

## Model Registry

The model registry is located at:

```text
models/model_registry.json
```

It records:

- task name;
- label mapping;
- fold protocol;
- input manifest;
- checkpoint root;
- model architectures;
- pretrained weights;
- classifier heads;
- input size;
- normalization type;
- checkpoint path pattern;
- SHA256 hashes of trained checkpoints;
- training timestamps.

Checkpoint path pattern:

```text
models/checkpoints/<model_name>/<fold>.pt
```

---

## Training Data

The proxy models are trained on the official clean binary folds generated from:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

The split manifest is:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

Training samples include only:

```text
weapon
non_weapon
```

OOD samples are excluded from training.

---

## Fold-Aware Training Protocol

Training is fold-aware. For each target fold, the model is trained on the other four folds and evaluated or attacked on the held-out fold.

Protocol:

```text
checkpoint for fold_1: train on fold_2 + fold_3 + fold_4 + fold_5
checkpoint for fold_2: train on fold_1 + fold_3 + fold_4 + fold_5
checkpoint for fold_3: train on fold_1 + fold_2 + fold_4 + fold_5
checkpoint for fold_4: train on fold_1 + fold_2 + fold_3 + fold_5
checkpoint for fold_5: train on fold_1 + fold_2 + fold_3 + fold_4
```

This protocol prevents a model-dependent adversarial attack from using a checkpoint trained on the same fold that is later attacked.

Official training entry point:

```text
models/scripts/12_train_proxy_models.py
```

---

## Training Configuration

The default official training command is documented in:

```text
models/README.md
```

Typical configuration:

```text
epochs            = 10
batch_size        = 16 for CNN models, 32 for CLIP head training
learning_rate     = 0.0001
weight_decay      = 0.0001
validation_ratio  = 0.15
seed              = 42
input_size        = 224
```

The actual training parameters should be verified against the executed command, training reports, and model registry before final reporting.

---

## Pretraining and Fine-Tuning

### ResNet18

- backbone: `torchvision.resnet18`;
- pretrained weights: ImageNet;
- classifier replaced with a binary head;
- trained on the thesis clean binary folds.

### EfficientNet-B0

- backbone: `torchvision.efficientnet_b0`;
- pretrained weights: ImageNet;
- classifier replaced with a binary head;
- trained on the thesis clean binary folds;
- used as the primary target model for model-dependent adversarial attacks.

### CLIP

- visual encoder: `open_clip.ViT-B-32`;
- pretrained weights: OpenAI CLIP weights through the selected implementation;
- visual encoder treated as frozen feature extractor;
- binary classification head trained on the thesis clean binary folds.

---

## Evaluation Conditions

The proxy models are evaluated on:

```text
clean
ood
adversarial
anti_forensic
```

Official evaluation entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Main outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_adversarial_metrics.csv
results/metrics/proxy_model_anti_forensic_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/final_core_metrics.csv
results/metrics/final_robustness_metrics.csv
results/metrics/final_confusion_matrices.csv
results/metrics/final_ood_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

---

## Reported Clean Baseline Performance

The clean baseline reported in the thesis pipeline is:

| Model | Accuracy | FNR | FPR |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.978 | 0.016 | 0.028 |
| `resnet18` | 0.964 | 0.030 | 0.042 |
| `clip` | 0.927 | 0.012 | 0.134 |

These values should be interpreted as clean-condition baselines, not as complete robustness indicators.

---

## OOD Behavior

OOD evaluation is treated as an operational robustness condition rather than a standard closed-set classification task.

Reported OOD behavior:

| Model | OOD weapon rate | Mean confidence | High-confidence rate |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.3696 | 0.8505 | 0.5008 |
| `resnet18` | 0.4376 | 0.8896 | 0.6468 |
| `clip` | 0.8356 | 0.5358 | 0.0000 |

The OOD results highlight that clean accuracy alone is insufficient to characterize operational forensic reliability.

---

## Adversarial and Anti-Forensic Robustness

The models are evaluated under the following adversarial perturbations:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

and the following anti-forensic transformations:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Model-dependent adversarial attacks are generated primarily against `efficientnet_b0` using fold-aware checkpoints. Anti-forensic transformations and `color_shift` are model-agnostic.

The adversarial and anti-forensic evaluations are intended as operational stress tests, not as exhaustive optimization studies of the adversarial machine learning attack space.

---

## Explainability

Explainability is performed on transparent proxy models, primarily EfficientNet-B0, using Integrated Gradients.

Official XAI entry points:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
```

The XAI layer is used for qualitative diagnostic analysis and representative case studies. It is not a primary robustness metric.

Integrated Gradients outputs should not be interpreted as explanations of commercial black-box forensic tools such as Magnet AXIOM/Magnet.AI, X-Ways/Excire, Cellebrite UFED, or Oxygen Forensic Detective.

---

## Intended Uses

The proxy models are intended for:

- academic robustness evaluation;
- controlled binary image-classification experiments;
- adversarial attack generation in a fold-aware protocol;
- comparison with black-box commercial forensic AI behavior;
- qualitative explainability case studies;
- forensic AI methodology development.

---

## Out-of-Scope Uses

The proxy models are not intended for:

- operational law-enforcement deployment;
- autonomous forensic decision-making;
- evidentiary classification without human review;
- real-time surveillance;
- safety-critical weapon detection;
- biometric analysis;
- person identification;
- replacement of commercial validated forensic tools.

The models are research artifacts and must be interpreted within the experimental protocol of the thesis.

---

## Human-in-the-Loop Requirement

The model outputs must be interpreted within a human-in-the-loop forensic workflow. Predictions, confidence scores, saliency maps, and robustness metrics are decision-support information, not autonomous evidentiary conclusions.

In the thesis methodology, the analyst remains responsible for:

- reviewing flagged and non-flagged items;
- interpreting false positives and false negatives;
- assessing OOD or ambiguous samples;
- validating tool outputs against case context;
- documenting limitations and uncertainty.

---

## Ethical and Legal Considerations

The models operate on weapon-related and forensic-relevant imagery. Their use requires careful handling of dataset provenance, redistribution rights, institutional constraints, and operational interpretation.

The model checkpoints, generated adversarial samples, forensic bundle outputs, and commercial-tool exports may be subject to separate distribution restrictions. They should not be redistributed unless their status has been verified.

---

## Known Limitations

- The models are trained on a limited thesis-specific dataset.
- The binary task does not represent all possible weapon categories or real-world forensic contexts.
- OOD samples are intentionally heterogeneous and should not be interpreted as a conventional third class.
- Clean accuracy does not imply robustness under adversarial or anti-forensic manipulation.
- Confidence scores may be poorly calibrated and must not be interpreted as forensic certainty.
- Adversarial robustness is evaluated under a selected set of attacks, not an exhaustive adversarial benchmark.
- Commercial forensic tools are black boxes and cannot be directly explained using proxy-model attribution maps.
- Results depend on the specific dataset, split protocol, attack parameters, model checkpoints, and software environment.

---

## Reproducibility Artifacts

Relevant files:

```text
models/model_registry.json
models/scripts/12_train_proxy_models.py
evaluation/scripts/15_evaluate_proxy_models.py
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
results/metrics/proxy_model_evaluation_summary.json
```

Relevant checkpoint root:

```text
models/checkpoints/
```

Checkpoint files should be handled through Git LFS or external storage when needed.

---

## Citation

Citation details will be added upon thesis completion.

Until then, cite the repository and the corresponding MSc thesis when referring to the proxy models, methodology, or results.
