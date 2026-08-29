# Model Card

## Model Suite

**FAIRLab Proxy Models for Forensic Image Robustness Evaluation**

This card documents the transparent proxy models used in the MSc thesis
repository. They are experimental instruments for controlled robustness
assessment, not commercial forensic tools or operational weapon detectors.

## Task

Binary image classification:

```text
0 = non_weapon
1 = weapon
```

The `ood` branch is excluded from training and from model-dependent attack
generation. It is evaluated separately as an operational stress condition and is
not a supervised third class.

## Models

| Name | Architecture | Training role |
|---|---|---|
| `resnet18` | `torchvision.resnet18`, ImageNet initialization | Supervised CNN baseline |
| `efficientnet_b0` | `torchvision.efficientnet_b0`, ImageNet initialization | Primary supervised CNN and adversarial target |
| `clip` | `open_clip` ViT-B/32 visual encoder plus binary head | Frozen visual encoder with trained binary head |

The CLIP checkpoints contain only the trained binary head. Reconstructing the
complete CLIP proxy also requires the external `open_clip` ViT-B/32 weights
identified as `openai` in the registry.

## Training Data and Fold Protocol

Training uses the clean binary split manifest:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

For each target fold, the corresponding checkpoint is trained on the other four
folds. The official record contains 15 checkpoints:

```text
3 models × 5 held-out folds
```

Each run uses 680 training images, 120 internal validation images, and a
200-image held-out fold. This prevents model-dependent attacks from using a
checkpoint trained on the same fold later attacked or evaluated.

## Frozen Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| CNN batch size | 16 |
| CLIP-head batch size | 32 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Internal validation ratio | 0.15 |
| Seed | 42 |
| Input size | 224 × 224 |

ResNet18 and EfficientNet-B0 are fine-tuned from ImageNet initialization unless
`--freeze-backbone` is explicitly supplied. The CLIP visual encoder is always
frozen; only its binary head is trained.

## Checkpoints and Integrity

Checkpoint pattern:

```text
models/checkpoints/<model_name>/<fold>.pt
```

The checkpoints are distributed through Git LFS. Their SHA-256 identifiers and
training timestamps are recorded in:

```text
models/model_registry.json
models/reports/proxy_model_training_summary.csv
```

The training script validates the official split files and their SHA-256 hashes
before training. Re-running one model/fold updates only that report record and
preserves the remaining frozen records.

## Evaluation

Official evaluation entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Evaluation conditions:

```text
clean
ood
adversarial
anti_forensic
```

### Clean Baseline

| Model | Accuracy | FNR | FPR |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.978 | 0.016 | 0.028 |
| `resnet18` | 0.964 | 0.030 | 0.042 |
| `clip` | 0.927 | 0.012 | 0.134 |

### OOD Behavior

| Model | OOD weapon rate | Mean Max-P | Max-P ≥ 0.9 rate |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.3696 | 0.8505 | 0.5008 |
| `resnet18` | 0.4376 | 0.8896 | 0.6468 |
| `clip` | 0.8356 | 0.5358 | 0.0000 |

`Max-P` denotes maximum predicted-class probability. It is retained only as an
intra-model diagnostic and is not a calibrated probability, forensic certainty,
or a quantity directly comparable across architectures.

The OOD table aggregates 2,500 predictions per model: the same 500 clean OOD
images are evaluated with each of the five fold-specific checkpoints. It does
not represent 2,500 distinct OOD images. Clean binary metrics instead contain
one fold-matched prediction for each of the 1,000 binary samples.

## Adversarial and Anti-Forensic Role

EfficientNet-B0 is the primary target for the four model-dependent attacks:
FGSM, One Pixel, Sigma-Zero, and SuperDeepFool. ResNet18 and CLIP support empirical
cross-architecture transfer analysis. Color Shift is deterministic and
model-agnostic. The five anti-forensic transformations are also model-agnostic.

Transfer results do not establish direct robustness of a non-target model,
because matched attacks were not independently optimized against ResNet18, CLIP,
or the commercial black-box systems.

## Explainability Boundary

Integrated Gradients is applied to the transparent EfficientNet-B0 proxy for
five qualitative diagnostic case studies discussed in Chapter 6. Proxy
attribution maps must not be interpreted as explanations of the commercial
black-box systems:

- Magnet AXIOM / Magnet.AI;
- Excire Foto 2025;
- Cellebrite Inseyets;
- Griffeye / T3K CORE.

## Intended Use

The proxy models support:

- controlled academic robustness experiments;
- fold-aware adversarial generation;
- comparison with black-box forensic AI behavior;
- qualitative explainability analysis;
- traceable methodological evaluation.

They are not intended for autonomous evidentiary decisions, operational law
enforcement deployment, real-time surveillance, biometric analysis, or
replacement of human forensic review.

## Limitations

- The models are trained on a thesis-specific and controlled-access dataset.
- The binary task does not represent all weapon categories or forensic contexts.
- OOD samples are deliberately heterogeneous and are not a supervised third class.
- Max-P values are not forensic certainty and may be poorly calibrated.
- The attack suite is selective rather than exhaustive.
- Results depend on the frozen dataset, folds, checkpoints, software environment,
  and perturbation parameters.

## Citation

Citation metadata for the repository and thesis are maintained in
[`CITATION.cff`](../CITATION.cff).
