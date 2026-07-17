# Model Card

## Model Suite

**FAIR-Lab Proxy Models for Forensic Image Robustness Evaluation**

This card documents the transparent proxy models used in the MSc thesis
repository. They are experimental instruments for controlled robustness
assessment, not commercial forensic tools or operational weapon detectors.

## Task

Binary image classification:

```text
0 = non_weapon
1 = weapon
```

The `ood` category is excluded from training and from model-dependent attack
generation. It is evaluated separately as an operational stress condition.

## Models

| Name | Architecture | Training role |
|---|---|---|
| `resnet18` | `torchvision.resnet18`, ImageNet initialization | Supervised CNN baseline |
| `efficientnet_b0` | `torchvision.efficientnet_b0`, ImageNet initialization | Primary supervised CNN and adversarial target |
| `clip` | `open_clip` ViT-B/32 visual encoder plus binary head | Frozen visual encoder with trained binary head |

The CLIP checkpoints contain only the trained binary head. Reconstructing the
complete CLIP proxy also requires the external `open_clip` ViT-B/32 weights
identified as `openai` in the registry.

## Training data and fold protocol

Training uses the clean binary split manifest:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

For each target fold, the corresponding checkpoint is trained on the other four
folds. The official record contains 15 checkpoints:

```text
3 models x 5 held-out folds
```

Each run uses 680 training images, 120 internal validation images, and a
200-image held-out fold. This prevents model-dependent attacks from using a
checkpoint trained on the same fold later attacked or evaluated.

## Frozen training configuration

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| CNN batch size | 16 |
| CLIP-head batch size | 32 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Internal validation ratio | 0.15 |
| Seed | 42 |
| Input size | 224 x 224 |

ResNet18 and EfficientNet-B0 are fine-tuned from ImageNet initialization unless
`--freeze-backbone` is explicitly supplied. The CLIP visual encoder is always
frozen; only its binary head is trained.

## Checkpoints and integrity

Checkpoint pattern:

```text
models/checkpoints/<model_name>/<fold>.pt
```

The checkpoints are distributed through Git LFS. Their SHA256 identifiers and
training timestamps are recorded in:

```text
models/model_registry.json
models/reports/proxy_model_training_summary.csv
```

The training script validates the official split files and their SHA256 hashes
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

### Clean baseline

| Model | Accuracy | FNR | FPR |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.978 | 0.016 | 0.028 |
| `resnet18` | 0.964 | 0.030 | 0.042 |
| `clip` | 0.927 | 0.012 | 0.134 |

### OOD behavior

| Model | OOD weapon rate | Mean confidence | High-confidence rate |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.3696 | 0.8505 | 0.5008 |
| `resnet18` | 0.4376 | 0.8896 | 0.6468 |
| `clip` | 0.8356 | 0.5358 | 0.0000 |

The OOD table aggregates 2,500 predictions per model: the same 500 clean OOD
images are evaluated with each of the five fold-specific checkpoints. It does
not represent 2,500 distinct OOD images. Clean binary metrics instead contain
one fold-matched prediction for each of the 1,000 binary samples.

## Adversarial and anti-forensic role

EfficientNet-B0 is the primary target for model-dependent adversarial attacks.
ResNet18 and CLIP support cross-model and transferability analysis. Color Shift
and the anti-forensic transformations are model-agnostic.

## Explainability boundary

Integrated Gradients is applied to transparent proxy models, primarily
EfficientNet-B0, for qualitative diagnostic case studies. Proxy attribution maps
must not be interpreted as explanations of the commercial black-box systems:

- Magnet AXIOM / Magnet.AI;
- Excire Foto 2025;
- Cellebrite Inseyets;
- Griffeye / T3K CORE.

## Intended use

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
- Confidence values are not forensic certainty and may be poorly calibrated.
- The attack suite is selective rather than exhaustive.
- Results depend on the frozen dataset, folds, checkpoints, software environment,
  and perturbation parameters.

## Citation

Citation metadata for the repository and thesis are maintained in
[`CITATION.cff`](../CITATION.cff).
