# Milestone 06 — Proxy Model Training

## Status

Completed.

## Purpose

This milestone documents the training of transparent proxy models used as the reproducible experimental baseline for the thesis.

Proxy models are not commercial forensic tools. They are local, inspectable models used to:

- establish clean baseline performance;
- evaluate robustness under adversarial and anti-forensic perturbations;
- compare architectures under a common protocol;
- provide a transparent reference point for later forensic-tool evaluation;
- support explainability case studies.

---

## Input

Clean binary folds:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

The training and evaluation setup uses the official binary subset:

| Class | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

OOD samples are not used as training targets. They remain reserved for separate reliability evaluation.

---

## Official script

```text
models/scripts/12_train_proxy_models.py
```

---

## Proxy models

The proxy model set is:

```text
efficientnet_b0
resnet18
clip
```

Methodological role:

| Model | Role |
|---|---|
| `efficientnet_b0` | Primary CNN proxy and adversarial source model |
| `resnet18` | Secondary CNN baseline for architectural comparison |
| `clip` | Vision-language proxy model for transfer and semantic robustness comparison |

---

## Output areas

Expected model artifacts:

```text
models/checkpoints/
models/reports/
models/model_registry.json
```

The registry records the proxy model configuration and label mapping used by later evaluation scripts.

---

## Methodological notes

- Proxy models are evaluated fold-aware where applicable.
- The positive class for binary reporting is `weapon`.
- The negative class is `non_weapon`.
- OOD images are not part of binary training and are evaluated separately.
- EfficientNet-B0 is the primary adversarial source model for model-dependent attacks.

---

## Completion criteria

This milestone is complete when:

- `efficientnet_b0`, `resnet18`, and `clip` are available as proxy models;
- the downstream evaluation script can load the trained/registered models;
- the model registry and checkpoint/report areas are available for reproducibility;
- the models can be evaluated on clean, adversarial, anti-forensic, and OOD inputs.

Status: **completed**.
