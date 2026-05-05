# Proxy Models

This directory contains the reproducible configuration for the open proxy models used in the FAIR-Lab adversarial attack pipeline.

The proxy models are not commercial forensic tools. They are transparent, reproducible binary classifiers used to generate model-dependent adversarial perturbations and to study transferability toward other AI systems and forensic tools.

## Official task

All proxy models use the same binary classification task:

```text
0 = non_weapon
1 = weapon
```

OOD samples are not used to train or generate adversarial attacks. They remain reserved for separate robustness and out-of-distribution evaluation.

## Directory layout

```text
models/
├── README.md
├── model_registry.json
├── scripts/
│   ├── 12_train_proxy_models.py
│   └── train_proxy_models.py        # implementation kept for backward compatibility
└── checkpoints/
    ├── .gitkeep
    ├── resnet18/
    │   ├── fold_1.pt
    │   ├── fold_2.pt
    │   ├── fold_3.pt
    │   ├── fold_4.pt
    │   └── fold_5.pt
    ├── efficientnet_b0/
    │   ├── fold_1.pt
    │   ├── fold_2.pt
    │   ├── fold_3.pt
    │   ├── fold_4.pt
    │   └── fold_5.pt
    └── clip/
        ├── fold_1.pt
        ├── fold_2.pt
        ├── fold_3.pt
        ├── fold_4.pt
        └── fold_5.pt
```

## Official numbered entry point

The official pipeline entry point for proxy model training is:

```text
models/scripts/12_train_proxy_models.py
```

The unnumbered implementation file is kept only for backward compatibility:

```text
models/scripts/train_proxy_models.py
```

Use the numbered script in documentation, experiments, and reproducible commands.

## Per-fold training protocol

The official training protocol is fold-aware:

```text
checkpoint for fold_1: train on fold_2 + fold_3 + fold_4 + fold_5
checkpoint for fold_2: train on fold_1 + fold_3 + fold_4 + fold_5
checkpoint for fold_3: train on fold_1 + fold_2 + fold_4 + fold_5
checkpoint for fold_4: train on fold_1 + fold_2 + fold_3 + fold_5
checkpoint for fold_5: train on fold_1 + fold_2 + fold_3 + fold_4
```

This avoids training a proxy model on the same images that are later attacked for that fold.

## Supported proxy models

| Model name | Meaning | Checkpoint path pattern |
|---|---|---|
| `resnet18` | ResNet18 binary classifier | `models/checkpoints/resnet18/<fold>.pt` |
| `efficientnet_b0` | EfficientNet-B0 binary classifier | `models/checkpoints/efficientnet_b0/<fold>.pt` |
| `clip` | Frozen CLIP visual encoder + trained binary head | `models/checkpoints/clip/<fold>.pt` |

## Training commands

Run the interactive launcher directly:

```bash
python models/scripts/12_train_proxy_models.py
```

Smoke test on ResNet18 for `fold_1`:

```bash
python models/scripts/12_train_proxy_models.py \
  --model resnet18 \
  --fold fold_1 \
  --epochs 2 \
  --batch-size 16 \
  --device auto
```

Train all ResNet18 fold checkpoints:

```bash
python models/scripts/12_train_proxy_models.py \
  --model resnet18 \
  --fold all \
  --epochs 10 \
  --batch-size 16 \
  --device auto
```

Train all EfficientNet-B0 fold checkpoints:

```bash
python models/scripts/12_train_proxy_models.py \
  --model efficientnet_b0 \
  --fold all \
  --epochs 10 \
  --batch-size 16 \
  --device auto
```

Train all CLIP binary-head fold checkpoints:

```bash
python models/scripts/12_train_proxy_models.py \
  --model clip \
  --fold all \
  --epochs 10 \
  --batch-size 32 \
  --device auto
```

Train all official proxy models:

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
  --num-workers 2 \
  --force
```

## Adversarial generation with per-fold checkpoints

The official fold-aware adversarial entry point is:

```text
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

The script resolves checkpoints deterministically as:

```text
models/checkpoints/<target_model>/<fold>.pt
```

For example, an image from `fold_1` attacked against `efficientnet_b0` uses:

```text
models/checkpoints/efficientnet_b0/fold_1.pt
```

Official FGSM generation against the primary proxy target:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

Smoke test before full generation:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --limit 10 \
  --force
```

FGSM outputs are saved as lossless PNG files to preserve epsilon-bounded perturbations.

## Git LFS

Checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`) must be tracked with Git LFS or stored externally and downloaded through the model registry.
