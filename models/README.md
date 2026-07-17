# Proxy Models

This directory contains the transparent proxy-model artifact used by the
FAIR-Lab thesis experiments. The models are research instruments for controlled
robustness evaluation and are not commercial forensic tools.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the complete scope, limitations, and
reported baseline behavior.

## Structure

```text
models/
├── README.md
├── MODEL_CARD.md
├── model_registry.json
├── checkpoints/
│   ├── resnet18/fold_1.pt ... fold_5.pt
│   ├── efficientnet_b0/fold_1.pt ... fold_5.pt
│   └── clip/fold_1.pt ... fold_5.pt
├── reports/
│   └── proxy_model_training_summary.csv
└── scripts/
    └── 12_train_proxy_models.py
```

## Task and protocol

```text
0 = non_weapon
1 = weapon
```

OOD samples are excluded from training. For every target fold, the checkpoint is
trained on the other four folds. The complete frozen suite therefore contains 15
checkpoints: three models across five held-out folds.

| Model | Implementation | Checkpoint contents |
|---|---|---|
| `resnet18` | ImageNet-initialized `torchvision.resnet18` | Complete binary classifier |
| `efficientnet_b0` | ImageNet-initialized `torchvision.efficientnet_b0` | Complete binary classifier |
| `clip` | `open_clip` ViT-B/32 | Trained binary head only; external base weights required |

## Controlled-data prerequisite

Image corpora are not tracked on `main`. Before training, obtain authorized
access to the raw bundle and regenerate the clean split files through steps
00–11. The canonical split manifest remains:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

## Training

Official entry point:

```bash
python models/scripts/12_train_proxy_models.py
```

Frozen CNN training command:

```bash
python models/scripts/12_train_proxy_models.py \
  --model resnet18 efficientnet_b0 \
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

Frozen CLIP-head training command:

```bash
python models/scripts/12_train_proxy_models.py \
  --model clip \
  --fold all \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --validation-ratio 0.15 \
  --seed 42 \
  --device auto \
  --input-size 224 \
  --num-workers 2 \
  --force
```

The interactive launcher remains available when the script is run without
arguments. For CLIP, the visual encoder is always frozen and only the binary
head is trained. The `--freeze-backbone` option affects only ResNet18 and
EfficientNet-B0.

Before training, the script validates image presence, fold/class balance,
identifier uniqueness, and SHA256 correspondence with the split manifest.
Training-report updates use an upsert keyed by `model_name + fold`, preventing a
partial rerun from deleting records for the other checkpoints.

## Registry, reports, and checkpoints

```text
models/model_registry.json
models/reports/proxy_model_training_summary.csv
models/checkpoints/<model_name>/<fold>.pt
```

The registry stores architecture metadata, checkpoint paths, SHA256 identifiers,
and training timestamps. Checkpoints are tracked through Git LFS; no public
image corpus is required to inspect their recorded identities.

Official evaluation entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```
