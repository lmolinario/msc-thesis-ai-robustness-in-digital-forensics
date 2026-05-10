# Attacks

This directory contains the structure for adversarial and anti-forensic perturbations used in the thesis pipeline.

The attack stage starts from the official clean binary folds generated from:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

and from the corresponding clean split manifest:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

Out-of-distribution samples are not used as primary attack targets. They are kept separate and evaluated through:

```text
datasets/splits/ood/ood_eval_set/
datasets/splits/manifests/ood_eval_manifest.csv
```

---

## Directory Structure

```text
attacks/
├── README.md
├── manifests/
├── adversarial/
│   ├── README.md
│   ├── fgsm/
│   ├── superdeepfool/
│   ├── sigma_zero/
│   ├── one_pixel/
│   └── color_shift/
└── anti_forensic/
    ├── README.md
    ├── jpeg_recompression/
    ├── resample_resize/
    ├── gaussian_blur/
    ├── histogram_modification/
    └── contrast_stretching/
```

---

## Methodological Role

The attack stage has two complementary goals:

1. generate controlled adversarial perturbations against AI-based image classifiers;
2. generate realistic anti-forensic image transformations that may affect automated forensic triage.

Each attack family must preserve traceability through manifest files, original hashes, generated hashes, fold identifiers, labels, and attack parameters.

---

## Current Operational Status

Official numbered operational scripts:

```text
models/scripts/12_train_proxy_models.py
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```



Implemented anti-forensic transformations:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Implemented adversarial/adversarial-style attacks:

```text
fgsm
color_shift
```

Planned but not currently implemented adversarial attacks:

```text
superdeepfool
sigma_zero
one_pixel
```

---

## Official Generation Order

Recommended order after proxy checkpoints are available:

```bash
python datasets/scripts/attacks/12_generate_anti_forensic_attacks.py --force
```

```bash
python datasets/scripts/attacks/13_generate_adversarial_attacks.py \
  --attack color_shift \
  --force
```

```bash
python datasets/scripts/attacks/13_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

FGSM must be smoke-tested first using `--limit 10` before full generation:

```bash
python datasets/scripts/attacks/13_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --limit 10 \
  --force
```
