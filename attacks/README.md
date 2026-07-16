# Attacks

This directory contains the frozen adversarial and anti-forensic artifacts used
to evaluate the operational robustness of image-classification systems in the
thesis.

Perturbations are generated only from the official 1,000-image binary subset:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
```

The 500 OOD samples remain clean and are evaluated separately.

## Structure

```text
attacks/
├── adversarial/
│   ├── color_shift/
│   ├── fgsm/
│   ├── one_pixel/
│   ├── sigma_zero/
│   └── superdeepfool/
├── anti_forensic/
│   ├── jpeg_recompression/
│   ├── resample_resize/
│   ├── gaussian_blur/
│   ├── histogram_modification/
│   └── contrast_stretching/
└── manifests/
```

## Frozen composition

| Family | Variants | Inputs per variant | Generated files |
|---|---:|---:|---:|
| Adversarial / adversarial-style | 5 | 1000 | 5000 |
| Anti-forensic | 5 | 1000 | 5000 |
| **Total** | **10** |  | **10000** |

All variants preserve the five-fold structure and the balanced binary labels:
500 `weapon` and 500 `non_weapon` inputs per variant.

## Methodological distinction

**Adversarial attacks** use transparent proxy models to generate controlled
perturbations. FGSM, One Pixel, Sigma-Zero, and SuperDeepFool are
model-dependent and use fold-aware EfficientNet-B0 checkpoints. Color Shift is
model-agnostic and is retained in the adversarial family as an
adversarial-style robustness stressor.

**Anti-forensic transformations** are model-agnostic image-processing
operations intended to simulate realistic manipulations that may occur before
forensic acquisition or automated triage.

## Official generation scripts

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

The scripts validate clean-input hashes, preserve fold assignments, generate
stable identifiers, and record output hashes and transformation parameters.

## Fold-aware rule

For every sample in fold `F`, a model-dependent attack uses:

```text
models/checkpoints/<target_model>/F.pt
```

The corresponding checkpoint was trained on the other four folds, preventing
the attacked image from appearing in the target model's training partition.

## Canonical records

Sample-level traceability and generation summaries are stored under
[`manifests/`](manifests/). CSV manifests are the source of truth for mappings
between clean and perturbed files; JSON summaries contain aggregate generation
counts and validation checks.

Detailed documentation:

- [`adversarial/README.md`](adversarial/README.md)
- [`anti_forensic/README.md`](anti_forensic/README.md)
- [`manifests/README.md`](manifests/README.md)

## Distribution status

The final decision on retaining or removing generated image files from `main`
is handled together with the dataset-image distribution policy. Regardless of
that decision, scripts, manifests, hashes, summaries, normalized predictions,
and metrics remain the canonical reproducibility record.
