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

## Current Status

This directory currently defines the planned attack structure. Attack generation scripts and generated outputs will be added in the next experimental phase.
