# Anti-Forensic Transformations

This directory contains realistic image transformations used to evaluate the robustness of AI-based classifiers and forensic AI tools under anti-forensic conditions.

The official anti-forensic transformation targets are the clean binary folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

Each transformation preserves the original fold structure and class labels.

---

## Implemented Transformations

Official numbered entry point:

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
```


Transformations:

```text
jpeg_recompression/
resample_resize/
gaussian_blur/
histogram_modification/
contrast_stretching/
```

These transformations are controlled image-processing operations, not model-optimized adversarial examples.

---

## Official Command

Run the interactive launcher directly:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
```

Generate all implemented anti-forensic transformations:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py --force
```

Generate one transformation only:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py \
  --attack gaussian_blur \
  --force
```

Smoke test all transformations:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py \
  --limit 10 \
  --force
```

---

## Manifest Requirements

Each generated anti-forensic artifact must be traceable through a manifest containing at least:

```text
generated_image_id
original_image_id
fold
final_label
source_dataset
clean_relative_path
perturbed_relative_path
attack_family
attack_name
attack_parameters
sha256_original
sha256_perturbed
md5_perturbed
size_bytes
created_at
```

---

## Methodological Note

Anti-forensic transformations are not necessarily optimization-based adversarial examples. They are implemented as controlled image-processing operations with explicit parameters, reproducible outputs, and forensic traceability through hashes and manifests.

They are evaluated separately from adversarial FGSM perturbations because their purpose is to simulate realistic image-processing manipulations that may occur in operational or anti-forensic scenarios.
