# Adversarial Attacks

This directory contains adversarial perturbations generated against AI-based image classifiers.

The official adversarial attack targets are the clean binary folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

Each attack should preserve the original fold structure and class labels.

---

## Planned Attacks

```text
fgsm/
superdeepfool/
sigma_zero/
one_pixel/
color_shift/
```

---

## Manifest Requirements

Each generated adversarial artifact should be traceable through a manifest containing at least:

```text
image_id
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

The preferred methodological reference is a Cagliari-aligned adversarial machine learning workflow, using SecML/SecML-Torch where practical and academically justified. Where specific attacks are not directly available or are easier to reproduce through widely used frameworks, Foolbox or ART may be used as implementation backends, provided that parameters and outputs are normalized into the thesis manifest format.
