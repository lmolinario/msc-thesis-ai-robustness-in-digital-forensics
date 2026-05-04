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

Each transformation should preserve the original fold structure and class labels.

---

## Planned Transformations

```text
jpeg_recompression/
resample_resize/
gaussian_blur/
histogram_modification/
contrast_stretching/
```

---

## Manifest Requirements

Each generated anti-forensic artifact should be traceable through a manifest containing at least:

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

Anti-forensic transformations are not necessarily optimization-based adversarial examples. They should be implemented as controlled image-processing operations with explicit parameters, reproducible outputs, and forensic traceability through hashes and manifests.
