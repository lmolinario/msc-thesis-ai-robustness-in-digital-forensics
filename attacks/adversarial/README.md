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

Each attack preserves the original fold structure and class labels.

---

## Implemented and Planned Attacks

Implemented:

```text
fgsm/
color_shift/
```

Planned but not currently implemented:

```text
superdeepfool/
sigma_zero/
one_pixel/
```

The official generation script raises `NotImplementedError` if a planned but unsupported attack is requested.

---

## Target-Model and Transferability Protocol

The adversarial protocol follows a proxy-based, limited-knowledge threat model.

White-box proxy generation target:

```text
efficientnet_b0
```

EfficientNet-B0 is the primary target for adversarial generation because it is a realistic CNN-based surrogate for image classification systems and provides a sustainable computational compromise for the thesis.

Transfer targets for cross-model evaluation:

```text
resnet18
clip
```

Semantic evaluation model:

```text
BLIP
```

BLIP is used only as a semantic/caption-based evaluator. It is not used as a primary adversarial target.

Black-box forensic evaluation targets:

```text
Magnet AI
X-Ways / Excire
Cellebrite
Oxygen
```

Commercial forensic tools are treated as operational black boxes. The goal is not to know or reproduce their internal models, but to evaluate whether perturbations generated on transparent local proxy models transfer to AI-assisted forensic triage systems.

---

## Fold-Aware Checkpoint Protocol

For every image belonging to fold `F`, model-dependent attacks must use the checkpoint:

```text
models/checkpoints/<target_model>/F.pt
```

Example:

```text
image in fold_1 + target efficientnet_b0
→ models/checkpoints/efficientnet_b0/fold_1.pt
```

This ensures that the proxy model used for attack generation was trained on the other four folds and never on the images being attacked.

Official numbered entry point:

```text
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```


Use the numbered script in documentation, experiments, and reproducible commands.

Official FGSM command:

```bash
python datasets/scripts/attacks/13_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

Smoke test:

```bash
python datasets/scripts/attacks/13_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --limit 10 \
  --force
```

---

## Output Format

FGSM outputs are saved in lossless PNG format:

```text
attacks/adversarial/fgsm/<target_model>/<fold>/<label>/<image_id>__fgsm__<target_model>.png
```

PNG is required for gradient-based attacks because JPEG compression may introduce artifacts comparable to the epsilon-bounded perturbation.

Color Shift outputs remain JPEG because Color Shift is a model-agnostic image-processing perturbation:

```text
attacks/adversarial/color_shift/model_agnostic/<fold>/<label>/<image_id>__color_shift__model_agnostic.jpg
```

---

## Manifest Requirements

Each generated adversarial artifact must be traceable through a manifest containing at least:

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
target_model
model_dependency
checkpoint_path
checkpoint_sha256
sha256_original
sha256_perturbed
md5_perturbed
size_bytes
extension
created_at
```

---

## Methodological Note

The preferred methodological reference is a Cagliari-aligned adversarial machine learning workflow, using SecML/SecML-Torch where practical and academically justified. Where specific attacks are not directly available or are easier to reproduce through widely used frameworks, Foolbox or ART may be used as implementation backends, provided that parameters and outputs are normalized into the thesis manifest format.
