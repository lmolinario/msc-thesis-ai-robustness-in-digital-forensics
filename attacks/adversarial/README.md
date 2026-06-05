# Adversarial Attacks

This directory contains adversarial and adversarial-style perturbations generated for the thesis pipeline.

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

## Generated Attacks

Generated adversarial/adversarial-style attacks:

```text
fgsm/
superdeepfool/
sigma_zero/
one_pixel/
color_shift/
```

Operational status:

```text
adversarial generation completed
proxy model evaluation completed
commercial-tool comparison completed for Magnet AXIOM / Magnet.AI and Excire Foto 2025
Chapter 5 XAI case selection completed and integrated
```

---

## Target-Model and Transferability Protocol

The adversarial protocol follows a proxy-based, limited-knowledge threat model.

Primary white-box proxy generation target:

```text
efficientnet_b0
```

EfficientNet-B0 is the primary target for model-dependent adversarial generation because it is a realistic CNN-based surrogate for image classification systems and provides a sustainable computational compromise for the thesis.

Transfer/evaluation targets:

```text
resnet18
clip
```

Semantic evaluator, if used in the thesis discussion:

```text
BLIP
```

BLIP is used only as a semantic/caption-based evaluator. It is not used as a primary adversarial target.

Black-box / operational forensic evaluation perimeter:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673

Completed / analyzed:
- X-Ways Forensics / Excire Foto 2025, version 4.1.5

Pending / to be consolidated:
- Cellebrite Inseyets, version 10.9

Excluded from the final experimental perimeter:
- Oxygen Forensic Detective
- Autopsy
```

Commercial forensic tools are treated as operational black boxes. The goal is not to know or reproduce their internal models, but to evaluate whether perturbations generated on transparent local proxy models correspond to observable robustness risks in AI-assisted forensic triage systems.

---

## Attack Set

| Attack | Type | Model dependency | Main role |
|---|---|---|---|
| `fgsm` | gradient-based adversarial attack | model-dependent | baseline white-box evasion attack |
| `superdeepfool` | decision-boundary adversarial attack | model-dependent | stronger/iterative adversarial perturbation |
| `sigma_zero` | adversarial attack | model-dependent | high-impact adversarial perturbation |
| `one_pixel` | sparse adversarial attack | model-dependent | localized perturbation stress test |
| `color_shift` | adversarial-style image transformation | model-agnostic | color/channel robustness stress test |

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

---

## Official Commands

Model-agnostic Color Shift:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack color_shift \
  --force
```

FGSM:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

SuperDeepFool:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack superdeepfool \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

Sigma Zero:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack sigma_zero \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

One Pixel:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack one_pixel \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

Smoke tests can be performed before full regeneration by adding:

```text
--limit 10
```

---

## Output Format

Model-dependent adversarial outputs are saved in lossless PNG format:

```text
attacks/adversarial/<attack_name>/<target_model>/<fold>/<label>/<image_id>__<attack_name>__<target_model>.png
```

PNG is required for gradient-based and model-dependent attacks because JPEG compression may introduce artifacts comparable to or stronger than the intended perturbation.

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

## Evaluation Outputs

The proxy model evaluation stage is performed by:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Main outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

The comparative metrics match perturbed predictions to clean predictions through model, fold, and original image identifier.

---

## XAI Status

Integrated Gradients case studies for Chapter 5 have been completed and integrated into the thesis text. The selected cases include clean, OOD, anti-forensic, and adversarial scenarios, with `sigma_zero` represented as a high-confidence adversarial failure.

---

## Methodological Note

The preferred methodological reference is a Cagliari-aligned adversarial machine learning workflow, using SecML/SecML-Torch where practical and academically justified. Where specific attacks are not directly available or are easier to reproduce through widely used frameworks, Foolbox or ART may be used as implementation backends, provided that parameters and outputs are normalized into the thesis manifest format.

For this thesis, adversarial machine learning is not the final object of study. It is used as an experimental stressor to evaluate the operational robustness of AI-based image classification systems in a digital forensic workflow.
