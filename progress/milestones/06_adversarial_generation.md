# Milestone 06 — Adversarial Attack Generation

## Status

Completed.

## Official script

`datasets/scripts/attacks/14_generate_adversarial_attacks.py`

## Input

`datasets/splits/manifests/clean_folds_manifest.csv`

The input manifest contains the official clean binary subset:

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |
| total | 1000 |

The OOD subset is not attacked in this stage and remains reserved for out-of-distribution evaluation.

## Output directories

`attacks/adversarial/`

Generated adversarial/adversarial-style outputs are organized by attack, target model where applicable, fold, and class.

General model-dependent structure:

```text
attacks/adversarial/<attack_name>/<target_model>/<fold>/<label>/<image_id>__<attack_name>__<target_model>.png
```

General model-agnostic structure:

```text
attacks/adversarial/<attack_name>/model_agnostic/<fold>/<label>/<image_id>__<attack_name>__model_agnostic.<ext>
```

## Generated attacks

| Attack name | Type | Model dependency | Primary target |
|---|---|---|---|
| `fgsm` | gradient-based adversarial attack | model-dependent | `efficientnet_b0` |
| `superdeepfool` | decision-boundary adversarial attack | model-dependent | `efficientnet_b0` |
| `sigma_zero` | adversarial attack | model-dependent | `efficientnet_b0` |
| `one_pixel` | sparse adversarial attack | model-dependent | `efficientnet_b0` |
| `color_shift` | adversarial-style image transformation | model-agnostic | none |

## Fold-aware checkpoint protocol

For every image belonging to fold `F`, model-dependent attacks use the checkpoint:

```text
models/checkpoints/<target_model>/F.pt
```

Example:

```text
image in fold_1 + target efficientnet_b0
→ models/checkpoints/efficientnet_b0/fold_1.pt
```

This ensures that the proxy model used for attack generation was trained on the other four folds and never on the images being attacked.

## Output manifests

Adversarial generation is expected to preserve traceability through attack-specific and/or global manifests under:

```text
attacks/manifests/
```

The manifest layer must allow mapping from every perturbed file back to:

- original image identifier;
- original fold;
- original class label;
- clean image path;
- perturbed image path;
- attack name;
- attack parameters;
- target model where applicable;
- checkpoint path and checkpoint hash where applicable;
- original SHA256;
- perturbed SHA256;
- MD5 for forensic tool compatibility where available.

## Validation summary

| Check | Result |
|---|---:|
| input images | 1000 |
| selected adversarial/adversarial-style attacks | 5 |
| attack names generated | `fgsm`, `superdeepfool`, `sigma_zero`, `one_pixel`, `color_shift` |
| primary model-dependent target | `efficientnet_b0` |
| fold-aware protocol used | true |
| OOD attacked in this stage | false |
| proxy evaluation available after generation | true |

## Methodological notes

The adversarial generation stage is not intended to turn the thesis into a pure Adversarial Machine Learning benchmark. In this work, adversarial attacks are used as controlled experimental stressors for evaluating the operational robustness of AI-based image classification systems in a digital forensic workflow.

Model-dependent attacks are generated against a transparent local proxy model and later evaluated across local proxy models and forensic AI tools. This supports a limited-knowledge transferability analysis, where commercial forensic tools are treated as black-box systems.

Color Shift is retained as an adversarial-style model-agnostic perturbation because it stresses color/channel robustness without requiring access to model gradients or decision boundaries.
