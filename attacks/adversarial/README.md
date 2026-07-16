# Adversarial Attacks

This directory contains the adversarial and adversarial-style perturbations
generated from the frozen 1,000-image binary subset.

## Threat model

The protocol adopts a proxy-based limited-knowledge setting. Transparent local
models are used to generate or evaluate perturbations, while commercial tools
are evaluated separately as operational black boxes.

The primary generation target is:

```text
efficientnet_b0
```

ResNet18 and CLIP are used as transfer/evaluation targets in the proxy-model
evaluation stage.

## Frozen attack set

| Attack | Type | Dependency | Frozen role |
|---|---|---|---|
| `fgsm` | gradient-based, untargeted | model-dependent | white-box baseline |
| `one_pixel` | score-based sparse attack | model-dependent | localized stress test |
| `sigma_zero` | sparse optimization attack | model-dependent | high-impact L0-oriented stressor |
| `superdeepfool` | iterative decision-boundary attack | model-dependent | stronger boundary-oriented stressor |
| `color_shift` | image transformation | model-agnostic | adversarial-style color robustness stressor |

Color Shift is not a gradient-based adversarial example. It remains under
`attacks/adversarial/` because it was frozen as one of the five adversarial-side
variants used by the bundle, evaluation scripts, and thesis reporting.

## Fold-aware checkpoint protocol

For an image belonging to fold `F`, every model-dependent attack uses:

```text
models/checkpoints/<target_model>/F.pt
```

The checkpoint is trained on the other four folds and never on the attacked
sample's fold.

## Frozen parameters

| Attack | Main frozen configuration |
|---|---|
| FGSM | epsilon `8/255`, untargeted, pixel space `[0,1]` |
| One Pixel | 50 iterations, population-size multiplier 20, seed 42 |
| Sigma-Zero | 1000 steps, eta 1.0, sigma 0.001, tau 0.3, tau factor 0.01, infinity gradient norm |
| SuperDeepFool | 20 outer iterations, 50 DeepFool iterations, 1 projection step, `SDF(infinity,1)` |
| Color Shift | R `+12`, G `0`, B `-12`, saturation `1.10`, contrast `1.00`, JPEG quality 95 |

Exact run parameters are preserved in the corresponding JSON summaries and in
each manifest's `attack_parameters` field.

## Output format

Model-dependent outputs are stored as lossless PNG files:

```text
attacks/adversarial/<attack>/<target_model>/<fold>/<label>/<image_id>__<attack>__<target_model>.png
```

Color Shift is stored as JPEG because it is an explicit image-processing
transformation:

```text
attacks/adversarial/color_shift/model_agnostic/<fold>/<label>/<image_id>__color_shift__model_agnostic.jpg
```

## Official entry point

```text
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

General command:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py   --attack <attack_name>   --target-model efficientnet_b0   --checkpoint-root models/checkpoints   --device auto   --force
```

For Color Shift, `--target-model` and `--checkpoint-root` are not operationally
used. A smoke test can be run by adding `--limit 10`.

## Traceability

Each generated artifact records, at minimum:

```text
generated_image_id
original_image_id
fold
final_label
clean_relative_path
perturbed_relative_path
attack_name
attack_parameters
target_model
model_dependency
checkpoint_path
checkpoint_sha256
sha256_original
sha256_perturbed
md5_perturbed
created_at
```

Manifest CSV files and strict JSON summaries are stored in
[`../manifests/`](../manifests/).
