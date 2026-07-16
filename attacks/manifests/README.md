# Attack Manifests

This directory contains the frozen sample-level manifests and strict JSON
generation summaries for the adversarial and anti-forensic stages.

## Canonical artifacts

### Anti-forensic

```text
anti_forensic_attacks_manifest.csv
anti_forensic_generation_summary.json
```

### Adversarial and adversarial-style

```text
adversarial_color_shift_manifest.csv
adversarial_color_shift_summary.json

adversarial_fgsm_efficientnet_b0_manifest.csv
adversarial_fgsm_efficientnet_b0_summary.json

adversarial_one_pixel_efficientnet_b0_manifest.csv
adversarial_one_pixel_efficientnet_b0_summary.json

adversarial_sigma_zero_efficientnet_b0_manifest.csv
adversarial_sigma_zero_efficientnet_b0_summary.json

adversarial_superdeepfool_efficientnet_b0_manifest.csv
adversarial_superdeepfool_efficientnet_b0_summary.json
```

These twelve files are the canonical generation records used by downstream
evaluation, forensic-bundle construction, and explainability workflows.

## Source-of-truth rule

CSV manifests define sample-level traceability. JSON summaries record run
parameters, aggregate counts, attack-success summaries where applicable, and
generation validation checks. A JSON summary does not replace the associated
CSV manifest when individual clean-to-perturbed mappings are required.

## Common traceability fields

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
created_at
```

Model-specific fields are empty or marked as not applicable for model-agnostic
transformations.

## Integrity expectations

Frozen summaries verify:

- the expected number of generated files;
- uniqueness of generated identifiers;
- uniqueness of perturbed SHA256 values;
- successful manifest creation;
- balanced per-fold and per-label counts.

JSON files use strict JSON syntax. Non-finite mathematical configuration values
are represented as strings such as `"inf"`, not as the non-standard token
`Infinity`.

## Evaluation outputs

Proxy-model predictions and metrics are canonical under `evaluation/` and
`results/`. Optional evaluation files produced directly by the anti-forensic
generator are treated as local working outputs and are not part of this frozen
manifest directory.
