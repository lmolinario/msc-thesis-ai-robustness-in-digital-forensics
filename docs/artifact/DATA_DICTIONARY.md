# Data Dictionary

This document summarizes the principal CSV and JSON artifacts used by the thesis repository.

## General Identifiers

| Field | Meaning |
|---|---|
| `image_id` | Stable identifier in the curated source dataset |
| `original_image_id` | Identifier of the clean source image from which another artifact derives |
| `generated_image_id` | Identifier assigned to a perturbed image |
| `bundle_id` | Anonymous identifier in the 11,500-item forensic evaluation bundle |
| `fold` | Fold used for fold-aware proxy training or evaluation |
| `final_label` | Frozen semantic label: `weapon`, `non_weapon`, or `ood` |
| `sample_type` | `clean`, `ood`, `adversarial`, or `anti_forensic` |
| `attack_family` | `none`, `adversarial`, or `anti_forensic` |
| `attack_name` | Specific perturbation, transformation, or `clean` condition |
| `tool_name` | Normalized commercial configuration identifier |
| `evaluated_model` | Transparent proxy architecture being evaluated |

Commercial-tool fields are observable operational signals. They must not be interpreted as access to proprietary internal probabilities or decision logic.

## Frozen Dataset Manifest

### `datasets/final/manifests/manual_selection_final_1500.csv`

Official 1,500-image dataset manifest.

Expected class profile:

```text
weapon      500
non_weapon  500
ood         500
```

Important fields may include:

| Field | Meaning |
|---|---|
| `image_id` | Stable curated identifier |
| `relative_path` | Prepared-image path used in the local pipeline |
| `source_dataset` | Source collection identifier |
| `sha256` | Primary integrity digest |
| `final_label` | Frozen semantic label |
| `review_state` | Human-review status |
| `reviewer_id` | Reviewer identifier |
| `review_timestamp` | Review timestamp |
| `review_notes` | Optional selection note |

### `datasets/final/manifests/manual_selection_adversarial_subset.csv`

Official 1,000-image balanced binary subset:

```text
weapon      500
non_weapon  500
```

It is the source for clean folds, adversarial generation, anti-forensic generation, and binary robustness evaluation.

## Split Manifests

### `datasets/splits/manifests/clean_folds_manifest.csv`

Fold-aware binary manifest. Typical fields:

```text
image_id
fold
final_label
relative_path
sha256
source_dataset
```

Each of the five test folds contains 200 items: 100 weapon and 100 non-weapon.

### `datasets/splits/manifests/ood_eval_manifest.csv`

Clean OOD evaluation manifest. OOD images are not used as a supervised third class and are not perturbed by the attack-generation stages.

## Attack Manifests

Files under `attacks/manifests/` provide traceability for generated adversarial and anti-forensic items.

Common fields:

| Field | Meaning |
|---|---|
| `generated_image_id` | Perturbed artifact identifier |
| `original_image_id` | Source clean image identifier |
| `fold` | Inherited fold |
| `final_label` | Original binary ground truth |
| `attack_family` | `adversarial` or `anti_forensic` |
| `attack_name` | Specific condition |
| `attack_parameters` | Serialized generation parameters |
| `target_model` | Target architecture where applicable |
| `model_dependency` | Model-dependent or model-agnostic status |
| `checkpoint_sha256` | Checkpoint integrity digest where applicable |
| `sha256_original` | Source-image digest |
| `sha256_perturbed` | Generated-image digest |
| `md5_perturbed` | Compatibility digest used for some commercial matching workflows |
| `perturbed_relative_path` | Local generated-image path |

## Forensic Evaluation Bundle

### `datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv`

Official metadata source for the 11,500-item blind bundle.

| Field | Meaning |
|---|---|
| `bundle_id` | Anonymous stable bundle identifier |
| `original_image_id` | Source clean image, where applicable |
| `generated_image_id` | Perturbed artifact identifier, where applicable |
| `sample_type` | Clean, OOD, adversarial, or anti-forensic |
| `attack_family` | Condition family |
| `attack_name` | Specific condition |
| `attack_target_model` | Generation target for model-dependent attacks |
| `fold` | Fold inherited from source data |
| `final_label` | Hidden evaluation ground truth |
| `blind_relative_path` | Semantically neutral import path |
| `sha256_actual` | Bundle-file SHA256 |
| `md5_actual` | Bundle-file MD5 |

The metadata and structured audit views must not be imported into commercial tools.

## Proxy Prediction Table

### `evaluation/proxy_models/proxy_model_predictions.csv`

Prediction-level output for EfficientNet-B0, ResNet18, and CLIP.

Typical fields:

| Field | Meaning |
|---|---|
| `evaluated_model` | Proxy architecture |
| `fold` | Fold-specific checkpoint |
| `bundle_id` | Bundle item |
| `sample_type` | Experimental condition |
| `attack_family` | Condition family |
| `attack_name` | Specific condition |
| `attack_target_model` | Attack generation target where applicable |
| `final_label` | Ground truth where defined |
| `predicted_label` | Binary proxy prediction |
| `confidence` | Model-specific prediction confidence |
| `correct` | Ground-truth agreement where meaningful |

Confidence is an intra-model diagnostic and is not assumed to be calibrated across architectures.

## Canonical Commercial-Tool Prediction Table

### `evaluation/forensic_tools/normalized_predictions.csv`

This is the public repository-wide source for commercial prediction-level analysis.

Frozen profile:

```text
magnet_axiom             11,500
excire_foto_2025_d20     11,500
excire_foto_2025_d50     11,500
excire_foto_2025_d80     11,500
cellebrite_inseyets      11,500
griffeye                 11,500
TOTAL                     69,000
```

Common fields:

| Field | Meaning |
|---|---|
| `tool_name` | Normalized tool/configuration identifier |
| `bundle_id` | Anonymous bundle identifier |
| `matched` | Always `true` in the canonical validated table |
| `match_method` | `validated_public_extract` |
| `sample_type` | Clean, OOD, adversarial, or anti-forensic |
| `attack_family` | Condition family |
| `attack_name` | Specific condition |
| `final_label` | Hidden bundle ground truth joined after processing |
| `weapon_detected` | Normalized boolean operational signal |
| `normalized_prediction` | `weapon` or `non_weapon` |

Tool-specific minimum observable fields:

| Field group | Tool/configuration |
|---|---|
| `tags` | Magnet AXIOM |
| `classifications` | Cellebrite Inseyets |
| `excire_distance_limit` | Excire D20/D50/D80 |
| `n_prompt_hits`, `hit_prompts` | Excire semantic retrieval summary |
| `prompt_*_hit` fields | Fixed Excire firearm-oriented prompts |
| `firearm_bookmark` | Griffeye/T3K CORE primary mapping |
| `secondary_weapon_bookmarks` | Additional retained Griffeye weapon bookmarks |

The canonical table deliberately excludes:

```text
raw_export_file
raw_row_number
raw_filename_or_path
tool_input_filename
sha256
md5
metadata_json
PhotoDNA
serial numbers
local absolute paths
```

Its schema and SHA256 are recorded in:

```text
evaluation/forensic_tools/normalized_predictions.schema.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
```

Exact equivalence is recorded in:

```text
forensic_tools/public_extracts_validation.json
```

## Commercial Metric Table

### `results/metrics/forensic_tools_metrics.csv`

Consolidated 186-row commercial metric table.

Key dimensions:

```text
tool_name
scope
sample_type
attack_family
attack_name
```

Key metrics:

| Metric | Meaning |
|---|---|
| `rows_total` | Rows in the metric group |
| `matched_rows` | Successfully normalized rows |
| `binary_rows` | Weapon/non-weapon rows |
| `unknown_rows` | Rows without an interpretable binary signal |
| `tp`, `fp`, `tn`, `fn` | Confusion counts |
| `accuracy` | Binary accuracy |
| `balanced_accuracy` | Mean recall across binary classes |
| `precision_weapon` | Weapon precision |
| `recall_weapon` | Weapon recall |
| `false_negative_rate` | Missed weapon rate |
| `false_positive_rate` | Non-weapon false alarm rate |
| `ood_weapon_flag_rate` | Share of OOD items mapped to weapon |

## Proxy Core Metrics

### `results/metrics/final_core_metrics.csv`

Contains clean and condition-level proxy metrics. Important fields include:

```text
evaluated_model
sample_type
attack_family
attack_name
attack_target_model
total
accuracy
balanced_accuracy
precision_weapon
recall_weapon
f1_weapon
macro_f1
false_positive_rate
false_negative_rate
misclassification_rate
confidence_mean
```

## Proxy Robustness Metrics

### `results/metrics/final_robustness_metrics.csv`

Clean-to-perturbed comparison fields include:

```text
clean_accuracy
perturbed_accuracy
accuracy_drop
clean_macro_f1
perturbed_macro_f1
f1_drop
attack_success_rate
induced_error_count
confidence_shift
weapon_to_non_weapon_count
non_weapon_to_weapon_count
```

Positive `accuracy_drop` means degradation relative to the clean baseline.

## Proxy OOD Metrics

### `results/metrics/final_ood_metrics.csv`

Important fields:

```text
evaluated_model
sample_type
total
predicted_weapon
predicted_non_weapon
predicted_weapon_rate
confidence_mean
high_confidence_threshold
high_confidence_count
high_confidence_rate
```

### OOD denominator rule

The dataset contains 500 unique OOD images. Each image is evaluated by five fold-specific checkpoints for each architecture:

```text
500 unique OOD images × 5 folds = 2,500 predictions per architecture
2,500 × 3 architectures = 7,500 proxy OOD prediction rows
```

Therefore `total = 2500` does not mean 2,500 distinct OOD images.

## XAI Selection Manifest

### `explainability/manifests/chapter5/thesis_selection.csv`

Documents the five Integrated Gradients cases used in Chapter 5, including model/fold, scenario, prediction, confidence, historical convergence metadata, thesis asset paths, and selection rationale.

XAI results are qualitative diagnostics for transparent proxies only.

## Interpretation Warning

Proxy metrics and commercial-tool metrics are not interchangeable:

- proxy outputs come from known architectures and fold-specific checkpoints;
- commercial results are normalized from observable black-box exports;
- commercial tools may not expose probabilities or homogeneous thresholds;
- all commercial mappings are operational recodings specific to the frozen protocol.
