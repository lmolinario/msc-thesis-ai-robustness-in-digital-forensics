# Data Dictionary

This document summarizes the principal CSV and JSON artifacts used by the thesis
repository.

## Terminology and Historical Field Names

The final thesis describes transparent-proxy probability outputs as **maximum
predicted-class probability** (`Max-P`). Several frozen CSV schemas retain the
historical field names `confidence`, `confidence_mean`, `confidence_shift`, and
`high_confidence_*`. In this repository those fields must be interpreted as
model-specific Max-P values or summaries. They are not calibrated confidence,
forensic certainty, or quantities directly comparable across architectures.

Likewise, paths and identifiers containing `chapter5` or `chapter_5` predate the
final separation between implementation (Chapter 5) and results (Chapter 6).
They are preserved as frozen artifact identifiers. The associated results are
discussed in Chapter 6.

## General Identifiers

| Field | Meaning |
|---|---|
| `image_id` | Stable identifier in the curated source dataset |
| `original_image_id` | Identifier of the clean source image from which another artifact derives |
| `generated_image_id` | Identifier assigned to a perturbed image |
| `bundle_id` | Anonymous identifier in the 11,500-item forensic evaluation bundle |
| `fold` | Fold used for fold-aware proxy training or evaluation |
| `final_label` | Frozen operational assignment: `weapon`, `non_weapon`, or `ood` |
| `sample_type` | `clean`, `ood`, `adversarial`, or `anti_forensic` |
| `attack_family` | `none`, `adversarial`, or `anti_forensic` |
| `attack_name` | Specific perturbation, transformation, or `clean` condition |
| `tool_name` | Normalized commercial configuration identifier |
| `evaluated_model` | Transparent proxy architecture being evaluated |

Commercial-tool fields are observable operational signals. They must not be
interpreted as access to proprietary internal probabilities or decision logic.

## Frozen Dataset Manifests

### `datasets/final/manifests/manual_selection_final_1500.csv`

Official 1,500-image source-population manifest:

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
| `final_label` | Frozen operational assignment |
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

It is the source for clean folds, adversarial generation, anti-forensic
generation, and binary robustness evaluation. OOD remains a separate evaluation
branch and is not a supervised third class.

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

Each of the five held-out folds contains 200 items: 100 `weapon` and 100
`non_weapon`.

### `datasets/splits/manifests/ood_eval_manifest.csv`

Clean OOD evaluation manifest. OOD images are not used as a supervised third
class and are not perturbed by the attack-generation stages.

## Attack Manifests

Files under `attacks/manifests/` provide traceability for generated adversarial
and anti-forensic items.

| Field | Meaning |
|---|---|
| `generated_image_id` | Perturbed artifact identifier |
| `original_image_id` | Source clean image identifier |
| `fold` | Inherited fold |
| `final_label` | Original binary reference assignment |
| `attack_family` | `adversarial` or `anti_forensic` |
| `attack_name` | Specific condition |
| `attack_parameters` | Serialized generation parameters |
| `target_model` | Target architecture where applicable |
| `model_dependency` | Model-dependent or model-agnostic status |
| `checkpoint_sha256` | Checkpoint integrity digest where applicable |
| `sha256_original` | Source-image digest |
| `sha256_perturbed` | Generated-image digest |
| `md5_perturbed` | Auxiliary compatibility digest for some commercial matching workflows |
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
| `final_label` | Hidden reference operational assignment |
| `blind_relative_path` | Semantically neutral import path |
| `sha256_actual` | Bundle-file SHA-256 |
| `md5_actual` | Auxiliary bundle-file MD5 |

The metadata and structured audit views must not be imported into commercial
tools.

## Proxy Prediction Table

### `evaluation/proxy_models/proxy_model_predictions.csv`

Prediction-level output for EfficientNet-B0, ResNet18, and the CLIP-based proxy.

| Field | Meaning |
|---|---|
| `evaluated_model` | Proxy architecture |
| `fold` | Fold-specific checkpoint |
| `bundle_id` | Bundle item |
| `sample_type` | Experimental condition |
| `attack_family` | Condition family |
| `attack_name` | Specific condition |
| `attack_target_model` | Attack generation target where applicable |
| `final_label` | Reference assignment where defined |
| `predicted_label` | Binary proxy prediction |
| `confidence` | Historical field name containing prediction Max-P |
| `correct` | Reference-assignment agreement where meaningful |

`confidence` is an intra-model maximum predicted-class probability and is not
assumed to be calibrated across architectures.

## Canonical Commercial-Tool Prediction Table

### `evaluation/forensic_tools/normalized_predictions.csv`

This is the public repository-wide source for commercial prediction-level
analysis.

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
| `final_label` | Hidden bundle reference assignment joined after processing |
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

The canonical table deliberately excludes raw export paths, raw row numbers,
local filenames, image hashes, unrelated metadata, PhotoDNA, serial numbers,
and local absolute paths.

Its schema and SHA-256 are recorded in:

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
| `false_negative_rate` | Missed `weapon` rate |
| `false_positive_rate` | `non_weapon` false-alarm rate |
| `ood_weapon_flag_rate` | Share of OOD items mapped to `weapon` |

## Proxy Core Metrics

### `results/metrics/final_core_metrics.csv`

Important fields include:

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

`confidence_mean` is the historical schema name for mean Max-P.

## Proxy Robustness Metrics

### `results/metrics/final_robustness_metrics.csv`

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

`confidence_shift` is the historical schema name for the change in mean Max-P.
Positive `accuracy_drop` means degradation relative to the clean baseline.
`attack_success_rate` is interpreted as attack success only for the four
model-dependent adversarial attacks; for model-agnostic and anti-forensic
conditions, the corresponding transition is an induced-error rate.

## Proxy OOD Metrics

### `results/metrics/final_ood_metrics.csv`

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

The `confidence_*` and `high_confidence_*` names are historical schema fields.
They describe Max-P and the rate above the recorded Max-P threshold, not
calibrated confidence.

### OOD Denominator Rule

The dataset contains 500 unique OOD images. Each image is evaluated by five
fold-specific checkpoints for each architecture:

```text
500 unique OOD images × 5 folds = 2,500 predictions per architecture
2,500 × 3 architectures = 7,500 proxy OOD prediction rows
```

Therefore `total = 2500` does not mean 2,500 distinct OOD images.

## XAI Selection Manifest

### `explainability/manifests/chapter5/thesis_selection.csv`

Documents the five Integrated Gradients cases discussed in Chapter 6, including
model/fold, scenario, prediction, historical `confidence` field (Max-P),
convergence metadata, thesis asset paths, and selection rationale.

XAI results are qualitative diagnostics for transparent proxies only.

## Interpretation Warning

Proxy metrics and commercial-tool metrics are not interchangeable:

- proxy outputs come from known architectures and fold-specific checkpoints;
- commercial results are normalized from observable black-box exports;
- commercial tools do not expose homogeneous probabilities or thresholds;
- all commercial mappings are operational recodings specific to the frozen
  protocol.
