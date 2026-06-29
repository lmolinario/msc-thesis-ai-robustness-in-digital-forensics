# Data Dictionary

This document summarizes the main CSV and JSON artifacts used by the thesis repository. It is designed to help reviewers interpret manifests, predictions, normalized outputs, and final metrics without inspecting every script.

---

## Reading Principles

- `image_id` identifies an image in the curated dataset before bundle expansion.
- `bundle_id` identifies an item in the blind forensic evaluation bundle.
- `final_label` is the frozen semantic label used for the binary forensic task or OOD separation.
- `attack_name` identifies the perturbation or transformation condition.
- `tool_name` identifies the commercial or proxy evaluation source.
- Commercial-tool labels are normalized operational signals, not internal model outputs.

---

## `datasets/final/manifests/manual_selection_final_1500.csv`

Purpose:

```text
official frozen 1500-image dataset manifest
```

Main columns:

| Column | Meaning |
|---|---|
| `image_id` | Stable image identifier in the curated dataset |
| `relative_path` | Path to the prepared image relative to the dataset root |
| `source_dataset` | Source collection identifier |
| `source_group` | Higher-level source group |
| `sha256` | SHA256 hash of the image file |
| `final_label` | Frozen semantic label: `weapon`, `non_weapon`, or `ood` |
| `review_state` | Manual review status |
| `review_notes` | Optional reviewer notes |
| `reviewer_id` | Reviewer identifier |
| `review_timestamp` | Review timestamp |
| `source_priority` | Source-priority value used during selection |

---

## `datasets/final/manifests/manual_selection_adversarial_subset.csv`

Purpose:

```text
official 1000-image binary subset used for clean folds and perturbation generation
```

Expected labels:

```text
weapon      = 500
non_weapon  = 500
```

Main columns follow the same structure as `manual_selection_final_1500.csv`.

---

## `datasets/splits/manifests/clean_folds_manifest.csv`

Purpose:

```text
fold-aware clean binary manifest used for proxy-model training, attack generation, and clean evaluation
```

Typical columns:

| Column | Meaning |
|---|---|
| `image_id` | Original curated image identifier |
| `fold` | Fold assignment used for fold-aware training/evaluation |
| `final_label` | Binary label: `weapon` or `non_weapon` |
| `relative_path` | Path to the clean image |
| `sha256` | SHA256 hash |
| `source_dataset` | Source dataset identifier |

---

## `datasets/splits/manifests/ood_eval_manifest.csv`

Purpose:

```text
clean out-of-distribution evaluation manifest
```

Important rule:

```text
OOD samples remain clean and are not used for adversarial or anti-forensic perturbation generation.
```

Typical columns include `image_id`, `relative_path`, `source_dataset`, `sha256`, and OOD-related labels or split identifiers.

---

## `attacks/manifests/`

Purpose:

```text
traceability layer for generated adversarial and anti-forensic artifacts
```

Typical columns:

| Column | Meaning |
|---|---|
| `generated_image_id` | Identifier of the generated perturbed item |
| `original_image_id` | Source clean image identifier |
| `fold` | Fold inherited from the clean image |
| `final_label` | Original clean binary label |
| `attack_family` | `adversarial` or `anti_forensic` |
| `attack_name` | Specific perturbation or transformation |
| `attack_parameters` | Serialized parameters used for generation |
| `target_model` | Target proxy model for model-dependent attacks, if applicable |
| `model_dependency` | Whether the perturbation is model-dependent or model-agnostic |
| `checkpoint_path` | Fold-aware checkpoint path, when applicable |
| `checkpoint_sha256` | Checkpoint hash, when applicable |
| `sha256_original` | Hash of the original image |
| `sha256_perturbed` | Hash of the generated image |
| `md5_perturbed` | MD5 hash used for commercial-tool matching |
| `perturbed_relative_path` | Path to the generated image |

---

## `datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv`

Purpose:

```text
official metadata source for the 11500-item blind forensic evaluation bundle
```

Important fields:

| Column | Meaning |
|---|---|
| `bundle_id` | Stable bundle item identifier |
| `original_image_id` | Original clean image identifier, when applicable |
| `sample_type` | `clean`, `ood`, `adversarial`, or `anti_forensic` |
| `attack_family` | Perturbation family or `none` |
| `attack_name` | Specific condition name |
| `final_label` | Ground-truth label used for evaluation |
| `blind_relative_path` | Semantically neutral bundle path used for tool import |
| `sha256_actual` | SHA256 hash of the bundle file |
| `md5_actual` | MD5 hash of the bundle file |

The metadata directory is not imported into commercial tools. It is used only after export for normalization and audit.

---

## `evaluation/proxy_models/proxy_model_predictions.csv`

Purpose:

```text
prediction-level output for transparent proxy models
```

Typical columns:

| Column | Meaning |
|---|---|
| `evaluated_model` | Proxy model name: `efficientnet_b0`, `resnet18`, or `clip` |
| `sample_type` | Clean, OOD, adversarial, or anti-forensic condition |
| `attack_family` | Attack/transformation family |
| `attack_name` | Specific condition |
| `attack_target_model` | Target model used for generation, if applicable |
| `image_id` / `bundle_id` | Image or bundle identifier |
| `true_label` | Ground-truth label where defined |
| `predicted_label` | Predicted binary label |
| `confidence` | Model confidence associated with the prediction |
| `correct` | Whether prediction matches the ground truth, where applicable |

---

## `evaluation/forensic_tools/normalized_predictions.csv`

Purpose:

```text
common normalized prediction schema for commercial forensic-tool outputs
```

Typical columns:

| Column | Meaning |
|---|---|
| `tool_name` | Normalized tool identifier |
| `tool_version` | Tool version or module version |
| `configuration` | Tool-specific configuration or threshold setting |
| `bundle_id` | Matched bundle item |
| `sample_type` | Clean, OOD, adversarial, or anti-forensic condition |
| `final_label` | Ground-truth label from bundle metadata |
| `weapon_detected` | Normalized operational binary signal |
| `raw_label` | Original exported label/category/bookmark/query result |
| `match_method` | Matching method used to link export row to bundle metadata |
| `match_status` | Matched, unmatched, duplicate, or audited status |

Commercial-tool outputs are black-box operational observations. They are not interpreted as access to internal model predictions.

---

## `results/metrics/forensic_tools_metrics.csv`

Purpose:

```text
consolidated commercial-tool metric summary
```

Typical metrics:

| Metric | Meaning |
|---|---|
| `recall_weapon` | Weapon-class recall |
| `false_negative_rate` | Weapon images missed by the tool |
| `false_positive_rate` | Non-weapon images flagged as weapon |
| `unknown_rate` | Share of rows that could not be mapped or interpreted, if applicable |
| `ood_weapon_flag_rate` | Rate at which OOD images are flagged as weapon |
| `total` | Number of evaluated items |

---

## `results/metrics/final_core_metrics.csv`

Purpose:

```text
thesis-oriented core proxy-model metrics across clean and perturbed conditions
```

Important columns include:

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

---

## `results/metrics/final_robustness_metrics.csv`

Purpose:

```text
robustness-oriented comparison between clean and perturbed proxy-model performance
```

Important columns include:

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

---

## `results/metrics/final_ood_metrics.csv`

Purpose:

```text
OOD-specific proxy-model reliability summary
```

Important columns include:

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

---

## Interpretation Warning

Metrics for transparent proxy models and commercial black-box tools are not interchangeable:

- proxy-model metrics are derived from controlled model outputs with known architectures;
- commercial-tool metrics are derived from observable exports after normalization;
- commercial tools may not expose calibrated confidence, class probabilities, or internal thresholds;
- all commercial-tool mappings are operational recodings of exported labels, tags, bookmarks, or semantic retrieval results.
