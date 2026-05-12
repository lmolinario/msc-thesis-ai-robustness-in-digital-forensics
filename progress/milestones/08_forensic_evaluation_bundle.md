# Milestone 08 — Forensic Evaluation Bundle

## Status

Completed and verified.

## Official script

`datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py`

## Input

The forensic evaluation bundle is built after dataset freezing, split generation, perturbation generation, and proxy model evaluation.

Main input areas:

```text
datasets/splits/clean/
datasets/splits/ood/
attacks/adversarial/
attacks/anti_forensic/
datasets/splits/manifests/
attacks/manifests/
```

## Output directory

```text
datasets/forensic_evaluation_bundle/
```

## Verified logical structure

```text
datasets/forensic_evaluation_bundle/
├── blind_tool_input/
│   └── files/
├── metadata/
│   ├── bundle_manifest.csv
│   ├── bundle_hashes_sha256.csv
│   └── bundle_summary.json
└── structured_audit_view/
```

## Purpose

The forensic evaluation bundle is the operational bridge between the local experimental pipeline and commercial or professional forensic AI tools.

Its purpose is to:

- provide a blind input set for forensic tools;
- avoid leaking class labels, attack names, or provenance through filenames;
- preserve full internal traceability through metadata and hashes;
- support later normalization of forensic tool outputs;
- make local proxy model results and forensic tool outputs comparable.

## Verified bundle content

The generated bundle contains the complete clean, OOD, adversarial, and anti-forensic corpus prepared for forensic-tool evaluation.

| Sample type / family | Count |
|---|---:|
| clean | 1000 |
| OOD | 500 |
| adversarial | 5000 |
| anti-forensic | 5000 |
| total bundle rows | 11500 |
| blind files | 11500 |
| structured files | 11500 |

Detailed attack distribution:

| Attack / group | Count |
|---|---:|
| clean | 1000 |
| ood | 500 |
| color_shift | 1000 |
| fgsm | 1000 |
| one_pixel | 1000 |
| sigma_zero | 1000 |
| superdeepfool | 1000 |
| jpeg_recompression | 1000 |
| resample_resize | 1000 |
| gaussian_blur | 1000 |
| histogram_modification | 1000 |
| contrast_stretching | 1000 |

Label distribution:

| Label | Count |
|---|---:|
| weapon | 5500 |
| non_weapon | 5500 |
| ood | 500 |

## Metadata and hashes

Verified metadata outputs:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
datasets/forensic_evaluation_bundle/metadata/bundle_summary.json
```

The metadata layer preserves both SHA256 and MD5 values for forensic-tool matching and post-export normalization.

The bundle generation script writes:

```text
sha256_actual
md5_actual
sha256_manifest
md5_manifest
sha256_matches_manifest
```

and the compact hash file contains:

```text
bundle_id
sha256
md5
tool_input_filename
tool_input_relative_path
blind_relative_path
structured_relative_path
sample_type
attack_family
attack_name
final_label
original_image_id
generated_image_id
```

## Verified traceability fields

The generated `bundle_manifest.csv` preserves the following key mapping fields:

```text
bundle_id
tool_input_filename
sample_type
attack_family
attack_name
attack_target_model
fold
final_label
source_dataset
original_image_id
generated_image_id
source_manifest
source_relative_path
blind_relative_path
structured_relative_path
tool_input_relative_path
original_sha256
original_md5
sha256_manifest
md5_manifest
sha256_actual
md5_actual
sha256_matches_manifest
size_bytes
extension
layout
created_at
```

This satisfies the required mapping toward:

```text
original_image_id
attack_name
sample_type
```

and also preserves the additional fields needed for forensic output normalization.

## Verified checks

| Check | Result |
|---|---:|
| bundle ID unique | true |
| actual SHA256 unique | true |
| SHA256 values match source manifests when manifest hashes are present | true |
| blind paths semantically clean | true |
| metadata separated from tool input | true |
| clean samples included | true |
| OOD samples included | true |
| adversarial samples included | true |
| anti-forensic samples included | true |
| SHA256 available | true |
| MD5 available | true |
| internal mapping to original image identifiers | true |
| internal mapping to attack metadata | true |

## Blind-input rule

For black-box forensic-tool evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files
```

Do not import:

```text
datasets/forensic_evaluation_bundle/metadata
datasets/forensic_evaluation_bundle/structured_audit_view
```

The blind flat layout is designed to reduce path-induced and analyst-induced bias. Ground-truth labels, perturbation metadata, source information, and hash mappings are preserved only in metadata manifests for post-export normalization.

## Forensic tool targets

The bundle is designed to be processed by black-box forensic AI tools, including:

```text
Magnet AXIOM / Magnet AI
X-Ways / Excire
Cellebrite UFED
Oxygen Forensic Detective
```

Tool-specific outputs should later be stored under:

```text
forensic_tools/
evaluation/forensic_tools/
results/metrics/
```

## Methodological notes

The bundle separates tool-facing filenames from internal experimental metadata. Forensic tools receive opaque filenames, while the thesis pipeline preserves the mapping through manifests and hash-based traceability.

This design supports an operationally realistic scenario: the forensic tool is not given direct information about whether a file is clean, adversarial, anti-forensic, OOD, weapon, or non-weapon.

The bundle is not the final evaluation result. It is the controlled input package required to obtain forensic tool outputs, normalize them, and compare them with local proxy model metrics.

## Next step

The next milestone documents commercial forensic tool execution and output normalization planning:

```text
progress/milestones/09_commercial_forensic_tools_evaluation.md
```
