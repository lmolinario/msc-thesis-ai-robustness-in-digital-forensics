# Milestone 08 — Forensic Evaluation Bundle

## Status

Started.

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

## Expected logical structure

```text
datasets/forensic_evaluation_bundle/
├── blind_tool_input/
│   └── files/
├── metadata/
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

## Bundle content

The bundle is expected to include selected or complete samples from:

| Sample type | Source |
|---|---|
| clean binary samples | `datasets/splits/clean/` |
| OOD samples | `datasets/splits/ood/` |
| adversarial samples | `attacks/adversarial/` |
| anti-forensic samples | `attacks/anti_forensic/` |

## Required traceability fields

The bundle metadata should preserve, at minimum:

```text
blind_filename
sample_type
original_image_id
generated_image_id
fold
final_label
source_dataset
attack_family
attack_name
attack_target_model
original_relative_path
bundle_relative_path
sha256
md5
extension
size_bytes
```

Additional fields may be included when useful for auditing or tool-output normalization.

## Validation checklist

Before running forensic tools, the bundle must be checked for:

| Check | Expected result |
|---|---|
| blind filenames do not expose labels | true |
| blind filenames do not expose attack names | true |
| clean samples included | true |
| OOD samples included | true |
| adversarial samples included | true |
| anti-forensic samples included | true |
| SHA256 available for all bundle files | true |
| MD5 available where required for tool matching | true |
| internal mapping to original image identifiers | true |
| internal mapping to attack metadata | true |

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

The bundle must separate tool-facing filenames from internal experimental metadata. Forensic tools should receive opaque filenames, while the thesis pipeline preserves the mapping through manifests and hash-based traceability.

This design supports an operationally realistic scenario: the forensic tool is not given direct information about whether a file is clean, adversarial, anti-forensic, OOD, weapon, or non-weapon.

The bundle is not the final evaluation result. It is the controlled input package required to obtain forensic tool outputs, normalize them, and compare them with local proxy model metrics.

## Next step

The next milestone should document forensic tool execution and output normalization:

```text
progress/milestones/09_forensic_tool_evaluation.md
```
