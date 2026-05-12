# Milestone 09 — Commercial Forensic Tools Evaluation

## Status

Pending.

## Purpose

This milestone documents the planned black-box evaluation of commercial and professional forensic AI tools on the validated forensic evaluation bundle.

The goal is to assess the operational reliability of AI-assisted image classification in realistic Digital/Computer Forensics workflows. The focus is not on optimizing adversarial attacks, but on measuring how forensic tools behave when clean, adversarial, anti-forensic, and out-of-distribution images are processed in a controlled and traceable protocol.

---

## Input bundle

Commercial tools must process only the blind input directory:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import the following directories into the tools:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

These directories contain ground truth, perturbation metadata, source information, and hash mappings. They are reserved for post-export normalization and audit.

---

## Bundle composition

The validated forensic evaluation bundle contains:

| Sample type / family | Count |
|---|---:|
| clean | 1000 |
| OOD | 500 |
| adversarial | 5000 |
| anti-forensic | 5000 |
| **total** | **11500** |

The bundle has already passed the required structural and integrity checks:

```text
bundle_id_unique                       = true
sha256_actual_unique                   = true
all_sha256_match_when_manifest_present = true
blind_paths_semantically_clean         = true
metadata_separated_from_tool_input     = true
```

---

## Planned tools

The planned forensic tools are:

```text
Magnet AXIOM / Magnet.AI
X-Ways Forensics / Excire
Cellebrite UFED
Oxygen Forensic Detective
```

The availability of each tool, its exact version, enabled AI modules, export format, and runtime configuration must be documented during execution.

---

## Expected raw output structure

Tool exports should be stored under:

```text
forensic_tools/<tool_name>/raw_exports/
```

Recommended tool identifiers:

```text
magnet_axiom
xways_excire
cellebrite_ufed
oxygen_forensic_detective
```

Expected normalized outputs should later be stored under:

```text
forensic_tools/<tool_name>/normalized/
evaluation/forensic_tools/
```

---

## Planned normalization script

The planned normalization entry point is:

```text
evaluation/scripts/19_normalize_forensic_tool_outputs.py
```

The number `19` is reserved for forensic-tool normalization because the number `18` is already used by:

```text
explainability/scripts/18_xai_interactive_launcher.py
```

---

## Expected final metrics

The expected final metric output is:

```text
results/metrics/forensic_tools_metrics.csv
```

Additional intermediate outputs may include:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
```

---

## Methodological requirements

For each tool, record:

- tool name;
- exact version/build;
- operating system and workstation context;
- enabled AI modules or classifiers;
- import path used;
- export format;
- export timestamp;
- any manual filtering or interaction performed;
- any tool errors, skipped files, unsupported files, or warnings.

The evaluation must preserve the black-box protocol: the tool receives only semantically neutral files and must not receive class labels, perturbation names, source dataset labels, or ground-truth metadata.

---

## Post-export matching

Forensic tool outputs should be matched back to the experimental ground truth using:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
```

Hash-based matching is preferred whenever possible because forensic tools may rename files, alter paths, or export results using tool-specific identifiers.

---

## Completion criteria

This milestone will be complete when:

- each available forensic tool has processed the blind input directory;
- raw exports are stored under `forensic_tools/<tool_name>/raw_exports/`;
- exports are normalized into a common schema;
- predictions are matched back to bundle identifiers and ground truth through metadata and hashes;
- `results/metrics/forensic_tools_metrics.csv` is produced;
- tool-specific limitations, failures, and configuration details are documented.

Status: **pending**.
