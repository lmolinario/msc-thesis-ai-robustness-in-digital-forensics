# Milestone 09 — Commercial Forensic Tools Evaluation

## Status

Completed and normalized.

## Purpose

This milestone documents the completed black-box evaluation of the final commercial forensic-tool perimeter on the validated forensic evaluation bundle.

The goal is to assess the operational reliability of AI-assisted image classification and media-triage behavior in realistic Digital/Computer Forensics workflows. The focus is not on optimizing adversarial attacks, but on measuring how commercial tools behave when clean, adversarial, anti-forensic, and out-of-distribution images are processed in a controlled and traceable protocol.

---

## Input Bundle

Commercial tools processed only the blind input directory:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

The following directories were not imported into the tools:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

These directories contain ground truth, perturbation metadata, source information, and hash mappings. They are reserved for post-export normalization and audit.

---

## Bundle Composition

The validated forensic evaluation bundle contains:

| Sample type / family | Count |
|---|---:|
| clean | 1000 |
| OOD | 500 |
| adversarial | 5000 |
| anti-forensic | 5000 |
| **total** | **11500** |

The bundle passed the required structural and integrity checks:

```text
bundle_id_unique                       = true
sha256_actual_unique                   = true
all_sha256_match_when_manifest_present = true
blind_paths_semantically_clean         = true
metadata_separated_from_tool_input     = true
```

---

## Final Tool Perimeter

The final commercial / black-box evaluation perimeter is:

| Tool | Version / module | Status |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Completed and normalized |
| Excire Foto 2025 | 4.1.5 | Completed and normalized as standalone AI-assisted semantic retrieval |
| Cellebrite Inseyets | 10.9 | Completed and normalized |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE v1.18.0 | Completed and normalized |

Commercial tools are treated as operational black boxes. Their outputs are interpreted as observable operational signals, not as direct evidence of internal model behavior.

---

## Raw Export Areas

Raw commercial-tool exports are stored under:

```text
forensic_tools/magnet_axiom/raw_exports/
forensic_tools/excire_foto_2025/raw_exports/
forensic_tools/cellebrite_inseyets/raw_exports/
forensic_tools/griffeye/raw_exports/
```

The final documented runs are:

```text
forensic_tools/magnet_axiom/raw_exports/FAIRLAB_AXIOM_RUN_02
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D20_FIREARM_PROMPTS
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D50_FIREARM_PROMPTS
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D80_FIREARM_PROMPTS
forensic_tools/cellebrite_inseyets/raw_exports/FAIRLAB_CELLEBRITE_INSEYETS_RUN_01
forensic_tools/griffeye/raw_exports/FAIRLAB_GRIFFEYE_T3_RUN_01
```

---

## Normalization Script

The official normalization entry point is:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

The normalization process:

- ingests commercial tool exports;
- maps exported items back to `bundle_manifest.csv` using filename, path, SHA256, MD5, or exported metadata;
- normalizes labels, bookmarks, categories, and semantic retrieval outputs into a common schema;
- deduplicates to one prediction per tool/configuration and bundle item;
- generates audit logs, version logs, normalized predictions, and thesis-ready metrics.

---

## Normalized Outputs

Main normalized outputs:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/magnet_axiom_normalized_predictions.csv
evaluation/forensic_tools/excire_foto_2025_d20_normalized_predictions.csv
evaluation/forensic_tools/excire_foto_2025_d50_normalized_predictions.csv
evaluation/forensic_tools/excire_foto_2025_d80_normalized_predictions.csv
evaluation/forensic_tools/cellebrite_inseyets_normalized_predictions.csv
evaluation/forensic_tools/griffeye_normalized_predictions.csv
```

Main metric outputs:

```text
results/metrics/forensic_tools_metrics.csv
results/metrics/magnet_axiom_metrics.csv
results/metrics/excire_foto_2025_d20_metrics.csv
results/metrics/excire_foto_2025_d50_metrics.csv
results/metrics/excire_foto_2025_d80_metrics.csv
results/metrics/cellebrite_inseyets_metrics.csv
results/metrics/griffeye_metrics.csv
```

---

## Mapping Strategy

Forensic tool outputs are matched back to the experimental ground truth using:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Preferred matching keys are:

1. exported filename / bundle filename;
2. SHA256 hash;
3. MD5 hash;
4. exported path and file size;
5. manual audit only when automatic matching fails.

Hash-based matching is preferred whenever possible because forensic tools may rename files, alter paths, or export results using tool-specific identifiers.

---

## Completion Criteria

This milestone is complete because:

- the final commercial-tool perimeter processed the blind input directory;
- raw exports are stored under tool-specific `raw_exports/` folders;
- exports have been normalized into a common schema;
- predictions have been matched back to bundle identifiers and ground truth through metadata and hashes;
- consolidated and tool-specific metrics have been produced;
- tool-specific limitations, mappings, and configuration details are documented in the repository and thesis text.
