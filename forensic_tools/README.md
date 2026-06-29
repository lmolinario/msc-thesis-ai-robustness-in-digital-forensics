# Forensic Tools

This directory is reserved for documentation, export organization, normalization artifacts, and audit notes related to the commercial forensic-tool evaluation phase of the frozen thesis.

Commercial forensic tools are treated as **operational black boxes**. The goal is not to reproduce, inspect, or infer their internal AI models, but to evaluate how their observable AI-assisted image-analysis behavior changes when the same forensic corpus contains clean, out-of-distribution (OOD), adversarial, and anti-forensic inputs.

The evaluation follows the thesis methodology and uses the official forensic evaluation bundle as the common blind input corpus.

---

## Final Tool Status

| Tool | Version | Status | Notes |
|---|---:|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Completed and normalized | The forensic evaluation bundle has been processed and the `Pictures.csv` export has been normalized against the bundle manifest. |
| Excire Foto 2025 | 4.1.5 | Completed and normalized | Evaluated as a standalone general-purpose AI-assisted image retrieval tool in a controlled forensic context with `D20`, `D50`, and `D80` semantic-distance configurations. |
| Cellebrite Inseyets | 10.9 | Completed and normalized | Evaluated through the Cellebrite Physical Analyzer report export. Observable image classifications are normalized as black-box operational signals. |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108 / T3K CORE v1.18.0 | Completed and normalized | Evaluated through automatically generated T3K CORE semantic bookmarks. Primary mapping uses `CORE/Violence/Firearm` only. |

---

## Final Commercial-Tool Perimeter

The final commercial-tool evaluation phase covers the following tools:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673
- Excire Foto 2025, version 4.1.5
- Cellebrite Inseyets, version 10.9
- Magnet Griffeye x64, version 26.2.108, with T3K CORE v1.18.0
```

Each included tool is documented and normalized according to:

- import procedure;
- software version;
- analysis configuration;
- export format;
- observable AI labels, categories, tags, bookmarks, or search outputs;
- mapping strategy to the forensic evaluation bundle;
- operational limitations observed during analysis.

The evaluation does not assume access to internal AI model logic, proprietary training data, internal thresholds, calibrated confidence scores, or undocumented decision rules.

---

## Input Rule

For blind black-box evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

The excluded directories contain labels, attack names, provenance, hashes, source information, and ground-truth metadata. They are reserved for normalization and evaluation after the commercial-tool export has been produced.

---

## Actual Directory Structure

```text
forensic_tools/
├── README.md
├── magnet_axiom/
│   └── raw_exports/
├── excire_foto_2025/
│   └── raw_exports/
├── cellebrite_inseyets/
│   └── raw_exports/
└── griffeye/
    └── raw_exports/
```

Large proprietary case files, installer files, licensed databases, license files, and heavy exports should not be committed unless strictly necessary and legally/ethically appropriate.

Prefer:

- normalized CSV/JSON outputs;
- methodological notes;
- audit logs;
- export summaries;
- reproducible mapping artifacts;
- aggregated metric files.

---

## Magnet AXIOM / Magnet.AI Consolidated Run

The consolidated Magnet AXIOM / Magnet.AI run currently available in the repository is:

```text
forensic_tools/magnet_axiom/raw_exports/FAIRLAB_AXIOM_RUN_02
```

The normalized outputs are stored in:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/magnet_axiom_normalized_predictions.csv
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/forensic_tools/normalization_summary.json
results/metrics/forensic_tools_metrics.csv
results/metrics/magnet_axiom_metrics.csv
```

The Magnet normalization maps the exported `Possible weapons` tag to:

```text
weapon_detected = true
```

and the absence of that tag to:

```text
weapon_detected = false
```

This is an operational recoding of observable tool output, not access to Magnet.AI internal model logic.

---

## Excire Foto 2025 Evaluation

Excire Foto 2025 is evaluated as a standalone AI-assisted image retrieval tool, not as a native forensic software package and not as a binary weapon classifier.

The evaluation is based on observable semantic retrieval behavior under controlled configurations:

```text
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D20_FIREARM_PROMPTS
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D50_FIREARM_PROMPTS
forensic_tools/excire_foto_2025/raw_exports/FAIRLAB_EXCIRE_D80_FIREARM_PROMPTS
```

The normalization script treats each distance configuration as a separate operational setting:

```text
excire_foto_2025_d20
excire_foto_2025_d50
excire_foto_2025_d80
```

An image retrieved by at least one fixed firearm-oriented prompt is mapped to `weapon_detected=true`; all remaining bundle images are completed as `weapon_detected=false` for that configuration.

---

## Cellebrite Inseyets Evaluation

Cellebrite Inseyets is included in the final experimental perimeter as the Cellebrite commercial black-box tool.

Run/export folder:

```text
forensic_tools/cellebrite_inseyets/raw_exports/FAIRLAB_CELLEBRITE_INSEYETS_RUN_01
```

Documented environment:

```text
Cellebrite Inseyets version 10.9
Physical Analyzer 10.9.0.3029 / UFED 10.9.0.284
```

The normalized output is:

```text
evaluation/forensic_tools/cellebrite_inseyets_normalized_predictions.csv
results/metrics/cellebrite_inseyets_metrics.csv
```

The mapping is based on the observable `Classifications` column exported from the Cellebrite report. The extended operational mapping treats an image as `weapon_detected=true` when `Classifications` contains at least one among:

```text
Armi
Pistola
Fucile
```

This is an operational recoding of exported tool output and does not imply access to Cellebrite internal AI model logic.

---

## Magnet Griffeye / T3K CORE Evaluation

Magnet Griffeye is included as a fourth commercial black-box forensic media-triage tool.

Run/export folder:

```text
forensic_tools/griffeye/raw_exports/FAIRLAB_GRIFFEYE_T3_RUN_01
```

Documented environment:

```text
Magnet Griffeye x64 26.2.108
T3K CORE v1.18.0
```

The normalized output is:

```text
evaluation/forensic_tools/griffeye_normalized_predictions.csv
results/metrics/griffeye_metrics.csv
```

The evaluation relies exclusively on automatically generated T3K CORE semantic bookmarks. No manual bookmark addition, removal, or correction is used for quantitative metrics.

The primary thesis mapping is firearm-oriented:

```text
weapon_detected = true  if Bookmarks contains CORE/Violence/Firearm
weapon_detected = false otherwise
```

The following bookmarks are intentionally excluded from the primary metric and retained only as secondary semantic indicators:

```text
CORE/Violence/Explosive Weapon
CORE/Violence/Bladed Weapon
CORE/Violence/Archery Weapon
CORE/Military/Military Equipment
```

Griffeye normalization checks:

```text
rows in normalized prediction file = 11501 (1 header + 11500 predictions)
matched_rows                       = 11500
unmatched_rows                     = 0
unknown_rows                       = 0
positive firearm bookmarks         = 5399
negative / no-firearm rows         = 6101
```

---

## Normalization Target

Raw commercial-tool exports are normalized by the official script:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Expected normalized output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

The normalization process supports:

- Magnet AXIOM / Magnet.AI exports through `Pictures.csv`;
- Excire Foto 2025 semantic retrieval prompt exports;
- Cellebrite Inseyets / Physical Analyzer report exports;
- Griffeye / T3K CORE CSV exports with automatic semantic `Bookmarks`;
- matching through filename, SHA256, and MD5;
- deduplication to one prediction per tool and bundle item;
- export audit and tool-version logging.

---

## Mapping Strategy

Tool outputs are mapped back to the forensic evaluation bundle through:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Preferred matching keys, in order:

1. exported filename / bundle filename;
2. SHA256 hash;
3. MD5 hash;
4. exported path and file size;
5. manual audit only when automatic matching fails.

Manual matching decisions must be logged. Manual correction of model/tool predictions is not used for quantitative performance evaluation.

---

## Reporting Principle

Commercial-tool results should be reported separately from proxy-model results unless their exports have been normalized and mapped back to the forensic evaluation bundle.

The thesis must distinguish:

- transparent proxy-model robustness;
- black-box commercial-tool behavior;
- Magnet AXIOM / Magnet.AI as a consolidated commercial-tool result;
- Excire Foto 2025 as a standalone AI-assisted semantic retrieval tool evaluated in a controlled forensic context;
- Cellebrite Inseyets 10.9 as a commercial black-box AI-assisted media-analysis tool;
- Magnet Griffeye / T3K CORE as a commercial black-box semantic-bookmark media-triage tool;
- operational implications for AI-assisted triage;
- limitations caused by proprietary labels, export formats, unavailable confidence scores, semantic mappings, and unknown internal model behavior.

Commercial-tool outputs are interpreted as observable operational signals, not as direct evidence of internal AI model performance.
