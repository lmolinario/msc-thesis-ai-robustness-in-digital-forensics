# Forensic Tools

This directory is reserved for documentation, export organization, normalization artifacts, and audit notes related to the commercial forensic-tool evaluation phase of the thesis.

Commercial forensic tools are treated as **operational black boxes**. The goal is not to reproduce, inspect, or infer their internal AI models, but to evaluate how their observable AI-assisted image-analysis behavior changes when the same forensic corpus contains clean, out-of-distribution (OOD), adversarial, and anti-forensic inputs.

The evaluation follows the thesis methodology and uses the official forensic evaluation bundle as the common input corpus.

---

## Current Tool Status

| Tool | Version | Status | Notes |
|---|---:|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Completed and normalized | The forensic evaluation bundle has been processed and the `Pictures.csv` export has been normalized against the bundle manifest. |
| Excire Foto 2025 | 4.1.5 | Completed / analyzed | Evaluated as a standalone general-purpose AI-assisted image retrieval tool in a controlled forensic context. Results are interpreted as semantic retrieval behavior, not as native forensic classification. |
| Cellebrite Inseyets | 10.9 | Pending / to be consolidated | Included in the final experimental perimeter. Requires documentation of the media-analysis workflow, export format, observable AI labels/categories, and comparability with the bundle ground truth. |
| Oxygen Forensic Detective | -- | Excluded | Not included in the final experimental perimeter. |
| Autopsy | -- | Excluded | Not included in the final experimental perimeter. |

---

## Target Tools

The commercial-tool evaluation phase covers the following tools:

```text
Consolidated / analyzed:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673
- Excire Foto 2025, version 4.1.5

Pending / to be consolidated:
- Cellebrite Inseyets, version 10.9

Excluded from final experimental perimeter:
- Oxygen Forensic Detective
- Autopsy
```

Each included tool must be documented separately, including:

* import procedure;
* software version;
* analysis configuration;
* export format;
* observable AI labels, categories, tags, or search outputs;
* mapping strategy to the forensic evaluation bundle;
* operational limitations observed during analysis.

The evaluation does not assume access to internal AI model logic, proprietary training data, internal thresholds, or undocumented decision rules.

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

## Recommended Directory Structure

```text
forensic_tools/
├── README.md
├── magnet_axiom/
│   ├── notes.md
│   └── raw_exports/
├── excire_foto_2025/
│   ├── notes.md
│   └── raw_exports/
└── cellebrite_inseyets/
    ├── notes.md
    └── raw_exports/
```

Large proprietary case files, installer files, licensed databases, and heavy exports should not be committed unless strictly necessary and legally/ethically appropriate.

Prefer:

* normalized CSV/JSON outputs;
* methodological notes;
* audit logs;
* export summaries;
* reproducible mapping artifacts.

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

The evaluation is based on observable semantic retrieval behavior under controlled configurations. Any mapping from Excire outputs to the binary `weapon` / `non_weapon` task must therefore be documented as an operational evaluation choice.

The documentation for Excire Foto 2025 should include:

```text
forensic_tools/excire_foto_2025/notes.md
forensic_tools/excire_foto_2025/raw_exports/
```

The notes should specify:

* software version: `Excire Foto 2025, version 4.1.5`;
* query or search configuration used;
* distance or similarity thresholds, if applicable;
* exported fields;
* mapping rule to the forensic evaluation bundle;
* operational limitations of treating semantic retrieval outputs as forensic triage signals.

---

## Cellebrite Inseyets Evaluation

Cellebrite Inseyets is included in the final experimental perimeter as the Cellebrite tool to be evaluated.

The documentation should specify:

```text
forensic_tools/cellebrite_inseyets/notes.md
forensic_tools/cellebrite_inseyets/raw_exports/
```

The notes should include:

* software version: `Cellebrite Inseyets, version 10.9`;
* import procedure for the forensic evaluation bundle;
* media-analysis or AI-assisted categorization workflow used;
* exported fields and export format;
* observable labels, categories, tags, or detections;
* mapping strategy to the bundle manifest;
* whether the output supports quantitative comparison with Magnet AXIOM / Magnet.AI and Excire Foto 2025;
* limitations caused by proprietary processing, export granularity, or unavailable internal model information.

The thesis should avoid attributing generic AI-based image-classification behavior to Cellebrite unless the specific Inseyets component and export used in the experiment support that claim.

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

The normalization process may support:

* Magnet AXIOM / Magnet.AI exports through `Pictures.csv`;
* Excire Foto 2025 semantic retrieval exports;
* Cellebrite Inseyets media-analysis exports, if exportable and mappable;
* generic CSV, TSV, JSON, JSONL, and TXT forensic AI exports;
* matching through filename, SHA256, and MD5;
* deduplication to one prediction per tool and bundle item;
* export audit and tool-version logging.

---

## Mapping Strategy

Tool outputs must be mapped back to the forensic evaluation bundle through:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Preferred matching keys, in order:

1. exported filename / bundle filename;
2. SHA256 hash;
3. MD5 hash;
4. exported path and file size;
5. manual audit only when automatic matching fails.

Manual matching decisions must be logged.

---

## Reporting Principle

Commercial-tool results should be reported separately from proxy-model results unless their exports have been normalized and mapped back to the forensic evaluation bundle.

The thesis must distinguish:

* transparent proxy-model robustness;
* black-box commercial-tool behavior;
* Magnet AXIOM / Magnet.AI as a consolidated commercial-tool result;
* Excire Foto 2025 as a standalone AI-assisted semantic retrieval tool evaluated in a controlled forensic context;
* Cellebrite Inseyets 10.9 as the Cellebrite tool included in the final experimental perimeter;
* Oxygen Forensic Detective and Autopsy as excluded tools;
* operational implications for AI-assisted triage;
* limitations caused by proprietary labels, export formats, unavailable confidence scores, and unknown internal model behavior.

Commercial-tool outputs are interpreted as observable operational signals, not as direct evidence of internal AI model performance.
