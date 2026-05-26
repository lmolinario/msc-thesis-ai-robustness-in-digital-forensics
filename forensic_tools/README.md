# Forensic Tools

This directory is reserved for documentation, export organization, and audit notes related to commercial forensic-tool evaluation.

Commercial forensic tools are treated as **operational black boxes**. The goal is not to reproduce their internal AI models, but to evaluate how their AI-assisted classification behavior changes when the same forensic corpus contains clean, OOD, adversarial, and anti-forensic inputs.

---

## Current Tool Status

| Tool | Status | Notes |
|---|---|---|
| Magnet AXIOM / Magnet.AI | Completed and normalized | The forensic evaluation bundle has been processed and the `Pictures.csv` export has been normalized against the bundle manifest. |
| X-Ways Forensics / Excire Photo AI | Pending / planned extension | Requires tool/module availability and an export format that can be mapped back to the bundle. |
| Cellebrite UFED | Pending / planned extension | Requires confirmation of AI-based image-classification support and exportability for this experimental setup. |
| Oxygen Forensic Detective | Pending / planned extension | Requires available AI modules and normalized export mapping. |

---

## Target Tools

The commercial forensic-tool evaluation phase covers one consolidated tool and three optional extensions:

```text
Consolidated:
- Magnet AXIOM / Magnet.AI

Pending / planned extensions:
- X-Ways Forensics / Excire Photo AI
- Cellebrite UFED
- Oxygen Forensic Detective
```

Each tool should be documented separately, including import procedure, export format, relevant AI labels/categories, software version, and operational limitations observed during analysis.

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

The excluded directories contain labels, attack names, provenance, hashes, source information, and ground-truth metadata. They are reserved for normalization after tool export.

---

## Recommended Directory Structure

```text
forensic_tools/
├── README.md
├── magnet_axiom/
│   ├── notes.md
│   └── raw_exports/
├── xways_excire/
│   ├── notes.md
│   └── raw_exports/
├── cellebrite_ufed/
│   ├── notes.md
│   └── raw_exports/
└── oxygen_forensic_detective/
    ├── notes.md
    └── raw_exports/
```

Large proprietary case files and heavy exports should not be committed unless strictly necessary and legally/ethically appropriate. Prefer normalized CSV/JSON outputs and methodological notes.

---

## Magnet AXIOM / Magnet.AI Consolidated Run

The consolidated commercial-tool run currently available in the repository is:

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

The Magnet normalization maps the exported `Possible weapons` tag to `weapon_detected=true` and the absence of that tag to `weapon_detected=false`. This is an operational recoding of observable tool output, not access to Magnet.AI internal model logic.

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

The current script supports:

- Magnet AXIOM / Magnet.AI exports through `Pictures.csv`;
- generic CSV, TSV, JSON, JSONL and TXT forensic AI exports;
- matching through filename, SHA256 and MD5;
- deduplication to one prediction per tool and bundle item;
- export audit and tool-version logging.

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

Manual matching decisions should be logged.

---

## Reporting Principle

Commercial tool results should be reported separately from proxy model results unless their exports have been normalized and mapped back to the forensic evaluation bundle.

The thesis should distinguish:

- transparent proxy model robustness;
- black-box commercial forensic-tool behavior;
- Magnet AXIOM / Magnet.AI as the consolidated commercial-tool result;
- X-Ways/Excire, Cellebrite UFED and Oxygen as pending extensions unless comparable exports become available;
- operational implications for AI-assisted triage;
- limitations caused by proprietary labels, export formats, and unknown internal model behavior.
