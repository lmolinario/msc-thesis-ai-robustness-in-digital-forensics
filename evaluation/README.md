# Evaluation

This directory contains the evaluation layer of the thesis pipeline.

The evaluation stage has two distinct purposes:

1. evaluate transparent local proxy models on clean, OOD, adversarial, and anti-forensic inputs;
2. normalize and evaluate commercial forensic-tool outputs after black-box analysis of the forensic evaluation bundle.

The two streams must remain separated until commercial tool exports have been normalized and mapped back to the bundle manifest.

---

## Directory Structure

```text
evaluation/
├── README.md
├── scripts/
│   ├── 15_evaluate_proxy_models.py
│   └── 19_normalize_forensic_ai_tool_predictions.py
├── proxy_models/
│   └── proxy_model_predictions.csv
└── forensic_tools/
    ├── normalized_predictions.csv
    ├── tool_export_audit.csv
    ├── tool_version_log.csv
    ├── normalization_summary.json
    ├── magnet_axiom_normalized_predictions.csv
    ├── xways_excire_normalized_predictions.csv
    └── cellebrite_inseyets_normalized_predictions.csv
```

`cellebrite_inseyets_normalized_predictions.csv` is a target filename only if comparable Cellebrite Inseyets exports become available and can be mapped to the forensic evaluation bundle.

---

## Proxy Model Evaluation

Official entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Evaluated proxy models:

```text
efficientnet_b0
resnet18
clip
```

Evaluation inputs:

```text
datasets/splits/clean/
datasets/splits/ood/
attacks/adversarial/
attacks/anti_forensic/
```

Main outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

The proxy evaluation is considered consolidated for the thesis reporting phase. It provides the reproducible baseline against which black-box forensic-tool behavior can be compared.

---

## Commercial Forensic Tool Evaluation

Commercial forensic tools must be evaluated as black boxes.

Input to tools:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

These directories contain ground truth, perturbation metadata, provenance, labels, and hash mappings. They are reserved for post-export normalization.

Final tool perimeter:

```text
Completed and normalized:
- Magnet AXIOM / Magnet.AI, version 10.1.0.48673

Completed / analyzed:
- X-Ways Forensics / Excire Foto 2025, version 4.1.5

Pending / to be consolidated:
- Cellebrite Inseyets, version 10.9

Excluded:
- Oxygen Forensic Detective
- Autopsy
```

Excire Foto 2025 is evaluated as a standalone AI-assisted semantic retrieval tool. Its results must be interpreted as controlled semantic retrieval behavior, not as native forensic binary classification.

---

## Normalization Step

Official entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Purpose:

- ingest commercial tool exports;
- map exported items back to `bundle_manifest.csv` using filename, path, SHA256, MD5, or exported metadata;
- normalize tool labels into a common schema;
- compute tool-level metrics;
- generate thesis-ready metrics and audit tables.

Implemented / consolidated behavior:

- Magnet AXIOM / Magnet.AI normalization from `Pictures.csv`;
- mapping of `Possible weapons` to `weapon_detected=true`;
- mapping of empty Magnet tags to `weapon_detected=false`;
- generic parsing for CSV, TSV, JSON, JSONL and TXT exports;
- deduplication to one prediction per `tool_name` + `bundle_id`;
- export audit and tool version log generation.

Expected normalized output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

Cellebrite Inseyets outputs should be normalized only if the export structure supports reproducible mapping to the bundle manifest and the exported labels/categories can be operationally related to the thesis ground truth.

---

## XAI / Explainability Status

Integrated Gradients is not part of the commercial-tool normalization layer. It is used only for transparent proxy models and has been completed for the Chapter 5 representative case-study discussion.

Selected Chapter 5 cases:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

---

## Separation Rule

Proxy model outputs and commercial forensic-tool outputs must not be merged manually.

The only valid bridge between black-box tool results and ground truth is:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

This preserves the forensic logic of blind evaluation and prevents label leakage during tool analysis.
