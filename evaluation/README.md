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
│   └── 19_normalize_forensic_tool_outputs.py        # planned / next implementation step
├── proxy_models/
│   └── proxy_model_predictions.csv
└── forensic_tools/
    ├── magnet_axiom/
    ├── xways_excire/
    ├── cellebrite_ufed/
    └── oxygen/
```

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

The proxy evaluation is considered consolidated for the thesis reporting phase. It provides the reproducible baseline against which black-box forensic-tool behavior can later be compared.

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

Target tools:

```text
Magnet AXIOM / Magnet.AI
X-Ways Forensics / Excire
Cellebrite UFED
Oxygen Forensic Detective
```

---

## Planned Normalization Step

Planned official entry point:

```text
evaluation/scripts/19_normalize_forensic_tool_outputs.py
```

Purpose:

- ingest commercial tool exports;
- map exported items back to `bundle_manifest.csv` using filename, path, SHA256, MD5, or exported metadata;
- normalize tool labels into a common schema;
- compute tool-level metrics;
- generate thesis-ready metrics and audit tables.

Expected normalized output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

---

## Separation Rule

Proxy model outputs and commercial forensic-tool outputs must not be merged manually.

The only valid bridge between black-box tool results and ground truth is:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

This preserves the forensic logic of blind evaluation and prevents label leakage during tool analysis.
