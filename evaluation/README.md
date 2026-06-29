# Evaluation

This directory contains the evaluation layer of the frozen thesis pipeline.

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
    ├── excire_foto_2025_d20_normalized_predictions.csv
    ├── excire_foto_2025_d50_normalized_predictions.csv
    ├── excire_foto_2025_d80_normalized_predictions.csv
    ├── cellebrite_inseyets_normalized_predictions.csv
    └── griffeye_normalized_predictions.csv
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

Main outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

The proxy evaluation is consolidated for thesis reporting and provides the reproducible baseline against which black-box forensic-tool behavior can be compared.

---

## Commercial Forensic Tool Evaluation

Commercial forensic tools are evaluated as black boxes.

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
- Excire Foto 2025, version 4.1.5
- Cellebrite Inseyets, version 10.9
- Magnet Griffeye x64, version 26.2.108, with T3K CORE v1.18.0
```

Excire Foto 2025 is evaluated as a standalone AI-assisted semantic retrieval tool. Cellebrite Inseyets is evaluated through observable image classifications exported by the Cellebrite Physical Analyzer report. Magnet Griffeye is evaluated through automatic T3K CORE semantic bookmarks.

---

## Normalization Step

Official entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Purpose:

- ingest commercial tool exports;
- map exported items back to `bundle_manifest.csv` using filename, path, SHA256, MD5, or exported metadata;
- normalize tool labels, bookmarks, and search outputs into a common schema;
- compute tool-level metrics;
- generate thesis-ready metrics and audit tables.

Implemented / consolidated behavior:

- Magnet AXIOM / Magnet.AI normalization from `Pictures.csv`;
- Excire Foto 2025 prompt-hit normalization for `D20`, `D50`, and `D80` configurations;
- Cellebrite Inseyets normalization from the report sheet `Immagini` and the observable `Classifications` column;
- Griffeye / T3K CORE normalization from automatic semantic `Bookmarks`;
- deduplication to one prediction per `tool_name` + `bundle_id`;
- export audit and tool version log generation.

Expected normalized output area:

```text
evaluation/forensic_tools/
results/metrics/forensic_tools_metrics.csv
```

Current normalization summary:

```text
bundle_rows                         = 11500
tools_requested                     = magnet_axiom, excire_foto_2025, cellebrite_inseyets, griffeye
normalized_rows_after_deduplication = 69000
matched_rows_after_deduplication    = 69000
unmatched_rows_after_deduplication  = 0
interpretable_rows_after_dedup      = 69000
unknown_predictions                 = 0
metric_outputs_consistent           = true
```

---

## Commercial Tool Global Binary Metrics

The following values are computed on the 11,000 binary bundle items. OOD behavior is reported separately as `OOD flag rate` over the 500 OOD samples.

| Tool / configuration | Accuracy | Recall | FNR | FPR | OOD flag rate |
|---|---:|---:|---:|---:|---:|
| Magnet Griffeye / T3K CORE | 0.971727 | 0.950727 | 0.049273 | 0.007273 | 0.260000 |
| Cellebrite Inseyets 10.9 | 0.958091 | 0.957818 | 0.042182 | 0.041636 | 0.292000 |
| Magnet AXIOM / Magnet.AI | 0.933364 | 0.901455 | 0.098545 | 0.034727 | 0.360000 |
| Excire Foto 2025 D20 | 0.910727 | 0.857091 | 0.142909 | 0.035636 | 0.238000 |
| Excire Foto 2025 D50 | 0.924545 | 0.948545 | 0.051455 | 0.099455 | 0.340000 |
| Excire Foto 2025 D80 | 0.887091 | 0.981273 | 0.018727 | 0.207091 | 0.522000 |

---

## Griffeye / T3K CORE Normalization Details

Griffeye is evaluated as a commercial black-box forensic media-triage tool through automatic semantic bookmarks.

```text
Tool           = Magnet Griffeye x64
Version        = 26.2.108
AI module      = T3K CORE
Module version = 1.18.0
Run folder     = forensic_tools/griffeye/raw_exports/FAIRLAB_GRIFFEYE_T3_RUN_01
```

The primary mapping is firearm-oriented and relies only on the corresponding T3K CORE bookmark. Other related T3K CORE categories are retained in the raw label field as secondary semantic indicators, but they are not used as primary positives.

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
