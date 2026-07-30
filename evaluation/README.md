# Evaluation

This directory contains the frozen evaluation layer of the thesis artifact. It
keeps transparent proxy-model inference separate from the normalization of
commercial black-box tool exports.

## Structure

```text
evaluation/
├── README.md
├── scripts/
│   ├── 15_evaluate_proxy_models.py
│   ├── _15_evaluate_proxy_models_impl.py
│   ├── 19_normalize_forensic_ai_tool_predictions.py
│   ├── _19_normalize_forensic_ai_tool_predictions_impl.py
│   └── compute_proxy_operational_risk_metrics.py
├── proxy_models/
│   └── proxy_model_predictions.csv
└── forensic_tools/
    ├── README.md
    ├── normalized_predictions.csv
    ├── normalized_predictions_public_summary.json
    ├── normalized_predictions.schema.csv
    ├── unmatched_predictions.schema.csv
    ├── tool_export_audit.csv
    ├── tool_version_log.csv
    └── normalization_summary.json
```

The underscored scripts preserve the implementation used for the frozen
experiments. The numbered entry points add validation, output protection, and
public-data minimization without changing the experimental algorithms.

## Controlled-Data Boundary

Image corpora are not distributed on `main`. Restore the authorized dataset and
regenerate the local clean, OOD, adversarial, anti-forensic, and blind-bundle
image trees before executing image-dependent evaluation stages.

Complete commercial raw exports are also excluded from `main`. The curated
public artifact instead retains:

- a canonical sanitized prediction table containing 69,000 decisions;
- four tool-specific sanitized extracts used to reconstruct that table;
- schemas, normalization summaries, export audits, version logs, and metrics;
- an exact equivalence report covering all decisions and 186 metric rows.

The canonical sanitized table is public and committed at:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

It must not be confused with the excluded complete raw exports.

## Proxy-Model Evaluation

Official entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

The frozen profile evaluates:

- 1,000 clean binary images;
- 500 unique OOD images;
- 5,000 adversarial or adversarial-style artifacts;
- 5,000 anti-forensic artifacts;
- EfficientNet-B0, ResNet18, and the CLIP-based binary-head proxy.

Binary samples use their fold-matched checkpoint. The same 500 OOD images are
evaluated with all five fold-specific checkpoints, producing 2,500 OOD
predictions per architecture and 7,500 OOD rows in total.

The entry point validates:

- the exact 11,500-item composition;
- the five canonical adversarial manifests;
- local image SHA-256 values against the manifests;
- all 15 checkpoint SHA-256 values against `models/model_registry.json`;
- absence of sample-level inference errors.

Partial, limited, or single-fold runs must use `--diagnostic-output-dir`; they
cannot overwrite the canonical outputs.

Public frozen outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_*_metrics.csv
results/metrics/final_*_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

Fields historically named `confidence*` contain maximum predicted-class
probability (`Max-P`) information. Max-P is an intra-model diagnostic and is not
calibrated confidence or forensic certainty.

## Commercial-Tool Normalization

Official entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Final evaluated configurations:

- Magnet AXIOM / Magnet.AI 10.1.0.48673;
- Excire Foto 2025 4.1.5 at D20, D50, and D80;
- Cellebrite Inseyets 10.9 / Physical Analyzer 10.9.0.3029;
- Magnet Griffeye x64 26.2.108 with T3K CORE 1.18.0.

The frozen normalization produced six complete 11,500-row configurations and
69,000 matched bundle decisions (`6 × 11,500`). A further 329 exported
Cellebrite rows did not map to the official blind bundle; they were excluded
from metrics and are recorded in the normalization summary.

Normalization retains only the fields necessary for traceability and the
observable operational recoding. It does not expose complete raw paths, volume
names, offsets, image hashes, unrelated metadata, case-management fields, or
proprietary internals in the canonical public table.

Canonical public outputs:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool_name>_metrics.csv
forensic_tools/public_extracts_validation.json
```

Locally generated outputs from a new authorized raw-export rerun must remain
separate until they have been fully validated. Complete raw exports and
unmatched raw rows must not be committed.

## Operational-Risk Summary

The reporting helper:

```text
evaluation/scripts/compute_proxy_operational_risk_metrics.py
```

reads the canonical proxy prediction table and computes one operational-risk
summary row for each transparent proxy model. It combines those computed rows
with the consolidated black-box values reported in Chapter 6 so that the final
comparison table and figure can be regenerated without manual editing. It does
not rerun commercial tools, alter normalized decisions, or replace the
canonical metric files produced by steps 15 and 19.

Run from the repository root:

```bash
python evaluation/scripts/compute_proxy_operational_risk_metrics.py
```

Outputs:

```text
results/metrics/proxy_operational_risk_metrics.csv
results/metrics/operational_risk_summary_data.csv
docs/LatexThesis/images/fig_results_operational_risk_summary.pdf
docs/LatexThesis/images/fig_results_operational_risk_summary.png
```

The script validates the expected frozen proxy counts before writing the
reporting artifacts. Its embedded black-box rows must remain aligned with the
consolidated Chapter 6 results.

## Separation Rule

Proxy-model and commercial-tool predictions must not be merged manually. The
only valid bridge to hidden ground truth is:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Commercial binary metrics cover 11,000 binary-condition items. OOD behavior is
reported separately over 500 OOD items. XAI applies only to transparent proxy
models and belongs under `explainability/`.
