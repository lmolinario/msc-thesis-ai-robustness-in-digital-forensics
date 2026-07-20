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
    ├── normalized_predictions.schema.csv
    ├── unmatched_predictions.schema.csv
    ├── tool_export_audit.csv
    ├── tool_version_log.csv
    └── normalization_summary.json
```

The underscored scripts preserve the implementation used for the frozen
experiments. The numbered entry points add validation, output protection and
public-data minimization without changing the experimental algorithms.

## Controlled-data boundary

Image corpora are not distributed on `main`. Restore the authorized dataset and
regenerate the local clean, OOD, adversarial, anti-forensic and blind-bundle
image trees before executing the evaluation pipeline.

Commercial prediction-level outputs are also generated locally. They contain
fine-grained per-sample mappings derived from proprietary exports and are not
published on `main`. The public repository retains their schemas, normalization
summary, export audit, version log and final metric tables.

## Proxy-model evaluation

Official entry point:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

The frozen profile evaluates:

- 1,000 clean binary images;
- 500 OOD images;
- 5,000 adversarial samples;
- 5,000 anti-forensic samples;
- EfficientNet-B0, ResNet18 and the CLIP binary-head proxy.

Binary samples use their fold-matched checkpoint. The same 500 OOD images are
evaluated with all five fold-specific checkpoints.

The entry point validates:

- the exact 11,500-sample composition;
- the five canonical adversarial manifests;
- local image SHA256 values against the manifests;
- all 15 checkpoint SHA256 values against `models/model_registry.json`;
- absence of sample-level inference errors.

Partial, limited or single-fold runs must use `--diagnostic-output-dir`; they
cannot overwrite the canonical outputs.

Public frozen outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_*_metrics.csv
results/metrics/final_*_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

## Commercial-tool normalization

Official entry point:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Final evaluated configurations:

- Magnet AXIOM / Magnet.AI 10.1.0.48673;
- Excire Foto 2025 4.1.5 at D20, D50 and D80;
- Cellebrite Inseyets 10.9 / Physical Analyzer 10.9.0.3029;
- Magnet Griffeye x64 26.2.108 with T3K CORE 1.18.0.

The frozen normalization produced six configurations and 69,000 matched
predictions (`6 × 11,500`). A further 329 exported Cellebrite rows did not map
to the official blind bundle; they were excluded from metrics and are recorded
in the public summary.

The wrapper minimizes public/local normalized fields by retaining only the
anonymized bundle filename and the observable classification needed for the
binary recoding. It does not copy full local paths, volume names, offsets or
unrelated file-system metadata into normalized prediction rows.

Canonical output files are protected against accidental partial overwrites.
Use `--force` only for an intentional complete regeneration. Per-tool prediction
copies are optional through `--write-per-tool-files`; the aggregate file is the
local source of truth.

Public frozen outputs:

```text
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool_name>_metrics.csv
```

Locally generated and Git-ignored outputs:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/unmatched_predictions.csv
evaluation/forensic_tools/*_normalized_predictions.csv
```

## Operational-risk summary

The reporting helper:

```text
evaluation/scripts/compute_proxy_operational_risk_metrics.py
```

reads the canonical proxy prediction table and computes one operational-risk
summary row for each transparent proxy model. It combines those computed rows
with the consolidated black-box values already reported in Chapter 5 so that
the final comparison table and figure can be regenerated without manual editing.
It does not rerun commercial tools, alter normalized decisions, or replace the
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
consolidated Chapter 5 results.

## Separation rule

Proxy-model and commercial-tool predictions must not be merged manually. The
only valid bridge to hidden ground truth is:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_manifest.csv
```

Commercial binary metrics cover 11,000 bundle items. OOD behavior is reported
separately over the 500 OOD items. XAI applies only to transparent proxy models
and belongs under `explainability/`.
