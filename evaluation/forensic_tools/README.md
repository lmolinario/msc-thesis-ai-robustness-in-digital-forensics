# Commercial-tool Evaluation Outputs

This directory stores the public audit layer for the commercial black-box
normalization performed by step 19.

## Public artifacts

```text
normalization_summary.json
normalized_predictions.schema.csv
unmatched_predictions.schema.csv
tool_export_audit.csv
tool_version_log.csv
```

The final quantitative outputs remain under:

```text
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool_name>_metrics.csv
```

## Prediction-level outputs

The following files are intentionally generated locally and excluded from
`main`:

```text
normalized_predictions.csv
unmatched_predictions.csv
<tool_name>_normalized_predictions.csv
```

They contain one row per tool configuration and bundle item, including mappings
derived from proprietary exports. Their public distribution is unnecessary for
verifying the thesis-level results and may retain local export details if
produced by older pipeline versions.

Use the numbered entry point to regenerate minimized files:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py \
  --no-interactive \
  --strict \
  --force \
  --tools magnet_axiom excire_foto_2025 cellebrite_inseyets griffeye
```

For the official frozen profile, supply the selected run directories documented
in `normalization_summary.json`. The wrapper validates six configurations,
69,000 matched predictions, 11,500 unique bundle IDs per configuration, no
`unknown` predictions, and the frozen unmatched-row profile.

## Data minimization

The current entry point retains only:

- the anonymized `bundle_XXXXXX` filename;
- bundle identifiers and hashes needed for traceability;
- the observable tool classification/bookmark/tag;
- the normalized binary recoding and correctness fields.

It does not propagate full local paths, volume names, file-system offsets or
unrelated report fields into normalized prediction rows.
