# Commercial-tool Evaluation Outputs

This directory stores the public audit layer for the commercial black-box normalization performed by step 19.

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

Validated sanitized prediction-level extracts are public under:

```text
forensic_tools/*/public_extracts/
```

Their equivalence report and hashes are:

```text
forensic_tools/public_extracts_validation.json
forensic_tools/public_extracts_summary.json
```

## Complete normalized outputs

The following complete pipeline outputs are generated locally and excluded from `main`:

```text
normalized_predictions.csv
unmatched_predictions.csv
<tool_name>_normalized_predictions.csv
```

The complete commercial-tool raw exports are also local/controlled-access inputs and are not distributed on `main`.

Use the numbered entry point to regenerate normalized files:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py \
  --no-interactive \
  --strict \
  --force \
  --tools magnet_axiom excire_foto_2025 cellebrite_inseyets griffeye
```

For the official frozen profile, supply the local run directories documented in `forensic_tools/run_registry.json`. The wrapper validates six configurations, 69,000 matched predictions, 11,500 unique bundle IDs per configuration, no `unknown` predictions, and the frozen unmatched-row profile.

## Data minimization

The current entry point and public-extract workflow retain only:

- anonymized `bundle_XXXXXX` identifiers;
- condition fields required to recompute metrics;
- observable classifications, tags, prompt hits, or bookmarks;
- normalized binary recoding and correctness fields.

They do not propagate full local paths, volume names, file-system offsets, serial numbers, PhotoDNA, unrelated EXIF data, or case-management fields.

`tool_export_audit.csv` is retained as a historical record of the files processed during the frozen experiment. Its `raw_export_file` paths describe the original repository layout and are not expected to resolve on current `main`.

The complete pre-cleanup export layout remains preserved at:

```text
archive/pre-commission-cleanup-2026-07-16
snapshot/pre-commission-cleanup-2026-07-16
309a4580537ebc3bb7950f29c090bb2729fc603b
```
