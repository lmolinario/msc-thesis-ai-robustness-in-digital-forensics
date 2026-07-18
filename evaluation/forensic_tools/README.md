# Commercial-tool Evaluation Outputs

This directory stores the public audit and canonical prediction layer for the commercial black-box normalization performed by step 19.

## Canonical public prediction table

The repository-wide prediction-level source for commercial tools is:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

It contains exactly 69,000 sanitized decisions:

- 11,500 Magnet AXIOM decisions;
- 11,500 Excire D20 decisions;
- 11,500 Excire D50 decisions;
- 11,500 Excire D80 decisions;
- 11,500 Cellebrite decisions;
- 11,500 Griffeye decisions.

The table is reconstructed only from the four validated public extracts and contains anonymized bundle identifiers, experimental-condition fields, normalized decisions, and the minimum observable tool signals required for audit. It does not contain raw-export paths, local file-system paths, device identifiers, image hashes, unrelated EXIF data, PhotoDNA, or case-management fields.

Its provenance, SHA256 digest, source-extract hashes, schema, and row profile are recorded in:

```text
evaluation/forensic_tools/normalized_predictions_public_summary.json
```

The public schema is recorded separately in:

```text
evaluation/forensic_tools/normalized_predictions.schema.csv
```

## Other public audit artifacts

```text
normalization_summary.json
normalized_predictions.schema.csv
normalized_predictions_public_summary.json
unmatched_predictions.schema.csv
tool_export_audit.csv
tool_version_log.csv
```

The final quantitative outputs remain under:

```text
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool_name>_metrics.csv
```

Validated tool-specific extracts remain public under:

```text
forensic_tools/*/public_extracts/
```

Their equivalence report and hashes are:

```text
forensic_tools/public_extracts_validation.json
forensic_tools/public_extracts_summary.json
```

## Exact equivalence status

The committed validation report confirms:

```text
69,000 prediction rows are identical
186 frozen metric rows are identical
```

The validation source is the canonical table itself:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

## Locally generated non-canonical outputs

The following pipeline outputs remain local and excluded from `main`:

```text
unmatched_predictions.csv
<tool_name>_normalized_predictions.csv
```

The complete commercial-tool raw exports are also local or controlled-access inputs and are not distributed on `main`.

## Rebuilding from the validated public extracts

The canonical table can be reproduced without the proprietary raw exports:

```bash
python forensic_tools/scripts/build_canonical_normalized_predictions.py --force
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --report forensic_tools/public_extracts_validation.json \
  --force
```

## Full local normalization from raw exports

Where authorized raw exports are locally available, use the numbered entry point:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py \
  --no-interactive \
  --strict \
  --force \
  --tools magnet_axiom excire_foto_2025 cellebrite_inseyets griffeye
```

For the official frozen profile, supply the local run directories documented in `forensic_tools/run_registry.json`. The wrapper validates six configurations, 69,000 matched predictions, 11,500 unique bundle IDs per configuration, no `unknown` predictions, and the frozen unmatched-row profile.

## Data minimization

The public prediction workflow retains only:

- anonymized `bundle_XXXXXX` identifiers;
- condition fields required to recompute metrics;
- observable classifications, tags, prompt hits, or bookmarks;
- normalized binary decisions.

It does not propagate full local paths, volume names, file-system offsets, serial numbers, PhotoDNA, unrelated EXIF data, or case-management fields.

`tool_export_audit.csv` is retained as a historical record of the files processed during the frozen experiment. Its `raw_export_file` paths describe the original repository layout and are not expected to resolve on current `main`.

The complete pre-cleanup export layout remains preserved at:

```text
archive/pre-commission-cleanup-2026-07-16
snapshot/pre-commission-cleanup-2026-07-16
309a4580537ebc3bb7950f29c090bb2729fc603b
```
