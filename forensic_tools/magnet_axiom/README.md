# Magnet AXIOM / Magnet.AI

## Experimental role

Magnet AXIOM / Magnet.AI was evaluated as a commercial black-box forensic image-triage system. The tool received only the semantically neutral blind bundle; ground-truth labels and attack metadata were kept outside the case and used only after export.

## Frozen run

| Field | Value |
|---|---|
| Tool | Magnet AXIOM / Magnet.AI |
| Version | `10.1.0.48673` |
| Run ID | `FAIRLAB_AXIOM_RUN_02` |
| Input | `datasets/forensic_evaluation_bundle/blind_tool_input/files/` |
| Analysis mode | Full search with Magnet.AI media analysis enabled |
| Enabled category | Weapons |
| Prediction export | `Pictures.csv` |
| Export rows | 11,500 |
| Matched bundle rows | 11,500 |
| Unmatched rows | 0 |

## Observable mapping

The quantitative evaluation uses the exported `Tags` field:

```text
Possible weapons  -> weapon_detected = true
otherwise         -> weapon_detected = false
```

This is an operational recoding of observable output. It does not imply access to Magnet.AI probabilities, thresholds, training data, or internal model logic.

## Public artifacts

- `forensic_tools/magnet_axiom/public_extracts/magnet_axiom_predictions_extract.csv`
- `forensic_tools/public_extracts_summary.json`
- `forensic_tools/public_extracts_validation.json`
- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `results/metrics/magnet_axiom_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The complete AXIOM export included file-system timestamps, hashes, paths, EXIF-derived values, device identifiers, and case-export metadata not required by the weapon-tag metric.

After exact equivalence validation of all 69,000 sanitized decisions and all 186 metric rows, the complete AXIOM raw export was removed from `main`. It remains preserved in the protected historical snapshot and may be restored locally under:

```text
forensic_tools/magnet_axiom/raw_exports/
```

That local path is ignored by Git.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```

Canonical regeneration requires the local blind-bundle metadata and all official commercial-tool raw exports.
