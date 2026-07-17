# Cellebrite Inseyets / Physical Analyzer

## Experimental role

Cellebrite Inseyets was evaluated as a commercial black-box AI-assisted media-analysis system. The quantitative analysis is based only on observable classifications exported through Physical Analyzer.

## Frozen run

| Field | Value |
|---|---|
| Tool | Cellebrite Inseyets |
| Version | `10.9` |
| Physical Analyzer | `10.9.0.3029` |
| UFED component | `10.9.0.284` |
| Run ID | `FAIRLAB_CELLEBRITE_INSEYETS_RUN_01` |
| Export format | XLSX report |
| Raw report rows | 11,829 |
| Matched bundle rows | 11,500 |
| Unmatched rows | 329 |

The 329 unmatched rows are preserved as an audit count and are excluded from quantitative bundle metrics.

## Observable mapping

The normalization reads only the exported `Classifications` field. A row is mapped to `weapon_detected=true` when the field contains at least one of:

```text
Armi
Pistola
Fucile
```

All other matched bundle rows are mapped to `weapon_detected=false`.

This operational mapping does not imply access to internal Cellebrite model probabilities, decision thresholds, training data, or undocumented categorization logic.

## Public artifacts

- `forensic_tools/cellebrite_inseyets/public_extracts/cellebrite_classifications_extract.csv`
- `forensic_tools/public_extracts_summary.json`
- `forensic_tools/public_extracts_validation.json`
- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `results/metrics/cellebrite_inseyets_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The original XLSX report contained file-system and case-export details not required by the `Classifications` mapping. The public extract retains anonymized bundle identifiers, the observable classification field, and normalized decision fields; the 329 non-bundle rows remain documented only as aggregate audit counts.

After exact equivalence validation, the complete report was removed from `main`. It remains preserved in the protected historical snapshot and may be restored locally under:

```text
forensic_tools/cellebrite_inseyets/raw_exports/
```

That local path is ignored by Git.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```
