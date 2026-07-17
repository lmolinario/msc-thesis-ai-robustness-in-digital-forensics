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

- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `evaluation/forensic_tools/normalization_summary.json`
- `results/metrics/cellebrite_inseyets_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The original report contains substantially more information than the `Classifications` field required for this experiment, including file-system and case-export details. It is temporarily retained on `main` while a minimized classification extract is generated and validated.

The planned public extract will contain only anonymized bundle identifiers, the observable classification field, match status, and normalized binary decision. The complete report will not be removed before exact equivalence validation and an explicit final decision.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```
