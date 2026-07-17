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

- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `evaluation/forensic_tools/normalization_summary.json`
- `results/metrics/magnet_axiom_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The original AXIOM export includes fields not required by the weapon-tag metric, such as file-system timestamps, hashes, paths, EXIF-derived values, device identifiers, and case-export metadata. The raw export is temporarily retained on `main` while a minimized public extract is built and validated.

No raw export will be removed until the sanitized extract has been shown to reproduce the frozen bundle-level decisions and metrics exactly. The protected branch and tag documented in `docs/artifact/ARCHIVE_SNAPSHOT.md` preserve the complete pre-cleanup repository state for provenance.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```

Canonical regeneration also requires the local blind-bundle metadata and all official commercial-tool exports.
