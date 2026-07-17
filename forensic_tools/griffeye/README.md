# Magnet Griffeye / T3K CORE

## Experimental role

Magnet Griffeye with T3K CORE was evaluated as a commercial black-box forensic media-triage and semantic-bookmarking system. Only automatically generated bookmarks were used for quantitative analysis; no manual bookmark addition, removal, or correction was permitted.

## Frozen run

| Field | Value |
|---|---|
| Tool | Magnet Griffeye x64 |
| Version | `26.2.108` |
| Semantic module | T3K CORE `1.18.0` |
| Run ID | `FAIRLAB_GRIFFEYE_T3_RUN_01` |
| Export format | CSV |
| Raw export rows | 12,053 |
| Non-bundle rows excluded | 553 |
| Matched bundle rows | 11,500 |
| Unmatched bundle rows | 0 |

## Observable mapping

The primary thesis mapping is firearm-specific:

```text
CORE/Violence/Firearm present  -> weapon_detected = true
otherwise                       -> weapon_detected = false
```

The following automatically generated bookmarks are retained only as secondary semantic indicators and are excluded from the primary weapon metric:

```text
CORE/Violence/Explosive Weapon
CORE/Violence/Bladed Weapon
CORE/Violence/Archery Weapon
CORE/Military/Military Equipment
```

## Public artifacts

- `forensic_tools/griffeye/public_extracts/griffeye_bookmarks_extract.csv`
- `forensic_tools/public_extracts_summary.json`
- `forensic_tools/public_extracts_validation.json`
- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `results/metrics/griffeye_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The original Griffeye CSV contained multiple hashes, PhotoDNA, paths, timestamps, EXIF fields, face-detection fields, RIC/CSA columns, file-system records, and case-management metadata unrelated to the firearm-bookmark metric.

The sanitized public extract preserves only anonymized bundle identifiers, the firearm bookmark, selected secondary weapon bookmarks, and normalized decision fields. After exact equivalence validation, the complete raw CSV was removed from `main`.

It remains preserved in the protected historical snapshot and may be restored locally under:

```text
forensic_tools/griffeye/raw_exports/
```

That local path is ignored by Git.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```
