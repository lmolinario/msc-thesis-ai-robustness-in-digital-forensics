# Excire Foto 2025

## Experimental role

Excire Foto 2025 was evaluated as a standalone AI-assisted semantic image-retrieval system in a controlled forensic context. It is not presented as a native forensic suite or as an internally binary weapon classifier.

## Frozen configurations

| Configuration | Run ID | Version | Raw prompt-hit rows | Completed bundle rows |
|---|---|---:|---:|---:|
| D20 | `FAIRLAB_EXCIRE_D20_FIREARM_PROMPTS` | 4.1.5 | 20,329 | 11,500 |
| D50 | `FAIRLAB_EXCIRE_D50_FIREARM_PROMPTS` | 4.1.5 | 32,076 | 11,500 |
| D80 | `FAIRLAB_EXCIRE_D80_FIREARM_PROMPTS` | 4.1.5 | 43,736 | 11,500 |

Each configuration used the same fixed firearm-oriented prompt set:

```text
firearm
gun
pistol
handgun
revolver
rifle
shotgun
assault rifle
```

## Observable mapping

For each distance configuration, an image retrieved by at least one fixed prompt is mapped to `weapon_detected=true`. Images not retrieved by any prompt are completed against the frozen 11,500-item bundle as `weapon_detected=false`.

The three settings remain separate normalized configurations:

```text
excire_foto_2025_d20
excire_foto_2025_d50
excire_foto_2025_d80
```

## Public artifacts

- `forensic_tools/excire_foto_2025/public_extracts/excire_prompt_hits_extract.csv`
- `forensic_tools/public_extracts_summary.json`
- `forensic_tools/public_extracts_validation.json`
- `forensic_tools/run_registry.json`
- `evaluation/forensic_tools/tool_version_log.csv`
- `evaluation/forensic_tools/tool_export_audit.csv`
- `results/metrics/excire_foto_2025_d20_metrics.csv`
- `results/metrics/excire_foto_2025_d50_metrics.csv`
- `results/metrics/excire_foto_2025_d80_metrics.csv`
- `results/metrics/forensic_tools_metrics.csv`

## Raw-export boundary

The 24 original prompt-hit CSV files included local Windows paths in addition to anonymized bundle filenames. The sanitized public extract retains only bundle identifiers, semantic-distance configuration, prompt membership, and normalized decision fields.

After exact equivalence validation, the 24 raw CSV files were removed from `main`. They remain preserved in the protected historical snapshot and may be restored locally under:

```text
forensic_tools/excire_foto_2025/raw_exports/
```

That local path is ignored by Git.

## Normalization

The official entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```
