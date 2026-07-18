# Forensic Tools

This directory documents the commercial black-box evaluation layer of the frozen MSc thesis artifact.

The tools are evaluated only through observable exports. The repository does not claim access to proprietary architectures, internal probabilities, model weights, training data, thresholds, or undocumented decision logic.

## Experimental perimeter

| Tool / configuration | Version | Operational role | Status |
|---|---:|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Forensic AI-assisted image categorization | Completed |
| Excire Foto 2025 D20 | 4.1.5 | Semantic image retrieval | Completed |
| Excire Foto 2025 D50 | 4.1.5 | Semantic image retrieval | Completed |
| Excire Foto 2025 D80 | 4.1.5 | Semantic image retrieval | Completed |
| Cellebrite Inseyets / Physical Analyzer | 10.9 / 10.9.0.3029 | AI-assisted media classification | Completed |
| Magnet Griffeye / T3K CORE | 26.2.108 / 1.18.0 | Semantic media triage and bookmarking | Completed |

The six normalized configurations each cover the same 11,500-item forensic evaluation bundle.

## Blind-input rule

Commercial tools receive only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

They must not receive:

```text
datasets/forensic_evaluation_bundle/metadata/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

Ground truth, attack conditions, source information, and hash mappings are used only after export for normalization and audit.

## Public structure

```text
forensic_tools/
├── README.md
├── run_registry.json
├── public_extracts_summary.json
├── public_extracts_validation.json
├── scripts/
│   ├── build_public_tool_extracts.py
│   ├── build_canonical_normalized_predictions.py
│   └── validate_public_extract_equivalence.py
├── magnet_axiom/
│   ├── README.md
│   └── public_extracts/
├── excire_foto_2025/
│   ├── README.md
│   └── public_extracts/
├── cellebrite_inseyets/
│   ├── README.md
│   └── public_extracts/
└── griffeye/
    ├── README.md
    └── public_extracts/
```

The combined canonical prediction table is stored separately under:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

Complete commercial-tool raw exports are not distributed on `main`.

## Run registry

The consolidated run metadata are recorded in:

```text
forensic_tools/run_registry.json
```

The registry provides tool versions, frozen run IDs, export formats, observable fields, positive mapping rules, raw and matched row counts, public extract paths, the combined canonical table, metric artifacts, and the immutable historical snapshot reference.

## Observable mappings

| Configuration | Observable signal | Positive mapping |
|---|---|---|
| Magnet AXIOM | `Tags` | `Possible weapons` |
| Excire D20/D50/D80 | fixed semantic prompt retrieval | at least one firearm-oriented prompt hit |
| Cellebrite Inseyets | `Classifications` | `Armi`, `Pistola`, or `Fucile` |
| Griffeye / T3K CORE | `Bookmarks` | `CORE/Violence/Firearm` |

These are operational recodings of exported fields, not direct measurements of proprietary internal model probabilities.

## Canonical sanitized prediction table

The repository-wide commercial-tool prediction source is:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

It contains exactly 69,000 rows:

- 11,500 Magnet AXIOM decisions;
- 11,500 Excire D20 decisions;
- 11,500 Excire D50 decisions;
- 11,500 Excire D80 decisions;
- 11,500 Cellebrite decisions;
- 11,500 Griffeye decisions.

The table is reconstructed from the validated public extracts by:

```text
forensic_tools/scripts/build_canonical_normalized_predictions.py
```

Its SHA256 digest, columns, source-extract hashes, and decision profile are recorded in:

```text
evaluation/forensic_tools/normalized_predictions_public_summary.json
```

The table contains only anonymized bundle identifiers, experimental condition fields, normalized decisions, and minimum observable tool signals. It excludes raw-export paths, image hashes, device identifiers, unrelated EXIF data, PhotoDNA, local file-system information, and case-management fields.

## Tool-specific sanitized extracts

The source extracts used to construct the canonical table are:

```text
forensic_tools/magnet_axiom/public_extracts/magnet_axiom_predictions_extract.csv
forensic_tools/excire_foto_2025/public_extracts/excire_prompt_hits_extract.csv
forensic_tools/cellebrite_inseyets/public_extracts/cellebrite_classifications_extract.csv
forensic_tools/griffeye/public_extracts/griffeye_bookmarks_extract.csv
```

They also contain 69,000 rows in total and retain the minimum tool-specific observable signals required for audit.

## Exact equivalence validation

The validation report:

```text
forensic_tools/public_extracts_validation.json
```

records that:

```text
69,000 canonical decisions are identical
186 frozen metric rows are identical
```

The corresponding source hashes and row counts are recorded in:

```text
forensic_tools/public_extracts_summary.json
evaluation/forensic_tools/normalized_predictions_public_summary.json
```

The validator fails if any tool/bundle decision, condition field, or frozen metric changes.

## Rebuild from committed public artifacts

The canonical table can be regenerated without proprietary raw exports:

```bash
python forensic_tools/scripts/build_canonical_normalized_predictions.py --force
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --report forensic_tools/public_extracts_validation.json \
  --force
```

## Full local regeneration from authorized raw exports

Where the raw exports are legitimately available, restore them only under the ignored local directories:

```text
forensic_tools/**/raw_exports/
```

Then run:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
python forensic_tools/scripts/build_public_tool_extracts.py --force
python forensic_tools/scripts/build_canonical_normalized_predictions.py --force
python forensic_tools/scripts/validate_public_extract_equivalence.py --force
```

Raw exports, case files, and local staging outputs must not be committed to `main`.

## Raw-export distribution policy

The 31 complete raw export files used in the experiment are not distributed on `main`.

They remain available only through:

- the protected historical branch `archive/pre-commission-cleanup-2026-07-16`;
- the protected annotated tag `snapshot/pre-commission-cleanup-2026-07-16`;
- the exact immutable commit `309a4580537ebc3bb7950f29c090bb2729fc603b`;
- controlled local or authorized research storage.

The historical snapshot supports provenance and audit. Its existence does not grant permission to redistribute third-party images, proprietary exports, or controlled underlying data.

`evaluation/forensic_tools/tool_export_audit.csv` remains on `main` as a historical record of the raw files processed by the frozen pipeline. Paths recorded there identify the original experiment layout and are not expected to resolve on the curated branch.

## Historical preservation

The full pre-cleanup repository state is preserved through:

```text
branch: archive/pre-commission-cleanup-2026-07-16
tag:    snapshot/pre-commission-cleanup-2026-07-16
commit: 309a4580537ebc3bb7950f29c090bb2729fc603b
```

Both branch and tag are protected against updates, deletion, and force pushes. See:

```text
docs/artifact/ARCHIVE_SNAPSHOT.md
```

`main` remains the authoritative curated research artifact.

## Tool-specific documentation

- `forensic_tools/magnet_axiom/README.md`
- `forensic_tools/excire_foto_2025/README.md`
- `forensic_tools/cellebrite_inseyets/README.md`
- `forensic_tools/griffeye/README.md`

## Public audit and result artifacts

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/normalized_predictions.schema.csv
evaluation/forensic_tools/unmatched_predictions.schema.csv
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool>_metrics.csv
results/figures/chapter_5/
```

Commercial-tool results remain distinct from transparent proxy-model results. Their exported labels and bookmarks are observable operational signals, not evidence of internal model behavior.
