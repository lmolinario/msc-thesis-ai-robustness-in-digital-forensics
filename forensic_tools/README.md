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

## Directory structure

```text
forensic_tools/
├── README.md
├── run_registry.json
├── scripts/
│   ├── build_public_tool_extracts.py
│   └── validate_public_extract_equivalence.py
├── magnet_axiom/
│   ├── README.md
│   ├── raw_exports/
│   └── public_extracts/              # generated locally after validation workflow
├── excire_foto_2025/
│   ├── README.md
│   ├── raw_exports/
│   └── public_extracts/              # generated locally after validation workflow
├── cellebrite_inseyets/
│   ├── README.md
│   ├── raw_exports/
│   └── public_extracts/              # generated locally after validation workflow
└── griffeye/
    ├── README.md
    ├── raw_exports/
    └── public_extracts/              # generated locally after validation workflow
```

The `public_extracts/` directories are created by the sanitization script. They are not yet the basis for removing the original exports.

## Run registry

The consolidated run metadata are recorded in:

```text
forensic_tools/run_registry.json
```

The registry provides, for each normalized configuration:

- tool and version;
- frozen run ID;
- export format;
- observable field;
- positive mapping rule;
- raw, matched, and unmatched row counts;
- normalized tool identifier;
- public metric artifact;
- immutable historical snapshot reference.

## Observable mappings

| Configuration | Observable signal | Positive mapping |
|---|---|---|
| Magnet AXIOM | `Tags` | `Possible weapons` |
| Excire D20/D50/D80 | fixed semantic prompt retrieval | at least one firearm-oriented prompt hit |
| Cellebrite Inseyets | `Classifications` | `Armi`, `Pistola`, or `Fucile` |
| Griffeye / T3K CORE | `Bookmarks` | `CORE/Violence/Firearm` |

These are operational recodings of exported fields, not direct measurements of proprietary internal model probabilities.

## Canonical normalization

The official normalization entry point is:

```bash
python evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py --force
```

Canonical regeneration requires:

- the locally restored blind bundle and metadata;
- the official commercial-tool raw exports;
- all six frozen configurations;
- output coverage of 11,500 bundle items per configuration;
- 69,000 matched normalized decisions in total;
- zero `unknown` normalized decisions.

Prediction-level normalized outputs are generated locally and are not currently distributed on `main`. Public audit and metric artifacts remain under:

```text
evaluation/forensic_tools/
results/metrics/
```

## Sanitized public-extract workflow

The raw exports are **not being removed at this stage**.

The planned transition is deliberately fail-closed:

```text
raw commercial exports
→ local normalized predictions
→ minimized public extracts
→ decision equivalence validation
→ 186-row metric equivalence validation
→ explicit final decision on raw-export retention
```

### 1. Build minimized extracts

After regenerating `evaluation/forensic_tools/normalized_predictions.csv` locally:

```bash
python forensic_tools/scripts/build_public_tool_extracts.py
```

Use `--force` only for an intentional replacement:

```bash
python forensic_tools/scripts/build_public_tool_extracts.py --force
```

The extracts retain only:

- anonymized `bundle_id`;
- experimental condition fields required to recompute metrics;
- the observable tool signal required for audit;
- normalized decision fields.

They exclude local paths, case locations, device names, unrelated EXIF fields, serial numbers, PhotoDNA, case-management fields, and other export metadata unnecessary for the experiment.

### 2. Validate exact equivalence

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py
```

The validator fails if:

- a tool/bundle decision is missing or added;
- any `weapon_detected` value changes;
- sample type, attack family, attack name, or final label changes;
- the six configurations do not each contain 11,500 rows;
- the 69,000 sanitized decisions differ from the local normalized source;
- the recomputed 186 metric rows differ from `results/metrics/forensic_tools_metrics.csv`.

Only a successful validation report can support a later proposal to remove or relocate complete raw exports.

## Current raw-export policy

The original exports remain temporarily tracked on `main` pending sanitized-extract equivalence validation.

This temporary retention supports review of the complete transformation chain:

```text
commercial-tool export
→ normalization
→ bundle matching
→ metric generation
```

No raw export will be removed without:

1. generation of the sanitized extracts;
2. exact decision equivalence;
3. exact metric equivalence;
4. review of the generated files;
5. explicit approval of the final retention policy.

## Historical preservation

The full pre-cleanup repository state is independently preserved through:

```text
branch: archive/pre-commission-cleanup-2026-07-16
tag:    snapshot/pre-commission-cleanup-2026-07-16
commit: 309a4580537ebc3bb7950f29c090bb2729fc603b
```

Both branch and tag are protected against updates, deletion, and force pushes. See:

```text
docs/artifact/ARCHIVE_SNAPSHOT.md
```

The archive supports provenance and recovery, while `main` remains the authoritative curated research artifact.

## Tool-specific documentation

- `forensic_tools/magnet_axiom/README.md`
- `forensic_tools/excire_foto_2025/README.md`
- `forensic_tools/cellebrite_inseyets/README.md`
- `forensic_tools/griffeye/README.md`

## Public audit and result artifacts

```text
evaluation/forensic_tools/tool_export_audit.csv
evaluation/forensic_tools/tool_version_log.csv
evaluation/forensic_tools/normalization_summary.json
evaluation/forensic_tools/normalized_predictions.schema.csv
evaluation/forensic_tools/unmatched_predictions.schema.csv
results/metrics/forensic_tools_metrics.csv
results/metrics/<tool>_metrics.csv
results/figures/chapter_5/
```

Commercial-tool results must remain distinct from transparent proxy-model results. Their exported labels and bookmarks are observable operational signals, not evidence of internal model behavior.
