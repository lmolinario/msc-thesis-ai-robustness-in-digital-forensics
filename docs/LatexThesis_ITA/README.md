# Italian LaTeX Thesis Reference

This directory contains the Italian reference version of the MSc thesis.

The authoritative frozen thesis source is the English tree:

```text
docs/LatexThesis/
```

The Italian tree is maintained to support review, terminology checks, and internal
comparison. It must not silently override the English source, experimental claims,
metric values, figure provenance, or bibliography state.

## Main Entry Point

```text
docs/LatexThesis_ITA/main.tex
```

A typical local build from this directory is:

```bash
latexmk -pdf main.tex
```

Generated auxiliary files and the local `main.pdf` are ignored by Git.

## Synchronization Rules

When synchronizing the Italian and English versions:

- preserve all numerical values from the frozen CSV/JSON sources;
- preserve labels, figure identifiers, table identifiers, and code paths;
- do not translate filenames, script names, class labels, attack identifiers, or
  repository paths;
- treat the English version as authoritative when wording or structure diverges;
- record any substantive methodological correction in both versions;
- do not introduce new experimental claims only in the Italian reference version.

Chapter 5 numerical values should be checked against:

```text
results/metrics/
evaluation/forensic_tools/normalized_predictions.csv
forensic_tools/public_extracts_validation.json
explainability/manifests/chapter5/thesis_selection.csv
```

## Figures

Thesis-ready figures are stored in:

```text
docs/LatexThesis_ITA/images/
```

Before deleting or replacing a figure copied from the reporting layer, run:

```bash
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

The audit checks references in both thesis trees and compares existing copies by
SHA256.

## Public Repository Safety

Do not commit local paths, temporary PDFs, editor configuration, controlled-access
URLs, proprietary forensic exports, credentials, or case material.
