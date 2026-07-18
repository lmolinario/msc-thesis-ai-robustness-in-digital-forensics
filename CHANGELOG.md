# Changelog

This file records repository-level states relevant to thesis auditability, reproducibility, minimization, and release management. It does not enumerate every intermediate editing commit.

## v1.0.0-thesis-freeze — Planned Release

Final frozen MSc thesis research artifact supporting:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

### Added

- authoritative thesis source under `docs/LatexThesis/`;
- frozen dataset and split manifests;
- 11,500-item forensic evaluation bundle metadata;
- fold-aware proxy checkpoints and registry;
- adversarial and anti-forensic generation workflows;
- transparent proxy predictions and robustness metrics;
- canonical sanitized commercial-tool prediction table with 69,000 rows;
- four tool-specific sanitized prediction extracts;
- exact public validation of 69,000 decisions and 186 commercial metric rows;
- five-case Chapter 5 Integrated Gradients manifest and twenty thesis-ready XAI assets;
- result, XAI, reporting-asset, and LaTeX-image audit utilities;
- protected pre-cleanup branch and annotated snapshot tag;
- organized artifact documentation under `docs/artifact/`;
- maintenance and release records under `docs/maintenance/`;
- security/data-exposure policy under `.github/SECURITY.md`;
- Linux/Kali and PowerShell audit helpers under `tools/`.

### Final Commercial Perimeter

- Magnet AXIOM / Magnet.AI 10.1.0.48673;
- Excire Foto 2025 4.1.5 with D20, D50, and D80 configurations;
- Cellebrite Inseyets 10.9 / Physical Analyzer 10.9.0.3029;
- Magnet Griffeye 26.2.108 with T3K CORE 1.18.0.

### Changed

- aligned the root README with the final curated repository structure;
- established `evaluation/forensic_tools/normalized_predictions.csv` as the canonical public commercial prediction table;
- documented `500 unique OOD images × 5 folds = 2,500 predictions per architecture`;
- retained a single authoritative LaTeX thesis source;
- updated CI to validate final paths, JSON, Python syntax, canonical prediction SHA256, decision profile, metric count, and documentation guards;
- moved governance documents out of the repository root;
- moved local repository utilities under `tools/`;
- ignored LaTeX auxiliary files and local thesis PDFs.

### Removed or Excluded from Current `main`

- raw and derived image corpora;
- complete commercial-tool raw exports;
- historical progress and milestone working documents;
- bulk historical XAI output directories;
- the redundant Italian LaTeX thesis tree;
- redundant root-level governance documents and audit utilities;
- stale references to excluded tools, private paths, and obsolete source directories.

### Preservation and Access Notes

The pre-cleanup state remains preserved at:

```text
branch: archive/pre-commission-cleanup-2026-07-16
tag:    snapshot/pre-commission-cleanup-2026-07-16
commit: 309a4580537ebc3bb7950f29c090bb2729fc603b
```

Current `main` is authoritative. The public artifact supports structural audit, prediction and metric inspection, canonical-table reconstruction, reporting validation, and thesis-source review. Full raw-data and commercial-tool reruns require controlled-access data and licensed software.

## Current Source-of-Truth Areas

```text
README.md
docs/artifact/
.github/SECURITY.md
docs/LatexThesis/
datasets/final/manifests/
datasets/splits/manifests/
datasets/forensic_evaluation_bundle/metadata/
models/model_registry.json
evaluation/proxy_models/
evaluation/forensic_tools/normalized_predictions.csv
forensic_tools/public_extracts_validation.json
results/metrics/
explainability/manifests/chapter5/thesis_selection.csv
```

## Pre-Freeze Development

Earlier states contained working notes, partial milestones, raw commercial exports, image artifacts or LFS pointers, intermediate reports, and evolving documentation. Those states are retained only for historical provenance and are not the current source of truth.
