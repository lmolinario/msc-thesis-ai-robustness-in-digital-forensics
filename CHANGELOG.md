# Changelog

This file records repository-level states relevant to thesis auditability,
reproducibility, minimization, and release management. It does not enumerate
every intermediate editing commit.

## Unreleased — Thesis/Repository Consistency Audit

### Final repository cleanup

- consolidated the defense material while retaining `main.tex`, `main2.tex`, and
  `main3.tex` as active LaTeX working variants until the presentation is finalized;
- removed superseded PowerPoint defense drafts while retaining the discussion
  vademecum and full Git provenance;
- separated post-freeze defense material from the frozen experimental artifact;
- replaced complete third-party literature PDFs in the current release tree with
  a bibliographic-access note pointing to `docs/LatexThesis/tesi.bib`;
- clarified that the manual-review `pending` count concerns unneeded candidate
  pool items, not unresolved labels in the frozen 1,500-image dataset.

### Corrected

- aligned all central documentation with the final seven-chapter thesis
  structure: Chapter 5 implementation, Chapter 6 results, and Chapter 7
  conclusions;
- documented `chapter5` and `chapter_5` paths as historical frozen artifact
  identifiers rather than current chapter assignments;
- replaced calibrated-sounding `confidence` prose with maximum
  predicted-class probability (`Max-P`) terminology while preserving historical
  CSV field names;
- clarified that the canonical sanitized commercial prediction table containing
  69,000 decisions is public, while complete raw commercial exports remain
  excluded;
- removed a personal Windows absolute path from the committed prepared-dataset
  summary;
- hardened the prepared-dataset generator so future summaries emit portable,
  repository-relative paths;
- expanded CI to validate the prepared summary, detect local-path leakage, and
  reject obsolete thesis file and chapter references;
- scoped the Chapter 6 finalization report to its exact historical validation
  commit rather than implying that later commits had been compiled and tested;
- clarified the OOD denominator, derived-variant dependence, attack-target
  boundary, and non-universality of commercial-tool comparisons;
- recorded the successful local execution of the principal final artifact
  validators at commit `1018f307038f1a27d380df18c2725825cd8ab6b9`, including
  exact commercial equivalence, XAI, result, reporting-asset, and LaTeX-image
  checks;
- separated the reusable release procedure from the live audit-status record and
  retained final LaTeX compilation/log inspection as a release-commit check;
- aligned validation entry points across the root documentation and the
  `docs/` artifact documentation, including explicit Windows PowerShell guidance.

## v1.0.0-thesis-freeze — Planned Release

Final frozen MSc thesis research artifact supporting:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

### Added

- authoritative thesis source under `docs/LatexThesis/`;
- frozen dataset and split manifests;
- 11,500-item forensic evaluation bundle metadata;
- controlled restoration of the exact 11,500-file commercial black-box input;
- authoritative SHA-256 digests for the raw and frozen controlled ZIP archives;
- automatic complete-ZIP verification and frozen per-file verification in the
  step-00 restoration script;
- fold-aware proxy checkpoints and registry;
- adversarial and anti-forensic generation workflows;
- transparent proxy predictions and robustness metrics;
- canonical sanitized commercial-tool prediction table with 69,000 rows;
- four tool-specific sanitized prediction extracts;
- exact public validation of 69,000 decisions and 186 commercial metric rows;
- five-case Integrated Gradients manifest and twenty thesis-ready XAI assets,
  discussed in Chapter 6;
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
- established `evaluation/forensic_tools/normalized_predictions.csv` as the
  canonical public commercial prediction table;
- documented `500 unique OOD images × 5 folds = 2,500 predictions per
  architecture`;
- distinguished full pipeline regeneration from exact frozen black-box input
  restoration;
- centralized controlled archive digests in
  `docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256`;
- retained a single authoritative LaTeX thesis source;
- updated CI to validate final paths, JSON, Python syntax, the controlled
  checksum registry, canonical prediction SHA-256, decision profile, metric
  count, and documentation guards;
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
- stale references to excluded tools, private paths, and obsolete source
  directories.

### Preservation and Access Notes

The pre-cleanup state remains preserved at:

```text
branch: archive/pre-commission-cleanup-2026-07-16
tag:    snapshot/pre-commission-cleanup-2026-07-16
commit: 309a4580537ebc3bb7950f29c090bb2729fc603b
```

Current `main` is authoritative. The public artifact supports structural audit,
prediction and metric inspection, canonical-table reconstruction, reporting
validation, and thesis-source review. Full pipeline regeneration requires the
controlled raw archive. Exact restoration of the original black-box input uses
the separately controlled frozen 11,500-file archive. Commercial-tool
reprocessing additionally requires licensed software.

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

The `chapter5` path above is a historical frozen identifier. The corresponding
results are reported in Chapter 6.

## Pre-Freeze Development

Earlier states contained working notes, partial milestones, raw commercial
exports, image artifacts or LFS pointers, intermediate reports, and evolving
documentation. Those states are retained only for historical provenance and are
not the current source of truth.
