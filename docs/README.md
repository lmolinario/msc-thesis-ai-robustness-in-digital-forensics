# Documentation

This directory contains research-artifact documentation, maintenance records, repository assets, and both LaTeX thesis trees.

## Directory Structure

```text
docs/
├── README.md
├── artifact/
│   ├── THESIS_ARTIFACT.md
│   ├── ARTIFACT_EVALUATION.md
│   ├── REPOSITORY_MAP.md
│   ├── DATA_DICTIONARY.md
│   ├── ENVIRONMENT.md
│   ├── REPRODUCIBILITY.md
│   ├── DATA_ACCESS.md
│   └── ARCHIVE_SNAPSHOT.md
├── maintenance/
│   ├── ACADEMIC_REPOSITORY_AUDIT.md
│   └── RELEASE_CHECKLIST.md
├── assets/
│   └── repository_header.png
├── LatexThesis/
│   ├── README.md
│   ├── main.tex
│   ├── packages.sty
│   ├── title.tex
│   ├── sections/
│   ├── methodology/
│   ├── images/
│   └── tesi.bib
└── LatexThesis_ITA/
    ├── README.md
    ├── main.tex
    ├── packages.sty
    ├── title.tex
    ├── sections/
    ├── images/
    └── tesi.bib
```

## Thesis Authority

Authoritative frozen English source:

```text
docs/LatexThesis/
```

Italian reference source:

```text
docs/LatexThesis_ITA/
```

The English source controls final wording, structure, experimental claims, numerical values, labels, and release status. The Italian tree is maintained for reference and review.

## Thesis Structure

```text
Chapter 1 - Introduction
Chapter 2 - Background
Chapter 3 - State of the Art
Chapter 4 - Methodology
Chapter 5 - Experiments and Results
Chapter 6 - Conclusions
Appendix
References
```

The appendix source is included after `\appendix` and is not a seventh main chapter.

## Writing and Archival Principles

- keep the focus on Digital/Computer Forensics;
- treat attacks and transformations as experimental stressors;
- preserve the human-in-the-loop selection protocol;
- distinguish transparent proxy results from black-box commercial observations;
- synchronize all numerical statements with committed CSV/JSON sources;
- avoid introducing methodological changes through silent documentation edits;
- keep repository paths, identifiers, labels, and script names exact;
- do not commit local paths, temporary build files, credentials, private links, or proprietary case material.

## LaTeX Build Products

Local compilation products, including `.acn`, `.acr`, `.alg`, `.aux`, `.log`, `.toc`, and `main.pdf`, are ignored in both thesis trees. The source of truth is the tracked LaTeX, bibliography, acronym definitions, figures, and supporting research artifacts.

## Reporting Sources of Truth

```text
results/metrics/
results/figures/chapter_5/
evaluation/proxy_models/
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
forensic_tools/public_extracts_validation.json
explainability/manifests/chapter5/thesis_selection.csv
datasets/forensic_evaluation_bundle/metadata/
```

The five selected XAI cases are documented by the canonical manifest. Their thesis-ready images are retained in the LaTeX image directories; the larger historical XAI output tree is not distributed on current `main`.

## Validation

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

Do not delete a reporting image merely because a byte-identical LaTeX copy exists. Review usage, source provenance, and release purpose first.

## Governance

Artifact scope and reproducibility documents are under `docs/artifact/`. Ongoing audit and release documents are under `docs/maintenance/`. Security and data-exposure handling are defined in `.github/SECURITY.md`.

## Historical Snapshot

The protected pre-cleanup branch and annotated tag are documented in:

```text
docs/artifact/ARCHIVE_SNAPSHOT.md
```

The snapshot supports provenance only. Current `main` remains authoritative and historical preservation does not grant redistribution rights.
