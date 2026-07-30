# Documentation

This directory contains research-artifact documentation, maintenance records,
repository assets, and the authoritative LaTeX thesis source.

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
│   ├── CHAPTER6_FINALIZATION_REPORT.md
│   └── RELEASE_CHECKLIST.md
├── assets/
│   └── repository_header.png
└── LatexThesis/
    ├── README.md
    ├── main.tex
    ├── packages.sty
    ├── title.tex
    ├── sections/
    ├── methodology/
    ├── images/
    └── tesi.bib
```

## Thesis Authority

The only current thesis source is:

```text
docs/LatexThesis/
```

It controls the final wording, structure, experimental claims, numerical values,
labels, and release status.

## Final Thesis Structure

```text
Chapter 1 - Introduction
Chapter 2 - Background
Chapter 3 - State of the Art
Chapter 4 - Methodology
Chapter 5 - Implementation and Experimental Setup
Chapter 6 - Experimental Results and Operational Robustness Analysis
Chapter 7 - Conclusions, Limitations, and Future Work
Appendix
References
```

The appendix source is included after `\appendix` and is not an eighth main
chapter.

## Historical `chapter5` Names

Several frozen XAI and reporting artifacts retain `chapter5` or `chapter_5` in
their paths, identifiers, and script names. Those names were created before the
final separation of implementation and results into Chapters 5 and 6. They are
preserved for artifact identity and reproducibility; they do **not** indicate that
the final quantitative and qualitative results are reported in Chapter 5.

Examples:

```text
explainability/manifests/chapter5/
explainability/scripts/validate_chapter5_xai_artifacts.py
results/figures/chapter_5/
```

The final results discussion is in `sections/06_results.tex`.

## Writing and Archival Principles

- keep the focus on Digital/Computer Forensics;
- treat attacks and transformations as experimental stressors;
- preserve the human-in-the-loop selection protocol;
- distinguish transparent proxy results from black-box commercial observations;
- synchronize numerical statements with committed CSV/JSON sources;
- describe proxy probability outputs as maximum predicted-class probabilities
  (`Max-P`), not calibrated confidence;
- avoid silent methodological changes;
- keep repository paths, identifiers, labels, and script names exact;
- do not commit local paths, build files, credentials, private links, or case
  material.

## LaTeX Build Products

Local compilation products, including `.acn`, `.acr`, `.alg`, `.aux`, `.log`,
`.toc`, and `main.pdf`, are ignored. The source of truth is the tracked LaTeX,
bibliography, acronym definitions, figures, and supporting research artifacts.

## Reporting Sources of Truth

```text
results/metrics/
results/figures/chapter_5/                 # historical frozen path
evaluation/proxy_models/
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
forensic_tools/public_extracts_validation.json
explainability/manifests/chapter5/thesis_selection.csv
datasets/forensic_evaluation_bundle/metadata/
```

The five selected XAI cases are documented by the canonical manifest. Their
thesis-ready images are retained in `docs/LatexThesis/images/` and discussed in
Chapter 6.

## Validation

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

Do not delete a reporting image solely because a byte-identical thesis copy
exists. Review usage, provenance, and release purpose first.

## Governance

Artifact documentation is under `docs/artifact/`. Audit and release records are
under `docs/maintenance/`. Security handling is defined in
`.github/SECURITY.md`.

## Historical Snapshot

The protected pre-cleanup branch and annotated tag are documented in
`docs/artifact/ARCHIVE_SNAPSHOT.md`. Current `main` remains authoritative.
