# Documentation

This directory contains thesis documentation, LaTeX sources, figures, and repository-level supporting material.

The final frozen thesis source is:

```text
docs/LatexThesis_ITA/
```

The thesis is written in Italian academic style. The repository documentation remains compatible with a possible later academic-English adaptation, but the submitted thesis source is the Italian frozen version.

---

## Directory Structure

```text
docs/
├── README.md
├── assets/
│   └── repository_header.png
└── LatexThesis_ITA/
    ├── main.tex
    ├── packages.sty
    ├── title.tex
    ├── sections/
    ├── methodology/
    └── tesi.bib
```

---

## Thesis Structure Reference

The frozen thesis structure is:

```text
Chapter 1 - Introduction
Chapter 2 - Background
Chapter 3 - State of the Art
Chapter 4 - Methodology
Chapter 5 - Results and Operational Robustness Analysis
Chapter 6 - Discussion / Legal and Operational Implications
Chapter 7 - Conclusions
```

Chapter 5 follows the stabilized order:

```text
status and dataset overview
clean baseline
OOD behavior
anti-forensic robustness
adversarial robustness
forensic evaluation bundle
commercial forensic tools evaluation
comparative discussion
XAI case studies
operational implications and limitations
```

---

## Writing Principles

When updating thesis text for archival corrections only:

- keep the focus on Digital/Computer Forensics;
- frame adversarial and anti-forensic attacks as experimental stressors;
- avoid turning the work into pure Adversarial Machine Learning optimization;
- keep the human-in-the-loop selection protocol explicit;
- distinguish methodology, results, and critical discussion;
- preserve terminology used in the frozen thesis.

---

## LaTeX Notes

The thesis uses glossary/acronym entries defined in:

```text
docs/LatexThesis_ITA/sections/000_acronyms.tex
```

Bibliography entries are maintained in:

```text
docs/LatexThesis_ITA/tesi.bib
```

Use `\texttt{...}` only for technical identifiers such as class names, file names, folders, scripts, or experimental parameters.

---

## Reporting Source of Truth

For thesis tables and figures, prefer the following sources:

```text
results/metrics/
evaluation/proxy_models/
evaluation/forensic_tools/
explainability/outputs/integrated_gradients/
datasets/forensic_evaluation_bundle/metadata/
```

Do not derive final thesis values from screenshots or tool interfaces when a CSV/JSON export or normalized manifest is available.

---

## Public Repository Note

Private editor URLs, temporary collaboration links, local paths, installer links, and controlled-access dataset URLs must not be documented in this public repository. Use `DATA_ACCESS.md` and `.env.example` for controlled access and local restoration conventions.
