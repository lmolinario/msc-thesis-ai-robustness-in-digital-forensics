# Documentation

Overleaf link at `https://www.overleaf.com/project/6a203c229eda2e2462eba7b3`

This directory contains thesis documentation, LaTeX sources, figures, and repository-level supporting material.

The current thesis writing base is:

```text
docs/LatexThesis/
```

The thesis is written in Italian academic style, but the structure and terminology are kept compatible with a later translation into academic English.

---

## Directory Structure

```text
docs/
├── README.md
├── assets/
│   └── repository_header.png
└── LatexThesis/
    ├── main.tex
    ├── packages.sty
    ├── title.tex
    ├── sections/
    └── tesi.bib
```

---

## Thesis Structure Reference

The working thesis structure is:

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
xAI case studies
operational implications and limitations
```

---

## Writing Principles

When updating thesis text:

- keep the focus on Digital/Computer Forensics;
- frame adversarial and anti-forensic attacks as experimental stressors;
- avoid turning the work into pure Adversarial Machine Learning optimization;
- keep the human-in-the-loop selection protocol explicit;
- distinguish methodology, results, and critical discussion;
- keep terminology compatible with later English translation.

---

## LaTeX Notes

The thesis uses glossary/acronym entries defined in:

```text
docs/LatexThesis/sections/000_acronyms.tex
```

Bibliography entries are maintained in:

```text
docs/LatexThesis/tesi.bib
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
