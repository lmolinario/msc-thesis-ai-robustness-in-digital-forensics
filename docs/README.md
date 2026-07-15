# Documentation

This directory contains the thesis sources, figures, repository assets, and documentation-support material.

The final English thesis source is:

```text
docs/LatexThesis/
```

The Italian reference version is maintained separately under:

```text
docs/LatexThesis_ITA/
```

The two directories must not be treated as interchangeable during compilation or archival checks. The English source under `docs/LatexThesis/` is the current source of truth for the frozen thesis artifact.

---

## Directory Structure

```text
docs/
├── README.md
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
    ├── main.tex
    ├── packages.sty
    ├── title.tex
    ├── sections/
    ├── images/
    └── tesi.bib
```

Additional audit scripts and generated LaTeX support files may be present within the two thesis directories.

---

## Thesis Structure Reference

The frozen English thesis structure is:

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

The source sequence is defined in:

```text
docs/LatexThesis/main.tex
```

and currently includes:

```text
sections/01_introduction.tex
sections/02_background.tex
sections/03_StateoftheArt.tex
sections/04_methodology.tex
sections/05_experiments.tex
sections/06_conclusions.tex
sections/07_appendix.tex
```

The appendix is introduced after `\appendix`; it is not a seventh main chapter.

---

## Chapter 5 Reporting Order

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

When making archival corrections:

- keep the focus on Digital/Computer Forensics;
- frame adversarial and anti-forensic attacks as experimental stressors;
- avoid turning the work into pure Adversarial Machine Learning optimization;
- keep the human-in-the-loop selection protocol explicit;
- distinguish methodology, experimental results, and critical interpretation;
- preserve the terminology used in the frozen English thesis;
- keep numerical claims synchronized with frozen CSV/JSON sources.

---

## LaTeX Notes

The English thesis uses glossary and acronym entries defined in:

```text
docs/LatexThesis/sections/000_acronyms.tex
```

Bibliography entries are maintained in:

```text
docs/LatexThesis/tesi.bib
```

Use `\texttt{...}` only for technical identifiers such as class names, filenames, folders, scripts, or experimental parameters.

Private editor URLs, temporary collaboration links, local paths, installer links, credentials, and controlled-access dataset URLs must not be committed to this public repository.

---

## Reporting Source of Truth

For thesis tables and figures, prefer the following sources:

```text
results/metrics/
results/figures/chapter_5/
evaluation/proxy_models/
evaluation/forensic_tools/
explainability/outputs/integrated_gradients/
datasets/forensic_evaluation_bundle/metadata/
```

Do not derive final thesis values from screenshots or tool interfaces when a CSV/JSON export or normalized manifest is available.

Presentation-layer copies in `docs/LatexThesis/images/` must remain traceable to the corresponding source metrics, manifests, or reporting scripts.
