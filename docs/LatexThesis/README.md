# English LaTeX Thesis Source

This directory contains the current English source of the MSc thesis:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

It is the thesis source of truth for the frozen research artifact. The separate Italian reference version is maintained under:

```text
docs/LatexThesis_ITA/
```

Private editor or collaboration URLs are intentionally not documented in the public repository.

---

## Main Entry Point

Compile the thesis from:

```text
docs/LatexThesis/main.tex
```

From this directory, a typical local build is:

```bash
latexmk -pdf main.tex
```

The toolchain must support the packages and bibliography workflow configured in `packages.sty` and `main.tex`. When using a manual build sequence, ensure that bibliography, glossary/acronym generation, and repeated LaTeX passes are completed.

Generated auxiliary files and local editor configuration should not be committed unless they are explicitly required by the frozen artifact.

---

## Directory Structure

```text
docs/LatexThesis/
├── README.md
├── main.tex
├── packages.sty
├── title.tex
├── tesi.bib
├── sections/
│   ├── 000_acronyms.tex
│   ├── 0_acknowledgment.tex
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_StateoftheArt.tex
│   ├── 04_methodology.tex
│   ├── 05_experiments.tex
│   ├── 06_conclusions.tex
│   └── 07_appendix.tex
├── methodology/
└── images/
```

Audit scripts and generated support files may also be present in the directory.

---

## Thesis Structure

The main document includes six chapters followed by an appendix and references:

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

The appendix source is:

```text
sections/07_appendix.tex
```

It is included after `\appendix` in `main.tex` and must not be described as a seventh main chapter.

---

## Acronyms and Glossary

Acronym definitions are maintained in:

```text
sections/000_acronyms.tex
```

The front matter prints the acronym list, then resets acronym usage before the thesis body so that first occurrences in the main chapters are expanded according to the configured formatting rules.

---

## Bibliography

The bibliography database is:

```text
tesi.bib
```

Bibliographic metadata should be checked with the repository audit utilities before archival release. Citation keys used by the chapter sources must remain synchronized with `tesi.bib`.

---

## Figures and Reporting Assets

Thesis-ready images are stored in:

```text
images/
```

Numerical figures and tables should be generated or verified from the frozen source artifacts, primarily:

```text
results/metrics/
results/figures/chapter_5/
evaluation/proxy_models/
evaluation/forensic_tools/
explainability/outputs/integrated_gradients/
datasets/forensic_evaluation_bundle/metadata/
```

Do not manually alter numerical values in presentation-layer figures when a CSV/JSON source or generation script is available.

---

## Archival Editing Rules

After thesis freeze, edits should be limited to:

- typographical and language corrections;
- bibliography or acronym consistency fixes;
- broken-reference or compilation fixes;
- documentation hygiene;
- clearly documented archival corrections.

Methodological changes, dataset changes, attack regeneration, model retraining, or metric replacement require explicit versioning and must not be introduced as silent documentation edits.

---

## Public Repository Safety

Do not commit:

- private editor or collaboration URLs;
- local absolute paths;
- credentials, tokens, or license files;
- proprietary forensic-tool installers or case files;
- controlled-access dataset URLs;
- temporary exports containing unnecessary personal or sensitive data.

Use the repository-level `DATA_ACCESS.md`, `SECURITY.md`, `.env.example`, and reproducibility documentation for controlled local configuration.
