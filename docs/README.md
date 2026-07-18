# Documentation

This directory contains the thesis sources, thesis-ready figures, repository assets, and artifact-preservation documentation.

The final English thesis source is:

```text
docs/LatexThesis/
```

The Italian reference version is maintained separately under:

```text
docs/LatexThesis_ITA/
```

The two directories must not be treated as interchangeable during compilation or archival checks. The English source under `docs/LatexThesis/` is the source of truth for the frozen thesis artifact. The Italian tree is a synchronized reference version and must not silently override English thesis content or numerical claims.

---

## Directory Structure

```text
docs/
├── README.md
├── artifact/
│   └── ARCHIVE_SNAPSHOT.md
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

Repository-governance documents currently located at the repository root are scheduled for the final root cleanup. Their content remains authoritative until the move is completed and all links are updated.

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
- keep numerical claims synchronized with frozen CSV/JSON sources;
- treat commercial-tool outputs as observable black-box signals, not internal model evidence.

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

Compilation products such as `.acn`, `.acr`, `.alg`, `.aux`, `.log`, `.toc`, and the local `main.pdf` files are ignored. They may remain in a local working tree without appearing in `git status` after the current `.gitignore` is pulled.

---

## Reporting Sources of Truth

Use the following authoritative sources for thesis tables, figures, and numerical statements:

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

The five thesis-selected XAI cases are documented in the canonical manifest. Their 20 thesis-ready image assets are stored under `docs/LatexThesis/images/`; the larger historical XAI output directory is not distributed on current `main`.

Do not derive final thesis values from screenshots or tool interfaces when a CSV/JSON export, normalized manifest, or validated reporting table is available.

Presentation-layer copies in `docs/LatexThesis/images/` must remain traceable to the corresponding source metrics, manifests, or reporting scripts. Use:

```bash
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

before removing or replacing any duplicated reporting image.

---

## Historical Snapshot

The protected pre-cleanup state is documented in:

```text
docs/artifact/ARCHIVE_SNAPSHOT.md
```

The snapshot supports provenance and audit. It is not the authoritative current source and does not grant redistribution rights for third-party data or proprietary exports.
