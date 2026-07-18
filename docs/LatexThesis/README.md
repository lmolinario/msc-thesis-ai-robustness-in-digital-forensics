# English LaTeX Thesis Source

This directory contains the authoritative source of the MSc thesis:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

## Main Entry Point

```text
docs/LatexThesis/main.tex
```

Typical local build:

```bash
cd docs/LatexThesis
latexmk -pdf main.tex
```

The toolchain must complete bibliography, glossary/acronym generation, and repeated LaTeX passes as configured by `main.tex` and `packages.sty`.

Generated auxiliary files and local `main.pdf` are ignored. The tracked LaTeX sources, bibliography, figures, manifests, and numerical source artifacts define the repository source of truth.

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

The document contains six chapters followed by an appendix and references. `sections/07_appendix.tex` is included after `\appendix` and must not be described as a seventh main chapter.

## Acronyms and Bibliography

Acronym definitions:

```text
sections/000_acronyms.tex
```

Bibliography database:

```text
tesi.bib
```

Citation keys, bibliography metadata, and first-use acronym behavior must remain synchronized with the final source.

## Figures and Numerical Sources

Thesis-ready images are stored in `images/`. Numerical values and figure content must be traceable to:

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

The five selected XAI cases and their twenty thesis-ready assets are the public Chapter 5 XAI layer. The larger historical Integrated Gradients output tree is not distributed on current `main`.

Do not manually alter numerical values in presentation-layer figures when a CSV/JSON source or generation script exists.

## Validation

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

The result validator checks prediction and metric counts, canonical commercial SHA256, OOD accounting, reporting-manifest counts, and metadata-sensitivity counts. The asset audit checks the authoritative thesis tree and compares existing copies by SHA256.

## Archival Editing Rules

After thesis freeze, edits should be limited to:

- typographical and language corrections;
- bibliography or acronym consistency fixes;
- broken-reference or compilation fixes;
- documentation hygiene;
- explicitly documented archival corrections.

Dataset changes, attack regeneration, model retraining, metric replacement, or new experimental claims require explicit versioning.

## Public Repository Safety

Do not commit private editor URLs, local absolute paths, credentials, license files, proprietary installers, case material, reusable private download URLs, or temporary exports containing unnecessary personal or sensitive data.

Governance and local configuration guidance:

```text
docs/artifact/DATA_ACCESS.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/ENVIRONMENT.md
.github/SECURITY.md
.env.example
```
