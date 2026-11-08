# English LaTeX Thesis Source

This directory contains the authoritative source of the MSc thesis:

> **Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

## Main Entry Point

```text
docs/LatexThesis/main.tex
```

The build requires pdfLaTeX, Biber, `makeglossaries`/MakeIndex, and the LaTeX
packages loaded by `packages.sty`, including `biblatex`, `glossaries`, and
`fontawesome5`. A complete explicit build from the repository root is:

```bash
cd docs/LatexThesis
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
makeglossaries main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If the final log still requests another pass, repeat pdfLaTeX until references
and page labels are stable. Inspect `main.log` and `main.blg` for errors,
undefined references or citations, missing glyphs, and layout warnings, then
visually inspect the generated PDF. A bare `latexmk -pdf main.tex` must not be
assumed to generate the glossaries without an appropriate local configuration;
no `.latexmkrc` is supplied in this directory.

Generated auxiliary files and local `main.pdf` are ignored. The tracked LaTeX
sources, bibliography, figures, manifests, and numerical source artifacts define
the repository source of truth.

## Current Directory Structure

```text
docs/LatexThesis/
├── README.md
├── main.tex
├── packages.sty
├── title.tex
├── tesi.bib
├── sections/
│   ├── 000_acronyms.tex
│   ├── 0_acknowledgments.tex
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_StateoftheArt.tex
│   ├── 04_methodology.tex
│   ├── 05_implementation.tex
│   ├── 06_results.tex
│   ├── 07_conclusions.tex
│   └── 08_appendix.tex
├── methodology/
└── images/
```

The document contains seven main chapters followed by an appendix and
references. `sections/08_appendix.tex` is included after `\appendix` and must not
be described as an eighth main chapter.

## Final Chapter Map

```text
1  Introduction
2  Background
3  State of the Art
4  Methodology
5  Implementation and Experimental Setup
6  Experimental Results and Operational Robustness Analysis
7  Conclusions, Limitations, and Future Work
```

## Acronyms and Bibliography

Acronym definitions:

```text
sections/000_acronyms.tex
```

Bibliography database:

```text
tesi.bib
```

Citation keys, bibliography metadata, and first-use acronym behavior must remain
synchronized with the final source.

## Figures and Numerical Sources

Thesis-ready images are stored in `images/`. Numerical values and figure content
must be traceable to:

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

`chapter5` and `chapter_5` are historical frozen artifact names created before
the final split between implementation and results. They are preserved for
artifact identity and reproducibility. The five selected XAI cases and their
twenty thesis-ready assets are discussed in Chapter 6 (`sections/06_results.tex`).

Proxy probability values reported in the thesis are maximum predicted-class
probabilities (`Max-P`). They are intra-model diagnostic outputs and must not be
presented as calibrated confidence or forensic certainty.

The controlled classifiers are white-box proxy models: `proxy` describes their
experimental reference role, and `white-box` describes the experimenter's
access, not the information used by every attack. EfficientNet-B0 is the
designated source; ResNet18 and CLIP are receiving architectures for empirical
transfer. One Pixel remains score-based. Generation-time success flags may
refer to pre-serialization tensors; the thesis results use predictions on
reloaded serialized files and a clean-correct ASR denominator.

FAIRLab produces task- and protocol-specific validation profiles, not system
certification, a universal robustness score, or a product ranking. Pooled binary
metrics assign equal weight to the 11 equally sized test conditions, but the
clean, adversarial, and anti-forensic branches have weights `1:5:5`.

Do not manually alter numerical values in presentation-layer figures when a
CSV/JSON source or generation script exists.

## Validation

Run the principal artifact checks from the repository root:

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --force

python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

The result validator checks prediction and metric counts, canonical commercial
SHA-256, OOD accounting, reporting-manifest counts, and metadata-sensitivity
counts. The asset audit scans the authoritative thesis tree and compares existing
copies by SHA-256.

The latest recorded local validation outcomes and the exact commit or explicitly
identified pre-commit working-tree state to which they apply are maintained in
`docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md`.
Regardless of earlier successful checks, the thesis must be compiled and its log
reviewed again at the exact official release commit/tag.

## Archival Editing Rules

After thesis freeze, edits should be limited to:

- typographical and language corrections;
- bibliography or acronym consistency fixes;
- broken-reference or compilation fixes;
- documentation hygiene;
- explicitly documented archival corrections.

Dataset changes, attack regeneration, model retraining, metric replacement, or
new experimental claims require explicit versioning.

## Public Repository Safety

Do not commit private editor URLs, local absolute paths, credentials, license
files, proprietary installers, case material, reusable private download URLs, or
temporary exports containing unnecessary personal or sensitive data.

Governance and local configuration guidance:

```text
docs/artifact/DATA_ACCESS.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/ENVIRONMENT.md
.github/SECURITY.md
.env.example
```
