# Thesis Defense Slides

This directory contains the LaTeX sources currently used to develop the MSc thesis defense presentation.

The defense material was developed **after** the experimental thesis artifact was frozen at tag `thesis-freeze-2026-08-04`. It may evolve independently of the frozen scientific artifact, provided that all numerical and methodological statements remain consistent with the thesis and committed results.

## Working sources

The presentation is still under active development. The current tree intentionally retains:

```text
main.tex
main2.tex
main3.tex
```

These files represent active working variants and may coexist until the defense deck is finalized. No single variant should be treated as permanently authoritative while this development phase is ongoing.

## Build

Compile the variant currently being reviewed, for example:

```bash
pdflatex main3.tex
pdflatex main3.tex
```

Generated PDF and auxiliary build products are local artifacts and should not be committed.

## Theme

The presentation uses the included SINTEF Beamer theme files and assets. See `LICENSE.txt` for the theme licensing terms.
