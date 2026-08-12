# Thesis Defense Slides

This directory contains the LaTeX source for the MSc thesis defense presentation.

The defense material was developed **after** the experimental thesis artifact was frozen at tag `thesis-freeze-2026-08-04`. It may evolve independently of the frozen scientific artifact, provided that all numerical and methodological statements remain consistent with the thesis and committed results.

## Authoritative source

Compile only:

```text
main.tex
```

Historical working variants are intentionally not retained in the current tree; Git history preserves their provenance.

## Build

From this directory, a standard Beamer build can be produced with a compatible LaTeX installation, for example:

```bash
pdflatex main.tex
pdflatex main.tex
```

Generated PDF and auxiliary build products are local artifacts and should not be committed.

## Theme

The presentation uses the included SINTEF Beamer theme files and assets. See `LICENSE.txt` for the theme licensing terms.
