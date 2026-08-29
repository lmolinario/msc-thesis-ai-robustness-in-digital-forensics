# Thesis Defense Slides

This directory contains the LaTeX source used to develop the MSc thesis defense
presentation.

The defense material was developed **after** the experimental thesis artifact
was frozen at tag `thesis-freeze-2026-08-04`. It may evolve independently of the
frozen scientific artifact, provided that all numerical and methodological
statements remain consistent with the thesis and committed results.

## Authoritative Source

The only documented and authoritative presentation entry point is:

```text
main.tex
```

Intermediate drafting alternatives are temporary personal working copies, not
scientific sources of truth or release artifacts. Before the presentation is
finalized, the selected version must be consolidated into `main.tex` and any
superseded alternatives removed.

## Build

Run from this directory:

```bash
latexmk -pdf main.tex
```

The included SINTEF Beamer theme requires a compatible LaTeX installation with
Beamer, TikZ, and the Caladea package (`caladea.sty`). Generated PDF and
auxiliary build products are local artifacts and should not be committed.

## Theme

The presentation uses the included SINTEF Beamer theme files and assets. See
`LICENSE.txt` for the theme licensing terms.

## Release Boundary

Defense slides and discussion-preparation material are maintained only as a
working backup for the author. They are excluded from the official MSc thesis
research-artifact release and do not redefine its scope, claims, or validation
status.
