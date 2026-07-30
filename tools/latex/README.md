# LaTeX Image Audit

`audit_latex_images_used.py` scans the thesis from `main.tex`, follows `\input`,
`\include`, and related inclusion commands, resolves `\includegraphics`
references, and produces CSV/JSON audit reports.

Run from the repository root:

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex
```

Use `--help` for all options.

The audit can identify:

- resolved image references;
- missing or ambiguous references;
- files present in image directories but not used by the thesis;
- duplicate image content through SHA-256;
- raster dimensions when Pillow is available.

The generated reports are evidence for manual review. Do not delete an image
solely because it appears unused or duplicated: first verify LaTeX inclusion
order, reporting-layer provenance, and release requirements.

For the frozen results-reporting copies, also run:

```bash
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

The audited path `results/figures/chapter_5/` is a historical frozen artifact
name. The associated results are reported in Chapter 6.
