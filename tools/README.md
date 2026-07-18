# Repository Tools

This directory contains non-destructive local audit helpers. These utilities are not
part of the numbered experimental pipeline and must not regenerate frozen datasets,
attacks, predictions, or metrics unless a command explicitly says so.

## PowerShell helper

Run from any location:

```powershell
.\tools\tasks.ps1 status
.\tools\tasks.ps1 check-json
.\tools\tasks.ps1 check-python-syntax
.\tools\tasks.ps1 check-text-guards
.\tools\tasks.ps1 check-results
.\tools\tasks.ps1 check-xai
.\tools\tasks.ps1 check-assets
.\tools\tasks.ps1 check-thesis-log
.\tools\tasks.ps1 audit-all
```

The script resolves the repository root relative to its own location.

## LaTeX image audit

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex
```

See `tools/latex/README.md` for output and interpretation notes.
