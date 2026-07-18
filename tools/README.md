# Repository Tools

This directory contains non-destructive local audit helpers. These utilities are not part of the numbered experimental pipeline and must not regenerate frozen datasets, attacks, predictions, or metrics unless a command explicitly says so.

## Kali/Linux helper

Run from the repository root:

```bash
bash tools/tasks.sh status
bash tools/tasks.sh check-json
bash tools/tasks.sh check-python-syntax
bash tools/tasks.sh check-text-guards
bash tools/tasks.sh check-xai
bash tools/tasks.sh check-results
bash tools/tasks.sh check-assets
bash tools/tasks.sh check-latex-images
bash tools/tasks.sh check-thesis-log
bash tools/tasks.sh audit-all
```

The script resolves the repository root relative to its own location.

## PowerShell helper

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

## LaTeX image audit

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex
```

See `tools/latex/README.md` for output and interpretation notes.
