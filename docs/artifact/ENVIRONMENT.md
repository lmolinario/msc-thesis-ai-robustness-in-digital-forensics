# Environment Notes

This document summarizes execution-environment assumptions for the MSc thesis research artifact.

The repository is not distributed as a software package. It contains scripts, manifests, checkpoints, predictions, metrics, reporting assets, and LaTeX thesis sources.

## Tested Context

The project was developed primarily with:

- Python virtual environments;
- Windows PowerShell for the principal local workflow;
- Linux/Kali for additional repository and script checks;
- CPU or CUDA-enabled GPU execution depending on the stage;
- licensed commercial forensic tools for black-box export generation;
- a LaTeX toolchain for thesis compilation.

Exact usernames, storage-device paths, credentials, signed URLs, and commercial license information are intentionally excluded.

## Python Environment

The human-maintained dependency list is:

```text
requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA-dependent stages, install PyTorch using a build compatible with the local GPU driver and CUDA runtime.

## Environment Variables

Use `.env.example` only as a safe variable-name template. Never commit:

```text
.env
private or signed URLs
API keys
credentials
session cookies
commercial license keys
```

Controlled data restoration may use:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

but the value must remain local.

## Compute Expectations

| Stage | Typical requirement |
|---|---|
| Documentation, JSON, CSV, and manifest audit | Low |
| Canonical commercial-table rebuild | Low |
| Reporting and result validators | Low |
| Reporting asset generation | Low to moderate |
| Proxy inference | Moderate |
| Proxy training | Moderate to high |
| Iterative adversarial generation | High |
| Integrated Gradients regeneration | Moderate to high |
| Commercial-tool evaluation | Licensed software and compatible workstation |

## Commercial Tool Environment

Frozen black-box perimeter:

```text
Magnet AXIOM / Magnet.AI 10.1.0.48673
Excire Foto 2025 4.1.5
Cellebrite Inseyets 10.9 / Physical Analyzer 10.9.0.3029
Magnet Griffeye 26.2.108 / T3K CORE 1.18.0
```

Full commercial reruns require:

- licensed software;
- compatible import/export environments;
- the controlled blind bundle;
- tool-specific processing procedures;
- post-export normalization.

The public repository contains sanitized observable outputs, not proprietary internal models.

## LaTeX Environment

Authoritative source:

```text
docs/LatexThesis/main.tex
```

Italian reference source:

```text
docs/LatexThesis_ITA/main.tex
```

Typical local build:

```bash
cd docs/LatexThesis
latexmk -pdf main.tex
```

Common generated files include:

```text
main.acn
main.acr
main.alg
main.aux
main.bbl
main.bcf
main.blg
main.fdb_latexmk
main.fls
main.glg
main.glo
main.gls
main.log
main.out
main.run.xml
main.synctex.gz
main.toc
main.pdf
```

These files are ignored. A final PDF should normally be attached to a versioned release rather than committed to the source tree.

## Lightweight Validation

From the repository root:

```bash
python -m py_compile \
  forensic_tools/scripts/build_canonical_normalized_predictions.py \
  forensic_tools/scripts/validate_public_extract_equivalence.py \
  explainability/scripts/validate_chapter5_xai_artifacts.py \
  results/scripts/23_validate_results_artifacts.py \
  results/scripts/24_audit_reporting_asset_usage.py \
  tools/latex/audit_latex_images_used.py

python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

Windows helper:

```powershell
.\tools\tasks.ps1 audit-all
```

## Thesis Log Check on Windows

After a local compilation:

```powershell
Select-String `
  -Path .\docs\LatexThesis\main.log `
  -Pattern "Undefined references","Citation.*undefined","LaTeX Error","Package glossaries Warning"
```

## Reproducibility Boundary

The public repository supports:

- script and structural inspection;
- committed manifest and metric inspection;
- reconstruction of the canonical sanitized commercial prediction table;
- metric and reporting validation;
- thesis-source review.

It does not independently provide unrestricted images, licensed forensic software, proprietary AI internals, or a guaranteed byte-identical operating-system environment.

Related documents:

```text
docs/artifact/REPRODUCIBILITY.md
docs/artifact/DATA_ACCESS.md
.github/SECURITY.md
```
