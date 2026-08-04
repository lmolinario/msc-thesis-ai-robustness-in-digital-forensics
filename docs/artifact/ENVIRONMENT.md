# Environment Notes

This document summarizes execution-environment assumptions for the MSc thesis
research artifact.

The repository is not distributed as a software package. It contains scripts,
manifests, checkpoints, predictions, metrics, reporting assets, and the LaTeX
thesis source.

## Tested Context

The project was developed primarily with:

- Python virtual environments;
- Windows PowerShell for part of the local workflow;
- Linux/Kali for repository and script checks;
- CPU or CUDA-enabled GPU execution depending on the stage;
- licensed commercial forensic tools for black-box export generation;
- a LaTeX toolchain for thesis compilation.

Exact usernames, storage-device paths, credentials, signed URLs, and commercial
license information are intentionally excluded.

## Python Environment

The human-maintained dependency list is:

```text
requirements.txt
```

### Linux/Kali

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA-dependent stages, install PyTorch using a build compatible with the
local GPU driver and CUDA runtime.

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

Controlled artifact access may use:

```text
FAIRLAB_RAW_DATASET_BUNDLE_REQUEST_URL
FAIRLAB_RAW_DATASET_BUNDLE_URL
FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_REQUEST_URL
FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_URL
```

Stable request pages may be documented publicly. Authorized direct-download URLs
must remain local when they are private, signed, temporary, or account-specific.

The authoritative complete-ZIP digests are tracked in:

```text
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

The restoration script reads this file automatically. The
`--expected-sha256` option is available only as an explicit override.

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

Full commercial reruns require licensed software, the controlled blind bundle,
compatible import/export environments, and post-export normalization.

## LaTeX Environment

Authoritative source:

```text
docs/LatexThesis/main.tex
```

Typical local build:

```bash
cd docs/LatexThesis
latexmk -pdf main.tex
```

Common generated files include `.acn`, `.acr`, `.alg`, `.aux`, `.bbl`, `.bcf`,
`.blg`, `.fdb_latexmk`, `.fls`, `.glg`, `.glo`, `.gls`, `.log`, `.out`,
`.run.xml`, `.synctex.gz`, `.toc`, and `main.pdf`. These files are ignored.

## Lightweight Validation on Kali/Linux

```bash
bash tools/tasks.sh check-json
bash tools/tasks.sh check-python-syntax
bash tools/tasks.sh check-text-guards
bash tools/tasks.sh check-xai
bash tools/tasks.sh check-results
bash tools/tasks.sh check-assets
bash tools/tasks.sh check-latex-images
```

Complete helper audit:

```bash
bash tools/tasks.sh audit-all
```

The Linux helper also checks the local thesis log when present. The commercial
public-extract equivalence validator remains a separate explicit release check.

## Lightweight Validation on Windows PowerShell

The PowerShell helper exposes the tasks implemented in `tools/tasks.ps1`:

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

The PowerShell helper does not currently expose the standalone LaTeX-image audit
or the commercial public-extract equivalence validator. Run those explicitly
from the repository root when performing the final artifact checks:

```powershell
python forensic_tools/scripts/validate_public_extract_equivalence.py --source evaluation/forensic_tools/normalized_predictions.csv --metrics results/metrics/forensic_tools_metrics.csv --force
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

PowerShell uses the backtick (`` ` ``), not a trailing backslash, for command
continuation. Keeping these two commands on one line avoids shell-specific
continuation issues.

## Reproducibility Boundary

The public repository supports script and structural inspection, manifest and
metric inspection, reconstruction of the canonical sanitized commercial
prediction table, reporting validation, and thesis-source review.

Controlled raw access supports complete pipeline regeneration. The separately
controlled frozen evaluation archive supports restoration of the exact 11,500
files used as commercial black-box input. Commercial reprocessing still requires
licensed software and a compatible execution environment.

The repository does not independently provide unrestricted images, licensed
forensic software, proprietary AI internals, or a guaranteed byte-identical
operating-system environment.

Related documents:

```text
docs/artifact/REPRODUCIBILITY.md
docs/artifact/DATA_ACCESS.md
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
.github/SECURITY.md
```
