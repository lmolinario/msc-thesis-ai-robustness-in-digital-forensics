# Release Checklist

Complete this checklist before creating the official thesis-artifact release.

This file is a **reusable release procedure**, not a live record of completed
work. The current audit state, latest successful validator execution, and exact
validated commit are recorded in:

```text
docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md
```

Recommended tag:

```text
v1.0.0-thesis-freeze
```

Recommended title:

```text
MSc Thesis Research Artifact — Final Frozen Version
```

## 1. Synchronize the Repository

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git status -sb
git log --oneline origin/main..HEAD
```

Expected:

```text
## main...origin/main
```

## 2. Verify the Final Layout

Expected root:

```text
.github/
attacks/
datasets/
docs/
evaluation/
explainability/
forensic_tools/
models/
results/
tools/
.env.example
.gitattributes
.gitignore
CHANGELOG.md
CITATION.cff
LICENSE
README.md
requirements.txt
```

The obsolete Italian thesis tree and old root-level governance files must not
reappear.

Defense working material under `docs/LatexSlides/` and `../presentation/` may remain
available in `main` as an author backup, but it is outside the scientific release
perimeter and must not be included among the official release assets.

## 3. Verify the Final Thesis Map

The authoritative source must contain:

```text
Chapter 1 - Introduction
Chapter 2 - Background
Chapter 3 - State of the Art
Chapter 4 - Methodology
Chapter 5 - Implementation and Experimental Setup
Chapter 6 - Experimental Results and Operational Robustness Analysis
Chapter 7 - Conclusions, Limitations, and Future Work
Appendix
```

Paths containing `chapter5` or `chapter_5` are allowed only where they are
historical frozen artifact identifiers. Documentation must explain this boundary
and must not describe the final results as belonging to Chapter 5.

## 4. Run the Repository Audit Helper

### Kali/Linux

```bash
bash tools/tasks.sh status
bash tools/tasks.sh check-json
bash tools/tasks.sh check-python-syntax
bash tools/tasks.sh check-text-guards
bash tools/tasks.sh check-xai
bash tools/tasks.sh check-results
bash tools/tasks.sh check-assets
bash tools/tasks.sh check-latex-images
bash tools/tasks.sh audit-all
```

### Windows PowerShell

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

The PowerShell helper does not currently expose `check-latex-images`; Section 7
therefore remains an explicit cross-platform release check. Neither helper
replaces the explicit commercial equivalence check in Section 5.

## 5. Validate Commercial Predictions

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --report forensic_tools/public_extracts_validation.json \
  --force
```

PowerShell one-line equivalent:

```powershell
python forensic_tools/scripts/validate_public_extract_equivalence.py --source evaluation/forensic_tools/normalized_predictions.csv --metrics results/metrics/forensic_tools_metrics.csv --report forensic_tools/public_extracts_validation.json --force
```

Expected:

```text
69,000 identical decisions
186 identical metric rows
```

Confirm that the canonical CSV, public summary, and equivalence report record the
same SHA-256.

## 6. Validate XAI and Results

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py \
  --strict-thesis-text

python results/scripts/23_validate_results_artifacts.py

python results/scripts/24_audit_reporting_asset_usage.py \
  --strict \
  --report results/reporting_asset_usage_summary.json
```

PowerShell equivalents may be run on single lines. The `chapter5` validator and
manifest names are historical; their authoritative thesis target is
`docs/LatexThesis/sections/06_results.tex`.

Review the ignored local report before removing or replacing any reporting
asset. Unreferenced generated reporting assets are not automatically errors;
missing outputs or mismatched thesis copies are the release-blocking conditions
reported by the strict audit.

## 7. Audit LaTeX Images

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex
```

PowerShell one-line equivalent:

```powershell
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

Review missing, ambiguous, unused, and duplicate image reports.

## 8. Compile the Thesis

```bash
cd docs/LatexThesis
latexmk -pdf main.tex
cd ../..
```

Check:

- bibliography entries resolve;
- cross-references resolve;
- acronym/glossary warnings are understood;
- no LaTeX error remains;
- Chapter 5 implementation figures and tables render correctly;
- Chapter 6 result figures, tables, metrics, and XAI values render correctly;
- Chapter 7 conclusions remain consistent with Chapter 6;
- the PDF reflects the canonical metrics and Max-P terminology.

Log check:

```bash
grep -En 'Undefined references|Citation.*undefined|LaTeX Error|Package glossaries Warning' \
  docs/LatexThesis/main.log || true
```

On Windows PowerShell, the repository helper can inspect an existing log:

```powershell
.\tools\tasks.ps1 check-thesis-log
```

The local `main.pdf` and auxiliary files are ignored.

## 9. Documentation Check

Verify:

```text
README.md
docs/README.md
docs/LatexSlides/README.md
docs/LatexThesis/README.md
docs/artifact/THESIS_ARTIFACT.md
docs/artifact/ARTIFACT_EVALUATION.md
docs/artifact/REPOSITORY_MAP.md
docs/artifact/DATA_DICTIONARY.md
docs/artifact/ENVIRONMENT.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/DATA_ACCESS.md
docs/artifact/ARCHIVE_SNAPSHOT.md
.github/SECURITY.md
docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md
docs/maintenance/CHAPTER6_FINALIZATION_REPORT.md
docs/maintenance/RELEASE_CHECKLIST.md
CHANGELOG.md
CITATION.cff
```

The historical `CHAPTER6_FINALIZATION_REPORT.md` must remain scoped to its
recorded commit. Do not rewrite it to imply validation of later repository
states.

## 10. Data and Security Check

Confirm that current `main` does not contain:

- `.env` or credential files;
- API tokens, passwords, or commercial license keys;
- reusable private or signed download URLs;
- excluded image corpora;
- complete raw commercial exports;
- proprietary case databases or evidence material;
- unnecessary local absolute paths;
- LaTeX compilation products.

Use `gitleaks` or an equivalent secret scanner where available.

## 11. Numerical and Terminological Check

Confirm the frozen profile:

```text
source images                         1,500
binary subset                         1,000
unique OOD images                       500
forensic evaluation bundle           11,500
commercial configurations                 6
commercial decisions                 69,000
commercial metric rows                  186
proxy prediction rows                40,500
proxy OOD rows                        7,500
XAI cases                                  5
XAI thesis assets                         20
```

Confirm that:

- OOD is not described as a supervised third class;
- 2,500 OOD predictions per architecture are not described as 2,500 distinct
  images;
- the four model-dependent attacks are distinguished from model-agnostic Color
  Shift and anti-forensic transformations;
- historical `confidence*` fields are described as Max-P outputs;
- no direct robustness claim is inferred from transferred attacks alone;
- commercial metrics are tied to frozen versions, configurations, observable
  signals, and normalization rules.

## 12. Historical Snapshot Check

Both protected references must still resolve to:

```text
309a4580537ebc3bb7950f29c090bb2729fc603b
```

References:

```text
archive/pre-commission-cleanup-2026-07-16
snapshot/pre-commission-cleanup-2026-07-16
```

Do not move or modify them.

## 13. Release Assets

Preferred assets:

```text
thesis-final.pdf
artifact-checksums.sha256
repository-audit-summary.md
```

Do not attach defense slides, slide PDFs, or discussion-preparation files to the
official research-artifact release. Their presence as working material in the
repository does not make them part of the release evidence package.

Do not add the generated PDF to the Git tree solely for distribution. Attach it
to the release.

Example checksum:

```bash
sha256sum docs/LatexThesis/main.pdf
```

## 14. DOI Archival

After GitHub release creation, archive through Zenodo or an institutional
repository. Then update `CITATION.cff`, add the DOI badge, update `CHANGELOG.md`,
and record the DOI in the academic audit document.

## 15. Post-Release Rule

Allowed maintenance includes citation/DOI corrections, documentation fixes,
release clarifications, security corrections, and non-substantive archival
improvements. Substantive experimental changes require a new versioned release.
