# Release Checklist

Complete this checklist before creating the official thesis-artifact release.

Recommended tag:

```text
v1.0.0-thesis-freeze
```

Recommended title:

```text
MSc Thesis Research Artifact — Final Frozen Version
```

## 1. Synchronize the Local Repository

```powershell
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

No local commit should be ahead of `origin/main` unless it is intentionally being prepared for the release.

## 2. Verify the Final Root Layout

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

The old governance files and audit utilities must not reappear in the root.

## 3. Run Local Audit Helpers

```powershell
.\tools\tasks.ps1 status
.\tools\tasks.ps1 check-json
.\tools\tasks.ps1 check-python-syntax
.\tools\tasks.ps1 check-text-guards
```

Full helper:

```powershell
.\tools\tasks.ps1 audit-all
```

## 4. Validate Canonical Commercial Predictions

```powershell
python .\forensic_tools\scripts\validate_public_extract_equivalence.py `
  --source .\evaluation\forensic_tools\normalized_predictions.csv `
  --metrics .\results\metrics\forensic_tools_metrics.csv `
  --report .\forensic_tools\public_extracts_validation.json `
  --force
```

Expected:

```text
69,000 identical decisions
186 identical metric rows
```

Confirm:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
forensic_tools/public_extracts_validation.json
```

all report the same canonical CSV SHA256.

## 5. Validate XAI and Results

```powershell
python .\explainability\scripts\validate_chapter5_xai_artifacts.py `
  --strict-thesis-text

python .\results\scripts\23_validate_results_artifacts.py

python .\results\scripts\24_audit_reporting_asset_usage.py `
  --strict `
  --report .\results\reporting_asset_usage_summary.json
```

Review the ignored local asset-audit report before removing or replacing any duplicate figure.

## 6. Audit LaTeX Images

```powershell
python .\tools\latex\audit_latex_images_used.py `
  --main .\docs\LatexThesis\main.tex
```

Review missing, ambiguous, unused, and duplicate image reports.

## 7. Compile the English Thesis

```powershell
Set-Location .\docs\LatexThesis
latexmk -pdf main.tex
Set-Location ..\..
```

Check:

- bibliography entries resolve;
- cross-references resolve;
- acronym/glossary warnings are understood;
- no LaTeX error remains;
- Chapter 5 figures and tables render correctly;
- the PDF reflects the canonical metrics and XAI confidence values.

Log check:

```powershell
Select-String `
  -Path .\docs\LatexThesis\main.log `
  -Pattern "Undefined references","Citation.*undefined","LaTeX Error","Package glossaries Warning"
```

The local `main.pdf` and auxiliary files are ignored.

## 8. Compile the Italian Reference Version

```powershell
Set-Location .\docs\LatexThesis_ITA
latexmk -pdf main.tex
Set-Location ..\..
```

Verify that numerical values, labels, identifiers, and figure references remain synchronized with the English source. The English thesis remains authoritative.

## 9. Documentation Check

Verify:

```text
README.md
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
docs/maintenance/RELEASE_CHECKLIST.md
CHANGELOG.md
CITATION.cff
```

## 10. Data and Security Check

Confirm that current `main` does not contain:

- `.env` or credential files;
- API tokens, passwords, or commercial license keys;
- reusable private or signed download URLs;
- image corpora excluded by policy;
- complete raw commercial exports;
- proprietary case databases or evidence material;
- unnecessary local absolute paths;
- LaTeX compilation products.

Run a secret scanner such as `gitleaks` where available.

## 11. Historical Snapshot Check

Verify that both references still resolve exactly to:

```text
309a4580537ebc3bb7950f29c090bb2729fc603b
```

References:

```text
archive/pre-commission-cleanup-2026-07-16
snapshot/pre-commission-cleanup-2026-07-16
```

Do not move or modify them.

## 12. Release Assets

Preferred release assets:

```text
thesis-final.pdf
artifact-checksums.sha256
repository-audit-summary.md
```

Do not add the generated PDF to the Git tree solely for distribution. Attach it to the release.

Generate checksums, for example:

```powershell
Get-FileHash .\docs\LatexThesis\main.pdf -Algorithm SHA256
```

## 13. GitHub Release

Recommended release notes:

```text
Final frozen research artifact supporting the MSc thesis
"Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks".

The release includes the final thesis source, numbered scripts, frozen manifests,
proxy checkpoints and predictions, sanitized commercial-tool decisions, exact metric
equivalence records, XAI case-study material, reporting assets, and governance documentation.

Raw images, controlled-access datasets, complete proprietary exports, licensed forensic
software, and commercial AI internals are not redistributed.
```

## 14. DOI Archival

After GitHub release creation, archive through Zenodo or an institutional repository. After DOI assignment:

1. update `CITATION.cff`;
2. add the DOI badge to `README.md`;
3. update `CHANGELOG.md`;
4. record the release date and DOI in the academic audit document.

## 15. Post-Release Rule

Allowed maintenance:

- citation and DOI corrections;
- documentation typo fixes;
- release-note clarifications;
- security corrections;
- non-substantive archival improvements.

Substantive experimental changes require a new versioned release.
