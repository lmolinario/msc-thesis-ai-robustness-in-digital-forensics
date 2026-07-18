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

The obsolete Italian thesis tree and old root-level governance files must not reappear.

## 3. Run the Kali/Linux Audit Helper

```bash
bash tools/tasks.sh status
bash tools/tasks.sh check-json
bash tools/tasks.sh check-python-syntax
bash tools/tasks.sh check-text-guards
bash tools/tasks.sh check-xai
bash tools/tasks.sh check-results
bash tools/tasks.sh check-assets
bash tools/tasks.sh check-latex-images
```

Complete audit:

```bash
bash tools/tasks.sh audit-all
```

## 4. Validate Commercial Predictions

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --report forensic_tools/public_extracts_validation.json \
  --force
```

Expected:

```text
69,000 identical decisions
186 identical metric rows
```

Confirm that the canonical CSV, public summary, and equivalence report record the same SHA256.

## 5. Validate XAI and Results

```bash
python explainability/scripts/validate_chapter5_xai_artifacts.py \
  --strict-thesis-text

python results/scripts/23_validate_results_artifacts.py

python results/scripts/24_audit_reporting_asset_usage.py \
  --strict \
  --report results/reporting_asset_usage_summary.json
```

Review the ignored local report before removing or replacing any reporting asset.

## 6. Audit LaTeX Images

```bash
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex
```

Review missing, ambiguous, unused, and duplicate image reports.

## 7. Compile the Thesis

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
- Chapter 5 figures and tables render correctly;
- the PDF reflects the canonical metrics and XAI values.

Log check:

```bash
grep -En 'Undefined references|Citation.*undefined|LaTeX Error|Package glossaries Warning' \
  docs/LatexThesis/main.log || true
```

The local `main.pdf` and auxiliary files are ignored.

## 8. Documentation Check

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

## 9. Data and Security Check

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

## 10. Historical Snapshot Check

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

## 11. Release Assets

Preferred assets:

```text
thesis-final.pdf
artifact-checksums.sha256
repository-audit-summary.md
```

Do not add the generated PDF to the Git tree solely for distribution. Attach it to the release.

Example checksum:

```bash
sha256sum docs/LatexThesis/main.pdf
```

## 12. DOI Archival

After GitHub release creation, archive through Zenodo or an institutional repository. Then update `CITATION.cff`, add the DOI badge, update `CHANGELOG.md`, and record the DOI in the academic audit document.

## 13. Post-Release Rule

Allowed maintenance includes citation/DOI corrections, documentation fixes, release clarifications, security corrections, and non-substantive archival improvements. Substantive experimental changes require a new versioned release.
