# Chapter 6 Finalization Report

> **Historical validation record.** This report applies only to commit
> `14246371e4551726e02a7e23c5fb76b78591148d`. The thesis and repository were
> modified after that commit. Therefore, the success statements below must not be
> interpreted as validation of the current branch tip. A new complete validation
> and LaTeX build are required before the official thesis-artifact release.

## Applied Changes at the Recorded Commit

- Replaced the embedded-metadata sensitivity subsection.
- Replaced the comparative operational robustness analysis.
- Replaced the explainability and representative-failure-case section.
- Replaced the operational interpretation and limitations section.
- Updated the XAI scripts for the results chapter and Max-P terminology.

## Validation Outcomes at Commit `14246371e455...`

- Repository audits: success.
- Complete LaTeX compilation: success.
- LaTeX log validation: success.
- Validation source commit:
  `14246371e4551726e02a7e23c5fb76b78591148d`.

These results establish only that the repository state at that exact commit
passed the recorded checks.

## Recorded Frozen Counts

```text
canonical commercial decisions        69,000
commercial metric rows                    186
proxy prediction rows                  40,500
unique OOD images                         500
OOD predictions per architecture        2,500
reporting-manifest rows                    41
unique reporting asset IDs                 24
thesis XAI cases                             5
thesis XAI assets                           20
```

The reporting manifest and paths containing `chapter5` or `chapter_5` are
historical frozen identifiers. The final quantitative and qualitative results are
reported in Chapter 6.

## Recorded Audit Summary

At the validation commit, the audit reported:

- JSON and Python syntax checks passed;
- repository layout and text guards passed;
- XAI public-artifact validation passed;
- result validation passed;
- 500 unique OOD images × 5 fold-specific checkpoints = 2,500 predictions per
  architecture;
- 41 reporting-manifest rows and 24 unique asset identifiers;
- no missing reporting outputs;
- no mismatched existing thesis copies;
- 21 LaTeX image references resolved and no missing references.

The historical audit output referred to a “Chapter 5 manifest” and “Chapter 5
reporting assets”. Those phrases described the frozen artifact names, not the
final thesis chapter allocation.

## Current Release Requirement

Before release, rerun from the current branch tip:

```bash
bash tools/tasks.sh audit-all

python explainability/scripts/validate_chapter5_xai_artifacts.py \
  --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py \
  --main docs/LatexThesis/main.tex

cd docs/LatexThesis
latexmk -pdf main.tex
```

Then inspect the current LaTeX log and record the exact validated commit SHA in a
new release-validation report. This historical file must not be updated merely
to imply that later commits were tested.
