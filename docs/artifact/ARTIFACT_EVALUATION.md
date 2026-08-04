# Artifact Evaluation Statement

This document defines what can be evaluated, audited, and reproduced from the public MSc thesis research artifact.

## Artifact Type

The repository contains experimental code, frozen manifests, transparent proxy-model outputs, sanitized commercial-tool decisions, metric summaries, XAI case-study material, reporting assets, the authoritative LaTeX thesis source, and governance documentation.

It is not a general-purpose software library, an unrestricted raw-image benchmark, or a reproduction package for proprietary forensic software.

## Claims Supported by the Public Repository

| Claim area | Public support |
|---|---|
| Dataset construction and freezing | `datasets/final/manifests/`, `datasets/splits/manifests/` |
| Clean/OOD separation | clean fold and OOD evaluation manifests |
| Perturbation protocol | numbered attack scripts and `attacks/manifests/` |
| Proxy-model evaluation | `evaluation/proxy_models/`, `models/model_registry.json`, `results/metrics/` |
| Commercial black-box normalization | canonical sanitized prediction table, public extracts, registry, validation report |
| Commercial metric reproduction | 69,000 committed decisions and 186 frozen metric rows |
| XAI qualitative analysis | five-case manifest and thesis-ready image assets |
| Thesis reporting | `docs/LatexThesis/`, reporting manifest, figures, and source metrics |
| Historical provenance | protected branch, protected annotated tag, exact snapshot commit |

## Canonical Commercial Evaluation Evidence

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/normalized_predictions_public_summary.json
forensic_tools/public_extracts_summary.json
forensic_tools/public_extracts_validation.json
results/metrics/forensic_tools_metrics.csv
```

The committed validation report establishes:

```text
69,000 decision rows identical
186 metric rows identical
```

The public prediction table excludes raw-export paths, image hashes, unrelated metadata, device identifiers, and case-management fields.

## Claims Not Supported by the Public Repository Alone

The public repository cannot independently provide:

- unrestricted access to source or derived image corpora;
- commercial forensic-tool licenses;
- proprietary commercial models, thresholds, weights, or training data;
- complete raw commercial exports on current `main`;
- private acquisition credentials or reusable direct-download URLs;
- a guaranteed byte-identical full training environment without a separately frozen lock file.

These limitations are intentional and documented in:

```text
docs/artifact/DATA_ACCESS.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/ENVIRONMENT.md
.github/SECURITY.md
```

## Evaluation Levels

| Level | Description | Status |
|---|---|---:|
| Structural audit | Verify repository layout, scripts, manifests, and documented paths | Public |
| Prediction audit | Inspect proxy predictions and 69,000 commercial decisions | Public |
| Metric audit | Inspect and recompute frozen CSV/JSON metrics | Public |
| Result validation | Run committed read-only validators | Public |
| Thesis-source audit | Inspect LaTeX, bibliography, labels, figures, and tables | Public |
| Partial rerun | Run stages when controlled data and dependencies are available | Controlled |
| Full proxy rerun | Reproduce training and evaluation with original images | Controlled |
| Full commercial rerun | Process the blind bundle in licensed software | Licensed / controlled |
| Commercial AI internals | Inspect proprietary implementation details | Unsupported |

## Reviewer Checklist

A reviewer can inspect:

- root `README.md` for orientation;
- `docs/artifact/THESIS_ARTIFACT.md` for the official scope;
- `docs/artifact/REPOSITORY_MAP.md` for navigation;
- `docs/artifact/DATA_DICTIONARY.md` for schema interpretation;
- `docs/artifact/REPRODUCIBILITY.md` for rerun guidance;
- `docs/artifact/DATA_ACCESS.md` for controlled data access;
- `.github/SECURITY.md` for exposure handling;
- `docs/LatexThesis/` for the authoritative thesis source;
- `results/metrics/` for final quantitative outputs;
- `evaluation/forensic_tools/normalized_predictions.csv` for canonical commercial decisions;
- `forensic_tools/public_extracts_validation.json` for exact equivalence;
- `explainability/manifests/chapter5/thesis_selection.csv` for the final XAI selection.

Recommended read-only checks:

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --force

python explainability/scripts/validate_chapter5_xai_artifacts.py --strict-thesis-text
python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

The exact commit and outcomes of the latest recorded local execution are kept in
`docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md`. This document defines what can
be evaluated; it should not be used as a moving execution log.

## Expected Interpretation

The repository should be interpreted as:

```text
final thesis research artifact with public audit material and controlled-access reproduction boundaries
```



## Frozen Status

Future changes should be limited to documented archival corrections, citation or DOI updates, dependency hardening, security fixes, and release metadata. Substantive experimental changes require a new version rather than silent replacement of the thesis artifact.
