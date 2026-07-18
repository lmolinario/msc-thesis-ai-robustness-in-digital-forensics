# Artifact Evaluation Statement

This document defines what can be evaluated, audited, and reproduced from this repository as an academic research artifact.

---

## Artifact Type

This repository is a thesis-oriented research artifact containing:

- experimental code;
- dataset manifests and split definitions;
- perturbation manifests;
- proxy-model evaluation outputs;
- normalized commercial-tool outputs;
- metric summaries;
- explainability case-study artifacts;
- LaTeX thesis source;
- governance and reproducibility documentation.

It is not a general-purpose software library and not a public raw-image dataset mirror.

---

## Claims Supported by the Repository

The repository supports audit and inspection of the following claims:

| Claim area | Supported by |
|---|---|
| Dataset construction and freezing | `datasets/final/manifests/`, `datasets/splits/manifests/` |
| Clean/OOD separation | `clean_folds_manifest.csv`, `ood_eval_manifest.csv` |
| Perturbation generation design | `attacks/`, `attacks/manifests/`, numbered generation scripts |
| Proxy-model robustness evaluation | `evaluation/proxy_models/`, `results/metrics/` |
| Commercial-tool black-box normalization | `evaluation/forensic_tools/`, `forensic_tools/`, `results/metrics/` |
| XAI qualitative case studies | `explainability/`, `docs/LatexThesis/images/`, Chapter 5 source |
| Thesis reporting | `docs/LatexThesis/`, `results/figures/`, `results/metrics/` |

---

## Claims Not Supported by the Public Repository Alone

The public repository alone does not support unrestricted rerun of every experiment because several components depend on controlled resources.

The following are not publicly redistributable or not fully reproducible without external access:

- raw third-party image sources;
- controlled-access source bundles;
- commercial forensic software;
- proprietary AI models inside commercial forensic tools;
- licensed forensic case formats and proprietary tool databases;
- private acquisition credentials or dataset links.

This limitation is intentional and documented in `DATA_ACCESS.md`, `SECURITY.md`, and `REPRODUCIBILITY.md`.

---

## Evaluation Levels

| Level | Description | Supported |
|---|---|---:|
| Structural audit | Verify repository layout, manifests, scripts, and documented paths | Yes |
| Metric audit | Inspect CSV/JSON metrics, normalized outputs, and thesis-reported results | Yes |
| Thesis-source audit | Inspect LaTeX source, figures, bibliography, and section-level reporting | Yes |
| Code review | Review dataset, perturbation, model, evaluation, XAI, and reporting scripts | Yes |
| Partial rerun | Rerun code stages when controlled-access data and dependencies are available | Controlled |
| Full proxy rerun | Reproduce training/evaluation using original images and compute environment | Controlled |
| Full commercial-tool rerun | Reprocess the blind bundle in licensed forensic tools | Licensed / controlled |
| Commercial AI internals | Inspect proprietary models, thresholds, or training data | No |

---

## Artifact Scope

The artifact supports the thesis as follows:

1. It defines the experimental dataset and split structure.
2. It preserves the perturbation generation protocol.
3. It stores transparent proxy-model evaluation outputs.
4. It normalizes commercial black-box forensic-tool outputs into a common schema.
5. It preserves thesis-ready metric summaries and figures.
6. It documents limitations, access boundaries, and security constraints.

---

## Artifact Evaluation Checklist

A reviewer can check:

- root `README.md` for high-level orientation;
- `THESIS_ARTIFACT.md` for official artifact boundaries;
- `REPOSITORY_MAP.md` for directory-level navigation;
- `DATA_DICTIONARY.md` for CSV/JSON schema interpretation;
- `REPRODUCIBILITY.md` for controlled rerun guidance;
- `DATA_ACCESS.md` for raw data access constraints;
- `SECURITY.md` for secret and proprietary-data handling;
- `docs/LatexThesis/` for the final thesis source;
- `results/metrics/` for final quantitative outputs;
- `evaluation/forensic_tools/` for normalized commercial-tool predictions.

---

## Expected Interpretation

The correct interpretation of this repository is:

```text
final thesis research artifact with controlled-access reproducibility and public audit material
```

The repository should not be interpreted as:

```text
unrestricted public dataset release or complete reproduction package for licensed commercial forensic software
```

---

## Final Artifact Status

The repository is intended to represent the final frozen state of the MSc thesis artifact. Future changes should be limited to:

- documentation corrections;
- citation updates;
- release metadata updates;
- non-substantive reproducibility clarifications;
- post-freeze archival metadata such as DOI or release notes.

Substantive experimental changes should be documented as a new version rather than silently replacing the thesis artifact.
