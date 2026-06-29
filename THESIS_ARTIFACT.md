# Thesis Research Artifact

## Title

**Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks**

## Academic Context

| Field | Value |
|---|---|
| Degree | MSc / Master's Degree |
| Programme | Computer Engineering, Cybersecurity and Artificial Intelligence |
| Institution | University of Cagliari |
| Author | Lello Molinario |
| Supervisor | Davide Maiorca |
| Repository role | Final frozen research artifact supporting the MSc thesis |

---

## Artifact Purpose

This repository preserves the experimental and documentary artifact supporting the thesis.

The thesis evaluates the operational robustness of AI-based image-classification and media-triage systems in Digital/Computer Forensics when exposed to:

- clean in-distribution inputs;
- out-of-distribution images;
- adversarial perturbations;
- anti-forensic image transformations;
- observable black-box outputs from commercial forensic tools.

The repository is designed to support auditability, traceability, and controlled reproducibility. It is not intended to be a general-purpose software package or a redistributable dataset mirror.

---

## What This Repository Contains

The repository contains:

- source code for dataset preparation, split generation, perturbation generation, proxy-model training, evaluation, XAI generation, and reporting;
- frozen dataset manifests and split manifests;
- perturbation manifests and metadata;
- proxy-model evaluation outputs and metrics;
- normalized commercial forensic-tool outputs and metric summaries;
- Integrated Gradients case-study artifacts for transparent proxy models;
- LaTeX thesis source;
- governance documents for data access, security, reproducibility, and repository audit.

---

## What This Repository Does Not Contain

The repository does not intentionally redistribute:

- raw third-party image datasets;
- controlled-access source collections;
- licensed forensic software;
- proprietary forensic case files;
- commercial tool databases;
- secrets, credentials, access tokens, or private dataset URLs;
- evidence material or operational investigative data.

Some experiments can therefore be structurally audited from the public repository, while full end-to-end reruns require controlled-access data and, for commercial-tool evaluation, licensed software environments.

---

## Official Source of Truth

The official thesis source is:

```text
docs/LatexThesis/
```

The official dataset manifests are:

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
```

The official forensic evaluation bundle metadata are:

```text
datasets/forensic_evaluation_bundle/metadata/
```

The official metric and reporting outputs are:

```text
results/metrics/
results/figures/
```

The official commercial-tool normalization outputs are:

```text
evaluation/forensic_tools/
```

---

## Final Commercial-Tool Perimeter

The final black-box commercial-tool perimeter is:

| Tool | Version / module | Role |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Commercial forensic AI categorization |
| Excire Foto 2025 | 4.1.5 | Standalone AI-assisted semantic image retrieval |
| Cellebrite Inseyets | 10.9 | Commercial black-box AI-assisted media analysis |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE v1.18.0 | Commercial forensic media triage and semantic bookmarking |

The internal models, thresholds, training data, and proprietary decision logic of these tools are not inspected. Only observable exported outputs are normalized and evaluated.

---

## Reproducibility Level

| Component | Publicly auditable | Publicly rerunnable | Notes |
|---|---:|---:|---|
| Code structure | Yes | Yes | Scripts and configuration are tracked. |
| Manifests and metrics | Yes | Yes | CSV/JSON outputs support inspection and consistency checks. |
| Raw data acquisition | Partially | Controlled | Raw sources are governed by access, licensing, ethical, and platform constraints. |
| Proxy-model pipeline | Partially | Controlled | Full rerun requires controlled-access images and compute environment. |
| Commercial-tool evaluation | Yes, through normalized outputs | No, unless licensed | Requires licensed forensic software and controlled exports. |
| Thesis source | Yes | Yes | LaTeX sources are tracked under `docs/LatexThesis/`. |

---

## Artifact Boundary

This repository supports the experimental claims made in the thesis by preserving the reproducible structure, manifests, scripts, metrics, normalized outputs, and thesis source.

It does not claim to provide unrestricted public redistribution of all source images, proprietary forensic exports, or commercial AI internals.

The correct interpretation is therefore:

```text
public research artifact + controlled-access data policy + black-box commercial-tool evaluation outputs
```

not:

```text
fully open raw dataset + fully reproducible commercial-tool rerun
```
