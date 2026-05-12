# Milestone 10 — XAI Case Studies

## Status

Prepared, not yet produced.

## Purpose

This milestone documents the explainability stage of the FAIR-Lab thesis pipeline.

The goal is to generate selected qualitative case studies after the quantitative proxy-model evaluation and, where available, after commercial forensic-tool evaluation. Explainability is used to support forensic interpretation of model behavior, not to replace the quantitative robustness metrics.

---

## Current state

The XAI generation script is already present:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
```

The interactive launcher is also present:

```text
explainability/scripts/18_xai_interactive_launcher.py
```

XAI outputs are not yet part of the completed repository state. They should be generated only after selecting the most relevant cases from the proxy-model results and, later, from forensic-tool outputs.

---

## Main input for case selection

The primary proxy-model prediction file is:

```text
evaluation/proxy_models/proxy_model_predictions.csv
```

Supporting metric files include:

```text
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_adversarial_metrics.csv
results/metrics/proxy_model_anti_forensic_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/final_core_metrics.csv
results/metrics/final_robustness_metrics.csv
results/metrics/final_confusion_matrices.csv
results/metrics/final_ood_metrics.csv
```

---

## Recommended case-selection criteria

XAI case studies should prioritize examples that are relevant to the forensic objective of the thesis:

- clean-correct images that become misclassified after perturbation;
- `weapon` images classified as `non_weapon` after adversarial or anti-forensic manipulation;
- high-confidence wrong predictions;
- cases where `efficientnet_b0`, `resnet18`, and `clip` disagree;
- examples from the strongest adversarial degradation conditions, especially `sigma_zero` and `fgsm`;
- operationally meaningful anti-forensic failures such as blur, histogram modification, or contrast changes;
- OOD samples incorrectly assigned to binary forensic categories;
- cases that are also interesting for later comparison with commercial forensic-tool behavior.

---

## Expected output areas

Expected XAI outputs should be stored under:

```text
explainability/case_studies/
explainability/manifests/
results/figures/xai/
```

The exact output structure may be refined when the case-selection manifest is created.

---

## Methodological role

The XAI stage supports:

- qualitative inspection of model attention/sensitivity;
- explanation of representative failures;
- comparison between clean and perturbed image behavior;
- thesis figures for selected case studies;
- forensic interpretation of why certain perturbations may be operationally risky.

The XAI stage does not redefine the quantitative metrics. It complements the robustness results by providing interpretable examples.

---

## Completion criteria

This milestone will be complete when:

- a case-selection manifest is created;
- selected clean, adversarial, anti-forensic, and OOD cases are documented;
- Integrated Gradients outputs are generated;
- representative visual outputs are stored in a thesis-ready location;
- the generated cases are referenced in the Results chapter or appendix;
- the XAI conclusions remain consistent with the quantitative proxy-model and forensic-tool findings.

Status: **prepared, not yet produced**.
