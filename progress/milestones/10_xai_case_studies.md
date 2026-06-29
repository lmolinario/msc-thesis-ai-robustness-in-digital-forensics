# Milestone 10 — XAI Case Studies

## Status

Completed and integrated in Chapter 5.

## Purpose

This milestone documents the explainability stage of the FAIR-Lab thesis pipeline.

The goal is to provide selected qualitative case studies after quantitative proxy-model evaluation. Explainability is used to support forensic interpretation of model behavior, not to replace the quantitative robustness metrics.

---

## Current State

The XAI generation script is present:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
```

The interactive launcher is also present:

```text
explainability/scripts/18_xai_interactive_launcher.py
```

Integrated Gradients outputs and thesis-ready visual material have been generated for the final representative cases and integrated into Chapter 5.

---

## Selected Representative Cases

The final Chapter 5 XAI cases are:

```text
xai_case_0001 = clean correct weapon
xai_case_0006 = clean false negative weapon
xai_case_0009 = OOD classified as weapon
xai_case_0010 = anti-forensic false negative under histogram modification
xai_case_0015 = high-confidence adversarial false positive under sigma_zero
```

These cases cover clean, OOD, anti-forensic, and adversarial scenarios and are used as qualitative diagnostic support for the quantitative proxy-model robustness analysis.

---

## Main Input for Case Selection

The primary proxy-model prediction file is:

```text
evaluation/proxy_models/proxy_model_predictions.csv
```

Supporting metric files include:

```text
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

---

## Output Areas

The XAI workflow is documented through:

```text
explainability/manifests/
explainability/logs/
explainability/outputs/integrated_gradients/
docs/LatexThesis/images/fig_xai_case*_input.png
docs/LatexThesis/images/fig_xai_case*_heatmap.png
docs/LatexThesis/images/fig_xai_case*_overlay.png
```

The final thesis discussion is integrated in:

```text
docs/LatexThesis/sections/05_experiments.tex
```

---

## Methodological Role

The XAI stage supports:

- qualitative inspection of model attention and sensitivity;
- explanation of representative failures;
- comparison between clean and perturbed image behavior;
- thesis figures for selected case studies;
- forensic interpretation of why certain perturbations may be operationally risky.

The XAI stage does not redefine the quantitative metrics. It complements the robustness results by providing interpretable examples for transparent proxy models only.

---

## Completion Criteria

This milestone is complete because:

- representative clean, OOD, anti-forensic, and adversarial cases have been selected;
- Integrated Gradients outputs have been generated;
- thesis-ready visual outputs have been integrated into the LaTeX thesis source;
- the generated cases are referenced in Chapter 5;
- the XAI conclusions remain consistent with the quantitative proxy-model and commercial-tool findings.
