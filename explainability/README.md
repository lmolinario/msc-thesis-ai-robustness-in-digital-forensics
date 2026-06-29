# Explainability

This directory contains the qualitative Integrated Gradients workflow used in Chapter 5 of the thesis.

Integrated Gradients are used only for transparent proxy models and are not used to explain proprietary black-box tools.

---

## Directory Structure

```text
explainability/
├── README.md
├── scripts/
│   ├── 17_generate_integrated_gradients_case_studies.py
│   └── 18_xai_interactive_launcher.py
├── manifests/
├── logs/
└── outputs/
    └── integrated_gradients/
```

---

## Official Entry Points

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
explainability/scripts/18_xai_interactive_launcher.py
```

Supported proxy models:

```text
efficientnet_b0
resnet18
clip
```

---

## Thesis Integration

The final Chapter 5 XAI cases are integrated in:

```text
docs/LatexThesis/sections/05_experiments.tex
```

The corresponding thesis figures use:

```text
docs/LatexThesis/images/fig_xai_case*_input.png
docs/LatexThesis/images/fig_xai_case*_heatmap.png
docs/LatexThesis/images/fig_xai_case*_overlay.png
```

---

## Traceability

Manual XAI selection should be preserved through manifests and logs in:

```text
explainability/manifests/
explainability/logs/
```

The XAI layer is qualitative and diagnostic. It is not a primary robustness metric.
