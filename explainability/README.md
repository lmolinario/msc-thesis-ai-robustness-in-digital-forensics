# Explainability

This directory contains the qualitative explainability workflow used to support the operational robustness analysis in Chapter 5.

The thesis uses **Integrated Gradients (IG)** as diagnostic support for transparent proxy models. XAI outputs are not treated as primary robustness metrics and are not generated for commercial black-box forensic tools.

The Chapter 5 XAI selection is completed and integrated into the thesis text.

---

## Methodological Role

The explainability workflow supports the thesis by helping to inspect representative cases such as:

- clean correct classifications;
- clean false negatives on `weapon` images;
- adversarial failures;
- anti-forensic failures;
- OOD samples classified as weapon or non-weapon with relevant confidence patterns.

The goal is to provide qualitative evidence about what parts of the image influenced a proxy model decision, not to prove causal forensic relevance of a region.

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

Main generation script:

```text
explainability/scripts/17_generate_integrated_gradients_case_studies.py
```

Interactive launcher:

```text
explainability/scripts/18_xai_interactive_launcher.py
```

The launcher builds commands for the official generation script and helps run automatic, manual, attack-stratified, and Chapter 5-oriented XAI workflows.

---

## Supported Models

```text
efficientnet_b0
resnet18
clip
```

Integrated Gradients are computed only on transparent proxy models. Commercial forensic tools such as Magnet AXIOM / Magnet.AI, Excire Foto 2025 and Cellebrite Inseyets are treated as black boxes or operationally opaque tools and are not explained through this workflow.

---

## Supported Strategies

Representative strategies include:

```text
weapon_to_non_weapon
ood_high_confidence
perturbed_failures
attack_stratified
chapter5_core
all
```

Chapter 5 case selection follows the human-in-the-loop methodology: candidate cases can be generated automatically, but final thesis figures are selected manually for interpretability, visual quality, and methodological relevance.

---

## Completed Chapter 5 Case Selection

The final Chapter 5 XAI section uses five representative EfficientNet-B0 Integrated Gradients cases:

| Case ID | Scenario | Manual label | Prediction | Operational role |
|---|---|---|---|---|
| `xai_case_0001` | `clean` | `weapon` | `weapon` | Clean correct reference case |
| `xai_case_0006` | `clean` | `weapon` | `non_weapon` | Clean false negative |
| `xai_case_0009` | `ood` | `ood` | `weapon` | OOD sample classified as weapon |
| `xai_case_0010` | `histogram_modification` | `weapon` | `non_weapon` | Anti-forensic false negative |
| `xai_case_0015` | `sigma_zero` | `non_weapon` | `weapon` | High-confidence adversarial false positive |

These cases are already integrated in:

```text
docs/LatexThesis_ITA/sections/05_experiments.tex
```

The corresponding thesis figure references use the following naming convention:

```text
docs/LatexThesis_ITA/images/fig_xai_case*_input.png
docs/LatexThesis_ITA/images/fig_xai_case*_heatmap.png
docs/LatexThesis_ITA/images/fig_xai_case*_overlay.png
```

---

## Output Asset Policy

The official XAI scripts save separate assets for each case:

```text
__input.png                          = input image
__overlay.png                        = Integrated Gradients overlay
__heatmap.png                        = heatmap only
__comparison.png                     = diagnostic side-by-side panel
__mask.png                           = normalized attribution mask
__top*_mask.png                      = top-percentile attribution mask
__distribution.png                   = attribution distribution
```

For thesis inclusion, prefer separate input and overlay images or compact manually curated panels. Avoid turning Chapter 5 into an image gallery.

---

## Thesis Usage

The XAI section is positioned after the quantitative robustness results and is framed as diagnostic interpretation of selected proxy-model failures and successes.

Its status is:

```text
completed and integrated in Chapter 5
```

XAI is not used to explain Magnet AXIOM / Magnet.AI, Excire Foto 2025, Cellebrite Inseyets, or any other proprietary black-box tool.

---

## Traceability

Manual XAI selection should be preserved through manifests and logs in:

```text
explainability/manifests/
explainability/logs/
```

The final thesis-level selection is also traceable through the LaTeX section and figure references in:

```text
docs/LatexThesis_ITA/sections/05_experiments.tex
```

This keeps the qualitative selection process consistent with the human-in-the-loop methodology of the thesis.
