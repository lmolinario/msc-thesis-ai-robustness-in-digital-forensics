# Explainability

This directory contains the qualitative explainability workflow used to support the operational robustness analysis in Chapter 5.

The thesis uses **Integrated Gradients (IG)** as diagnostic support for transparent proxy models. XAI outputs are not treated as primary robustness metrics and are not generated for commercial black-box forensic tools.

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

Integrated Gradients are computed only on transparent proxy models. Commercial forensic tools such as Magnet.AI, Excire, Cellebrite UFED, and Oxygen are treated as black boxes and are not explainable through this workflow.

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

Chapter 5 case selection should remain human-in-the-loop: candidate cases can be generated automatically, but final thesis figures should be selected manually for interpretability, visual quality, and methodological relevance.

---

## Output Asset Policy

The official XAI scripts save separate assets for each case:

```text
__input.png                         = input image
__overlay.png                       = Integrated Gradients overlay
__heatmap.png                       = heatmap only
__comparison.png                    = diagnostic side-by-side panel
__mask.png                          = normalized attribution mask
__top*_mask.png                     = top-percentile attribution mask
__distribution.png                  = attribution distribution
```

For thesis inclusion, prefer separate input and overlay images or compact manually curated panels. Avoid turning Chapter 5 into an image gallery.

---

## Thesis Usage

Recommended use in Chapter 5:

```text
5 main XAI cases in the chapter
+ optional additional cases in appendix or supplementary material
```

The XAI section should be positioned after the quantitative robustness results and should be framed as diagnostic interpretation of selected proxy-model failures and successes.

---

## Traceability

Manual XAI selection must be preserved through manifests and logs in:

```text
explainability/manifests/
explainability/logs/
```

This keeps the qualitative selection process reproducible and consistent with the human-in-the-loop methodology of the thesis.
