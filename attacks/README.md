# Attacks

This directory contains the adversarial and anti-forensic perturbations used in the thesis pipeline.

The attack stage starts from the official clean binary folds generated from:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

and from the corresponding clean split manifest:

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

Out-of-distribution samples are not used as primary attack targets. They are kept separate and evaluated through:

```text
datasets/splits/ood/ood_eval_set/
datasets/splits/manifests/ood_eval_manifest.csv
```

---

## Directory Structure

```text
attacks/
├── README.md
├── manifests/
├── adversarial/
│   ├── README.md
│   ├── fgsm/
│   ├── superdeepfool/
│   ├── sigma_zero/
│   ├── one_pixel/
│   └── color_shift/
└── anti_forensic/
    ├── README.md
    ├── jpeg_recompression/
    ├── resample_resize/
    ├── gaussian_blur/
    ├── histogram_modification/
    └── contrast_stretching/
```

---

## Methodological Role

The attack stage has two complementary goals:

1. generate controlled adversarial perturbations against AI-based image classifiers;
2. generate realistic anti-forensic image transformations that may affect automated forensic triage.

Each attack family preserves traceability through manifest files, original hashes, generated hashes, fold identifiers, labels, attack parameters, and model/checkpoint metadata where applicable.

---

## Current Operational Status

Official numbered operational scripts:

```text
models/scripts/12_train_proxy_models.py
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
datasets/scripts/attacks/14_generate_adversarial_attacks.py
evaluation/scripts/15_evaluate_proxy_models.py
datasets/scripts/bundle/16_build_forensic_evaluation_bundle.py
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Implemented and generated anti-forensic transformations:

```text
jpeg_recompression
resample_resize
gaussian_blur
histogram_modification
contrast_stretching
```

Implemented and generated adversarial/adversarial-style attacks:

```text
fgsm
superdeepfool
sigma_zero
one_pixel
color_shift
```

Operational status:

```text
attack generation completed
proxy model evaluation completed
forensic evaluation bundle generated and validated
Magnet AXIOM / Magnet.AI evaluation completed and normalized
additional commercial forensic tools pending / planned extension
```

---

## Anti-Forensic Transformations

Anti-forensic transformations are model-agnostic image-processing operations designed to simulate realistic manipulations that may occur before forensic acquisition or analysis.

Official script:

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
```

Generated transformations:

| Transformation | Family | Model dependency |
|---|---|---|
| `jpeg_recompression` | anti-forensic | model-agnostic |
| `resample_resize` | anti-forensic | model-agnostic |
| `gaussian_blur` | anti-forensic | model-agnostic |
| `histogram_modification` | anti-forensic | model-agnostic |
| `contrast_stretching` | anti-forensic | model-agnostic |

---

## Adversarial Attacks

Adversarial attacks are generated on the binary clean folds. Model-dependent attacks use fold-aware proxy checkpoints.

Official script:

```text
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

Generated attacks:

| Attack | Family | Model dependency | Primary target |
|---|---|---|---|
| `fgsm` | adversarial | model-dependent | `efficientnet_b0` |
| `superdeepfool` | adversarial | model-dependent | `efficientnet_b0` |
| `sigma_zero` | adversarial | model-dependent | `efficientnet_b0` |
| `one_pixel` | adversarial | model-dependent | `efficientnet_b0` |
| `color_shift` | adversarial-style / image transformation | model-agnostic | none |

---

## Fold-Aware Protocol

For every image belonging to fold `F`, model-dependent attacks use the checkpoint:

```text
models/checkpoints/<target_model>/F.pt
```

Example:

```text
image in fold_1 + target efficientnet_b0
→ models/checkpoints/efficientnet_b0/fold_1.pt
```

This ensures that the proxy model used for attack generation was trained on the other four folds and never on the images being attacked.

---

## Evaluation Protocol

Proxy model evaluation is performed by:

```text
evaluation/scripts/15_evaluate_proxy_models.py
```

Main outputs:

```text
evaluation/proxy_models/proxy_model_predictions.csv
results/metrics/proxy_model_clean_metrics.csv
results/metrics/proxy_model_ood_metrics.csv
results/metrics/proxy_model_comparative_metrics.csv
results/metrics/proxy_model_evaluation_summary.json
```

The evaluation compares clean, OOD, adversarial, and anti-forensic samples across the selected local proxy models.

---

## Commercial Forensic-Tool Evaluation Status

The forensic evaluation bundle generated from the clean, OOD, adversarial and anti-forensic artifacts has already been processed with:

```text
Magnet AXIOM / Magnet.AI
```

The Magnet export has been normalized through:

```text
evaluation/scripts/19_normalize_forensic_ai_tool_predictions.py
```

Consolidated outputs:

```text
evaluation/forensic_tools/normalized_predictions.csv
evaluation/forensic_tools/magnet_axiom_normalized_predictions.csv
evaluation/forensic_tools/normalization_summary.json
results/metrics/forensic_tools_metrics.csv
results/metrics/magnet_axiom_metrics.csv
```

The same bundle can later be used for additional black-box commercial-tool extensions, provided that comparable exports are available:

```text
X-Ways Forensics / Excire Photo AI
Cellebrite UFED
Oxygen Forensic Detective
```

---

## Official Generation Commands

Anti-forensic transformations:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py --force
```

Model-agnostic Color Shift:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack color_shift \
  --force
```

Model-dependent adversarial attacks:

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack fgsm \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack superdeepfool \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack sigma_zero \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

```bash
python datasets/scripts/attacks/14_generate_adversarial_attacks.py \
  --attack one_pixel \
  --target-model efficientnet_b0 \
  --checkpoint-root models/checkpoints \
  --device auto \
  --force
```

Smoke tests can be performed with `--limit 10` before full regeneration.

---

## Current Reporting Step

The attack generation, proxy evaluation, forensic bundle construction, and Magnet AXIOM / Magnet.AI normalization stages are complete. The current operational focus is:

```text
finalize Chapter 5 reporting → finalize XAI case studies → keep additional commercial tools as optional extensions unless normalized exports become available
```
