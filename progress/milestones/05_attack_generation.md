# Milestone 05 — Attack and Transformation Generation

## Status

Completed.

## Purpose

This milestone documents the generation of all perturbation artifacts used to stress-test the operational robustness of AI-based image classifiers in the FAIR-Lab thesis pipeline.

The milestone includes two distinct families of manipulations:

1. **adversarial attacks**, used to evaluate model sensitivity to input perturbations generated with adversarial objectives;
2. **anti-forensic transformations**, used to evaluate robustness under realistic image-processing operations that can occur in forensic and anti-forensic scenarios.

The goal is not to optimize Adversarial Machine Learning performance as an isolated research problem. The goal is to evaluate whether AI-based systems used in forensic triage remain reliable when exposed to manipulated images.

---

## Input

```text
datasets/splits/manifests/clean_folds_manifest.csv
```

The input manifest contains the official clean binary subset:

| Class | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| **Total** | **1000** |

OOD samples are not attacked in this stage. They remain reserved for out-of-distribution reliability evaluation.

---

## Official scripts

Anti-forensic generation:

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
```

Adversarial generation:

```text
datasets/scripts/attacks/14_generate_adversarial_attacks.py
```

---

## Generated adversarial attacks

| Attack | Status | Notes |
|---|---|---|
| `fgsm` | Completed | Model-dependent adversarial attack |
| `superdeepfool` | Completed | Model-dependent adversarial attack |
| `sigma_zero` | Completed | Model-dependent adversarial attack |
| `one_pixel` | Completed | Model-dependent adversarial attack |
| `color_shift` | Completed | Model-agnostic color perturbation |

Model-dependent adversarial attacks are generated against the EfficientNet-B0 proxy target where required by the attack implementation.

Expected adversarial outputs:

```text
5 attacks × 1000 binary clean samples = 5000 adversarial images
```

---

## Generated anti-forensic transformations

| Transformation | Status | Main parameter |
|---|---|---|
| `jpeg_recompression` | Completed | `quality = 70` |
| `resample_resize` | Completed | `scale = 0.50`, bicubic |
| `gaussian_blur` | Completed | `radius = 1.50` |
| `histogram_modification` | Completed | global histogram equalization |
| `contrast_stretching` | Completed | `cutoff = 1.0` |

Expected anti-forensic outputs:

```text
5 transformations × 1000 binary clean samples = 5000 anti-forensic images
```

---

## Output directories

Adversarial outputs:

```text
attacks/adversarial/
```

Anti-forensic outputs:

```text
attacks/anti_forensic/
```

Attack manifests:

```text
attacks/manifests/
```

---

## Main output manifests

Adversarial manifests include one manifest per generated adversarial attack and the associated generation metadata under:

```text
attacks/manifests/
```

Anti-forensic manifests:

```text
attacks/manifests/anti_forensic_attacks_manifest.csv
attacks/manifests/anti_forensic_generation_summary.json
```

The anti-forensic evaluation outputs generated during this phase are retained as intermediate attack-level artifacts, but the canonical cross-model evaluation is documented in Milestone 07.

---

## Methodological notes

- Clean binary samples are the only samples perturbed in this phase.
- OOD samples are kept unmodified and evaluated separately.
- Adversarial and anti-forensic artifacts remain traceable to the original clean image through image identifiers, hashes, fold metadata, and manifest mappings.
- Technical filenames may preserve internal identifiers for debugging and reproducibility.
- The later forensic evaluation bundle renames files into opaque identifiers to reduce path-induced and analyst-induced bias.

---

## Completion criteria

This milestone is complete when:

- all five adversarial attacks are generated;
- all five anti-forensic transformations are generated;
- all generated images are tracked through manifests;
- generated files can be mapped back to the clean source images;
- the resulting artifacts are available for proxy-model evaluation and forensic-bundle construction.

Status: **completed**.
