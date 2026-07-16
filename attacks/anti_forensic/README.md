# Anti-Forensic Transformations

This directory contains controlled image-processing transformations used to
evaluate robustness under realistic anti-forensic conditions.

The transformations are model-agnostic and are applied to the official clean
binary folds. They are not optimization-based adversarial examples.

## Frozen transformation set

| Transformation | Frozen configuration |
|---|---|
| `jpeg_recompression` | JPEG quality 70 |
| `resample_resize` | scale 0.5, bicubic downsampling and restoration to the original dimensions |
| `gaussian_blur` | Gaussian radius 1.5 |
| `histogram_modification` | histogram equalization |
| `contrast_stretching` | autocontrast cutoff 1% |

Non-recompression outputs are saved as JPEG quality 95 after the selected
transformation. Exact parameters are preserved in the manifest.

## Official entry point

```text
datasets/scripts/attacks/13_generate_anti_forensic_attacks.py
```

Generate the frozen set:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py --force
```

Generate one transformation:

```bash
python datasets/scripts/attacks/13_generate_anti_forensic_attacks.py   --attack gaussian_blur   --force
```

A smoke test can be run with `--limit 10`.

## Output layout

```text
attacks/anti_forensic/<transformation>/<fold>/<label>/<image_id>__<transformation>.jpg
```

The fold and label hierarchy is preserved for controlled evaluation; the blind
forensic-tool bundle later replaces semantic paths with neutral identifiers.

## Traceability

Each generated artifact records, at minimum:

```text
generated_image_id
original_image_id
fold
final_label
clean_relative_path
perturbed_relative_path
attack_name
attack_parameters
sha256_original
sha256_perturbed
md5_perturbed
size_bytes
created_at
```

Canonical generation records:

```text
attacks/manifests/anti_forensic_attacks_manifest.csv
attacks/manifests/anti_forensic_generation_summary.json
```

The transformations are evaluated as a separate robustness family from FGSM,
One Pixel, Sigma-Zero, SuperDeepFool, and the adversarial-style Color Shift
variant.
