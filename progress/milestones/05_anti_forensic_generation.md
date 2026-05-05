# Milestone 05 — Anti-Forensic Attack Generation

## Status

Completed.

## Official script

`datasets/scripts/attacks/13_generate_anti_forensic_attacks.py`

## Input

`datasets/splits/manifests/clean_folds_manifest.csv`

The input manifest contains the official clean binary subset:

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |
| total | 1000 |

The OOD subset is not attacked in this stage and remains reserved for out-of-distribution evaluation.

## Output directories

`attacks/anti_forensic/`

The generated outputs are organized by attack, fold, and class:

```text
attacks/anti_forensic/<attack_name>/<fold>/<label>/<image_id>__<attack_name>.jpg
```
## Output manifests

- `attacks/manifests/anti_forensic_attacks_manifest.csv`
- `attacks/manifests/anti_forensic_generation_summary.json`

## Transformations

| Attack name | Description | Parameter |
|---|---|---|
| `jpeg_recompression` | JPEG recompression | `quality = 70` |
| `resample_resize` | Downsample and resize back to the original dimensions | `scale = 0.50`, bicubic |
| `gaussian_blur` | Gaussian filtering | `radius = 1.50` |
| `histogram_modification` | Global histogram equalization | `ImageOps.equalize` |
| `contrast_stretching` | Automatic contrast stretching | `cutoff = 1.0` |

## Validation summary

| Check | Result |
|---|---:|
| input images | 1000 |
| selected attacks | 5 |
| expected generated images | 5000 |
| actual generated images | 5000 |
| images per attack | 1000 |
| images per fold per attack | 200 |
| weapon per attack | 500 |
| non_weapon per attack | 500 |
| generated image IDs unique | true |
| perturbed SHA256 unique | true |
| manifest written | true |

## Methodological notes

The anti-forensic transformations are controlled image post-processing operations, not model-optimized adversarial examples.

The technical filenames intentionally preserve the source `image_id` and the `attack_name` to support traceability and debugging during internal experimental development.

A later forensic evaluation bundle should rename files using opaque identifiers, such as `sample_0000001.jpg`, while preserving the complete mapping only through a dedicated manifest.

All generated images are saved in JPEG format. Therefore, all transformations include final JPEG serialization. This design choice reflects a realistic operational scenario in which images may be recompressed, exported, forwarded, or processed by external tools.