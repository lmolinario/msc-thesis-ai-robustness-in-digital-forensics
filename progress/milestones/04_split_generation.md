# Milestone 04 — Split Generation

## Status

Completed.

## Purpose

This milestone documents the generation of the clean binary folds and the OOD evaluation set from the final human-reviewed dataset.

The split generation stage creates the reproducible dataset structure used for:

- clean baseline evaluation;
- adversarial attack generation;
- anti-forensic transformation generation;
- OOD reliability evaluation;
- later forensic evaluation bundle construction.

## Input files

```text
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

## Script used

```text
datasets/scripts/splits/11_generate_clean_and_ood_splits.py
```

## Output directories

Clean binary folds:

```text
datasets/splits/clean/fold_1/
datasets/splits/clean/fold_2/
datasets/splits/clean/fold_3/
datasets/splits/clean/fold_4/
datasets/splits/clean/fold_5/
```

OOD evaluation set:

```text
datasets/splits/ood/ood_eval_set/ood/
```

Split manifests:

```text
datasets/splits/manifests/clean_folds_manifest.csv
datasets/splits/manifests/ood_eval_manifest.csv
datasets/splits/manifests/split_generation_summary.json
```

## Clean fold distribution

Each clean fold contains exactly 200 images:

| Fold | Weapon | Non-weapon | Total |
|---|---:|---:|---:|
| `fold_1` | 100 | 100 | 200 |
| `fold_2` | 100 | 100 | 200 |
| `fold_3` | 100 | 100 | 200 |
| `fold_4` | 100 | 100 | 200 |
| `fold_5` | 100 | 100 | 200 |

Total clean binary samples:

```text
1000
```

## OOD evaluation set

OOD samples are not split into folds.

They are stored in a single evaluation set:

```text
datasets/splits/ood/ood_eval_set/ood/
```

Total OOD samples:

```text
500
```

## Source distribution — clean binary subset

| Source dataset | Weapon | Non-weapon |
|---|---:|---:|
| `01_kaggle_weapon` | 100 | 0 |
| `02_deepfirearm` | 100 | 0 |
| `03_google_scraped` | 100 | 30 |
| `04_telegram_youtube` | 100 | 57 |
| `05_deepweb` | 100 | 413 |

## Source distribution — OOD evaluation set

| Source dataset | OOD |
|---|---:|
| `01_kaggle_weapon` | 100 |
| `03_google_scraped` | 71 |
| `04_telegram_youtube` | 225 |
| `05_deepweb` | 104 |

## Integrity checks

The split generation summary reports the following checks as passed:

```text
clean_total_1000      = true
ood_total_500         = true
clean_sha256_unique   = true
ood_sha256_unique     = true
clean_image_id_unique = true
ood_image_id_unique   = true
```

## Methodological role

The clean folds provide the stable experimental units for clean baseline evaluation and for generating adversarial and anti-forensic perturbations.

The OOD evaluation set is kept separate because it answers a different research question: whether AI systems or forensic tools incorrectly treat out-of-distribution or borderline images as weapons.

## Important decisions

- The binary `weapon` / `non_weapon` subset is split into five deterministic clean folds.
- The repository uses `fold_1`, `fold_2`, etc., rather than `test_set_1`, because this is more consistent with machine learning evaluation terminology.
- Each fold is exactly class-balanced: 100 `weapon` and 100 `non_weapon` images.
- OOD images are not attacked and are not folded.
- OOD images are evaluated as a single OOD evaluation set.
- Each copied image is traced through SHA256 and MD5 in the split manifests.

## Next milestone

The next milestone is attack generation:

```text
progress/milestones/05_attack_generation.md
```
