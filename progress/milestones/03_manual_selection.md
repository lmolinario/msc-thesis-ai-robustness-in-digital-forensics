# Milestone 03 — Manual Selection

## Status

Completed.

## Purpose

This milestone documents the human-in-the-loop manual selection stage used to construct the final frozen dataset for the thesis.

The manual selection stage transforms the prepared technical image pool into a semantically reviewed dataset with three final groups:

- `weapon`;
- `non_weapon`;
- `ood`.

## Input file

```text
datasets/prepared/manifests/review_manifest_full.csv
```

This manifest is generated from:

```text
datasets/scripts/prepared/09_generate_review_manifest_full.py
```

## Script used

```text
datasets/scripts/final/10_manual_selection_protocol_reviewer.py
```

## Main outputs

Final manifests:

```text
datasets/final/manifests/manual_selection_protocol_db.csv
datasets/final/manifests/manual_selection_final_1500.csv
datasets/final/manifests/manual_selection_adversarial_subset.csv
datasets/final/manifests/manual_selection_removed.csv
```

Reports:

```text
datasets/final/reports/manual_selection_summary.json
datasets/final/reports/manual_selection_log.csv
datasets/final/reports/manual_selection_state.json
datasets/final/reports/backups/
```

## Final dataset

The official frozen dataset is:

```text
datasets/final/manifests/manual_selection_final_1500.csv
```

The previous `33_final_frozen_dataset.csv` naming convention is no longer used.

## Final counts

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |
| ood | 500 |
| exclude | 0 |

Total reviewed final samples:

```text
1500
```

## Adversarial / anti-forensic subset

The official binary subset used for clean fold generation and attack generation is:

```text
datasets/final/manifests/manual_selection_adversarial_subset.csv
```

It contains:

| Class | Count |
|---|---:|
| weapon | 500 |
| non_weapon | 500 |

Total binary samples:

```text
1000
```

## Source distribution at final review

| Source dataset | Weapon | Non-weapon | OOD | Reviewed |
|---|---:|---:|---:|---:|
| `01_kaggle_weapon` | 100 | 0 | 100 | 200 |
| `02_deepfirearm` | 100 | 0 | 0 | 100 |
| `03_google_scraped` | 100 | 30 | 71 | 201 |
| `04_telegram_youtube` | 100 | 57 | 225 | 382 |
| `05_deepweb` | 100 | 413 | 104 | 617 |

## Methodological role

This milestone is the central human-in-the-loop component of the pipeline.

The final dataset is not generated through automatic labeling alone. It is produced through manual semantic inspection and logged reviewer decisions.

This provides:

- explicit human validation;
- traceable class assignment;
- reproducible final manifests;
- a clear distinction between binary task samples and OOD samples.

## Important decisions

- `manual_selection_final_1500.csv` is the official frozen dataset.
- `manual_selection_adversarial_subset.csv` is the official 1000-image binary subset.
- OOD images are preserved for separate reliability/OOD evaluation.
- OOD images are not used as primary targets for adversarial or anti-forensic attack generation.
- The source imbalance in `non_weapon` and OOD samples is a documented methodological limitation, not a technical error.

## Next milestone

The next milestone is split generation:

```text
progress/milestones/04_split_generation.md
```
