# Milestone 02 — Prepared Dataset

## Status

Completed.

## Purpose

This milestone documents the technical preparation of the raw image sources into a validated, deduplicated, and indexed image pool.

This stage is intentionally limited to technical dataset preparation. It does not perform semantic annotation, class assignment, manual selection, or final dataset freezing.

## Input directory

```text
datasets/raw/
```

Expected source directories:

```text
01_kaggle_weapon/
02_deepfirearm/
03_google_scraped/
04_telegram_youtube/
05_deepweb/
```

## Scripts used

Prepared dataset construction:

```text
datasets/scripts/prepared/08_build_prepared_dataset.py
```

Review-manifest generation:

```text
datasets/scripts/prepared/09_generate_review_manifest_full.py
```

The second script is the bridge between the technical prepared pool and the human-in-the-loop review stage documented in Milestone 03.

## Main operations

The preparation stage performs:

- recursive image discovery from the raw source directories;
- technical image validation;
- minimum size filtering;
- SHA256 computation;
- global exact-duplicate removal;
- copy of valid unique images into the prepared final pool;
- generation of technical metadata;
- generation of invalid-image and duplicate-discard reports;
- generation of the full review manifest consumed by the manual-selection tool.

## Output directory

```text
datasets/prepared/final_pool/
```

Expected structure:

```text
datasets/prepared/final_pool/images/
datasets/prepared/final_pool/metadata.csv
datasets/prepared/final_pool/reports/prepared_build_summary.json
datasets/prepared/final_pool/reports/invalid_images.csv
datasets/prepared/final_pool/reports/duplicates_discarded.csv
```

## Output files

Main technical output:

```text
datasets/prepared/final_pool/metadata.csv
```

Reports:

```text
datasets/prepared/final_pool/reports/prepared_build_summary.json
datasets/prepared/final_pool/reports/invalid_images.csv
datasets/prepared/final_pool/reports/duplicates_discarded.csv
```

Review manifest for the next stage:

```text
datasets/prepared/manifests/review_manifest_full.csv
```

## Methodological role

This milestone establishes the technical evidence base for the following human-in-the-loop annotation stage.

The resulting metadata file records technical provenance information such as:

- `image_id`;
- prepared filename;
- source dataset;
- source group;
- source relative path;
- SHA256;
- dimensions;
- file size;
- extension;
- image validity status.

The review manifest preserves these technical fields and exposes the prepared image pool to the manual-selection protocol.

## Important decision

The prepared dataset is a technical pool, not the final thesis dataset.

Semantic labels such as `weapon`, `non_weapon`, and `ood` are introduced only in later stages.

## Next milestone

The next milestone is manual selection and final dataset freezing:

```text
progress/milestones/03_manual_selection.md
```
