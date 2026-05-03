# Milestone 01 — Dataset Acquisition

## Status

Completed as part of the current working research repository.

## Purpose

This milestone documents the acquisition and organization of the raw image sources used to build the thesis dataset.

The acquisition stage is responsible for collecting or reconstructing the heterogeneous raw sources before any technical validation, deduplication, manual annotation, or final selection is performed.

## Repository location

```text
datasets/scripts/acquisition/
datasets/raw/
```

## Script directory

The acquisition scripts are stored in:

```text
datasets/scripts/acquisition/
```

Expected scripts:

```text
00_download_raw_datasets_bundle.py
01_download_kaggle.py
02_download_github.py
03_build_subset_deepfirearm.py
04_scrape_google.py
05_scrape_telegram.py
06_scrape_youtube.py
07_scrape_deepweb.py
```

## Raw dataset directory

The raw data are stored in:

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

## Methodological role

This stage provides the heterogeneous input pool for the thesis pipeline. The sources are intentionally diverse in order to approximate a realistic forensic scenario involving images obtained from public datasets, OSINT-style collection, web scraping, Telegram/YouTube material, and deep web-oriented sources.

## Important decision

The directory:

```text
datasets/raw/
```

contains raw acquired data and must not be renamed.

The directory:

```text
datasets/scripts/acquisition/
```

contains acquisition scripts. This replaces the previous script-folder naming convention `datasets/scripts/raw/`.

## Output of this milestone

The output of this milestone is the organized raw source tree:

```text
datasets/raw/01_kaggle_weapon/
datasets/raw/02_deepfirearm/
datasets/raw/03_google_scraped/
datasets/raw/04_telegram_youtube/
datasets/raw/05_deepweb/
```

## Next milestone

The next milestone is technical dataset preparation:

```text
progress/milestones/02_prepared_dataset.md
```
