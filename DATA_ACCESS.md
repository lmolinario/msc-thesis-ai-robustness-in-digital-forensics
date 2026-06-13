# Data Access Policy

This repository documents a research pipeline for evaluating the operational robustness of AI-based forensic image-classification and media-triage tools.

The repository intentionally separates reproducible code, manifests, metrics, documentation, and thesis material from raw image data that may be subject to third-party licensing, platform terms, source-specific limitations, ethical constraints, or institutional review considerations.

---

## Publicly available in the repository

The public repository may include:

- source code for dataset preparation, model training, perturbation generation, evaluation, normalization, and reporting;
- dataset manifests and audit metadata where redistribution is appropriate;
- hash-based traceability artifacts;
- aggregate metrics and thesis-ready result tables;
- LaTeX thesis source files;
- documentation describing the experimental protocol.

---

## Not publicly distributed

The archived raw dataset bundle is not distributed through a public URL in this repository.

Raw images and source-specific exports are kept under controlled access because they may include material obtained from heterogeneous sources, including public datasets, web scraping, social-media-like sources, and other operationally relevant acquisition contexts.

The repository therefore does not expose:

- public raw dataset download links;
- private Google Drive URLs;
- local acquisition credentials;
- forensic-tool proprietary case files;
- commercial-tool working databases;
- temporary signed URLs or installer links.

---

## Controlled access procedure

Access to the raw dataset bundle may be requested from the thesis author or repository maintainer.

Requests should specify:

1. the requester identity and affiliation;
2. the intended research or review purpose;
3. whether access is needed for thesis verification, academic review, or reproducibility assessment;
4. any applicable legal, ethical, or institutional constraints.

Access may be denied or restricted when redistribution would be incompatible with source terms, ethical limitations, or institutional constraints.

---

## Local restoration mechanism

The public bootstrap script is:

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The script does not contain a hardcoded URL. After controlled access has been granted, the raw bundle URL must be configured locally through the environment variable:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

Example for Windows PowerShell:

```powershell
$env:FAIRLAB_RAW_DATASET_BUNDLE_URL="<controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Example for Linux/macOS:

```bash
export FAIRLAB_RAW_DATASET_BUNDLE_URL="<controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The environment variable must not be committed to the repository.

---

## Reproducibility note

The repository is designed for controlled reproducibility rather than unrestricted redistribution of all raw data. The methodological record is preserved through scripts, manifests, hashes, metrics, normalized outputs, and thesis documentation.

Where raw images cannot be redistributed, auditability is supported through documented acquisition procedures, file-level hashes, frozen manifests, and aggregate experimental outputs.
