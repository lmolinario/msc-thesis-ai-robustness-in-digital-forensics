# Data Access and Local Restoration

This repository documents a research pipeline for evaluating the operational
robustness of AI-based forensic image-classification and media-triage tools.

## Main-branch distribution policy

The `main` branch intentionally excludes image corpora. It retains the
reproducibility record required to inspect the frozen thesis artifact:

- acquisition, preparation, training, perturbation, evaluation, and reporting code;
- frozen dataset, split, attack, and bundle manifests;
- hashes, metadata, review logs, and generation summaries;
- normalized predictions and aggregate metrics;
- thesis sources and documentation.

The following data areas are restored or generated locally and are not tracked
on `main`:

```text
datasets/raw/
datasets/prepared/final_pool/images/
datasets/splits/clean/
datasets/splits/ood/
attacks/adversarial/<variant>/
attacks/anti_forensic/<variant>/
datasets/forensic_evaluation_bundle/blind_tool_input/
datasets/forensic_evaluation_bundle/structured_audit_view/
```

## Controlled access procedure

The raw source bundle is stored on Google Drive with **Restricted** access. The
repository retains the Drive page only so that an interested academic reviewer
or researcher can submit an access request:

```text
https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link
```

Opening the page does not grant access or permit download. A signed-in Google
account without permission should receive the Google Drive access-request page.
The requester must select **Request access** and wait for approval by the thesis
author or repository maintainer.

A request should identify:

1. the requester and institutional affiliation;
2. the intended research, academic-review, or reproducibility purpose;
3. the required access period;
4. any applicable legal, ethical, or institutional constraints.

Access may be denied, restricted, time-limited, or revoked when distribution or
use would conflict with source terms, ethical limitations, or institutional
requirements. The Drive link and any downloaded archive must not be
redistributed.

## Local restoration

The restoration entry point is:

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

### 1. Submit the access request

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --request-access
```

The command opens the restricted Drive page in the default browser. Sign in with
the account that should receive access and submit the request.

### 2. Download after approval

For restricted Drive files, the recommended procedure is to download the ZIP
through the authenticated browser after approval and then run:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

An authorized direct-download URL may alternatively be supplied through
`--url` or the local environment variable `FAIRLAB_RAW_DATASET_BUNDLE_URL`.
Private or temporary URLs must never be committed.

The script validates that the supplied object is a ZIP archive, prints its
SHA256 digest, checks archive paths before extraction, and rejects symbolic-link
entries.

## Regeneration boundary

Step 00 restores the raw source bundle only. Prepared images, the frozen local
image pool, clean/OOD splits, adversarial and anti-forensic outputs, and the
blind forensic evaluation bundle are regenerated through the numbered pipeline.

Manifests and hashes committed to the repository provide the reference against
which regenerated files can be checked.

## Legal and ethical note

Source images originate from heterogeneous collections and may remain subject
to third-party licenses, platform terms, ethical restrictions, or institutional
handling requirements. Researchers are responsible for verifying that their
use, storage, and redistribution are lawful and compatible with the original
sources.

The frozen thesis artifact is designed for traceable controlled reproducibility,
not as an unrestricted benchmark redistribution package.
