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

## Controlled artifacts

Two distinct controlled-access archives support different reproducibility goals:

| Artifact | Purpose | Archive | Authoritative SHA-256 |
|---|---|---|---|
| Raw dataset bundle | Restore the heterogeneous source corpora and rerun the numbered image pipeline | `00_raw_datasets_bundle.zip` | `a6103ec76e47c7951b11bfc42f932b5bf59f24532784adf82d42c470ba89a12e` |
| Frozen forensic evaluation bundle | Restore the exact 11,500 files used as black-box commercial-tool input | `16_frozen_forensic_evaluation_bundle.zip` | `1ced63e6dff01379a26770d5e942f4a3c16e02bfc5bbb3d82319c20a7058050d` |

The machine-readable source of truth is:

```text
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

The archive-level digest identifies the complete downloaded ZIP. For the frozen
bundle, the restoration script additionally verifies every blind input against:

```text
datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv
```

## Controlled access procedure

The raw source bundle is stored with **Restricted** access. The repository
retains the stable access-request page so that an interested academic reviewer
or researcher can request authorization:

```text
https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link
```

The frozen forensic evaluation bundle is also distributed only under controlled
access. Its stable request page must be provided through the repository
configuration, `--request-page`, or the local environment variable
`FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_REQUEST_URL` after the upload has been
frozen. Signed, private, or temporary direct-download URLs must never be
committed.

Opening an access-request page does not grant permission or permit download. A
requester should identify:

1. the requester and institutional affiliation;
2. the intended research, academic-review, or reproducibility purpose;
3. the required access period;
4. any applicable legal, ethical, or institutional constraints.

Access may be denied, restricted, time-limited, or revoked when distribution or
use would conflict with source terms, ethical limitations, or institutional
requirements. Access links and downloaded archives must not be redistributed.

## Restoration entry point

Both archives are handled by:

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The script automatically reads the authoritative archive digest from
`docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256`. The optional
`--expected-sha256` argument is retained as an explicit override.

## Raw dataset restoration

### Request access

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --request-access
```

### Restore after browser download

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

An authorized direct-download URL may alternatively be supplied through `--url`
or the local environment variable `FAIRLAB_RAW_DATASET_BUNDLE_URL`.

Step 00 restores the raw source bundle only. Prepared images, the frozen local
image pool, clean/OOD splits, adversarial and anti-forensic outputs, and the
forensic evaluation bundle can then be regenerated through the numbered
pipeline. Manifests and hashes committed to the repository provide the reference
against which regenerated files can be checked.

## Exact frozen-bundle restoration

Request access after the stable frozen-bundle request page has been configured:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact frozen \
  --request-access
```

Restore the approved browser download with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact frozen \
  --archive "/path/to/16_frozen_forensic_evaluation_bundle.zip"
```

The script performs two integrity layers:

1. complete-ZIP SHA-256 verification against the repository checksum file;
2. filename and per-file SHA-256 verification of all 11,500 blind inputs against
   `datasets/forensic_evaluation_bundle/metadata/bundle_hashes_sha256.csv`.

For black-box forensic-tool evaluation, import only:

```text
datasets/forensic_evaluation_bundle/blind_tool_input/files/
```

Do not import `metadata/` or `structured_audit_view/` into the evaluated tools.

An authorized direct-download URL may be supplied through `--url` or the local
environment variable `FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_URL`.

## Manual checksum verification

From the directory containing both downloaded archives and a copy of the
repository checksum file:

```bash
sha256sum --check docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

Individual checks can also be performed with:

```bash
sha256sum 00_raw_datasets_bundle.zip
sha256sum 16_frozen_forensic_evaluation_bundle.zip
```

## Legal and ethical note

Source images originate from heterogeneous collections and may remain subject
to third-party licenses, platform terms, ethical restrictions, or institutional
handling requirements. Researchers are responsible for verifying that their
use, storage, and redistribution are lawful and compatible with the original
sources.

The frozen thesis artifact is designed for traceable controlled reproducibility,
not as an unrestricted benchmark redistribution package.
