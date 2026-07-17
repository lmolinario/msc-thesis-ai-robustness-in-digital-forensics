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

The raw source bundle is stored externally and is not available through a
publicly usable download link. Access is granted case by case by the thesis
author or repository maintainer.

A request should identify:

1. the requester and institutional affiliation;
2. the intended research, academic-review, or reproducibility purpose;
3. the required access period;
4. any applicable legal, ethical, or institutional constraints.

Access may be denied, restricted, time-limited, or revoked when distribution or
use would conflict with source terms, ethical limitations, or institutional
requirements.

After authorization, the requester receives an access-controlled URL. The URL
must remain local and must not be committed, published, or redistributed.

## Local restoration

The restoration entry point is:

```text
datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Provide the authorized URL through either the command line or the environment
variable `FAIRLAB_RAW_DATASET_BUNDLE_URL`.

Windows PowerShell:

```powershell
$env:FAIRLAB_RAW_DATASET_BUNDLE_URL="<authorized-controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Linux/macOS:

```bash
export FAIRLAB_RAW_DATASET_BUNDLE_URL="<authorized-controlled-access-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Alternative one-time invocation:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --url "<authorized-controlled-access-url>"
```

The script validates that the downloaded object is a ZIP archive, prints its
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
