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

## External raw bundle

The raw source bundle is hosted externally on Google Drive and is restored with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

Default bundle URL:

```text
https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link
```

The Drive file must be shared as **Anyone with the link – Viewer** for
unattended restoration. Publishing the URL does not grant redistribution rights
beyond those applicable to the original sources.

URL precedence is:

1. `--url <bundle-url>`;
2. `FAIRLAB_RAW_DATASET_BUNDLE_URL`;
3. the default URL embedded in the script.

Examples:

```powershell
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
$env:FAIRLAB_RAW_DATASET_BUNDLE_URL="<alternative-url>"
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py --force-download
```

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
FAIRLAB_RAW_DATASET_BUNDLE_URL="<alternative-url>" \
  python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py --force-download
```

The archive is validated as ZIP, its SHA256 is printed, archive paths are
checked before extraction, and symbolic-link entries are rejected.

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

The frozen thesis artifact is designed for traceable reproducibility, not as an
unrestricted benchmark redistribution package.
