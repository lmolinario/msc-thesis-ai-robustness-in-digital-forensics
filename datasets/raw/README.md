# Raw Dataset Restoration

Raw image corpora are intentionally not tracked on the `main` branch.

The raw bundle is stored on Google Drive with **Restricted** access. The Drive
page is retained only to allow an interested reviewer or researcher to request
authorization:

```text
https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link
```

Opening the link does not grant access. Use:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --request-access
```

Sign in to Google Drive, select **Request access**, and wait for approval by the
thesis author or repository maintainer.

After approval, download the ZIP through the authenticated browser and restore
it locally with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

An authorized direct-download URL may alternatively be provided through `--url`
or `FAIRLAB_RAW_DATASET_BUNDLE_URL`. Private or temporary URLs must not be
committed or redistributed.

Downloaded and extracted data remain local and are ignored by Git. Prepared
images, clean/OOD splits, perturbations, and the forensic evaluation bundle must
be regenerated through the numbered pipeline.
