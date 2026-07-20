# Raw Dataset Restoration

Raw image corpora are intentionally not tracked on the `main` branch.

The controlled archive is named:

```text
00_raw_datasets_bundle.zip
```

Its authoritative SHA-256 digest is:

```text
a6103ec76e47c7951b11bfc42f932b5bf59f24532784adf82d42c470ba89a12e
```

The machine-readable source of truth is:

```text
docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256
```

The raw bundle is stored with **Restricted** access. The stable access-request
page is retained only to allow an interested reviewer or researcher to request
authorization:

```text
https://drive.google.com/file/d/1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link
```

Opening the link does not grant access. Use:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --request-access
```

After approval, download the ZIP through the authenticated browser and restore
it locally with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py \
  --artifact raw \
  --archive "/path/to/00_raw_datasets_bundle.zip"
```

The script automatically verifies the archive SHA-256 against the authoritative
repository checksum before extraction.

An authorized direct-download URL may alternatively be provided through `--url`
or `FAIRLAB_RAW_DATASET_BUNDLE_URL`. Private or temporary URLs must not be
committed or redistributed.

Downloaded and extracted data remain local and are ignored by Git. Prepared
images, clean/OOD splits, and perturbations can be regenerated through the
numbered pipeline. The exact 11,500-file black-box input can alternatively be
restored from the separately controlled frozen forensic evaluation bundle.

See [`../../docs/artifact/DATA_ACCESS.md`](../../docs/artifact/DATA_ACCESS.md) for
access conditions, both archive digests, and the complete restoration boundary.
