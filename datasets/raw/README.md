# Raw Dataset Restoration

Raw image corpora are intentionally not tracked on the `main` branch.

Restore the externally hosted raw bundle with:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The script uses the repository's default Google Drive bundle URL. It can be
overridden with `--url` or the `FAIRLAB_RAW_DATASET_BUNDLE_URL` environment
variable.

Downloaded and extracted data remain local and are ignored by Git. Prepared
images, clean/OOD splits, perturbations, and the forensic evaluation bundle must
be regenerated through the numbered pipeline.
