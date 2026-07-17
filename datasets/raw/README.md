# Raw Dataset Restoration

Raw image corpora are intentionally not tracked on the `main` branch.

Access to the externally hosted raw bundle is granted case by case by the thesis
author or repository maintainer. No private download URL is stored in this
repository.

After authorization, provide the received URL locally through:

```text
FAIRLAB_RAW_DATASET_BUNDLE_URL
```

and run:

```bash
python datasets/scripts/acquisition/00_download_raw_datasets_bundle.py
```

The URL may alternatively be supplied with `--url`. It must not be committed,
published, or redistributed.

Downloaded and extracted data remain local and are ignored by Git. Prepared
images, clean/OOD splits, perturbations, and the forensic evaluation bundle must
be regenerated through the numbered pipeline.
