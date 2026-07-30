# Results Scripts

The numbered scripts in this directory operate only on frozen downstream
artifacts. They do not change the official dataset, perturbations, model
checkpoints, or source predictions unless an explicit installation option is
requested.

The path `results/figures/chapter_5/` and the `chapter5_*` file names are
historical frozen identifiers created before the final thesis reorganization.
The generated material is used in the experimental-results chapter, Chapter 6.

## `20_generate_experimental_reporting_assets.py`

Generates Chapter 6 reporting figures and tables from consolidated metric files.
The authoritative numerical sources remain under `results/metrics/`.

## `21_generate_embedded_metadata_sensitivity_check.py`

Reproduces the frozen embedded-metadata leave-out sensitivity analysis using:

```text
evaluation/forensic_tools/normalized_predictions.csv
```

The retained detailed output is:

```text
results/figures/chapter_5/tab_embedded_metadata_sensitive_hits_detail.csv
```

The analysis is descriptive and does not establish causal influence of metadata
on a proprietary decision process.

## `22_generate_public_embedded_metadata_sensitivity_check.py`

Optional privacy-reduced workflow. It requires a separately generated minimized
metadata audit and does not replace the frozen detailed workflow unless
explicitly run and validated.

## `23_validate_results_artifacts.py`

Read-only validation of:

- 69,000 canonical commercial-tool decisions;
- 186 commercial metric rows;
- 40,500 proxy prediction rows;
- OOD accounting (`500 images × 5 folds = 2,500 predictions per architecture`);
- the historically named reporting-manifest counts and provenance;
- embedded-metadata sensitivity counts.

Run:

```bash
python results/scripts/23_validate_results_artifacts.py
```

## `24_audit_reporting_asset_usage.py`

Audits the historically named reporting assets against the authoritative LaTeX
thesis tree. It identifies:

- asset IDs referenced by the thesis;
- reporting files absent from the repository;
- thesis-ready copies with the same filename;
- byte-identical copies;
- existing copies whose bytes differ;
- asset IDs not referenced by the thesis.

Run without modifying files:

```bash
python results/scripts/24_audit_reporting_asset_usage.py --strict
```

To create an ignored local report:

```bash
python results/scripts/24_audit_reporting_asset_usage.py \
  --strict \
  --report results/reporting_asset_usage_summary.json
```

No duplicate or unreferenced asset should be deleted solely from this audit.
Removal requires a separate review of LaTeX references, binary equivalence, and
the role of the asset as a reproducibility/reporting output.
