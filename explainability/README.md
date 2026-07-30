# Explainability

This directory contains the qualitative Integrated Gradients workflow discussed
in Chapter 6 of the thesis. Integrated Gradients are applied only to transparent
proxy models; they are not used to explain proprietary black-box tools.

## Historical Naming Note

The paths, script names, strategy identifier, and frozen manifests containing
`chapter5` were created before the final thesis reorganization that separated
implementation (Chapter 5) from results (Chapter 6). They are preserved to avoid
breaking artifact identity, provenance, and validation. They do not indicate the
current chapter in which the results are discussed.

## Public Artifact Boundary

`main` retains only the material required to audit the thesis-level XAI
discussion:

```text
explainability/
├── README.md
├── scripts/
│   ├── 17_generate_integrated_gradients_case_studies.py
│   ├── _17_generate_integrated_gradients_case_studies_impl.py
│   ├── 18_xai_interactive_launcher.py
│   ├── sync_chapter5_xai_metadata.py
│   └── validate_chapter5_xai_artifacts.py
├── manifests/
│   └── chapter5/
│       ├── run_summary.json
│       ├── candidate_selection_summary.json
│       ├── thesis_selection.csv
│       └── run_provenance.json
└── logs/
    ├── chapter5_manual_selection.csv
    └── xai_interactive_launcher_commands.jsonl
```

Generated diagnostic images under `explainability/outputs/` are local artifacts
and are excluded from `main`. The thesis retains only the 20 assets used in
Chapter 6 under `docs/LatexThesis/images/`: input, heatmap, overlay, and top-10%
attribution mask for each of five cases.

The complete pre-minimization XAI output tree remains available in the protected
historical snapshot documented in `docs/artifact/ARCHIVE_SNAPSHOT.md`.

## Official Entry Points

```bash
python explainability/scripts/17_generate_integrated_gradients_case_studies.py --help
python explainability/scripts/18_xai_interactive_launcher.py
python explainability/scripts/sync_chapter5_xai_metadata.py --check
python explainability/scripts/validate_chapter5_xai_artifacts.py
```

The public entry point wraps the frozen implementation and adds:

- headless-safe batch execution (`Agg`) and an interactive backend only for
  manual review;
- SHA-256 validation of checkpoints against `models/model_registry.json`;
- SHA-256 validation of input images when an expected hash is available;
- run-specific cleanup when `--force` is used;
- adaptive Integrated Gradients steps up to `--max-n-steps`;
- normalized convergence diagnostics and a fail-visible status;
- pseudonymous reviewer identifiers by default;
- runtime and integrity metadata in the run summary.

## Frozen Five-Case Selection

The original review considered 1,175 candidates and selected 15 cases, three per
narrative bucket. The thesis uses five representative cases:

| Case | Historical bucket identifier | Diagnostic role |
|---|---|---|
| `xai_case_0001` | `clean_correct_weapon` | clean correct reference |
| `xai_case_0006` | `clean_false_negative_weapon` | clean missed detection |
| `xai_case_0009` | `ood_as_weapon` | OOD input assigned to `weapon` with high Max-P |
| `xai_case_0010` | `anti_forensic_failure` | histogram-modification false negative |
| `xai_case_0015` | `adversarial_high_conf_failure` | Sigma-Zero incorrect prediction with Max-P 1.000 |

`adversarial_high_conf_failure` is a historical frozen bucket name. In the final
thesis, the associated numerical value is described as maximum predicted-class
probability (`Max-P`), not calibrated confidence.

The authoritative public mapping is:

```text
explainability/manifests/chapter5/thesis_selection.csv
```

The synchronizer checks or updates the Max-P values displayed in the authoritative
results source:

```text
docs/LatexThesis/sections/06_results.tex
```

```bash
python explainability/scripts/sync_chapter5_xai_metadata.py --check
python explainability/scripts/sync_chapter5_xai_metadata.py --write
```

## Regeneration and Convergence

The historical thesis figures were generated with 32 integration steps and a
zero baseline in the preprocessed input space. Some legacy convergence deltas
were large, especially for Sigma-Zero cases. The hardened entry point therefore
supports adaptive regeneration:

```bash
python explainability/scripts/17_generate_integrated_gradients_case_studies.py \
  --selection-manifest <LOCAL_SELECTION_MANIFEST.csv> \
  --model efficientnet_b0 \
  --strategy chapter5_core \
  --attribution-target predicted_label \
  --n-steps 64 \
  --max-n-steps 256 \
  --convergence-threshold 0.05 \
  --device auto \
  --force
```

A regenerated attribution is marked `passed` only when its normalized convergence
delta meets the configured threshold. A run that reaches the maximum number of
steps without meeting the threshold remains explicit as `threshold_not_met`; no
silent claim of convergence is made.

## Interpretation Limits

The XAI layer is qualitative and diagnostic. Attribution maps indicate
sensitivity of the proxy-model output along the selected baseline path; they are
not semantic segmentations, causal proofs, calibrated probabilities, robustness
metrics, or explanations of commercial tools. The final interpretation remains
human-in-the-loop.
