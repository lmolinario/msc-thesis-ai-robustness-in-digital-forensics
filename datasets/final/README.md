# Final Dataset

This directory contains the frozen human-reviewed dataset manifests and the
audit reports produced by the manual selection protocol.

For class definitions, intended uses, limitations, and distribution policy,
see [`../DATASET_CARD.md`](../DATASET_CARD.md).

## Canonical manifests

| Artifact | Purpose |
|---|---|
| `manifests/manual_selection_protocol_db.csv` | Review database and semantic assignments |
| `manifests/manual_selection_final_1500.csv` | Official frozen 1,500-image dataset |
| `manifests/manual_selection_adversarial_subset.csv` | Official 1,000-image binary subset |
| `manifests/manual_selection_removed.csv` | Samples excluded from the final selection |

## Canonical reports

| Artifact | Purpose |
|---|---|
| `reports/manual_selection_log.csv` | Chronological audit log of review actions |
| `reports/manual_selection_summary.json` | Final counts, outputs, and consistency checks |

The GUI session-resume state is generated locally by the reviewer script and
is intentionally excluded from the frozen repository. It is not a scientific
result and is not required to verify the final selection.

## Frozen composition

| Group | Count |
|---|---:|
| `weapon` | 500 |
| `non_weapon` | 500 |
| `ood` | 500 |
| **Total** | **1500** |

The adversarial and anti-forensic subset contains 500 `weapon` and 500
`non_weapon` samples. OOD inputs are evaluated separately and are not used for
perturbation generation.

## Downstream use

The frozen manifests feed:

- clean fold generation under `datasets/splits/`;
- OOD evaluation;
- adversarial and anti-forensic generation under `attacks/`;
- proxy-model evaluation;
- construction of the forensic evaluation bundle.

Downstream processing must use stable identifiers and hashes rather than
relying only on filenames. The frozen manifests are immutable for the thesis
release.
