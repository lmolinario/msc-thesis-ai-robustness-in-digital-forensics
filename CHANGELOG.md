# Changelog

All notable repository-level changes for the thesis research artifact are documented here.

This changelog is not intended to track every intermediate working edit. It records research-artifact states relevant to thesis auditability, reproducibility, and release management.

---

## v1.0.0-thesis-freeze — Planned Release

Final frozen MSc thesis research artifact.

### Added

- Final thesis source under `docs/LatexThesis/`.
- Frozen dataset manifests under `datasets/final/manifests/`.
- Clean and OOD split manifests under `datasets/splits/manifests/`.
- Forensic evaluation bundle metadata under `datasets/forensic_evaluation_bundle/metadata/`.
- Proxy-model training and evaluation scripts.
- Adversarial and anti-forensic generation scripts and manifests.
- Commercial forensic-tool normalization workflow.
- Final commercial-tool perimeter:
  - Magnet AXIOM / Magnet.AI 10.1.0.48673;
  - Excire Foto 2025 4.1.5;
  - Cellebrite Inseyets 10.9;
  - Magnet Griffeye x64 26.2.108 with T3K CORE v1.18.0.
- Integrated Gradients XAI case studies for transparent proxy models.
- Thesis-oriented reporting scripts and final metric outputs.
- Repository governance documentation:
  - `DATA_ACCESS.md`;
  - `SECURITY.md`;
  - `REPRODUCIBILITY.md`;
  - `ACADEMIC_REPOSITORY_AUDIT.md`;
  - `THESIS_ARTIFACT.md`;
  - `REPOSITORY_MAP.md`;
  - `ARTIFACT_EVALUATION.md`;
  - `DATA_DICTIONARY.md`;
  - `ENVIRONMENT.md`.

### Changed

- Root `README.md` aligned with the final frozen repository structure.
- Official script sequence expanded to steps `00–21` with paths and purposes.
- Progress documentation aligned with completed Cellebrite, Griffeye/T3K, and XAI stages.
- Commercial-tool documentation aligned with the final four-tool perimeter.
- Local absolute paths removed from public split summary metadata.

### Removed / Excluded

- Stale references to excluded tools from final public-facing documentation.
- Stale references to the previous thesis source directory were removed from final source-of-truth documentation.
- Public documentation claims implying access to proprietary commercial-tool internals.

### Notes

This release is intended to represent the official thesis research artifact. The public repository supports structural audit, metric inspection, thesis-source review, and controlled reproducibility. Full reruns of raw-data and commercial-tool stages require controlled-access data and licensed forensic software.

---

## Pre-freeze Development

Earlier repository states included working notes, partial milestones, local scripts, intermediate reports, and evolving documentation. Those states are not treated as the final source of truth for the thesis.

The final source of truth is defined by:

```text
README.md
THESIS_ARTIFACT.md
REPOSITORY_MAP.md
ARTIFACT_EVALUATION.md
REPRODUCIBILITY.md
DATA_ACCESS.md
SECURITY.md
docs/LatexThesis/
results/metrics/
evaluation/forensic_tools/
```
