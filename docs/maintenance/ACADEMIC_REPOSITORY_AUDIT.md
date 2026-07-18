# Academic Repository Audit

This document records the repository-level review performed to align the public project with the frozen MSc thesis artifact.

Repository:

```text
lmolinario/msc-thesis-ai-robustness-in-digital-forensics
```

Thesis:

```text
Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks
```

## Review Criteria

The audit considers:

1. clear thesis scope and contribution;
2. traceable experimental workflow;
3. frozen manifests and integrity metadata;
4. controlled data-access policy;
5. public/private artifact separation;
6. documented execution entry points;
7. consistent proxy and commercial result layers;
8. citation, license, and security metadata;
9. consistency between repository documentation and thesis source;
10. absence of exposed secrets, raw controlled corpora, and unnecessary proprietary exports.

## Completed Alignment Work

| Area | Final state |
|---|---|
| Historical working notes | Removed from `main` and preserved in the protected snapshot |
| Image corpora | Excluded from `main`; manifests and access procedure retained |
| Commercial raw exports | Excluded from `main`; protected snapshot and controlled storage retained |
| Commercial prediction evidence | 69,000 sanitized canonical rows committed |
| Commercial metric evidence | 186 frozen rows exactly reproduced |
| Tool-specific extracts | Four sanitized extracts committed and hash-registered |
| Proxy models | 15 fold-aware checkpoints retained and registry-hashed |
| XAI | Historical bulk outputs removed; five cases and 20 thesis assets retained |
| Results | Validators added; OOD denominator documented |
| LaTeX | English source authoritative; Italian reference documented; build outputs ignored |
| Root structure | Governance docs moved to `docs/`; utilities moved to `tools/`; security policy moved to `.github/` |
| CI | Required paths, canonical prediction SHA256, schema, metrics, and stale references checked |

## Final Commercial Perimeter

| Tool/configuration | Version | Status |
|---|---:|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Included |
| Excire Foto D20 | 4.1.5 | Included |
| Excire Foto D50 | 4.1.5 | Included |
| Excire Foto D80 | 4.1.5 | Included |
| Cellebrite Inseyets | 10.9 / Physical Analyzer 10.9.0.3029 | Included |
| Griffeye / T3K CORE | 26.2.108 / 1.18.0 | Included |

## Canonical Evidence Profile

```text
Frozen source dataset                     1,500 images
Binary subset                             1,000 images
Unique OOD set                              500 images
Forensic evaluation bundle              11,500 files
Commercial configurations                    6
Commercial prediction rows              69,000
Commercial metric rows                      186
Proxy prediction rows                    40,500
Proxy OOD rows                            7,500
Thesis XAI cases                              5
Thesis XAI assets                            20
```

Proxy OOD accounting:

```text
500 unique OOD images × 5 folds = 2,500 predictions per architecture
2,500 × 3 architectures = 7,500 prediction rows
```

## Public Governance Structure

```text
docs/artifact/THESIS_ARTIFACT.md
docs/artifact/ARTIFACT_EVALUATION.md
docs/artifact/REPOSITORY_MAP.md
docs/artifact/DATA_DICTIONARY.md
docs/artifact/ENVIRONMENT.md
docs/artifact/REPRODUCIBILITY.md
docs/artifact/DATA_ACCESS.md
docs/artifact/ARCHIVE_SNAPSHOT.md
.github/SECURITY.md
docs/maintenance/ACADEMIC_REPOSITORY_AUDIT.md
docs/maintenance/RELEASE_CHECKLIST.md
```

## Validation Entry Points

```bash
python forensic_tools/scripts/validate_public_extract_equivalence.py \
  --source evaluation/forensic_tools/normalized_predictions.csv \
  --metrics results/metrics/forensic_tools_metrics.csv \
  --force

python explainability/scripts/validate_chapter5_xai_artifacts.py \
  --strict-thesis-text

python results/scripts/23_validate_results_artifacts.py
python results/scripts/24_audit_reporting_asset_usage.py --strict
python tools/latex/audit_latex_images_used.py --main docs/LatexThesis/main.tex
```

Windows helper:

```powershell
.\tools\tasks.ps1 audit-all
```

## Current Strengths

The repository now provides:

- a clear forensic robustness scope;
- a numbered and documented experimental pipeline;
- human-in-the-loop dataset selection;
- checkpoint, image, and bundle traceability;
- transparent proxy evaluation;
- black-box commercial normalization with exact public equivalence;
- a minimal thesis-focused XAI layer;
- controlled data governance;
- protected historical preservation;
- source-level English and Italian thesis trees;
- lightweight CI and local audit utilities.

## Remaining Archival Considerations

These are not methodological blockers:

- optional creation of a fully pinned environment lock generated from the verified working environment;
- local execution and review of the new result and asset validators;
- independent final LaTeX compilation and log inspection;
- creation of the official thesis release tag after owner approval;
- optional Zenodo or institutional DOI archival.

## Archival Readiness Checklist

- [x] root README reflects the final frozen scope;
- [x] governance documentation is organized under `docs/` and `.github/`;
- [x] image corpora are excluded from current `main`;
- [x] raw commercial exports are excluded from current `main`;
- [x] canonical sanitized commercial predictions are committed;
- [x] decision and metric equivalence are recorded;
- [x] XAI public artifacts are minimized to thesis-selected cases;
- [x] historical state is preserved by protected branch and annotated tag;
- [x] citation metadata and MIT license are present;
- [x] controlled access is documented;
- [x] LaTeX build products are ignored;
- [x] CI guards the final file layout and canonical prediction profile;
- [ ] local execution of all final validators after the last pull;
- [ ] final English and Italian LaTeX build check;
- [ ] optional environment lock;
- [ ] final release and DOI archival after approval.

## Conclusion

The repository is aligned with the frozen MSc thesis as a controlled academic research artifact. Remaining work concerns final local verification and release management rather than experimental redesign.
