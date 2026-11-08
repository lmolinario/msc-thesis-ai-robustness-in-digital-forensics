# Academic Repository Audit

This document records the repository-level review performed to align the public
project with the frozen MSc thesis artifact.

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
10. absence of exposed secrets, raw controlled corpora, and unnecessary
    proprietary exports;
11. correct final chapter structure and source-file map;
12. consistent Max-P terminology without calibrated-confidence claims;
13. explicit separation between direct attacks, transferred attacks,
    model-agnostic transformations, and anti-forensic conditions;
14. denominator and dependency transparency for OOD and derived variants.

## Final Thesis Structure

```text
Chapter 1 - Introduction
Chapter 2 - Background
Chapter 3 - State of the Art
Chapter 4 - Methodology
Chapter 5 - Implementation and Experimental Setup
Chapter 6 - Experimental Results and Operational Robustness Analysis
Chapter 7 - Conclusions, Limitations, and Future Work
Appendix
```

Frozen paths and identifiers containing `chapter5` or `chapter_5` are historical
artifact names created before the final reorganization. They are preserved for
reproducibility and must not be interpreted as current chapter assignments.

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
| LaTeX | A single authoritative seven-chapter thesis source retained; build outputs ignored |
| Defense material | Maintained in the separate `Molinario_Thesis_Defense_2026` repository and excluded from this repository's scientific release assets |
| Chapter references | Documentation aligned to Chapters 5, 6, and 7; historical names explained |
| Probability terminology | Proxy outputs documented as Max-P; historical schema names preserved |
| Public commercial boundary | Sanitized canonical table distinguished from excluded raw exports |
| Local path hygiene | Prepared summary sanitized; generator and CI hardened |
| Root structure | Governance docs under `docs/`; utilities under `tools/`; security policy under `.github/` |
| CI | Required paths, JSON, Python syntax, canonical results, stale references, and local-path leakage checked |

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
Frozen source population                   1,500 images
Binary subset                              1,000 images
Unique OOD set                               500 images
Forensic evaluation bundle               11,500 files
Commercial configurations                     6
Commercial prediction rows               69,000
Commercial metric rows                       186
Proxy prediction rows                     40,500
Proxy OOD rows                             7,500
Thesis XAI cases                               5
Thesis XAI assets                             20
```

Proxy OOD accounting:

```text
500 unique OOD images × 5 folds = 2,500 predictions per architecture
2,500 × 3 architectures = 7,500 prediction rows
```

The 10,000 perturbed bundle files are derived variants of the same 1,000 binary
source images. They are not independent underlying cases.

## Attack Interpretation

```text
model-dependent attacks targeting EfficientNet-B0:
  fgsm
  one_pixel
  sigma_zero
  superdeepfool

model-agnostic adversarial-style condition:
  color_shift

model-agnostic anti-forensic transformations:
  jpeg_recompression
  resample_resize
  gaussian_blur
  histogram_modification
  contrast_stretching
```

Transfer results describe empirical cross-model effects and do not establish
direct robustness of a non-target model.

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
docs/maintenance/CHAPTER6_FINALIZATION_REPORT.md
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

Kali/Linux helper:

```bash
bash tools/tasks.sh audit-all
```

The helper covers repository structure, JSON, Python syntax, text guards, XAI,
results, reporting assets, LaTeX-image references, and the local thesis log when
present. The commercial public-extract equivalence validator remains an explicit
separate release check.

## Latest Local Validation Record

On **2026-08-04**, immediately before the documentation synchronization recorded
by later documentation-only commits, the local checkout was aligned with
`origin/main` at:

```text
1018f307038f1a27d380df18c2725825cd8ab6b9
```

The following final artifact checks completed successfully at that exact commit:

```text
Public extract equivalence validation
  69,000 identical decision rows
  186 identical metric rows

Results-chapter XAI public-artifact validation
  5 thesis cases
  20 thesis assets
  local path leakage: none

Results artifact validation
  69,000 canonical commercial decisions
  186 commercial metric rows
  40,500 proxy prediction rows
  500 unique OOD images × 5 folds = 2,500 predictions per architecture
  41 reporting-manifest rows
  24 unique reporting asset IDs

Reporting-asset usage audit
  missing reporting outputs: 0
  mismatched existing thesis copies: 0
  13 reporting asset IDs not referenced by the thesis

LaTeX image audit
  21 includegraphics references
  21 resolved references
  0 missing references
  0 duplicate image groups
```

The 13 unreferenced reporting asset IDs are retained generated reporting outputs;
they are not missing assets and do not by themselves indicate an error. Their
presence remains subject to the reporting-layer retention rules documented in
`results/README.md` and `results/scripts/README.md`.

This validation record applies to commit `1018f307...`. Subsequent
**documentation-only** synchronization does not redefine the frozen experimental
artifacts, but the official release must still be compiled and checked at the
exact release commit/tag before archival publication.

The historical Chapter 6 validation report remains independently scoped to
commit `14246371e4551726e02a7e23c5fb76b78591148d` and must not be interpreted as
validation of later commits.

## Current Strengths

The repository provides:

- a clear forensic robustness scope;
- a numbered and documented experimental pipeline;
- human-in-the-loop dataset selection;
- checkpoint, image, and bundle traceability;
- transparent proxy evaluation;
- black-box commercial normalization with exact public equivalence;
- a minimal thesis-focused XAI layer;
- controlled data governance;
- protected historical preservation;
- one authoritative thesis source;
- explicit chapter and terminology boundaries;
- lightweight CI and local audit utilities.

## Remaining Release Considerations

These are not methodological blockers:

- independent final LaTeX compilation and log inspection at the exact release
  commit;
- optional creation of a fully pinned environment lock from a verified working
  environment;
- creation of the official thesis release tag after owner approval;
- optional Zenodo or institutional DOI archival.

## Archival Readiness Checklist

- [x] root README reflects the final frozen scope;
- [x] final seven-chapter structure documented;
- [x] historical `chapter5` names explained without renaming frozen artifacts;
- [x] governance documentation organized under `docs/` and `.github/`;
- [x] image corpora excluded from current `main`;
- [x] raw commercial exports excluded from current `main`;
- [x] canonical sanitized commercial predictions committed;
- [x] decision and metric equivalence recorded;
- [x] XAI public artifacts minimized to thesis-selected cases;
- [x] Max-P terminology aligned while preserving historical schemas;
- [x] OOD and derived-variant denominators documented;
- [x] local absolute path removed from the prepared summary;
- [x] CI hardened against local-path and stale-chapter regressions;
- [x] historical state preserved by protected branch and annotated tag;
- [x] citation metadata and MIT license present;
- [x] controlled access documented;
- [x] LaTeX build products ignored;
- [x] redundant thesis source tree removed;
- [x] defense working material separated from the scientific release perimeter;
- [x] final artifact validators executed locally at commit `1018f307...`;
- [ ] final LaTeX build check at the exact release commit;
- [ ] optional environment lock;
- [ ] final release and DOI archival after approval.

## Conclusion

The repository documentation and public artifact boundaries are aligned with the
final thesis structure and experimental claims. The final artifact validators
were successfully executed locally at commit `1018f307...`; their observed
counts and integrity checks are recorded above. The remaining release work is
limited to final LaTeX compilation and log inspection at the exact release
commit, followed by release management and, optionally, DOI archival. No
experimental redesign is required by this repository audit.

## Pre-Commit Thesis Revision - 2026-11-07

This record applies to an **uncommitted working-tree revision**, not to a new
commit or release. Its base is:

```text
6dbfb2c1564b5a848c1e3e7be59db74cc192e102
```

Earlier local, unpushed revision objects were unavailable in this checkout and
were not found on the remote. Their wording was reconstructed using the
previously revised thesis PDF as a reference; no recovery of the original Git
history or commit identifiers is claimed. No defense-repository edit, commit,
or push was performed in this revision.

### Editorial Scope

- distinguish the white-box proxy role, experimenter access, designated source,
  score-based generation, and empirical transfer, consistently with the GKC
  threat model already present in the base revision;
- restore FAIRLab validation-profile framing, including the compact
  methodological-positioning table, without certification or universal-score
  claims;
- align RQ1 between introduction and conclusions, separating supervised
  perturbation performance from OOD assignment behavior;
- distinguish pre-serialization generation diagnostics, predictions on reloaded
  serialized files, and the clean-correct ASR denominator;
- qualify the SuperDeepFool result table as the internal bounded, single-cycle
  adapter already described in the implementation;
- correct bundle-weighting prose and notes: equal-sized test conditions, but
  `1:5:5` clean/adversarial/anti-forensic branch weights;
- document complete bibliography/glossary compilation and repair stale
  documentation-guard requirements for the removed local slide directory;
- remove one pre-existing minor overfull line through a shorter implementation
  sentence, preserving the API identifier and numeric cutoff.

The title, author information, main chapter structure, bibliography, package
configuration, datasets, checkpoints, attacks, manifests, predictions, metrics,
and numerical reporting figures were preserved. Only the two documentation
guards changed executable repository files; attack and evaluation behavior did
not change.

### Identified Source State

The complete tracked `docs/LatexThesis/` working-tree content was fingerprinted
after the final source edit:

```text
SHA-256: e8924c12f741a1469be5734c74b02ae32ba5dd9a4fe53b228198c2cada9ad676
```

Fingerprint procedure: sort the tracked repository-relative paths under
`docs/LatexThesis/`; for each path concatenate its UTF-8 bytes, one NUL byte,
and the raw SHA-256 digest bytes of its working-tree content; then SHA-256 the
complete concatenation. This identifies the reviewed source independently of
any future commit hash. This audit file is outside that fingerprint's scope.

### Verification Outcomes

| Check | Observed result |
|---|---|
| Public commercial equivalence | 69,000 identical decisions; 186 identical metric rows |
| Frozen result-artifact validator | 40,500 proxy rows; commercial and OOD counts passed |
| Commercial aggregate weighting | Six configurations; 11 conditions of 1,000 binary items each; pooled accuracy equals the condition mean |
| Existing numerical thesis tables | Numeric tokens unchanged in all 75 pre-existing table environments in the edited chapters |
| Existing citations and image paths | Unchanged in the edited source files |
| Main/title/packages/bibliography | Byte-identical to the base revision |
| RQ1/RQ2 alignment | Identical wording in introduction and conclusions |
| XAI validation | Five cases; 20 assets; no local-path leakage |
| Reporting assets | 41 manifest rows; 24 asset IDs; zero missing outputs or mismatched thesis copies |
| LaTeX image audit | 21 references resolved; zero missing references or duplicate image groups |
| Linux documentation guards | Passed; shell syntax check passed |
| PowerShell guard | Updated Python check body passed; native PowerShell was not available |
| Full PDF build | 205 A4 pages; Biber and both glossary indexes completed |
| Final LaTeX log | No errors, undefined references/citations, missing glyphs, overfull boxes, or rerun requests |
| PDF structural/render check | All 205 pages rendered; no unresolved placeholder markers or text spans outside page bounds |

Ordinary underfull-box spacing diagnostics remain in the LaTeX log. The revised
abstract, positioning table, threat-model table, model definitions, generation
diagnostics, ASR/results, weighting explanation, conclusions, and limitations
were visually inspected. The 13 unreferenced generated reporting asset IDs
remain retained outputs, not missing assets.

The compiled pre-commit candidate has SHA-256:

```text
ee9c262ab4be49627cc1f906ba0caee2fb831f8a6b4d99c8d2e1364a30612e44
```

### Remaining Interpretive Boundary

Recounting the frozen records confirms 544 generation-time success flags for
SuperDeepFool versus 69 final serialized-file errors, of which 47 are newly
induced; Sigma-Zero has 1,000 generation flags versus 985 final errors, of which
963 are newly induced. All 1,000 SuperDeepFool records report one executed
outer cycle. These are observations at different representation stages and
with different success accounting, not interchangeable attack-success totals.

This revision does not establish a causal decomposition of those discrepancies.
It does not attribute them entirely to quantization and does not claim that an
ASR denominator change explains the absolute differences. A dedicated
matched-stage rerun would be needed for causal attribution. The thesis metrics
continue to describe the frozen, serialized artifacts actually evaluated.

No retraining, new attack generation, commercial rerun, new statistical
experiment, paper-only population, or replacement numerical result was added.
The prior historical validation records and release checklist status above
remain scoped to their stated commits. Approval of this working-tree revision,
creation of its commit, and the exact release-commit build are still pending.
