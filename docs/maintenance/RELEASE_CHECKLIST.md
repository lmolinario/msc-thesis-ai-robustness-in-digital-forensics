# Release Checklist

This checklist should be completed before creating the official thesis-artifact release.

Recommended release name:

```text
v1.0.0-thesis-freeze
```

Recommended release title:

```text
MSc Thesis Research Artifact — Final Frozen Version
```

---

## 1. Local Git State

From the repository root:

```powershell
git fetch origin
git status -sb
git log --oneline origin/main..HEAD
```

Expected:

```text
## main...origin/main
```

and no local commits ahead of `origin/main`.

---

## 2. Repository Audit

Run:

```powershell
.\tasks.ps1 audit-all
```

At minimum, verify:

```powershell
.\tasks.ps1 check-json
.\tasks.ps1 check-python-syntax
.\tasks.ps1 check-text-guards
```

If the thesis has just been compiled, also run:

```powershell
.\tasks.ps1 check-thesis-log
```

---

## 3. Documentation Check

Verify that the following documents are present and coherent:

```text
README.md
THESIS_ARTIFACT.md
REPOSITORY_MAP.md
ARTIFACT_EVALUATION.md
DATA_DICTIONARY.md
ENVIRONMENT.md
REPRODUCIBILITY.md
DATA_ACCESS.md
SECURITY.md
ACADEMIC_REPOSITORY_AUDIT.md
CHANGELOG.md
CITATION.cff
```

---

## 4. Thesis Source Check

Official thesis source:

```text
docs/LatexThesis/
```

Check that:

- `main.tex` compiles locally;
- bibliography references are resolved;
- glossary/acronym warnings are under control;
- generated temporary files are not committed;
- the final PDF, if distributed, is attached as a release asset rather than committed by default.

---

## 5. Data and Security Check

Confirm that the repository does not contain:

```text
.env
credentials
API keys
private URLs
raw controlled datasets
licensed commercial-tool databases
unnecessary proprietary case files
```

Confirm that public documentation does not reintroduce stale or excluded experimental references.

---

## 6. Release Asset Recommendation

Do not commit generated PDFs or archives unless intentionally required.

Preferred release assets:

```text
thesis-final.pdf
artifact-checksums.sha256
repository-audit-summary.md
```

The Git tree should remain focused on source code, manifests, normalized outputs, metrics, and thesis source.

---

## 7. GitHub Release

Create a GitHub release from `main` after final checks.

Recommended tag:

```text
v1.0.0-thesis-freeze
```

Recommended release title:

```text
MSc Thesis Research Artifact — Final Frozen Version
```

Recommended release notes:

```text
Final frozen research artifact supporting the MSc thesis "Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks".

This release includes the final thesis source, scripts, manifests, normalized commercial-tool outputs, metric summaries, XAI case-study material, and repository governance documentation.

Raw images, controlled-access datasets, proprietary forensic software, and licensed tool internals are not redistributed.
```

---

## 8. Zenodo / DOI

After creating the GitHub release, archive the release through Zenodo or an institutional repository.

After DOI assignment:

1. update `CITATION.cff` with DOI and release date;
2. add DOI badge to `README.md`;
3. update `CHANGELOG.md` release section from planned to released;
4. optionally cite the repository artifact in the thesis or appendix.

---

## 9. Post-Release Rule

After the official thesis-artifact release, avoid silent experimental changes.

Allowed post-release changes:

- citation metadata corrections;
- DOI updates;
- documentation typo fixes;
- release-note clarifications;
- security corrections.

Substantive experimental changes should produce a new versioned release.
