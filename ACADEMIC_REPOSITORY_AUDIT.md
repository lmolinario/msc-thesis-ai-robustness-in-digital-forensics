# Academic Repository Audit

This document records the repository-level review performed to align the project with expectations for an MSc research artifact and to move it toward a more research-grade, PhD-style standard of reproducibility, auditability, and controlled data governance.

Repository:

```text
lmolinario/msc-thesis-ai-robustness-in-digital-forensics
```

Thesis title:

```text
Evaluating the Robustness of AI-based Forensic Tools under Adversarial and Anti-Forensic Attacks
```

---

## 1. Review criteria

The repository is assessed against the following academic and research-artifact criteria:

1. clear research scope and contribution;
2. traceable experimental workflow;
3. documented dataset access policy;
4. reproducible execution entry points;
5. explicit separation between public artifacts and controlled raw data;
6. versioned dependencies and environment documentation;
7. clear citation metadata;
8. license and security policy;
9. consistency between README files, scripts, thesis text, and final experimental perimeter;
10. absence of exposed secrets, private links, credentials, or proprietary files.

---

## 2. Current strengths

The repository satisfies several important research-artifact requirements:

- clear thesis scope and methodological framing in the root README;
- structured numbered pipeline scripts;
- frozen dataset manifests;
- human-in-the-loop selection protocol;
- hash-based traceability;
- proxy-model evaluation layer;
- adversarial and anti-forensic perturbation layer;
- blind forensic evaluation bundle;
- commercial-tool normalization layer;
- metrics exported as CSV/JSON artifacts;
- LaTeX thesis source included;
- MIT license included;
- local secret and proprietary artifact ignores hardened in `.gitignore`;
- controlled raw data access documented in `DATA_ACCESS.md`;
- citation metadata provided in `CITATION.cff`;
- security handling described in `SECURITY.md`;
- safe environment-variable template provided in `.env.example`.

---

## 3. Changes applied during this review

| Area | Change | Status |
|---|---|---|
| Citation | Added `CITATION.cff` for formal repository citation metadata. | Completed |
| Data governance | Added `DATA_ACCESS.md` describing controlled raw dataset access. | Completed |
| Security | Added `SECURITY.md` for secret/data exposure handling. | Completed |
| Environment | Added `.env.example` documenting required environment variables without secrets. | Completed |
| Raw bundle access | Replaced hardcoded dataset URL with local `FAIRLAB_RAW_DATASET_BUNDLE_URL`. | Completed |
| Secret prevention | Hardened `.gitignore` against `.env`, session files, tokens, keys, credential JSON files, and proprietary forensic artifacts. | Completed |
| Root README | Rewritten as a concise academic research-artifact overview aligned with the final thesis perimeter. | Completed |
| Reproducibility guide | Rewritten to remove historical tool names and document the final experimental perimeter. | Completed |
| Dataset README | Rewritten to align dataset documentation with the final frozen dataset, bundle, and commercial-tool perimeter. | Completed |

---

## 4. Final commercial-tool perimeter

The public documentation is now aligned to the following final commercial / black-box evaluation perimeter:

| Tool | Version / module | Final status |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Included |
| Excire Foto 2025 | 4.1.5 | Included as standalone AI-assisted semantic retrieval |
| Cellebrite Inseyets | 10.9 | Included |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE 1.18.0 | Included |

Excluded from the final experimental perimeter:

```text
Oxygen Forensic Detective
Autopsy
X-Ways Forensics
```

These tools should appear only as historical, preliminary, or non-final references where contextually necessary.

---

## 5. Remaining consistency issues

The remaining issues are limited and do not require architectural redesign.

### 5.1 Dependency reproducibility can be strengthened

`requirements.txt` contains a mixture of pinned and unpinned dependencies. This is acceptable for a working thesis repository but not ideal for archival reproducibility.

Recommended future improvement:

```text
requirements.txt       = human-maintained main dependency list
requirements-lock.txt  = fully pinned frozen environment generated from the working environment
```

### 5.2 Final LaTeX consistency check remains external to this audit

The repository documentation has been aligned, but the final thesis source should still be compiled and checked for:

- unresolved references;
- undefined citations;
- glossary/acronym issues;
- missing figures;
- stale textual references to excluded tools.

### 5.3 Final release/tag is still pending

After thesis freeze, create a final archival commit or release tag, for example:

```text
v1.0-thesis-submission
```

Do this only after the thesis text, metrics, figures, and repository documentation are frozen.

---

## 6. Recommended final academic structure

The repository now follows this public-facing research-artifact structure:

```text
README.md                         Main overview and current status
CITATION.cff                      Citation metadata
LICENSE                           Software license
DATA_ACCESS.md                    Controlled dataset access policy
SECURITY.md                       Secret/data exposure policy
REPRODUCIBILITY.md                End-to-end reproducibility guide
ACADEMIC_REPOSITORY_AUDIT.md      Repository-level review and remaining issues
.env.example                      Safe environment variable template
requirements.txt                  Main dependency list
requirements-lock.txt             Optional frozen dependency snapshot
datasets/                         Dataset manifests, scripts, and controlled access bootstrap
attacks/                          Generated perturbation artifacts and manifests
models/                           Proxy-model scripts, checkpoints, reports
evaluation/                       Proxy and commercial-tool evaluation outputs
explainability/                   Integrated Gradients/XAI workflow
docs/                             Thesis source and supporting documentation
results/                          Metric tables and final outputs
forensic_tools/                   Normalized commercial-tool export structure
progress/                         Logs, notes, and milestones
```

---

## 7. Archival readiness checklist

Before final thesis submission or public archival release, verify:

- [x] root README reflects the final thesis status;
- [x] `REPRODUCIBILITY.md` reflects the final tool perimeter;
- [x] `datasets/README.md` reflects the final tool perimeter;
- [x] no hardcoded raw dataset links remain in the public bootstrap script;
- [x] no `.env`, session, token, or credential files are intentionally tracked;
- [x] no proprietary forensic case files should be tracked under the public artifact policy;
- [x] citation metadata are available through `CITATION.cff`;
- [x] controlled data access is documented through `DATA_ACCESS.md`;
- [x] security/data exposure handling is documented through `SECURITY.md`;
- [ ] final metrics should be regenerated or revalidated after the last script change;
- [ ] LaTeX thesis should compile without unresolved references or undefined citations;
- [ ] all figures and tables referenced in the thesis should exist;
- [ ] final commit or release tag should be created after thesis freeze.

---

## 8. Review conclusion

The repository is now substantially aligned with the expectations of an academic MSc research artifact and approaches the standard of a controlled, research-grade experimental repository.

The main remaining work is final validation rather than redesign: freeze the environment more precisely if needed, compile the thesis, revalidate metrics after final edits, and create an archival tag after the thesis submission state is reached.
