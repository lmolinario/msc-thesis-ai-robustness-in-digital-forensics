# Academic Repository Audit

This document records the repository-level review performed to align the project with the frozen MSc thesis artifact and to preserve a research-grade standard of reproducibility, auditability, and controlled data governance.

Repository:

```text
lmolinario/msc-thesis-ai-robustness-in-digital-forensics
```

Thesis title:

```text
Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and Anti-Forensic Attacks
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

The repository satisfies the main research-artifact requirements:

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

## 3. Final commercial-tool perimeter

The public documentation is aligned to the following final commercial / black-box evaluation perimeter:

| Tool | Version / module | Final status |
|---|---|---|
| Magnet AXIOM / Magnet.AI | 10.1.0.48673 | Included |
| Excire Foto 2025 | 4.1.5 | Included as standalone AI-assisted semantic retrieval |
| Cellebrite Inseyets | 10.9 | Included |
| Magnet Griffeye / T3K CORE | Griffeye x64 26.2.108, T3K CORE 1.18.0 | Included |

---

## 4. Documentation changes applied during final alignment

| Area | Change | Status |
|---|---|---|
| Root README | Marked thesis reporting as completed and frozen; aligned title capitalization and final tool perimeter. | Completed |
| Citation | Updated `CITATION.cff` to use the final thesis title. | Completed |
| Documentation README | Removed private editor link and marked `docs/LatexThesis/` as the final frozen thesis source. | Completed |
| Dataset README | Marked dataset workflow as frozen and removed remaining-progress wording. | Completed |
| Attacks README | Removed stale historical wording and retained only the final tool perimeter. | Completed |
| Evaluation README | Retained only the final included tool perimeter. | Completed |
| Forensic tools README | Clarified standalone Excire evaluation and retained only the final tool perimeter. | Completed |
| Results README | Reinforced final reporting rules and retained only final evaluated tools. | Completed |

---

## 5. Remaining archival considerations

The thesis is treated as frozen by the author. The following items are repository-archival considerations rather than methodological blockers:

### 5.1 Dependency reproducibility

`requirements.txt` contains a mixture of pinned and unpinned dependencies. This is acceptable for a working thesis repository but not ideal for long-term archival reproducibility.

Optional future improvement:

```text
requirements.txt       = human-maintained main dependency list
requirements-lock.txt  = fully pinned frozen environment generated from the working environment
```

### 5.2 Independent LaTeX build verification

The GitHub-level audit confirms documentation consistency, but it does not independently compile the LaTeX project. The submitted thesis source remains under:

```text
docs/LatexThesis/
```

### 5.3 Final release/tag

After the final repository state is accepted, create an archival release tag, for example:

```text
v1.0-thesis-submission
```

---

## 6. Recommended final academic structure

The repository follows this public-facing research-artifact structure:

```text
README.md                         Main overview and frozen thesis status
CITATION.cff                      Citation metadata
LICENSE                           Software license
DATA_ACCESS.md                    Controlled dataset access policy
SECURITY.md                       Secret/data exposure policy
REPRODUCIBILITY.md                End-to-end reproducibility guide
ACADEMIC_REPOSITORY_AUDIT.md      Repository-level review and remaining archival notes
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

Before public archival release, verify:

- [x] root README reflects the frozen thesis status;
- [x] `REPRODUCIBILITY.md` describes the reproducibility workflow;
- [x] `datasets/README.md` reflects the final dataset, bundle, and tool perimeter;
- [x] `attacks/README.md` reflects the final perturbation and tool-normalization status;
- [x] `evaluation/README.md` reflects the final tool perimeter;
- [x] `forensic_tools/README.md` reflects the final commercial-tool perimeter;
- [x] `results/README.md` reflects the final metric/reporting perimeter;
- [x] no hardcoded raw dataset links remain in the public bootstrap script;
- [x] no `.env`, session, token, or credential files are intentionally tracked;
- [x] citation metadata are available through `CITATION.cff`;
- [x] controlled data access is documented through `DATA_ACCESS.md`;
- [x] security/data exposure handling is documented through `SECURITY.md`;
- [ ] optional `requirements-lock.txt` can be generated for long-term environment freeze;
- [ ] optional final release tag can be created after the repository owner approves this state.

---

## 8. Review conclusion

The repository is aligned with the frozen MSc thesis state as a controlled academic research artifact.

The main remaining actions are optional archival hardening: generating a lock file from the actual working environment and creating a final release tag after owner approval.
