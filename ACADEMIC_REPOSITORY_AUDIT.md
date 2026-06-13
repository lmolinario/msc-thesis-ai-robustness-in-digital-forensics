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

The repository already satisfies several important research-artifact requirements:

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
- local secret and proprietary artifact ignores hardened in `.gitignore`.

---

## 3. Changes already applied during this review

The following repository-level improvements were applied:

| Area | Change |
|---|---|
| Citation | Added `CITATION.cff` for formal repository citation metadata. |
| Data governance | Added `DATA_ACCESS.md` describing controlled raw dataset access. |
| Security | Added `SECURITY.md` for secret/data exposure handling. |
| Environment | Added `.env.example` documenting required environment variables without secrets. |
| Raw bundle access | Replaced hardcoded dataset URL with local `FAIRLAB_RAW_DATASET_BUNDLE_URL`. |
| Secret prevention | Hardened `.gitignore` against `.env`, session files, tokens, keys, credential JSON files, and proprietary forensic artifacts. |

---

## 4. Remaining consistency issues

The following issues should be resolved before considering the repository fully thesis-final or archival-ready.

### 4.1 Reproducibility documentation still contains historical tool names

`REPRODUCIBILITY.md` contains an older commercial-tool perimeter mentioning historical or excluded tools such as:

```text
X-Ways Forensics / Excire
Cellebrite UFED
Oxygen Forensic Detective
```

This should be updated to the final perimeter:

```text
Magnet AXIOM / Magnet.AI, version 10.1.0.48673
Excire Foto 2025, version 4.1.5
Cellebrite Inseyets, version 10.9
Magnet Griffeye x64, version 26.2.108, with T3K CORE v1.18.0
```

Oxygen, Autopsy, and X-Ways should be excluded from the final experimental perimeter, except where mentioned historically or as non-final candidates.

### 4.2 Dataset README contains older commercial-tool status

`datasets/README.md` still includes a historical commercial-tool section indicating that Cellebrite is pending and referring to X-Ways / Excire.

This should be aligned with the current finalized state:

- Magnet AXIOM / Magnet.AI completed and normalized;
- Excire Foto 2025 completed as standalone AI-assisted semantic retrieval;
- Cellebrite Inseyets 10.9 completed and normalized;
- Magnet Griffeye / T3K CORE completed and normalized;
- Oxygen and Autopsy excluded;
- X-Ways not part of the final experimental perimeter.

### 4.3 Dependency reproducibility can be strengthened

`requirements.txt` contains a mixture of pinned and unpinned dependencies. This is acceptable for a working thesis repository but not ideal for archival reproducibility.

Recommended future improvement:

```text
requirements.txt       = human-maintained main dependency list
requirements-lock.txt  = fully pinned frozen environment generated from the working environment
```

### 4.4 README should eventually link the new policy files

The root README should include a short `Research Artifact Governance` or `Repository Policies` section linking:

```text
CITATION.cff
DATA_ACCESS.md
SECURITY.md
REPRODUCIBILITY.md
.env.example
```

This is not functionally necessary, but it improves readability and reviewer navigation.

---

## 5. Recommended final academic structure

The repository should converge toward this public-facing structure:

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

## 6. Archival readiness checklist

Before final thesis submission or public archival release, verify:

- [ ] root README reflects the final thesis status;
- [ ] `REPRODUCIBILITY.md` reflects the final tool perimeter;
- [ ] `datasets/README.md` reflects the final tool perimeter;
- [ ] no hardcoded raw dataset links remain;
- [ ] no `.env`, session, token, or credential files are tracked;
- [ ] no proprietary forensic case files are tracked;
- [ ] final metrics are generated from scripts rather than manual transcription;
- [ ] LaTeX thesis compiles without unresolved references or undefined citations;
- [ ] all figures and tables referenced in the thesis exist;
- [ ] final commit or release tag is created after thesis freeze.

---

## 7. Review conclusion

The repository is already substantially stronger than a typical MSc code dump because it includes a documented end-to-end workflow, hash-based traceability, frozen manifests, normalized commercial-tool outputs, and thesis source integration.

The main remaining work is not architectural redesign. It is final consistency hardening: update stale historical documentation, freeze dependencies more precisely, and ensure that all public-facing files describe the same final experimental perimeter.
