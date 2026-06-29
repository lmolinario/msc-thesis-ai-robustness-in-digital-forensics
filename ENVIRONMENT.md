# Environment Notes

This document summarizes the execution environment assumptions for the MSc thesis research artifact.

The repository is not distributed as a software package. It is a research artifact containing scripts, manifests, metrics, normalized outputs, and thesis source.

---

## Tested Context

The project has been developed and executed primarily in a local research environment with:

- Python virtual environment;
- PowerShell / Windows-oriented local workflow;
- GPU or CPU execution depending on the stage;
- licensed commercial forensic tools for black-box export generation;
- LaTeX environment for thesis compilation.

Exact local paths, usernames, private drives, credentials, and controlled-access URLs are intentionally not part of the public repository.

---

## Python Dependencies

The canonical dependency list is:

```text
requirements.txt
```

Create and activate a local virtual environment before running scripts.

Example PowerShell workflow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Environment Variables

Use:

```text
.env.example
```

as the safe template for local configuration.

Do not commit:

```text
.env
*.env
private URLs
API keys
credentials
session cookies
licensed software keys
```

Raw dataset restoration, if permitted, must use controlled local environment variables and not hard-coded public links.

---

## Compute Notes

Some stages can be inspected or rerun with limited compute, while others require more substantial resources.

| Stage | Compute expectation |
|---|---|
| Manifest inspection | Low |
| JSON/CSV audit | Low |
| Reporting asset generation | Low to moderate |
| Proxy-model evaluation | Moderate |
| Proxy-model training | Moderate to high |
| Iterative adversarial attack generation | High |
| Commercial-tool evaluation | Requires licensed tools, not just compute |

---

## Commercial Forensic Tools

The following tools are part of the final black-box perimeter:

```text
Magnet AXIOM / Magnet.AI 10.1.0.48673
Excire Foto 2025 4.1.5
Cellebrite Inseyets 10.9
Magnet Griffeye x64 26.2.108 with T3K CORE v1.18.0
```

Their internal AI models and proprietary resources are not reproduced by this repository. Only observable exports and normalized outputs are part of the public artifact when appropriate.

Full commercial-tool reruns require:

- licensed software;
- compatible workstation environment;
- controlled forensic evaluation bundle;
- tool-specific import/export workflow;
- post-export normalization.

---

## LaTeX Environment

The official thesis source is:

```text
docs/LatexThesis/
```

Compilation generates auxiliary files such as:

```text
main.aux
main.log
main.out
main.toc
main.acn
main.acr
main.alg
main.pdf
```

These generated files should not be committed unless explicitly selected as a final release asset. The preferred repository source of truth remains the LaTeX source, bibliography, acronym file, figures, and tracked thesis assets.

---

## Recommended Non-Destructive Checks

From the repository root:

```powershell
git status -sb
python -m compileall datasets models evaluation explainability results
```

For thesis log checks after compiling LaTeX:

```powershell
cd docs\LatexThesis
Select-String -Path .\main.log -Pattern "Undefined references","Citation.*undefined","LaTeX Error","Package glossaries Warning"
```

---

## Reproducibility Boundary

The public repository supports:

- structural audit;
- script inspection;
- manifest inspection;
- metric inspection;
- thesis-source review;
- normalized commercial-tool output inspection.

The public repository alone does not provide:

- unrestricted raw image access;
- commercial forensic-tool licenses;
- proprietary AI models;
- private source credentials;
- full rerun capability for every stage.

This is a controlled-access research artifact, not an unrestricted data release.
