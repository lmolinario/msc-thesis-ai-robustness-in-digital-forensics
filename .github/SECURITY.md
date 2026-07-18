# Security and Data-Exposure Policy

This repository is a public academic research artifact. It contains code,
documentation, manifests, sanitized prediction-level outputs, frozen metrics, and
thesis material related to AI robustness evaluation in Digital/Computer Forensics.

## Sensitive material that must not be committed

Do not commit:

- API keys, tokens, passwords, private credentials, or license files;
- `.env` files or scripts containing secrets or temporary signed URLs;
- Telegram session files or authentication artifacts;
- private direct-download URLs or authorization-bearing dataset links;
- proprietary forensic-tool databases, cases, installers, or complete raw exports;
- raw third-party datasets when redistribution is not explicitly permitted;
- operational evidence, investigative case material, or personal data;
- local absolute paths when they reveal unnecessary user, device, or storage details.

An access-controlled landing page may be documented when its purpose is to allow
researchers to request access. Such a page must not expose a reusable private download
token or bypass the access-control process.

The repository `.gitignore` includes rules for common local credentials, session
files, proprietary tool material, raw corpora, LaTeX build products, and local staging
outputs. Contributors remain responsible for inspecting staged changes.

## Public artifacts and minimization boundary

The curated `main` branch may distribute:

- anonymized bundle identifiers;
- source-code and methodological documentation;
- frozen manifests and metric tables;
- the canonical sanitized commercial-tool prediction table;
- validated tool-specific public extracts;
- thesis-ready reporting and XAI assets selected for the final document.

The complete commercial-tool raw exports and image corpora are not distributed on
current `main`. Historical preservation does not grant permission to redistribute
third-party or proprietary content.

## Reporting a security or data-exposure issue

When a potential secret, private data link, proprietary export, raw-data disclosure,
or unintended personal-data exposure is found, contact the repository maintainer
through GitHub without reproducing the sensitive value in a public issue.

Provide only a minimal description and request a private communication channel when
necessary.

## Supported scope

This policy covers accidental disclosure or misuse of repository artifacts, including:

- credential leakage;
- private or signed data-link exposure;
- raw dataset redistribution risk;
- proprietary forensic-tool export disclosure;
- local-path and personal-data leakage;
- misleading documentation that could cause improper handling of controlled data.

It does not provide operational security support for third-party tools, commercial
forensic platforms, or external services used during the research workflow.

## Local checks before pushing

```bash
git status --short
git diff --cached
```

Also run the repository validators relevant to the modified area. Secret scanners such
as `gitleaks` may be used as an additional control.
