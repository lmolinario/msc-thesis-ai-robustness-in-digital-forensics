# Security Policy

This repository is a public academic research artifact. It contains code, documentation, manifests, normalized metrics, and thesis material related to AI robustness evaluation in Digital/Computer Forensics.

---

## Sensitive material that must not be committed

Do not commit:

- API keys, tokens, passwords, or private credentials;
- `.env` files or shell scripts containing private URLs;
- Telegram session files or authentication artifacts;
- private Google Drive links for controlled-access data bundles;
- proprietary forensic-tool databases, cases, or export containers;
- temporary signed download URLs;
- raw datasets when redistribution is not explicitly allowed;
- personal contact information beyond what is intentionally published in the repository profile or documentation.

The `.gitignore` file includes rules for common local secrets, credentials, session files, and proprietary forensic-tool artifacts, but contributors remain responsible for checking staged changes before committing.

---

## Reporting a security or data-exposure issue

If you find a potential exposed secret, private dataset link, proprietary forensic-tool artifact, or unintended raw-data disclosure, contact the repository maintainer through the public GitHub repository interface.

Do not open a public issue containing the exposed secret or private URL. Instead, provide a minimal description and request a private communication channel if needed.

---

## Supported scope

This policy covers accidental disclosure or misuse of repository artifacts, including:

- credential leakage;
- private data-link exposure;
- raw dataset redistribution risk;
- forensic-tool export disclosure;
- misleading documentation that could cause improper use of controlled data.

It does not provide operational security support for third-party tools, commercial forensic platforms, or external services used during the research workflow.

---

## Local handling recommendation

Before pushing changes, run at least one local secret scan and inspect staged files manually:

```bash
git status --short
git diff --cached
```

Recommended additional tools include secret scanners such as `gitleaks` or equivalent repository-audit utilities.
