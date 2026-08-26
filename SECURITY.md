# Security Policy

## Supported versions

The latest released version of Orthonym receives security fixes.

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ |

## Reporting a vulnerability

Please report security vulnerabilities privately rather than opening a public issue.

Use GitHub's **[private vulnerability reporting](https://github.com/Kohulan/Orthonym/security/advisories/new)**
("Report a vulnerability" under the repository's Security tab) to disclose the issue
confidentially.

Please include:

- a description of the vulnerability and its impact,
- steps to reproduce (a minimal SMILES input or command is ideal),
- the Orthonym version and your environment (Python and Java versions).

We will acknowledge your report, investigate, and coordinate a fix and disclosure timeline
with you.

## Scope note

Orthonym invokes a bundled OPSIN Java process to validate names. Vulnerabilities in OPSIN
itself should be reported upstream at <https://github.com/dan2097/opsin>; we will update the
bundled version as needed.
