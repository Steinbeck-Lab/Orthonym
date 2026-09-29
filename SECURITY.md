# Security Policy

## Supported versions

The latest released version of Orthonym receives security fixes.

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ |

## Reporting a vulnerability

Please report security vulnerabilities privately rather than opening a public issue.

Use GitHub's **[private vulnerability reporting](https://github.com/Steinbeck-Lab/Orthonym/security/advisories/new)**
("Report a vulnerability" under the repository's Security tab) to disclose the issue
confidentially.

Please include:

- a description of the vulnerability and its impact,
- steps to reproduce (a minimal SMILES input or command is ideal),
- the Orthonym version and your environment (Python and Java versions).

We will acknowledge your report, investigate, and coordinate a fix and disclosure timeline
with you.

## Scope note

Orthonym runs two unmodified third-party Java programs, OPSIN and centres. They are not part of
Orthonym: it downloads them from their official releases and checks each file against a pinned
SHA-256 before use (see [`NOTICE`](NOTICE)). A jar given by `ORTHONYM_OPSIN_JAR` or
`ORTHONYM_CENTRES_JAR` is used without that check.

Report a problem in OPSIN at <https://github.com/dan2097/opsin> and in centres at
<https://github.com/SiMolecule/centres>. We update the pinned versions when a fixed release is out.
