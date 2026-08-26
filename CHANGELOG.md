# Changelog

All notable changes to Orthonym are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-26

First public release.

### Added
- Deterministic, rule-based SMILES → IUPAC name generation targeting Preferred IUPAC
  Names (PINs) per the IUPAC 2013 recommendations.
- Coverage of acyclic and cyclic compounds, functional-group derivatives, heterocycles
  (Hantzsch–Widman), fused, bridged (von Baeyer), and spiro ring systems.
- Stereochemistry (R/S, E/Z) via a high-accuracy CIP engine.
- Round-trip validation against OPSIN: a name is not emitted for the wrong structure.
- Public Python API (`name_compound`, `name_with_tree`, `Orthonym`) and a command-line
  interface (`orthonym`, `python -m orthonym`).
- OPSIN and CIP jars shipped in the repository and invoked as external Java processes
  (a Java runtime is required for validation); licenses documented in NOTICE.
- Accuracy harness and fixed evaluation splits under the project tooling.
