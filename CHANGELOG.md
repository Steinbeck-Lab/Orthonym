# Changelog

All notable changes to Orthonym are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.0...v1.0.0) (2026-09-29)


### Features

* more preferred IUPAC names, fewer lost names, two wrong-name classes closed, honest labels ([e3a9a63](https://github.com/Steinbeck-Lab/Orthonym/commit/e3a9a63b1e69dea3b683f38977d8bbdb855854fb))
* nomenclature coverage and correctness updates ([931fe63](https://github.com/Steinbeck-Lab/Orthonym/commit/931fe63e5c1222958aba9fef92c31320a4229a21))
* nomenclature coverage and correctness updates ([1d835f6](https://github.com/Steinbeck-Lab/Orthonym/commit/1d835f61c4d9756d9d07bc62ecf51af7072a3d39))
* nomenclature coverage and correctness updates ([327e28c](https://github.com/Steinbeck-Lab/Orthonym/commit/327e28c43e58881d85b0a036ae33419c5eaa2f4b))
* radical names, safer salt names, stable numbering and Blue Book PIN spelling fixes ([34ef0d3](https://github.com/Steinbeck-Lab/Orthonym/commit/34ef0d3af8c4727790941323a6907efe52d23a18))
* systematic names for N-substituted amino acids, long alkanes named again, and many preferred-name fixes ([68f50d1](https://github.com/Steinbeck-Lab/Orthonym/commit/68f50d1284c7ae860d51804f0c7661a408f6a778))


### Bug Fixes

* every shown name passes its own round trip, N-substituted amino acids get systematic names, and ChEBI losses are restored ([0be43e0](https://github.com/Steinbeck-Lab/Orthonym/commit/0be43e0e5d12344a3ba3c14e72eee3b16e228414))
* honest tier labels, faster naming of very large molecules, and every paper-named ChEBI structure named again ([eb25c33](https://github.com/Steinbeck-Lab/Orthonym/commit/eb25c3337e22020030f2eac6164ea85569347fa4))
* nomenclature correctness updates ([57eaa92](https://github.com/Steinbeck-Lab/Orthonym/commit/57eaa92ccb972cd52e6a654037230636d16dea92))


### Documentation

* credits back in the README under Built on ([2153b79](https://github.com/Steinbeck-Lab/Orthonym/commit/2153b7930e420f1b61ab26b3aa757b7e9b0c26df))
* lighter README, with guide pages for how it works, declines and accuracy ([e4f4ae3](https://github.com/Steinbeck-Lab/Orthonym/commit/e4f4ae382d538fe4f62510c2ab8198977254713a))
* project links in the README, the citation file, the package metadata and the documentation site ([d2f3118](https://github.com/Steinbeck-Lab/Orthonym/commit/d2f3118ef24ad9bd39dd9dc65b3b3393bbe9f7d2))
* the documentation site shows the Orthonym mark and the web app's credit, and the README links the docs and the web app ([7fc9231](https://github.com/Steinbeck-Lab/Orthonym/commit/7fc9231a7e8cb00efd45aba2d0cefcbd62f8fc78))
* the Orthonym documentation site, with API docstrings and command-line help in plain language ([6b3cb01](https://github.com/Steinbeck-Lab/Orthonym/commit/6b3cb017cda588e946bdd1e35289811063e772ff))

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
