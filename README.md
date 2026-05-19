# Orthonym

**Open Structure-To-IUPAC-Name Generator**

A comprehensive rule-based system to generate IUPAC systematic names from molecular structures (SMILES). The first open-source implementation targeting IUPAC 2013 (Blue Book) compliance.

## Why Orthonym?

- **OPSIN** converts IUPAC names → structures (open-source)
- **Orthonym** converts structures → IUPAC names (this project!)
- Commercial tools (ACD/Name, ChemDraw) are proprietary and expensive
- STOUT uses machine learning (non-deterministic, requires training data)

Orthonym is the **first deterministic, rule-based, open-source** solution for structure-to-name conversion.

## Installation

```bash
pip install orthonym
```

Or install from source:

```bash
git clone https://github.com/Kohulan/Orthonym.git
cd Orthonym
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from orthonym import name_compound

# Simple molecules
print(name_compound("CCO"))           # → ethanol
print(name_compound("CC(=O)O"))       # → acetic acid
print(name_compound("c1ccccc1"))      # → benzene

# With stereochemistry
print(name_compound("C/C=C/C"))       # → (E)-but-2-ene
print(name_compound("C[C@H](O)CC"))   # → (2R)-butan-2-ol
```

### Command Line

```bash
# Single molecule
orthonym "CCO"
# → ethanol

# Verbose output
orthonym "CC(=O)O" --verbose
# → SMILES: CC(=O)O
# → Style:  pin
# → Name:   acetic acid

# Batch processing
orthonym --batch molecules.txt --output names.txt
```

## Features

- **IUPAC 2013 Compliant**: Implements Preferred IUPAC Names (PINs) from the Blue Book
- **Comprehensive Coverage**:
  - Acyclic compounds (alkanes, alkenes, alkynes)
  - Functional groups (alcohols, acids, aldehydes, ketones, amines, etc.)
  - Cyclic compounds (cycloalkanes, aromatics)
  - Heterocycles (furan, pyridine, imidazole, etc.)
  - Fused ring systems
  - Stereochemistry (R/S, E/Z)
- **Round-trip Validated**: Names can be converted back via OPSIN
- **100% Open Source**: MIT License

## Accuracy

Measured against the v18 multi-corpus benchmark (ChEBI 5,000 + PubChem 2,000 + OPSIN self-test 500 = 7,500 compounds), Orthonym v19.0 achieves:

**default-OFF** (rule-based pipeline only; primary):

| Metric | Overall (7,500) | ChEBI 5,000 | PubChem 2,000 | OPSIN self-test 500 |
|---|---:|---:|---:|---:|
| Round-trip accuracy (absolute) | 27.12% | 26.02% | 13.65% | 92.00% |
| Round-trip accuracy (ceiling-relative) | 31.84% | 31.62% | 15.40% | 92.00% |
| Graded score (mean) | 2.7603/5.0 | 2.627/5.0 | 2.5537/5.0 | 4.9199/5.0 |

**`--allow-ml-fallback=ON`** (opt-in ML augmentation per ADR-19-08; predicted byte-identical to default-OFF under R-02 sentinel ACTIVE per CONTEXT D-10; ml_fallback_used_count = 0): same headline (2,034 / 7,500 = 27.12% absolute / 31.84% ceiling-relative); ML attach-rate 0% pending Phase 162.1 amendment commit upon Zenodo republish.

**v18.0 → v19.0 delta**: +7 RT overall (Phase 160.1 substituent enumerator +2 + Phase 161 P-69 organometallics +5). All other v19 phases ship substrate-only or canary-only (no multi-corpus RT delta at ship).

Ceiling-relative round-trip accounts for the fact that not all reference IUPAC names in ChEBI / PubChem round-trip back to their structures themselves; see []() and [Phase 147.1 strategic decisions]() for full methodology.

Salt-class compounds (314 of 7,500) are reported separately because reference-quality issues dominate salt-class round-trip behavior; non-salt accuracy (27.75%) is reported alongside salt-included accuracy (27.12%). See []() for details.

Stereochemistry is handled via RDKit's `rdCIPLabeler.AssignCIPLabels()`, which implements CIP rules 1-2 fully and rules 3-5 partially. The Hanson 2018 CIP Validation Suite reports 182/290 PASS on the v19 codebase (RDKit 2025.09.3 + 2026.03.1; stable across RDKit versions); affected stereo-dense compounds are documented in [](). v19 ships ZERO new stereo substrate per ADR-19-07 / ADR-19-08 / ADR-19-09 ZERO mutation invariants.

**Honest verdict**: v19.0 ships **PASS-WITH-CAVEATS**. Of the 8 V19 §4 minimum success thresholds, 2 are NOT MET on data: Overall RT (ceiling-relative) 31.84% vs ≥ 65% target (-33.16 pp gap); Overall RT (absolute) 27.12% vs ≥ 50% target (-22.88 pp gap). The remaining 6 thresholds PASS (graded total +0.0031 vs v18 baseline; OPSIN parse rate within determinism floor; canary +39 surplus over ≥ 1,354 floor; tests passing ≥ Phase 163 ship; HIGH-severity code review 0; test failures 0 NEW). The gap is at the score-extraction layer per `v19_final_report.md` §10 (substrate-clean-but-score-extraction-lag posture inherited from v18 §10 lesson); v20.0 architectural phase targets the remediation. NO V19 §4 minimum threshold has been relaxed per `the contributor guide` memory rule #4 ("no band-aids").

**Historical context:** HERITAGE (Wisniewski, Beilstein Institute, *J. Chem. Inf. Comput. Sci.* 1990, 30, 324-332) achieved 61% expert-status agreement with human nomenclaturists on stereo-bypassed random samples — the first general-purpose structure→IUPAC-name system. v19.0 explicitly handles stereo throughout and reports full numbers; v19.0 31.84% ceiling-relative remains below the 1990 algorithmic-naming benchmark on the modern multi-corpus suite (with stereo). v20.0 ≥ 65% ceiling-relative target carries forward.

For the comprehensive measurement report (per-phase attribution, per-compound-class accuracy, handler performance, stereo analysis, OPSIN failure deep dive, theoretical-ceiling analysis, v20.0 roadmap), see [`v19_final_report.md`]() (862 LOC, 12 sections + 4 appendices).

## Development

### Running Tests

```bash
# Quick unit tests
pytest tests/ -m "not slow" -v

# Full validation (100k compounds)
pytest tests/ -m slow -v

# Specific compound class
pytest tests/ -k "alcohol" -v
```

### Project Structure

```
Orthonym/
├── src/orthonym/
│   ├── perception/     # RDKit molecular feature extraction
│   ├── rules/          # IUPAC naming rules
│   ├── assembly/       # Name fragment composition
│   └── data/           # Lookup tables
├── tests/              # Test suite
└── ./skills/     #  Code development guides
```

## Contributing

Contributions welcome! See the development guide in `the contributor guide` for architecture details and how to add support for new compound classes.

## License

MIT License

## Citation

If you use Orthonym in your research, please cite:

```bibtex
@software{orthonym,
  author = {Rajan, Kohulan},
  title = {Orthonym: Open Structure-To-IUPAC-Name Generator},
  year = {2024},
  url = {https://github.com/Kohulan/Orthonym}
}
```

## Acknowledgments

- IUPAC Blue Book 2013 for nomenclature rules
- OPSIN for name-to-structure conversion (used for validation)
- RDKit for molecular perception
- ChEBI for validation dataset
