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

Measured against the v18 multi-corpus benchmark (ChEBI 5,000 + PubChem 2,000 + OPSIN self-test 500 = 7,500 compounds), Orthonym v18.0 achieves:

| Metric | Overall (7,500) | ChEBI 5,000 | PubChem 2,000 | OPSIN self-test 500 |
|---|---:|---:|---:|---:|
| Round-trip accuracy (absolute) | 27.03% | 25.94% | 13.50% | 92.00% |
| Round-trip accuracy (ceiling-relative) | 31.73% | 31.52% | 15.23% | 92.00% |
| Graded score (mean) | 2.7572/5.0 | 2.624/5.0 | 2.5496/5.0 | 4.9199/5.0 |

Ceiling-relative round-trip accounts for the fact that not all reference IUPAC names in ChEBI / PubChem round-trip back to their structures themselves; see []() and [Phase 147.1 strategic decisions]() for full methodology.

Salt-class compounds (314 of 7,500) are reported separately because reference-quality issues dominate salt-class round-trip behavior; non-salt accuracy (27.65%) is reported alongside salt-included accuracy (27.03%). See []() for details.

Stereochemistry is handled via RDKit's `rdCIPLabeler.AssignCIPLabels()`, which implements CIP rules 1-2 fully and rules 3-5 partially. The Hanson 2018 CIP Validation Suite reports 182/290 PASS on the v18 codebase (RDKit 2026.03.1; stable across RDKit versions); affected stereo-dense compounds are documented in []().

**Historical context:** HERITAGE (Wisniewski, Beilstein Institute, *J. Chem. Inf. Comput. Sci.* 1990, 30, 324-332) achieved 61% expert-status agreement with human nomenclaturists on stereo-bypassed random samples — the first general-purpose structure→IUPAC-name system. v18 explicitly handles stereo (Phases 152, 153) and reports full numbers; v19 targets ≥ 65% ceiling-relative round-trip to exceed the 1990 expert-status baseline on the v18 multi-corpus suite. See [`v18_final_report.md`]() §10 + §11 for the full v19 roadmap.

For the comprehensive measurement report (per-phase attribution, per-compound-class accuracy, handler performance, stereo analysis, OPSIN failure deep dive, theoretical-ceiling analysis, v19 roadmap), see [`v18_final_report.md`]().

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
