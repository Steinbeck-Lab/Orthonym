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
