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

Measured against the multi-corpus benchmark (ChEBI 5,000 + PubChem 2,000 + OPSIN self-test 500 = 7,500 compounds), Orthonym v21.0 achieves (deterministic rule-based pipeline; the ML fallback was retired this milestone — ADR-21-01):

| Metric | Overall (7,500) | ChEBI 5,000 | PubChem 2,000 | OPSIN self-test 500 |
|---|---:|---:|---:|---:|
| Round-trip accuracy (absolute) | 30.36% | 29.58% | 16.85% | 92.20% |
| Round-trip accuracy (ceiling-relative) | 35.65% | 35.94% | 19.01% | 92.20% |
| Graded score (mean) | 2.7749/5.0 | 2.6454/5.0 | 2.5618/5.0 | 4.9219/5.0 |

Orthonym v21.0 is **deterministic-rules-only**: the dormant ML fallback (default-OFF, never fired in any prior run) was removed entirely (ADR-21-01), proven byte-identical. There is no longer an `--allow-ml-fallback` option.

**v20.0 → v21.0 delta**: **+125 RT overall** (2152 → 2277), measured by per-row A/B flip vs the v20 baseline (137 gains / 12 losses): ChEBI +92, PubChem +32, OPSIN self-test +1. v21 shipped the WS-A parent chokepoint + name-tree production path (Phases 178/179), the WS-C lipid/steroid/carbohydrate/glycoside subsystems (176/180–183), the WS-E charge-first PCG subsystem (184), and the WS-B `centres` CIP plumbing (177).

Ceiling-relative round-trip accounts for the fact that not all reference IUPAC names in ChEBI / PubChem round-trip back to their structures themselves; see []() and [Phase 147.1 strategic decisions]() for full methodology.

Salt-class compounds (314 of 7,500) are reported separately because reference-quality issues dominate salt-class round-trip behavior; non-salt accuracy (31.07%) is reported alongside salt-included accuracy (30.36%). See []() for details.

Stereochemistry is handled via RDKit's `rdCIPLabeler.AssignCIPLabels()`, which implements CIP rules 1-2 fully and rules 3-5 partially. The Hanson 2018 CIP Validation Suite reports 182/290 PASS on RDKit; v21 plumbed the `centres` CIP engine (281/290 on the same suite, a verified +46) as an opt-in, held default-OFF at validation to keep the RT measurement uncoupled. Affected stereo-dense compounds are documented in []().

**Honest verdict**: v21.0 ships **PASS-WITH-CAVEATS**. There is **no regression on any anti-regression anchor** (full-7500 RT 2277 ≥ 2152; ChEBI 1479 ≥ 1387; PubChem 337 ≥ 305; OPSIN self-test 461 ≥ 460; OPSIN parse 73.80% ≥ 73.04%; graded 2.7749 ≥ 2.702; PIN-strict gold protect 1/atropine; among-rings gold protect 0; 0 NEW test failures). The aspirational ceiling-relative RT gates — 35.65% vs ≥ 45% (honest) and ≥ 65% — are **reported as failed on data, not relaxed**: the binding constraint is unbuilt complex-class depth (multi-defect; the ≥31-heavy-atom class at ~13% RT), not missing Blue Book rules in the neutral spine. NO success threshold was relaxed per `the contributor guide` memory rule #4 ("no band-aids").

**Historical context:** HERITAGE (Wisniewski, Beilstein Institute, *J. Chem. Inf. Comput. Sci.* 1990, 30, 324-332) achieved 61% expert-status agreement with human nomenclaturists on stereo-bypassed random samples — the first general-purpose structure→IUPAC-name system. v21.0 explicitly handles stereo throughout and reports full numbers; v21.0 35.65% ceiling-relative remains below the 1990 algorithmic-naming benchmark on the modern multi-corpus suite (with stereo).

For the comprehensive measurement report (per-phase attribution, per-compound-class accuracy, handler performance, stereo analysis, OPSIN failure deep dive, theoretical-ceiling analysis, v22 roadmap), see [`V21.0-FINAL-REPORT.md`]() (12 sections + 4 appendices).

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
