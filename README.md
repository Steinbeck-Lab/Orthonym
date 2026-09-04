# Orthonym

**Open Structure-TO-IUPAC-Name generator**

Orthonym is a deterministic, rule-based system that generates IUPAC systematic names
from molecular structures (SMILES), targeting Preferred IUPAC Names (PINs) as defined by
the IUPAC 2013 recommendations (the "Blue Book"). It is the structure→name counterpart to
[OPSIN](https://github.com/dan2097/opsin), which goes name→structure.

- **Deterministic** — the same structure always produces the same name; no model, no
  training data, no randomness.
- **Rule-based** — names are built from the IUPAC rules, not looked up or generated
  statistically.
- **Round-trip validated** — every candidate name is parsed back with OPSIN and checked to
  describe the input structure, so Orthonym does not emit a name for the wrong molecule.
- **Open source** — MIT licensed.

## Installation

Install from source:

```bash
git clone https://github.com/Kohulan/Orthonym.git
cd Orthonym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Java runtime (required for full validation)

Orthonym validates each candidate name by round-tripping it through OPSIN, which runs on
the Java Virtual Machine. **A Java runtime (JRE 11 or newer) must be on your `PATH`** for
full-fidelity naming. The OPSIN jar ships in the repository (invoked as an external Java
process — not linked), so a source install already has it; you only need a JVM on your `PATH`.
Without a JVM, Orthonym still runs but skips round-trip validation and operates in a
reduced-confidence mode. Bundled jars and their licenses are documented in [`NOTICE`](NOTICE).

## Quick start

### Python API

```python
from orthonym import name_compound

print(name_compound("CCO"))           # ethanol
print(name_compound("CC(=O)O"))       # acetic acid
print(name_compound("c1ccccc1"))      # benzene

# Stereochemistry
print(name_compound("C/C=C/C"))       # (E)-but-2-ene
print(name_compound("C[C@H](O)CC"))   # (2R)-butan-2-ol
```

### Command line

```bash
orthonym "CCO"                  # ethanol
orthonym "CC(=O)O" --verbose    # SMILES, style, and name
python -m orthonym "c1ccccc1"   # module form
```

## Features

- Preferred IUPAC Names (PINs) per the IUPAC 2013 Blue Book.
- Acyclic compounds: alkanes, alkenes, alkynes, and their functional-group derivatives
  (alcohols, acids, esters, aldehydes, ketones, amines, amides, nitriles, …).
- Cyclic compounds: cycloalkanes, arenes, heterocycles (Hantzsch–Widman), fused,
  bridged (von Baeyer), and spiro ring systems.
- Stereochemistry: R/S and E/Z descriptors, assigned through a high-accuracy CIP engine.
- Round-trip validation against OPSIN so a name is never emitted for the wrong structure.

## Accuracy

The headline metric is **round-trip exact match**: name the structure, parse the name back
with OPSIN, and compare canonical identifiers over every molecule in a fixed set, counting
any non-emission as a failure. This metric is reference-free — it does not depend on a
possibly-noisy database name.

On a 1,500-molecule benchmark drawn from ChEBI and PubChem:

| Metric | Value |
|---|---:|
| Round-trip exact match | **94.8%** |
| Wrong structures emitted | **0** |

The design priority is **never to emit a name for the wrong molecule**. When a preferred
name cannot be built with confidence, Orthonym degrades to a less-preferred but still
correct systematic name, or abstains — it does not guess.

See [`eval/README.md`](eval/README.md) for how the harness works and how to reproduce a
measurement.

## Development

### Running tests

Run tests on targeted file sets (the OPSIN-backed tests require a JVM):

```bash
python -m pytest tests/unit/rules/test_chain_names.py -q
python -m pytest tests/unit/assembly -q
```

Test markers (`unit`, `integration`, `roundtrip`, `slow`, `benchmark`) are defined in
`pyproject.toml`.

### Project structure

```
Orthonym/
├── src/orthonym/
│   ├── perception/   # structure perception (functional groups, rings, CIP stereo)
│   ├── rules/        # IUPAC naming rules
│   ├── assembly/     # name assembly (locants, ordering, selection)
│   └── data/         # naming tables
├── eval/             # accuracy harness and fixed splits
├── benchmarks/       # benchmark corpora and gold reference sets
├── scripts/          # utility and validation scripts
└── tests/            # test suite
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to set up a
development environment, run tests, and add support for new compound classes, and
[`CLAUDE.md`](CLAUDE.md) for the project's engineering conventions.

## License

MIT License — see [`LICENSE`](LICENSE). Bundled third-party components (OPSIN, the CIP
engine) are documented in [`NOTICE`](NOTICE).

## Citation

If you use Orthonym in your research, please cite:

```bibtex
@software{orthonym,
  author = {Rajan, Kohulan},
  title  = {Orthonym: Open Structure-TO-IUPAC-Name generator},
  year   = {2026},
  url    = {https://github.com/Kohulan/Orthonym}
}
```

## Acknowledgments

- The IUPAC 2013 recommendations (Blue Book) for the nomenclature rules.
- [OPSIN](https://github.com/dan2097/opsin) for name→structure conversion, used for validation.
- [RDKit](https://github.com/rdkit/rdkit) for molecular perception.
- ChEBI and PubChem for benchmark data.
