# Contributing to Orthonym

Thanks for your interest in improving Orthonym. This guide covers how to set up a
development environment, run the tests, and contribute a change.

## Development setup

```bash
git clone https://github.com/Steinbeck-Lab/Orthonym.git
cd Orthonym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

A **Java runtime (JRE 11+)** must be on your `PATH`: Orthonym validates candidate names
by round-tripping them through OPSIN, which is a Java program. The OPSIN and centres jars are
not part of the repository; `pip install` tries to fetch them when it builds the package, a
missing jar is downloaded on first use, and `orthonym --fetch-jars` fetches them or re-checks the
ones in the jar directory at any time (see the README, "The OPSIN and centres jars").

## Running tests

Run tests on targeted file sets (the OPSIN-backed tests need a Java runtime and the jars):

```bash
python -m pytest tests/unit/rules/test_multiplicative.py -q
python -m pytest tests/unit/rules/test_d1_coordination_v36.py -q
```

Markers (`unit`, `integration`, `roundtrip`, `slow`, `benchmark`) are defined in
`pyproject.toml`.

## How Orthonym is built

The engine perceives structure (`perception/`), dispatches by compound class (`routing/`,
`decomposition/`), applies nomenclature rules (`rules/`), and assembles the name (`assembly/`),
drawing on naming tables in `data/`. The orchestrator `namer.py` runs the final OPSIN check,
using helpers in `validation/` (which also holds the atom-coverage check);
`metrics/` records the provenance and the tier of each name.
[`guide/how-it-works.md`](guide/how-it-works.md) describes the main stages.

```
Orthonym/
├── src/orthonym/
│   ├── namer.py        # the naming pipeline and the final OPSIN check
│   ├── cli.py          # the command line
│   ├── perception/     # structure perception: rings, characteristic groups, CIP stereo
│   ├── routing/        # compound-class dispatch
│   ├── decomposition/  # fragment-based naming of large structures
│   ├── rules/          # IUPAC nomenclature rules
│   ├── assembly/       # name assembly: locants, ordering, selection
│   ├── validation/     # OPSIN round-trip and atom-coverage checks
│   ├── metrics/        # provenance, tiers and abstention codes
│   └── data/           # naming tables
└── tests/              # unit and integration tests
```

To add a compound class, add a test that pins the expected name and cites the governing IUPAC
rule. Release notes are in [`CHANGELOG.md`](CHANGELOG.md), vulnerability reports go by
[`SECURITY.md`](SECURITY.md), and questions and bug reports go to
[open an issue](https://github.com/Steinbeck-Lab/Orthonym/issues/new/choose).

## Contribution guidelines

1. **Fix at the root cause.** Fix a naming defect at the rule or data source that produced
   it — not with a per-molecule special case or a blind rewrite of the emitted name.
2. **Never emit a wrong structure.** A change must not cause Orthonym to emit a name that
   describes a different molecule than the input. When a preferred name cannot be built with
   confidence, degrade to a correct systematic name rather than guess.
3. **Cite the rule.** When a change touches nomenclature correctness, add or extend a test
   that pins the expected name, and cite the governing IUPAC rule in the code or test.
4. **Keep it deterministic.** Any algorithm that resolves a choice (ring numbering, locant
   assignment, ordering) must be deterministic.

## Submitting a change

1. Fork the repository and create a topic branch.
2. Make your change with tests.
3. Run the relevant tests and confirm they pass.
4. Open a pull request describing the change and the IUPAC rule it implements or corrects.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you
agree to uphold it.

## License

By contributing, you agree that your contributions are licensed under the MIT License.
