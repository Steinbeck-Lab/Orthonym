# Contributing to Orthonym

Thanks for your interest in improving Orthonym. This guide covers how to set up a
development environment, run the tests, and contribute a change.

## Development setup

```bash
git clone https://github.com/Kohulan/Orthonym.git
cd Orthonym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

A **Java runtime (JRE 11+)** must be on your `PATH` — Orthonym validates candidate names
by round-tripping them through OPSIN, which runs on the JVM. The OPSIN jar ships with the
package.

## Running tests

Run tests on targeted file sets (OPSIN-backed tests require a JVM):

```bash
python -m pytest tests/unit/rules/test_chain_names.py -q
python -m pytest tests/unit/assembly -q
```

Markers (`unit`, `integration`, `roundtrip`, `slow`, `benchmark`) are defined in
`pyproject.toml`.

## How Orthonym is built

The engine perceives structure (`perception/`), applies nomenclature rules (`rules/`), and
assembles the name (`assembly/`), drawing on naming tables in `data/`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the engineering conventions.

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
5. **Do not modify the evaluation splits** under  — they are the measurement
   contract.

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
