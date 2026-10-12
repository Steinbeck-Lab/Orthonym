# Contributing to Orthonym

Thanks for your interest in improving Orthonym. This guide covers how to set up a
development environment, run the tests, find the code that builds a name, and make a change
to a nomenclature rule.

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

The test suite turns the OPSIN validity gate **off** by default, and with it the default
tier's rule that only a verified preferred name is returned. A plain `name_compound` call in a
test therefore shows what the strict path builds, which is not always what a user gets. A test
that depends on the gate carries `@pytest.mark.opsin_gate`.

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

Two pages go deeper, and you will want both before you change a rule:

- [`guide/architecture.md`](guide/architecture.md) follows one SMILES through the code: the
  routing, the producers, the checks and the rescues. It explains how the tiers are flags and
  not separate code paths, and which names in the code mislead.
- [`guide/rule-index.md`](guide/rule-index.md) says where each rule lives, which rules are
  written in more than one place, and which file holds the code behind each class and handler id.

Release notes are in [`CHANGELOG.md`](CHANGELOG.md), vulnerability reports go by
[`SECURITY.md`](SECURITY.md), and questions and bug reports go to
[open an issue](https://github.com/Steinbeck-Lab/Orthonym/issues/new/choose).

## Finding the code that builds a name

There is no single naming path: several producers can name one molecule, and the first name
that passes the checks wins. Before you change anything, find out which code named your
molecule.

```bash
orthonym "CCOC(=O)C(C)O" --provenance                    # tier, source, how it was checked
orthonym "CCOC(=O)C(C)O" --emit-tier best-effort --provenance
orthonym "CCOC(=O)C(C)O" --dispatch-stats                # outer classes and inner handlers that matched
orthonym "CCOC(=O)C(C)O" --dump-tree                     # the parts of the name; class_id is the handler
```

The ids that these print are looked up in the tables of the rule index. Then check that the code
you are about to edit runs for your molecule at all: a function can be documented as the place
where a rule is decided and still never be called for your class. The architecture page shows how
to count calls with `cProfile`.

## Adding a rule or changing a name

1. **Start from the Blue Book.** Find the rule in the
   [IUPAC 2013 Recommendations](https://iupac.qmul.ac.uk/BlueBook/) and read the whole
   paragraph: the sentence that decides is often the last one. Note the section number, its
   heading and the criterion inside it. A rule number copied from a comment in the code may be
   wrong, so look it up. The book's examples marked `(PIN)` are the best test inputs.
2. **Reproduce the name** with the commands above, at the default tier and at
   `--emit-tier best-effort`. Note the `tier` and `source` of the row.
3. **Find the producer and the rule's home.** The rule index lists the canonical helper of a
   rule and the places that decide the same question beside it. If the rule is written in
   several places, fix the shared helper first, and say in the pull request which other places
   you did not change.
4. **Fix the cause, not the name.** Fix the rule or the data table that produced the defect.
   Do not add a case for one molecule, and do not rewrite the finished name by string
   surgery. When a preferred name cannot be built with confidence, degrade to a correct
   systematic name rather than guess.
5. **Write the test** (next section), run it, and run the neighbours.
6. **Check that nothing else moved** (the section after next).
7. **Open the pull request.**

## Writing the test

- Put the test beside the tests of the module you changed (`tests/unit/rules/`,
  `tests/unit/assembly/`, `tests/unit/perception/`, ...), mark it `@pytest.mark.unit`, and pin
  the exact name.
- Cite the rule in the docstring: the section number, its heading and the criterion, and the
  book's example when it has one. A name that OPSIN reads back is valid; it is not thereby the
  preferred name, so take the expected name from the book, not from the output of the code.
- Check the round trip independently of the engine. `tests/support/rt_assert.py` has
  `assert_full_rt(name, smiles)`, which has OPSIN read the name and compares the full InChIKey.
  `tests/support/default_tier.py` and `tests/support/pin_tiers.py` hold the assertions for the
  default tier and the best-effort tier.
- Name the molecule from two different SMILES spellings of it. The names must be equal: the
  same input always gives the same output.
- Keep history out of the test. Do not write "was X, now Y" notes or a dated log of earlier
  outputs next to an expectation; git log holds that history. Cite the rule once per test, not
  once per change.
- When a change is meant to alter existing names, change those expectations in the same
  commit, check each new name against the book, and list them in the pull request. Do not update
  an expected name only because the output now differs.

## Checking that nothing else moved

A change to a shared rule can move names far from the one you set out to fix.

- Run the tests of the package you changed and of the packages that call it. Some tests read the
  text of source files (to check that a function is not called from some module); if you renamed
  or moved a function, find them with `git grep -l -E "getsource|read_text" -- tests` and run them.
- Keep a list of a few dozen SMILES from the same class as your molecule, and compare the names
  before and after your change, at the default tier and at the tier you changed. This script
  prints one line per molecule; run it on the unchanged code and on your branch, and compare the
  two outputs with `diff`:

  ```python
  import json
  import sys

  from orthonym import Orthonym

  TIERS = {
      "pin": {},
      "valid": {"general_fallback": True},
      "complete": {"general_fallback": True, "allow_aromatic_general": True},
      "best-effort": {"general_fallback": True, "allow_aromatic_general": True,
                      "general_fallback_unverified": True},
  }

  tier, path = sys.argv[1], sys.argv[2]
  namer = Orthonym(**TIERS[tier])
  with open(path) as handle:
      for line in handle:
          smiles = line.split()[0]
          row = namer.name_tiered(smiles)
          print(json.dumps({"smiles": smiles, "name": row["name"], "tier": row["tier"]},
                           ensure_ascii=False))
  ```

  Save it as `snapshot.py` and run `python snapshot.py best-effort molecules.smi > before.jsonl` (the first
  argument is the tier, the second a file with one SMILES per line), then again after your change
  into `after.jsonl`, and run `diff before.jsonl after.jsonl`. Every difference must be one you
  intended. The default tier must not change unless the change is meant for it.

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
5. **Do not trust a comment about where a rule lives.** Count the calls before you edit.
6. **Match the code around you.** Keep the style, naming and comment density of the surrounding
   code.

## Submitting a change

1. Fork the repository and create a topic branch.
2. Make your change with tests.
3. Run the relevant tests and confirm they pass.
4. Open a pull request describing the change and the IUPAC rule it implements or corrects: the
   section, its heading and the criterion, the molecule, the name before and the name after.

Keep one topic to a pull request. The release notes are generated from the titles of the
commits, so begin the title with `fix:`, `feat:`, `refactor:`, `docs:` or `test:` followed by a
short description.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you
agree to uphold it.

## License

By contributing, you agree that your contributions are licensed under the MIT License.
