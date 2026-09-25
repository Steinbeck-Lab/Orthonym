<div align="center">

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-banner-narrow.svg">
  <img src="assets/readme-banner.svg" alt="Orthonym engine. Preferred IUPAC Names for chemical structures. Deterministic, rule-based, following the IUPAC 2013 recommendations." width="100%">
</picture>

<br>

<a href="https://orthonym.decimer.ai"><img src="assets/readme-try.svg" alt="Try it online" height="48"></a>&nbsp;&nbsp;<a href="#quick-start"><img src="assets/readme-start.svg" alt="Quick start" height="48"></a>

<br>

**A SMILES string goes in. Its Preferred IUPAC Name comes out, or a clear "no". Never a guess.**

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-1a1a1a?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-1a1a1a?style=flat-square)](https://www.python.org/)
[![IUPAC 2013](https://img.shields.io/badge/IUPAC-2013%20recommendations-2f6b28?style=flat-square)](https://doi.org/10.1039/9781849733069)
[![CI](https://github.com/Beilstein-Institut/Orthonym/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Beilstein-Institut/Orthonym/actions/workflows/ci.yml)

</div>

## How every name is checked

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-roundtrip-narrow.svg">
  <img src="assets/readme-roundtrip.svg" alt="The round trip for caffeine. Orthonym writes the name 1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione. OPSIN reads that name back into a structure. The InChIKey of your structure and the InChIKey of what OPSIN read are both RYYVLZVUVIJVGH-UHFFFAOYSA-N, so the name is a verified Preferred IUPAC Name. Below, the four tiers a name can land on: PIN, fallback, best effort and no name, with the engine's tier ids." width="100%">
</picture>

**Deterministic.** Orthonym builds every name from the nomenclature rules of the IUPAC 2013
recommendations, the "Blue Book". There is no neural network and no sampling: the same structure
always gets the same name.

**Checked.** In the default tier, every name is handed to [OPSIN](https://github.com/dan2097/opsin),
which never saw your structure, and parsed back. The two structures are compared by full InChIKey,
and an atom-coverage check confirms that every atom is named. `--provenance` reports the outcome
for each name in its `verified` field.

**Honest.** `--provenance` reports the tier each name landed on: `pin_verified`, `pin_unverified`,
`systematic_verified`, `best_effort` or `abstain`. A tier says how a name was built; whether OPSIN
read it back is its own field, `verified`. The pictures here draw the tiers with the web app's
four marks. The shape of a mark carries the tier and the colour only agrees, so the marks stay
distinct in greyscale.

## See it working

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-specimen-narrow.svg">
  <img src="assets/readme-specimen.svg" alt="Five commands and what they print. orthonym CCO prints ethanol, tier pin_verified. Ibuprofen prints (2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid, pin_verified. With --emit-tier valid, sphingosine prints sphingosine, pin_unverified, and retinol prints (2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-en-1-yl)nona-2,4,6,8-tetraen-1-ol, best_effort. Uranium trioxide prints inorganic compound (not supported), abstain." width="100%">
</picture>

<sub>Real output. The command line prints the plain line; the mark under it and the tier at its right
come from the same command run with <code>--provenance</code>. The pictures are generated from the
engine's own output, so they show what it prints.</sub>

The same from Python:

```python
>>> from orthonym import name_compound
>>> name_compound("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
'(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid'
```

## When it cannot name a molecule

It says so. The plain call returns a label in place of a name, such as
`inorganic compound (not supported)`, and `--provenance` marks the row `abstain` and says why:

```console
$ orthonym "O=[U](=O)=O" --provenance | python -m json.tool
{
    "name": "inorganic compound (not supported)",
    "tier": "abstain",
    "is_pin": false,
    "source": "abstain",
    "opsin": "n/a",
    "gates_passed": [],
    "gate_outcome": "suppressed",
    "formula": "O3U",
    "limit_code": "UNSUPPORTED_ELEMENT",
    "stereo_unexpressed": false,
    "suffix_free_prefix_name": false,
    "verified": "unverified"
}
```

In code, `orthonym.errors.is_failure_name(name)` tells such a label from a name.

## Quick start

Needs Python 3.10+ and a Java 11+ runtime on your `PATH` (see [Installation](#installation)).

```bash
pip install "git+https://github.com/Beilstein-Institut/Orthonym.git"
orthonym --fetch-jars          # one-time: downloads and checks the OPSIN and centres jars
```

```python
from orthonym import name_compound

name_compound("CCO")                          # 'ethanol'
name_compound("CC(=O)Oc1ccccc1C(=O)O")        # '2-(acetyloxy)benzoic acid'
name_compound("Cn1cnc2c1c(=O)n(C)c(=O)n2C")   # '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
name_compound("C/C=C/C")                      # '(2E)-but-2-ene'
name_compound("C[C@H](O)CC")                  # '(2S)-butan-2-ol'
```

From the command line:

```bash
orthonym "CCO"                                # ethanol
orthonym "CC(=O)Oc1ccccc1C(=O)O" --provenance # the name plus tier, validation and source, as JSON
orthonym --batch molecules.smi -o names.txt   # one SMILES per line
python -m orthonym "c1ccccc1"                 # module form
```

## What's inside

| Part | What it does |
|:--|:--|
| **Perception** | RDKit reads the structure; Orthonym finds rings, characteristic groups and stereocentres, with CIP descriptors from the centres labeller. |
| **Rules** | Seniority, the parent hydride, locants, alphanumerical order and spelling, built as whole nomenclature classes from the Blue Book, never as per-molecule special cases. |
| **Assembly** | Chains, rings, Hantzsch–Widman heterocycles, fused, bridged (von Baeyer) and spiro systems, and the characteristic-group families: acids, esters, amides, amines, nitriles and more. |
| **Validation** | The OPSIN round trip and the atom-coverage check that decide whether a name leaves the engine. |

Built on the [IUPAC 2013 recommendations](https://doi.org/10.1039/9781849733069) ·
[RDKit](https://www.rdkit.org/) · [OPSIN 2.9.0](https://github.com/dan2097/opsin) ·
[centres 1.2.1](https://github.com/SiMolecule/centres)

## Installation

| Requirement | Why |
|:--|:--|
| **Python 3.10+** | the engine |
| **Java runtime 11+** on your `PATH` | OPSIN round-trip validation and the CIP labeller run on the JVM |

Install from GitHub as in [Quick start](#quick-start), or from a clone, for development:

```bash
git clone https://github.com/Beilstein-Institut/Orthonym.git
cd Orthonym
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### The OPSIN and centres jars

Orthonym does **not** ship any Java jars. It uses two, and downloads them from their official
releases, checking each against a pinned SHA-256 checksum:

| Jar | Version | Used for | Licence |
|:--|:--|:--|:--|
| [OPSIN](https://github.com/dan2097/opsin) `opsin-cli-2.9.0-jar-with-dependencies.jar` | 2.9.0 | round-trip validation of every name | MIT (the jar bundles jna-inchi, LGPL-2.1, and others) |
| [centres](https://github.com/SiMolecule/centres) `centres.jar` | 1.2.1 | CIP stereo descriptors (R/S, E/Z) | BSD-2-Clause (the jar bundles CDK, LGPL-2.1+) |

`pip install` tries to fetch them for you. To fetch or re-check them at any time, run
`orthonym --fetch-jars`. If a jar is missing, Orthonym **stops with a clear error** rather than
quietly naming with less validation. For offline machines, point Orthonym at jars you copied
yourself:

| Setting | Effect |
|:--|:--|
| `ORTHONYM_OPSIN_JAR=/path/to/opsin-cli-2.9.0-jar-with-dependencies.jar` | use this OPSIN jar |
| `ORTHONYM_CENTRES_JAR=/path/to/centres-cli-1.2.1.jar` | use this centres jar |
| `ORTHONYM_JAR_DIR=/path/to/dir` | where downloaded jars are kept (default: your user cache) |

Licences and sources of the third-party components are listed in [`NOTICE`](NOTICE).

## Output tiers

The default is strict. Wider tiers are opt-in with `--emit-tier`:

| `--emit-tier` | Returns |
|:--|:--|
| `pin` *(default)* | the Preferred IUPAC Name, or nothing |
| `valid` | adds round-trip-verified general names |
| `complete` | adds round-trip-verified names from the wider general fallbacks (non-PIN allowed) |
| `best-effort` | adds names that pass the atom-coverage check but are not verified by OPSIN |

Whatever the tier, `--provenance` says what each name is: `pin_verified` (the strict PIN path built
and verified it), `pin_unverified` (a PIN-form name whose preferred status is not certified),
`systematic_verified` (verified, not the PIN), `best_effort` (the general engine built all or part
of it) or `abstain` (no name). The tier says how a name was built. The `verified` field says
whether OPSIN read it back to the same molecule; `--emit-tier best-effort` is the tier that adds
names OPSIN has not confirmed.

## How accuracy is measured

Orthonym is judged by what it gets **exactly right** and by what it gets **wrong**:

- **Round-trip exact match.** Name the structure, parse the name with OPSIN, and compare the full
  InChIKey with the input's. Every molecule counts; a declined molecule counts as a miss.
- **Wrong structures.** Any emitted name that parses to a different molecule. The design target is zero.
- **Preferred-name conformance.** Names checked character for character against molecules the
  Blue Book itself marks as the preferred name.

Benchmark results will be published with the accompanying paper.

## Development

Run the tests on targeted file sets (the OPSIN-backed tests need a Java runtime and the jars):

```bash
python -m pytest tests/unit/assembly -q
python -m pytest tests/unit/rules/test_multiplicative.py -q
```

Test markers (`unit`, `integration`, `roundtrip`, `slow`, `benchmark`) are defined in
`pyproject.toml`. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup and how to add a compound class:
add a test that pins the expected name and cites the governing IUPAC rule.

```
Orthonym/
├── src/orthonym/
│   ├── perception/   # structure perception: rings, characteristic groups, CIP stereo
│   ├── rules/        # IUPAC nomenclature rules
│   ├── assembly/     # name assembly: locants, ordering, selection
│   ├── validation/   # OPSIN round-trip and atom-coverage checks
│   └── data/         # naming tables
└── tests/            # unit and integration tests
```

Release notes are in [`CHANGELOG.md`](CHANGELOG.md), vulnerability reports go by
[`SECURITY.md`](SECURITY.md), and the community standards are in
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Questions and bug reports:
[open an issue](https://github.com/Beilstein-Institut/Orthonym/issues/new/choose).

## How to cite

A paper describing Orthonym is in preparation. Until it is published, please cite the software:
GitHub's **Cite this repository** button, built from [`CITATION.cff`](CITATION.cff), gives the
entry in APA and BibTeX. In BibTeX:

```bibtex
@software{orthonym,
  author  = {Rajan, Kohulan and Zielesny, Achim and Steinbeck, Christoph},
  title   = {{Orthonym}},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/Beilstein-Institut/Orthonym}
}
```

Orthonym stands on the IUPAC 2013 recommendations and on open cheminformatics software:

> Favre, H. A.; Powell, W. H. *Nomenclature of Organic Chemistry: IUPAC Recommendations and
> Preferred Names 2013*. Royal Society of Chemistry, **2013**.
> [doi:10.1039/9781849733069](https://doi.org/10.1039/9781849733069)
>
> Lowe, D. M.; Corbett, P. T.; Murray-Rust, P.; Glen, R. C. Chemical Name to Structure: OPSIN,
> an Open Source Solution. *J. Chem. Inf. Model.* **2011**, 51 (3), 739–753.
> [doi:10.1021/ci100384d](https://doi.org/10.1021/ci100384d)
>
> Hanson, R. M.; Musacchio, S.; Mayfield, J. W.; Vainio, M. J.; Yerin, A.; Redkin, D. Algorithmic
> Analysis of Cahn–Ingold–Prelog Rules of Stereochemistry: Proposals for Revised Rules and a Guide
> for Machine Implementation. *J. Chem. Inf. Model.* **2018**, 58 (9), 1755–1765.
> [doi:10.1021/acs.jcim.8b00324](https://doi.org/10.1021/acs.jcim.8b00324)

## Licence

MIT, see [LICENSE](LICENSE). The OPSIN and centres jars are not part of this repository; they are
downloaded from their official releases and keep their own licences (see [`NOTICE`](NOTICE)).

<div align="center">
<br>
<sub>Made with <img src="assets/readme-cup.svg" alt="coffee" height="14"> by <a href="https://www.kohulanr.com/">Kohulan Rajan</a> at</sub>
<br><br>
<a href="https://www.beilstein-institut.de/en/"><img src="assets/readme-beilstein.svg" alt="Beilstein-Institut" height="44"></a>
<img src="assets/readme-brush-x.svg" alt="and" height="44">
<a href="https://cheminf.uni-jena.de"><img src="assets/readme-steinbeck.svg" alt="Steinbeck Lab, Friedrich Schiller University Jena" height="44"></a>
<br><br>
<picture>
  <source media="(max-width: 700px) and (prefers-color-scheme: dark)" srcset="assets/readme-collab-line-narrow-dark.svg">
  <source media="(max-width: 700px)" srcset="assets/readme-collab-line-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme-collab-line-dark.svg">
  <img src="assets/readme-collab-line.svg" alt="An official collaboration for open science" width="560">
</picture>
</div>
