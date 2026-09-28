# How it works

A SMILES string goes in. The engine reads the structure, applies the nomenclature rules, assembles a name and, for nearly every name, checks it with an OPSIN round trip before it leaves ([How every name is checked](checking.md)). A few name classes that OPSIN cannot read leave on their construction alone; `--provenance` marks them.

Four stages do the work. The orchestrator, `namer.py`, runs them and holds the final OPSIN check.

| Stage | What it does |
|:--|:--|
| **Perception** | RDKit reads the structure; Orthonym finds rings, characteristic groups and stereocentres. The CIP descriptors come from the centres labeller; RDKit's CIP labeller fills the few double bonds centres leaves unlabelled and takes over for a molecule centres cannot label. |
| **Rules** | Seniority, the parent hydride, locants, alphanumerical order and spelling, built as nomenclature classes from the IUPAC 2013 recommendations. Names the recommendations list one by one (retained and natural-product names), and a last-resort table of trivial names, are looked up by exact structure. |
| **Assembly** | Chains, rings, Hantzsch–Widman heterocycles, fused, bridged (von Baeyer) and spiro systems, and the characteristic-group families: acids, esters, amides, amines, nitriles and more. |
| **Validation** | The OPSIN round trip that decides whether a name leaves the engine, with named exceptions for classes OPSIN cannot read, and, for names from the general engine, an atom-coverage certificate. |

Compound classes are tried in a fixed order; a class that cannot build a name declines, and the next one is tried. There is no special case for a single molecule and no rewriting of an emitted name.

## The source tree

```text
src/orthonym/
├── namer.py        # orchestration and the final OPSIN check
├── cli.py          # the command line
├── jars.py         # finding, downloading and checking the OPSIN and centres jars
├── perception/     # structure perception: rings, characteristic groups, CIP stereo
├── routing/        # choosing the compound class
├── decomposition/  # naming large structures from their fragments
├── rules/          # IUPAC nomenclature rules
├── assembly/       # name assembly: locants, ordering, selection, the general engine
├── validation/     # OPSIN round-trip helpers and atom-coverage checks
├── metrics/        # provenance, tier labels and reason codes
└── data/           # naming tables
```

Every module has a page under [Internals](reference/internals/index.md).

## Built on

RDKit, OPSIN and centres, on the IUPAC 2013 recommendations. The credits and references are in the README, under [Built on](https://github.com/Beilstein-Institut/Orthonym#built-on).
