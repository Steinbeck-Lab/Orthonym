# How it works

A SMILES string goes in. The engine reads the structure, applies the nomenclature rules, assembles a name and, for nearly every name, checks it with an OPSIN round trip before it leaves ([How every name is checked](checking.md)). A few name classes that OPSIN cannot read leave on their construction alone; `--provenance` marks them.

![An animated isometric drawing of a typesetting bench that shows how Orthonym names a molecule, with six coloured parts numbered to match a legend. 1, blue: the input structure on a plate under four dial gauges, next to a card holding its SMILES. 2, amber: a cabinet of type drawers sets the name, one block per part, in a small tray called a galley. 3, violet: a separate machine, OPSIN, reads only the type, draws the structure it reads and cuts a key from it. 4, green: a lock of 25 pins set by the input's InChIKey; for caffeine's name, 1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione, every pin sits level and the gate is open. 5, teal: trays for PIN, fallback and best-effort names. 6, red: a bin for withdrawn names and a flag that rises when no name can be built.](_static/orthonym-anatomy.svg)

*Orthonym as a machine: RDKit reads the structure (1), the IUPAC 2013 rules set the name (2), and OPSIN, which never sees the input, reads it back (3). A name ships only when the InChIKey of what OPSIN read matches the input's in every letter (4); it then lands on its tier (5), or is withdrawn or declined (6).*

Four stages do the work. The orchestrator, `namer.py`, runs them and holds the final OPSIN check.

| Stage | What it does |
|:--|:--|
| **Perception** | RDKit reads the structure; Orthonym finds rings, characteristic groups and stereocentres. The CIP descriptors come from the centres labeller; RDKit's CIP labeller fills the few double bonds centres leaves unlabelled and takes over for a molecule centres cannot label. |
| **Rules** | Seniority, the parent hydride, locants, alphanumerical order and spelling, built as nomenclature classes from the IUPAC 2013 recommendations. Retained names and natural-product parent names that the recommendations list one by one, the names of metal tetrapyrrole complexes (ChEBI names, matched by exact InChIKey) and a last-resort table of trivial names are looked up by exact structure. Natural-product names with substituents on such a parent are built from it. Where the engine builds a bridged fused PIN for a molecule (morphine), that PIN is returned instead of the natural-product name. |
| **Assembly** | Chains, rings, Hantzsch–Widman heterocycles, fused, bridged fused, bridged (von Baeyer) and spiro systems, and the characteristic-group families: acids, esters, amides, amines, nitriles and more. |
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

Every module has a page under [Internals](reference/internals/index.md). To change the code, read [Architecture](architecture.md), which follows one SMILES through it, and the [Rule index](rule-index.md), which says where each rule lives.

## Built on

RDKit, OPSIN and centres, on the IUPAC 2013 recommendations. The credits and references are in the README, under [Built on](https://github.com/Steinbeck-Lab/Orthonym#built-on).
