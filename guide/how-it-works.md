# How Orthonym works

A SMILES string goes in. The engine perceives the structure, applies the nomenclature rules,
assembles a name, and checks it before it leaves. Four parts do this work:

| Part | What it does |
|:--|:--|
| **Perception** | RDKit reads the structure; Orthonym finds rings, characteristic groups and stereocentres, with CIP descriptors from the centres labeller. |
| **Rules** | Seniority, the parent hydride, locants, alphanumerical order and spelling, built as whole nomenclature classes from the Blue Book, never as per-molecule special cases. |
| **Assembly** | Chains, rings, Hantzsch–Widman heterocycles, fused, bridged (von Baeyer) and spiro systems, and the characteristic-group families: acids, esters, amides, amines, nitriles and more. |
| **Validation** | The OPSIN round trip and the atom-coverage check that decide whether a name leaves the engine. |

The source tree follows the same four parts:

```
src/orthonym/
├── perception/   # structure perception: rings, characteristic groups, CIP stereo
├── rules/        # IUPAC nomenclature rules
├── assembly/     # name assembly: locants, ordering, selection
├── validation/   # OPSIN round-trip and atom-coverage checks
└── data/         # naming tables
```

## Built on

RDKit, OPSIN and centres, on the IUPAC 2013 recommendations. The credits and references are in the
README, under [Built on](../README.md#built-on).
