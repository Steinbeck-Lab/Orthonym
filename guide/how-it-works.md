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

Orthonym stands on the IUPAC 2013 recommendations and on open cheminformatics software:
[RDKit](https://www.rdkit.org/) · [OPSIN 2.9.0](https://github.com/dan2097/opsin) ·
[centres 1.2.1](https://github.com/SiMolecule/centres)

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
