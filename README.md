<div align="center">

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-banner-narrow.svg">
  <img src="assets/readme-banner.svg" alt="Orthonym engine. Checked IUPAC Names for Chemical Structures. Deterministic, rule-based, following the IUPAC 2013 recommendations." width="100%">
</picture>

<br>

<a href="https://orthonym.decimer.ai"><img src="assets/readme-try.svg" alt="Try it online" height="48"></a>&nbsp;&nbsp;<a href="#quick-start"><img src="assets/readme-start.svg" alt="Quick start" height="48"></a>

<br>

**A SMILES string goes in. A verified IUPAC name comes out, or a clear "no". Never a guess.**

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-1a1a1a?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-1a1a1a?style=flat-square)](https://www.python.org/)
[![IUPAC 2013](https://img.shields.io/badge/IUPAC-2013%20recommendations-2f6b28?style=flat-square)](https://doi.org/10.1039/9781849733069)
[![CI](https://github.com/Steinbeck-Lab/Orthonym/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Steinbeck-Lab/Orthonym/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-1a1a1a?style=flat-square)](https://steinbeck-lab.github.io/Orthonym/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23044199.svg)](https://doi.org/10.5281/zenodo.23044199)
[![Web app: Orthonym-Web](https://img.shields.io/badge/web%20app-Orthonym--Web-1a1a1a?style=flat-square&logo=github)](https://github.com/Steinbeck-Lab/Orthonym-Web)
[![Built with: orthonym-skills](https://img.shields.io/badge/built%20with-orthonym--skills-1a1a1a?style=flat-square&logo=github)](https://github.com/Kohulan/orthonym-skills)

</div>

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-gallery-narrow.svg">
  <img src="assets/readme-gallery.svg" alt="What it names: seven cards, each with a drawing, the name the engine prints and its tier. Bridged fused rings: (4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline-7,9-diol, pin_verified. Hypervalent iodine: iodylbenzene, pin_verified. Stereocentre at phosphorus: (1S,2R)-1,2-diphenylphospholane, pin_verified. Isotope labels: (2R,4S)-(4-2H1,2-3H1)pentane, pin_verified. Metal salt: zinc diacetate, pin_verified. Cage, with --emit-tier best-effort: pentacyclo[9.3.1.0^3,14.0^6,13.0^8,12]pentadecane, systematic_verified. Spiro: spiro[4.5]decane, pin_verified." width="100%">
</picture>

## Quick start

```bash
pip install orthonym                      # or: conda install -c conda-forge orthonym
orthonym --fetch-jars                     # one time: downloads and checks the OPSIN and centres jars
```

You need Python 3.10+ and a Java 11+ runtime on your `PATH` (the conda package brings Java with it).
More in [Install](guide/start/install.md).

```python
from orthonym import name_compound

name_compound("CCO")                          # 'ethanol'
name_compound("CC(=O)Oc1ccccc1C(=O)O")        # '2-(acetyloxy)benzoic acid'
name_compound("Cn1cnc2c1c(=O)n(C)c(=O)n2C")   # '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
name_compound("C/C=C/C")                      # '(2E)-but-2-ene'
name_compound("C[C@H](O)CC")                  # '(2S)-butan-2-ol'
```

`name_compound` also takes an RDKit `Chem.Mol`, for example a molecule read from an SDF file.

```bash
orthonym "CCO"                                # ethanol
orthonym "CC(=O)Oc1ccccc1C(=O)O" --provenance # the name plus its tier and how it was checked, as JSON
orthonym --batch molecules.smi -o names.txt   # one SMILES per line
```

These commands use the default tier, which returns a verified Preferred IUPAC Name (PIN) or no name.
For the wider tier, add `--emit-tier best-effort` (one SMILES or `--batch`), or in Python
`Orthonym(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)`.
See [Output tiers](#output-tiers).

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-specimen-narrow.svg">
  <img src="assets/readme-specimen.svg" alt="Five commands and what they print. orthonym CCO prints ethanol, tier pin_verified. Ibuprofen prints (2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid, pin_verified. With --emit-tier valid, sphingosine prints (2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol, pin_unverified, and retinol prints (2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-en-1-yl)nona-2,4,6,8-tetraen-1-ol, pin_unverified. Uranium trioxide, with --on-failure label, prints inorganic compound (not supported), abstain." width="100%">
</picture>

<sub>Real output, generated from the engine. The tier at the right comes from <code>--provenance</code>.</sub>

## How names are checked

<picture>
  <source media="(max-width: 700px)" srcset="assets/readme-roundtrip-narrow.svg">
  <img src="assets/readme-roundtrip.svg" alt="The round trip for caffeine. Orthonym writes the name 1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione. OPSIN reads that name back into a structure. The InChIKey of your structure and the InChIKey of what OPSIN read are both RYYVLZVUVIJVGH-UHFFFAOYSA-N, so OPSIN confirms that the name describes your structure; the strict PIN path built and certified it, so it is labelled pin_verified. Below, the four tiers a name can land on: PIN, fallback, best effort and no name, with the engine's tier ids." width="100%">
</picture>

- **Deterministic.** Names are built from the rules of the IUPAC 2013 recommendations (the "Blue
  Book"). No neural network, no sampling: the same input always gives the same name.
- **Checked.** [OPSIN](https://github.com/dan2097/opsin), which never saw your structure, reads
  every name back. The result must match your structure in constitution, charge and stereo. A few
  name forms OPSIN cannot read are checked another way, and `--provenance` marks them.
- **Honest.** A name that fails a check is withdrawn. When no name is left, Orthonym says so and
  gives the reason. It never guesses.

More in [How every name is checked](guide/checking.md) and [Declines](guide/declines.md).

## Output tiers

By default Orthonym returns only the Preferred IUPAC Name (PIN), verified, or no name. Wider tiers
are opt-in with `--emit-tier` (keyword switches of `Orthonym` in Python) and label every name
they return. The default is a rule about which names to return, not a limit of the engine. On 2,000
held-out ChEBI molecules the default tier names 1,106 (55.30 %) and declines 891, and
`--emit-tier best-effort` names 1,994 (99.70 %). A name counts only when OPSIN 2.9.0 reads it back
to the molecule ([both tiers on three sets](guide/accuracy.md#the-two-tiers)).

| `--emit-tier` | Returns |
|:--|:--|
| `pin` *(default)* | the verified PIN (with a few listed exceptions), or no name |
| `valid` | also other systematic names that OPSIN reads back to your structure |
| `complete` | also general names for aromatic and heterocyclic ring systems |
| `best-effort` | also the last-resort names, such as von Baeyer and spiro names for ring systems of up to 15 rings |
| `full-coverage` | also coordination names for metal tetrapyrrole and corrin complexes |

`--provenance` labels each name: `pin_verified` (the PIN, verified), `pin_unverified` (PIN form,
preferred status not certified), `systematic_verified` (a correct name that is not the PIN),
`best_effort` (a last-resort name) or `abstain` (no name). Metal compounds are admitted one metal
at a time. The exceptions, the labels and the metals are in [Output tiers](guide/tiers.md).

## Documentation

- [Docs site](https://steinbeck-lab.github.io/Orthonym/): install, use, tiers, provenance, reference.
- [How it works](guide/how-it-works.md) · [How accuracy is measured](guide/accuracy.md) ·
  [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Security](SECURITY.md) ·
  [Code of conduct](CODE_OF_CONDUCT.md)
- Found a wrong name? [Open an issue](https://github.com/Steinbeck-Lab/Orthonym/issues/new/choose).

## How Orthonym was built

Orthonym was built with [Claude Code](https://docs.claude.com/en/docs/claude-code) under a fixed
set of working rules, published as [**orthonym-skills**](https://github.com/Kohulan/orthonym-skills)
(20 skills, 2 hooks, 1 agent). Each rule makes a step rest on a measurement instead of a guess: fixed
test sets before and after each change, proof that the code to change is on the execution path, a
primary source before an expected value in a test changes, a regression gate before every merge,
and a written handoff between sessions.

## How to cite

Please cite the preprint:

> Rajan, K.; Zielesny, A.; Steinbeck, C. Orthonym: verified IUPAC names for chemical structures in
> the wild. *ChemRxiv* **2026**. [doi:10.26434/chemrxiv.15009769/v1](https://doi.org/10.26434/chemrxiv.15009769/v1)

```bibtex
@article{rajan2026orthonym,
  author  = {Rajan, Kohulan and Zielesny, Achim and Steinbeck, Christoph},
  title   = {{Orthonym}: verified {IUPAC} names for chemical structures in the wild},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15009769/v1}
}
```

To cite the software version you used, cite its Zenodo archive:

<!-- x-release-please-start-version -->
```bibtex
@software{orthonym,
  author  = {Rajan, Kohulan and Zielesny, Achim and Steinbeck, Christoph},
  title   = {{Orthonym}},
  version = {1.0.9},
  year    = {2026},
  doi     = {10.5281/zenodo.23044199},
  url     = {https://github.com/Steinbeck-Lab/Orthonym}
}
```
<!-- x-release-please-end -->

Every release is archived on Zenodo. The DOI [10.5281/zenodo.23044199](https://doi.org/10.5281/zenodo.23044199)
stands for all versions and resolves to the newest one.

## Built on

Orthonym stands on the IUPAC 2013 recommendations and on open cheminformatics software:
[RDKit](https://www.rdkit.org/) reads the structure, [OPSIN 2.9.0](https://github.com/dan2097/opsin)
reads the names back to check them, and [centres 1.2.1](https://github.com/SiMolecule/centres)
assigns the CIP descriptors. Without them there would be no Orthonym. `orthonym --opsin main` reads
the names back with OPSIN's main branch instead, built on your machine with a JDK
([Install](guide/start/install.md#choosing-the-opsin-version)); only then are phane names built for
cyclophanes and linear phanes ([Phane names](guide/use/phane-names.md)).

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
