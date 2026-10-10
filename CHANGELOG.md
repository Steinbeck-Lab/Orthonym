# Changelog

All notable changes to Orthonym are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.7](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.6...v1.0.7) (2026-10-10)


### Bug Fixes

* input handling and wrong-name guards -- racemic stereo groups, all-bracket SMILES, ions, chain choice ([#15](https://github.com/Steinbeck-Lab/Orthonym/issues/15)) ([f4f3a87](https://github.com/Steinbeck-Lab/Orthonym/commit/f4f3a87bd663796ca479e7f0d122aee0224f025a))
* preferred-name labels and seniority -- phosphorus acids and esters, multiplied names, ketones ([#18](https://github.com/Steinbeck-Lab/Orthonym/issues/18)) ([eddb929](https://github.com/Steinbeck-Lab/Orthonym/commit/eddb92953859bfba8820cc113fbe6bfcead68dee))
* ring assemblies -- hydro prefixes, indicated hydrogen, primed components, assembly labels ([#17](https://github.com/Steinbeck-Lab/Orthonym/issues/17)) ([73e79c2](https://github.com/Steinbeck-Lab/Orthonym/commit/73e79c25a1c4302f9336412d03123221110c2f92))
* substituent prefixes, decomposition and speed -- guanidines, azo groups, acylsulfanyl, mercapturates ([#19](https://github.com/Steinbeck-Lab/Orthonym/issues/19)) ([082ccab](https://github.com/Steinbeck-Lab/Orthonym/commit/082ccabfc52d63333693adcabf6ee1dd287bbaab))

## [1.0.6](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.5...v1.0.6) (2026-10-09)


### Features

* name quality -- fusion names, preferred-name labels, book prefix forms at the wider tiers, faster validation ([bb658a7](https://github.com/Steinbeck-Lab/Orthonym/commit/bb658a7c5e33a46312f26c2abb625b5b3584b51c))


### Bug Fixes

* name quality -- fusion names, preferred-name labels, book prefix forms at the wider tiers, faster validation ([dc79af0](https://github.com/Steinbeck-Lab/Orthonym/commit/dc79af04839a67d3b86409ff0beae9d779b56953))
* names built from structure for esters and partial producers; a first start says why it is slow ([695b360](https://github.com/Steinbeck-Lab/Orthonym/commit/695b360ff4f46feac68c7952ce36e1600f78a80f))
* names built from structure for esters and partial producers; a first start says why it is slow ([968ba64](https://github.com/Steinbeck-Lab/Orthonym/commit/968ba64b1d559246460da77786e83e554978732f))

## [1.0.5](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.4...v1.0.5) (2026-10-07)


### Bug Fixes

* name quality -- book forms at every tier, stricter PIN labels ([597858a](https://github.com/Steinbeck-Lab/Orthonym/commit/597858ac14e51e1a550c10b1a8044ab471384b64))
* name_compound and the other public naming functions accept an RDKit molecule as well as a SMILES string ([324bf1d](https://github.com/Steinbeck-Lab/Orthonym/commit/324bf1dac3e218a6abea8870d278c1af011bc323))
* name_compound and the other public naming functions accept an RDKit molecule as well as a SMILES string ([0d3ad47](https://github.com/Steinbeck-Lab/Orthonym/commit/0d3ad4707aa9f259b43764bc6554dc0e44ed60f4))

## [1.0.4](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.3...v1.0.4) (2026-10-03)


### Bug Fixes

* bridged fused PINs on every printed fused parent, the morphine class, ketones and imides on bridged parents, more PIN classes and exact spellings ([85759e7](https://github.com/Steinbeck-Lab/Orthonym/commit/85759e7bca1481c7c0cf5a5f9ce48554fb39711f))

## [1.0.3](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.2...v1.0.3) (2026-10-01)


### Bug Fixes

* bridged fused PINs on naphthalene and anthracene parents, larger ring systems at the best-effort tier, more PIN classes and exact spellings ([7bdb0d2](https://github.com/Steinbeck-Lab/Orthonym/commit/7bdb0d2e6c2f6817610f5d9a4607a2202e6e716e))

## [1.0.2](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.1...v1.0.2) (2026-09-30)


### Bug Fixes

* engine fixes and optimizations, tier updates ([6cab387](https://github.com/Steinbeck-Lab/Orthonym/commit/6cab38789a73993310344acf4d15dfd882391fde))

## [1.0.1](https://github.com/Steinbeck-Lab/Orthonym/compare/v1.0.0...v1.0.1) (2026-09-29)

The naming engine is the same as in 1.0.0: the Python code differs only in comments and docstrings. The PyPI package now lists all three authors and Development Status 5 - Production/Stable, and the documentation cites the Zenodo archive.


### Documentation

* the Zenodo DOI in the README, the docs and CITATION.cff ([9415508](https://github.com/Steinbeck-Lab/Orthonym/commit/94155089a8a4ed7180bd7f060c9f66a9859cb968))

## [1.0.0](https://github.com/Steinbeck-Lab/Orthonym/releases/tag/v1.0.0) (2026-09-29)

Orthonym 1.0.0 is the first public release of a deterministic, rule-based generator that turns a SMILES string into an IUPAC name, aiming at the Preferred IUPAC Name (PIN) of the IUPAC 2013 recommendations. Each name is parsed back by OPSIN and must match the input structure by full InChIKey before it is shown; otherwise Orthonym declines and gives the reason, and the few names OPSIN cannot read in full are labelled as such. The release gathers 2,799 public commits made from late January to 29 September 2026, which took the engine from chains, monocycles and the main functional groups to fused, bridged, spiro and phane ring systems, stereodescriptors, charged species, isotopes, natural products, peptides, glycans, lipids and metal compounds. On the corpora of the accompanying paper, run with the paper's recipe on the release branch, it gave no wrong names.

### Highlights

- Rules only, with no neural network and no sampling: the same SMILES always gives the same name.
- Names are read back by OPSIN and compared with the input in constitution, charge and stereo; a name that fails is withdrawn, and when none is left Orthonym declines with a reason.
- A PIN tier for names built and certified on the strict preferred-name path, and opt-in wider tiers up to best effort (`--emit-tier`) whose names must pass a full-InChIKey round trip (metal-complex list names excepted).
- `--provenance` prints each result as JSON with its tier, its source and a `verified` field that says how the name was checked.
- 0 wrong names on the paper's corpora (measured on the release branch with the paper's recipe); round-trip exact: QM9 133,860 of 133,885, ChEBI 107,946 of 111,843, PubChem 500k 499,666 of 500,000, ZINC22 500k 488,428 of 500,000.
- Ring systems: monocycles, Hantzsch-Widman heterocycles, fused systems with IUPAC numbering, von Baeyer polycycles, spiro compounds, phanes and ring assemblies.
- Stereo: CIP R/S from the centres engine, E/Z, ring cis/trans and pseudoasymmetric r/s, each checked against the input's CIP labels.
- Ions, salts, zwitterions, radicals, isotope-labelled compounds, natural products, peptides, oligosaccharides, lipids, nucleotides and organometallics.
- OPSIN runs in the same process through JPype; the OPSIN and centres jars are downloaded at install and checked against pinned SHA-256 checksums.
- Documentation at https://steinbeck-lab.github.io/Orthonym/ and a web app at https://orthonym.decimer.ai (source: https://github.com/Steinbeck-Lab/Orthonym-Web).

### Nomenclature coverage

**Chains and substituents**
- Acyclic chains with IUPAC numerical terms, lowest locants and double- and triple-bond locants ([956329481](https://github.com/Steinbeck-Lab/Orthonym/commit/956329481)), including unbranched alkanes of 80 carbons and more ([68f50d128](https://github.com/Steinbeck-Lab/Orthonym/commit/0941b15ba)).
- One substituent-naming path shared by the ester, lactone, lactam, amide, acid-halide, anhydride and benzene namers, so substituents are not silently dropped ([08b6e0d62](https://github.com/Steinbeck-Lab/Orthonym/commit/08b6e0d62)); alkenyl and alkynyl prefixes ([d63e482b3](https://github.com/Steinbeck-Lab/Orthonym/commit/d63e482b3)).
- Nested enclosing marks for compound substituents, including N-substituents on amides (P-16.5) ([eb5d40c69](https://github.com/Steinbeck-Lab/Orthonym/commit/eb5d40c69)); alphabetical tie-breaking of chain orientation ([ec90679f5](https://github.com/Steinbeck-Lab/Orthonym/commit/ec90679f5)).
- Multiplicative names (bis/tris/tetrakis) that keep every identical arm, with bridges such as sulfanediyl, peroxy and disulfanediyl ([bcea5da52](https://github.com/Steinbeck-Lab/Orthonym/commit/22899d493), [5b2fc9c87](https://github.com/Steinbeck-Lab/Orthonym/commit/8040c91f0)).

**Parent selection and functional groups**
- The parent chain or ring is chosen by the P-44 criteria in order: principal groups, ring or chain, length, multiple bonds, lowest locants ([863ff10db](https://github.com/Steinbeck-Lab/Orthonym/commit/863ff10db), [6cf607ac7](https://github.com/Steinbeck-Lab/Orthonym/commit/60cb38e0d)).
- Suffix seniority follows the functional-group order of P-41, and lower-ranked groups are cited as prefixes ([4b6158840](https://github.com/Steinbeck-Lab/Orthonym/commit/4b6158840), [d396a18c9](https://github.com/Steinbeck-Lab/Orthonym/commit/d396a18c9)).
- Nitriles, amides, esters, acid halides, anhydrides, hydroperoxides, carbamic and thiocarboxylic acids, and functional class names for oximes, N-oxides, isocyanates and carbamates ([b73e049cf](https://github.com/Steinbeck-Lab/Orthonym/commit/b73e049cf)).
- Seleno and telluro acids ([24fb4c019](https://github.com/Steinbeck-Lab/Orthonym/commit/57703b4a7)), sulfinamides ([62bd3f515](https://github.com/Steinbeck-Lab/Orthonym/commit/85b4775db)), amidines, imidates, carbamoylamino and ylidene prefixes, and principal chains of As, Sb, Se and Te oxoacids.

**Rings: monocycles, Hantzsch-Widman and fused systems**
- Cycloalkanes, cycloalkenes and Hantzsch-Widman names for 3- to 10-membered heterocycles, such as oxolane and thiazolidine ([785a1e0c3](https://github.com/Steinbeck-Lab/Orthonym/commit/785a1e0c3)); unsaturated 7- and 8-membered heterocycles take Hantzsch-Widman names ([c6599a59b](https://github.com/Steinbeck-Lab/Orthonym/commit/15fea9ec3)).
- Retained fused heterocycles with IUPAC peripheral numbering ([87146be5e](https://github.com/Steinbeck-Lab/Orthonym/commit/87146be5e)); fused systems not in the dictionary are named from their components, the base component chosen by P-25.3.1.3 ([a821b6b53](https://github.com/Steinbeck-Lab/Orthonym/commit/a821b6b53)).
- A fusion-numbering engine for cata-fused arenes and mixed 5/6-membered systems, which declines ring systems it does not cover ([6a82e08bf](https://github.com/Steinbeck-Lab/Orthonym/commit/a6c417359)).
- Substituted two-component ortho-fused heterocycles and naphtho-fused systems ([bb07d4ba6](https://github.com/Steinbeck-Lab/Orthonym/commit/e5f747895), [a2a12971b](https://github.com/Steinbeck-Lab/Orthonym/commit/829491b05)).
- Indicated hydrogen, added indicated hydrogen and hydro prefixes, with cyclic ketones and quinones named as PIN diones ([7472e8dca](https://github.com/Steinbeck-Lab/Orthonym/commit/b83a56176)); chromones and coumarins on the 1-benzopyran parent ([757586d58](https://github.com/Steinbeck-Lab/Orthonym/commit/d475b7d3b)).

**Rings: von Baeyer, spiro, phanes and ring assemblies**
- Von Baeyer names with secondary and zero-atom bridges and heteroatom replacement ([5fde4e994](https://github.com/Steinbeck-Lab/Orthonym/commit/5fde4e994)); main ring and main bridge chosen for PIN descriptors, preferring the largest main bridge ([ce8a95d7b](https://github.com/Steinbeck-Lab/Orthonym/commit/ec665fee3)).
- Spiro names, including dispiro and heterospiro systems, spiro compounds of von Baeyer and fluorene components ([e991c19c7](https://github.com/Steinbeck-Lab/Orthonym/commit/97939fc53)), and dispiro and trispiro systems of fused components ([8e66cc181](https://github.com/Steinbeck-Lab/Orthonym/commit/73a5820fe)).
- Preferred names for monocyclic all-benzene homophanes ([a3c878a2a](https://github.com/Steinbeck-Lab/Orthonym/commit/8f71aee6b)); ring assemblies with primes (1,1':4',1''-terphenyl), enclosing marks and multipliers up to deci.

**Stereochemistry**
- R/S and E/Z with locants ([6bb4693a9](https://github.com/Steinbeck-Lab/Orthonym/commit/6bb4693a9)), with the centres engine as the default source of CIP labels ([0537e1517](https://github.com/Steinbeck-Lab/Orthonym/commit/a0b546eb7)).
- Descriptors on substituents, oximes, esters, lactones, lactams, fused heterocycles and polycycles, with CIP labels recomputed on each fragment ([646fe501a](https://github.com/Steinbeck-Lab/Orthonym/commit/646fe501a), [0179bca74](https://github.com/Steinbeck-Lab/Orthonym/commit/0179bca74)).
- Chain and ring stereocentres expressed in full; a name that leaves out or invents a stereocentre is rejected ([c242eb2c7](https://github.com/Steinbeck-Lab/Orthonym/commit/90455d8ad)); cis/trans on rings with two stereocentres ([9288fd0e3](https://github.com/Steinbeck-Lab/Orthonym/commit/b92fcdc82)).
- Steroid alpha/beta descriptors ([e4056e902](https://github.com/Steinbeck-Lab/Orthonym/commit/5807751d6)).

**Charged species, salts, zwitterions and radicals**
- Each input is classified as neutral, ion, zwitterion, salt or radical ([867cc6026](https://github.com/Steinbeck-Lab/Orthonym/commit/867cc6026)), and salts are named as cation plus anion ([512aed128](https://github.com/Steinbeck-Lab/Orthonym/commit/5bb88bb06)).
- Semipolar oxides (oxido/-ium), azide, diazo, nitrooxy and isocyano groups ([2803b3a97](https://github.com/Steinbeck-Lab/Orthonym/commit/70e39c1fa)); nitro and N-oxide groups are named as neutral groups ([37461389e](https://github.com/Steinbeck-Lab/Orthonym/commit/37461389e)).
- Stereo ionic salts, protonated diamines, hemisalts and mixed multi-anion salts; a salt with a part that cannot be named is refused ([2c622fff5](https://github.com/Steinbeck-Lab/Orthonym/commit/d78da5586)).
- Carbon, chalcogen, ring-N, N-oxyl, acyl, nitrene and multicentre radicals, each shown only after a round trip that compares the radical sites ([34ef0d3af](https://github.com/Steinbeck-Lab/Orthonym/commit/f575d9083)).

**Isotopes**
- Isotopic descriptors ordered by symbol and mass, with counts as subscripts, as in (1,1-2H2) ([50331d9a9](https://github.com/Steinbeck-Lab/Orthonym/commit/7b3beaaa9)); labelled retained names fall back to a systematic parent, and labels go on the right word of a salt name ([f5eaa886a](https://github.com/Steinbeck-Lab/Orthonym/commit/e6c977a10)).

**Natural products and retained names**
- Steroid and alkaloid scaffolds named from the scaffold, with IUPAC atom numbering for nine steroid skeletons ([55ed43bb3](https://github.com/Steinbeck-Lab/Orthonym/commit/55ed43bb3)); nor-, homo- and seco- modifications ([c1bc100a2](https://github.com/Steinbeck-Lab/Orthonym/commit/c1bc100a2)).
- Terpene stereoparents (IUPAC Table 10.1c), a complete D/L set of aldonic acids and xylitol ([7b1175c72](https://github.com/Steinbeck-Lab/Orthonym/commit/15ddd47cd)); ring stereo on saturated steroid scaffolds ([815c950f6](https://github.com/Steinbeck-Lab/Orthonym/commit/f7b9e25d8)).
- Retained names that are not PINs give way to systematic ones (acetone becomes propan-2-one, picric acid 2,4,6-trinitrophenol); retained names from OPSIN data are round-trip checked before use ([163007b89](https://github.com/Steinbeck-Lab/Orthonym/commit/163007b89), [5a696fd61](https://github.com/Steinbeck-Lab/Orthonym/commit/5a696fd61)).

**Peptides, glycans, lipids and nucleotides**
- Peptides as acylamino chains, including capped termini, ester C-termini and non-standard backbones ([6515385de](https://github.com/Steinbeck-Lab/Orthonym/commit/6515385de), [d5371532a](https://github.com/Steinbeck-Lab/Orthonym/commit/ba7acaf0a)).
- An amino acid with a substituent on its nitrogen takes its systematic name, e.g. (2S)-1-acetylpyrrolidine-2-carboxylic acid ([68f50d128](https://github.com/Steinbeck-Lab/Orthonym/commit/0941b15ba)).
- About 50 retained sugar names and glycosides ([769783748](https://github.com/Steinbeck-Lab/Orthonym/commit/769783748)); disaccharides and oligosaccharides, including non-reducing ones such as raffinose ([419c678b8](https://github.com/Steinbeck-Lab/Orthonym/commit/63a6159c2)).
- Triglycerides and polyol polyesters ([6a85190b4](https://github.com/Steinbeck-Lab/Orthonym/commit/6a85190b4)); phosphatidylcholine, phosphatidylethanolamine, phosphatidylserine, phosphatidic acid, ceramides and sphingolipids ([9db641d5f](https://github.com/Steinbeck-Lab/Orthonym/commit/4855a60ba)).
- Nucleosides and nucleotides with modified sugars or substituted bases ([d64ccb421](https://github.com/Steinbeck-Lab/Orthonym/commit/131f21ab0)); acyl-CoA molecules on their 9H-purine parent ([9f236276e](https://github.com/Steinbeck-Lab/Orthonym/commit/67ed5a4db)).

**Organometallics and inorganic parents**
- Metallocenes such as ferrocene, mononuclear metal carbonyls and half-sandwich complexes ([9e0fe162a](https://github.com/Steinbeck-Lab/Orthonym/commit/863c1c025)); sigma-bonded main-group organometallics ([95ef27193](https://github.com/Steinbeck-Lab/Orthonym/commit/43d25fc74)).
- Additive names for sigma-coordinated and metallacycle compounds; compounds with two metals are refused rather than guessed ([a147f99a1](https://github.com/Steinbeck-Lab/Orthonym/commit/45c9e2929)).
- An exact-InChIKey table of retained coordination names drawn from ChEBI, including corrinoid precursors ([c75c88549](https://github.com/Steinbeck-Lab/Orthonym/commit/253ed2828)), and hydrate word forms (mono, di, hemi, sesqui).
- Parent hydrides of Group 15, the chalcogens, the halogens, boron and Group 14, polyazanes and lambda-convention hydrides ([376445ebf](https://github.com/Steinbeck-Lab/Orthonym/commit/74c6d148b)).

**Large molecules and mixtures**
- Large molecules split at ester, amide, ether, glycosidic, thioester, phosphodiester and sulfonamide bonds, including several bonds of mixed types, with each piece named and the name rebuilt ([26c75704c](https://github.com/Steinbeck-Lab/Orthonym/commit/26c75704c), [c2f90e2d0](https://github.com/Steinbeck-Lab/Orthonym/commit/c2f90e2d0)).
- Neutral multi-component SMILES, such as cocrystals, named one component at a time in a fixed order ([f43b8e3e4](https://github.com/Steinbeck-Lab/Orthonym/commit/f43b8e3e4)).

### Checks and correctness

- Every public entry point checks its own names by an OPSIN round trip that compares constitution, charge and stereo ([0be43e0e5](https://github.com/Steinbeck-Lab/Orthonym/commit/5d3ad6079)); a name that OPSIN reads as a different molecule is not shown ([5f43cd2bf](https://github.com/Steinbeck-Lab/Orthonym/commit/360b1eb86)).
- The default tier also names a few classes OPSIN cannot read in full: names from exact-match lists (metal-complex and natural-product parent names), a few name forms OPSIN's grammar lacks or misreads, and names whose stereodescriptors OPSIN cannot parse (constitution confirmed by OPSIN, each descriptor checked against its CIP label). `--provenance` marks each; the wider tiers ship none of them except the metal-complex list names.
- Tiers reported by `--provenance`: `pin_verified`, `pin_unverified`, `systematic_verified`, `best_effort` and `abstain`; `verified` is `opsin`, `opsin_constitution`, `identity` or `unverified` ([55756fccb](https://github.com/Steinbeck-Lab/Orthonym/commit/04569243d)); the best-effort tier gives a correct name or abstains ([574514b0d](https://github.com/Steinbeck-Lab/Orthonym/commit/4bac21b81)).
- An atom-coverage check rejects names that drop atoms ([0dfbc6eaa](https://github.com/Steinbeck-Lab/Orthonym/commit/0dfbc6eaa)), and a grammar check tests brackets, hyphens and stereo placement before a name is emitted ([317684d48](https://github.com/Steinbeck-Lab/Orthonym/commit/773403827)).
- Numbering ties are settled by the CIP criteria (P-14.4 (j)); the same input string always gives the same name; the centres jar is fixed to its tagged 1.2.1 release ([866f91477](https://github.com/Steinbeck-Lab/Orthonym/commit/583619cbc)).
- Unsupported inputs, such as wildcard atoms, many inorganic compounds, unnameable salt parts and ring systems the engine cannot number, are declined with a reason ([722efbc63](https://github.com/Steinbeck-Lab/Orthonym/commit/304972556)).

### Performance and robustness

- OPSIN runs in the same process through JPype instead of a new JVM for each name, which made naming about 2.6 times faster ([502c251cd](https://github.com/Steinbeck-Lab/Orthonym/commit/7db7d39c5)).
- Memoization scoped to each call or molecule, with byte-identical output ([342a40ab8](https://github.com/Steinbeck-Lab/Orthonym/commit/0cb1d833c)); SMARTS patterns compiled once and ring lookups hash-bucketed ([504084230](https://github.com/Steinbeck-Lab/Orthonym/commit/d64f9f0a2)).
- A per-molecule work budget abstains on macrocycles that would take too long ([90b265b39](https://github.com/Steinbeck-Lab/Orthonym/commit/7a88ab14c)); large peptides, glycopeptides and oligosaccharides name within the time limit.

### Command line, Python API and packaging

- Python API: `name_compound()`, the `Orthonym` class and `name_with_tree()`, which returns the name with its structured name tree ([89441dced](https://github.com/Steinbeck-Lab/Orthonym/commit/89441dced), [b4247b9ee](https://github.com/Steinbeck-Lab/Orthonym/commit/580eecd60)).
- The `orthonym` command names single SMILES or batch files, with `--emit-tier`, `--provenance` (JSON) and `--dump-tree` ([8dccfa14c](https://github.com/Steinbeck-Lab/Orthonym/commit/115ec06c4)); an abstention prints a clean line instead of 'None'.
- `orthonym --fetch-jars` downloads the OPSIN and centres jars and checks each against its pinned SHA-256 checksum; the jars' licences are listed in NOTICE.
- The package imports cleanly on a minimal install, declares lxml as a runtime dependency, and raises ValueError on an unparseable SMILES ([29120be67](https://github.com/Steinbeck-Lab/Orthonym/commit/21b8953cd)).
- Python 3.10+ and a Java 11+ runtime; MIT licence; public CI and release automation with release-please and PyPI Trusted Publishing ([ea1ec9532](https://github.com/Steinbeck-Lab/Orthonym/commit/dacac2da9)).

### Documentation

- A documentation site, https://steinbeck-lab.github.io/Orthonym/, covers install, a first name, tiers, how each name is checked, declines, accuracy, the Python API and the command line ([6b3cb017c](https://github.com/Steinbeck-Lab/Orthonym/commit/b123e4ab3)).
- The README has a round-trip diagram, real example output, a quick start and credits (IUPAC 2013 recommendations, RDKit, OPSIN, centres), with guide pages on how it works, declines and accuracy ([e4f4ae382](https://github.com/Steinbeck-Lab/Orthonym/commit/deb7cbdaf)).
- Public docstrings and `--help` text describe each option in plain language; changelog, contributing, code of conduct and security files are included ([a275e9958](https://github.com/Steinbeck-Lab/Orthonym/commit/99a94f073)).
- The data record of the accompanying paper is at https://doi.org/10.5281/zenodo.22946586.

### Development timeline

- **January 2026:** first public commits: chains, monocycles, benzene derivatives, Hantzsch-Widman heterocycles, fused and bridged ring systems, suffix seniority, CIP stereo, and the `orthonym` command with a Python API.
- **February 2026:** ions, salts, zwitterions and radicals; von Baeyer polycycles, lactones, lactams and macrocycles; steroids, alkaloids, sugars and peptides; splitting of large molecules; a confidence score for every name.
- **March 2026:** one shared substituent-naming path, parent selection by the P-44 criteria, fused names from components, dispiro names and splitting at several bonds.
- **April 2026:** retained names, amino acids and sugars extended with OPSIN data and round-trip checks; a 300-molecule CIP validation set; one shared candidate pool.
- **May 2026:** an OPSIN grammar check before emission, metallocenes, seleno and telluro acids, and `name_with_tree()` with `--dump-tree`.
- **June 2026:** rules only (the experimental machine-learning fallback was removed); OPSIN parse and self-consistency checks on every name; centres as the default CIP source; indicated hydrogen; fusion numbering; carbohydrates, lipids, nucleotides and organometallics.
- **July 2026:** output tiers, `--emit-tier` and `--provenance`; OPSIN in the same process; isotope descriptors; phanes; stereo completeness; sigma-coordination names.
- **August 2026:** the best-effort tier gated on a full round trip; sulfinamides, spiro and fused breadth, semipolar oxides, salts, capped peptides, non-reducing oligosaccharides and the coordination-name table.
- **September 2026:** every shown name passes its own round trip; radical names; systematic names for N-substituted amino acids; faster naming with scoped memoization; `--fetch-jars` with checksums; the documentation site; release 1.0.0 on 29 September.

### Changes since the first publish

#### Features

* more preferred IUPAC names, fewer lost names, two wrong-name classes closed, honest labels ([e3a9a63](https://github.com/Steinbeck-Lab/Orthonym/commit/e56e9821bda098c3ea264aada0de7909fdac729d))
* nomenclature coverage and correctness updates ([931fe63](https://github.com/Steinbeck-Lab/Orthonym/commit/c2beb0bc6df1d8e4b81f179e6eb3488adf784903))
* nomenclature coverage and correctness updates ([1d835f6](https://github.com/Steinbeck-Lab/Orthonym/commit/d468f7b65a760f039086b1b8f7fc352537aa7e53))
* nomenclature coverage and correctness updates ([327e28c](https://github.com/Steinbeck-Lab/Orthonym/commit/44c23872a36ebc859181dc709f59919ad52feab4))
* radical names, safer salt names, stable numbering and Blue Book PIN spelling fixes ([34ef0d3](https://github.com/Steinbeck-Lab/Orthonym/commit/f575d9083cfa7402a9ddf5de217aac6be529d1cf))
* systematic names for N-substituted amino acids, long alkanes named again, and many preferred-name fixes ([68f50d1](https://github.com/Steinbeck-Lab/Orthonym/commit/0941b15ba3db53d7596e505b1b947dc5ac6c7c28))


#### Bug Fixes

* every shown name passes its own round trip, N-substituted amino acids get systematic names, and ChEBI losses are restored ([0be43e0](https://github.com/Steinbeck-Lab/Orthonym/commit/5d3ad607909f2181afcf14b10ec7121c66331bc5))
* honest tier labels, faster naming of very large molecules, and every paper-named ChEBI structure named again ([eb25c33](https://github.com/Steinbeck-Lab/Orthonym/commit/748a6031fb0e32c0fdb101bdeff5cf90225f9531))
* nomenclature correctness updates ([57eaa92](https://github.com/Steinbeck-Lab/Orthonym/commit/7c5c6c52e75fc8c3af0b717a1d685cf520406f4a))


#### Documentation

* credits back in the README under Built on ([2153b79](https://github.com/Steinbeck-Lab/Orthonym/commit/2e5e1c9830505b96b11e2876585dff4a4d3a4eca))
* lighter README, with guide pages for how it works, declines and accuracy ([e4f4ae3](https://github.com/Steinbeck-Lab/Orthonym/commit/deb7cbdaf8665f96ccb8859248c64df2eb28fbe9))
* project links in the README, the citation file, the package metadata and the documentation site ([d2f3118](https://github.com/Steinbeck-Lab/Orthonym/commit/4d1bfb9d10509589199efd3729de6880b60f351e))
* the documentation site shows the Orthonym mark and the web app's credit, and the README links the docs and the web app ([7fc9231](https://github.com/Steinbeck-Lab/Orthonym/commit/87099ae2f74fe35039111e956544ca9070ed8f7e))
* the Orthonym documentation site, with API docstrings and command-line help in plain language ([6b3cb01](https://github.com/Steinbeck-Lab/Orthonym/commit/b123e4ab3f7b0964f96e75510d42502114263a75))
