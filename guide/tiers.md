# Output tiers

Every name Orthonym returns carries a tier. The tier says **how the name was built**. The `verified` field of the [provenance row](use/provenance.md) says **whether OPSIN read it back**. Read the two together.

## The five tiers

{tier}`pin_verified`
: The strict PIN path built the name and verified it: OPSIN read it back to your structure (for a PIN spelling OPSIN has no grammar for, a twin spelling of it: `verified` `opsin_twin`; for a poly-anhydride name, the acids it cites: `verified` `opsin_acids`), or, for a metal-complex name from the exact-match list, your structure's InChIKey matched the list entry exactly (`verified` `identity`). A compound without carbon (`sodium chloride`, `sulfuric acid`, `ammonia`) carries the tier its naming path gives it. `is_pin` is `true` only here.

{tier}`pin_unverified`
: A name in PIN form that only a breadth producer built, so its preferred status is not certified. It is also the tier of a strict-path name that OPSIN read back only in part (its constitution) or not at all. The `verified` field tells which. At `best-effort` it is the tier of the name the default tier ships for a molecule that has no other name, when OPSIN cannot read the stereodescriptors of that name: the same name with the same label.

{tier}`systematic_verified`
: A correct systematic name that is not the PIN: from the general engine (for example, a von Baeyer name for a fused ring system), from the table of retained trivial names, from the strict path when the name contains a part the engine records as not the preferred form (for example `(trimethylazaniumyl)acetate`, whose PIN is `(N,N-dimethylmethanaminiumyl)acetate`), or from the strict path for a class the Blue Book gives no PIN, such as organometallic compounds of the Group 1-12 metals (`ethenylsodium`) and compounds of aluminium, gallium, indium and thallium. A name reached through another drawing of the same compound is here too, as are the zinc names of [Metal compounds](#metal-compounds) that carry no PIN label. A natural-product parent name from the exact-match list lands here (`verified` `identity`), because the recommendations identify no preferred names for natural products. A natural-product name lands here too when the molecule's preferred name is a bridged fused name the engine does not build yet (`diamorphine`, `(9R,13S,14S)-3-methoxy-17-methylmorphinan`). Where the engine builds that bridged fused name (morphine, codeine, oxycodone), it is the PIN and is returned instead of the natural-product name. At `best-effort` the `cis` or `trans` name of a monocycle's pair of stereocentres (`trans-cyclohexane-1,4-diol`) lands here too: it is a general name of the Blue Book, and the PIN of the same structure cites lowercase `r` and `s`, which OPSIN cannot read.

{tier}`best_effort`
: A name from the last-resort producers of the `best-effort` tier, or a name whose own string no round trip confirmed.

{tier}`abstain`
: No name. The row gives the reason code instead ([Declines](declines.md)).

The marks are the ones the [web app](use/browser.md) uses. It shows the five tiers with four marks, because {tier}`pin_unverified` and {tier}`systematic_verified` share the FALLBACK mark, and it has a fifth state, Error, for input it cannot read. The shape carries the tier and the colour only agrees, so the marks stay distinct in greyscale: {lamp}`pin` {lamp}`fallback` {lamp}`best_effort` {lamp}`abstain`

## Choosing a tier: `--emit-tier`

The default is `pin`: a verified PIN, or no name. That is a rule about which names to return, not a limit of the engine, and `best-effort` names many more molecules ([Accuracy](accuracy.md#the-two-tiers) has both tiers side by side). Ask for the `best-effort` tier for one SMILES, for a file, or from Python:

```console
$ orthonym "SMILES" --emit-tier best-effort
$ orthonym --batch molecules.smi --emit-tier best-effort
```

```python
from orthonym import Orthonym

namer = Orthonym(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)
namer.name("SMILES")
```

Wider tiers are opt-in:

| `--emit-tier` | What it adds |
|:--|:--|
| `pin` *(default)* | A name only when the pipeline can build the preferred IUPAC name (PIN), that is when the strict PIN path built the name and verified it ({tier}`pin_verified`). The exceptions are names from the natural-product and metal-complex lists, the name formats absent from OPSIN's grammar (inositols, and phanes unless `--opsin main` lets OPSIN itself read them; otherwise a phane name is checked by building its structure), PIN spellings that OPSIN reads back through a twin spelling of the same structure (`ethan(2H)ol`, `diphenyliodanuide`, `methane-SO-thioperoxol`) or through their acids (dianhydrides), PINs whose stereodescriptors OPSIN cannot read, for which the default tier compares the constitution, and, with `--trivial`, a retained trivial name. Otherwise it declines, with the reason code `NO_VERIFIED_PIN` when it built a name that is not a verified PIN. The rule applies to the default `--style pin`. |
| `valid` | Names from the general engine, each with an atom-coverage certificate and a full-InChIKey round trip. |
| `complete` | General names for aromatic and heterocyclic ring systems as well. |
| `best-effort` | The last-resort producers as well, and von Baeyer and spiro names for ring systems of up to 100 skeletal atoms and 15 rings (the other tiers build these names for ring systems of up to 40 skeletal atoms and 8 rings), and adducts with a one-atom ion such as chloride. OPSIN reads no lowercase `r` or `s`, so this tier returns, for a saturated four- or six-membered monocycle (a ring that shares no atom with another ring) whose two stereocentres each carry one ligand and one hydrogen atom, the name with `cis` or `trans` (`trans-cyclohexane-1,4-diol`, tier {tier}`systematic_verified`) when the engine built a name with `r` and `s` for the whole molecule and OPSIN reads the `cis`/`trans` form back to your structure; otherwise a last-resort producer names the structure, or, for a molecule that has no other name and that the default tier names with stereodescriptors OPSIN cannot read, the tier returns the name the default tier gives it, with the label the default tier gives it ({tier}`pin_unverified`); otherwise the tier declines. |
| `full-coverage` | The coordination-name builder for metal tetrapyrrole and corrin complexes as well; it builds a name or declines. |

The wider tiers are not simply "the default plus more". They also return the names the default tier declines, each labelled with its tier. At `valid`, `complete` and `best-effort` every name must pass a full-InChIKey round trip, except a name from the natural-product and metal-complex lists (`verified` `identity`), a twin-checked name for which the engine has no other spelling (a halogen `-uide` anion, the anion of a thioperoxol, an organometallic name with halido ligands; `verified` `opsin_twin`) and, at `best-effort`, the name of a molecule that has no other name when the default tier ships it because OPSIN cannot read its stereodescriptors (`gate_outcome` `self_consistency_constitution_only`, the label of the default tier), so the other name formats absent from OPSIN's grammar, which the default tier ships without a full read-back, are declined there ([How every name is checked](checking.md)). A PIN spelling that the default tier ships through its twin gets a name OPSIN reads there (`1,4-dihydro-1,4-epithionaphthalene`, `ethan-1-(2H1)ol`, `(hydroxysulfanyl)methane` for `methane-SO-thioperoxol`), with the label of that name.

## Metal compounds

A standard InChIKey does not record the bonds between a metal and its ligands, so one compound drawn in two ways has one key. Orthonym names a metal compound through another drawing of it when the keys are equal and a separate count of atoms and charges agrees ([How every name is checked](checking.md#compounds-of-metals)). That name describes the other drawing, so it is {tier}`systematic_verified`, never {tier}`pin_verified`, and the default tier declines it.

Zinc is admitted to every mechanism that names a metal compound. Other metals are admitted one at a time. The table shows zinc, drawn in several ways, at three tiers:

| Zinc drawn as | Name | `pin` (default) | `valid` | `best-effort` |
|:--|:--|:--|:--|:--|
| a zinc ion and two chloride ions | `zinc dichloride` | {tier}`pin_verified` | {tier}`pin_verified` | {tier}`pin_verified` |
| two zinc-chlorine bonds | `zinc dichloride` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |
| a zinc ion and two acetate ions | `zinc diacetate` | {tier}`pin_verified` | {tier}`pin_verified` | {tier}`pin_verified` |
| two acetates joined to zinc by dative bonds | `zinc diacetate` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |
| a zinc-carbon bond, two ethyl groups | `diethylzinc` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |
| a zinc ion and two ethanide ions | `zinc bis(ethan-1-ide)` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |
| the atom, an ion, an isotope | `zinc`, `zinc(2+)`, `(65Zn)zinc` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |
| a zinc ion and four ammonia molecules | `ammonia—zinc(2+) (4/1)` | declined | no name | {tier}`systematic_verified` |
| a zinc ion and the tetraphenylporphyrin dianion | `zinc 5,10,15,20-tetraphenylporphyrin-21,23-diide` | declined | {tier}`systematic_verified` | {tier}`systematic_verified` |

At `valid` the engine gives no name for the adducts of zinc (the ammine of the table, the aqua ion, a complex with a neutral ligand such as pyridine, a zincate anion drawn alone, a porphyrin drawn beside a zinc ion), for a phthalocyanine complex, or for an organozinc compound with a halogen or an ester on its organyl group (`(3-ethoxy-3-oxopropyl)zinc chloride`). `best-effort` names them.

A zinc salt drawn as ions is {tier}`pin_verified` at the default tier when the strict PIN path builds its name and the name carries a PIN label: salts of acids, alcohols and thiols, and carbon-free salts. The names of organozinc compounds, of drawings with bonds to zinc, of the atom and its ions, of carbanion salts, of ammines and of complexes carry no PIN label, so the default tier declines them.

A compound of a metal that is not admitted is named where the general producers can name it (`copper(II) sulfate`, {tier}`pin_verified`) and is otherwise declined, usually with `UNSUPPORTED_ELEMENT` ([Declines](declines.md)).

## One molecule, three tiers

*cis*-Decalin shows all of this at once. The default tier gives a name in preferred-name form that OPSIN read back with the right constitution only:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --provenance
{"name": "(4as,8as)-decahydronaphthalene", "tier": "pin_unverified", "is_pin": false, "source": "pin_path", "opsin": "verified_constitution_only", "gates_passed": ["self_consistency_constitution_only"], "gate_outcome": "self_consistency_constitution_only", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin_constitution", "spelling_failures": [], "producer": "inner:tier_a_ring"}
```

At `valid` that name does not pass the full round trip, and no general-engine name does either, so the engine declines:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier valid

C1CC[C@H]2CCCC[C@H]2C1	UNNAMEABLE: unknown organic compound
```

The first line is the empty name, the second the reason on the error stream (see [Declines](declines.md)).

At `best-effort` a last-resort producer builds a name that OPSIN reads back to the same molecule, stereochemistry included:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier best-effort --provenance
{"name": "cis-decahydronaphthalene", "tier": "best_effort", "is_pin": false, "source": "t4_floor", "opsin": "verified", "gates_passed": ["full_key_round_trip"], "gate_outcome": "full_key_round_trip_verified", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin", "spelling_failures": [], "producer": "engine:t4"}
```

The `producer` shows the change: `inner:tier_a_ring` built the default-tier name, and the last-resort namer (`engine:t4`) built the `best-effort` one.

At a tier other than `pin`, the provenance row of a decline has `"name": null`, and `--on-failure label` prints `(no name — CODE)` on the command line.

## In Python

The tiers are keyword switches of `Orthonym`; the table is on the [Python](use/python.md) page.
