# The provenance row

`orthonym "SMILES" --provenance` prints one row of JSON instead of the bare name, and `Orthonym.name_tiered` returns the same row as a dictionary. It says what the name is, how it was built, which code built it and how it was checked.

```console
$ orthonym "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --provenance
{"name": "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", "tier": "pin_verified", "is_pin": true, "source": "pin_path", "opsin": "verified", "gates_passed": ["self_consistency"], "gate_outcome": "self_consistency_verified", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin", "spelling_failures": [], "producer": "inner:tier_a_ring"}
```

## Every field

`name`
: The name, or the label for a decline (the row is the record of the decline, so `--on-failure` does not change it). With `--emit-tier` other than `pin`, a decline gives `null` here.

`tier`
: How the name was built: {tier}`pin_verified`, {tier}`pin_unverified`, {tier}`systematic_verified`, {tier}`best_effort` or {tier}`abstain`. See [Output tiers](../tiers.md).

`is_pin`
: `true` only for a certified Preferred IUPAC Name.

`source`
: Which part of the engine produced the name: `pin_path` (the strict path; at `best-effort` it also labels the `cis` or `trans` name of a monocycle's pair of stereocentres, which is made from a name with lowercase `r` and `s` that the engine built for the whole molecule and OPSIN cannot read, with the tier {tier}`systematic_verified`, and the name the default tier ships although OPSIN cannot read its stereodescriptors, for a molecule that has no other name, with the tier {tier}`pin_unverified`), `general_engine`, `trivial_retained` (a retained trivial name: from the last-resort table, or a trivial natural-product name such as `diamorphine`), `t4_floor` (a last-resort producer) or `abstain`. It is coarse; `producer` is the finer field.

`opsin`
: What the OPSIN check found: `verified`, `verified_constitution_only` (the constitution matched; the stereodescriptors were not compared), `verified_twin` (OPSIN has no grammar for the name and read a twin spelling of it back to your structure), `unverified`, or `n/a` when there is no name.

`gates_passed`
: The checks this name passed, for example `self_consistency` (OPSIN read the name back to your structure), `atom_coverage` (every atom is named), `full_key_round_trip` or `twin_round_trip` (OPSIN read a twin spelling of the name back to your structure).

`gate_outcome`
: What the final OPSIN check did for this name: `self_consistency_verified`, `full_key_round_trip_verified`, `self_consistency_constitution_only`, `suppressed` (a candidate failed and was withdrawn), `not_run`, `unavailable` (no OPSIN check ran: the opt-in reduced mode without the jars), `carveout:<class>` for a name class that OPSIN cannot read, or `twin_verified:<rules>` for a name read back through its twin spelling (the ids of the rewrites that made the twin, joined by `+`).

`formula`
: The molecular formula, given when there is no name, so a decline still tells you what came in.

`limit_code`
: The reason code of a decline, for example `UNSUPPORTED_ELEMENT` or, at the default tier, `NO_VERIFIED_PIN` (the engine built a checked name that is not a verified PIN). See [Declines](../declines.md).

`stereo_unexpressed`
: `true` when your structure configures a stereo element (a stereocentre, or a double bond drawn with `/` and `\`) that the name does not state. The check counts the stereo elements that OPSIN's read-back of the name configures (the read-back the final check keeps, so a name it has read is not read again): fewer than your structure configures sets the flag. It is for a name of any producer, and it changes nothing else: not the name, not the tier, not `is_pin`. A Preferred IUPAC Name can carry it. `C[N+](C)=C/C=C/C=C/N1CCCC1` gives `1-[5-(dimethylamino)penta-2,4-dien-1-ylidene]pyrrolidin-1-ium`, {tier}`pin_verified`, with the flag `true`: the two double bonds you configured are single bonds in this form of the cation, so the name cannot state them, and the standard InChIKey does not see them, so the read-back still matches. The flag is `false` when the read-back states every element, and when there is no read-back, because OPSIN cannot read the name (a name with lowercase `r` and `s`, a name from an exact-match list): the flag does not guess from the text of the name.

`suffix_free_prefix_name`
: `true` when the name states the principal characteristic group as a prefix with no suffix, a form the recommendations do not allow.

`prefix_order_fallback`
: `true` when a {tier}`best_effort` name cites its substituent prefixes in an order other than the alphanumerical one, because the name in the standard order did not pass its round trip for its stereodescriptors alone. Such a name is never a Preferred IUPAC Name; the default tier does not use it.

`verified`
: Whether the name was read back, in one word:
  `opsin` (OPSIN read the name back to the same molecule),
  `opsin_constitution` (OPSIN read it back with the same constitution; the stereodescriptors were not confirmed by OPSIN),
  `identity` (a name from an exact-match list: a metal-complex name found by your structure's exact InChIKey, or a natural-product parent name found by its exact structure; OPSIN cannot read these names),
  `opsin_split` (OPSIN read the name back to your structure drawn without its dative bonds, because RDKit gives a drawing with dative bonds no standard InChIKey to read a name back against; a count of atoms and charges ties the two drawings),
  `opsin_twin` (a PIN spelling OPSIN has no grammar for: OPSIN read a twin spelling of the same structure back to your structure; [How every name is checked](../checking.md#where-opsin-cannot-read-the-name)),
  or `unverified` (no read-back recorded).

`spelling_failures`
: The spelling rules of the Preferred IUPAC Name that a name in preferred-name form breaks, as a list of `{"rule": ..., "detail": ...}` entries, and the rule of a part of the name that the engine records as not the preferred form. A name reached through another drawing of the same compound is listed here too, and the `detail` says so. OPSIN cannot see these: a name can read back to your structure and still break one. Such a name is {tier}`systematic_verified` at most; the default tier declines it with `NO_VERIFIED_PIN` and keeps the entries as the reason. The list is empty otherwise.

`producer`
: The code that built the name that ships, for example `inner:tier_a_ring` or `engine:general`; `null` when no name ships, a default-tier `NO_VERIFIED_PIN` decline included. See [Which code built the name](#which-code-built-the-name).

## Which code built the name

`source` has five values and records the last producer that wrote it. `producer` is the finer field: the id of the code that built the name that ships. Use it to find the code to look at before you change a rule ([Architecture](../architecture.md#before-you-edit-a-function)), or to count which producers name a set of molecules.

- **The deepest builder.** `producer` names the code that built the string, not the class that passed the molecule on. A peptide whose name an inner handler composed shows that handler, not `outer:peptide`. When two codes return the same name, the first one to return it is reported, which is the innermost.
- **Another drawing of a metal compound.** A metal compound that is named through another drawing of it ([Metal compounds](../tiers.md#metal-compounds)) shows the code that built the name of that drawing: `Cl[Zn]Cl` is named through its ionic drawing and gives `outer:salt`, a bare zinc atom gives `outer:metal_atom`.
- **The `cis` or `trans` name of a ring pair.** At `best-effort`, a name with `cis` or `trans` is made from the name with lowercase `r` and `s` that a code built for the whole molecule ([Tiers](../tiers.md)). The row names that code: `trans-cyclohexane-1,4-diol` gives `inner:general_acyclic`, the code that built `(1r,4r)-cyclohexane-1,4-diol` at the default tier.
- **`null` when no name ships.** The default tier can build a name and withhold it (`NO_VERIFIED_PIN`, see [Declines](../declines.md)). Nothing ships then, so no code is named; a wider tier returns the name, and its row names the code.
- **A diagnostic, not a contract.** The ids follow the code's class and handler ids and can change in any release: count with them, do not parse them.
- **Only the row has it.** `Orthonym.name_tiered` and `--provenance` return the key; `name_compound` returns the bare name.

| `producer` | The code that built the name |
|:--|:--|
| `outer:<class>` | A class of the outer table, with the class id in lower case: `outer:retained_name`, `outer:natural_product`, `outer:peptide`. The [rule index](../rule-index.md#from-a-handler-id-to-its-code) lists the classes and the file of each. |
| `inner:<handler>` | An inner handler: `inner:tier_a_ring` (rings), `inner:ester_family` (acids, alcohols and esters), `inner:general_acyclic` (chains). The same page lists the handlers. |
| `composer` | The composer assembled the name itself; no inner handler built it. |
| `engine:general` | The general engine. |
| `engine:t4` | The last-resort namer of the `best-effort` tier, with its variant that cites the substituent prefixes in another order. |
| `decomposition` | The fragment-based namer, which names a large structure from its parts. |
| `multifragment` | The recovery for an input of several components: each component is named and the names are joined. |
| `retained_fused` | A retained fused-ring name for a molecule that holds a purine without being a fused ring itself, such as an acyl coenzyme A or ADP-ribose. |
| `isotope` | The isotope step, which names an isotopically labelled molecule by adding the isotope descriptors to the name of its skeleton. |
| `table:coordination` | The list of metal-complex names, matched by the exact structure (`verified` is `identity`). |
| `table:trivial` | The last-resort table of retained trivial names. |
| `table:np_parent` | The list of natural-product parent names, matched by the exact structure, used as a fallback. |
| `table:retained` | A retained or fused-ring name for a molecule that is exactly such a structure, which the late recovery prefers to the engine's own name when it reads back. |

## Read `tier` and `verified` together

The tier says how a name was built. The `verified` field says whether OPSIN read it back. The two can differ: at the default tier, *cis*-decalin gets a name in preferred-name form that OPSIN read back with the right constitution only.

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --provenance
{"name": "(4as,8as)-decahydronaphthalene", "tier": "pin_unverified", "is_pin": false, "source": "pin_path", "opsin": "verified_constitution_only", "gates_passed": ["self_consistency_constitution_only"], "gate_outcome": "self_consistency_constitution_only", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin_constitution", "spelling_failures": [], "producer": "inner:tier_a_ring"}
```

`--provenance` works for one SMILES at a time; `--batch` ignores it. For many molecules, loop over `Orthonym.name_tiered` as shown on [Batch files](batch.md).
