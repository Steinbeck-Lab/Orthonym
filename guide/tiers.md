# Output tiers

Every name Orthonym returns carries a tier. The tier says **how the name was built**. The `verified` field of the [provenance row](use/provenance.md) says **whether OPSIN read it back**. Read the two together.

## The five tiers

{tier}`pin_verified`
: The strict PIN path built the name and verified it: OPSIN read it back to your structure, or, for a name from the exact-match lists, your structure matched the list entry exactly: a metal-complex name by your structure's InChIKey, a natural-product parent name by its exact structure (`verified` `identity`). A compound without carbon (`sodium chloride`, `sulfuric acid`, `ammonia`) carries the tier its naming path gives it. `is_pin` is `true` only here.

{tier}`pin_unverified`
: A name in PIN form that only a breadth producer built, so its preferred status is not certified. It is also the tier of a strict-path name that OPSIN read back only in part (its constitution) or not at all. The `verified` field tells which.

{tier}`systematic_verified`
: A correct systematic name that is not the PIN: from the general engine (for example, a von Baeyer name for a fused ring system), from the table of retained trivial names, from the strict path when the name contains a part the engine records as not the preferred form (for example `(trimethylazaniumyl)acetate`, whose PIN is `(N,N-dimethylmethanaminiumyl)acetate`), or from the strict path for a class the Blue Book gives no PIN, such as organometallic compounds of the Group 1-12 metals (`ethenylsodium`) and compounds of aluminium, gallium, indium and thallium. A natural-product name lands here too when the molecule's preferred name is a bridged fused name the engine does not build yet (`diamorphine`, `(9R,13S,14S)-3-methoxy-17-methylmorphinan`). Where the engine builds that bridged fused name (morphine, codeine, oxycodone), it is the PIN and is returned instead of the natural-product name.

{tier}`best_effort`
: A name from the last-resort producers of the `best-effort` tier, or a name whose own string no round trip confirmed.

{tier}`abstain`
: No name. The row gives the reason code instead ([Declines](declines.md)).

The marks are the ones the [web app](use/browser.md) uses. It shows the five tiers with four marks, because {tier}`pin_unverified` and {tier}`systematic_verified` share the FALLBACK mark, and it has a fifth state, Error, for input it cannot read. The shape carries the tier and the colour only agrees, so the marks stay distinct in greyscale: {lamp}`pin` {lamp}`fallback` {lamp}`best_effort` {lamp}`abstain`

## Choosing a tier: `--emit-tier`

The default is `pin`. Wider tiers are opt-in:

| `--emit-tier` | What it adds |
|:--|:--|
| `pin` *(default)* | A name only when the pipeline can build the preferred IUPAC name (PIN), that is when the strict PIN path built the name and verified it ({tier}`pin_verified`). The exceptions are names from the natural-product and metal-complex lists, the name formats absent from OPSIN's grammar (for example inositols, phanes and thioperoxols), PINs whose stereodescriptors OPSIN cannot read, for which the default tier compares the constitution, and, with `--trivial`, a retained trivial name. Otherwise it declines, with the reason code `NO_VERIFIED_PIN` when it built a name that is not a verified PIN. The rule applies to the default `--style pin`. |
| `valid` | Names from the general engine, each with an atom-coverage certificate and a full-InChIKey round trip. |
| `complete` | General names for aromatic and heterocyclic ring systems as well. |
| `best-effort` | The last-resort producers as well, and von Baeyer and spiro names for ring systems of up to 100 skeletal atoms and 11 rings (the other tiers build these names for ring systems of up to 40 skeletal atoms and 8 rings), and adducts with a one-atom ion such as chloride. |
| `full-coverage` | The coordination-name builder for metal tetrapyrrole and corrin complexes as well; it builds a name or declines. |

The wider tiers are not simply "the default plus more". They also return the names the default tier declines, each labelled with its tier. At `valid`, `complete` and `best-effort` every name must pass a full-InChIKey round trip, except a name from the natural-product and metal-complex lists (`verified` `identity`), so the name formats absent from OPSIN's grammar, which the default tier ships without a full read-back, are declined there ([How every name is checked](checking.md)).

## One molecule, three tiers

*cis*-Decalin shows all of this at once. The default tier gives a name in preferred-name form that OPSIN read back with the right constitution only:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --provenance
{"name": "(4as,8as)-decahydronaphthalene", "tier": "pin_unverified", "is_pin": false, "source": "pin_path", "opsin": "verified_constitution_only", "gates_passed": ["self_consistency_constitution_only"], "gate_outcome": "self_consistency_constitution_only", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin_constitution"}
```

At `valid` that name does not pass the full round trip, and no general-engine name does either, so the engine declines:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier valid
(no name — UNNAMEABLE)
```

At `best-effort` a last-resort producer builds a name that OPSIN reads back to the same molecule, stereochemistry included:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier best-effort --provenance
{"name": "cis-bicyclo[4.4.0]decane", "tier": "best_effort", "is_pin": false, "source": "t4_floor", "opsin": "verified", "gates_passed": ["full_key_round_trip"], "gate_outcome": "full_key_round_trip_verified", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "opsin"}
```

At a tier other than `pin`, a decline prints `(no name — CODE)` on the command line, and the provenance row has `"name": null`.

## In Python

The tiers are keyword switches of `Orthonym`; the table is on the [Python](use/python.md) page.
