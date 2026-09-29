# Output tiers

Every name Orthonym returns carries a tier. The tier says **how the name was built**. The `verified` field of the [provenance row](use/provenance.md) says **whether OPSIN read it back**. Read the two together.

## The five tiers

{tier}`pin_verified`
: The strict path for the Preferred IUPAC Name built the name, certified it as the preferred name, and OPSIN read it back to your structure. `is_pin` is `true` only here.

{tier}`pin_unverified`
: A name in preferred-name form whose preferred status the engine does not certify: a producer outside the strict path built it, or it carries a part that is not the preferred form. It is also the tier of a strict-path name that OPSIN read back only in part (its constitution) or not at all. The `verified` field tells which.

{tier}`systematic_verified`
: A checked name that is not certified as the preferred name: from the general engine, from the table of retained trivial names, from the exact-match list of metal complexes (matched by InChIKey, `verified` `identity`), or from the strict path for a class that has no preferred-name status, such as organometallic compounds and compounds without carbon (`iron(III) trichloride`, `sulfuric acid`, `ammonia`).

{tier}`best_effort`
: A name from the last-resort producers of the `best-effort` tier, or a name whose own string no round trip confirmed.

{tier}`abstain`
: No name. The row gives the reason code instead ([Declines](declines.md)).

The marks are the ones the [web app](use/browser.md) uses. It shows the five tiers with four marks, because {tier}`pin_unverified` and {tier}`systematic_verified` share the FALLBACK mark, and it has a fifth state, Error, for input it cannot read. The shape carries the tier and the colour only agrees, so the marks stay distinct in greyscale: {lamp}`pin` {lamp}`fallback` {lamp}`best_effort` {lamp}`abstain`

## Choosing a tier: `--emit-tier`

The default is `pin`. Wider tiers are opt-in:

| `--emit-tier` | What it adds |
|:--|:--|
| `pin` *(default)* | The name from the strict path for the Preferred IUPAC Name. Where the engine cannot certify the preferred name it can still return a name, labelled with a lower tier, or it declines. |
| `valid` | Names from the general engine, each with an atom-coverage certificate and a full-InChIKey round trip. |
| `complete` | General names for aromatic and heterocyclic ring systems as well. |
| `best-effort` | The last-resort producers as well. |
| `full-coverage` | The coordination-name builder for metal tetrapyrrole and corrin complexes as well; it builds a name or declines. |

The wider tiers are not simply "the default plus more". At `valid`, `complete` and `best-effort` every name must pass a full-InChIKey round trip, except a metal-complex name from the exact-match list ({tier}`systematic_verified`, `verified` `identity`), so a few name classes that the default tier ships without a full read-back are declined there ([How every name is checked](checking.md)).

## One molecule, three tiers

*cis*-Decalin shows all of this at once. The default tier gives a name in preferred-name form that OPSIN read back with the right constitution only:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --provenance
{"name": "(4as,8as)-decahydronaphthalene", "tier": "pin_unverified", "is_pin": false, "source": "pin_path", "opsin": "verified_constitution_only", "gates_passed": ["self_consistency_constitution_only"], "gate_outcome": "self_consistency_constitution_only", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "verified": "opsin_constitution"}
```

At `valid` that name does not pass the full round trip, and no general-engine name does either, so the engine declines:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier valid
(no name — UNNAMEABLE)
```

At `best-effort` a last-resort producer builds a name that OPSIN reads back to the same molecule, stereochemistry included:

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --emit-tier best-effort --provenance
{"name": "cis-bicyclo[4.4.0]decane", "tier": "best_effort", "is_pin": false, "source": "t4_floor", "opsin": "verified", "gates_passed": ["full_key_round_trip"], "gate_outcome": "full_key_round_trip_verified", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "verified": "opsin"}
```

At a tier other than `pin`, a decline prints `(no name — CODE)` on the command line, and the provenance row has `"name": null`.

## In Python

The tiers are keyword switches of `Orthonym`; the table is on the [Python](use/python.md) page.
