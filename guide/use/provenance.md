# The provenance row

`orthonym "SMILES" --provenance` prints one row of JSON instead of the bare name, and `Orthonym.name_tiered` returns the same row as a dictionary. It says what the name is, how it was built and how it was checked.

```console
$ orthonym "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --provenance
{"name": "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", "tier": "pin_verified", "is_pin": true, "source": "pin_path", "opsin": "verified", "gates_passed": ["self_consistency"], "gate_outcome": "self_consistency_verified", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "verified": "opsin"}
```

## Every field

`name`
: The name, or the label for a decline. With `--emit-tier` other than `pin`, a decline gives `null` here.

`tier`
: How the name was built: {tier}`pin_verified`, {tier}`pin_unverified`, {tier}`systematic_verified`, {tier}`best_effort` or {tier}`abstain`. See [Output tiers](../tiers.md).

`is_pin`
: `true` only for a certified Preferred IUPAC Name.

`source`
: Which part of the engine produced the name: `pin_path` (the strict path), `general_engine`, `trivial_retained` (the table of retained trivial names), `t4_floor` (a last-resort producer) or `abstain`.

`opsin`
: What the OPSIN check found: `verified`, `verified_constitution_only` (the constitution matched; the stereodescriptors were not compared), `unverified`, or `n/a` when there is no name.

`gates_passed`
: The checks this name passed, for example `self_consistency` (OPSIN read the name back to your structure), `atom_coverage` (every atom is named) or `full_key_round_trip`.

`gate_outcome`
: What the final OPSIN check did for this name: `self_consistency_verified`, `full_key_round_trip_verified`, `self_consistency_constitution_only`, `suppressed` (a candidate failed and was withdrawn), `not_run`, `unavailable` (OPSIN did not answer), or `carveout:<class>` for a name class that OPSIN cannot read.

`formula`
: The molecular formula, given when there is no name, so a decline still tells you what came in.

`limit_code`
: The reason code of a decline, for example `UNSUPPORTED_ELEMENT`. See [Declines](../declines.md).

`stereo_unexpressed`
: `true` when a stereocentre of your structure is not stated in the name.

`suffix_free_prefix_name`
: `true` when the name states the principal characteristic group as a prefix with no suffix, a form the recommendations do not allow.

`verified`
: Whether the name was read back, in one word:
  `opsin` (OPSIN read the name back to the same molecule),
  `opsin_constitution` (OPSIN read it back with the same constitution; the stereodescriptors were not confirmed by OPSIN),
  `identity` (a metal-complex name from the exact-match list, matched to your structure by InChIKey; OPSIN cannot read these names),
  or `unverified` (no read-back recorded).

## Read `tier` and `verified` together

The tier says how a name was built. The `verified` field says whether OPSIN read it back. The two can differ: at the default tier, *cis*-decalin gets a name in preferred-name form that OPSIN read back with the right constitution only.

```console
$ orthonym "C1CC[C@H]2CCCC[C@H]2C1" --provenance
{"name": "(4as,8as)-decahydronaphthalene", "tier": "pin_unverified", "is_pin": false, "source": "pin_path", "opsin": "verified_constitution_only", "gates_passed": ["self_consistency_constitution_only"], "gate_outcome": "self_consistency_constitution_only", "formula": null, "limit_code": null, "stereo_unexpressed": false, "suffix_free_prefix_name": false, "verified": "opsin_constitution"}
```

`--provenance` works for one SMILES at a time; `--batch` ignores it. For many molecules, loop over `Orthonym.name_tiered` as shown on [Batch files](batch.md).
