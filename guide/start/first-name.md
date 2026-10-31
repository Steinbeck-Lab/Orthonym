# Your first name

Give Orthonym a structure written as SMILES. It prints the name.

## From the command line

```console
$ orthonym "CCO"
ethanol
```

Put the SMILES in quotes: characters such as `(`, `=` and `#` mean something to the shell. `python -m orthonym "CCO"` does the same.

Five more, each the engine's own output:

```console
$ orthonym "CC(=O)Oc1ccccc1C(=O)O"
2-(acetyloxy)benzoic acid
$ orthonym "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione
$ orthonym "C/C=C/C"
(2E)-but-2-ene
$ orthonym "C[C@H](O)CC"
(2S)-butan-2-ol
$ orthonym "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid
```

That is aspirin, caffeine, *trans*-but-2-ene, (*S*)-butan-2-ol and (*R*)-ibuprofen. The stereodescriptors in the SMILES (`/`, `\`, `@`, `@@`) come back as *E*, *Z*, *R* and *S* in the name.

## From Python

```pycon
>>> from orthonym import name_compound
>>> name_compound("CCO")
'ethanol'
>>> name_compound("CC(=O)Oc1ccccc1C(=O)O")
'2-(acetyloxy)benzoic acid'
>>> name_compound("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
'1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
>>> name_compound("C/C=C/C")
'(2E)-but-2-ene'
>>> name_compound("C[C@H](O)CC")
'(2S)-butan-2-ol'
```

## How sure is the name? Ask for the provenance

Add `--provenance` and Orthonym prints one row of JSON instead of the bare name. It says which tier the name earned and whether OPSIN read it back to your structure:

```console
$ orthonym "CC(=O)Oc1ccccc1C(=O)O" --provenance | python -m json.tool
{
    "name": "2-(acetyloxy)benzoic acid",
    "tier": "pin_verified",
    "is_pin": true,
    "source": "pin_path",
    "opsin": "verified",
    "gates_passed": [
        "self_consistency"
    ],
    "gate_outcome": "self_consistency_verified",
    "formula": null,
    "limit_code": null,
    "stereo_unexpressed": false,
    "suffix_free_prefix_name": false,
    "prefix_order_fallback": false,
    "verified": "opsin",
    "spelling_failures": [],
    "producer": "inner:tier_a_ring"
}
```

The two lines to read first:

- `tier` is {tier}`pin_verified`: the strict path for the Preferred IUPAC Name built the name and verified it. The other tiers are on [Output tiers](../tiers.md).
- `verified` is `opsin`: OPSIN read the name back to the same molecule.

Every other field is explained on [The provenance row](../use/provenance.md), `producer` (the code that built the name) among them.

## When there is no name

Orthonym says so. The plain call prints an empty name and, on the error stream, the reason (a limit code and a label); `--provenance` gives the reason as a row:

```console
$ orthonym "F[Xe]F"

F[Xe]F	UNSUPPORTED_ELEMENT: inorganic compound (not supported)
```

More on [declines](../declines.md).

## Which names you get: the default and `--emit-tier pin`

Everything above uses the default, the `best-effort` tier. It returns the verified Preferred IUPAC Name (PIN) when the strict PIN path builds one, and otherwise a systematic name that is not the PIN, labelled with its tier. When a name is below {tier}`pin_verified`, `orthonym` says so once on the error stream, and `--provenance` shows its tier:

```console
$ orthonym "OCCNCc1ccccc1"
2-(benzylamino)ethan-1-ol
orthonym: note: the default returns the PIN when it can, else a systematic name that is not the PIN; --provenance shows the tier and the check of each name, --emit-tier pin gives PIN only
```

For the PIN only, a name or no name, ask for `--emit-tier pin`; this was the default up to release 1.0.x:

```console
$ orthonym "SMILES" --emit-tier pin
$ orthonym --batch molecules.smi --emit-tier pin
```

```python
from orthonym import name_compound

name_compound("SMILES", emit_tier="pin")
```

On 2,000 held-out ChEBI molecules the default names 1,994 (99.70 %) and `--emit-tier pin` names 1,106 (55.30 %); [Accuracy](../accuracy.md#the-two-tiers) has the numbers for both on three sets, and [Output tiers](../tiers.md) says what each tier adds.
