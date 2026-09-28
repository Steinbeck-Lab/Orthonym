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
    "verified": "opsin"
}
```

The two lines to read first:

- `tier` is {tier}`pin_verified`: the strict path for the Preferred IUPAC Name built the name and certified it. The other tiers are on [Output tiers](../tiers.md).
- `verified` is `opsin`: OPSIN read the name back to the same molecule.

Every other field is explained on [The provenance row](../use/provenance.md).

## When there is no name

Orthonym says so. The plain call prints a label in place of a name, and the provenance row gives the reason:

```console
$ orthonym "O=[U](=O)=O"
inorganic compound (not supported)
```

More on [declines](../declines.md).
