# Python

Everything the command line does is one import away. The full parameter lists are on the [Python API](../reference/python-api.md) page.

## One molecule, one name

```pycon
>>> from orthonym import name_compound
>>> name_compound("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
'(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid'
```

`name_compound` returns a string: the name, or the empty string when there is none. The default is the `best-effort` tier: the verified PIN when the strict PIN path builds one, and otherwise a systematic name that is not the PIN. `emit_tier="pin"` returns the verified PIN or nothing, which was the default up to release 1.0.x, and the other tiers of `--emit-tier` are values of the same argument. The call logs nothing about it; the command line prints a note, and `Orthonym.name_tiered` or `include_confidence=True` (for a tier above `pin`) tells the tier of a name:

```pycon
>>> name_compound("OCCNCc1ccccc1")
'2-(benzylamino)ethan-1-ol'
>>> name_compound("OCCNCc1ccccc1", emit_tier="pin")
''
>>> name_compound("OCCNCc1ccccc1", include_confidence=True)["tier"]
'pin_unverified'
```

```pycon
>>> name_compound("O=[U](=O)=O")
''
```

## When there is no name: `on_failure`

A label such as `inorganic compound (not supported)` is a reason, not a name, so by default it is not returned. The reason is logged instead, as one warning on the logger `orthonym` that holds the SMILES, the limit code and the label (`UNSUPPORTED_ELEMENT: inorganic compound (not supported)`). The record carries the parts as attributes `orthonym_smiles`, `orthonym_code` and `orthonym_label`, so a handler can read them without parsing the text.

`on_failure` chooses what a call gives for a structure with no name, in `name_compound` and in `Orthonym.name`:

| `on_failure` | the call returns |
|:--|:--|
| `"empty"` (the default) | `""`, and the reason is logged |
| `"label"` | the label, as earlier versions did |
| `"raise"` | nothing: it raises `OrthonymLimitError` (`raise_on_limit=True` is the same) |

```pycon
>>> name_compound("O=[U](=O)=O", on_failure="label")
'inorganic compound (not supported)'
```

The codes (the keys of `orthonym.errors.LIMIT_CATALOG`) are the stable part; the texts of labels can change. `is_failure_name` is true for the empty string and for a label, and false for a name:

```pycon
>>> from orthonym.errors import is_failure_name
>>> is_failure_name(name_compound("O=[U](=O)=O"))
True
```

The choice is the caller's: the engine's own calls to name the parts of a molecule always get labels. With `include_confidence=True` the `"name"` of the dictionary follows `on_failure`, and its `"limit"` entry holds the code. `Orthonym.name_tiered` and `name_with_tree` are unchanged: a row says `abstain` and gives a `limit_code`.

A SMILES that RDKit cannot read raises `ValueError`:

```pycon
>>> name_compound("C1CC")
Traceback (most recent call last):
  ...
ValueError: Invalid SMILES: C1CC
```

## Many molecules, one set of options: `Orthonym`

`name_compound` sets up a fresh engine for every call. For a loop, or for the wider tiers, build one `Orthonym` and reuse it. `Orthonym()` with no argument is the engine of the `pin` tier, as before; the default of `name_compound` is the `best-effort` row. The keyword switches are the ones behind `--emit-tier`:

| `--emit-tier` | `Orthonym(...)` |
|:--|:--|
| `pin` | `Orthonym()` |
| `valid` | `Orthonym(general_fallback=True)` |
| `complete` | `Orthonym(general_fallback=True, allow_aromatic_general=True)` |
| `best-effort` (the default of `name_compound`) | `Orthonym(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)` |
| `full-coverage` | as `best-effort`, plus `full_coverage=True` |

```pycon
>>> from orthonym import Orthonym
>>> namer = Orthonym(general_fallback=True)
>>> namer.name("CCCCCCCCCCCCC/C=C/[C@H]([C@H](CO)N)O")
'(2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol'
```

## The name and how it was checked: `name_tiered`

`Orthonym.name_tiered` returns the provenance row as a dictionary, the same row `--provenance` prints:

```pycon
>>> best_effort = Orthonym(general_fallback=True, allow_aromatic_general=True,
...                        general_fallback_unverified=True)
>>> row = best_effort.name_tiered("C1CC[C@H]2CCCC[C@H]2C1")
>>> row["name"], row["tier"], row["verified"]
('cis-decahydronaphthalene', 'best_effort', 'opsin')
```

Each key is explained on [The provenance row](provenance.md).

## The parts of a name: `name_with_tree`

```pycon
>>> from orthonym import name_with_tree
>>> result = name_with_tree("OC1CCCCC1")
>>> result.name
'cyclohexanol'
>>> result.tree.parent_stem, result.tree.suffix
('cyclohex', 'ol')
```

## Asking first: `classify_limit`

`classify_limit` says whether a structure is out of scope, without raising:

```pycon
>>> from orthonym import classify_limit
>>> classify_limit("O=[U](=O)=O").code
'UNSUPPORTED_ELEMENT'
>>> classify_limit("CCO") is None
True
```

To get an exception instead of an empty name, pass `on_failure="raise"` (or `raise_on_limit=True`):

```pycon
>>> from orthonym import OrthonymLimitError
>>> try:
...     Orthonym().name("O=[U](=O)=O", on_failure="raise")
... except OrthonymLimitError as err:
...     print(err.code, "|", err.message)
UNSUPPORTED_ELEMENT | inorganic compound (not supported)
```
