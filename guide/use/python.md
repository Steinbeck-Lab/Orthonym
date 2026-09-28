# Python

Everything the command line does is one import away. The full parameter lists are on the [Python API](../reference/python-api.md) page.

## One molecule, one name

```pycon
>>> from orthonym import name_compound
>>> name_compound("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
'(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid'
```

`name_compound` returns a string: the name, or a label when there is none. Tell the two apart with `is_failure_name`:

```pycon
>>> from orthonym.errors import is_failure_name
>>> is_failure_name(name_compound("O=[U](=O)=O"))
True
```

A SMILES that RDKit cannot read raises `ValueError`:

```pycon
>>> name_compound("C1CC")
Traceback (most recent call last):
  ...
ValueError: Invalid SMILES: C1CC
```

## Many molecules, one set of options: `Orthonym`

`name_compound` sets up a fresh engine for every call. For a loop, or for the wider tiers, build one `Orthonym` and reuse it. The keyword switches are the ones behind `--emit-tier`:

| `--emit-tier` | `Orthonym(...)` |
|:--|:--|
| `pin` (default) | `Orthonym()` |
| `valid` | `Orthonym(general_fallback=True)` |
| `complete` | `Orthonym(general_fallback=True, allow_aromatic_general=True)` |
| `best-effort` | `Orthonym(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)` |
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
('cis-bicyclo[4.4.0]decane', 'best_effort', 'opsin')
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

To get an exception instead of a label, pass `raise_on_limit=True`:

```pycon
>>> from orthonym import OrthonymLimitError
>>> try:
...     Orthonym().name("O=[U](=O)=O", raise_on_limit=True)
... except OrthonymLimitError as err:
...     print(err.code, "|", err.message)
UNSUPPORTED_ELEMENT | inorganic compound (not supported)
```
