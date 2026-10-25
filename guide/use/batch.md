# Batch files

Put one SMILES per line in a text file. An empty line gets an empty output line, so line N of the output belongs to line N of the input.

```console
$ cat molecules.smi
CCO
c1ccccc1
O=[U](=O)=O
$ orthonym --batch molecules.smi
CCO	ethanol
c1ccccc1	benzene
O=[U](=O)=O	
```

Each output line is the SMILES, a tab, and the name; the name is empty for a line that gets none, so line N of the output belongs to line N of the input. The reason for each such line, with its limit code, goes to the error stream:

```console
O=[U](=O)=O	UNSUPPORTED_ELEMENT: inorganic compound (not supported)
```

`--on-failure label` writes the label in the name column instead, and `--on-failure raise` gives the line `ERROR:` and the reason and ends the run with status `1` (see [Declines](../declines.md)). Write the lines to a file with `--output`, and add `--verbose` for a count at the end:

```console
$ orthonym --batch molecules.smi --output names.txt --verbose
Processed 3 SMILES, 0 errors, 1 unnamed
Output written to: names.txt
```

A line that RDKit cannot read does not stop the run. It gets `ERROR:` and the reason, and the exit status is then `1`:

```console
$ orthonym --batch bad.smi
CCO	ethanol
not-a-smiles	ERROR: Invalid SMILES: not-a-smiles
CC(=O)O	acetic acid
```

## Tiers and provenance rows

Batch runs use the default tier, unless you give `--emit-tier`. It works as it does for one SMILES. The second molecule here gets no name at the default tier and a name at `best-effort`:

```console
$ cat wide.smi
CCO
CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1
O=[U](=O)=O
$ orthonym --batch wide.smi --emit-tier best-effort
CCO	ethanol
CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1	2-{4-[4-(1-benzofuran-2-yl)-1,3-thiazol-2-yl]piperidin-1-yl}-N-(3-fluorophenyl)-2-methylpropanamide
O=[U](=O)=O	
```

A line that gets no name at the tier you chose follows `--on-failure`, as it does at the default tier. The whole file is named with one `Orthonym` instance, as in the Python loop below.

`--provenance` puts the row of [The provenance row](provenance.md) in the second column, on one line, in place of the name. It has the same keys as the row for one SMILES, and `--on-failure` does not change it: a line with no name has `limit_code` and, with `--emit-tier` other than `pin`, `null` as its `name`.

```console
$ orthonym --batch wide.smi --provenance
CCO	{"name": "ethanol", "tier": "pin_verified", "is_pin": true, ...}
```

`--confidence` cannot be combined with `--provenance` or with an `--emit-tier` other than `pin`, because a row has no coverage score. `orthonym` stops with a usage error and status `2`.

The flags that look inside the naming of one structure cannot be combined with `--batch`: `--validation-stats`, `--dispatch-stats`, `--engine-only`, `--dump-tree` and `--format` (which only sets the layout of `--dump-tree`). `orthonym` stops with a usage error and status `2`, before the file is read, and names the flags it refused. To look inside the naming of one line, run `orthonym` on that SMILES with the flag.

For rows with more keys, or to make the choices of the `Orthonym` constructor that the command line does not offer, loop in Python with one `Orthonym` instance:

```python
import json
from orthonym import Orthonym

namer = Orthonym(general_fallback=True)          # the valid tier
with open("molecules.smi") as src, open("rows.jsonl", "w") as out:
    for line in src:
        smiles = line.strip()
        if smiles:
            out.write(json.dumps({"smiles": smiles, **namer.name_tiered(smiles)}) + "\n")
```
