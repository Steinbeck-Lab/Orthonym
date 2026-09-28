# Batch files

Put one SMILES per line in a text file. Empty lines are skipped.

```console
$ cat molecules.smi
CCO
c1ccccc1
O=[U](=O)=O
$ orthonym --batch molecules.smi
CCO	ethanol
c1ccccc1	benzene
O=[U](=O)=O	inorganic compound (not supported)
```

Each output line is the SMILES, a tab, and the name or the label for a decline. Write the lines to a file with `--output`, and add `--verbose` for a count at the end:

```console
$ orthonym --batch molecules.smi --output names.txt --verbose
Processed 3 SMILES, 0 errors
Output written to: names.txt
```

A line that RDKit cannot read does not stop the run. It gets `ERROR:` and the reason, and the exit status is then `1`:

```console
$ orthonym --batch bad.smi
CCO	ethanol
not-a-smiles	ERROR: Invalid SMILES: not-a-smiles
CC(=O)O	acetic acid
```

Batch runs use the default tier. `--emit-tier` and `--provenance` do not apply to `--batch` yet. For tiers or provenance rows over many molecules, loop in Python with one `Orthonym` instance:

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
