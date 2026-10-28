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

## A time limit for each structure

A few structures take a long time (a large molecule, a cage ring system: some need half a minute or more), and in a long file they hold up everything behind them. `--time-limit SECONDS` gives each line at most that long. A line that is not named in time is a line with no name, in its own place, and the run goes on with the next one:

```console
$ cat slow.smi
CCO
CC1C(C=CC(C2CCC2CN3CC4(CCCC5=C4C=CC(=C5)Cl)COC6=C3C=C(C=C6)C(=O)N=[S-](=O)C1C(=O)N(C)OC7=NN(C=C7C(=O)N)C)OC)OC(=O)N(C)C
c1ccccc1
$ orthonym --batch slow.smi --time-limit 5
CCO	ethanol
CC1C(C=CC(C2CCC2CN3CC4(CCCC5=C4C=CC(=C5)Cl)COC6=C3C=C(C=C6)C(=O)N=[S-](=O)C1C(=O)N(C)OC7=NN(C=C7C(=O)N)C)OC)OC(=O)N(C)C	
c1ccccc1	benzene
```

The error stream has the reason of the second line, with the code `TIME_LIMIT`:

```console
CC1C(C=CC(C2CCC2CN3CC4(CCCC5=C4C=CC(=C5)Cl)COC6=C3C=C(C=C6)C(=O)N=[S-](=O)C1C(=O)N(C)OC7=NN(C=C7C(=O)N)C)OC)OC(=O)N(C)C	TIME_LIMIT: no name within the time limit of 5 s
```

The limit follows the other options for a structure that gets no name:

* `--on-failure empty` (the default): an empty name, and the reason `<SMILES><tab>TIME_LIMIT: no name within the time limit of 5 s` on the error stream.
* `--on-failure label`: the label `no name within the time limit of 5 s` in the name column.
* `--on-failure raise`: the line gets `ERROR: [TIME_LIMIT] no name within the time limit of 5 s`, the run goes on, and it ends with status `1`, as for any other limit.
* `--provenance`: the row has `"name": null`, `"tier": "abstain"` and `"limit_code": "TIME_LIMIT"`, and `--on-failure` does not change it. With `--confidence` the score column is `unverified` and the engine column `TIME_LIMIT`.

`--verbose` adds the number of lines that ran out of time, which are also counted as unnamed (as errors with `--on-failure raise`). `--output` writes the same lines to the file. The limit is for naming: the start-up of the run (the Java machine, OPSIN's tables, about 1.5 s) is paid before the first limit begins and is not counted against the first line.

The next line is named exactly as it is named without the limit: the limit puts back everything a cut-off naming had set, and the run builds a new namer. A limit that never expires changes nothing; a 50-line file gives the same output with `--time-limit 600` as without it. The limit also works for one SMILES (`orthonym "SMILES" --time-limit 30`), with the output of a structure that gets no name and, with `--on-failure raise`, status `1`.

How the limit stops a naming. It is a timer signal (`SIGALRM`) whose handler stops the naming at the next Python statement, and a lost signal is raised again every 0.1 s. A call that is already running inside the Java machine (OPSIN, the stereo labeller) or inside RDKit cannot be stopped from Python and finishes first, so a line can take a little longer than the limit. Measured on 168 structures (150 random drug-like molecules from PubChem, 6 large oligosaccharides, 12 macrocycles and cage ring systems) at limits of 0.05, 0.1, 0.25, 0.5 and 1 s, 268 of 840 runs ran out of time. They stopped 1.1 ms after the limit in the middle, within 10 ms for 99 in 100, and within 57 ms at worst. In 12 of them a call into the Java machine was running when the limit expired, and the extra time was at most 11 ms. The calls into the Java machine in the naming of a 90-second peptide (389 of them) lasted 0.42 s at most (the first call of the process, which starts the Java machine and which the start-up makes before the first limit) and 1.7 s together. A single call that lasts much longer is not covered by these numbers: OPSIN needs about 2 s to read a made-up name of 3,600 characters, and the limit of that line is exceeded by what is left of the call. A call that lasts minutes was met once (a polyphenylene of 37 benzene rings, GitHub issue 121, in the RDKit search that matches a ring fragment onto its canonical re-parse); the naming no longer makes it for rings joined by bonds. For one SMILES (not `--batch`), a call that is still running 5 s after the limit ends the run with the output of any other timeout; a batch has no such end, and the lines after a call that does not return wait for it. Where no timer signal exists (Windows) or `orthonym` is called from a thread other than the main one, the run says so once on the error stream and names every line without a limit.

## Tiers and provenance rows

A batch run has the same default as one SMILES: the `best-effort` tier, that is the verified PIN when the strict path builds one and otherwise a systematic name. The default prints one note on the error stream at the end of the run, once however many lines got a name below `pin_verified`, and none when no line did. `--emit-tier` works as it does for one SMILES. This file has a molecule with no PIN, which `--emit-tier pin` leaves without a name:

```console
$ cat wide.smi
CCO
CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1
O=[U](=O)=O
$ orthonym --batch wide.smi
CCO	ethanol
CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1	2-{4-[4-(1-benzofuran-2-yl)-1,3-thiazol-2-yl]piperidin-1-yl}-N-(3-fluorophenyl)-2-methylpropanamide
O=[U](=O)=O	
$ orthonym --batch wide.smi --emit-tier pin
CCO	ethanol
CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1	
O=[U](=O)=O	
```

A line that gets no name at the tier you chose follows `--on-failure`. The whole file is named with one `Orthonym` instance, as in the Python loop below.

`--provenance` puts the row of [The provenance row](provenance.md) in the second column, on one line, in place of the name. It has the same keys as the row for one SMILES, and `--on-failure` does not change it: a line with no name has `limit_code` and, with `--emit-tier` other than `pin`, `null` as its `name`. The row carries the tier, so the note is not printed.

```console
$ orthonym --batch wide.smi --provenance
CCO	{"name": "ethanol", "tier": "pin_verified", "is_pin": true, ...}
```

`--confidence` works with every tier; the line has the name, the score and the handler (without `--emit-tier` the note is printed when a name is below `pin_verified`). It cannot be combined with `--provenance`, because a row has no coverage score. `orthonym` stops with a usage error and status `2`.

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
