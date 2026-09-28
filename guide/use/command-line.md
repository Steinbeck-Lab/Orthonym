# Command line

`orthonym` names one SMILES, or a file of them. `python -m orthonym` is the same program. Every option is listed below in plain words, grouped by what it is for; the [option reference](../reference/command-line.md) prints the program's own help text.

```console
$ orthonym "c1ccccc1"
benzene
```

## Naming

`orthonym "SMILES"`
: Print the name of one structure.

`--style {pin,general,cas}`
: `pin` (the default) aims at the Preferred IUPAC Name. `general` allows a few general IUPAC forms where the recommendations offer one. `cas` is accepted and at present gives the same names as `pin`.

`--enable-triviality-controller`
: Where the IUPAC 2013 recommendations prefer a retained parent name (benzene, phenol, aniline, benzoic acid and others) to the systematic one, use it (P-15.1.8). Every change is checked by an OPSIN round trip. `ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1` turns it on too.

`--trivial`
: When no preferred name can be built, also allow a retained trivial name that is not a preferred name. A preferred name that can be built is never replaced: glycerol stays `propane-1,2,3-triol`. Without this option a small table of retained trivial names is still used as a last resort; the provenance row labels those names {tier}`systematic_verified` with source `trivial_retained`.

## Tiers

`--emit-tier {pin,valid,complete,best-effort,full-coverage}`
: Which names to return. The default, `pin`, returns the name from the strict path for the Preferred IUPAC Name; where the engine cannot certify the preferred name it can return another name, labelled with its tier. `valid` adds names from the general engine, `complete` adds general names for aromatic and heterocyclic ring systems, `best-effort` adds the last-resort producers, and `full-coverage` adds the coordination-name builder for metal tetrapyrrole and corrin complexes. See [Output tiers](../tiers.md).

## Output

`--provenance`
: Print a JSON row instead of the bare name: the name, its tier, and how it was checked. One SMILES at a time. See [The provenance row](provenance.md).

`--confidence`
: Print the name with a coverage score, the part of the engine that built it, and the parts of the score. When no measurement was taken it says so: `Confidence: unverified (no coverage measurement was taken)`.

`--verbose`, `-v`
: Also print the SMILES and the style. With `--batch` and `--output`, also say how many lines were named and how many failed.

`--dump-tree`, `--format {text,json}`
: Print the parts of the name as a tree instead of the name.

```console
$ orthonym --dump-tree "OC1CCCCC1"
NameTree: cyclohexanol  (class_id=general_acyclic, cite=P-14+P-23+P-44)
+- parent_stem: 'cyclohex'
   |- suffix: 'ol'
```

## Batch

`--batch FILE`, `-b FILE`
: Name every SMILES in the file, one per line. See [Batch files](batch.md).

`--output FILE`, `-o FILE`
: With `--batch`, write the lines to this file instead of the screen.

## Diagnostics

These are for looking inside the engine. None of them changes a name.

`--fetch-jars`
: Download the OPSIN and centres jars if they are missing, check them, print where they are, and stop. See [Install](../start/install.md).

`--validation-stats`
: After the name, print on the error stream how often the OPSIN grammar pre-check passed or repaired a candidate name.

`--dispatch-stats`
: After the name, print on the error stream which compound-class routes the engine took.

`--engine-only`
: Skip the strict path and print, as JSON, the general engine's own name, checked for atom coverage but not by OPSIN. Never a preferred name; do not use it as a name.

`--binding-proof {off,audit,enforce}`
: An extra check that every part of a general-engine name still maps onto its atoms in the final name. `audit` records the result and never changes the name; `enforce` also declines when the check fails.

`--version`, `-V`
: Print the version (`orthonym 1.0.0`).

## Exit status

`0` when the name (or the label for a decline) was printed; `1` for a SMILES that RDKit cannot read, or a batch with at least one such line; `2` when a jar is missing.
