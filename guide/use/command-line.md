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
: `pin` (the default) aims at the Preferred IUPAC Name. `general` allows general IUPAC forms where the recommendations offer one, for example functional class names (`dimethyl sulfoxide`, `methyl isocyanate`), hydrate names (`oxalic acid dihydrate`) and the axial descriptors `Ra` and `Sa`. The default tier's PIN-only rule applies to `pin` only, so with `general` or `cas` a name that `pin` declines with `NO_VERIFIED_PIN` is returned; the provenance row gives its tier. `cas` is accepted and at present gives the names of `general`, except that hydrate names and axial descriptors keep their `pin` form.

`--enable-triviality-controller`
: Where the IUPAC 2013 recommendations prefer a retained parent name (benzene, phenol, aniline, benzoic acid and others) to the systematic one, use it (P-15.1.8). Every change is checked by an OPSIN round trip. `ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1` turns it on too.

`--trivial`
: When no preferred name can be built, also allow a retained trivial name that is not a preferred name. A preferred name that can be built is never replaced: glycerol stays `propane-1,2,3-triol`. Without this option, two kinds of retained trivial name are still used at the wider tiers, and the default tier declines them: names from a small last-resort table, and the trivial natural-product names of molecules whose preferred bridged fused name the engine does not build yet (`diamorphine`). The provenance row labels both {tier}`systematic_verified` with source `trivial_retained`. A natural-product name built on a parent, such as `(9R,13S,14S)-3-methoxy-17-methylmorphinan`, is not a trivial name, and `--trivial` does not return it at the default tier.

## Tiers

`--emit-tier {pin,valid,complete,best-effort,full-coverage}`
: Which names to return. The default, `pin`, returns a name only when the pipeline can build the preferred IUPAC name (PIN), that is when the strict PIN path built the name and verified it ({tier}`pin_verified`). The exceptions are names from the natural-product and metal-complex lists, the name formats absent from OPSIN's grammar (for example inositols, phanes and thioperoxols), PINs whose stereodescriptors OPSIN cannot read, for which the default tier compares the constitution, and, with `--trivial`, a retained trivial name. Otherwise it declines, with the reason code `NO_VERIFIED_PIN` when it built a name that is not a verified PIN. The rule applies to the default `--style pin`. `valid` adds names from the general engine, `complete` adds general names for aromatic and heterocyclic ring systems, `best-effort` adds the last-resort producers, von Baeyer and spiro names for ring systems of up to 100 skeletal atoms and 11 rings (the other tiers build these names for ring systems of up to 40 skeletal atoms and 8 rings), and adducts with a one-atom ion such as chloride, and `full-coverage` adds the coordination-name builder for metal tetrapyrrole and corrin complexes. The wider tiers also return the names the default tier declines, each labelled with its tier. See [Output tiers](../tiers.md).

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

These are for looking inside the engine. None of them changes a name, except `--binding-proof enforce`, which can decline one.

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
: Print the installed version, in the form `orthonym X.Y.Z`.

## Exit status

`0` when the name (or the label for a decline) was printed, for `--help` and `--version`, and when `--fetch-jars` succeeded. `1` for a SMILES that RDKit cannot read, a batch with at least one such line, a call with neither a SMILES nor `--batch` (the help is printed), or a `--fetch-jars` that could not fetch or check a jar. `2` when a jar can be neither found nor downloaded; this is checked before the input is read, except with `--fetch-jars`.
