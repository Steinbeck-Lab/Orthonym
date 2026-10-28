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
: `pin` (the default) aims at the Preferred IUPAC Name. `general` allows general IUPAC forms where the recommendations offer one, for example functional class names (`dimethyl sulfoxide`, `methyl isocyanate`), hydrate names (`oxalic acid dihydrate`) and the axial descriptors `Ra` and `Sa`. The PIN-only tier's PIN-only rule applies to `pin` only, so with `general` or `cas` a name that `pin` declines with `NO_VERIFIED_PIN` is returned; the provenance row gives its tier. `cas` is accepted and at present gives the names of `general`, except that hydrate names and axial descriptors keep their `pin` form.

`--enable-triviality-controller`
: Where the IUPAC 2013 recommendations prefer a retained parent name (benzene, phenol, aniline, benzoic acid and others) to the systematic one, use it (P-15.1.8). Every change is checked by an OPSIN round trip. `ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1` turns it on too.

`--trivial`
: When no preferred name can be built, also allow a retained trivial name that is not a preferred name. A preferred name that can be built is never replaced: glycerol stays `propane-1,2,3-triol`. Without this option, two kinds of retained trivial name are still used at the wider tiers, and the PIN-only tier declines them: names from a small last-resort table, and the trivial natural-product names of molecules whose preferred bridged fused name the engine does not build yet (`diamorphine`). The provenance row labels both {tier}`systematic_verified` with source `trivial_retained`. A natural-product name built on a parent, such as `(9R,13S,14S)-3-methoxy-17-methylmorphinan`, is not a trivial name, and `--trivial` does not return it at the PIN-only tier.

## Tiers

`--emit-tier {pin,valid,complete,best-effort,full-coverage}`
: Which names to return. Without it, the `best-effort` tier: the strict PIN path first, whose {tier}`pin_verified` name is returned as it is, and otherwise a systematic name that is not the PIN, labelled with its tier. A name below `pin_verified` earns one note on the error stream per run, and `--provenance` shows the tier. `pin` returns a name only when the pipeline can build the preferred IUPAC name (PIN), that is when the strict PIN path built the name and verified it ({tier}`pin_verified`); it was the default up to release 1.0.x. Its exceptions are names from the natural-product and metal-complex lists, the name formats absent from OPSIN's grammar (inositols, and phanes unless `--opsin main` lets OPSIN itself read them; otherwise a phane name is checked by building its structure), PIN spellings that OPSIN reads back through a twin spelling of the same structure (`ethan(2H)ol`, `diphenyliodanuide`, `methane-SO-thioperoxol`) or through their acids (dianhydrides), PINs whose stereodescriptors OPSIN cannot read, for which the PIN-only tier compares the constitution, and, with `--trivial`, a retained trivial name. Otherwise `pin` declines, with the reason code `NO_VERIFIED_PIN` when it built a name that is not a verified PIN. The rule applies to the default `--style pin`. `valid` adds names from the general engine, `complete` adds general names for aromatic and heterocyclic ring systems, `best-effort` adds the last-resort producers, von Baeyer and spiro names for ring systems of up to 100 skeletal atoms and 15 rings (the other tiers build these names for ring systems of up to 40 skeletal atoms and 8 rings), and adducts with a one-atom ion such as chloride, and, for a monocycle whose two stereocentres each carry one ligand and one hydrogen atom, the `cis` or `trans` name (`trans-cyclohexane-1,4-diol`) in place of a name with lowercase `r` and `s`, which OPSIN cannot read (the conditions are in [Output tiers](../tiers.md)), and, for a molecule that has no other name, the name the PIN-only tier ships although OPSIN cannot read its stereodescriptors, with the label of the PIN-only tier ({tier}`pin_unverified`), and `full-coverage` adds the coordination-name builder for metal tetrapyrrole and corrin complexes. The wider tiers also return the names the PIN-only tier declines, each labelled with its tier. The default is the `pin` tier for `--style general` and `cas`, and when no OPSIN reads the names back (a host without Java). See [Output tiers](../tiers.md).

## Output

`--provenance`
: Print a JSON row instead of the bare name: the name, its tier, and how it was checked. With `--batch`, each line is the SMILES, a tab and the row. Not with `--confidence`. See [The provenance row](provenance.md).

`--confidence`
: Print the name with a coverage score, the tier the name earned (for a tier above `pin`), the part of the engine that built it, and the parts of the score. When no measurement was taken it says so: `Confidence: unverified (no coverage measurement was taken)`. With every tier; not with `--provenance`: a tier row has no coverage score, and the call stops with a usage error.

`--on-failure {empty,label,raise}`
: What to give for a structure that gets no name. `empty` (the default) prints an empty name and, on the error stream, one line `SMILES<tab>CODE: reason` with the limit code and the label. `label` prints the label in place of a name, as earlier versions did. `raise` stops with an error: the same line on the error stream and exit status `1` (with `--batch` the line gets `ERROR:` and the run ends with status `1`). It applies to a name, to `--confidence` and to `--batch` (with `--emit-tier` too); `--provenance` (with `--batch` too), `--dump-tree` and `--engine-only` report the structure as it is: in a `--provenance` row, `limit_code` says why there is no name. The codes are the stable part (see [Declines](../declines.md)); the label texts can change.

`--verbose`, `-v`
: Also print the SMILES and the style. With `--batch`, also say how many lines were read, how many failed and how many got no name (with `--time-limit`, how many of them ran out of time).

`--dump-tree`, `--format {text,json}`
: Print the parts of the name as a tree instead of the name. `--format` sets the layout: `text` (an indented tree; the default) or `json`. One SMILES only: not with `--batch`.

```console
$ orthonym --dump-tree "OC1CCCCC1"
NameTree: cyclohexanol  (class_id=general_acyclic, cite=P-14+P-23+P-44)
+- parent_stem: 'cyclohex'
   |- suffix: 'ol'
```

## Batch

`--batch FILE`, `-b FILE`
: Name every SMILES in the file, one per line. `--emit-tier`, `--provenance`, `--trivial`, `--enable-triviality-controller` and `--binding-proof` work as they do for one SMILES. `--validation-stats`, `--dispatch-stats`, `--engine-only`, `--dump-tree` and `--format` look inside the naming of one structure, so with `--batch` they stop with a usage error. See [Batch files](batch.md).

`--output FILE`, `-o FILE`
: With `--batch`, write the lines to this file instead of the screen.

`--time-limit SECONDS`
: Give each structure at most this many seconds: one SMILES, or each line of `--batch`. A structure that is not named in time has no name, as any structure that gets none (`--on-failure`), with the reason code `TIME_LIMIT`; with `--provenance` its row has the tier `abstain` and `limit_code` `TIME_LIMIT`. A batch goes on with the next line, which is named as in a run without the limit, and `--verbose` counts the lines that ran out of time. Off by default. A call into the JVM that is already running finishes before the limit stops the naming (for one SMILES, a call still running 5 s after the limit ends the run with the same output), and where no alarm can be set (Windows, a thread other than the main one) the run says so once and has no limit: see [A time limit for each structure](batch.md#a-time-limit-for-each-structure).

## Diagnostics

These are for looking inside the engine. None of them changes a name, except `--binding-proof enforce`, which can decline one.

`--fetch-jars`
: Download the OPSIN and centres jars if they are missing, check them, print where they are, and stop. With `--opsin main`, also build that OPSIN. See [Install](../start/install.md).

`--opsin {2.9.0,main,main@COMMIT}`
: Which OPSIN reads every name back: `2.9.0`, the release jar (the default), `main`, the newest commit of OPSIN's main branch, or `main@COMMIT`, one commit. A main-branch OPSIN is built with your JDK on first use and kept in the jar directory. `ORTHONYM_OPSIN` sets the same; a jar given by `ORTHONYM_OPSIN_JAR` wins over both. See [Choosing the OPSIN version](../start/install.md#choosing-the-opsin-version).

`--validation-stats`
: After the name, print on the error stream how often the OPSIN grammar pre-check passed or repaired a candidate name. One SMILES only: not with `--batch`.

`--dispatch-stats`
: After the name, print on the error stream which compound-class routes the engine took. One SMILES only: not with `--batch`.

`--engine-only`
: Skip the strict path and print, as JSON, the general engine's own name, checked for atom coverage but not by OPSIN. Never a preferred name; do not use it as a name. One SMILES only: not with `--batch`.

`--binding-proof {off,audit,enforce}`
: An extra check that every part of a general-engine name still maps onto its atoms in the final name. `audit` records the result and never changes the name; `enforce` also declines when the check fails.

`--version`, `-V`
: Print the installed version, in the form `orthonym X.Y.Z`. When `--opsin` or `ORTHONYM_OPSIN` selects OPSIN's main branch, a second line, `OPSIN main <commit>`, names the build.

## Exit status

`0` when the name was printed, when an empty name or a label was printed for a decline (`--on-failure empty` and `label`), for `--help` and `--version`, and when `--fetch-jars` succeeded. `1` for a SMILES that RDKit cannot read, a batch with at least one such line, a decline with `--on-failure raise`, a call with neither a SMILES nor `--batch` (the help is printed), or a `--fetch-jars` that could not fetch, build or check a jar. `2` for a usage error (an option the parser does not accept, `--confidence` together with `--provenance`, or `--batch` together with `--validation-stats`, `--dispatch-stats`, `--engine-only`, `--dump-tree` or `--format`), and when a jar can be neither found nor downloaded; the jar check is made before the input is read, except with `--fetch-jars`.
