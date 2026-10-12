# Architecture

This page is for people who change the code. [How it works](how-it-works.md) is the short version for users. Here one SMILES is followed through the code, the parts are named, and the names that mislead new readers are explained. Where a rule lives is on the [Rule index](rule-index.md), and how to make a change is in [Contributing](project/contributing.md).

Functions and classes are given by name, not by line number, so that you can find them with `git grep -n "def name_tiered" -- src`. Line numbers change in every release.

## Five facts to start from

1. **There is no single naming path.** Several families of code can build a name for one molecule. They are tried in a fixed order, and the first name that passes the checks is the one that is returned. Two families can both be right and spell the name differently.
2. **The tiers are not four code paths.** `pin`, `valid`, `complete` and `best-effort` run the same code with different flags. The flags are read in many places, so the same function can write a different name at a different tier.
3. **A producer can be wrong, and the checks are what makes the engine safe.** A name is returned only after OPSIN has read it back and the InChIKeys agree, with a few named exceptions. When a check rejects a name, the class that produced it is excluded and the next one is tried.
4. **The same pipeline runs again on parts of the molecule.** A substituent or a fragment is named by calling the naming code recursively, so the function you are reading may be working on a piece of your molecule, not on the whole of it.
5. **A comment that says "the one place where X is decided" is a claim to test.** A function can be well documented and still never run for the molecules you care about. [Before you edit](#before-you-edit-a-function) shows how to count calls.

## One call, step by step

```text
name_compound(smiles, ...)           one Orthonym per call
  Orthonym.name()                    three wrappers, then the body
    _name_with_pin_promotion         PIN tier: a second run if no name
    isotope decorator, wildcard check
    _name_impl()
      ClassFirstRouter.dispatch()    first class that fits (outer table)
        class handler -> rules/...   a name, or None: the next class
        class "general"              None on purpose: general pipeline
      _perceive(), _classify()       features, principal group, parent
      composer.assemble_name()
        dispatch_inner()             first handler that succeeds
          handler -> assembly/...    calls rules/... to build the name
    _final_stereo_check, _final_grammar_check
    _final_opsin_validity_gate       a rejection excludes the class
    rescues                          only if a failure label would ship
    _finish()                        offers, round-trip selection, exit
  Orthonym.name_tiered()             adds tier, source and gate fields
```

### 1. The entry: `name_compound` and `Orthonym`

`name_compound` builds one `Orthonym` and calls its `name` method, or `name_with_confidence` when you ask for the confidence. `Orthonym.name_tiered` calls `name` inside and then derives the provenance row (`_tier_row_and_pin_form`). A `Chem.Mol` is turned into a SMILES string first (`orthonym.input_mol`), and from then on the engine works with the string.

When `name_compound` is called while another naming is already running, which is how recursion happens, it builds a fresh `Orthonym` and looks the call up in a memo (`cached_nested_call`). A fresh instance does not have the flags of the outer call, so the flags travel in context variables (see [Tiers are flags](#tiers-are-flags)).

### 2. `Orthonym.name`: three wrappers and a body

`name` carries three decorators, outermost first.

| Wrapper | What it does |
|:--|:--|
| `_form_choice_scope` | Names the charge forms of your input (`_charge_forms`) and keeps the best one. |
| `_book_retry_scope` | When a name written in the book's spelling leaves the molecule unnamed at a wider tier, names it again with the older spelling (`mechanical_forms`). |
| `_budget_scope` | Counts how deep the naming is (`enter_name_scope`), arms the work budgets that stop a runaway molecule, and applies the default tier's emission rule (`_default_tier_decision`) to the caller's own call. |

The body then runs, in this order: the isotope decorator (`decorate_isotopic_name` in `rules/isotopes.py`, for an input with isotopes), a check for a wildcard atom, `_name_impl`, the checks, the rescues and `_finish`.

At the default tier `name` has one more step in front. `_name_with_pin_promotion` runs `name` once; when that returns no name, and only then, it runs it a second time with the best-effort ring-substituent producers allowed, and keeps the second name only if it passes the same gates. A name from the second run is labelled `pin_unverified`, because a breadth producer built it.

### 3. Class routing: the outer table

`_name_impl` parses the SMILES, removes the charges of a large zwitterion that counts as neutral, and asks `ClassFirstRouter.dispatch` (`routing/dispatcher.py`) for a class. The router walks `DISPATCH_TABLE` (`routing/dispatch_table.py`) in priority order and returns the first class whose predicate says yes. The class handler then runs. When it returns `None`, `_name_impl` goes on with the next higher priority that matches. The table is listed on the [Rule index](rule-index.md#from-a-handler-id-to-its-code).

Four things in this table surprise readers.

- **The `tier` number in an entry is a label.** The docstring of the router speaks of two-tier evaluation, but the code is one loop over all entries sorted by priority. `tier` marks whether a predicate needs only the molecule (1) or perceived features (2), and is not used to decide anything.
- **Most handlers are shims.** The function in `dispatch_table.py` imports the real code from `rules/` inside its body and calls it. Search `rules/` for the code, not the table.
- **A predicate often runs the naming function.** For most classes the predicate calls the same function as the handler and asks whether it returns a name (for example `name_mononuclear_hydride`). The class "can name this" exactly when its namer builds a name.
- **The last class, `general`, returns `None` on purpose.** It means "no special class fits": `_name_impl` then runs the general pipeline inline, because `routing/` must not import the composer.

When a gate rejects the name of a class later on, that class is added to `_excluded_dispatch_classes` and the routing is run again without it. This is how a molecule ends up named by the second-best class.

### 4. The general pipeline

For a molecule no special class took, `_name_impl` runs three steps.

1. `_perceive` finds the functional groups (SMARTS patterns in `perception/functional_groups.py`), the ring systems (`perception/rings.py`), the multiple bonds and the stereocentres, and fills a `MolecularFeatures` object. Stereo goes through `assign_stereochemistry` in `perception/stereo.py`. Do not call RDKit's CIP labeller directly: `assign_stereochemistry` chooses the most accurate labeller that is available.
2. `_classify` picks the principal group (`get_principal_group` in `rules/seniority.py`), finds out whether the molecule is polyfunctional, and chooses ring or chain as the parent (`select_parent` in `rules/parent_selection.py`, the P-44 rule).
3. `composer.assemble_name` builds the name. A few inline branches in `_assemble_name_impl` (amidines, hydrazides) go first. Then `dispatch_inner` (`assembly/inner_dispatch.py`) walks `INNER_DISPATCH_TABLE`. Unlike the outer table, this loop stops at the first handler that applies **and succeeds**: a handler that returns `None` hands the molecule to the next one. The last entry, `general_acyclic`, is a catch-all and handles what is left.

The inner handlers live in `assembly/handlers/` and call the nomenclature rules in `rules/`, the helpers of `assembly/naming_utils.py` (locants, enclosing marks, alphanumerical order) and the substituent namers. `--dump-tree` prints the handler that built a name. The handlers are listed with their files on the [Rule index](rule-index.md#from-a-handler-id-to-its-code).

### 5. Checks after a producer has returned

Still inside `name`, and only at the top level of the call, the name goes through three checks.

- `_final_stereo_check` adds stereodescriptors that a producer left out, for the few producers whose numbering it can trust. For the others it only logs: a missing descriptor is better than one at a guessed locant.
- `_final_grammar_check` checks the name against a model of OPSIN's grammar and repairs it only when the repaired name reads back.
- `_final_opsin_validity_gate` has OPSIN read the name back and compares the structure with the input (called SELF-01 in comments). Some name classes that OPSIN cannot read are let through by named carve-outs, and the provenance row says so (`gate_outcome`, `carveout:<class>`).

When the gate rejects a name, `_retry_cascade_on_gate_rejection` excludes the class that produced it and dispatches again. The atom-coverage check of the [checking page](checking.md) is a separate certificate (E1, `validation/e1_certificate.py`). It decides for names of the general engine.

### 6. Rescues

If the result is now a failure label, up to seven rescues get a turn. They are optional: each one either returns a name that passes the gate or declines.

| Rescue | Runs when |
|:--|:--|
| `_try_general_engine_recovery` | The general engine is on (`valid` and wider). It also runs when a handler raised `OrthonymLimitError` or any other exception. |
| `_try_demote_senior_group_rescue` | `valid` and wider: names the ring parent when a senior group on a substituent had taken over. |
| a second call of `try_decompose` | Any tier, with the gate on: the decomposition engine as a last resort. |
| `_try_alternate_parent_rescue` | `valid` and wider: tries the next parent of the P-44 ranking. |
| `_try_np_systematic_downgrade` | `best-effort`, even when a name exists: replaces a retained natural-product parent name by a systematic name if that reads back. |
| `_try_besteffort_clean_general_fallthrough` | `best-effort`: runs the general engine again in a clean context. |
| `_try_perf_budget_t4_rescue` | `best-effort`: when a work budget ran out. |

`_apply_trivial_fallback` runs at every tier. What it does depends on `--trivial` and on the last-resort table of retained names.

A budget that runs out raises `PerfBudgetExceeded`, which derives from `BaseException`, so a broad `except Exception` does not catch it. It unwinds to `_budget_scope`, which declines cleanly (at `best-effort` after one last rescue).

### 7. The single exit: `_finish`

Every way out of `name` goes through `_finish`; a test parses the code to keep it that way. In `_finish`:

- the exact-match list for metal tetrapyrrole complexes can replace the name (`_coordination_retained_name`);
- a coverage audit records a verdict, and in its `veto` mode declines an incomplete name (the default is `shadow`, which only records);
- the winner becomes one `Offer` (`assembly/offer_pool.py`); at `best-effort`, if the winner does not already read back exactly, `_maybe_append_t4_floor_offer` adds the output of the last-resort namer as a second offer; `_select_rt_passing_offer_name` returns the first offer that is complete and reads back;
- `_exit_round_trip_check` checks the name once more, and `_maybe_prefix_order_fallback` is the last resort of the best-effort tier;
- with `binding_proof` set to `audit` or `enforce`, a proof that every part of the name maps onto atoms is checked.

The pool of offers holds the winner and at most one more name, the floor. `assembly/candidate_pool.py` is a different thing; see the [glossary](#names-that-mislead).

### 8. The row: `name_tiered`

`name_tiered` calls `name`, then `_tier_row_and_pin_form` combines what the producers and gates recorded into the row of the [provenance page](use/provenance.md). The recording is done with context variables in `metrics/provenance.py`; `record_source` sets the `source` value. At the default tier the emission rule (`_default_tier_emits`) is applied inside `name` itself, so that `name_compound` and `name_tiered` agree: the row derived for that decision is the row `name_tiered` returns.

## Tiers are flags

`--emit-tier` becomes constructor arguments of `Orthonym` (`_emit_tier_flags` in `cli.py` is the one table that maps one to the other):

| Tier | `general_fallback` | `allow_aromatic_general` | `general_fallback_unverified` | `full_coverage` |
|:--|:-:|:-:|:-:|:-:|
| `pin` | no | no | no | no |
| `valid` | yes | no | no | no |
| `complete` | yes | yes | no | no |
| `best-effort` | yes | yes | yes | no |
| `full-coverage` | yes | yes | yes | yes |

The top-level `name` publishes each flag in a context variable (`general_fallback_ctx`, `allow_aromatic_general_ctx`, `best_effort_ctx`, `full_coverage_ctx`, all in `metrics/provenance.py`), because recursion builds a fresh `Orthonym` and a fresh instance would lose the flags. Code far from `namer.py` reads the variable it needs. `best_effort_ctx` alone is read in more than twenty files. `pin_promotion_var` (`assembly/memo.py`) marks the second run of the promotion step and keeps its memo entries apart from the first run's.

What this means for a change:

- A producer that is used by every tier can write a different name at each. The `source` value `pin_path` names a code path, not a result: at the default tier cis-decalin gets `(4as,8as)-decahydronaphthalene` from the `pin_path` with tier `pin_unverified`.
- Test a change at the tiers it can reach. `pin` is the byte-identical default, and a change that is meant for the wider tiers must leave it as it was.
- There is no `if tier == ...` to search for. Search for the context variables and for the instance attributes `_general_fallback`, `_general_fallback_unverified`, `_allow_aromatic_general` and `_full_coverage`.

## The producer families

| Family | Where | Runs |
|:--|:--|:--|
| Class handlers | `routing/dispatch_table.py` shims, code in `rules/` | Always, first |
| General pipeline | `_perceive`, `_classify`, `assembly/composer.py`, handlers in `assembly/handlers/`, code in `rules/` | Always, when no class took the molecule |
| Decomposition | `decomposition/engine.py`, `try_decompose` | As the class `decomposition_pre_general` before the general pipeline, and as a last resort |
| General engine | `assembly/general_engine.py`, `name_general` | For the whole molecule at `valid` and wider, when the strict path has no name. Every part of its name is bound to atoms, and the E1 certificate checks that no atom is lost. `tier_a_ring` also calls it at every tier, to prefer the suffix form for a spiro parent |
| Universal namer ("T4") | `assembly/t4_coverage.py`, `name_t4_complete`, with `assembly/universal_substituent.py` | For the whole molecule at `best-effort` only: as a rescue, and as the "floor" offer in `_finish`. For a part of a molecule it can run at any tier, because the fragment rescue builds a fresh best-effort engine (`best_effort_request_ctx` records the tier of the outer request) |
| Promotion re-run | `_name_with_pin_promotion` | Default tier, when the first run has no name |
| Substituent namers | `assembly/substituent_enumerator.py`, `assembly/substituent_naming.py`, `assembly/universal_substituent.py`, `rules/ring_substituents.py`, `rules/terminal_fragment.py`, and prefix writers inside `composer.py` | In every family, for every part of a molecule that becomes a prefix |

Because the families do not partition the molecules, a molecule can be named by two of them, and the spelling can differ. Two examples, each from a direct call of the producers:

| Molecule | Strict path (the name the user gets) | Universal namer |
|:--|:--|:--|
| ethyl lactate, `CCOC(=O)C(C)O` | `ethyl 2-hydroxypropanoate` | `1-ethoxy-2-hydroxy-1-oxopropane` |
| paracetamol, `CC(=O)Nc1ccc(O)cc1` | `N-(4-hydroxyphenyl)acetamide` | `N-(4-hydroxyphenyl)ethanamide` |

Which spelling is returned is decided by the order of the families and by which name survives the checks, not by a rule that compares them. When you fix a rule in one family, find out whether another family writes the same part of the name, and fix it there too or write down that you did not. The [Rule index](rule-index.md) lists the rules that are written in more than one place.

Ring systems show the overlap most: a bridged, spiro or large polycyclic system can be named by the ring modules of `rules/`, by the general engine and by the universal namer.

## Recursion and budgets

Naming a molecule names its parts, and naming a part calls the same entry points. Four mechanisms keep this finite and let the code know where it is:

- `name_scope_depth()` and `is_top_level_naming()` (`assembly/fragment_naming.py`) say where the code is. `name_scope_depth()` counts nested `name()` calls, and 1 is the caller's own call. `is_top_level_naming()` says whether the code is naming a whole molecule or a piece of one; the isolated sessions that some rescues open reset it, so the two are not the same test. The checks of section 5 and most rescues run only at the top level. A function that behaves differently for a piece of a molecule usually asks one of these.
- A set of SMILES currently being named (`name_fragment_recursively`) breaks cycles.
- Per-molecule work budgets (`spend_perf_work`, `spend_analysis_call`) stop a molecule whose analysis explodes, as in `PerfBudgetExceeded` above.
- Scoped memos (`assembly/memo.py`) keep the names of parts for the length of one call. Where the name of a part depends on the tier, the key carries the tier flags.

## Before you edit a function

Find out which code named your molecule, and whether the code you are about to change runs at all. Four ways, from cheap to precise.

1. **The row.** `orthonym "SMILES" --provenance` shows `tier`, `source` and `gate_outcome`. `source` is coarse: `pin_path`, `general_engine`, `trivial_retained`, `t4_floor` or `abstain`. On a decline it can still name the last producer that ran.
2. **The classes.** `orthonym "SMILES" --dispatch-stats` prints, on the error stream, how often each outer class and each inner handler matched. The counts include the parts of the molecule. `Orthonym.get_dispatch_stats` and `get_inner_dispatch_stats` give the same in Python.
3. **The tree.** `orthonym "SMILES" --dump-tree` prints the parts of the name. Its `class_id` is the inner handler that built the name (`ester_family`, `general_acyclic`), or `coarse_fallback` when a class of the outer table did.
4. **A call count.** Profile a few molecules of your class and look for the function:

```python
import cProfile, pstats
from orthonym import name_compound

smiles = ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "CCOC(=O)C(C)O", "CC(C)(C)CC(=O)O"]
pr = cProfile.Profile()
pr.enable()
for s in smiles:
    name_compound(s)
pr.disable()
for (path, line, func), row in pstats.Stats(pr).stats.items():
    if func in {"alpha_sort_key", "compare_locant_sets", "name_ketene"}:
        print(row[1], path.split("orthonym/")[-1], func)
```

Check the recipe against functions you know run before you trust a zero. A function that is imported inside another function (the shims do this) is still counted, because the profiler counts calls and not imports. To spy by patching a module attribute you must patch it in the module that calls it, since a function imported with `from x import f` keeps the old reference.

The three vocabularies below are easy to confuse.

| Where it shows | What the ids are |
|:--|:--|
| `--dispatch-stats`, first block | Outer classes (`StoutClass`): `GENERAL`, `RETAINED_NAME` |
| `--dispatch-stats`, second block, and `--dump-tree` | Inner handlers: `ester_family`, `tier_a_ring` |
| `--confidence`, line "Handler" | Labels of the coverage score: `chain`, `complex_ring`, `unmeasured` |

## Names that mislead

| Name | What it is |
|:--|:--|
| `ester_family` | The inner handler for polyfunctional molecules, multiple esters and esters, tried in that order. It names plain acids, alcohols and thiols as well (`3-sulfanylhexan-1-ol`, `6-hydroxy-2-oxohexanoic acid` come from it). |
| `tier_a_ring` | The ring cascade, a long handler. It tries complex rings, polycycles, partly saturated rings, heterocycles and benzene, picks among the candidates (`select_best_candidate`), and calls the general engine itself for spiro parents. |
| `general`, `GENERAL` | In the outer table: a pass-through that returns `None` on purpose. `general_acyclic` in the inner table is the real catch-all. |
| `composer`, `assemble_name` | Assembles a name from parts. The inner dispatch runs inside it. |
| `polycyclic` and `polycyclics` | `rules/polycyclic.py` generates von Baeyer descriptors (P-23). `rules/polycyclics.py` names polycyclic aromatic hydrocarbons. |
| `bicyclo`, `tricyclo`, `vonbaeyer_universal` | Von Baeyer namers for two rings, for three rings, and the general analysis of a cage. Several modules use each of them, `polycyclic.py` among them. |
| `T4`, `t4_floor`, `t4_coverage` | The universal namer of the `best-effort` tier. "T1" to "T5" in older comments are the tier ids: T1 `pin_verified`, T2 `pin_unverified`, T3 `systematic_verified`, T4 `best_effort`, T5 `abstain`. |
| `pin_path` | A `source` value for the strict path: class handlers, composer and inner handlers. It is not a module boundary, and not a promise that the name is a PIN. |
| `source` | Coarse. It records the last producer that wrote a value, and on a decline it can name a producer that ran and failed. Do not use it to count producers. |
| `candidate_pool`, `CandidatePool`, `HANDLER_POLICIES` | A selector that ranks parent candidates by the P-44 criteria. In the default configuration its pool almost always holds a single candidate, because handlers return their names instead of adding candidates, so the ranking has next to nothing to rank. The parent is chosen in `_classify`, by `select_parent`, and the real source of alternatives is the routing: a rejected class is excluded and the next one is tried. The module is kept because its unit tests call it. |
| `Offer`, `offer_pool` | Whole finished names that compete in `_finish`. Not the same as `candidate_pool`. |
| `SELF-01`, `E1` | In comments: SELF-01 is the OPSIN read-back of `_final_opsin_validity_gate`; E1 is the atom-coverage certificate. |
| `StoutClass` | The enum of the outer classes. The prefix is historical. |
| "Phase 158", "v25 G1", "WSB-03" | Tags in comments that name the internal work item that added the code. They are not rules, and the notes they refer to are not in the repository. The [internals pages](reference/internals/index.md) strip the leading tag of a docstring. |
| `BlueBookV2.md:4667` | A line number in a text conversion of the Blue Book that is not in the repository. Find the rule by its P-number in the Blue Book instead. |
| `P-31.1.4.2.4`, `P-31.1.4.3.4` | Not Blue Book ids. P-31.1.4 has sub-rules `.1` to `.4` only. The numbering criteria are in P-14.4. Check any id that you copy from a comment. |

## Citing the Blue Book

The authority is the [IUPAC 2013 Recommendations](https://iupac.qmul.ac.uk/BlueBook/) ("the Blue Book"). When you cite a rule in code or in a test, give the section that the book defines and the criterion inside it, and read the whole paragraph: the sentence that decides is often the last one. A rule number copied from a comment may be wrong; look it up.

If you search a text or OCR conversion of the book, expect a few traps. Headings can carry markup (`### **P-14.3.4.2**`), so a pattern anchored on `P-14.3.4.` misses them. In some conversions the hyphen of `P-16` is a double quote and spaces are exclamation marks, so a search for `P-16.5` finds nothing. Whole chapter headings can be missing (P-2, P-4 and P-7). Figures and ring-numbering diagrams are the least reliable part; check them against the web edition.

## Where the data lives

The naming tables are Python modules in `src/orthonym/data/`, not JSON files. The list, with what each holds, is on the [Rule index](rule-index.md#naming-tables). Retained names that are written by hand take precedence over the tables imported from OPSIN's resource files (`data/opsin_imports/`); `data/__init__.py` merges them.

## Where the tests live

`tests/unit/` mirrors the packages (`rules/`, `assembly/`, `perception/`, `routing/`, `namer/`), with many older tests directly in `tests/unit/`. Two things to know before you write one:

- The suite turns the OPSIN validity gate **off** by default. A plain `name_compound` call in a test shows what the strict path builds, which is not always what a user gets. A test that depends on the gate carries `@pytest.mark.opsin_gate`.
- `tests/support/` holds helpers that check the round trip independently of the engine (`rt_assert.py`), and helpers for the default tier and the best-effort tier (`default_tier.py`, `pin_tiers.py`).

Some tests read the text of source files, for example to check that a function is not called from a module. If you rename or move a function, find them with `git grep -l -E "getsource|read_text" -- tests` and run them.
