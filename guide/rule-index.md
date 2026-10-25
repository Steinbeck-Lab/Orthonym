# Rule index

Where each naming rule lives in the code, which rules are written in more than one place, and what else to look at when you change one. [Architecture](architecture.md) says how a name is built; this page says where the rules are.

## How to read this page

- **The Blue Book** is the IUPAC 2013 Recommendations ([web edition](https://iupac.qmul.ac.uk/BlueBook/)). Section numbers and titles below are those of its table of contents. When you cite a rule, cite the level the book defines, and the criterion inside it.
- A **canonical helper** is the function the code treats as the home of a rule. That a helper exists does not mean that every place that needs the rule calls it. For eight rules, part 1 lists the helpers, how widely they are used, and the places that decide the same question beside them.
- **The counts are pattern counts.** They come from a scan of `src/orthonym` that looks for code of a given shape; the shapes are listed at the end of the page. A match is a lead, not a proof of a copy: some matches are right for a reason of their own, and a copy that hides under other names is not seen. Read the site before you act on a count.
- **A rule can live in a place that never runs for your molecule.** Before you edit a helper, count its calls on a few molecules of your class ([Architecture](architecture.md#before-you-edit-a-function)).
- Names are given as the function name and its module, not as a line number.

## 1. Rules that are written in many places

### Writing locants into a name

Blue Book: <!-- rule-index:begin bb:P-14.3,P-16.2 -->P-14.3 Locants; P-16.2 Punctuation.<!-- rule-index:end bb:P-14.3,P-16.2 -->

The helpers join a list of locants and attach it to a prefix or a suffix, with the multiplier and the enclosing marks. Most code that needs this does not use them.

<!-- rule-index:begin locant-render -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `format_substituent_prefix` | `assembly/naming_utils.py` | 33 calls in 13 other files |
| `format_suffix_with_locants` | `assembly/naming_utils.py` | 6 calls in 4 other files |
| `multiplied_component` | `assembly/naming_utils.py` | 467 calls in 44 other files |

Code that writes a locant into a name by hand: **560 places** in **243 functions** of **70 files** (a locant joined with commas or hyphens: 253; an f-string `{locant}-...`: 365; a locant plus `"-"`: 31; a line can match more than one pattern). 209 of those functions call none of the helpers above. The files with most places: `rules/ions.py` (63), `assembly/composer.py` (54), `rules/natural_products.py` (52), `rules/ring_substituents.py` (30), `rules/heterocycles.py` (28). 20 of the comma joins sort the locants inside the join; the others write them in the order they are given.

<!-- rule-index:end locant-render -->

When you change it:

- Look at each place that writes a locant by hand and decide whether your change should reach it. Do not assume that fixing the helper fixes the name.
- Most comma joins write the locants in the order the caller gives. A helper that sorts would turn `2,1` into `1,2` for a caller that cites locants in citation order, as some multiplicative names do. Do not add a sort without reading the callers.
- Not every locant is an integer. `4a`, primed locants, Greek letters, `N1`, `1H` and the lambda tokens sort and compare differently. They have renderers of their own (`format_lambda_token` in `rules/lambda_convention.py`, `_render_primed_locant` in `assembly/composer.py`).

### Citing and omitting locants

Blue Book: <!-- rule-index:begin bb:P-14.3.3,P-14.3.4 -->P-14.3.3 Citation of locants; P-14.3.4 Omission of locants.<!-- rule-index:end bb:P-14.3.3,P-14.3.4 -->

The two sections must be read together. P-14.3.3 says that if any locant is essential for the parent structure or for a unit of structure as its enclosing marks define it, all are cited for that unit. P-14.3.4 then lists the cases in which locants are omitted, in six numbered licences (P-14.3.4.1 to P-14.3.4.6). Cite the licence, not the section. The list of exceptions is open: the Blue Book prints `prop-2-enamide` and the prefix `prop-2-enoyl` as preferred, and they keep a locant that they could do without.

The code has two homes for this decision, and they overlap: `assembly/locant_omission.py` (its docstring calls it the one place where the licences are decided) and `should_omit_locant_one` in `assembly/naming_utils.py`.

<!-- rule-index:begin locant-omit -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `l3_locant_omitted_for_parent_atoms` | `assembly/locant_omission.py` | 5 calls in 4 other files |
| `l4_no_isomer_by_relocation` | `assembly/locant_omission.py` | 2 calls in 2 other files |
| `l5_uniform_complete` | `assembly/locant_omission.py` | 5 calls in 4 other files |
| `scope_forces_locants` | `assembly/locant_omission.py` | 4 calls in 3 other files |
| `should_omit_locant_one` | `assembly/naming_utils.py` | 14 calls in 9 other files |

Sites that decide for themselves whether a locant may go:

- `rules/added_carbon_parent.py`: omits locants for a one-atom parent, with no licence list.
- `rules/multiplicative.py`: decides `omit_locant_one` locally and calls `l5_uniform_complete` for `omit_all_locants`.
- `rules/skeletal_replacement.py`: decides `omit_all_locants` and `omit_locants` locally.
- `rules/polyazane.py`, `_ene_locant_is_unambiguous`: its own test for a double bond that needs no locant.
- `assembly/handlers/_handler_shared.py`, `_l5_prefix_locants_omitted`: one of four adapters (`_l3_...`, `_l4_...`, `_l5_...`, `_l6_...`) that call the licence checkers.

<!-- rule-index:end locant-omit -->

When you change it: change the licence checker, not the name. A rule that removes a locant from one class can remove it from another that needs it. The spelling checks in `validation/spelling/` (`locant_omission_check` in `checks_locants.py` among them) re-encode the licences and read the finished name without any naming code, so a change to `locant_omission.py` does not change them. They are a second, independent reading of the rule; keep them that way.

### Enclosing marks

Blue Book: <!-- rule-index:begin bb:P-16.5 -->P-16.5 Enclosing Marks.<!-- rule-index:end bb:P-16.5 -->

The helpers decide whether a prefix needs brackets, which bracket, and how marks nest ( ) then [ ] then { }.

<!-- rule-index:begin enclose -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `apply_enclosing_marks` | `assembly/naming_utils.py` | 132 calls in 30 other files |
| `enclose_if_compound` | `assembly/naming_utils.py` | 164 calls in 44 other files |
| `needs_brackets` | `assembly/naming_utils.py` | 28 calls in 8 other files |
| `is_complex_substituent` | `assembly/naming_utils.py` | 43 calls in 20 other files |
| `renest_group_and_ancestors` | `assembly/naming_utils.py` | 2 calls in 2 other files |
| `top_level_tokens` | `assembly/naming_utils.py` | 2 calls in 2 other files |
| `matching_close` | `assembly/naming_utils.py` | 3 calls in 3 other files |

Beside them: **82** f-strings of the form `f"({x})"` (or with `[ ]`, `{ }`) and **7** concatenations `"(" + x + ")"` in **37 files**, and **10** loops that count brackets in a finished name, in **8 files**. A few of the wraps are fixed syntax (an oxidation number, a fusion descriptor, an added-hydrogen bracket) and are correct as they are. Local helpers that make the same decision:

- `assembly/composer.py`, `_wrap`: wraps in round brackets only; never moves to square brackets.
- `assembly/composer.py`, `_is_compound_prefix`: its own character class, and a carve-out for italic prefixes.
- `assembly/general_engine.py`, `_is_complex_prefix`: needs_brackets or is_complex_substituent, plus a carve-out.
- `assembly/handlers/azine.py`, `_enclose`: round brackets when is_complex_substituent says so.
- `rules/ring_substituents.py`, `_is_complex_prefix`: its own list of endings.
- `rules/polyfunctional.py`, `_enclose`: adds the outermost mark only when one is missing.
- `rules/oxoacid_group_prefix.py`, `_enclose`: thin wrapper over apply_enclosing_marks.
- `rules/ring_assemblies.py`, `_enclose_component`: a different rule that shares the glyph: von Baeyer brackets after 'bi'.
- `rules/heterocycles.py`, `_fully_enclosed`: reads round brackets only.
- `rules/oxoacid_group_prefix.py`, `_fully_enclosed`: delegates to the shared scanner.

<!-- rule-index:end enclose -->

When you change it:

- Some local helpers apply a different rule on purpose. `_enclose_component` in `rules/ring_assemblies.py` writes the brackets of a von Baeyer name, and fixed-syntax brackets (oxidation numbers, fusion descriptors, added hydrogen) must never go through the marks helper.
- A producer may hand over a prefix that is already enclosed (the docstring of `_enclose` in `rules/polyfunctional.py` says so). Enclosing it again gives `((x)y)`.
- Replacing a round-bracket-only wrapper by the helper can change a nested name from `((x)y)` to `[(x)y]`. That is the correct form of P-16.5.4, and still a different name.
- An italic prefix such as `tert-butyl` is treated as bare by some predicates and as complex by others.

### Multiplying prefixes

Blue Book: <!-- rule-index:begin bb:P-14.2,P-16.3 -->P-14.2 Multiplicative Prefixes; P-16.3 Multiplicative prefixes 'di', 'tri', etc. vs. 'bis', 'tris', etc.<!-- rule-index:end bb:P-14.2,P-16.3 -->

`di`, `tri`, ... for simple prefixes and `bis`, `tris`, ... for compound ones. The choice between the two series is made from the name of the multiplied part (`get_multiplier_prefix`), or fixed by the class.

<!-- rule-index:begin multiplier -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `SIMPLE_MULTIPLIERS` | `data/multipliers.py` | 81 reads in 28 other files |
| `COMPLEX_MULTIPLIERS` | `data/multipliers.py` | 33 reads in 15 other files |
| `multiplier_table` | `data/multipliers.py` | 92 calls in 46 other files |
| `get_multiplier_prefix` | `assembly/naming_utils.py` | 36 calls in 16 other files |
| `simple_multiplier_word` | `assembly/naming_utils.py` | 9 calls in 6 other files |
| `get_suffix_multiplier_prefix` | `assembly/naming_utils.py` | 23 calls in 10 other files |
| `multiplier_needs_hyphen` | `assembly/naming_utils.py` | 7 calls in 5 other files |

Beside them: **4** dictionary literals that map integers to `di`, `tri`, ... (3) or `bis`, `tris`, ... (1) in **3 files**, including the two canonical tables. **2** of them stop at 10 or below; the canonical tables go to 20. Key `1` is absent from 2 of them, maps to the empty string in 2. What a copy does above its own range can differ by site (for example a `KeyError`, an empty string, or no name).

<!-- rule-index:end multiplier -->

When you change it: the tables for the suffix words `diyl`, `triyl`, `tetrayl` (`_POLYOL_VALENCE` in `rules/esters.py`, `_VALENCE_WORD` in `rules/multiplicative_general.py`) are not prefix tables. Replacing a copy by the canonical table can let that site name a molecule with more repeated groups than before, which is a change of names and not a tidy-up; do it in a change of its own.

### Numbering: lowest locants

Blue Book: <!-- rule-index:begin bb:P-14.4 -->P-14.4 Numbering.<!-- rule-index:end bb:P-14.4 -->

The first point of difference decides between two sets of locants. `compare_locant_sets` implements it, with the rule for letters such as `4a`.

<!-- rule-index:begin numbering -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `compare_locant_sets` | `rules/locants.py` | 38 calls in 14 other files |
| `compare_numbering` | `rules/locants.py` | 4 calls in 4 other files |

Functions named `locant_sort_key` or `_locant_sort_key`: **9**. **6** are defined at module level, in `assembly/name_comparison.py`, `assembly/naming_utils.py`, `assembly/universal_substituent.py`, `rules/isotopes.py`, `rules/p44_scorer.py`, `rules/spiro.py`. **3** are nested inside another function, in `assembly/composer.py`, `rules/fused_rings.py`, `rules/partial_saturation.py`. Other sites that compare locant sets:

- `perception/chains.py`, `_compare_locants`: first point of difference, re-written; breaks ties with `len <=`.
- `assembly/form_choice.py`, `lower_locants`: compares the locant roles of two finished names of one species, a different object.

<!-- rule-index:end numbering -->

When you change it: most code that compares sets of locants calls `compare_locant_sets`, but the criteria of P-14.4 are not held in one function. Ring numbering is decided inside each ring-system module, and the order of the criteria is spread over the parent-selection and ring modules. The small `locant_sort_key` functions are twins, not one function.

### Elision of vowels

Blue Book: <!-- rule-index:begin bb:P-16.7 -->P-16.7 Elision of Vowels.<!-- rule-index:end bb:P-16.7 -->

`apply_vowel_elision` is the canonical helper.

<!-- rule-index:begin elision -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `apply_vowel_elision` | `assembly/naming_utils.py` | 6 calls in 4 other files |
| `_ELISION_VOWELS` | `assembly/naming_utils.py` | 3 reads in 2 other files |

Vowel sets written out in the code: **24** in **17 files** (`aeiou`: 11, `aeiouy`: 11, `aiouy`: 2). The canonical helper uses `aiouy`: it does not elide a final `e` before another `e` (`apply_vowel_elision("butane", "ene")` gives `butaneene`). The Blue Book rule P-16.7.1 (a) lists `e` among the letters before which a final `e` is elided, and most of the other sets contain it. A change of the set can change a name only where a stem that ends in `e` meets a suffix that begins with `e`. Sites that decide the same question:

- `rules/multiplicative_general.py`, `_elide`: the same vowel set as the canonical helper, and true for an empty suffix.
- `rules/spiro.py`: strips the terminal 'e' before 'ene' with its own code (P-31.1.5.2.1 spiro names).
- `rules/heterocycles.py`: its own vowel sets for Hantzsch-Widman stems and endings.
- `assembly/general_engine.py`, `_elide_before_ionic_suffix`: the set `aeiouy`, which includes 'e'.
- `rules/mononuclear_hydrides.py`, `_apply_hydride_suffix`: the set `aeiouy`, which includes 'e'.
- `rules/ions.py`, `_elide_terminal_e`: drops a final 'e' without looking at the next letter.

<!-- rule-index:end elision -->

When you change it: P-31.1.5.2.1 prints a spiro name in which the final `e` of a saturated name is elided before the ending `ene` (`2,2'-spirobi[bicyclo[2.2.1]heptan]-5-ene`). The code that writes that name, in `rules/spiro.py` (`if not _mult and component.endswith('e')`), strips the `e` itself and does not call the helper. If you make it call the helper as it is, the name becomes `...heptane]-5-ene`. Widen the helper's vowel set first, and check both.

### Hydro prefixes, indicated and added hydrogen

Blue Book: <!-- rule-index:begin bb:P-14.7,P-31.2 -->P-14.7 Indicated and 'Added Indicated' Hydrogen; P-31.2 Substituent Groups Modified by the Prefixes 'hydro' or 'dehydro'.<!-- rule-index:end bb:P-14.7,P-31.2 -->

`rules/ring_hydrogen.py` holds the shared description of the hydrogen of a ring system whose atoms carry a double bond to an atom outside the ring (a ketone, an imine, an oxo prefix). Its docstring quotes the sections of the Blue Book that govern it.

<!-- rule-index:begin hydro -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `ring_hydrogen` | `rules/ring_hydrogen.py` | 11 calls in 5 other files |
| `added_text` | `rules/ring_hydrogen.py` | 2 calls in 2 other files |
| `hydro_text` | `rules/ring_hydrogen.py` | 2 calls in 2 other files |
| `parent_prefix` | `rules/ring_hydrogen.py` | 5 calls in 3 other files |
| `numbering_key` | `rules/ring_hydrogen.py` | 7 calls in 3 other files |

The placement itself is computed by `rules/bridged_fused_pin/hydro.py`; `ring_hydrogen` is the entry that other producers call. Producers that write hydro prefixes or indicated hydrogen without it, or on only some of their paths:

- `rules/partial_saturation.py`: writes `{locant}H-` parents itself.
- `rules/purine.py`: writes `{indicated hydrogen}H-purine` itself.
- `rules/ring_substituents.py`: writes hydro and indicated-hydrogen strings itself and never calls the helpers.
- `rules/fused_rings.py`: uses the helpers on some paths and writes `hydro` and `H` strings directly on others.
- `rules/heterocycles.py`: uses the helpers on some paths and writes `hydro` and `H` strings directly on others.

<!-- rule-index:end hydro -->

When you change it: a producer that writes `<locants>-<multiplier>hydro` or `{locant}H-` itself is not changed by a change to the helpers. Search for the strings, not only for the function.

### Alphanumerical order

Blue Book: <!-- rule-index:begin bb:P-14.5 -->P-14.5 Alphanumerical Order.<!-- rule-index:end bb:P-14.5 -->

`alpha_sort_key` is the canonical key. It leaves out multiplying prefixes, stereodescriptors and Greek letters, and it keeps `iso`, `neo` and `cyclo` (they are part of the name). `sec-` and `tert-` are not part of the primary key: P-14.5.3 makes italicised letters a last tie-break, so `3-tert-butyl-1-(1-methylpropyl)benzene` comes out right. It is used widely.

<!-- rule-index:begin alpha -->

| Helper | Module | Used outside its module |
|:--|:--|:--|
| `alpha_sort_key` | `assembly/naming_utils.py` | 149 calls in 37 other files |
| `strip_alphanumerical_noise` | `assembly/naming_utils.py` | 2 calls in 2 other files |
| `cip_descriptor_rank_key` | `assembly/naming_utils.py` | 5 calls in 4 other files |
| `prefix_citation_sort_key` | `assembly/naming_utils.py` | 10 calls in 4 other files |

Orderings of names that do not go through `alpha_sort_key`:

- `rules/benzene.py`, `_alphanumerical_name_key`: letters only; no multiplier or italic handling (a P-45.5 variant).
- `rules/spiro.py`, `_component_alpha_key`: orders ring-component names.
- `rules/spiro.py`, `_pin_component_alpha_key`: orders ring-component names; adds a locant tie-break.
- `rules/spiro.py`, `_spiro_component_alpha_key`: orders ring-component names; strips every bracketed group.
- `rules/lipids.py`, `_alpha_key`: strip_alphanumerical_noise only.

<!-- rule-index:end alpha -->

When you change it: sorting a list of names with Python's `sorted` is not alphanumerical order; it differs for names with `tert-`, multiplying prefixes, Greek letters or italic `N-`. When you meet a sort of name strings, look for the key.

## 2. Where each kind of rule lives

An entry point is the function that the dispatch tables or the handlers call. Most of the modules hold many more functions; the [internals pages](reference/internals/index.md) list them all.

<!-- rule-index:begin topics -->

| Kind of rule (Blue Book section) | Where it lives |
|:--|:--|
| Which group is the principal characteristic group (the suffix)<br>*P-41 Seniority Order for Classes; P-43 Seniority Order for Suffixes* | `rules/seniority.py`: `get_principal_group`<br>`rules/seniority.py`: `SENIORITY_ORDER`<br>`perception/functional_groups.py`: `detect_functional_groups`<br>The SMARTS patterns that find groups are in `FUNCTIONAL_GROUP_SMARTS` of the same perception module. |
| Ring or chain as the parent<br>*P-44.1 Seniority Order for Parent Structures* | `rules/parent_selection.py`: `select_parent`<br>Called from `Orthonym._classify`. |
| Which ring system, which chain<br>*P-44.2 Seniority Order only for Rings and Ring Systems; P-44.3 Seniority of Acyclic Chains (the Principal Chain)* | `rules/ring_selection.py`: `select_principal_ring_system`<br>`perception/chains.py`: `find_principal_chain` |
| Lowest locants, choice of numbering<br>*P-14.4 Numbering* | `rules/locants.py`: `compare_locant_sets`<br>`rules/locants.py`: `orient_chain`<br>Ring numbering is decided inside each ring-system module. |
| Monocyclic heterocycles (Hantzsch-Widman names)<br>*P-22.2 Heteromonocyclic Parent Hydrides* | `rules/heterocycles.py`: `build_hw_name`<br>`rules/heterocycles.py`: `name_heterocycle`<br>Stems and heteroatom prefixes are in `data/hw_stems.py` and `data/hw_heteroatoms.py`. |
| Von Baeyer polycyclic parents<br>*P-23 Polyalicyclic Parent Hydrides (Extended von Baeyer System)* | `rules/polycyclic.py`: `name_polycyclic_complete`<br>`rules/bicyclo.py`: `name_bicyclo_system`<br>`rules/vonbaeyer_universal.py`: `analyze_cage_universal` |
| Spiro systems<br>*P-24 Spiro Ring Systems* | `rules/spiro.py`: `name_spiro_system` |
| Fused ring systems<br>*P-25 Fused and Bridged Fused Ring Systems* | `rules/fused_rings.py`: `name_fused_heterocycle`<br>`rules/fusion_descriptors.py`: `generate_systematic_name_for_fused_pair`<br>`rules/fusion_numbering.py`: `compute_fused_numbering`<br>Retained fused names are in `data/fused_heterocycles.py`; bridged fused names are built by `rules/bridged_fused_pin/`. |
| Phanes<br>*P-26 Phane Nomenclature* | `rules/phane.py`: `name_cyclophane` |
| Ring assemblies<br>*P-28 Ring Assemblies* | `rules/ring_assemblies.py`: `name_ring_assembly` |
| Benzene derivatives<br>*P-22.1 Monocyclic Hydrocarbons; P-22.1.2 Unsubstituted monocyclic hydrocarbon polyenes* | `rules/benzene.py`: `name_substituted_benzene`<br>P-22.1.2 makes benzene the retained name for C6H6 (`benzene (PIN; retained name)`); the substituent rules are those of P-29 and P-61. |
| Hydro prefixes, indicated and added hydrogen<br>*P-14.7 Indicated and 'Added Indicated' Hydrogen; P-31.2 Substituent Groups Modified by the Prefixes 'hydro' or 'dehydro'* | `rules/ring_hydrogen.py`: `ring_hydrogen`<br>`rules/partial_saturation.py`: `detect_partial_saturation` |
| Substituent prefixes | `assembly/substituent_enumerator.py`: `name_substituent`<br>`assembly/universal_substituent.py`: `name_universal_substituent_prefix`<br>`rules/ring_substituents.py`: `get_ring_substituent_name`<br>Three families name a prefix; see the architecture page. |
| Multiplicative nomenclature<br>*P-51 Selecting the Preferred Type of IUPAC Nomenclature* | `rules/multiplicative.py`: `name_multiplicative` |
| Esters | `rules/esters.py`: `name_ester` |
| Amides, nitriles | `rules/amides.py`: `name_amide`<br>`rules/nitriles.py`: `name_nitrile` |
| Acid halides, anhydrides, lactones, lactams | `rules/acid_halides.py`: `name_acid_halide`<br>`rules/anhydrides.py`: `name_anhydride`<br>`rules/lactones.py`: `name_monocyclic_lactone`<br>`rules/lactams.py`: `name_monocyclic_lactam` |
| Molecules with several kinds of group | `rules/polyfunctional.py`: `name_polyfunctional` |
| Ions, salts, zwitterions, radicals, adducts | `rules/charged_router.py`: `route_charged`<br>`rules/ions.py`: `name_anion`<br>`rules/ions.py`: `name_cation`<br>`rules/salts.py`: `name_salt`<br>`rules/radicals.py`: `name_radical`<br>`rules/adducts.py`: `name_adduct`<br>`route_charged` is the single choke point for the parent of a charged species. |
| Hydrides of the other main-group elements, inorganic acids | `rules/mononuclear_hydrides.py`: `name_mononuclear_hydride`<br>`rules/inorganic_acids.py`: `name_inorganic_acid`<br>`rules/phosphorus.py`: `name_phosphine` |
| Organometallic compounds | `rules/organometallics.py`: `assemble_organometallic_name` |
| Isotopic modification | `rules/isotopes.py`: `decorate_isotopic_name` |
| Stereodescriptors: finding them, writing them | `perception/stereo.py`: `assign_stereochemistry`<br>`rules/stereochemistry.py`: `collect_stereodescriptors`<br>Call `assign_stereochemistry`, never RDKit's CIP labelling directly. |
| Natural products, sugars, amino acids, peptides, lipids | `rules/natural_products.py`: `name_natural_product`<br>`rules/oligosaccharides.py`: `name_disaccharide`<br>`rules/amino_acids.py`: `name_amino_acid`<br>`rules/peptides.py`: `name_peptide`<br>`rules/lipids.py`: `name_lipid`<br>Their tables are in `data/natural_products.py`, `data/sugar_names.py` and `data/amino_acids.py`. |
| Skeletal replacement ('a' nomenclature) in chains | `rules/skeletal_replacement.py`: `try_skeletal_replacement_name` |
| Naming a large molecule from its fragments | `decomposition/engine.py`: `try_decompose` |

<!-- rule-index:end topics -->

## From a handler id to its code

`orthonym "SMILES" --dispatch-stats` prints ids of these two tables, and `--dump-tree` prints an inner handler id. The tables say which file holds the code. The outer table is walked in priority order and the first class that fits is used; the inner table is walked in priority order and the first handler that succeeds is used ([Architecture](architecture.md#3-class-routing-the-outer-table)).

### Outer classes

A class handler is a short shim in `routing/dispatch_table.py` that calls the code named in the last column.

<!-- rule-index:begin outer -->

| Priority | Class id | What it names | Code |
|:--|:--|:--|:--|
| 40 | `INORGANIC_ACID` | Free inorganic oxoacids with a retained name (sulfuric, phosphoric, carbonic, silicic acid) | `rules/inorganic_acids.py` |
| 45 | `MONONUCLEAR_HYDRIDE` | Mononuclear hydrides with a halogen-only or organyl hub (sulfur hexafluoride, trichlorophosphane, trimethylarsane) | `rules/mononuclear_hydrides.py` |
| 46 | `CHALCOGEN_CHAIN` | Chains of O, S, Se or Te atoms (disulfane, trioxidane) | `rules/polychalcogen.py` |
| 46.5 | `POLYCHALCOGEN_OXIDE` | Oxides of chalcogen chains (disulfane dione) | `rules/polychalcogen.py` |
| 47 | `POLYAZANE` | Hydrazine, diazene, triazane and relatives | `rules/polyazane.py` |
| 47.5 | `CATENATED_HYDRIDE` | Alternating Group 14 and 15 chains (disiloxane, disilazane) | `rules/catenated_hydrides.py` |
| 47.55 | `HETEROCHALCOGEN_ABA` | Chalcogen-only skeletal chains (dithioxane) | `rules/catenated_hydrides.py` |
| 47.6 | `HOMONUCLEAR_PNICTOGEN_CHAIN` | Chains of one Group 15 element (diphosphane, pentaarsane) | `rules/catenated_hydrides.py` |
| 47.65 | `PNICTOGEN_CARBOXYLIC_ACID` | Carboxylic acids on P, As or Sb hydrides (phosphanecarboxylic acid) | `rules/phosphorus.py` |
| 47.66 | `MONONUCLEAR_HYDRIDE_ADDED_CARBON` | Carbaldehydes and carbonitriles on Si, Ge, Sn, Pb, P, As, Sb, Bi hydrides (silanecarbonitrile) | `rules/mononuclear_hydrides.py` |
| 47.7 | `FREE_HOMONUCLEAR_G14_HYDRIDE` | Chains of one Group 14 element (disilane, digermene) | `rules/multiplicative.py` |
| 48 | `DINUCLEAR_HYDRIDE` | Two-atom Group 14 and 15 hydrides (silylarsane) | `rules/mononuclear_hydrides.py` |
| 48.3 | `AZINIC_DERIVATIVE` | Azinic acids (ethylideneazinic acid) | `rules/inorganic_acids.py` |
| 48.35 | `CUMULATIVE_ZWITTERION` | Zwitterions with onium and ide centres on one heteroatom chain | `rules/ions.py` |
| 48.36 | `YLIDE` | Ylides: an onium cation next to a carbanion | `rules/ions.py` |
| 48.4 | `HETERONE` | Heterones of Si, Ge, P, As (dimethylsilanone) | `rules/mononuclear_hydrides.py` |
| 48.5 | `SULFINE` | Sulfines (thiocarbonyl S-oxides) | `rules/mononuclear_hydrides.py` |
| 48.6 | `PSEUDOKETONE_HETERO` | Acyl groups on Si, Ge, P or As | `rules/pseudoketones.py` |
| 48.65 | `ACYL_CHALCOGENCHAIN_PSEUDOKETONE` | Acyl groups on a chalcogen chain of three or more atoms | `rules/pseudoketones.py` |
| 48.66 | `LAMBDA5_PHOSPHANIMINE` | Phosphanimines at lambda-5 | `rules/mononuclear_hydrides.py` |
| 48.7 | `HETEROIMINE` | Imines on a P, As or Si hydride | `rules/mononuclear_hydrides.py` |
| 48.75 | `LAMBDA_SULFANE_IMINE_OXIDE` | Sulfane imides and oxides at lambda-4 and lambda-6 | `rules/mononuclear_hydrides.py` |
| 49 | `KETENE` | Ketenes (ethenone and its halogen derivatives) | `rules/ketenes.py` |
| 49.5 | `RING_CHALCOGEN_OXIDE` | Oxides of ring chalcogens (dibenzo[b,d]thiophene 5-oxide) | `rules/ring_chalcogen_oxide.py` |
| 49.6 | `HYDRO_FUSED_PEROXOL` | Peroxols on partly hydrogenated fused carbocycles | `rules/partial_saturation.py` |
| 49.7 | `THIOIMIDE` | Acyclic thioimides | `rules/thioimides.py` |
| 49.75 | `NITRAMIDE_SUBSTITUTED` | Amides of the nitrogen and halogen oxoacids (nitramides, nitrous amide, hypochlorous amide) | `rules/nitramide.py` |
| 49.8 | `CYCLIC_POLYESTER` | Lactides and other cyclic di- or polyesters (glycolide) | `rules/lactones.py` |
| 50 | `ORGANOMETALLIC` | Metal-carbon compounds and sandwich complexes | `assembly/handlers/organometallic.py` |
| 100 | `SALT` | Salts of several components | `rules/salts.py` |
| 200 | `RADICAL` | Radicals | `data/`, `rules/charged_router.py`, `rules/radicals.py` |
| 250 | `LIPID` | Lipids built on glycerol, phosphoglycerol or a sphingoid base | `rules/lipids.py` |
| 300 | `ZWITTERION` | Zwitterions with net charge zero | `rules/salts.py` |
| 301 | `ESTER_ANION_ZWITTERION` | Net-charged ester-anion zwitterions (choline-type phosphate esters) | `perception/ions.py`, `rules/charged_router.py` |
| 302 | `MIXED_SIGN_ZWITTERION` | Net-charged ions with both cation and anion sites | `rules/charged_router.py`, `rules/ions.py` |
| 400 | `ANION_RETAINED` | Anions with a retained name (acetate, benzoate) | `rules/ions.py` |
| 480 | `CATION_QUATERNARY` | Quaternary ammonium cations | `rules/charged_router.py` |
| 490 | `METAL_ATOM` | Atoms, ions and isotopes of a metal the scope table admits (zinc, zinc(2+), (65Zn)zinc) | `rules/metal_atoms.py` |
| 500 | `CATION_RETAINED` | Cations with a retained name | `rules/ions.py` |
| 600 | `ANION_SMALL` | Single anions of up to 25 heavy atoms | `rules/acid_ester_anion.py`, `rules/ions.py`, `rules/charged_router.py`, `perception/ions.py` |
| 700 | `POLY_ANION` | Anions with two or more anionic sites | `rules/acid_ester_anion.py`, `rules/charged_router.py`, `rules/ions.py`, `perception/ions.py` |
| 800 | `MULTI_COMPONENT_NEUTRAL` | Neutral co-crystals and solvates, each component named in turn | `rules/adducts.py` |
| 900 | `MULTIPLICATIVE` | Symmetric molecules joined by a central group (multiplicative names) | `rules/multiplicative.py` |
| 1000 | `CARBOHYDRATE_LOOKUP` | Carbohydrates: table lookup, then anomer and configuration | `data/sugar_names.py`, `rules/oligosaccharides.py` |
| 1100 | `NATURAL_PRODUCT` | Steroids, alkaloids and terpenes | `data/natural_products.py`, `rules/bridged_fused_pin/` |
| 1200 | `PEPTIDE` | Peptides | `rules/peptides.py`, `decomposition/engine.py`, `assembly/nested_memo.py` |
| 1300 | `RETAINED_NAME` | Whole molecules that have a retained name (naphthalene) | `data/` |
| 1400 | `AMINO_ACID` | Amino acids | `rules/amino_acids.py` |
| 1500 | `SKELETAL_REPLACEMENT` | Chains with heteroatoms in the skeleton ('a' nomenclature) | `rules/skeletal_replacement.py` |
| 1600 | `CYCLOPHANE` | Cyclophanes | `rules/phane.py`, `rules/phane_pin/` |
| 1650 | `LINEAR_PHANE` | Linear phanes (four or more rings or ring systems, two terminal, seven or more nodes) | `rules/phane_pin/` |
| 1700 | `INOSITOL` | Inositols (cyclitols) | `rules/inositols.py` |
| 1800 | `NUCLEOSIDE` | Nucleosides and nucleotides with a decoration (phosphates, acyl esters) | `rules/nucleosides.py` |
| 98000 | `N_HETERO_AMINE` | Amines, carboxamides and sulfonamides with an azido, alkoxy, sulfanyl, nitro, nitroso or halogen group on N (N-azido-N-methylmethanamine) | `rules/n_hetero_amines.py` |
| 99000 | `DECOMPOSITION_PRE_GENERAL` | Naming a large molecule from its fragments; may decline, and the pipeline goes on | `decomposition/` |
| 99999 | `GENERAL` | Catch-all. It returns nothing on purpose, and `_name_impl` runs the general pipeline | `namer.py`, `_name_impl` |

<!-- rule-index:end outer -->

### Inner handlers

<!-- rule-index:begin inner -->

| Priority | Handler id | What it names | File |
|:--|:--|:--|:--|
| 50 | `ion_dispatch` | Ions, salts, zwitterions and radicals, before the candidate pool | `assembly/handlers/ion_dispatch.py` |
| 100 | `oxime` | Oximes | `assembly/handlers/oxime.py` |
| 200 | `hydrazone` | Hydrazones | `assembly/handlers/hydrazone.py` |
| 300 | `n_oxide` | N-oxides | `assembly/handlers/n_oxide.py` |
| 400 | `isocyanate` | Isocyanates | `assembly/handlers/isocyanate.py` |
| 500 | `isothiocyanate` | Isothiocyanates | `assembly/handlers/isothiocyanate.py` |
| 600 | `carbamic_acid` | Carbamic acids | `assembly/handlers/carbamic_acid.py` |
| 700 | `carbamate` | Carbamates | `assembly/handlers/carbamate.py` |
| 800 | `urea` | Ureas | `assembly/handlers/urea.py` |
| 810 | `thiourea` | Thioureas and chalcogen ureas | `assembly/handlers/thiourea.py` |
| 900 | `guanidine` | Guanidines | `assembly/handlers/guanidine.py` |
| 950 | `cyanamide` | Cyanamides | `assembly/handlers/cyanamide.py` |
| 975 | `nitrile_oxide` | Nitrile oxides | `assembly/handlers/nitrile_oxide.py` |
| 1000 | `boronic_acid` | Boronic acids | `assembly/handlers/boronic_acid.py` |
| 1100 | `acid_halide` | Acyl halides | `assembly/handlers/acid_halide.py` |
| 1150 | `sulfonyl_halide` | Sulfonyl and sulfinyl halides | `assembly/handlers/sulfonyl_halide.py` |
| 1175 | `imidoyl_thioyl_halide` | Imidoyl, thioyl and selenoyl halides | `assembly/handlers/imidoyl_thioyl_halide.py` |
| 1200 | `anhydride` | Acid anhydrides | `assembly/handlers/anhydride.py` |
| 1300 | `lactone` | Monocyclic lactones | `assembly/handlers/lactone.py` |
| 1400 | `lactam` | Monocyclic lactams | `assembly/handlers/lactam.py` |
| 1450 | `ring_ester` | Ester groups on a ring parent (acyloxy prefix) | `assembly/handlers/ring_ester.py` |
| 1490 | `chain_diamide` | Chain diamides, with N-substituents | `assembly/handlers/diamide.py` |
| 1495 | `noncarbon_ester` | Esters of non-carbon acids (sulfonic, sulfinic, pseudoesters) | `assembly/handlers/noncarbon_ester.py` |
| 1500 | `ester_family` | Polyfunctional molecules, multiple esters and esters, tried in that order. Names many molecules that are not esters | `assembly/handlers/ester_family.py` |
| 1800 | `sulfoxide` | Sulfoxides | `assembly/handlers/sulfoxide.py` |
| 1801 | `selenoxide` | Selenoxides | `assembly/handlers/chalcogen_oxide.py` |
| 1802 | `telluroxide` | Telluroxides | `assembly/handlers/chalcogen_oxide.py` |
| 1900 | `sulfone` | Sulfones | `assembly/handlers/sulfone.py` |
| 1901 | `selenone` | Selenones | `assembly/handlers/chalcogen_oxide.py` |
| 1902 | `tellurone` | Telluroxones | `assembly/handlers/chalcogen_oxide.py` |
| 2000 | `thioether` | Sulfides (thioethers) | `assembly/handlers/thioether.py` |
| 2100 | `phosphine_oxide` | Phosphine oxides | `assembly/handlers/phosphine_oxide.py` |
| 2200 | `phosphate_ester` | Phosphate esters | `assembly/handlers/phosphate_ester.py` |
| 2205 | `sulfate_ester` | Sulfate esters | `assembly/handlers/sulfate_ester.py` |
| 2210 | `pnictogen_inate_ester` | Esters of phosphinic, arsinic and stibinic acids | `assembly/handlers/pnictogen_inate_ester.py` |
| 2300 | `phosphine` | Phosphines | `assembly/handlers/phosphine.py` |
| 2350 | `phosphonic_acid` | Phosphonic acids | `assembly/handlers/phosphonic_acid.py` |
| 2352 | `arsonic_acid` | Arsonic acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2354 | `arsinic_acid` | Arsinic acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2356 | `stibonic_acid` | Stibonic acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2358 | `stibinic_acid` | Stibinic acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2360 | `phosphonous_acid` | Phosphonous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2362 | `phosphinous_acid` | Phosphinous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2364 | `arsonous_acid` | Arsonous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2366 | `arsinous_acid` | Arsinous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2368 | `stibonous_acid` | Stibonous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2370 | `stibinous_acid` | Stibinous acids | `assembly/handlers/pnictogen_oxoacid.py` |
| 2400 | `phosphinic_acid` | Phosphinic acids | `assembly/handlers/phosphinic_acid.py` |
| 2500 | `ring_assembly` | Ring assemblies (biphenyl and the like) | `assembly/handlers/ring_assembly.py` |
| 2600 | `polycyclic` | Polycyclic aromatic hydrocarbons with a retained name | `assembly/handlers/polycyclic.py` |
| 2700 | `partial_sat` | Partly saturated carbocycles (tetralin shape) | `assembly/handlers/partial_sat.py` |
| 2800 | `simple_molecule` | Single atoms, noble gases, small simple molecules | `assembly/handlers/simple_molecule.py` |
| 2900 | `imidate` | Imidates (iminoesters) | `assembly/handlers/imidate.py` |
| 2950 | `chalcogen_ester` | Selenium and tellurium esters | `assembly/handlers/chalcogen_ester.py` |
| 2960 | `nitrite_ester` | Nitrite esters | `assembly/handlers/nitrite_ester.py` |
| 2962 | `nitrate_ester` | Nitrate esters | `assembly/handlers/nitrate_ester.py` |
| 2965 | `thiocyanate` | Thiocyanates | `assembly/handlers/thiocyanate.py` |
| 2970 | `n_substituted_sulfonamide` | N-substituted sulfonamides | `assembly/handlers/n_substituted_sulfonamide.py` |
| 4500 | `tier_a_ring` | The ring cascade: complex rings, polycycles, partly saturated rings, heterocycles, benzene, then a chain; picks among the candidates | `assembly/handlers/tier_a_ring.py` |
| 5100 | `ring_nitrile` | Nitriles on a ring | `assembly/handlers/ring_nitrile.py` |
| 5200 | `amide` | Amides with one amide group | `assembly/handlers/amide.py` |
| 5240 | `azine` | Azines | `assembly/handlers/azine.py` |
| 5250 | `hydroxylamine` | Substituted hydroxylamines | `assembly/handlers/hydroxylamine.py` |
| 5290 | `imine` | N-substituted acyclic imines | `assembly/handlers/imine.py` |
| 5300 | `amine` | Secondary and tertiary amines | `assembly/handlers/amine.py` |
| 99999 | `general_acyclic` | Catch-all for chains and ring substituents; always returns a name or raises | `assembly/handlers/general_acyclic.py` |

<!-- rule-index:end inner -->

## Where the code mentions each Blue Book section

The number of times the code (comments, docstrings and strings) mentions a section, with the modules that mention it most. A mention is not a promise that the module implements the section, and a number copied into a comment can be wrong. Use the table to find the neighbourhood of a rule, then read.

<!-- rule-index:begin chapters -->

| Section | Title | Mentions | Modules that mention it most |
|:--|:--|:--|:--|
| P-14 | General Rules | 1446 | `assembly/naming_utils.py` (110), `assembly/composer.py` (95), `assembly/handlers/_handler_shared.py` (92) |
| P-15 | Types of Nomenclature | 276 | `rules/multiplicative.py` (92), `rules/skeletal_replacement.py` (29), `assembly/universal_substituent.py` (12) |
| P-16 | Name Writing | 1026 | `assembly/naming_utils.py` (220), `assembly/composer.py` (91), `assembly/substituent_naming.py` (68) |
| P-21 | Mononuclear and Acyclic Polynuclear Parent Hydrides | 108 | `routing/dispatch_table.py` (25), `rules/catenated_hydrides.py` (16), `rules/polychalcogen.py` (10) |
| P-22 | Monocyclic Parent Hydrides | 200 | `rules/heterocycles.py` (70), `rules/skeletal_replacement.py` (21), `rules/monocycle_forms.py` (17) |
| P-23 | Polyalicyclic Parent Hydrides (Extended von Baeyer System) | 199 | `rules/polycyclic.py` (82), `rules/ring_replacement.py` (25), `rules/vonbaeyer_universal.py` (22) |
| P-24 | Spiro Ring Systems | 262 | `rules/spiro.py` (196), `assembly/composer.py` (31), `data/fused_heterocycles.py` (7) |
| P-25 | Fused and Bridged Fused Ring Systems | 635 | `rules/fused_rings.py` (96), `data/fused_heterocycles.py` (62), `rules/bridged_fused_pin/fusion_components.py` (30) |
| P-26 | Phane Nomenclature | 133 | `rules/phane.py` (51), `rules/phane_pin/naming.py` (19), `rules/phane_pin/numbering.py` (17) |
| P-28 | Ring Assemblies | 140 | `rules/ring_assemblies.py` (101), `rules/ring_substituents.py` (13), `rules/ring_assembly_screen.py` (6) |
| P-29 | Prefixes Denoting Substituent Groups Derived from Parent Hydrides | 359 | `assembly/substituent_naming.py` (65), `assembly/substituent_enumerator.py` (59), `rules/ring_substituents.py` (38) |
| P-31 | Modification of the Degree of Hydrogenation of Parent Hydrides | 416 | `rules/spiro.py` (41), `rules/polycyclic.py` (27), `rules/heterocycles.py` (26) |
| P-35 | Prefixes Corresponding to Characteristic Groups | 61 | `rules/seniority.py` (17), `assembly/naming_utils.py` (12), `assembly/composer.py` (8) |
| P-41 | Seniority Order for Classes | 220 | `rules/benzene.py` (21), `rules/fused_rings.py` (15), `rules/polycyclics.py` (12) |
| P-44 | Seniority order for parent structures | 448 | `rules/ring_selection.py` (91), `rules/p44_scorer.py` (51), `namer.py` (45) |
| P-45 | Selection of the Preferred IUPAC Name | 175 | `perception/chains.py` (36), `rules/isotopes.py` (30), `rules/benzene.py` (20) |
| P-46 | The Principal Chain in Substituent Groups | 70 | `assembly/substituent_naming.py` (61), `assembly/capped_name.py` (2), `assembly/substituent_enumerator.py` (2) |
| P-51 | Selecting the Preferred Type of IUPAC Nomenclature | 104 | `rules/multiplicative.py` (29), `rules/skeletal_replacement.py` (23), `rules/pin_vocabulary.py` (13) |
| P-52 | Selecting Preferred IUPAC Names and Preselected Names for Parent Hydrides | 128 | `routing/dispatch_table.py` (12), `rules/bridged_fused_pin/selection.py` (11), `rules/phane.py` (9) |
| P-54 | Selecting the Preferred Method for Modifying the Degree of Hydrogenation | 44 | `rules/ring_assembly_screen.py` (10), `data/fused_heterocycles.py` (8), `rules/heterocycles.py` (8) |
| P-58 | Selecting Preferred IUPAC Names | 139 | `rules/partial_saturation.py` (27), `rules/fused_rings.py` (24), `rules/bridged_fused_pin/hydro.py` (17) |
| P-59 | Name Construction | 91 | `assembly/universal_substituent.py` (23), `rules/fused_rings.py` (9), `perception/ions.py` (8) |
| P-61 | Substitutive Nomenclature: Prefix Mode | 73 | `rules/benzene.py` (14), `rules/mononuclear_hydrides.py` (12), `assembly/substituent_naming.py` (7) |
| P-62 | Amines and Imines | 146 | `assembly/composer.py` (17), `rules/ring_substituents.py` (13), `assembly/substituent_naming.py` (10) |
| P-63 | Hydroxy Compounds, Ethers, Peroxols, Peroxides, and Chalcogen Analogues | 327 | `assembly/substituent_enumerator.py` (39), `rules/seniority.py` (36), `assembly/substituent_prefix_forms.py` (31) |
| P-64 | Ketones, Pseudoketones, Heterones, and Chalcogen Analogues | 137 | `routing/dispatch_table.py` (19), `rules/partial_saturation.py` (14), `assembly/composer.py` (12) |
| P-65 | Acids, Acyl Halides and Pseudohalides, Salts, Esters, and Anhydrides | 903 | `rules/seniority.py` (89), `perception/functional_groups.py` (78), `rules/esters.py` (66) |
| P-66 | Amides, Imides, Hydrazides, Nitriles, and Aldehydes | 809 | `assembly/composer.py` (77), `rules/benzene.py` (75), `assembly/substituent_naming.py` (57) |
| P-67 | Mononuclear and Polynuclear Noncarbon Acids and their Functional Replacement Analogues as Functional Parents for Naming Organic Compounds | 414 | `rules/inorganic_acids.py` (126), `rules/functional_replacement.py` (33), `rules/phosphorus.py` (27) |
| P-68 | Nomenclature of Organic Compounds of the Group 13, 14, 15, 16, and 17 Elements not Included in Sections P-62 through P-67 | 341 | `routing/dispatch_table.py` (51), `rules/mononuclear_hydrides.py` (31), `rules/polyazane.py` (22) |
| P-69 | Nomenclature for Organometallic Compounds | 147 | `rules/organometallics.py` (32), `perception/metals.py` (21), `namer.py` (14) |
| P-71 | Radicals | 132 | `rules/radicals.py` (76), `rules/charged_router.py` (25), `rules/ions.py` (18) |
| P-72 | Anions | 233 | `rules/ions.py` (92), `rules/charged_router.py` (61), `routing/dispatch_table.py` (15) |
| P-73 | Cations | 279 | `rules/ions.py` (101), `rules/charged_router.py` (65), `data/ion_retained_names.py` (14) |
| P-74 | Zwitterions | 237 | `rules/charged_router.py` (57), `rules/ions.py` (34), `assembly/general_engine.py` (23) |
| P-82 | Isotopically Substituted Compounds | 285 | `rules/isotopes.py` (242), `assembly/locant_omission.py` (12), `assembly/naming_utils.py` (5) |
| P-91 | Stereoisomer Graphical Representation and Naming | 110 | `assembly/substituent_naming.py` (18), `rules/stereochemistry.py` (18), `namer.py` (9) |
| P-93 | Configuration Specification | 70 | `namer.py` (13), `rules/stereochemistry.py` (10), `rules/relative_stereo.py` (9) |
| P-101 | Nomenclature for Natural Products Based on Parent Hydrides (Alkaloids, Steroids, Terpenes, Carotenes, Corrinoids, Tetrapyrroles, and Similar Compounds) | 55 | `rules/natural_products.py` (17), `namer.py` (15), `data/natural_products.py` (7) |
| P-102 | Carbohydrate Nomenclature | 150 | `data/sugar_names.py` (91), `rules/oligosaccharides.py` (26), `routing/dispatch_table.py` (13) |
| P-103 | Amino Acids and Peptides | 97 | `data/amino_acids.py` (35), `rules/peptides.py` (23), `decomposition/engine.py` (5) |

<!-- rule-index:end chapters -->

## Naming tables

The tables are Python modules in `src/orthonym/data/`, and the retained names written by hand take precedence over the tables imported from OPSIN's resource files (`data/opsin_imports/`).

<!-- rule-index:begin data -->

| Module | What it holds (first line of its docstring) |
|:--|:--|
| `data/amino_acids.py` | Amino acid data for trivial name lookups |
| `data/bicyclo_systems.py` | Retained names and data for bicyclic ring systems |
| `data/bluebook_not_names.py` | Blue Book tier-3 corpus: names the Blue Book marks ``not`` (discarded) |
| `data/cation_words.py` | Element -> cation word table for salt composition (IUPAC 2013 P-65.6.2.1) |
| `data/chain_names.py` | Centralized IUPAC chain naming module |
| `data/coordination_retained.py` | Retained-name table for N-coordinated metal-tetrapyrrole macrocycles |
| `data/fused_heterocycles.py` | Retained names for fused heterocyclic systems |
| `data/fusion_components.py` | Monocyclic component registry for algorithmic fused ring naming |
| `data/group_split_rules.py` | Phase 169 Plan-01: Group-split topology-table loader |
| `data/hw_heteroatoms.py` | Hantzsch-Widman heteroatom prefix data for systematic heterocycle naming |
| `data/hw_stems.py` | Hantzsch-Widman stem suffix data for systematic heterocycle naming |
| `data/ion_retained_names.py` | Retained names for common ions used in IUPAC nomenclature |
| `data/metal_scope.py` | Which metals the engine names beside organic ions, and by which mechanism |
| `data/multipliers.py` | The multiplying prefixes: ``di, tri, tetra, ...`` and ``bis, tris, tetrakis, ...`` |
| `data/natural_products.py` | Natural product scaffold data and derivative lookup tables |
| `data/organometallics.py` | Retained names + lookup tables for IUPAC P-69 organometallic compounds |
| `data/partial_saturation_refs.py` | Aromatic reference templates for partial saturation detection |
| `data/pin_policy.py` | Single loader for the curated IUPAC-2013 PIN allow/deny list |
| `data/polycyclic_data.py` | Polycyclic Aromatic Hydrocarbon (PAH) data for IUPAC naming |
| `data/resonance_templates.py` | Resonance-shifted (bond_orders, charges) chain templates for terminal |
| `data/retained_names.py` | Retained (trivial) names that are preferred over systematic names |
| `data/sugar_names.py` | Sugar retained names lookup table and glycosyloxy prefix formatting |
| `data/trivial_acids.py` | Acid name to acylate (-ate/-oate) conversion for ester naming |
| `data/triviality_controller_seed.py` | Phase 168 Plan-01: Triviality-controller seed-table loader |
| `data/xanthine_derivatives.py` | Xanthine derivatives data module for purine-based alkaloids |

<!-- rule-index:end data -->

## How the counts were made

Each count is made by walking the syntax trees of the modules under `src/orthonym` with Python's `ast` module. Modules of `validation/` are left out of the copy counts: the spelling checkers there re-encode the rules on purpose.

| Count | Pattern |
|:--|:--|
| Calls and reads of a helper | Calls of a function by name (a constant: reads of its name), outside the module that defines it |
| Locants written by hand | A string `","`, `", "`, `"/"` or `"-"` that joins an expression whose source mentions a locant; an f-string with a locant followed by text that starts with `-`; a locant plus `"-"`. The expression matches `loc` (not `block`, `local`), `locant`, `...locs` or `_l`. Counted by line |
| Wraps in brackets | An f-string that is exactly `(`, an expression and `)` (also `[ ]` and `{ }`); a concatenation `"(" + x + ")"`; an `if` that compares a character with a bracket and adds to a counter |
| Multiplier tables | A dictionary literal whose keys are all integers and whose values include `di` and `tri`, or `bis` and `tris` |
| Vowel sets | A string of at least four vowels from `aeiouy`, or a tuple, list or set of single vowels |
| Citations of a section | The text `P-` and a number, with optional sub-numbers, in the source of the module |
