# How every name is checked

A name that reads well can still describe the wrong molecule. Orthonym does not trust its own name: before a name leaves the engine, the name is checked against your structure, and a candidate that fails is withdrawn.

## The round trip

1. **You give a structure.** RDKit reads the SMILES.
2. **Orthonym writes a name** from the rules of the IUPAC 2013 recommendations.
3. **OPSIN reads the name back** into a structure. OPSIN is an independent program that turns names into structures; it never saw your SMILES.
4. **The two structures are compared** by InChIKey, a short code computed from a structure that changes with its constitution, charge and stereochemistry.

When they match, the name ships. Here is caffeine, from the engine's own output:

```{raw} html
<div class="ot-roundtrip" role="group" aria-label="The round trip for caffeine">
  <div class="ot-rt-step"><span class="ot-rt-label">Your structure</span><code>Cn1cnc2c1c(=O)n(C)c(=O)n2C</code></div>
  <div class="ot-rt-step"><span class="ot-rt-label">Orthonym writes</span><span class="ot-name ot-rule--pin">1,3,7-trimethyl-3,7-dihydro-1<i>H</i>-purine-2,6-dione</span></div>
  <div class="ot-rt-step"><span class="ot-rt-label">OPSIN reads back</span><code>CN1C(N(C=2N=CN(C2C1=O)C)C)=O</code></div>
  <div class="ot-rt-step ot-rt-verdict"><span class="ot-rt-label">InChIKey, both sides</span><code>RYYVLZVUVIJVGH-UHFFFAOYSA-N</code><span class="ot-rt-same"><svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path d="M3 8.5 6.5 12 13 4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>Same molecule</span></div>
</div>
```

## The four checks

**1. Every atom is named.** Every heavy atom of your structure must be claimed by exactly one part of the name, and every part must appear in the final name. This check decides for the names of the general engine (the `valid`, `complete` and `best-effort` tiers). On the strict path of the PIN-only tier it is recorded but does not decide; the OPSIN checks below do.

**2. OPSIN must be able to read the name.** At the `valid`, `complete` and `best-effort` tiers, a name OPSIN cannot read is not emitted. The `best-effort` tier has one exception, for a molecule that has no other name: if the PIN-only tier ships a name whose stereodescriptors OPSIN cannot read (the lowercase `r` and `s` of `(1r,4r)-4-chloro-4-methylcyclohexan-1-ol`, label {tier}`pin_unverified`, `gate_outcome` `self_consistency_constitution_only`), `best-effort` ships that name with the same label. It is the name the PIN-only tier checked, so check 3 below is the PIN-only tier's for it.

**3. What OPSIN read must be your molecule.** At the `valid`, `complete` and `best-effort` tiers the full InChIKey must match: constitution, charge and stereochemistry. A name with a missing or wrong stereodescriptor is therefore not emitted there (the one `best-effort` name of check 2 excepted). At the PIN-only tier the check accepts the same full InChIKey, or the same constitution, charge and protonation state with no stereodescriptor that disagrees with your structure.

**4. OPSIN must read the name once.** Orthonym also rejects a name that OPSIN reads as ambiguous (warning `APPEARS_AMBIGUOUS`): a name that leaves out a locant it needs can read back to the right structure and still not say where a group stands. The engine first names the molecule again with every locant cited. A molecule with no other name keeps the ambiguous one at the best-effort tier, labelled below the preferred name, and the PIN-only tier declines it. The few names the Blue Book itself prints in a form OPSIN reads as ambiguous (`propylium`, `[1-(methylperoxy)sulfanyl]butan-1-one`) are not rejected.

A candidate that fails is withdrawn, and the engine tries the next way of naming the molecule. When none passes, it declines and says why ([Declines](declines.md)).

## The one stereo repair

When a name read back with the right constitution but without some of its stereodescriptors, the engine adds the missing descriptors from your structure once and has OPSIN read the new name. The name ships only if the full InChIKey then matches. The provenance row shows it as `gate_outcome` `stereo_omission_full_key_recomposed`.

The best-effort tier uses the same repair as its last offer. A molecule with stereo that no other path names is named without its stereo, by an engine of the same tier, and the stereo of your structure is composed on that name. The name is offered only when OPSIN reads it back to your structure's full InChIKey, the radical graph and the protonation site are the same, the exact isomeric SMILES are the same, OPSIN does not read it as ambiguous, and, for a lone-pair stereocentre, the configuration check of the exit passes. An input with an AND or OR group of enhanced stereo (CXSMILES `|&1:1|`, `|o1:1|`) is not offered a name by this repair, because its centres are not absolute. The row shows the same `gate_outcome`, the tier {tier}`systematic_verified` and `is_pin` false: the name cites every descriptor at the front of the complete name, with the locants of the stereo-free name, which is not the form of a preferred name (P-91.3), so a descriptor of a substituent can repeat a locant of the parent. A repeated locant does not say which unit a descriptor belongs to (`(3S,3R,...)` and `(3R,3S,...)` are read to two diastereomers), so such a name is offered only when every descriptor at the repeated locant is the same (`4S,4S,4S`) and the descriptors state every stereo unit of your structure, none of which is unspecified. Any other name with a repeated locant is declined, and the molecule keeps its decline.

## Where OPSIN cannot read the name

Some correct names use words OPSIN does not know. Two kinds of name leave the engine without a full read-back, and the provenance row says so every time:

- **Names from exact-match lists.** Metal tetrapyrrole complexes (hemes, chlorophylls, cobalamins, siroheme, coenzyme F430) and a table of retained natural-product parent names are matched to your structure exactly, by InChIKey or canonical SMILES. OPSIN cannot read these names at all. A metal-complex list name is returned at every tier. A natural-product parent name is returned at the PIN-only tier, unless the engine builds a systematic PIN for the structure: kaurane, taxane and gibbane, for example, get their bridged fused PIN instead. At every other tier a natural-product parent name is the last resort, given only when no verified systematic name is found (`germacrane`). A list name shows `verified` `identity`. A metal-complex list name has the tier {tier}`pin_verified`. A natural-product parent name has the tier {tier}`systematic_verified`, because the recommendations identify no preferred names for natural products (P-100), and shows `gate_outcome` `carveout:np_stereoparent`.
- **Inositols at the best-effort tier.** OPSIN cannot read an inositol name (`myo-inositol`), and the recommendations give these names no PIN (P-100). The best-effort tier returns one only when the engine's recogniser gives exactly that name for your structure, matched by InChIKey to one of the seven meso stereoisomers; the row shows `gate_outcome` `carveout:inositol` and the tier {tier}`best_effort`. The PIN-only tier declines it.
- **A few name classes at the PIN-only tier.** Some preferred-name forms (phanes among them) are built by their rules and shipped at the PIN-only tier although OPSIN cannot read them, or reads only their constitution. OPSIN itself reads a phane name with `--opsin main`, and the name ships when that read-back gives your full InChIKey; the default OPSIN 2.9.0 reads no phane name, so there only the monocyclic cyclophane of benzene rings ships, when the engine's own phane builder turns the name back into your structure ([Phane names](use/phane-names.md)). The row shows `gate_outcome` `carveout:<class>` or `self_consistency_constitution_only`, and the tier is lowered accordingly. At the `valid`, `complete` and `best-effort` tiers these names are declined instead, except that `best-effort` ships a name of `self_consistency_constitution_only` (stereodescriptors OPSIN cannot read) with the same label when it has no other name for the molecule (check 2).

A third kind of name is read back whole, through a twin. A few preferred spellings that the Blue Book prints are ones OPSIN has no grammar for: the bridge prefixes `sulfano`, `disulfano` and `azano` (`1,4-dihydro-1,4-sulfanonaphthalene`), isotope descriptors (`(N-2H2)aniline`, `ethan(2H)ol`, `methan(2H,18O)ol`) and the interior-atom locant of a fused parent (`1,3a^1,4-triazaphenalene`). A twin is the same structure in a spelling OPSIN reads: `epithio` for `sulfano`, `(N,N-2H2)` for `(N-2H2)`, `(2H1)` for `(2H)`, the locant `9b` for `3a^1`. At the PIN-only tier the name ships when OPSIN rejects it, OPSIN reads its twin to your full InChIKey (radical electrons and protonation site included) and OPSIN gives no warning that it read the twin as ambiguous. The twin proves the structure and never the spelling, and it is never shipped. The row shows `verified` `opsin_twin`, `gate_outcome` `twin_verified:<rules>`, `gates_passed` `twin_round_trip` and the tier {tier}`pin_verified`. At the other tiers the same structure gets a name OPSIN reads, such as `1,4-dihydro-1,4-epithionaphthalene` or a von Baeyer name, with the labels of that name.

Three twin families are different, because the engine has no other spelling for them: the halogen `-uide` anions (`diphenyliodanuide`, twin `diphenyl-λ3-iodanide`), the anion of a thioperoxol (`methane(SO-thioperoxolate)`, twin `methanesulfenate`) and organometallic names with halido ligands (`bromido(ethyl)nickel`, twin `bromo(ethyl)nickel`). These names ship through their twin at every tier, with the same labels. An organometallic name has the tier {tier}`systematic_verified`, because the recommendations give no preferred name for these compounds, so the PIN-only tier declines it.

A neutral thioperoxol is read back through its twin at the PIN-only tier only (`methane-SO-thioperoxol`, twin `methanesulfenic acid`; an `-OS-thioperoxol` is read as its `-peroxol` with the sulfur an oxygen). The other tiers name the group by prefixes, a name OPSIN reads whole (`(hydroxysulfanyl)methane`), with the labels of that name. The PIN spelling ships through its twin there only for a molecule for which the engine builds no such name.

A fourth kind of name is read back through its parts. A poly-anhydride name (`diacetic butanedioic dianhydride`) has no twin spelling: OPSIN reads two-word anhydrides only. Each acid word is read by OPSIN as `<word> acid`, and the anhydride built from those acids must have your full InChIKey. The row shows `verified` `opsin_acids`, `gate_outcome` `acid_split_verified:anhydride_acids`, `gates_passed` `acid_split_round_trip` and the tier {tier}`pin_verified`. This is done at the PIN-only tier only. The other tiers name the structure in a form OPSIN reads (`1,4-bis(acetyloxy)-1,4-dioxobutane`).

A fifth kind of name is read back by rebuilding the structure from its text. OPSIN 2.9.0 has a numerical term up to 50 (`pentaconta`) and none above, so a name with a count word of 51 and more has no OPSIN reading and no twin spelling: `tetrapentacontafluorohexacosane`, or the cumulene of 55 carbon atoms `pentapentaconta-1,2,...,54-tetrapentacontaene`. A reader takes the parent chain, its multiple bonds and its prefixes from the text of the name alone, composes the numerical terms again from Table 1.4 of the Blue Book, and the reconstructor builds the structure from those facts. The structure must be the constitution of your input, and the reader gives no verdict for an input that asserts stereo or for any name it does not wholly understand: a token left over, a locant outside the chain, a locant list that does not match its multiplier, a prefix it does not know. The row shows `verified` `reconstructor`, `gate_outcome` `reconstructed:long_count`, `gates_passed` `name_reconstruction` and the tier {tier}`systematic_verified`. The check proves the structure and never the spelling, so the PIN-only tier declines the name (`NO_VERIFIED_PIN`) and the wider tiers return it.

If OPSIN itself does not answer (the Java runtime times out or stops), the candidate is withdrawn and the row shows `gate_outcome` `suppressed`. A name ships with `gate_outcome` `unavailable` only in the opt-in reduced mode without the jars (`ORTHONYM_ALLOW_REDUCED=1`, [Install](start/install.md)).

## Compounds of metals

A standard InChIKey does not record the bonds between a metal and its ligands. Zinc chloride drawn with two zinc-chlorine bonds and drawn as a zinc ion with two chloride ions have one key, so the round trip cannot tell the two drawings apart, and a producer may name one where it cannot name the other.

When Orthonym names a metal compound through another drawing of it, three things must hold:

- the two drawings have the same standard InChIKey;
- a separate count, which does not use InChI, finds the same atoms, isotopes, hydrogens and bonds apart from the broken bonds, and charges that differ only by the charge those bonds move: one positive charge on the metal and one negative charge on the other atom for each broken bond;
- the metal ends on a charge that Orthonym allows for it (zinc: 2+).

A drawing that fails a test is named as drawn, or not at all. The round trip then confirms the atoms, the charges and the organic part of the name, not which atom the metal is bonded to. The name describes the other drawing, so it is {tier}`systematic_verified`, never {tier}`pin_verified`. The provenance row says so in `spelling_failures`.

A drawing with dative bonds has no standard InChIKey in RDKit, so there is no key to read a name back against. Its name is read back to the same drawing without the dative bonds, which has a key, and the same count of atoms and charges ties the two drawings. The row shows `verified` `opsin_split`.

## See it for yourself

`--provenance` prints what happened to every name. [The provenance row](use/provenance.md) explains each field, and the web app draws the read-back structure next to yours ([In the browser](use/browser.md)).
