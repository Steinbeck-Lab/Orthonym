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

## The three checks

**1. Every atom is named.** Every heavy atom of your structure must be claimed by exactly one part of the name, and every part must appear in the final name. This check decides for the names of the general engine (the `valid`, `complete` and `best-effort` tiers). On the strict path of the default tier it is recorded but does not decide; the OPSIN checks below do.

**2. OPSIN must be able to read the name.** At the `valid`, `complete` and `best-effort` tiers, a name OPSIN cannot read is not emitted.

**3. What OPSIN read must be your molecule.** At the `valid`, `complete` and `best-effort` tiers the full InChIKey must match: constitution, charge and stereochemistry. A name with a missing or wrong stereodescriptor is therefore not emitted there. At the default tier the check accepts the same full InChIKey, or the same constitution, charge and protonation state with no stereodescriptor that disagrees with your structure.

A candidate that fails is withdrawn, and the engine tries the next way of naming the molecule. When none passes, it declines and says why ([Declines](declines.md)).

## The one stereo repair

When a name read back with the right constitution but without some of its stereodescriptors, the engine adds the missing descriptors from your structure once and has OPSIN read the new name. The name ships only if the full InChIKey then matches. The provenance row shows it as `gate_outcome` `stereo_omission_full_key_recomposed`.

## Where OPSIN cannot read the name

Some correct names use words OPSIN does not know. Two kinds of name leave the engine without a full read-back, and the provenance row says so every time:

- **Names from exact-match lists.** Metal tetrapyrrole complexes (hemes, chlorophylls, cobalamins, siroheme, coenzyme F430) and a table of retained natural-product parent names are matched to your structure exactly, by InChIKey or canonical SMILES, and the listed name is given. OPSIN cannot read these names at all. A metal-complex list name shows `verified` `identity`.
- **A few name classes at the default tier.** Some preferred-name forms (inositols, phanes, some anhydrides and natural-product stereoparents among them) are built by their rules and shipped at the default tier although OPSIN cannot read them, or reads only their constitution. The row shows `gate_outcome` `carveout:<class>` or `self_consistency_constitution_only`, and the tier is lowered accordingly. At the `valid`, `complete` and `best-effort` tiers these names are declined instead.

If OPSIN itself does not answer (the Java runtime times out or stops), the candidate is withdrawn and the row shows `gate_outcome` `suppressed`. A name ships with `gate_outcome` `unavailable` only in the opt-in reduced mode without the jars (`ORTHONYM_ALLOW_REDUCED=1`, [Install](start/install.md)).

## See it for yourself

`--provenance` prints what happened to every name. [The provenance row](use/provenance.md) explains each field, and the web app draws the read-back structure next to yours ([In the browser](use/browser.md)).
