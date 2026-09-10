""" core-namer: ring-bearing compound-substituent silent-drop witnesses.

At HEAD (before the core-namer fix) each witness ABSTAINS at the best-effort
tier: a ring-parent substituent enumerator names the parent but silently drops a
ring-bearing compound substituent it cannot render, and SELF-01 then suppresses
the atom-incomplete partial name (final output ``unknown organic compound``).

Two root causes were found (full trace trail:
`internal notes` + the fix
commits):

1. The fused-heterocycle substituent enumerator
   (``rules/fused_rings.py::_identify_fused_substituent``) never delegated a
   ring-bearing compound substituent rooted at a HETEROATOM linker. The
   pantoprazole C2 ``-S(=O)-CH2-[pyridine]`` half is rooted at the sulfinyl S, so
   it never reached the carbon branch's ring delegate at ``:1724`` -- it hit the
   sulfur branch, which has no ring delegate, returned ``None``, and the caller
   silently ``continue``d.
2. The chalcogen prefix primitive
   (``assembly/substituent_prefix_forms.py::get_sulfanyl_prefix``) named a
   sulfinyl ``-S(=O)-`` as ``sulfanyl``, silently dropping the =O -> a wrong
   (reduced-thioether) molecule that SELF-01 suppressed to an abstention.

The fix threads the best-effort context into the un-wired ring-parent substituent
enumerators and makes ``get_sulfanyl_prefix`` fail closed on a non-divalent
chalcogen (so the fragment degrades to the oxo-preserving replacement name).
Each witness must then emit a name that round-trips through OPSIN to the full
input InChIKey (0-wrong).

Every witness below was A/B-verified with ``scripts/an A/B check`` to ABSTAIN at
HEAD and to NAME + round-trip with the fix. Fresh process per witness
(warm-cache hazard -- CLAUDE.md).
"""
import json
import subprocess
import sys

import pytest

# --- fused-heterocycle parent, ring-bearing compound substituent (Tasks 2/4) --
# The pantoprazole class: a benzimidazole parent carrying an S(=O)-rooted
# ring-bearing compound substituent at C2. All three ABSTAIN at HEAD.
FUSED_HETEROCYCLE_WITNESSES = [
    # pantoprazole (trifluoromethoxy variant) -- the charter anchor.
    "COc1ccnc(CS(=O)c2nc3ccc(OC(F)(F)F)cc3[nH]2)c1OC",
    # pantoprazole (difluoromethoxy variant, the C4-trace shape).
    "COc1ccnc(CS(=O)c2nc3ccc(OC(F)F)cc3[nH]2)c1OC",
    # a benzimidazole-sulfinyl-benzyl (trifluoromethyl on the benzo ring).
    "O=S(Cc1ccccc1)c1nc2ccc(C(F)(F)F)cc2[nH]1",
]

WITNESSES = list(FUSED_HETEROCYCLE_WITNESSES)

# --- complex-ring (spiro / von-Baeyer) parent enumerator (Task 3) -------------
# The composer's complex-ring substituent enumerator
# (assembly/composer.py::_enrich_complex_ring_with_subs) is the spiro / von-Baeyer
# sibling of the fused enumerator. A ring-bearing compound substituent being
# DROPPED there is off-path in the sampled ChEBI corpus (0 in 520+ molecules;
# parent selection routes the ring substituent through the fused path instead --
# a project rule), so Task 3 is proven at the WIRING level: the composer must thread
# the best-effort context into ``name_substituent`` as ``allow_mancude=True``
# exactly as the fused enumerator now does. At HEAD it always passed ``False``.
# 9-methylspiro[5.5]undecane routes its substituent enumeration through the
# enricher, so it exercises the wiring deterministically.
COMPOSER_WIRING_SMILES = "CC1CCC2(CC1)CCCCC2"


def _cn_rt(smiles):
    """Best-effort name + OPSIN full-InChIKey round-trip, in a FRESH process.

    Returns ``{"name": str|None, "tier": str, "rt": bool|None}``. A subprocess
    per call is deliberate: the fragment memo cache is per-process and warm-cache
    state has repeatedly produced false greens in this project (CLAUDE.md).
    """
    out = subprocess.run(
        [sys.executable, __file__, smiles],
        capture_output=True, text=True, timeout=180,
    )
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"name"' in line:
            return json.loads(line)
    raise AssertionError(
        f"no result JSON for {smiles!r}\n"
        f"stdout:\n{out.stdout[-2000:]}\nstderr:\n{out.stderr[-2000:]}"
    )


@pytest.mark.parametrize("smiles", FUSED_HETEROCYCLE_WITNESSES)
def test_fused_heterocycle_ring_bearing_substituent_names_and_rt(smiles):
    """A ring-bearing compound substituent on a fused-heterocycle parent must be
    named (not dropped) and the whole name must round-trip (0-wrong)."""
    res = _cn_rt(smiles)
    name = res.get("name")
    assert name and "unknown" not in name, (
        f"abstained (ring-bearing compound substituent silently dropped): "
        f"{smiles} -> {res}")
    assert res.get("rt") is True, (
        f"emitted a name that does NOT round-trip to the input InChIKey: "
        f"{smiles} -> {name!r} (rt={res.get('rt')})")


def test_composer_threads_best_effort_allow_mancude():
    """Task 3: the complex-ring (spiro/VB) substituent enumerator must thread the
    best-effort context into ``name_substituent`` (``allow_mancude=True``). RED at
    HEAD, where the composer always passed ``allow_mancude=False``."""
    out = subprocess.run(
        [sys.executable, __file__, "--wiring-probe", COMPOSER_WIRING_SMILES],
        capture_output=True, text=True, timeout=180,
    )
    res = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "enrich_calls" in line:
            res = json.loads(line)
    assert res is not None, (
        f"no wiring-probe result\nstdout:\n{out.stdout[-1500:]}\n"
        f"stderr:\n{out.stderr[-1500:]}")
    assert res["enrich_calls"] > 0, (
        "the composer enricher (_enrich_complex_ring_with_subs) was not reached "
        f"for {COMPOSER_WIRING_SMILES}: {res}")
    assert res["any_allow_mancude_true"], (
        "composer did NOT thread best_effort_ctx -> allow_mancude into "
        f"name_substituent (v36 Task-3 wiring missing): {res}")


def _make_sub(locant, attach_mol_idx, frag_atoms):
    from orthonym.assembly.substituent_enumerator import SubstituentInfo
    return SubstituentInfo(None, locant, attach_mol_idx, list(frag_atoms))


def test_composer_fail_closed_on_unnameable_ring_bearing_substituent():
    """Task 4: a RING-BEARING compound substituent on a complex-ring parent that
    cannot render must ABORT the whole name (return None), not silently drop the
    substituent and leak an atom-incomplete partial (a wrong molecule that only
    the downstream RT/SELF-01 gate would catch). RED at HEAD, where the composer
    ``continue``d and returned the bare parent."""
    from rdkit import Chem
    import orthonym.assembly.composer as comp
    import orthonym.assembly.substituent_enumerator as se
    from orthonym.metrics.provenance import best_effort_ctx
    mol = Chem.MolFromSmiles("c1ccccc1C1CCCCC1")   # phenylcyclohexane
    ring_atoms = {6, 7, 8, 9, 10, 11}              # the cyclohexane parent
    a2l = {6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6}
    ring_sub = _make_sub(1, 6, [0, 1, 2, 3, 4, 5])  # the phenyl (ring-bearing)
    _od, _on = se.discover_substituents, se.name_substituent
    se.discover_substituents = lambda *a, **k: [ring_sub]
    se.name_substituent = lambda *a, **k: None      # unnameable
    tok = best_effort_ctx.set(True)
    try:
        out = comp._enrich_complex_ring_with_subs(
            mol, "cyclohexane", ring_atoms, a2l)
    finally:
        best_effort_ctx.reset(tok)
        se.discover_substituents, se.name_substituent = _od, _on
    assert out is None, (
        "expected fail-closed (None) on an unnameable ring-bearing substituent, "
        f"got {out!r} (an atom-incomplete partial)")


def test_composer_keeps_skip_for_unnameable_nonring_substituent():
    """Task 4 scoping (a project rule): a NON-ring unnameable / mis-discovered drop
    keeps the prior skip (returns the bare parent) -- only ring-bearing,
    atom-significant drops abort. Green at HEAD and with the fix."""
    from rdkit import Chem
    import orthonym.assembly.composer as comp
    import orthonym.assembly.substituent_enumerator as se
    from orthonym.metrics.provenance import best_effort_ctx
    mol = Chem.MolFromSmiles("CC1CCCCC1")           # methylcyclohexane
    ring_atoms = {1, 2, 3, 4, 5, 6}
    a2l = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    nonring_sub = _make_sub(1, 1, [0])              # the methyl (non-ring)
    _od, _on = se.discover_substituents, se.name_substituent
    se.discover_substituents = lambda *a, **k: [nonring_sub]
    se.name_substituent = lambda *a, **k: None
    tok = best_effort_ctx.set(True)
    try:
        out = comp._enrich_complex_ring_with_subs(
            mol, "cyclohexane", ring_atoms, a2l)
    finally:
        best_effort_ctx.reset(tok)
        se.discover_substituents, se.name_substituent = _od, _on
    assert out == "cyclohexane", (
        "a non-ring drop must keep the prior skip (return the bare parent), "
        f"got {out!r}")


def _composer_wiring_probe(smiles):
    """Report the ``allow_mancude`` values ``_enrich_complex_ring_with_subs``
    passes into ``name_substituent`` while naming ``smiles`` at the best-effort
    tier. The Task-3 wiring makes these True; at HEAD they are always False."""
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cn-wiring"):
        import orthonym.assembly.composer as comp
        import orthonym.assembly.substituent_enumerator as se
        from orthonym.namer import Orthonym
        state = {"in": False}
        seen = []
        _oe = comp._enrich_complex_ring_with_subs
        def _we(*a, **k):
            state["in"] = True
            try:
                return _oe(*a, **k)
            finally:
                state["in"] = False
        _ons = se.name_substituent
        def _wns(mol, frag, attach, *a, allow_mancude=False, **k):
            if state["in"]:
                seen.append(bool(allow_mancude))
            return _ons(mol, frag, attach, *a, allow_mancude=allow_mancude, **k)
        comp._enrich_complex_ring_with_subs = _we
        se.name_substituent = _wns
        try:
            Orthonym(
                style="pin", general_fallback=True,
                general_fallback_unverified=True,
                allow_aromatic_general=True).name_tiered(smiles)
        finally:
            comp._enrich_complex_ring_with_subs = _oe
            se.name_substituent = _ons
        return {"enrich_calls": len(seen),
                "any_allow_mancude_true": any(seen)}


def _name_one(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cn-test"):
        from orthonym.namer import Orthonym
        from orthonym.validation import opsin_roundtrip_check
        namer = Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True)
        res = namer.name_tiered(smiles)
        name = res.get("name")
        rt = None
        if name and "unknown" not in name:
            rt = bool(opsin_roundtrip_check(smiles, name)["passed"])
        return {"name": name, "tier": res.get("tier"), "rt": rt}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--wiring-probe":
        print(json.dumps(_composer_wiring_probe(sys.argv[2])))
    else:
        print(json.dumps(_name_one(sys.argv[1])))
