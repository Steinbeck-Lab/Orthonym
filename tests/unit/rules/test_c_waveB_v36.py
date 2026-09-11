""" Milestone-C Wave B — ring-breadth core-namer builds (research items 4 + 5).

BUILD 1 — parent-selection robustness (C3 Site-3 + C1/C2/C6 Pattern-D).
    A small FG-bearing ring chosen as parent while SPIRO-joined to a larger ring
    system used to be named alone (the added-indicated-H cyclic-oxo preempt
    ``rules/partial_saturation.py::name_cyclic_oxo_compound`` names only the
    carbonyl's FUSED system and silently drops the spiro-joined partner), leaving
    an atom-incomplete partial that then suppressed -> abstain.

    ROOT CAUSE (trace: `internal notes` Site-3, this session's
    trace): the preempt fires BEFORE the whole-molecule complex_ring/spiro path.
    FIX: ``name_cyclic_oxo_compound`` fails closed (returns None) when the
    carbonyl's fused ring system is spiro-/bridge-joined to further ring atoms it
    cannot express, so the molecule falls through to the complex_ring/spiro namer
    which names the WHOLE joined system (RT-gated; emit-or-abstain, never a silent
    atom-drop).

    Convertible witness (was abstain -> now a full RT-matching name):
        ``O=C1NNC2(CCCCC2)c2ccccc21`` (a phthalazinone spiro-joined to cyclohexane)
    Diagnosed-abstain witness (routed to complex_ring, whose spiro FG-enricher has
    a separate numbering bug -> OPSIN-unparseable -> RT-gate abstains, 0-wrong):
        ``O=C1CCC2(Oc3cccc4cccc(c34)O2)c2cccc(O)c21`` (C3-trace w16)

BUILD 2 — recursive-fragment retained-table reach (C4 glycan aglycone + C3 Site-2)
    a trace RESOLUTION (this session): the charter's stated wall is REFUTED / already
    solved, so no code change lands (a project rule — the named site is off-path).
      * ``assembly/fragment_naming.py::name_fragment_recursively`` ALREADY reaches
        the steroid/NP retained table: it calls ``namer.name_compound`` (line 543),
        which owns ``rules/natural_products.py``. Proven: it names estradiol ->
        ``(8R,9S,13S,14S,17S)-estra-1,3,5(10)-triene-3,17-diol``.
      * The 25-atom cap at ``substituent_enumerator.py:4283`` is OFF-PATH for this
        population: a >25-atom ring substituent (the steroid in cholesteryl
        glucoside) is named by a DIFFERENT substituent path not subject to it.
        Lifting the cap 25->999 converts 0/3 glycan witnesses and 0 net on a
        40-molecule large-abstention sample (the 2 that named there also name at
        cap=25). Lifting it only adds garbage-name + giant-molecule-hang exposure
        for zero yield, so it is NOT lifted.
      * Steroid glycosides whose aglycone IS retained-table-nameable ALREADY name
        and round-trip today (estradiol 3-glucoside, cholesteryl glucoside).
      * NAMED BLOCKER for the g1/g2/g3 glycan witnesses: the AGLYCONE itself is
        unnameable (``name_compound`` -> ``unknown``) — a cardenolide / sapogenin /
        macrolide fused steroid+lactone skeleton absent from the retained table and
        unbuildable by the systematic ring namer (its bare ring core is itself
        MULTI-FRAGMENT). That is C1/C2 ring-topology + NP-table COVERAGE work,
        out of Build-2's scope. The tests below are regression guards for the
        already-working class + the reach wiring, and a 0-wrong guard on the
        witnesses.

Fresh process per witness (warm-cache hazard, the contributor guide). Run ONLY this file:
    ``.venv/bin/python -m pytest tests/unit/rules/test_c_waveB_v36.py -q``
(the whole suite deadlocks on an OPSIN pipe).
"""
import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# BUILD 1 — spiro-joined ring-ketone parent selection
# ---------------------------------------------------------------------------

# A phthalazinone (fused N-N-C=O ring) spiro-joined at its sp3 carbon to a
# cyclohexane. The KIH cyclic-oxo preempt used to name only the phthalazinone
# ("3,4-dihydrophthalazin-1(2H)-one") and drop the cyclohexane -> abstain.
SPIRO_JOINED_KETONE_CONVERT = "O=C1NNC2(CCCCC2)c2ccccc21"

# C3-trace w16: a tetralone (ring ketone) spiro-joined to a naphtho-dioxole. The
# complex_ring namer builds a full-system name but its spiro FG-enricher mis-numbers
# the second component (separate numbering bug), so this stays a clean abstain.
W16_DIAGNOSED_ABSTAIN = "O=C1CCC2(Oc3cccc4cccc(c34)O2)c2cccc(O)c21"

# Plain KIH canaries — NO spiro/bridge partner: the guard must NOT fire, these
# keep naming byte-identically.
KIH_PLAIN_CANARIES = {
    "O=C1CCCc2ccccc21": "3,4-dihydronaphthalen-1(2H)-one",
    "O=C1NNCc2ccccc21": "3,4-dihydrophthalazin-1(2H)-one",
    "O=c1ccc2ccccc2[nH]1": "quinolin-2(1H)-one",
    "O=C1C=Cc2ccccc21": "1H-inden-1-one",
}


def _cn_rt(smiles):
    """Best-effort name + OPSIN full-InChIKey round-trip, in a FRESH process."""
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
        f"stdout:\n{out.stdout[-2000:]}\nstderr:\n{out.stderr[-2000:]}")


def test_spiro_joined_ring_ketone_names_and_round_trips():
    """A ring ketone spiro-joined to a larger ring system must name the WHOLE
    joined system (not drop the spiro partner) and round-trip (0-wrong).
    RED at HEAD (abstains: the KIH preempt named the phthalazinone alone)."""
    res = _cn_rt(SPIRO_JOINED_KETONE_CONVERT)
    name = res.get("name")
    assert name and "unknown" not in name, (
        f"abstained (spiro partner silently dropped): "
        f"{SPIRO_JOINED_KETONE_CONVERT} -> {res}")
    assert res.get("rt") is True, (
        f"emitted a name that does NOT round-trip to the input InChIKey: "
        f"{name!r} (rt={res.get('rt')})")


def test_w16_spiro_ketone_never_emits_wrong_partial():
    """w16 must NEVER ship the atom-incomplete tetralone partial (a wrong
    molecule). It either round-trips or abstains cleanly (0-wrong)."""
    res = _cn_rt(W16_DIAGNOSED_ABSTAIN)
    name = res.get("name") or ""
    if name and "unknown" not in name:
        assert res.get("rt") is True, (
            f"w16 emitted a non-abstain name that does NOT round-trip "
            f"(wrong molecule): {name!r}")
        # and it must NOT be the dropped-partner tetralone partial:
        assert "naphthalen-1(2h)-one" not in name.lower(), (
            f"w16 shipped the atom-incomplete tetralone partial: {name!r}")


@pytest.mark.parametrize("smiles,expected", list(KIH_PLAIN_CANARIES.items()))
def test_plain_kih_canaries_unchanged(smiles, expected):
    """A cyclic-oxo compound with NO spiro/bridge partner keeps its KIH name
    (the guard must be scoped to spiro/bridge-joined systems only)."""
    from orthonym.rules.partial_saturation import name_cyclic_oxo_compound
    from rdkit import Chem
    got = name_cyclic_oxo_compound(Chem.MolFromSmiles(smiles))
    assert got == expected, f"{smiles}: {got!r} != {expected!r}"


def test_kih_fail_closed_on_spiro_joined_carbonyl_system():
    """``name_cyclic_oxo_compound`` must return None when the carbonyl's fused
    ring system is spiro-/bridge-joined to further ring atoms (it can only name
    the fused component and would silently drop the joined partner).
    RED at HEAD (returns the dropped-partner partial)."""
    from orthonym.rules.partial_saturation import name_cyclic_oxo_compound
    from rdkit import Chem
    for smi in (SPIRO_JOINED_KETONE_CONVERT, W16_DIAGNOSED_ABSTAIN):
        got = name_cyclic_oxo_compound(Chem.MolFromSmiles(smi))
        assert got is None, (
            f"expected fail-closed (None) on spiro-joined carbonyl system, "
            f"got {got!r} for {smi} (an atom-incomplete partial)")


# ---------------------------------------------------------------------------
# BUILD 2 — recursive-fragment retained-table reach (findings + regression guards)
# ---------------------------------------------------------------------------

# Steroid glycoside whose aglycone IS retained-table-nameable — already names +
# round-trips today (the class Build-2's charter thought was blocked by the cap).
STEROID_GLYCOSIDE_WORKS = "C[C@]12CC[C@H]3[C@@H](CCc4cc(OC5OC(CO)C(O)C(O)C5O)ccc34)[C@@H]1CC[C@@H]2O"

# TRUE-glycan witnesses (V36-a trace-C4.md g1/g2/g3): abstain because the AGLYCONE is
# an uncovered fused steroid+lactone/macrolide skeleton (ring-topology COVERAGE gap,
# out of Build-2's scope) — NOT the recursive-reach / 25-atom-cap wall.
GLYCAN_COVERAGE_GAP_WITNESSES = [
    "CC1CCC2(OC1)OC1CC3C4CCC5CC(OC6OC(CO)C(O)C(OC7OC(CO)C(O)C(O)C7O)C6O)CCC5(C)C4CCC3(C)C1C2C",
    "CCC(C)/C=C/CCCC(C)C(O)C/C=C/C=C/C(=O)OC1C(OC2OC(CO)C(O)C(O)C2O)C(CO)OC2(OCc3cc(O)cc(O)c32)C1O",
    "CC(=O)OC(CC1C=C(C)C(=O)O1)C(C)C1CCC2C3(C)CCC(OC4OC(CO)C(O)C(O)C4OC4OC(C)C(O)C(O)C4O)C(C)(C)C3CCC2(C)C12COC(=O)C2",
]


def test_recursive_fragment_naming_reaches_steroid_retained_table():
    """Build-2 a trace: ``name_fragment_recursively`` ALREADY reaches the steroid/NP
    retained table (it delegates to ``name_compound``). Regression guard against
    breaking that wiring — if this fails, the retained table is unreachable from
    the recursive substituent path (the exact gap the charter feared)."""
    out = subprocess.run(
        [sys.executable, __file__, "--reach",
         "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O"],
        capture_output=True, text=True, timeout=180)
    res = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "frag_name" in line:
            res = json.loads(line)
    assert res is not None, f"no reach result\nstdout:\n{out.stdout[-1500:]}"
    assert res["frag_name"] and "estra" in res["frag_name"], (
        "name_fragment_recursively did NOT reach the steroid retained table "
        f"for estradiol: {res}")


def test_steroid_glycoside_with_nameable_aglycone_round_trips():
    """Regression guard: a steroid glycoside whose aglycone is retained-table-
    nameable names + round-trips today (the >25-atom-substituent case the 25-atom
    cap was thought to block — it is off-path)."""
    res = _cn_rt(STEROID_GLYCOSIDE_WORKS)
    name = res.get("name")
    assert name and "unknown" not in name, (
        f"steroid glycoside abstained unexpectedly: {res}")
    assert res.get("rt") is True, (
        f"steroid glycoside name does not round-trip: {name!r}")


@pytest.mark.parametrize("smiles", GLYCAN_COVERAGE_GAP_WITNESSES)
def test_glycan_coverage_gap_witnesses_are_zero_wrong(smiles):
    """The g1/g2/g3 glycan witnesses abstain because their aglycone is an uncovered
    ring skeleton (documented blocker).  0-wrong guard: each must ABSTAIN or emit a
    name that round-trips — NEVER a wrong-molecule name."""
    res = _cn_rt(smiles)
    name = res.get("name") or ""
    if name and "unknown" not in name:
        assert res.get("rt") is True, (
            f"glycan witness emitted a NON-round-tripping name (wrong molecule): "
            f"{name!r} for {smiles}")


# ---------------------------------------------------------------------------
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
def _reach_one(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cwaveb-reach"):
        from orthonym.assembly.fragment_naming import (
            name_fragment_recursively, start_naming_session, end_naming_session)
        start_naming_session()
        try:
            frag = name_fragment_recursively(smiles)
        finally:
            end_naming_session()
        return {"frag_name": frag}


def _name_one(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cwaveb-test"):
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
    if len(sys.argv) >= 3 and sys.argv[1] == "--reach":
        print(json.dumps(_reach_one(sys.argv[2])))
    else:
        print(json.dumps(_name_one(sys.argv[1])))
