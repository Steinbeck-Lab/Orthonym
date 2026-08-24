"""v36 Milestone-C Wave B — ring-breadth core-namer builds (research items 4 + 5).

BUILD 1 — parent-selection robustness (C3 Site-3 + C1/C2/C6 Pattern-D).
    A small FG-bearing ring chosen as parent while SPIRO-joined to a larger ring
    system used to be named alone (the added-indicated-H cyclic-oxo preempt
    ``rules/partial_saturation.py::name_cyclic_oxo_compound`` names only the
    carbonyl's FUSED system and silently drops the spiro-joined partner), leaving
    an atom-incomplete partial that SELF-01 then suppressed -> abstain.

    ROOT CAUSE (spy: `` Site-3, this session's
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
        ``O=C1CCC2(Oc3cccc4cccc(c34)O2)c2cccc(O)c21`` (C3-spy w16)

BUILD 2 — recursive-fragment retained-table reach (C4 glycan aglycone + C3 Site-2)
    (added below once its spy pins the exact wall.)

Fresh process per witness (warm-cache hazard, the contributor guide).  Run ONLY this file:
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
# cyclohexane.  The KIH cyclic-oxo preempt used to name only the phthalazinone
# ("3,4-dihydrophthalazin-1(2H)-one") and drop the cyclohexane -> abstain.
SPIRO_JOINED_KETONE_CONVERT = "O=C1NNC2(CCCCC2)c2ccccc21"

# C3-spy w16: a tetralone (ring ketone) spiro-joined to a naphtho-dioxole.  The
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
    molecule).  It either round-trips or abstains cleanly (0-wrong)."""
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
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
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
    print(json.dumps(_name_one(sys.argv[1])))
