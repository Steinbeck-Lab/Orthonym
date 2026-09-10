""" Milestone-C Wave C — ring-CONSTRUCTION builds (trace: V36-a trace-C1C2C6.md).

BUILD 1 — Pattern-A stereo/numbering + invariant-18 RT-gated re-anchor
    (mixed-spiro-fused, the leak sub-mode).
    A spiro-of-fused-component parent (``name_spiro_vonbaeyer`` /
    ``name_mixed_spiro_fused``) builds the correct FLAT descriptor, but the
    ``combined_locants`` map it hands to the complex_ring stereo injector is
    numbered by a DIFFERENT (and collision-bearing) system than the printed
    spiro descriptor. The injected stereo descriptor lands on the WRONG locant
    (``(3R)-spiro[1,3-dihydro-2-benzofuran-1,1'-2,3-dihydro-1H-indene]`` — locant
    3 is the O-CH2, not the spiro stereocentre), so the FULL name is
    OPSIN-unparseable. It used to SHIP anyway via the stereo carve-out
    (constitution verified on the stereo-stripped parse) — a live 0-wrong leak:
    an emitted name that does not round-trip.

    ROOT CAUSE (this session's trace): ``_name_spiro_vonbaeyer_core`` /
    ``_name_fused_component`` produce a non-bijective atom->locant map for these
    fused components; ``_canonical_spiro_locant`` renumbers only the spiro atom
    for the NAME, leaving the map inconsistent with the descriptor.

    FIX (a project rule — offer many, keep the RT-passing one): at the complex_ring
    stereo-injection site the injected candidate is RT-gated; when it does NOT
    full-round-trip, the atom->locant map is RE-ANCHORED to OPSIN's OWN numbering
    of the built parent name (``-o extendedsmi`` $_AV locants, mapped back onto
    the input graph), stereo is re-injected on that authoritative numbering, and
    the re-anchored candidate is kept ONLY if it full-round-trips. Otherwise the
    original candidate is returned unchanged (current behaviour preserved — the
    change is purely additive, so a legitimate OPSIN-can't-parse-the-stereo-layer
    carve-out PIN is untouched because its re-anchored form also fails full-RT).

BUILD 2 — general N-component ortho/ortho-peri-fused mancude construction
    (pure-fused, trace Pattern C). ``name_ortho_fused_bicyclic`` is 2-component
    only; a genuine 3+-component novel mancude skeleton (e.g. the target that
    OPSIN names ``pyrimido[4,5-b]quinoline``) had NO construction path -> abstain.
    See the BUILD 2 section below for the reached component count + named blocker.

Fresh process per witness (warm-cache hazard, CLAUDE.md). Run ONLY this file:
    ``.venv/bin/python -m pytest tests/unit/rules/test_c_waveC_v36.py -q``
(the whole suite deadlocks on an OPSIN pipe).
"""
import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# BUILD 1 — Pattern-A stereo/numbering re-anchor (the 7 unique leak witnesses)
# ---------------------------------------------------------------------------

# The 9 mixed-spiro-fused witnesses that BUILT a spiro name (7 unique) from the
# V36-a trace-C1C2C6 classified set. Every one emitted an OPSIN-UNPARSEABLE name at
# with a wrong-locant stereo descriptor and shipped via the stereo carve-out.
PATTERN_A_WITNESSES = [
    "c1ccc2c(c1)CC[C@@]21OCc2ccccc21",
    "c1ccc2c(c1)CC[C@]1(C2)OCc2ccccc21",
    "c1ccc2c(c1)CO[C@@]21Oc2cccc3cccc1c23",
    "c1ccc2c(c1)CN[C@@]21Oc2cccc3cccc1c23",
    "c1ccc2c(c1)CN[C@]21Oc2cccc3cccc1c23",
    "C1=CC[C@@H]2[C@@H](C1)CC=CC21Oc2cccc3cccc(c23)O1",
    "C1CC[C@@H]2C[C@]3(CC[C@H]2C1)CO3",
]


def _cn_rt(smiles):
    """Best-effort name + OPSIN full-InChI round-trip, in a FRESH process."""
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


@pytest.mark.parametrize("smiles", PATTERN_A_WITNESSES)
def test_pattern_a_spiro_stereo_names_and_round_trips(smiles):
    """A mixed-spiro-fused parent with a stereocentre must emit a name whose
    stereo descriptor is on the RIGHT locant, so the FULL name round-trips.
    RED at HEAD (ships an OPSIN-unparseable wrong-locant stereo descriptor)."""
    res = _cn_rt(smiles)
    name = res.get("name")
    assert name and "unknown" not in name, (
        f"abstained unexpectedly: {smiles} -> {res}")
    assert res.get("rt") is True, (
        f"emitted a name that does NOT full-round-trip (wrong-locant stereo "
        f"descriptor): {name!r} (rt={res.get('rt')}) for {smiles}")


def test_pattern_a_no_wrong_locant_stereo_leak():
    """The canonical witness must cite the spiro stereocentre at its spiro locant
    (1 / 1'), never at the O-CH2 locant 3. RED at HEAD ((3R)-...)."""
    res = _cn_rt("c1ccc2c(c1)CC[C@@]21OCc2ccccc21")
    name = res.get("name") or ""
    assert res.get("rt") is True and name, f"expected a round-tripping name: {res}"
    assert not name.startswith("(3R)"), (
        f"still shipping the wrong-locant stereo descriptor: {name!r}")


# ---------------------------------------------------------------------------
# CANARIES — shared ring dispatch (spiro/tier_a_ring/fused_rings/composer) must
# stay byte-identical. Pinned pre-change (HEAD 3fa69a42). The stereo-bearing
# complex_ring names (steroid, von-Baeyer androstanedione) exercise the exact
# injection path BUILD 1 modifies and MUST be unchanged (RT-gate returns the
# original candidate untouched because it already full-round-trips).
# ---------------------------------------------------------------------------
CANARIES = {
    "c1ccc2ccccc2c1": "naphthalene",
    "c1ccc2cc3ccccc3cc2c1": "anthracene",
    "c1ccc2ncccc2c1": "quinoline",
    "c1ccc2[nH]ccc2c1": "1H-indole",
    "c1ccc2nc3ccccc3cc2c1": "acridine",
    "c1ccc2c(c1)ncc1ccccc21": "phenanthridine",
    "c1ccc2c(c1)[nH]c1ccccc12": "9H-carbazole",
    "c1ccc2c(c1)ccc1ccccc12": "phenanthrene",
    "C1CCC2(CC1)CCCC2": "spiro[4.5]decane",
    "C1C[C@]2(CCCC2)CC1": "spiro[4.4]nonane",
    "C1C2CC3CC1CC(C2)C3": "adamantane",
    "C[C@]12CC[C@H]3[C@@H](CC[C@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O":
        "17β-hydroxy-5α-androstan-3-one",
    "O=C1CC[C@]2(C)[C@H]3CC[C@@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C":
        "(5R,8S,9S,10S,13S,14R)-14-methylandrostane-3,17-dione",
}


@pytest.mark.parametrize("smiles,expected", list(CANARIES.items()))
def test_shared_ring_dispatch_canaries_unchanged(smiles, expected):
    """Every shared-ring-dispatch canary keeps its exact HEAD name (and RTs)."""
    res = _cn_rt(smiles)
    assert res.get("name") == expected, (
        f"canary REGRESSED: {smiles} -> {res.get('name')!r} != {expected!r}")
    assert res.get("rt") is True, f"canary no longer round-trips: {res}"


# ---------------------------------------------------------------------------
# BUILD 2 — general N-component ortho/ortho-peri-fused mancude construction
# ---------------------------------------------------------------------------

# 3-component novel fused mancude systems whose senior base is a 2-ring retained
# component (pyrimido[4,5-b]quinoline class). RED at HEAD (name_ortho_fused_
# bicyclic is 2-component-only; _try_polycomponent_fusion_name builds only
# star-of-monocycles bases) -> abstained / von-Baeyer.
BUILD2_3COMP_WITNESSES = [
    "c1ccc2nc3ncncc3cc2c1",   # -> pyrimido[4,5-b]quinoline (a trace-verified)
    "c1ccc2nc3ncncc3nc2c1",   # a 3-ring diaza/tetraaza fused mancude
]


@pytest.mark.parametrize("smiles", BUILD2_3COMP_WITNESSES)
def test_three_component_fused_mancude_names_and_round_trips(smiles):
    """A genuine 3-component ortho-fused mancude system must build a fusion name
    that round-trips (0-wrong; the generate-and-test path is OPSIN-RT-gated)."""
    res = _cn_rt(smiles)
    name = res.get("name")
    assert name and "unknown" not in name, f"abstained: {smiles} -> {res}"
    assert res.get("rt") is True, (
        f"emitted a non-round-tripping name: {name!r} for {smiles}")


def test_pyrimido_quinoline_exact_pin():
    """The a trace-verified 3-component target must build exactly the OPSIN-feasible
    fusion PIN ``pyrimido[4,5-b]quinoline``. RED at HEAD (abstained)."""
    res = _cn_rt("c1ccc2nc3ncncc3cc2c1")
    assert res.get("name") == "pyrimido[4,5-b]quinoline", (
        f"expected pyrimido[4,5-b]quinoline, got {res.get('name')!r}")
    assert res.get("rt") is True


def test_four_component_fused_is_zero_wrong():
    """0-wrong guard: a 4-component fused mancude either round-trips or abstains,
    never a wrong molecule (it degrades to a von-Baeyer name today)."""
    res = _cn_rt("c1ccc2nc3ccc4ncccc4c3nc2c1")
    name = res.get("name") or ""
    if name and "unknown" not in name:
        assert res.get("rt") is True, (
            f"4-component fused emitted a non-round-tripping name: {name!r}")


@pytest.mark.xfail(strict=True, reason=(
    "v37 CT.4 built the 4-component CARBOCYCLIC-child case "
    "(naphtho[2,3-g]quinoxaline). This witness is the HETEROCYCLIC-bicyclic-prefix "
    "case (quinoxaline + quinoline attached) -- a NAMED FOLLOW-ON not built this "
    "session: it needs a nameable heterocyclic bicyclic PREFIX + descriptor "
    "construction, not the naphtho carbocyclic path. The molecule still names+RTs "
    "via the von-Baeyer no-abstain fallback (0-wrong), just not as a fusion PIN."))
def test_four_component_fused_gets_fusion_name_NAMED_BLOCKER():
    res = _cn_rt("c1ccc2nc3ccc4ncccc4c3nc2c1")
    name = res.get("name") or ""
    assert res.get("rt") is True
    # DESIRED (not yet built): a fusion name, not a von-Baeyer 'tetracyclo[...]'.
    assert "cyclo[" not in name and "[" in name, (
        f"got a von-Baeyer name, not a fusion PIN: {name!r}")


# ---------------------------------------------------------------------------
# subprocess entrypoint (fresh process per witness)
# ---------------------------------------------------------------------------
def _name_one(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="v36-cwaveC-test"):
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
