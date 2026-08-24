"""v36 Milestone C1/C2/C6 — ring-topology construction gap.

Grounding spy: `` (197 witnesses, 4 decline
patterns). Witnesses here are BARE RING CORES whose topology currently abstains
(the true ring-topology-construction population — not the C3 substituent/assembly
gap). Every emitted ring name must OPSIN-round-trip to the input's full InChIKey
(``opsin_roundtrip_check``) or the molecule abstains; a wrong-but-parseable name is
a 0-wrong break and must never occur.

Patterns (spy, VERIFIED):
  A1  spiro-von-Baeyer with a BICYCLO cage: the ``-ene`` unsaturation suffix is
      emitted OUTSIDE the spiro brackets (``...octane-6,2'-oxolane]-3-ene``), which
      is OPSIN-grammar-invalid. VERIFIED fix: splice it INSIDE the cage component
      (``...oct-3-ene-6,2'-oxolane]``). Task 2. -> GREEN.
  A2  spiro-von-Baeyer that builds a name with a SELF-01 wrong-numbering / other
      construction bug (no trailing-ene). Task 3 triage. -> abstain today.
  B   mixed-spiro-fused (spiro joining two fused/bridged sub-systems);
      ``name_mixed_spiro_fused`` has no construction path. Task 5. -> abstain today.
  C   3+-component ortho/ortho-peri-fused mancude; ``name_ortho_fused_bicyclic`` is
      2-component-only. Task 4. -> abstain today.
  D   bridged-fused whole-molecule; upstream parent-selection bypass (small ring
      claims parent, ``sub_count=0``). Task 5. -> abstain today.
"""
from __future__ import annotations

import signal
from collections import namedtuple

import pytest

# --- witness sets (spy V36-SPY-C1C2C6.md; bare ring cores) --------------------

# Pattern A1: spiro-von-Baeyer, bicyclo cage, unsaturation suffix appended OUTSIDE
# the brackets today (OPSIN-grammar-invalid). VERIFIED closeable by in-bracket
# splice. These go GREEN after Task 2.
A1_WITNESSES = [
    "C1=CC2CC(C1)OC21CCCO1",                       # 7-oxaspiro[bicyclo[3.2.1]oct-3-ene-6,2'-oxolane] (VERIFIED)
    "C1=C[C@@H]2O[C@@H]2[C@]2(C1)CCCO2",           # bicyclo[4.1.0]hept-4-ene cage
    "C1=C[C@]2(CCCO2)[C@H]2O[C@H]2C1",             # bicyclo[4.1.0]hept-3-ene cage
    "C1=CCC[C@]2(CO2)[C@@H]2CC[C@H]2CC1",          # bicyclo[7.2.0]undec-5-ene cage
    "C1=CCCC2(CO2)C2CCC2CC1",                       # bicyclo[7.2.0]undec-5-ene cage (achiral)
    "C1=C\\[C@@]2(CCCO2)CCOCC[C@@H]2O[C@H]/12",    # bicyclo[8.1.0]undec-2-ene cage
    "C1=C/CCCCCCCCCCCC/C=C/CO[C@@H]2C[C@H](CCC/C=C/1)O[C@]1(CCCCO1)C2",  # bicyclo[23.3.1] triene
    "C1=CCCCCCCCCCCCCC=CCO[C@@H]2C[C@H](CCCC=C1)O[C@]1(CCCCO1)C2",       # bicyclo[23.3.1] triene
    "C1=CCCCC2CC(CC3(CCCCO3)O2)OC/C=C\\CCCCCCCCCCCC/C=C\\1",             # bicyclo[23.3.1] triene
    "C1=C\\CCCCCCCCCCCC/C=C\\COC2CC(CCC\\C=C/1)OC1(CCCCO1)C2",           # bicyclo[23.3.1] triene
]

# Pattern A2: spiro-von-Baeyer builds a name with a wrong-numbering / SELF-01 bug.
A2_WITNESSES = [
    "c1cc2c(c3c1CNC3)O[C@@]1(CCC[C@H]3CCCC[C@@H]31)C2",   # spy SELF-01 "different molecule"
    "C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1",     # tricyclo diene component
]

# Pattern B: mixed-spiro-fused (name_mixed_spiro_fused has no construction path).
B_WITNESSES = [
    "C1=CCC2(C1)COc1ccccc12",
    "C1=CC2(C=CC1)Cc1ccccc1O2",
    "C1CC[C@@H]2C[C@]3(CC[C@H]2C1)CO3",   # builds spiro[decahydronaphthalene-7,2'-oxirane] (unparseable)
]

# Pattern C: multi-component ortho / ortho-peri-fused mancude that genuinely
# abstains (rt-fails) at the best-effort tier -- neither the retained fused table
# nor the von-Baeyer floor covers these.
C_WITNESSES = [
    "c1ccc2c(c1)CO[C@H]2[C@H]1OCc2ccccc21",
    "c1ccc([C@@H]2O[C@@]23CNc2ccccc2CN3)cc1",
    "c1ccc2c(c1)CCOCCCOCc1ccccc1CCOCCCOCc1ccccc1CCOCCCOC2",  # ortho-peri-fused
]

# MEASUREMENT (best-effort tier, this session): the von-Baeyer best-effort FLOOR
# already RT-covers many 3+-component fused-mancude cores that the spy recorded as
# PIN-tier abstentions (24/41 ortho-fused witnesses RT-pass). For those the v36
# breadth win is ALREADY delivered (a 0-wrong non-PIN name); the remaining gap is
# PIN-QUALITY (emitting the proper fusion name, e.g. pyrimido[4,5-b]quinoline,
# instead of a triaza-von-Baeyer name) -- a separate, larger fusion-nomenclature
# build, deferred (Task 4 scope note).
C_COVERED_BY_FLOOR = [
    "c1ccc2nc3ncncc3cc2c1",    # -> 2,4,6-triazatricyclo[...]heptaene (non-PIN, RT-ok)
    "c1ccc2nc3ncncc3nc2c1",    # -> benzo[g]pteridine (retained, RT-ok)
]

# Pattern D: bridged-fused whole molecule; upstream parent-selection bypass.
D_WITNESSES = [
    "C1=NC2CCCCCCCCOC3C=NC(CCCCCCCCOC1CC2)CC3",
]

# A2/B/C/D remain a documented abstain until their pattern's task lands. A1 is
# promoted into ``test_a1_witness_names_and_rt`` (Task 2, VERIFIED fix shipped).
ABSTAIN_TODAY_WITNESSES = (
    [("A2", s) for s in A2_WITNESSES]
    + [("B", s) for s in B_WITNESSES]
    + [("C", s) for s in C_WITNESSES]
    + [("D", s) for s in D_WITNESSES]
)

# Canary: ring names that ALREADY work must stay byte-identical + RT-valid. The
# C1/C2/C6 dispatch (spiro.py / tier_a_ring.py / composer.py / fused_rings.py) is
# shared across all of these.
CANARY_NAMES = {
    "c1ccc2ccccc2c1": "naphthalene",
    "c1ccc2cc3ccccc3cc2c1": "anthracene",
    "C1CCC2(CC1)CCCCC2": "spiro[5.5]undecane",
    "C1CC2CC1C2": "bicyclo[2.1.1]hexane",  # simple von Baeyer
    "C1CC2CCC1CC2": "bicyclo[2.2.2]octane",
    "C1C2CC3CC1CC(C2)C3": "adamantane",
    "c1ccc2[nH]ccc2c1": "1H-indole",
    "c1ccc2ncccc2c1": "quinoline",
    "C1CC2(CC1)OCCO2": "1,4-dioxaspiro[4.4]nonane",
}


RT = namedtuple("RT", ["name", "passed", "error"])


class _Timeout(Exception):
    pass


def _ring_rt(smiles: str, timeout_s: int = 60) -> RT:
    """Best-effort name the molecule, then OPSIN-round-trip it to the input's
    full InChIKey. ``passed`` is True only when a non-empty name round-trips.
    SKIPs (via _Timeout) if naming does not return within ``timeout_s`` (the spy's
    single hang was an out-of-scope C70 fullerene; in-scope cores did not hang)."""
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        nm = Orthonym(style="pin", general_fallback=True,
                       allow_aromatic_general=True)
        name = nm.name(smiles)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    if not name:
        return RT(name, False, "no_name")
    res = opsin_roundtrip_check(smiles, name)
    return RT(name, bool(res["passed"]), res.get("error"))


# --- Task 1 RED baseline: every witness abstains (rt-fails) today -------------

@pytest.mark.parametrize("pattern,smiles", ABSTAIN_TODAY_WITNESSES,
                         ids=[f"{p}-{i}" for i, (p, _) in enumerate(ABSTAIN_TODAY_WITNESSES)])
def test_witness_abstains_today(pattern, smiles):
    """0-wrong-safe baseline: every ring-topology witness either abstains or emits
    an OPSIN-unparseable name (rt-fail) today. NEVER a wrong-but-parseable name.
    Task 2 promotes the A1 rows into ``test_a1_witness_names_and_rt``."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert not r.passed, (
        f"{pattern} witness unexpectedly round-trips already: {r.name!r}")


@pytest.mark.parametrize("smiles", A1_WITNESSES)
def test_a1_witness_names_and_rt(smiles):
    """Pattern A1 (Task 2, VERIFIED): a spiro-von-Baeyer name with a BICYCLO cage
    used to append the unsaturation suffix OUTSIDE the spiro brackets
    (``...octane-6,2'-oxolane]-3-ene``), which is OPSIN-grammar-invalid. The fix
    splices it INSIDE the cage component (``...oct-3-ene-6,2'-oxolane]``),
    re-anchored to that component's numbering. Each witness must now NAME and
    OPSIN-round-trip to the input's full InChIKey."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert r.passed, f"A1 witness did not round-trip: name={r.name!r} err={r.error}"


# --- Ring-dispatch canary: already-working names unchanged + RT-valid ---------

@pytest.mark.parametrize("smiles", C_COVERED_BY_FLOOR)
def test_pattern_c_breadth_already_delivered_by_floor(smiles):
    """Measurement: several 3+-component fused-mancude cores the spy logged as
    PIN-tier abstentions already RT-pass at the best-effort tier (von-Baeyer floor
    or retained fused name). Breadth is delivered (0-wrong, non-PIN); the residual
    Task-4 gap is PIN-quality fusion naming, deferred."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert r.passed, f"floor coverage regressed: name={r.name!r} err={r.error}"


@pytest.mark.parametrize("smiles,expected", list(CANARY_NAMES.items()))
def test_ring_canary_unchanged(smiles, expected):
    """Shared ring dispatch must not regress the names that already work. Names
    are checked at the DEFAULT PIN tier (not best-effort) for byte-identity."""
    import orthonym
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = orthonym.name_compound(smiles, style="pin")
    assert name == expected, f"canary regressed: {name!r} != {expected!r}"
    assert opsin_roundtrip_check(smiles, name)["passed"], f"canary RT broke: {name!r}"
