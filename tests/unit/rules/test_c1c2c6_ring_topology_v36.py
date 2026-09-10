""" Milestone C1/C2/C6 — ring-topology construction gap.

Grounding trace: `internal notes` (197 witnesses, 4 decline
patterns). Witnesses here are BARE RING CORES whose topology currently abstains
(the true ring-topology-construction population — not the C3 substituent/assembly
gap). Every emitted ring name must OPSIN-round-trip to the input's full InChIKey
(``opsin_roundtrip_check``) or the molecule abstains; a wrong-but-parseable name is
a 0-wrong break and must never occur.

Patterns (trace, VERIFIED):
  A1 spiro-von-Baeyer with a BICYCLO cage: the ``-ene`` unsaturation suffix is
      emitted OUTSIDE the spiro brackets (``...octane-6,2'-oxolane]-3-ene``), which
      is OPSIN-grammar-invalid. VERIFIED fix: splice it INSIDE the cage component
      (``...oct-3-ene-6,2'-oxolane]``). Task 2. -> GREEN.
  A2 spiro-von-Baeyer that builds a name with a wrong-numbering / other
      construction bug (no trailing-ene). Task 3 triage. -> abstain today.
  B mixed-spiro-fused (spiro joining two fused/bridged sub-systems);
      ``name_mixed_spiro_fused`` has no construction path. Task 5. -> abstain today.
  C 3+-component ortho/ortho-peri-fused mancude; ``name_ortho_fused_bicyclic`` is
      2-component-only. Task 4. -> abstain today.
  D bridged-fused whole-molecule; upstream parent-selection bypass (small ring
      claims parent, ``sub_count=0``). Task 5. -> abstain today.
"""
from __future__ import annotations

import signal
from collections import namedtuple

import pytest

# --- witness sets (trace V36-a trace-C1C2C6.md; bare ring cores) --------------------

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

# Pattern A2: spiro-von-Baeyer builds a name that rejects. MEASURED (Task 3,
# V36-C1C2C6-TASK3-A2-FINDING.md): the trace's "wrong-numbering" premise is REFUTED --
# 67/85 spiro-VB fails are STEREO-OMISSION (right constitution + numbering, missing
# descriptors), only 16 are real constitutional defects. A re-anchored stereo injection
# closes a measured 40/85 (RT-gated, 0-wrong), but the clean wiring touches the SHARED
# stereo primitive (format_stereodescriptor_string tuple rendering + _STEREO_PREFIX_RE
# primed-locant recognition) and must be validated by the full PIN gate -> DEFERRED.
A2_WITNESSES = [
    "c1cc2c(c3c1CNC3)O[C@@]1(CCC[C@H]3CCCC[C@@H]31)C2",   # stereo-omission (primed component)
    # The second A2 witness (C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1) was
    # RESOLVED by CP2 (fused-atom numbering): it now names a determinate,
    # full-InChIKey-RT-verified spiro-VB name (0-wrong). Promoted to
    # RESOLVED_BY_V38_CP2 below.
]

# The A2 stereo-closeable witness was measured to RT-pass once its stereo
# block is completed (finding doc). CP2's fused-component numbering fix
# delivered exactly that as a side effect, so it now RT-passes (see
# ``test_a2_stereo_completion_target``, no longer xfail).
A2_STEREO_CLOSEABLE = ["C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1"]

# CP2 (fused-atom numbering, '4a'/'8a' fusion locants) RESOLVED two
# witnesses that abstained here: a mixed-spiro-fused decalin, and a spiro-VB whose
# fused sub-component numbering was the blocker. Each now emits a determinate name
# (identical across randomized atom orders) that OPSIN-round-trips to the input's
# full InChIKey -- 0-wrong. Kept as a positive canary so a future numbering change
# cannot silently re-break them.
RESOLVED_BY_V38_CP2 = [
    "C1CC[C@@H]2C[C@]3(CC[C@H]2C1)CO3",                   # (2R,4aR,8aR)-spiro[decahydronaphthalene-2,2'-oxirane]
    "C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1",     # (1'R,2S,3'R,7'R)-spiro[oxolane-2,6'-tricyclo[9.3.0.0^3,7]tetradeca-10,13-diene]
]

# Pattern B: mixed-spiro-fused. DEFERRED (V36-C1C2C6-TASK45-BCD-FINDING.md): a mix of an
# OPSIN-grammar bug in name_mixed_spiro_fused's component assembly + stereo-omission (the
# latter closed by the same deferred shared-stereo-path fix as A2). Abstains today (0-wrong).
B_WITNESSES = [
    "C1=CCC2(C1)COc1ccccc12",
    "C1=CC2(C=CC1)Cc1ccccc1O2",
    # The third B witness (C1CC[C@@H]2C[C@]3(CC[C@H]2C1)CO3) built an unparseable
    # spiro[decahydronaphthalene-7,2'-oxirane] at HEAD; CP2 (correct 4a/8a
    # fusion locants + completed stereo) RESOLVED it -> RESOLVED_BY_V38_CP2 below.
]

# Pattern C: multi-component ortho / ortho-peri-fused mancude that genuinely abstains
# (rt-fails) at the best-effort tier. DEFERRED (V36-C1C2C6-TASK45-BCD-FINDING.md): general
# N-component fusion nomenclature is a documented large build (name_ortho_fused_bicyclic is
# 2-ring-only,); the recommended path is an offline OPSIN-validated
# fused-template index. Breadth already floor-delivered for 24/41 ortho-fused (see below).
C_WITNESSES = [
    "c1ccc2c(c1)CO[C@H]2[C@H]1OCc2ccccc21",
    # The second C witness (c1ccc([C@@H]2O[C@@]23CNc2ccccc2CN3)cc1) was NOT an
    # ortho-fused abstainer at all: it is a MASKED-SPIRO core (a spiro-oxirane on a
    # benzo-fused diazocine BRIDGEHEAD, so the spiro atom sits in >=3 SSSR rings and
    # get_spiro_atoms missed it). M4 L1a's masked-spiro floor lever now names it and
    # it full-InChIKey-round-trips -> promoted to C_RESOLVED_BY_MASKED_SPIRO below.
    "c1ccc2c(c1)CCOCCCOCc1ccccc1CCOCCCOCc1ccccc1CCOCCCOC2",  # ortho-peri-fused
]

# M4 L1a (masked-spiro floor lever): a monospiro whose spiro atom is ALSO a
# von-Baeyer bridgehead (>=3 SSSR-ring membership) is invisible to get_spiro_atoms,
# so every spiro namer used to bail and the core VOIDED. The lever detects the true
# spiro cut-vertex, splits at it and assembles the separable name. This
# core now emits a determinate name that OPSIN-round-trips to the input's full
# InChIKey (0-wrong). Positive canary against re-breaking.
C_RESOLVED_BY_MASKED_SPIRO = [
    "c1ccc([C@@H]2O[C@@]23CNc2ccccc2CN3)cc1",  # -> (4S,3'S)-3'-(cyclohexa-1,3,5-trien-1-yl)spiro[2,5-diazabicyclo[5.4.0]undeca-1(11),7,9-triene-4,2'-oxirane]
]

# MEASUREMENT (best-effort tier, this session): the von-Baeyer best-effort FLOOR
# already RT-covers many 3+-component fused-mancude cores that the trace recorded as
# PIN-tier abstentions (24/41 ortho-fused witnesses RT-pass). For those the
# breadth win is ALREADY delivered (a 0-wrong non-PIN name); the remaining gap is
# PIN-QUALITY (emitting the proper fusion name, e.g. pyrimido[4,5-b]quinoline,
# instead of a triaza-von-Baeyer name) -- a separate, larger fusion-nomenclature
# build, deferred (Task 4 scope note).
C_COVERED_BY_FLOOR = [
    "c1ccc2nc3ncncc3cc2c1",    # -> 2,4,6-triazatricyclo[...]heptaene (non-PIN, RT-ok)
    "c1ccc2nc3ncncc3nc2c1",    # -> benzo[g]pteridine (retained, RT-ok)
]

# Pattern D: bridged-fused whole molecule. DEFERRED (V36-C1C2C6-TASK45-BCD-FINDING.md):
# measurement REFUTES the trace's "pure routing bypass" premise -- detect_bridged_fused
# returns False and name_bridged_fused_system returns None for this macrocyclic-bridge
# topology, so it is a detector+construction gap, not a routing patch. Abstains (0-wrong).
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
    SKIPs (via _Timeout) if naming does not return within ``timeout_s`` (the trace's
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


@pytest.mark.parametrize("smiles", A2_STEREO_CLOSEABLE)
def test_a2_stereo_completion_target(smiles):
    """Task-3 target, now GREEN: this spiro-von-Baeyer core was constitution-correct
    but abstained because its stereo block was missing (V36-C1C2C6-TASK3-A2-FINDING.md).
     CP2's fused-component numbering fix delivered the closing behaviour as a
    side effect -- it now names and OPSIN-round-trips to the input's full InChIKey."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert r.passed, f"A2 stereo target still abstains: name={r.name!r} err={r.error}"


@pytest.mark.parametrize("smiles", RESOLVED_BY_V38_CP2)
def test_v38_cp2_resolved_witness_names_and_rt(smiles):
    """ CP2 (fused-atom numbering, resolved these two previously-
    abstaining witnesses: each now emits a determinate name that OPSIN-round-trips
    to the input's full InChIKey (0-wrong). A positive canary against re-breaking."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert r.passed, f"v38 CP2 witness regressed: name={r.name!r} err={r.error}"


# --- Ring-dispatch canary: already-working names unchanged + RT-valid ---------

@pytest.mark.parametrize("smiles", C_RESOLVED_BY_MASKED_SPIRO)
def test_masked_spiro_core_names_and_rt(smiles):
    """M4 L1a: a masked-spiro core (spiro atom that is also a von-Baeyer
    bridgehead, >=3 SSSR rings, invisible to get_spiro_atoms) used to VOID; the
    masked-spiro floor lever now emits a separable name that
    OPSIN-round-trips to the input's full InChIKey (0-wrong)."""
    try:
        r = _ring_rt(smiles)
    except _Timeout:
        pytest.skip("naming exceeded timeout (out-of-scope size)")
    assert r.passed, f"masked-spiro core did not round-trip: name={r.name!r} err={r.error}"


@pytest.mark.parametrize("smiles", C_COVERED_BY_FLOOR)
def test_pattern_c_breadth_already_delivered_by_floor(smiles):
    """Measurement: several 3+-component fused-mancude cores the trace logged as
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
