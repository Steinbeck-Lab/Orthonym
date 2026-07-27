"""ARCH-2 — the stereo carve-out must PROVE constitution, not skip the proof.

`namer._final_opsin_validity_gate` carries two mechanisms: a parseability gate
(OPSIN cannot parse -> suppress) and the SELF-01 constitutional check (OPSIN
parses -> re-perceive and compare constitutions). Between them sits the stereo
carve-out: OPSIN's *generation-side* stereo grammar is narrower than IUPAC's, so
the verbatim Blue Book PIN ``(1s,4s)-cyclohexane-1,4-diol`` is REJECTED by OPSIN
even though it is correct. The carve-out asks "does the stereo-STRIPPED form
parse?" and, if yes, ships the full stereo name.

THE DEFECT this file pins: that branch used to ``return name`` without ever
running SELF-01, so ANY constitutional defect rode out free as long as the name
happened to carry a stereo prefix OPSIN rejects. Verification was strongest on
well-formed names and ABSENT on malformed ones -- exactly backwards. Measured
blast radius on  (gates ON): 107 names shipped
through the carve-out, 98 of them constitutionally WRONG; 0 gold regressions.

The fix keeps the carve-out doing what it exists for -- not requiring OPSIN to
parse the stereo LAYER -- while restoring the burden it was never entitled to
skip: the constitution must be proven. ``_self_consistency_verdict`` compares the
InChIKey SKELETON block, which excludes stereochemistry by construction
(namer.py:514-528, ADR-18-07), so it can judge the stereo-STRIPPED parse with no
loss of validity.

WHY A NEW FILE: `tests/unit/test_bbr_gate_stereo.py`'s fixture sets
``_SC_MODE = "off"`` and stubs ``_validity_gate_name_to_smiles`` to the constant
``"CCO"``, deliberately isolating the parseability POLICY from the SELF-01 layer.
Under that fixture the new behaviour is a no-op, so a green run there is not
evidence for it. Every OPSIN fact stubbed below was captured from the REAL oracle
(measurement report D1/D2/D4/D5) and the strip_stereo outputs verified verbatim.
"""
import pytest

import orthonym.namer as nm
from orthonym import Orthonym

# --- The witness (measurement D1/D2) --------------------------------------
# A dibromo-bicyclic carbamate that shipped as an OPSIN-UNPARSEABLE name for a
# DIFFERENT molecule (ring + both Br dropped): the stereo carve-out was its ship
# path (traced stats delta {"gate_stereo_kept": 1}, SELF-01 called 0 times).
WITNESS_SMILES = "COC(=O)NCC[C@@H]1CC[C@H]2[C@@H]1C2(Br)Br"
WITNESS_BAD_NAME = "(1S,4S,5R)-methyl N-octylcarbamate"
WITNESS_STRIPPED_OPSIN_SMILES = "CCCCCCCCNC(=O)OC"  # real OPSIN output, verdict `mismatch`

# --- The D5 case (measurement D5, corpus row 478) -------------------------
# OPSIN PARSES the stripped name but emits SMILES RDKit cannot canonicalise, so
# the constitution is UNVERIFIABLE. Exactly 1 such case on the corpus, 0 in gold.
D5_SMILES = "CO[C@H]1CCCC[C@H]1OC(=O)C[NH+](CI)C2CCCCCC2"
D5_BAD_NAME = "(1R,2S)-1-(decanoyloxy)-2-hydroxymethylcyclohexanium"

# --- The protected class (measurement D4: all 8 gold carve-out rows) ------
# Lowercase pseudo-asymmetric / cis-trans ring descriptors -- the genuine OPSIN
# generation-grammar gap the carve-out exists for. (name, input SMILES,
# stereo-stripped name, real OPSIN SMILES for the stripped name). All verdict `ok`.
PROTECTED = [
    ("(1s,4s)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1",
     "cyclohexane-1,4-diol", "OC1CCC(O)CC1"),
    ("(4as,8as)-decahydronaphthalene", "C1CC[C@@H]2CCCC[C@@H]2C1",
     "decahydronaphthalene", "C1CCC2CCCCC2C1"),
    ("[(1R,3r,5S)-8-methyl-8-azabicyclo[3.2.1]octan-3-yl] 3-hydroxy-2-phenylpropanoate",
     "CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c1ccccc1",
     "[8-methyl-8-azabicyclo[3.2.1]octan-3-yl] 3-hydroxy-2-phenylpropanoate",
     "CN1C2CCC1CC(OC(=O)C(CO)c1ccccc1)C2"),
]


def _stub_opsin(monkeypatch, parsed_map):
    """Enable the gate with SELF-01 ON and a REALISTIC OPSIN stub.

    ``parsed_map`` maps name -> the SMILES the real OPSIN emits for it. A name in
    the map is 'parsed'; every other name is DEFINITIVELY 'rejected' (SMILES
    None) -- which is what OPSIN really does to the stereo forms below.
    """
    monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(nm, "_SC_MODE", "on", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True, raising=False)
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles",
                        lambda n: parsed_map.get(n), raising=False)
    monkeypatch.setattr(nm, "_validity_gate_status",
                        lambda n: "parsed" if n in parsed_map else "rejected",
                        raising=False)


# ---------------------------------------------------------------------------
# 1. The witness, end-to-end, at PRODUCTION settings (real OPSIN, no stubs)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_witness_wrong_molecule_no_longer_ships():
    """The whole point: a name for a DIFFERENT molecule must not ride out on a
    stereo prefix. Real OPSIN, real SELF-01 mode, gate re-enabled -- the only
    monkeypatch undoes conftest's suite-wide gate disable.

    Invariant 11: assert the string that ACTUALLY ships, not merely that the bad
    name is gone -- suppressing a wrong output can unmask a worse generator.
    """
    import orthonym.namer as _nm
    _prev = _nm._DISABLE_VALIDITY_GATE
    _nm._DISABLE_VALIDITY_GATE = False
    try:
        out = Orthonym().name(WITNESS_SMILES)
    finally:
        _nm._DISABLE_VALIDITY_GATE = _prev
    assert out != WITNESS_BAD_NAME, "the wrong-molecule name still ships"
    assert out == "unknown organic compound", (
        f"expected the honest descriptive fallback, got {out!r} -- if this is a "
        "different wrong name, a second generator has been unmasked (invariant 11)")


# ---------------------------------------------------------------------------
# 2. The protected class still ships -- end-to-end, SELF-01 ON, realistic stub
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_protected_pin_still_ships_end_to_end(monkeypatch):
    """DEF-9 gold row (P-93.5 / P-91). With the gate ENABLED, ``_SC_MODE = "on"``
    and a REALISTIC stripped-parse stub, the verbatim Blue Book PIN must still
    ship with its stereo descriptors intact. This is the assertion
    test_bbr_gate_stereo.py cannot make (it pins ``_SC_MODE = "off"``)."""
    _stub_opsin(monkeypatch, {"cyclohexane-1,4-diol": "OC1CCC(O)CC1"})
    assert Orthonym().name("O[C@H]1CC[C@@H](O)CC1") == "(1s,4s)-cyclohexane-1,4-diol"


# ---------------------------------------------------------------------------
# 3. Unit-level, on the gate function directly
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,stripped,opsin_smiles", PROTECTED,
                         ids=["cyclohexanediol", "decalin", "tropane-ester"])
def test_carveout_ships_when_constitution_matches(monkeypatch, name, smiles,
                                                  stripped, opsin_smiles):
    """Stereo-rejected + stripped form parses + constitution MATCHES -> the FULL
    stereo name ships (stereo descriptors preserved) and the carve-out counter
    records the ship."""
    _stub_opsin(monkeypatch, {stripped: opsin_smiles})
    stats = {}
    assert nm._final_opsin_validity_gate(name, smiles, stats) == name
    assert stats.get("gate_stereo_kept") == 1
    assert "self_consistency_suppressed" not in stats


@pytest.mark.unit
def test_carveout_suppresses_when_constitution_differs(monkeypatch):
    """Stereo-rejected + stripped form parses + constitution DIFFERS -> suppressed
    to the honest fallback. This is the 98-name class the hole let through."""
    _stub_opsin(monkeypatch, {"methyl N-octylcarbamate": WITNESS_STRIPPED_OPSIN_SMILES})
    stats = {}
    out = nm._final_opsin_validity_gate(WITNESS_BAD_NAME, WITNESS_SMILES, stats)
    assert out == nm._descriptive_fallback(WITNESS_SMILES)
    assert stats.get("self_consistency_suppressed") == 1
    # The carve-out must NOT claim a ship it did not make.
    assert "gate_stereo_kept" not in stats


# ---------------------------------------------------------------------------
# 4. `inconclusive` still fails OPEN -- never suppress on a comparison you
#    could not make.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("input_smiles,opsin_smiles,why", [
    ("O[C@H]1CC[C@@H](O)CC1", "@@not-smiles@@", "OPSIN output unparseable by RDKit"),
    ("!!! not smiles !!!", "OC1CCC(O)CC1", "input SMILES yields no skeleton"),
], ids=["bad-opsin-smiles", "bad-input-smiles"])
def test_carveout_inconclusive_fails_open(monkeypatch, input_smiles, opsin_smiles, why):
    _stub_opsin(monkeypatch, {"cyclohexane-1,4-diol": opsin_smiles})
    stats = {}
    out = nm._final_opsin_validity_gate("(1s,4s)-cyclohexane-1,4-diol", input_smiles, stats)
    assert out == "(1s,4s)-cyclohexane-1,4-diol", f"suppressed on an unmade comparison ({why})"
    assert "self_consistency_suppressed" not in stats


@pytest.mark.unit
def test_carveout_ships_when_input_smiles_missing(monkeypatch):
    """No input structure at all -> nothing to compare against -> fail OPEN."""
    _stub_opsin(monkeypatch, {"cyclohexane-1,4-diol": "OC1CCC(O)CC1"})
    assert nm._final_opsin_validity_gate(
        "(1s,4s)-cyclohexane-1,4-diol", None, {}) == "(1s,4s)-cyclohexane-1,4-diol"


# ---------------------------------------------------------------------------
# 5. `unavailable` still fails OPEN -- no-JAR / timeout must not suppress
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_transient_unavailable_still_fails_open(monkeypatch):
    """A stereo name whose OPSIN probe merely TIMED OUT must survive: a transient
    OPSIN failure must never turn a valid name into a fallback (CR-01)."""
    monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(nm, "_SC_MODE", "on", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True, raising=False)
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None, raising=False)
    monkeypatch.setattr(nm, "_validity_gate_status", lambda n: "unavailable", raising=False)
    assert nm._final_opsin_validity_gate(
        "(1s,4s)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", {}
    ) == "(1s,4s)-cyclohexane-1,4-diol"


@pytest.mark.unit
def test_no_jar_still_fails_open(monkeypatch):
    """JAR absent -> the gate is a no-op for a stereo name too (D-13).

    The oracle underneath is stubbed HOSTILE (everything rejected, no SMILES) so
    the JAR guard is the ONLY thing that can save the name. Without this the test
    is vacuous: mutation M7 deleted the guard and the test still passed, because
    the unstubbed real OPSIN happened to rescue the name via the carve-out.
    """
    monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(nm, "_SC_MODE", "on", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: False, raising=False)
    monkeypatch.setattr(nm, "_validity_gate_status", lambda n: "rejected", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None, raising=False)
    assert nm._final_opsin_validity_gate(
        "(1s,4s)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", {}
    ) == "(1s,4s)-cyclohexane-1,4-diol"


# ---------------------------------------------------------------------------
# 6. The D5 decision: OPSIN PARSED the stripped name but emitted SMILES RDKit
#    cannot canonicalise -> UNVERIFIABLE -> suppressed, matching the precedent
#    the same function already set for the non-stereo path (namer.py:650-661).
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_d5_parsed_but_uncanonicalisable_is_suppressed(monkeypatch):
    """`parsed` + no usable SMILES means OPSIN emitted a structure RDKit rejects
    as chemically impossible, so the name's constitution can never be proven. The
    stereo layer's presence must not lower the burden of proof: same input, same
    verdict as the non-stereo path -- suppress."""
    monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(nm, "_SC_MODE", "on", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True, raising=False)
    # Both forms yield no usable SMILES; only the STRIPPED form 'parsed'.
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None, raising=False)
    stripped = "1-(decanoyloxy)-2-hydroxymethylcyclohexanium"
    monkeypatch.setattr(nm, "_validity_gate_status",
                        lambda n: "parsed" if n == stripped else "rejected", raising=False)
    stats = {}
    out = nm._final_opsin_validity_gate(D5_BAD_NAME, D5_SMILES, stats)
    assert out == nm._descriptive_fallback(D5_SMILES)
    assert "gate_stereo_kept" not in stats
    assert stats.get("gate_stereo_unverifiable") == 1


# ---------------------------------------------------------------------------
# 7. Tripwire: the carve-out must still be REACHED, i.e. the fix must not have
#    quietly become "suppress every OPSIN-rejected stereo name". If SELF-01 is
#    OFF (the historical mode, and what test_bbr_gate_stereo.py pins) the
#    carve-out must remain a pure pass-through.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_carveout_is_passthrough_when_self01_off(monkeypatch):
    _stub_opsin(monkeypatch, {"methyl N-octylcarbamate": WITNESS_STRIPPED_OPSIN_SMILES})
    monkeypatch.setattr(nm, "_SC_MODE", "off", raising=False)
    stats = {}
    # SELF-01 disabled -> even the wrong-molecule name ships (mode contract), and
    # the carve-out counts the ship.
    assert nm._final_opsin_validity_gate(
        WITNESS_BAD_NAME, WITNESS_SMILES, stats) == WITNESS_BAD_NAME
    assert stats.get("gate_stereo_kept") == 1


@pytest.mark.unit
def test_constitutional_defect_without_stereo_still_suppressed(monkeypatch):
    """Tripwire on the OTHER side: a name whose stripped form ALSO fails to parse
    has no rescuing stereo layer and must stay suppressed (the carve-out must not
    have been widened into a blanket pass)."""
    _stub_opsin(monkeypatch, {})  # nothing parses
    stats = {}
    out = nm._final_opsin_validity_gate("(1s,4s)-cyclohexane-1,4-diolz",
                                        "O[C@H]1CC[C@@H](O)CC1", stats)
    assert out == nm._descriptive_fallback("O[C@H]1CC[C@@H](O)CC1")
    assert stats.get("opsin_suppressed") == 1


# ---------------------------------------------------------------------------
# 8. ARCH-2-FOLLOWUP: the STRIPPED probe is three-valued too. `unavailable`
#    means OPSIN could not be consulted -- it is NOT evidence against the name,
#    so it must fail OPEN exactly as the primary probe already does
#    (namer.py:660-661, CR-01). Lumping it with `rejected` suppressed a name the
#    carve-out exists to rescue, on nothing but a subprocess hiccup.
#
#    WHY THE EXISTING `test_transient_unavailable_still_fails_open` DOES NOT
#    COVER THIS: it stubs `_validity_gate_status` to "unavailable" for EVERY
#    name, so the PRIMARY probe's fail-OPEN at namer.py:660 returns first and the
#    stripped probe is never reached. The stub below is asymmetric on purpose --
#    the primary name is DEFINITIVELY rejected (as real OPSIN rejects these
#    stereo forms) and only the STRIPPED probe hiccups.
# ---------------------------------------------------------------------------
def _stub_opsin_3valued(monkeypatch, status_map, smiles_map=None):
    """Enable the gate with SELF-01 ON and a HOSTILE THREE-VALUED OPSIN stub.

    ``status_map``: name -> ``"parsed"``/``"rejected"``/``"unavailable"``; any name
    NOT listed is ``"rejected"``. ``smiles_map``: name -> the SMILES OPSIN emits;
    any name NOT listed yields ``None``.

    Hostile by construction: nothing is rescued unless this test says so. The real
    oracle must never be reachable here -- on the previous task a guard-deletion
    mutation survived because the unstubbed real OPSIN rescued the name anyway and
    the test passed vacuously.
    """
    smiles_map = smiles_map or {}
    monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(nm, "_SC_MODE", "on", raising=False)
    monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True, raising=False)
    monkeypatch.setattr(nm, "_validity_gate_status",
                        lambda n: status_map.get(n, "rejected"), raising=False)
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles",
                        lambda n: smiles_map.get(n), raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,stripped", [(p[0], p[1], p[2]) for p in PROTECTED],
                         ids=["cyclohexanediol", "decalin", "tropane-ester"])
def test_stripped_probe_unavailable_ships_full_stereo_name(monkeypatch, name,
                                                           smiles, stripped):
    """Primary probe DEFINITIVELY rejected + stripped probe merely UNAVAILABLE ->
    the FULL stereo name ships unchanged.

    Invariant 11 -- assert the exact string that ships, not just that the fallback
    is absent: fail-OPEN must ship the name WITH its stereo descriptors, never the
    stripped constitutional form (which would silently drop stereochemistry) and
    never a fallback.
    """
    _stub_opsin_3valued(monkeypatch, {name: "rejected", stripped: "unavailable"})
    stats = {}
    out = nm._final_opsin_validity_gate(name, smiles, stats)
    assert out == name, "suppressed/altered on a comparison that could not be made"
    assert out != stripped, "shipped the stereo-STRIPPED form -- stereo silently dropped"
    assert out != nm._descriptive_fallback(smiles)
    assert stats.get("gate_stereo_unavailable") == 1
    # It was NOT verified, so it must not be counted as a verified ship, and
    # nothing may be recorded as suppressed.
    assert "gate_stereo_kept" not in stats
    assert "gate_stereo_unverifiable" not in stats
    assert "opsin_suppressed" not in stats
    assert "self_consistency_suppressed" not in stats


@pytest.mark.unit
def test_stripped_probe_rejected_still_suppressed(monkeypatch):
    """Regression guard on the narrowing: widening `unavailable` must NOT widen
    `rejected`. A stripped form OPSIN DEFINITIVELY rejects is a genuine
    constitutional defect and stays suppressed to the honest fallback."""
    name, smiles, stripped = PROTECTED[0][0], PROTECTED[0][1], PROTECTED[0][2]
    _stub_opsin_3valued(monkeypatch, {name: "rejected", stripped: "rejected"})
    stats = {}
    out = nm._final_opsin_validity_gate(name, smiles, stats)
    assert out == nm._descriptive_fallback(smiles), (
        "a DEFINITIVELY rejected stripped form must stay suppressed")
    assert stats.get("opsin_suppressed") == 1
    assert "gate_stereo_unavailable" not in stats


@pytest.mark.unit
def test_carveout_suppression_has_its_own_counter(monkeypatch):
    """Telemetry: the carve-out's SELF-01 suppression must be distinguishable from
    the primary parsed path's, which shares `self_consistency_suppressed`."""
    _stub_opsin_3valued(
        monkeypatch,
        {WITNESS_BAD_NAME: "rejected", "methyl N-octylcarbamate": "parsed"},
        {"methyl N-octylcarbamate": WITNESS_STRIPPED_OPSIN_SMILES})
    stats = {}
    out = nm._final_opsin_validity_gate(WITNESS_BAD_NAME, WITNESS_SMILES, stats)
    assert out == nm._descriptive_fallback(WITNESS_SMILES)
    assert stats.get("gate_stereo_mismatch") == 1
    assert stats.get("self_consistency_suppressed") == 1  # shared counter still moves
    assert "gate_stereo_kept" not in stats
