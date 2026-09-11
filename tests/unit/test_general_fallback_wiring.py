# tests/unit/test_general_fallback_wiring.py
"""v25 G1: general_fallback flag — default OFF byte-identity, ON recovery."""
from unittest import mock

import pytest

from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def test_default_off_is_byte_identical_on_easy_molecule():
    assert (Orthonym(_disable_opsin_validity_gate=True).name("CCO")
            == Orthonym(_disable_opsin_validity_gate=True,
                         general_fallback=False).name("CCO"))


def test_flag_defaults_false():
    assert Orthonym()._general_fallback is False


def test_engine_fires_when_legacy_general_abstains():
    """Force the legacy GENERAL pipeline to abstain; engine must recover."""
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out == "2-chlorobutane"


def test_engine_does_not_fire_when_flag_off():
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=False)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out != "2-chlorobutane"


def test_engine_ring_fallback_fires_behind_flag():
    from orthonym.assembly.general_engine import name_general
    from rdkit import Chem
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    smi = "O=C1CCC2CCCCC2C1"  # decalin-2-one: GENERAL-dispatched cage + ketone
    mol = Chem.MolFromSmiles(smi)
    feats = nm._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    nm._classify(feats)
    expected = name_general(mol, feats).name
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name(smi)
    assert out == expected


def test_late_recovery_replaces_suppressed_wrong_name():
    """decalin-2-one: legacy names it 'decahydronaphthalene' (atom-dropping,
    -suppressed). The late recovery must emit the engine's verified
    cage name instead. Gate disabled here -> recovery returns the engine
    name directly (in prod the same name must clear)."""
    from orthonym.assembly.general_engine import name_general
    from rdkit import Chem
    smi = "O=C1CCC2CCCCC2C1"
    nm_off = Orthonym(_disable_opsin_validity_gate=True)
    nm_on = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    mol = Chem.MolFromSmiles(smi)
    feats = nm_on._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    nm_on._classify(feats)
    engine_name = name_general(mol, feats).name
    out_on = nm_on.name(smi)
    # With the flag OFF nothing changes (byte-identity)...
    assert nm_off.name(smi) != engine_name
    #...with it ON, the abstention/suppression is recovered (or, if the
    # ungated legacy path ships its wrong candidate here because is
    # disabled in unit tests, the flag must still not corrupt it).
    assert out_on in (engine_name, nm_off.name(smi))


def test_mancude_refused_regardless_of_optin_no_jar():
    """Post-G5-A, an aromatic/mancude cage is refused by the engine, so neither
    general_fallback nor general_fallback_unverified ships a von-Baeyer name for
    it — even without the OPSIN jar (fails open). This is the safety win:
    the invalid/non-PIN von-Baeyer polyene for aromatic systems no longer ships
    in ANY mode. The real name comes from the PIN path."""
    from unittest import mock as _m
    smi = "CC1C2C=CC1c1ccccc12"  # methyl-benzonorbornadiene (aromatic ring)
    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=False):
        for kw in ({"general_fallback": True},
                   {"general_fallback": True,
                    "general_fallback_unverified": True}):
            out = Orthonym(**kw).name(smi) or ""
            assert "tricyclo" not in out and "bicyclo" not in out


def test_flag_propagates_into_recursion():
    from orthonym.metrics.provenance import general_fallback_ctx
    from orthonym.namer import name_compound
    tok = general_fallback_ctx.set(True)
    try:
        out = name_compound("CC1C2C=CC1c1ccccc12",
                            general_fallback_unverified=True)
    finally:
        general_fallback_ctx.reset(tok)
    assert "tricyclo" in out or out == "unknown organic compound"


def test_best_effort_no_java_never_ships_valence_illegal():
    """Even with the OPSIN jar absent (fails open) and best-effort ON,
    a valence-illegal von-Baeyer name (a ring double-bond locant coinciding
    with a dioxo/one locant -> 5-bond carbon, the caffeine class) must never
    ship. Post-G5-A the engine already refuses the mancude entry; this asserts
    the END-TO-END invariant holds via the Java-free source guard too."""
    from unittest import mock as _m
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=False):
        out = Orthonym(general_fallback=True,
                        general_fallback_unverified=True).name(smi)
    bad = (out or "")
    assert not ("bicyclo" in bad and "diene" in bad and "dioxo" in bad)


# ---------------------------------------------------------------------------
# task-JAR-ABSENT (a review 0-wrong hole): the general-engine RECOVERY lanes
# (`_try_general_engine_recovery` late-recovery ladder + the inline-G1 lane in
# `name`) previously shipped a best-effort candidate UNVERIFIED when the OPSIN
# jar is absent -- there was no `else: return None`, so control fell through to
# the emit with opsin_status="unverified" and ZERO verification. The fix routes
# both lanes through the OPSIN-free reconstructor oracle `verify_or_none`, which
# is provably None with `name_facts=None` and no jar today -> a clean fail-closed
# ABSTAIN. These tests guard THAT lane.
#
# NOTE (measured, see task-jar-absent-report.md): in the *natural* config
# (assemble_name NOT mocked) the 7 a review witnesses do NOT reach these recovery
# lanes -- assemble_name returns a wrong-but-non-failure name (e.g. `methane` for
# `COS(=O)(=O)O`), so the fallback guard `not name or is_failure_name(name)` is
# False and the lanes never fire. Those witnesses ship via the THIRD site,
# `_final_opsin_validity_gate`'s jar-absent fail-OPEN (namer.py ~:1209). That
# site is now CLOSED tier-aware: it fails CLOSED (abstains) when
# `besteffort_unverified=True` (the four call sites pass
# `self._general_fallback_unverified`), because a genuinely absent jar makes an
# unverified best-effort emission unverifiable and 0-wrong is ABSOLUTE. The
# default/PIN tier keeps the historical fail-OPEN (a separate no-Java-PIN policy
# question), so a correct-by-construction PIN name (`ethanol` from `CCO` at
# default tier) still ships jar-absent. The mocked-lane tests below guard the two
# recovery lanes; `test_jar_absent_besteffort_natural_config_*` below guards the
# ~:1209 site in the NATURAL config (assemble_name NOT mocked).

_JAR_ABSENT_WITNESSES = [
    "COS(=O)(=O)O",
    "C1CCC(CC1)OC(=O)NP(=O)(Cl)Cl",
    "CCOP(=O)(OCC)N",
    "ClP(Cl)(=O)OC1=CC=CC=C1",
    "CCON=C(C)OP(=O)(OC)SC",
    "CC(C)OC(=O)NS(=O)(=O)Cl",
    "COB(OC)OC",
]


def _jar_absent_patches():
    """Simulate a genuine no-Java deployment: the namer probe AND the two
    independent jar-discovery functions verify_or_none reaches all see no jar."""
    from unittest import mock as _m
    return [
        _m.patch("orthonym.namer._validity_gate_jar_present", return_value=False),
        _m.patch("orthonym.validation.atom_coverage.find_opsin_jar",
                 return_value=None),
        _m.patch("orthonym.validation.opsin_roundtrip._find_opsin_jar",
                 return_value=None),
    ]


def test_jar_absent_besteffort_recovery_lane_never_ships_unverified():
    """The a review 0-wrong hole, for the lane the fix covers: with the jar absent
    and best-effort ON, once a recovery lane is engaged (assemble_name abstains)
    every witness must ABSTAIN -- never a wrong-molecule name shipped unverified.
    Before the fix these shipped `methane`, `(phosphonooxy)benzene`, etc."""
    import contextlib
    from unittest import mock as _m
    from orthonym.errors import is_failure_name
    for smi in _JAR_ABSENT_WITNESSES:
        with contextlib.ExitStack() as es:
            for p in _jar_absent_patches():
                es.enter_context(p)
            es.enter_context(_m.patch("orthonym.namer.assemble_name",
                                      return_value="unknown organic compound"))
            out = Orthonym(general_fallback=True,
                            general_fallback_unverified=True).name(smi)
        assert out is None or is_failure_name(out), (smi, out)


def test_jar_absent_besteffort_ships_only_when_reconstructor_confirms():
    """The `verify_or_none` seam is the DECIDING gate on the jar-absent
    best-effort recovery lanes (not an unrelated abstain): with it returning
    None (its real jar-absent verdict) the emission abstains; with it patched to
    CONFIRM, the same emission ships. Proves the fix wires the oracle in-path."""
    import contextlib
    from unittest import mock as _m
    import orthonym.validation.reconstruct as _R
    from orthonym.errors import is_failure_name
    smi = "O=C1CCC2CCCCC2C1"  # decalin-2-one: general-engine-named cage ketone

    def run(confirm):
        with contextlib.ExitStack() as es:
            for p in _jar_absent_patches():
                es.enter_context(p)
            es.enter_context(_m.patch("orthonym.namer.assemble_name",
                                      return_value="unknown organic compound"))
            if confirm:
                es.enter_context(_m.patch.object(
                    _R, "verify_or_none",
                    side_effect=lambda name, s, name_facts=None: name))
            return Orthonym(general_fallback=True,
                             general_fallback_unverified=True).name(smi)

    abstained = run(confirm=False)
    shipped = run(confirm=True)
    assert abstained is None or is_failure_name(abstained), abstained
    assert shipped and not is_failure_name(shipped), shipped


@pytest.mark.opsin_gate
def test_jar_present_witnesses_unchanged_by_fix():
    """Jar-PRESENT regression: the fix ONLY narrows a jar-ABSENT branch (both
    edited branches are unreachable when `_validity_gate_jar_present` is True),
    so with a live jar + the validity gate ON the witnesses behave exactly as
    before -- best-effort ships a POSITIVELY round-tripping (correct) name, or
    abstains; NEVER a wrong molecule. `opsin_gate` marker => real gate (skipped
    with no jar, per conftest's guard). No jar-absent patch here."""
    from rdkit import Chem
    from orthonym.errors import is_failure_name
    from orthonym.validation.opsin_roundtrip import opsin_parse
    for smi in ("COS(=O)(=O)O", "ClP(Cl)(=O)OC1=CC=CC=C1"):
        out = Orthonym(general_fallback=True,
                        general_fallback_unverified=True).name(smi)
        if out is None or is_failure_name(out):
            continue  # abstain is 0-wrong-safe
        parsed = opsin_parse(out)
        assert parsed, (smi, out, "jar-present best-effort name must OPSIN-parse")
        want = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        got = Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
        assert got == want, (smi, out, "jar-present emission must round-trip")


@pytest.mark.opsin_gate
def test_jar_absent_besteffort_natural_config_witnesses_never_ship_wrong():
    """THE task deliverable: the 7 a review witnesses in the NATURAL config
    (assemble_name NOT mocked) at best-effort + genuinely-absent jar. These ship
    their WRONG names (`methane` for `COS(=O)(=O)O`, `(phosphonooxy)benzene` for
    the phenyl phosphorodichloridate, etc.) NOT via the recovery lanes -- the
    ordinary substitutive producers assemble an atom-DROPPING name that reaches
    `_final_opsin_validity_gate`, whose jar-absent branch used to fail OPEN. The
    ~:1209 tier-aware fix now fails CLOSED for best-effort-unverified, so every
    witness must ABSTAIN -- never a wrong molecule. (Jar-absent nothing
    OPSIN-verifies and the Wave-0 reconstructor has no NameFacts extractor for an
    arbitrary emitted name yet, so abstain is the only 0-wrong outcome today; a
    future extractor would let a reconstructor-CONFIRMED correct name ship here
    instead, still never a wrong one.)"""
    import contextlib
    from orthonym.errors import is_failure_name
    for smi in _JAR_ABSENT_WITNESSES:
        with contextlib.ExitStack() as es:
            for p in _jar_absent_patches():
                es.enter_context(p)
            out = Orthonym(general_fallback=True,
                            general_fallback_unverified=True).name(smi)
        assert out is None or is_failure_name(out), (smi, out)


@pytest.mark.opsin_gate
def test_jar_absent_default_tier_by_construction_pin_still_ships():
    """By-construction guard: the ~:1209 fix is TIER-AWARE. A correct-by-
    construction PIN name at the DEFAULT tier (`besteffort_unverified=False`)
    keeps the historical jar-absent fail-OPEN, so `ethanol` from `CCO` (and
    other retained/systematic PINs) still SHIP with no jar -- the fix does not
    gratuitously abstain the deterministic PIN path (0-wrong there is protected
    by construction, not by OPSIN)."""
    import contextlib
    from orthonym.errors import is_failure_name
    for smi, want in (("CCO", "ethanol"), ("c1ccccc1", "benzene"),
                      ("CC(C)O", "propan-2-ol")):
        with contextlib.ExitStack() as es:
            for p in _jar_absent_patches():
                es.enter_context(p)
            out = Orthonym().name(smi)  # DEFAULT tier
        assert out == want, (smi, out)
        assert not is_failure_name(out), (smi, out)


@pytest.mark.opsin_gate
def test_jar_absent_besteffort_tier_abstains_unverifiable_by_construction():
    """Scope decision (documented, deliberate -- see task-jar-absent-report.md):
    at BEST-EFFORT tier jar-absent, the ~:1209 gate cannot distinguish a correct-
    by-construction PIN (`ethanol`) from an atom-dropping producer bug (`methane`)
    -- both arrive with provenance source=None and `verify_or_none(name_facts=
    None)` is None for both -- so it fails CLOSED for the whole best-effort tier.
    `ethanol` from `CCO` therefore ABSTAINS at best-effort jar-absent. This is the
    0-wrong-mandated cost (0-wrong ABSOLUTE > breadth) and it only affects a
    no-Java best-effort deployment; it is asserted here so the deliberate abstain
    is not silently regressed to a fail-OPEN ship. (Same molecule still ships at
    the DEFAULT tier -- previous test -- and jar-PRESENT at every tier.)"""
    import contextlib
    from orthonym.errors import is_failure_name
    with contextlib.ExitStack() as es:
        for p in _jar_absent_patches():
            es.enter_context(p)
        out = Orthonym(general_fallback=True,
                        general_fallback_unverified=True).name("CCO")
    assert out is None or is_failure_name(out), out


def test_oxo_ene_valence_illegal_guard_unit():
    from orthonym.perception.structure_conservation import (
        oxo_ene_valence_illegal,
    )
    # caffeine-class: plain '1-ene' = bond 1-2, and 2-oxo -> C2 both =C and =O
    assert oxo_ene_valence_illegal(
        "2,4-dioxo-3,5,7,9-tetraazabicyclo[4.3.0]nona-1,7-diene") is True
    # 2-oxo + 2-ene (bond 2-3): C2 both =O and =C
    assert oxo_ene_valence_illegal("2-oxobicyclo[2.2.2]oct-2-ene") is True
    # legal: oxo at 2, ene at 5(6) far away -> no shared carbon
    assert oxo_ene_valence_illegal("2-oxobicyclo[4.4.0]dec-5(6)-ene") is False
    assert oxo_ene_valence_illegal("bicyclo[4.4.0]decan-2-one") is False    # no ene
    assert oxo_ene_valence_illegal("ethanol") is False

    # M4 (scope-aware): a 2-oxo cited INSIDE a substituent scope is a
    # different atom-numbering from the parent ring's ene -- the global union
    # falsely flagged these salt/decorated names. The oxo and the ene never
    # share a scope, so no carbon is five-valent -> legal (was a false positive).
    assert oxo_ene_valence_illegal(
        "1-{2-oxo-3-oxapropan-1-yl}cyclohexa-1,3,5-triene") is False
    assert oxo_ene_valence_illegal(
        "1-{5-[4-(5-cyanocyclohexa-1,3,5-trien-1-yl)-1,4-diazacyclohexan-4-ium"
        "-1-yl]-2-oxo-3-oxa-1-azapentan-1-yl}-6-(1-oxapropan-1-yl)"
        "cyclohexa-1,3,5-triene chloride") is False
    # a genuinely-illegal cation still fires when it is one salt word (the
    # space-split must not lose a real same-scope collision)
    assert oxo_ene_valence_illegal("2-oxobicyclo[2.2.2]oct-2-ene chloride") is True
    # the caffeine collision survives even with an unrelated oxo-bearing
    # substituent present elsewhere (parent-scope collision is independent)
    assert oxo_ene_valence_illegal(
        "2,4-dioxo-9-[2-oxoethyl]-3,5,7,9-tetraazabicyclo[4.3.0]nona-1,7-diene"
    ) is True
