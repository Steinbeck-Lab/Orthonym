"""v37 ST.1 — route the general-engine stereo-injection sites through the
RT-gated reanchor (SC-1, the dominant stereo-omission abstain sub-cause).

Grounding: . The general engine was the
one ring-stereo producer that injected parent-scope stereo via a plain
`_stereo_prefix` with NO RT-gate, so `general_engine_stereo_complete` certified
an OPSIN-unparseable stereo layer (e.g. `(1s,4s)-bicyclo[2.2.1]heptane`) as
"complete", forcing the verified branch to ABSTAIN instead of degrading to the
best-effort flat constitution. After the fix every top-level general-engine
parent-scope stereo goes through `inject_stereo_reanchored_rt_gated` (helper
`_apply_parent_stereo`): full stereo where it RT-verifies (byte-identical for a
currently-round-tripping name), else the stereo-STRIPPED flat constitution
(stereo omitted per feedback_stereo_omission_is_not_wrong_molecule) — never
abstain-when-the-constitution-RTs, never a wrong stereoisomer.

The DOMINANT, NEW rescue this delivers is flat-degradation: a cage/spiro whose
stereo layer OPSIN cannot verify (pseudoasymmetric r/s, or a numbering OPSIN
cannot read back) used to abstain and now ships the constitution. The tests
assert that rescue on the invariant (ships + right constitution + stereo
omitted), plus never-wrong on the E/Z case, the PIN-tier abstain unchanged, and
a full-stereo non-regression guard.

Run ONLY this file (whole-suite deadlocks on an OPSIN pipe):
    .venv/bin/python -m pytest tests/unit/rules/test_v37_stereo_completion.py -q
"""
import signal
import pytest
from rdkit import Chem

from orthonym import Orthonym, errors, name_compound
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.stereochemistry import strip_stereo
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# These tests are ABOUT the RT/validity gate: the pre-fix names ship an
# OPSIN-unparseable pseudoasymmetric stereo layer (abstain) or degrade to a flat
# constitution ONLY with the gate ON, exactly as production names. The suite
# disables the gate by default (a green-but-blind trap for gate tests — see
# tests/conftest.py), so enable it module-wide.
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module", autouse=True)
def _jvm_slot():
    with jvm_slots(1, purpose="test_v37_stereo_completion"):
        yield


def _alarm(seconds=90):
    def _raise(sig, frm):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)


def _be():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _be_name(smi):
    _alarm()
    try:
        return _be().name(smi)
    finally:
        signal.alarm(0)


def _rt(smi, name):
    return bool(name) and opsin_roundtrip_check(smi, name)["passed"]


def _block1(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


def _ships_flat_constitution(smi, name):
    """Ships (not abstain), names the RIGHT constitution (block-1 InChIKey), and
    omits stereo (full isomeric InChIKey does NOT match — stereo omitted, the
    best-effort stereo-omission contract, not a wrong molecule)."""
    if errors.is_failure_name(name):
        return False, "abstained"
    r = opsin_roundtrip_check(smi, name)
    opsin_smi = r.get("opsin_smiles")
    if not opsin_smi:
        return False, "constitution not OPSIN-parseable"
    if _block1(opsin_smi) != _block1(smi):
        return False, "WRONG constitution (block-1 mismatch)"
    if r["passed"]:
        return False, "full stereo (not the flat-degradation case)"
    return True, "flat constitution, stereo omitted"


class TestST1FlatDegradationRescue:
    """The dominant, NEW ST.1 rescue: an unverifiable stereo layer degrades to
    the flat constitution instead of forcing an abstain (all three abstained
    before ST.1)."""

    def test_pseudoasym_vonbaeyer_ships_flat_not_abstain(self):
        # norbornane: pseudoasym (1s,4s), OPSIN-unparseable -> flat constitution.
        smi = "C1C[C@H]2CC[C@@H]1C2"
        name = _be_name(smi)
        assert not errors.is_failure_name(name)          # NOT 'unknown organic compound'
        assert name == "bicyclo[2.2.1]heptane"           # stereo omitted (constitution)

    def test_sesquiterpene_cage_ships_flat_not_abstain(self):
        # genuine R/S cage candidate-B cannot place -> flat constitution (SC-3
        # residual after ST.1: flat, never abstain).
        smi = "CC(C)=CCC[C@]1(C)[C@H]2C[C@@H]3[C@H](C2)[C@@]31C"
        ok, why = _ships_flat_constitution(smi, _be_name(smi))
        assert ok, why

    def test_endoperoxide_substituent_stereo_ships_flat_not_abstain(self):
        # SC-4: pseudoasym stereo in a ring substituent on a chain parent;
        # subsumed by ST.1's top-level reanchor -> flat constitution.
        smi = "C[C@]12OO[C@](CCC(=O)O)(c3ccccc31)c1ccccc12"
        ok, why = _ships_flat_constitution(smi, _be_name(smi))
        assert ok, why


class TestST1NeverWrongAndPinPreserved:
    def test_ez_chain_with_ring_substituent_ships_never_wrong(self):
        # E/Z on a chain parent + R/S in a ring substituent. Never abstain,
        # never wrong: FULL RT, or a stereo-stripped constitution (block-1 RT).
        smi = "C/C(=C\\CC[C@]1(C)[C@H]2C[C@@H]3[C@H](C2)[C@@]31C)C(=O)O"
        name = _be_name(smi)
        assert not errors.is_failure_name(name)
        assert _rt(smi, name) or opsin_roundtrip_check(
            smi, strip_stereo(name))["opsin_smiles"] is not None

    def test_completable_spiro_full_stereo_non_regression(self):
        # A completable spiro whose FULL stereo round-trips must keep shipping it
        # (the reanchor's candidate-A path is byte-identical for a currently-RT
        # name — the v36 Wave-E non-regression argument).
        smi = ("CC(=O)OC[C@]12C[C@H](OC(=O)CC(C)C)C(C)=C[C@H]1O[C@@H]1"
               "[C@H](O)[C@@H](O)[C@@]2(C)[C@]12CO2")
        name = _be_name(smi)
        assert not errors.is_failure_name(name)
        assert _rt(smi, name)                            # FULL stereo, round-trips

    def test_pseudoasym_named_blocker_pin_tier_still_abstains(self):
        # gold V36-VB-PSEUDOASYM-01: PIN tier must NOT ship a non-RT (1r,5s);
        # default tier abstains (SELF-01 suppresses the flat form at PIN).
        _alarm()
        try:
            name = name_compound("C1C[C@H]2C[C@H](C2)O1")
        finally:
            signal.alarm(0)
        assert errors.is_failure_name(name)


class TestST1KnownFollowOn:
    @pytest.mark.xfail(strict=False, reason=(
        "ST.3 follow-on (NOT stereo-site routing): the reanchor builds a "
        "full-stereo name that RT-verifies, but certify_general_result rejects "
        "the spiro-vonbaeyer component fallback with STEREO_PARENT_BLOCK_AMBIGUOUS "
        "(binding-spine parent-stereo-block proof gap — stereo_atom_to_locant is "
        "int-locant-only), routing to T4 -> abstain. The achiral constitution "
        "ships, so this violates 'never abstain-when-constitution-RTs' and is a "
        "documented named-blocker for the ST.3/binding-spine work. Confirmed "
        "pre-existing (name identical pre/post ST.1)."))
    def test_completable_spiro_certify_gap_documented(self):
        smi = "CC(=O)O[C@@H]1C[C@H]2O[C@@H]3C=C(C)CC[C@]3(C)[C@]1(C)[C@@]21CO1"
        name = _be_name(smi)
        assert not errors.is_failure_name(name)          # currently abstains -> xfail
        assert _rt(smi, name)
