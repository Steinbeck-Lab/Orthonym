"""The von Baeyer cage ceilings by tier.

``vonbaeyer_universal`` analyses a ring system only up to ``MAX_CAGE_ATOMS`` = 40
skeletal atoms and ``MAX_CAGE_RINGS`` = 8 rings at the PIN tier: the
main-bridge selection there is not yet an exhaustive search, so a larger system
could get a correct but non-preferred name. At the best-effort tier a name need
only be a verified systematic name, so the tier takes its measured ceilings
(``BEST_EFFORT_MAX_CAGE_ATOMS`` / ``BEST_EFFORT_MAX_CAGE_RINGS``; TRIAGE 'Large
molecules -- part 3'). The tier is the tier of the outermost naming request
(``provenance.best_effort_request_ctx``), so a best-effort recovery that a PIN
request runs keeps the PIN ceilings, and a name built on a ring system beyond the
PIN ceilings is recorded as not a PIN.

Every expected name is read back by a fresh OPSIN call that does not go through
the engine (``tests.support.rt_assert``).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.jvm_budget import jvm_slots
from orthonym.metrics.provenance import best_effort_ctx, best_effort_request_ctx
from orthonym.rules import vonbaeyer_universal as vbu
from tests.support.rt_assert import name_is_rt_exact

#: calix[6]arene: one ring system of 42 skeletal atoms and 7 rings.
CALIX6 = ("Oc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)"
          "Cc1cccc(c1O)C2")
CALIX6_NAME = (
    "heptacyclo[31.3.1.1^3,7.1^9,13.1^15,19.1^21,25.1^27,31]dotetraconta-"
    "1(37),3(42),4,6,9(41),10,12,15(40),16,18,21,23,25(39),27,29,31(38),33,35-"
    "octadecaene-37,38,39,40,41,42-hexol")
#: a cyclic thiazole peptide (ChEBI:107473): a 45-atom ring system, named on the
#: PIN path of the best-effort tier.
CYCLIC_PEPTIDE = (
    "CC[C@H](C)[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)"
    "[C@@H](CO)NC(=O)CNC(=O)CNC(=O)c2csc(n2)[C@@H](CC(N)=O)NC(=O)[C@H]"
    "(Cc2ccc(O)cc2)NC(=O)c2csc(n2)[C@H](CC(C)C)NC(=O)c2csc1n2")


def _row(smiles, tier):
    with jvm_slots(1, purpose="cage-caps-test"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.unit
def test_pin_tier_ceilings_are_unchanged():
    assert (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS) == (40, 8)
    assert vbu.cage_caps() == (40, 8)


def _in_request(request_tier, fn, *args):
    token = best_effort_request_ctx.set(request_tier)
    try:
        return fn(*args)
    finally:
        best_effort_request_ctx.reset(token)


@pytest.mark.unit
def test_best_effort_ceilings_follow_the_request_tier():
    raised = (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    assert raised[0] > vbu.MAX_CAGE_ATOMS and raised[1] > vbu.MAX_CAGE_RINGS
    assert _in_request(True, vbu.cage_caps) == raised
    assert _in_request(False, vbu.cage_caps) == (40, 8)
    # the best-effort breadth context alone is no best-effort request: a
    # recovery's fresh best-effort engine sets it inside a PIN request
    token = best_effort_ctx.set(True)
    try:
        assert _in_request(False, vbu.cage_caps) == (40, 8)
        assert vbu.cage_caps() == (40, 8)
    finally:
        best_effort_ctx.reset(token)
    assert vbu.cage_caps() == (40, 8)


@pytest.mark.unit
def test_analyzer_refuses_the_42_atom_system_at_pin_and_analyses_it_at_best_effort():
    from orthonym.metrics.provenance import clear_provenance, get_provenance
    mol = Chem.MolFromSmiles(CALIX6)
    assert vbu.analyze_cage_universal(mol, allow_mancude=True) is None
    def analyse():
        return vbu.analyze_cage_universal(mol, allow_mancude=True)

    assert _in_request(False, analyse) is None
    clear_provenance()
    cage = _in_request(True, analyse)
    assert cage is not None and len(cage.cage_atoms) == 42
    assert cage.descriptor == "heptacyclo[31.3.1.1^3,7.1^9,13.1^15,19.1^21,25.1^27,31]"
    # a system beyond the PIN ceilings is recorded as a part that is never a PIN
    assert cage.descriptor in (get_provenance().get("non_pin_fragments") or ())


@pytest.mark.opsin_gate
def test_a_best_effort_recovery_inside_a_pin_request_keeps_the_pin_ceilings():
    # The fragment rescue builds a fresh best-effort engine inside a request of
    # any tier (decomposition.engine._name_fragment_t4_rescue_fresh). In a PIN
    # request it analyses no ring system beyond the PIN ceilings.
    from orthonym.decomposition.engine import _name_fragment_t4_rescue_fresh
    with jvm_slots(1, purpose="cage-caps-test"):
        assert _in_request(True, _name_fragment_t4_rescue_fresh, CALIX6) == CALIX6_NAME
        assert _in_request(False, _name_fragment_t4_rescue_fresh, CALIX6) is None
        assert _name_fragment_t4_rescue_fresh(CALIX6) is None


@pytest.mark.opsin_gate
def test_best_effort_names_the_calixarene_and_the_pin_tier_refuses():
    row = _row(CALIX6, "best-effort")
    assert row.get("name") == CALIX6_NAME, row.get("name")
    assert name_is_rt_exact(CALIX6_NAME, CALIX6)
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]
    pin = _row(CALIX6, "pin")
    assert pin["tier"] == "abstain" and is_failure_name(pin.get("name") or "unknown"), (
        pin.get("name"), pin["tier"])


@pytest.mark.opsin_gate
def test_a_pin_path_name_on_a_raised_cage_is_not_labelled_a_pin():
    row = _row(CYCLIC_PEPTIDE, "best-effort")
    name = row.get("name")
    assert name and "tetracyclo[38.2.1.1^23,26.1^33,36]pentatetraconta" in name, name
    assert name_is_rt_exact(name, CYCLIC_PEPTIDE)
    assert row["tier"] != "pin_verified" and row["is_pin"] is False, row["tier"]
    pin = _row(CYCLIC_PEPTIDE, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])


@pytest.mark.unit
def test_name_with_confidence_sets_the_request_tier_for_its_call():
    # name_with_confidence calls _name_impl without name's _budget_scope, so it
    # sets the request tier itself: the ceilings inside follow the engine's tier,
    # and nothing is left behind after the call.
    seen = []

    def fake_impl(self, smiles):
        seen.append((best_effort_request_ctx.get(), vbu.cage_caps()))
        return "methane"

    raised = (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Orthonym, "_name_impl", fake_impl)
        Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_with_confidence("C")
        Orthonym().name_with_confidence("C")
        # inside a request the call keeps that request's tier
        _in_request(False, Orthonym(style="pin", **_emit_tier_flags(
            "best-effort")).name_with_confidence, "C")
    assert seen == [(True, raised), (False, (40, 8)), (False, (40, 8))], seen
    assert best_effort_request_ctx.get() is None


#: (41-methyl-1-oxacyclohentetracontan-2-yl)acetic acid: a 41-membered ring,
#: beyond the PIN ceiling of 40 atoms, named as a decorated ring substituent.
OXACYCLE41 = "OC(=O)CC1" + "C" * 38 + "C(C)O1"


@pytest.mark.opsin_gate
def test_a_decorated_ring_substituent_beyond_the_pin_ceilings_is_not_labelled_a_pin():
    from orthonym.metrics.provenance import get_provenance
    row = _row(OXACYCLE41, "best-effort")
    name = row.get("name")
    assert name == "(41-methyl-1-oxacyclohentetracontan-2-yl)acetic acid", name
    assert name_is_rt_exact(name, OXACYCLE41)
    # recorded as not a PIN, as the bare ring substituent is
    assert "1-oxacyclohentetracontan-2-yl" in (get_provenance().get("non_pin_fragments") or ())
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row["tier"]
