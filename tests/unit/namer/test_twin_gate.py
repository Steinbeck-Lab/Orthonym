"""The twin branch of the validity gate: a PIN spelling OPSIN has no grammar for ships when its
twin spelling (``validation.twin_spellings``) reads to the input's full InChIKey, at the default
tier only, and every reader of the gate outcome knows the outcome."""
import pytest

from orthonym import Orthonym, namer
from orthonym.errors import is_failure_name
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.opsin_gate

SULFANO = ("1,4-dihydro-1,4-sulfanonaphthalene", "C12C=CC(C3=CC=CC=C13)S2")
AZANO_SMILES = "C12C=CC(C3=CC=CC=C13)N2"


def _outcome(name):
    row = pv.get_provenance()
    return pv.resolve_gate_outcome(row["gate_outcome"], row["gate_outcome_name"], name)


@pytest.mark.parametrize("name,smiles,rule", [
    SULFANO + ("bridge_sulfano",),
    ("(1R,4S)-1,4-dihydro-1,4-sulfanonaphthalene", "C1=C[C@@H]2c3ccccc3[C@H]1S2", "bridge_sulfano"),
    ("1,4-dihydro-1,4-azanonaphthalene", AZANO_SMILES, "bridge_azano"),
    ("9,10-dihydro-9,10-disulfanoanthracene", "C12c3ccccc3C(SS1)c1ccccc12", "bridge_disulfano"),
    ("(N-2H2)aniline", "[2H]N([2H])c1ccccc1", "isotope_letter_once"),
    ("ethan(2H)ol", "[2H]OCC", "isotope_count_one"),
    ("(1-2H1)ethan-1-(2H)ol", "CC([2H])O[2H]", "isotope_count_one"),
])
def test_gate_ships_the_pin_spelling_through_its_twin(name, smiles, rule):
    assert namer._final_opsin_validity_gate(name, smiles) == name
    out = _outcome(name)
    assert out == f"twin_verified:{rule}"
    assert pv.opsin_label_for_gate_outcome(out) == "verified_twin"
    assert pv.gate_token_for_gate_outcome(out) == "twin_round_trip"


def test_gate_refuses_a_twin_of_another_molecule():
    name, _ = SULFANO
    assert is_failure_name(namer._final_opsin_validity_gate(name, AZANO_SMILES))
    assert not _outcome(name).startswith(pv.GATE_OUTCOME_TWIN_PREFIX)


def test_the_wider_tier_keeps_be_strict():
    name, smiles = SULFANO
    assert is_failure_name(namer._final_opsin_validity_gate(name, smiles, general_fallback_tier=True))


def test_a_name_without_a_twin_rule_is_still_suppressed():
    assert is_failure_name(namer._final_opsin_validity_gate("1,4-dihydro-1,4-selanonaphthalene",
                                                           "C12C=CC(C3=CC=CC=C13)[Se]2"))


def test_a_twin_that_opsin_reads_as_ambiguous_is_refused(monkeypatch):
    name, smiles = SULFANO
    monkeypatch.setattr("orthonym.validation.twin_spellings.ambiguity_messages",
                        lambda n: ("Positioning of radical on: x",))
    assert is_failure_name(namer._final_opsin_validity_gate(name, smiles))


def test_every_reader_of_the_outcome_knows_the_twin():
    name, smiles = SULFANO
    namer._final_opsin_validity_gate(name, smiles)
    assert _outcome(name) == "twin_verified:bridge_sulfano"
    assert namer._self01_lookup(name) == (True, False, "")
    assert Orthonym()._exit_round_trip_check_rt(name, smiles) == name       # not withdrawn at the exit
    assert pv.opsin_label_for_gate_outcome("a-state-that-does-not-exist-yet") == "unverified"


def test_the_exit_withdraws_a_name_whose_outcome_is_not_a_passing_one():
    name, smiles = SULFANO
    pv.record_gate_outcome("a-state-that-does-not-exist-yet", name)
    assert is_failure_name(Orthonym()._exit_round_trip_check_rt(name, smiles))


def test_offer_rt_ok_accepts_its_own_twin_outcome_only():
    """OPSIN rejects the name itself, so the full-key check of an offer runs on the twin."""
    name, smiles = SULFANO
    namer._final_opsin_validity_gate(name, smiles)           # records twin_verified:bridge_sulfano for this string
    assert namer._offer_rt_ok(name, smiles) is True
    assert namer._offer_rt_ok(name, AZANO_SMILES) is False   # the twin is the twin of another molecule
    assert namer._offer_rt_ok("1,4-dihydro-1,4-azanonaphthalene", smiles) is False   # an outcome recorded for another string never counts


def test_tier_order_independence():
    """A call at one tier leaves no spelling for the next: the default tier spells the preselected bridge prefix
    and the best-effort tier the general one, in any order."""
    from tests.support.rt_assert import name_best_effort
    smiles = SULFANO[1]
    first = name_best_effort(smiles)["name"]
    default = Orthonym(style="pin").name_tiered(smiles)["name"]
    second = name_best_effort(smiles)["name"]
    assert first == second == "1,4-dihydro-1,4-epithionaphthalene"
    assert default == SULFANO[0]
