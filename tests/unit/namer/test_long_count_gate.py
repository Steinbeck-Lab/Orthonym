"""The validity gate ships a name with a count word above 50 when the reconstructor builds the structure from its
text (``validation.long_count_read_back``): OPSIN 2.9.0 has no numerical term above 50, and until the check the
gate withdrew each of these names (``opsin_unparseable``).

The label is systematic_verified at best, ``verified`` is ``reconstructor``: the reconstructor proves the
constitution of the name and never its spelling, and the default (PIN-only) tier declines the name as it declines
every name that is not a verified PIN. Basic multiplicative prefixes (the Blue Book, Table 1.4),
 Derivation of basic numerical terms (:2811), Hydrocarbons (:7984, 'heptacontane (PIN)').
"""
import pytest

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.opsin_gate


def _locants(first, last):
    return ",".join(str(i) for i in range(first, last + 1))


PERFLUORO = ("tetrapentacontafluorohexacosane", "C(F)(F)(F)" + "C(F)(F)" * 24 + "C(F)(F)F")
CUMULENE = ("pentapentaconta-" + _locants(1, 54) + "-tetrapentacontaene", "=".join(["C"] * 55))
BRANCHED = ("79-methyl-78-(propan-2-yl)octaconta-" + _locants(1, 77) + "-heptaheptacontaene",
            "=".join(["C"] * 77) + "=C(C(C)C)C(C)C")
POLYYNE = ("tetracosahecta-" + ",".join(str(i) for i in range(2, 123, 2)) + "-henhexacontayne",
           "C" + "C#C" * 61 + "C")
ROWS = [PERFLUORO, CUMULENE, BRANCHED, POLYYNE]


def _outcome(name):
    row = pv.get_provenance()
    return pv.resolve_gate_outcome(row["gate_outcome"], row["gate_outcome_name"], name)


@pytest.mark.parametrize("general", [False, True], ids=["pin_tier_gate", "general_tier_gate"])
@pytest.mark.parametrize("name,smiles", ROWS, ids=[r[0][:30] for r in ROWS])
def test_the_gate_ships_a_name_the_reconstructor_reads(name, smiles, general):
    assert namer._final_opsin_validity_gate(name, smiles, general_fallback_tier=general) == name
    out = _outcome(name)
    assert out == "reconstructed:long_count"
    assert pv.opsin_label_for_gate_outcome(out) == "verified_reconstructor"
    assert pv.gate_token_for_gate_outcome(out) == "name_reconstruction"


def _corrupt(name):
    """A locant, a count and a halogen of ``name`` changed, one at a time: three names of another molecule."""
    out = [name.replace("pentaconta", "hexaconta", 1), name.replace("-1,2,", "-1,3,", 1),
           name.replace("fluoro", "chloro", 1), name.replace("henhexaconta", "hexaconta", 1)]
    return [n for n in out if n != name]


@pytest.mark.parametrize("general", [False, True], ids=["pin_tier_gate", "general_tier_gate"])
@pytest.mark.parametrize("name,smiles", ROWS, ids=[r[0][:30] for r in ROWS])
def test_the_gate_withdraws_a_corrupted_name(name, smiles, general):
    wrong = _corrupt(name)
    assert wrong
    for corrupted in wrong:
        assert is_failure_name(namer._final_opsin_validity_gate(corrupted, smiles, general_fallback_tier=general))
        assert not _outcome(corrupted).startswith(pv.GATE_OUTCOME_RECONSTRUCTED_PREFIX)


def test_the_gate_withdraws_the_name_of_another_structure():
    name, _ = PERFLUORO
    assert is_failure_name(namer._final_opsin_validity_gate(name, "C(F)(F)(F)" + "C(F)(F)" * 23 + "C(F)(F)F"))
    assert is_failure_name(namer._final_opsin_validity_gate(name, "C" * 26))


def test_the_gate_withdraws_a_name_for_an_input_with_stereo():
    # 2E stereo the name does not state
    assert is_failure_name(namer._final_opsin_validity_gate("henpentacont-2-ene", "C/C=C/" + "C" * 48))


def test_every_reader_of_the_outcome_knows_the_reconstruction():
    name, smiles = PERFLUORO
    namer._final_opsin_validity_gate(name, smiles, general_fallback_tier=True)
    assert _outcome(name) == "reconstructed:long_count"
    assert namer._self01_lookup(name) == (True, False, "")
    assert Orthonym(**_emit_tier_flags("best-effort"))._exit_round_trip_check_rt(name, smiles) == name
    assert Orthonym()._exit_round_trip_check_rt(name, smiles) == name           # the default tier's exit too
    assert namer._offer_rt_ok(name, smiles) is True
    assert namer._offer_rt_ok(name, "C" * 26) is False                          # another structure never counts
    pv.record_gate_outcome("reconstructed:a-rule-that-does-not-exist", name)
    assert namer._offer_rt_ok(name, smiles) is True                             # the reader decides, not the rule id
    pv.record_gate_outcome("a-state-that-does-not-exist-yet", name)
    assert is_failure_name(Orthonym(**_emit_tier_flags("best-effort"))._exit_round_trip_check_rt(name, smiles))


@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
@pytest.mark.parametrize("name,smiles", ROWS, ids=[r[0][:30] for r in ROWS])
def test_the_wider_tiers_return_the_name_as_systematic_verified(name, smiles, tier):
    row = Orthonym(**_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert row["verified"] == "reconstructor" and row["opsin"] == "verified_reconstructor"
    assert row["gate_outcome"] == "reconstructed:long_count"
    assert row["gates_passed"] == ["name_reconstruction"]


@pytest.mark.parametrize("name,smiles", ROWS, ids=[r[0][:30] for r in ROWS])
def test_the_default_pin_tier_declines_the_name(name, smiles):
    """The reconstructor proves the structure of the name, not that the spelling is the preferred one: the PIN-only
    tier names no molecule it has not verified as a PIN."""
    row = Orthonym(**_emit_tier_flags("pin")).name_tiered(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN"
    assert not Orthonym(**_emit_tier_flags("pin")).name(smiles, on_failure="empty")


def test_the_label_is_the_readers_and_no_pin_tier_claims_it():
    assert pv.pin_path_tier("verified_reconstructor") == ("systematic_verified", False)
    assert pv.pin_path_tier("verified") == ("pin_verified", True)
    assert "reconstructor" not in pv.VERIFIED_FIELD_FOR_PIN_TIER
    assert "reconstructor" in pv.VERIFIED_FIELD_FOR_TIER
    assert pv.opsin_label_for_gate_outcome("reconstructed") == "unverified"      # no rule id: not an outcome
    assert pv.opsin_label_for_gate_outcome("reconstructor:long_count") == "unverified"


# the grammar of the reader below 51, where OPSIN reads the same names: both must build the same structure
OPSIN_READS = [
    "pentacontane", "hexacosane", "hexacosa-1,2,3-triene", "penta-1,2,3,4-tetraene", "heptacos-1-ene",
    "tetracosa-2,4-diyne", "hexacos-3-yne", "tetradecafluorohexane",
    "1,1,1,2,2,3,3,4,4,5,5,6,6,6-tetradecafluorohexane", "1,1,1-trifluorohexacosane",
    "15,28-diiodo-1,1,1-trifluorodotetracontane", "26-methyl-1,1,1-trifluoroheptacosane",
    "3-(propan-2-yl)-2,2-dimethylundecane", "dotriaconta-1,31-diene", "henicosa-1,3,5-triene",
    "pentadeca-" + _locants(1, 14) + "-tetradecaene",
]


@pytest.mark.parametrize("name", OPSIN_READS)
def test_the_reader_builds_what_opsin_builds_for_a_name_opsin_reads(name):
    from rdkit import Chem
    from tests.support.rt_assert import _independent_parse
    from orthonym.validation import reconstruct
    from orthonym.validation.long_count_read_back import name_facts_from_text
    facts = name_facts_from_text(name, 0)
    assert facts is not None
    read = _independent_parse(name)
    assert read
    assert reconstruct._normalize(reconstruct._build_from_facts(facts)) == reconstruct._normalize(
        Chem.MolFromSmiles(read))
