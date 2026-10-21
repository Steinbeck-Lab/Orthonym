"""UN lane A: the validity gate and the exit round-trip check read back OPSIN's own SMILES.

RDKit's canonical SMILES of OPSIN's SMILES can denote the MIRROR IMAGE of the structure OPSIN read
(RDKit 2026.03.5, CID101003382: OPSIN's string reads (R,R) for rdCIPLabeler, the canonical SMILES of
it reads (S,S), and so does the canonical SMILES of the input as written). The InChIKey is faithful:
the key of OPSIN's string is the key of the input, the key of the canonical form is the key of the
enantiomer. The headline metric (``eval/harness.py::_inchikey`` on ``opsin_smiles``) and the
independent read-back of the tests (``tests/support/rt_assert.py::name_is_rt_exact``) both read
OPSIN's string, so the gate must too. Reading the canonical form had two effects:

* at the general tiers the correct name of the diazatricyclododecadiene series (CID101003378/79/81/82)
  was withdrawn (``full_rt_unconfirmed_general_fallback``, ``exit_full_key_round_trip``): its
  canonical form is the mirror image of the input;
* the mirror-image name could be accepted: with the input written as RDKit's canonical SMILES (S,S for
  rdCIPLabeler) the name ``methyl (6R,12R)-...`` was shipped at the default tier (verdict
  'ok') and at the general tiers' read-back.

The default tier's verdict now reads OPSIN's own string as well (user decision 2026-10-21).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi, rdCIPLabeler

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.metrics import provenance as pv
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# CID101003382 (census 2), written as in PubChem: (R,R) for rdCIPLabeler
SMILES = "COC(=O)C1=CNC2C([C@@H]1C3=CC=CC=C3)(C4C2([C@@H](C(=CN4)C(=O)OC)C5=CC=CC=C5)C(=O)OC)C(=O)OC"
NAME = ("methyl (6R,12R)-1,7,11-tris(methoxycarbonyl)-6,12-diphenyl-3,9-diazatricyclo"
        "[6.4.0.0^2,7]dodeca-4,10-diene-5-carboxylate")
NAME_SS = NAME.replace("(6R,12R)", "(6S,12S)")
# OPSIN 2.9.0's own SMILES of NAME
OPSIN_WROTE = ("COC(=O)C12C3NC=C([C@H](C3(C2NC=C([C@H]1C1=CC=CC=C1)C(=O)OC)C(=O)OC)"
               "C1=CC=CC=C1)C(=O)OC")
# the input written as RDKit's canonical SMILES: the molecule of (S,S) for rdCIPLabeler
SS_INPUT = "COC(=O)C1=CNC2C(C(=O)OC)(C3NC=C(C(=O)OC)[C@H](c4ccccc4)C23C(=O)OC)[C@H]1c1ccccc1"
# the other rows of the series (every one read back exact by the independent OPSIN call)
SERIES = [
    "CCOC(=O)C1=CNC2C([C@@H]1C3=CC=CC=C3)(C4C2([C@@H](C(=CN4)C(=O)OCC)C5=CC=CC=C5)C(=O)OCC)C(=O)OCC",
    "CCOC(=O)C1=CN(C2C([C@@H]1C3=CC=CC=C3)(C4C2([C@@H](C(=CN4CC5=CC=CC=C5)C(=O)OCC)C6=CC=CC=C6)"
    "C(=O)OCC)C(=O)OCC)CC7=CC=CC=C7",
]


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _cip(smiles):
    """The CIP labels of the stereocentres, from RDKit's rdCIPLabeler: an implementation of the rules
    independent of the engine's (``centres``) and of InChI."""
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)
    return sorted(a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode"))


def _best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def _stub_oracle(monkeypatch, canonical, opsin):
    monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda name: canonical)
    monkeypatch.setattr(namer, "_validity_gate_name_to_opsin_smiles", lambda name: opsin)
    monkeypatch.setattr(namer, "_validity_gate_status", lambda name: "parsed")


def test_premise_rdkits_canonical_smiles_of_opsins_string_is_the_mirror_image():
    """The fact this module rests on. If RDKit stops showing it, the two reads agree and this fails."""
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(OPSIN_WROTE))
    assert _cip(SMILES) == _cip(OPSIN_WROTE) == ["R", "R"]
    assert _cip(canonical) == ["S", "S"], "the premise is gone: RDKit's canonical form is faithful again"
    assert canonical == SS_INPUT == Chem.MolToSmiles(Chem.MolFromSmiles(SMILES))
    # InChI is faithful: OPSIN's string has the key of the input, the canonical form that of the mirror
    assert _key(OPSIN_WROTE) == _key(SMILES) != _key(canonical)


def test_the_lookup_hands_out_the_string_opsin_wrote():
    assert namer._validity_gate_name_to_opsin_smiles(NAME) == OPSIN_WROTE
    assert namer._validity_gate_name_to_smiles(NAME) != OPSIN_WROTE        # the canonical form


def test_the_general_tier_gate_ships_a_name_opsin_reads_to_the_inputs_own_key():
    out = namer._final_opsin_validity_gate(NAME, SMILES, general_fallback_tier=True)
    assert out == NAME and not is_failure_name(out)


def test_the_exit_check_of_a_general_tier_keeps_it():
    pv.clear_provenance()
    assert _best_effort()._exit_round_trip_check_rt(NAME, SMILES) == NAME
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED


def test_shipped_name_round_trip_reads_opsins_string():
    assert namer._shipped_name_round_trip(NAME, SMILES) == "verified"


@pytest.mark.parametrize("smiles", [SMILES] + SERIES)
def test_best_effort_names_the_series_and_the_name_reads_back_exact(smiles):
    res = _best_effort().name_tiered(smiles)
    name = res["name"]
    assert name and res["tier"] != "abstain", res
    assert_full_rt(name, smiles)
    assert name_is_rt_exact(name, smiles)             # the independent OPSIN call of the tests


@pytest.mark.parametrize("general", [False, True], ids=["default_tier", "general_tier"])
def test_the_mirror_image_name_is_refused_and_its_enantiomers_name_ships(general):
    """The input is the (S,S) molecule (rdCIPLabeler, see the premise); the name '(6R,12R)-...' is read
    by OPSIN as the (R,R) molecule, a different molecule, and was shipped from the canonical form."""
    assert _cip(SS_INPUT) == ["S", "S"] and _cip(namer._validity_gate_name_to_opsin_smiles(NAME)) == ["R", "R"]
    assert is_failure_name(namer._final_opsin_validity_gate(NAME, SS_INPUT, general_fallback_tier=general))
    assert namer._final_opsin_validity_gate(NAME_SS, SS_INPUT, general_fallback_tier=general) == NAME_SS


def test_shipped_name_round_trip_refuses_the_mirror_image_name():
    assert namer._shipped_name_round_trip(NAME, SS_INPUT) == "failed"
    assert namer._shipped_name_round_trip(NAME_SS, SS_INPUT) == "verified"
    assert not name_is_rt_exact(NAME, SS_INPUT) and name_is_rt_exact(NAME_SS, SS_INPUT)    # the metric agrees


@pytest.mark.parametrize("general", [False, True], ids=["default_tier", "general_tier"])
def test_every_tier_hands_opsins_own_string_to_self01(monkeypatch, general):
    """What is handed: the string OPSIN wrote, at the default tier too. This replaces the
    assertion of 448f25700 that the default tier is handed RDKit's canonical form, which pinned the
    mirror-image hole above (user decision 2026-10-21). Rests on: the metric's definition (OPSIN's
    string keyed, ``eval/harness.py::_inchikey``); rdCIPLabeler, independent of the engine, reading the
    canonical form as the other enantiomer (the premise test); a mutation test (the lookup removed from
    the gate: this test and the refusal above fail)."""
    seen = []
    real = namer._self_consistency_decision

    def spy(name, smiles, opsin_smiles, stats, ignore_stereo=False):
        seen.append(opsin_smiles)
        return real(name, smiles, opsin_smiles, stats, ignore_stereo)

    monkeypatch.setattr(namer, "_self_consistency_decision", spy)
    namer._final_opsin_validity_gate(NAME, SMILES, general_fallback_tier=general)
    assert seen == [OPSIN_WROTE]


def test_a_name_is_refused_when_opsins_own_string_has_another_key(monkeypatch):
    """No wrong name ships: the key is the key of OPSIN's string, exactly. The canonical form here
    is the input (the old read-back accepted it); the string OPSIN wrote is another molecule."""
    _stub_oracle(monkeypatch, SMILES, "CCO")
    assert is_failure_name(namer._final_opsin_validity_gate("some name", SMILES, general_fallback_tier=True))
    assert namer._shipped_name_round_trip("some name", SMILES) == "failed"


@pytest.mark.parametrize("general", [False, True], ids=["default_tier", "general_tier"])
def test_a_lookup_without_opsins_string_is_failed(monkeypatch, general):
    """The canonical form exists but OPSIN's string cannot be had: nothing is verified, so the name is
    withdrawn at every tier and the round trip fails (the canonical form is never a stand-in)."""
    _stub_oracle(monkeypatch, "CCO", None)
    assert is_failure_name(namer._final_opsin_validity_gate("some name", "CCO", general_fallback_tier=general))
    assert namer._shipped_name_round_trip("some name", "CCO") == "failed"


def test_a_key_that_cannot_be_computed_stays_closed(monkeypatch):
    """OPSIN's string that RDKit cannot key abstains, as an unkeyable canonical form always did."""
    _stub_oracle(monkeypatch, "CCO", "not a smiles")
    assert is_failure_name(namer._final_opsin_validity_gate("some name", "CCO", general_fallback_tier=True))
