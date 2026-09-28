"""Claims conformance, job claims-code-a (2026-09-27): 0-wrong and honest labels.

Public claims audit ids critic-1 / R63, R19, R68 / critic-2 / X03 (the audit is
`internal notes`):

1. The stereo-layer carve-out of the validity gate (a name OPSIN rejects only for its
   stereo layer ships after judged the stereo-STRIPPED parse) shipped
   descriptors nobody checked, some on the wrong atoms: '(7R,8S)-7-chloro-1-(3,5-
   dihydroxyphenyl)-2-methylundecyl acetate' (the input's stereocentres are C1 and C2)
   and '(2S,3R)-3-oxoestr-4-en-17β-yl 3-phenylpropanoate' (C2 is a CH2, C3 the ketone
   carbon), both at pin_verified. Every descriptor is now checked against the input's
   CIP label at its locant (``namer._stereo_descriptors_verified``); on a mismatch the
   gate ships the recomposed name whose full key round-trips, else the stereo-free name
   when IT round-trips (labelled below pin), else nothing.
2. 'inconclusive' (a skeleton key could not be computed) shipped the name
   unchecked; it now ships only after a full-key round trip passes.
3. Retained trivial names were labelled systematic_verified with no check.
4. pin_verified / systematic_verified only for a name that passed a full round trip
   (``verified`` 'opsin') or the documented identity table ('identity'); the grammar
   carve-outs and the constitution-only stereo branch keep shipping at the default tier
   one label down, and ``verified`` says 'opsin_constitution' for the latter. Gate
   outcome strings are unchanged.

Every shipped name asserted here is also read back by a FRESH OPSIN call that does not
go through the engine (``tests.support.rt_assert._independent_parse``) and compared by
full InChIKey.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym import namer
from orthonym.metrics import provenance as pv
from tests.support.rt_assert import _independent_parse

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


def _independent_full_key(name):
    """Full InChIKey of what a fresh OPSIN call reads for ``name``; '' if nothing."""
    return _key(_independent_parse(name))


def _pin_row(smiles):
    return Orthonym().name_tiered(smiles)


# ---------------------------------------------------------------------------
# 1. stereodescriptors of the stereo-layer carve-out are checked
# ---------------------------------------------------------------------------

CHLORO = "CCCCC(Cl)CCCC[C@H](C)[C@@H](OC(C)=O)c1cc(O)cc(O)c1"
CHLORO_WRONG = "(7R,8S)-7-chloro-1-(3,5-dihydroxyphenyl)-2-methylundecyl acetate"
CHLORO_RIGHT = "(1R,2S)-7-chloro-1-(3,5-dihydroxyphenyl)-2-methylundecyl acetate"

NANDROLONE_PP = ("C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@H]43)[C@@H]1CC[C@@H]2"
                 "OC(=O)CCc1ccccc1")
NANDROLONE_PP_WRONG = "(2S,3R)-3-oxoestr-4-en-17β-yl 3-phenylpropanoate"
NANDROLONE_PP_RIGHT = "3-oxoestr-4-en-17β-yl 3-phenylpropanoate"

# milestone1500: the same class found by the census of every constitution-only name
# (descriptors on the two ring oxygens 2 and 9, and one of six centres missing).
SPIRO_N_OXIDE = ("C[C@@H]1C[C@@]2(O[C@H]2C)C(=O)O[C@@H]2CC[N+]3([O-])CC=C(COC(=O)"
                 "[C@]1(C)O)[C@H]23")
SPIRO_N_OXIDE_WRONG = ("(2R,3R,8R,9R,11S)-7-hydroxy-3',6,7-trimethyl-3,8-dioxospiro"
                       "[2,9-dioxa-14-azatricyclo[9.5.1.0^14,17]heptadec-11-ene-4,2'-"
                       "oxirane] N-oxide")


def test_wrong_locants_are_recomposed_to_the_round_tripping_name():
    assert _independent_full_key(CHLORO_WRONG) == ""  # OPSIN cannot place 8S on a CH2
    row = _pin_row(CHLORO)
    assert row["name"] == CHLORO_RIGHT, row
    assert _independent_full_key(row["name"]) == _key(CHLORO)
    assert row["gate_outcome"] == pv.GATE_OUTCOME_STEREO_RECOMPOSED, row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin", row


def test_wrong_descriptors_on_a_stereoparent_fall_back_to_its_full_key_name():
    assert _independent_full_key(NANDROLONE_PP_WRONG) == ""
    row = _pin_row(NANDROLONE_PP)
    assert row["name"] == NANDROLONE_PP_RIGHT, row
    assert _independent_full_key(row["name"]) == _key(NANDROLONE_PP)
    assert row["opsin"] == "verified" and row["verified"] == "opsin", row
    # not what the PIN path built, so labelled below pin
    assert row["tier"] not in ("pin_verified", "pin_unverified"), row
    assert row["is_pin"] is False, row


def test_census_row_with_descriptors_on_ring_oxygens_is_repaired():
    assert _independent_full_key(SPIRO_N_OXIDE_WRONG) == ""
    row = _pin_row(SPIRO_N_OXIDE)
    assert row["name"] != SPIRO_N_OXIDE_WRONG, row
    assert row["name"] and _independent_full_key(row["name"]) == _key(SPIRO_N_OXIDE), row


# Correct names OPSIN rejects only for the stereo grammar it lacks (lowercase r/s,
# a descriptor restating a retained parent's configuration, a locant-free code) keep
# shipping at the default tier, constitution-only, now one label down.
CORRECT_CARVEOUT_NAMES = [
    ("O[C@H]1CC[C@@H](O)CC1", "(1s,4s)-cyclohexane-1,4-diol"),
    ("CN1[C@@H]2CC[C@H]1C[C@H](O)C2", "(1R,3s,5S)-tropan-3-ol"),
    ("Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@H]1Cl",
     "(1R,2R,3R,4R,5S,6S)-1,2,3,4,5,6-hexachlorocyclohexane"),
]


@pytest.mark.parametrize("smiles,name", CORRECT_CARVEOUT_NAMES)
def test_correct_carveout_names_still_ship_labelled_constitution_only(smiles, name):
    row = _pin_row(smiles)
    assert row["name"] == name, row
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY, row
    assert row["opsin"] == "verified_constitution_only", row
    assert row["verified"] == "opsin_constitution", row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row
    # OPSIN reads the stereo-stripped form to the input's constitution
    from orthonym.rules.stereochemistry import strip_stereo
    stripped = strip_stereo(name)
    assert _independent_full_key(stripped).split("-")[0] == _key(smiles).split("-")[0]


@pytest.mark.parametrize("name,smiles,ok", [
    # the descriptor check itself, with negative controls
    ("(1s,4s)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", True),
    ("(1r,4r)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", False),   # trans for cis
    ("(1s,4r)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", False),   # one code wrong
    ("(1r,3r)-cyclobutane-1,3-diol", "O[C@@H]1C[C@@H](O)C1", True),
    ("(1r,3r)-cyclobutane-1,3-diol", "O[C@@H]1C[C@H](O)C1", False),
    (CHLORO_WRONG, CHLORO, False),
    (CHLORO_RIGHT, CHLORO, True),
    (NANDROLONE_PP_WRONG, NANDROLONE_PP, False),
    ("(1R,3s,5S)-tropan-3-ol", "CN1[C@@H]2CC[C@H]1C[C@H](O)C2", True),
    ("(1R,3r,5S)-tropan-3-ol", "CN1[C@@H]2CC[C@H]1C[C@H](O)C2", False),
    ("(R)-4-nitrobenzene-1-sulfinic acid", "O=[N+]([O-])c1ccc([S@](=O)O)cc1", True),
    ("(S)-4-nitrobenzene-1-sulfinic acid", "O=[N+]([O-])c1ccc([S@](=O)O)cc1", False),
    ("rel-(1R,4R)-cyclohexane-1,4-diol", "O[C@H]1CC[C@@H](O)CC1", False),  # not checkable
])
def test_stereo_descriptor_check(name, smiles, ok):
    from orthonym.rules.stereochemistry import strip_stereo
    assert namer._stereo_descriptors_verified(name, smiles, strip_stereo(name)) is ok


# ---------------------------------------------------------------------------
# 2. 'inconclusive' is not a pass
# ---------------------------------------------------------------------------

def test_inconclusive_self01_ships_only_after_a_full_key_round_trip(monkeypatch):
    monkeypatch.setattr(namer, "_self_consistency_verdict",
                        lambda *a, **k: "inconclusive")
    # a name whose full key round-trips ships, recorded as the full-key proof
    assert namer._final_opsin_validity_gate("propan-2-ol", "CC(C)O") == "propan-2-ol"
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED
    assert _independent_full_key("propan-2-ol") == _key("CC(C)O")
    # one that does not (a different molecule the compare could not see) is withdrawn
    shipped = namer._final_opsin_validity_gate("propan-1-ol", "CC(C)O")
    assert shipped != "propan-1-ol" and namer.is_failure_name(shipped)
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_SUPPRESSED
    assert _independent_full_key("propan-1-ol") != _key("CC(C)O")


# ---------------------------------------------------------------------------
# 3. retained trivial names: labelled verified only after their round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,name", [
    ("N=C=S", "isothiocyanic acid"),
    ("[SH-]", "bisulfide"),
])
def test_trivial_name_is_round_tripped_and_labelled_verified(smiles, name):
    row = _pin_row(smiles)
    assert row["name"] == name and row["source"] == "trivial_retained", row
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01, row  # it was 'not_run'
    assert row["tier"] == "systematic_verified" and row["verified"] == "opsin", row
    assert _independent_full_key(name) == _key(smiles)


def test_unverified_trivial_name_is_not_labelled_systematic_verified():
    # with no check at all (the gate disabled) the same name is labelled below
    # systematic_verified
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered("N=C=S")
    assert row["name"] == "isothiocyanic acid", row
    assert row["tier"] == "best_effort" and row["verified"] == "unverified", row


# ---------------------------------------------------------------------------
# 4. labels: 'verified' tiers only for a round trip or the identity table
# ---------------------------------------------------------------------------

def test_grammar_carveout_ships_at_pin_unverified_with_its_gate_string():
    row = _pin_row("CSO")
    assert row["name"] == "methane-SO-thioperoxol", row
    assert row["gate_outcome"] == "carveout:thioperoxol", row  # gate string unchanged
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row
    assert row["verified"] == "unverified" and row["opsin"] == "unverified", row
    assert _independent_full_key(row["name"]) == ""  # OPSIN has no grammar for it


def test_round_tripping_pin_keeps_pin_verified():
    row = _pin_row("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert _independent_full_key(row["name"]) == _key("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")


def test_identity_table_name_keeps_pin_verified():
    from tests.unit.rules.test_d1_coordination_v36 import FIXTURES
    heme = next(r["smiles"] for r in FIXTURES
                if r["name"] == "(protoporphyrinato)iron(II)")
    row = _pin_row(heme)
    assert row["name"] == "(protoporphyrinato)iron(II)", row
    assert row["verified"] == "identity", row
    assert row["tier"] == "pin_verified", row
