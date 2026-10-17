"""The preselected bridge prefixes at the default tier, the Blue Book): 'sulfano'
(-S-), 'disulfano' (-S-S-) and 'azano' (-NH-). OPSIN 2.9.0 has no grammar for them, so the default tier
ships the name through its twin spelling ('epithio', 'epidithio', 'epimino', ``validation.twin_spellings``);
a wider tier spells the general prefixes OPSIN reads."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _canon, _independent_parse, assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (the Blue Book-14110): the table rows '-S- sulfano (preselected prefix)' (:14103),
# '-S-S- disulfano (preselected prefix)' (:14105) and '-NH- azano (preselected prefix)' (:14110); the
# PIN example '1,4-dihydro-1,4-sulfanonaphthalene (PIN)' under CYCLIC ETHERS, SULFIDES,
# SELENIDES AND TELLURIDES (:28063).
PIN_ROWS = [
    ("C12C=CC(C3=CC=CC=C13)S2", "1,4-dihydro-1,4-sulfanonaphthalene"),
    ("C1=C[C@@H]2c3ccccc3[C@H]1S2", "(1R,4S)-1,4-dihydro-1,4-sulfanonaphthalene"),
    ("c1ccc2c(c1)C1CCC2S1", "1,2,3,4-tetrahydro-1,4-sulfanonaphthalene"),
    ("Clc1ccc2c(c1)C1C=CC2S1", "6-chloro-1,4-dihydro-1,4-sulfanonaphthalene"),
    ("O=C(O)C1=CC2SC1c1ccccc12", "1,4-dihydro-1,4-sulfanonaphthalene-2-carboxylic acid"),
    ("O=C(O)CC12C=CC(S1)c1ccccc12", "(1,4-sulfanonaphthalen-1(4H)-yl)acetic acid"),
    ("C12C=CC(C3=CC=CC=C13)N2", "1,4-dihydro-1,4-azanonaphthalene"),
    ("CN1C2C=CC1c1ccccc12", "9-methyl-1,4-dihydro-1,4-azanonaphthalene"),
    ("c1ccc2c(c1)C1CCC2N1", "1,2,3,4-tetrahydro-1,4-azanonaphthalene"),
    ("C12c3ccccc3C(SS1)c1ccccc12", "9,10-dihydro-9,10-disulfanoanthracene"),
    # the checks a builder makes by reading its own name read the twin: the tautomer of a drawing
    # with a ring nitrogen next to a hydro position, the tautomer of an N-bridge prefix (a free
    # valence), the stereo of a substituent outside the ring system
    ("OCCN1C2C=CC1c1ccccc12", "2-(1,4-dihydro-1,4-azanonaphthalen-9-yl)ethan-1-ol"),
    ("O=C(O)CC12C=CC(N1)c1ccccc12", "(1,4-azanonaphthalen-1(4H)-yl)acetic acid"),
    ("C[C@H](Cl)C1=CC2SC1c1ccccc12", "2-[(1S)-1-chloroethyl]-1,4-dihydro-1,4-sulfanonaphthalene"),
    # the component and cation checks of the adduct and salt routes read the twin of a component name
    # adducts; cations: the bridge atom is 9, 'Bridge atoms are numbered
    # continuing from the highest locant of the fused ring system', so the 'ium' locant is 9)
    ("OC(=O)C1=CC2NC1c1ccccc12.Cl", "1,4-dihydro-1,4-azanonaphthalene-2-carboxylic acid\u2014hydrogen chloride (1/1)"),
    ("CN1C2C=CC1c1ccccc12.Cl", "9-methyl-1,4-dihydro-1,4-azanonaphthalene\u2014hydrogen chloride (1/1)"),
    ("C1=CC2SC1c1ccccc12.C1=CC2NC1c1ccccc12",
     "1,4-dihydro-1,4-azanonaphthalene\u20141,4-dihydro-1,4-sulfanonaphthalene (1/1)"),
    ("C[NH+]1C2C=CC1c1ccccc12", "9-methyl-1,4-dihydro-1,4-azanonaphthalen-9-ium"),
    ("C[NH+]1C2C=CC1c1ccccc12.[Cl-]", "9-methyl-1,4-dihydro-1,4-azanonaphthalen-9-ium chloride"),
    ("C[NH+]1C2CCC1c1ccccc12.[Br-]", "9-methyl-1,2,3,4-tetrahydro-1,4-azanonaphthalen-9-ium bromide"),
]
TWIN_ROWS = {
    "1,4-dihydro-1,4-sulfanonaphthalene": "1,4-dihydro-1,4-epithionaphthalene",
    "(1R,4S)-1,4-dihydro-1,4-sulfanonaphthalene": "(1R,4S)-1,4-dihydro-1,4-epithionaphthalene",
    "1,2,3,4-tetrahydro-1,4-sulfanonaphthalene": "1,2,3,4-tetrahydro-1,4-epithionaphthalene",
    "6-chloro-1,4-dihydro-1,4-sulfanonaphthalene": "6-chloro-1,4-dihydro-1,4-epithionaphthalene",
    "1,4-dihydro-1,4-sulfanonaphthalene-2-carboxylic acid":
        "1,4-dihydro-1,4-epithionaphthalene-2-carboxylic acid",
    "(1,4-sulfanonaphthalen-1(4H)-yl)acetic acid": "(1,4-epithionaphthalen-1(4H)-yl)acetic acid",
    "1,4-dihydro-1,4-azanonaphthalene": "1,4-dihydro-1,4-epiminonaphthalene",
    "9-methyl-1,4-dihydro-1,4-azanonaphthalene": "9-methyl-1,4-dihydro-1,4-epiminonaphthalene",
    "1,2,3,4-tetrahydro-1,4-azanonaphthalene": "1,2,3,4-tetrahydro-1,4-epiminonaphthalene",
    "9,10-dihydro-9,10-disulfanoanthracene": "9,10-dihydro-9,10-epidithioanthracene",
    "2-(1,4-dihydro-1,4-azanonaphthalen-9-yl)ethan-1-ol": "2-(1,4-dihydro-1,4-epiminonaphthalen-9-yl)ethan-1-ol",
    "(1,4-azanonaphthalen-1(4H)-yl)acetic acid": "(1,4-epiminonaphthalen-1(4H)-yl)acetic acid",
    "2-[(1S)-1-chloroethyl]-1,4-dihydro-1,4-sulfanonaphthalene":
        "2-[(1S)-1-chloroethyl]-1,4-dihydro-1,4-epithionaphthalene",
    "1,4-dihydro-1,4-azanonaphthalene-2-carboxylic acid\u2014hydrogen chloride (1/1)":
        "1,4-dihydro-1,4-epiminonaphthalene-2-carboxylic acid\u2014hydrogen chloride (1/1)",
    "9-methyl-1,4-dihydro-1,4-azanonaphthalene\u2014hydrogen chloride (1/1)":
        "9-methyl-1,4-dihydro-1,4-epiminonaphthalene\u2014hydrogen chloride (1/1)",
    "1,4-dihydro-1,4-azanonaphthalene\u20141,4-dihydro-1,4-sulfanonaphthalene (1/1)":
        "1,4-dihydro-1,4-epiminonaphthalene\u20141,4-dihydro-1,4-epithionaphthalene (1/1)",
    "9-methyl-1,4-dihydro-1,4-azanonaphthalen-9-ium": "9-methyl-1,4-dihydro-1,4-epiminonaphthalen-9-ium",
    "9-methyl-1,4-dihydro-1,4-azanonaphthalen-9-ium chloride":
        "9-methyl-1,4-dihydro-1,4-epiminonaphthalen-9-ium chloride",
    "9-methyl-1,2,3,4-tetrahydro-1,4-azanonaphthalen-9-ium bromide":
        "9-methyl-1,2,3,4-tetrahydro-1,4-epiminonaphthalen-9-ium bromide",
}
DISULFANO = "C12c3ccccc3C(SS1)c1ccccc12"


@pytest.mark.parametrize("smiles,name", PIN_ROWS)
def test_default_tier_gives_the_preselected_prefix(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin" and row["opsin"] == "verified_twin", row
    assert row["gate_outcome"].startswith("twin_verified:") and row["gates_passed"] == ["twin_round_trip"], row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles,name", PIN_ROWS)
def test_the_twin_of_the_default_name_is_the_structure_of_the_input(smiles, name):
    """An independent OPSIN read of the twin gives the input; OPSIN rejects the name itself."""
    twin = TWIN_ROWS[name]
    assert _independent_parse(name, smiles) is None
    assert _canon(_independent_parse(twin, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles,name", PIN_ROWS)
def test_best_effort_keeps_the_general_prefix(smiles, name):
    twin = TWIN_ROWS[name]
    row = name_best_effort(smiles)
    assert row["name"] == twin and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(twin, smiles)


def test_the_selenium_bridge_has_no_twin_and_keeps_its_behaviour():
    # OPSIN reads 'episeleno' as a [SeH2] bridge, another structure: no twin rule, no preselected spelling
    smiles = "C12C=CC(C3=CC=CC=C13)[Se]2"
    assert default_tier_row(smiles)["limit_code"] == "UNNAMEABLE"
    assert "selenatricyclo" in name_best_effort(smiles)["name"]


def test_multiplied_one_atom_bridges_stay_general():
    # The prefixes 'bis', 'tris', 'tetrakis', etc. are also used to avoid ambiguity (the Blue Book),
    # item (a) (:7136): bis(sulfanyl) "defines two -SH groups, see; whereas disulfanyl defines the
    # -SSH group" (:7140). Applied to the bridge prefix by analogy: 'disulfano' names the -S-S- bridge.
    smiles = "C12SC(C=C1)c1c3SC(C=C3)c(c12)"
    assert default_tier_row(smiles)["limit_code"] == "NO_VERIFIED_PIN"
    assert name_best_effort(smiles)["name"] == "4,8-dihydro-1H-1,4:5,8-diepithiobenzo[7]annulene"


def test_the_two_sulfur_bridge_has_a_best_effort_name():
    row = name_best_effort(DISULFANO)
    assert row["name"] == "9,10-dihydro-9,10-epidithioanthracene" and row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False
    assert_full_rt(row["name"], DISULFANO)


@pytest.mark.parametrize("smiles,name", [
    ("C12C=CC(C3=CC=CC=C13)O2", "1,4-dihydro-1,4-epoxynaphthalene"),
    ("C12c3ccccc3C(O1)c1ccccc12", "9,10-dihydro-9,10-epoxyanthracene"),
    ("C12c3ccccc3C(OO1)c1ccccc12", "9,10-dihydro-9,10-epidioxyanthracene"),
])
def test_epoxy_controls_are_read_by_opsin_and_do_not_use_the_twin(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["verified"] == "opsin" and row["tier"] == "pin_verified", row



def test_a_default_flag_namer_nested_in_a_wider_tier_call_spells_the_general_prefix():
    """The cation of a salt is named by a namer built with the default flags inside the best-effort
    call; the PIN spelling stays off there, so the name is one OPSIN reads."""
    smiles = "C[NH+]1C2C=CC1c1ccccc12.[Cl-]"
    row = name_best_effort(smiles)
    assert row["name"] == "9-methyl-1,4-dihydro-1,4-epiminonaphthalen-9-ium chloride", row
    assert row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


def test_the_wider_tier_flag_is_reset_when_the_call_ends():
    from orthonym.validation.twin_spellings import pin_spellings_on
    name_best_effort("C12C=CC(C3=CC=CC=C13)S2")
    assert pin_spellings_on() is True


def test_the_cation_charge_veto_reads_the_twin():
    """A neutral name offered for a cation is vetoed whether it is spelled with the general prefix
    (read by OPSIN) or the preselected one (read through its twin)."""
    from rdkit import Chem

    from orthonym.rules.ions import _validate_cation_name
    cation = Chem.MolFromSmiles("C[NH+]1C2C=CC1c1ccccc12")
    for neutral in ("9-methyl-1,4-dihydro-1,4-epiminonaphthalene", "9-methyl-1,4-dihydro-1,4-azanonaphthalene"):
        assert _validate_cation_name(cation, neutral) == ""
    assert _validate_cation_name(cation, "9-methyl-1,4-dihydro-1,4-azanonaphthalen-9-ium") != ""
