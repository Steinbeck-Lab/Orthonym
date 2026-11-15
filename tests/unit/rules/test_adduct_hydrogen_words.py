"""Hydrogen ions and isotopic dihydrogens as adduct components "Names for hydrogen atoms and ions", Table 8.1,
the Blue Book; nuclide descriptors,:43728; "Organic adducts",:3765).

OPSIN reads none of 'hydron', 'deuteron', '(2H2)dihydrogen', and reads a dihydrogen beside a radical as one
component; such a name ships through ``validation.hydrogen_word_read_back`` (each component read on its own) only when
it reproduces the drawn structure with every hydrogen an atom, because the standard InChIKey puts a hydron into its
protonation layer. Never a PIN,:4667: mixed adducts have none).
"""
import pytest
from rdkit import Chem

from orthonym.rules.adducts import HYDROGEN_ION_AND_ISOTOPE_NAMES, isotopes_only_in_hydrogen_components
from orthonym.validation import hydrogen_word_read_back as H
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


def test_every_word_has_its_structure():
    assert len(H.HYDROGEN_WORDS) == 14
    assert H.HYDROGEN_WORDS["hydron"] == "[H+]" and H.HYDROGEN_WORDS["deuteron"] == "[2H+]"
    assert H.HYDROGEN_WORDS["(2H2)dihydrogen"] == "[2H][2H]"
    for word in HYDROGEN_ION_AND_ISOTOPE_NAMES.values():
        assert word in H.HYDROGEN_WORDS


def test_the_component_table_keys_are_rdkit_canonical():
    for smi in HYDROGEN_ION_AND_ISOTOPE_NAMES:
        assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == smi


def test_reader_needs_an_adduct_name_with_a_hydrogen_word():
    assert H.read_back_with_hydrogen_words("hydron") is None
    assert H.read_back_with_hydrogen_words("ethane—methane (1/1)") is None
    assert H.read_back_with_hydrogen_words("benzene—hydron (1/1/1)") is None


def test_reader_reads_each_component():
    assert H.read_back_with_hydrogen_words("benzene—(2H2)dihydrogen (1/1)") == \
        Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1.[2H][2H]")))


def test_a_key_match_is_not_enough_for_a_hydron():
    # 'acetate—hydron (2/1)' and the drawing share no graph; acetic acid and acetate + hydron share one key
    assert H.hydrogen_word_round_trip("acetate—hydron (2/1)", "CC(=O)[O-].[H+]") is None
    assert H.hydrogen_word_round_trip("acetate—hydron (1/1)", "CC(=O)[O-].[H+].[H+]") is None
    assert H.hydrogen_word_round_trip("acetate—hydron (1/2)", "CC(=O)[O-].[H+].[H+]") == H.HYDROGEN_WORDS_RULE


def test_labels_only_in_hydrogen_components_skip_the_decorator():
    assert isotopes_only_in_hydrogen_components(Chem.MolFromSmiles("c1ccccc1.[2H][2H]"))
    assert not isotopes_only_in_hydrogen_components(Chem.MolFromSmiles("[2H]c1ccccc1.[2H][2H]"))
    assert not isotopes_only_in_hydrogen_components(Chem.MolFromSmiles("c1ccccc1.[HH]"))


@pytest.mark.parametrize("smiles, name", [
    ("c1ccccc1.[2H][2H]", "benzene—(2H2)dihydrogen (1/1)"),
    ("CC.CC.[3H][3H]", "ethane—(3H2)dihydrogen (2/1)"),
    ("[2H+].CC(=O)[O-]", "acetate—deuteron (1/1)"),
    ("CC(=O)[O-].[H+].[H+]", "acetate—hydron (1/2)"),
])
def test_hydrogen_word_adducts_ship_through_the_reader(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name
    assert r["tier"] == "systematic_verified" and not r["is_pin"]
    assert r["verified"] == "reconstructor" and r["gate_outcome"] == "reconstructed:hydrogen_words"


def test_a_labelled_organic_component_still_goes_to_the_decorator():
    r = name_best_effort("[2H]c1ccccc1.[2H][2H]")
    assert r["gate_outcome"] != "reconstructed:hydrogen_words"
