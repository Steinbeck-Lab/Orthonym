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


# OPSIN reads an adduct name as a whole: it balances the charges of its components and reads some words otherwise
# than alone. The reader reads the components other than the neutral dihydrogens as one name, so a name ships only for
# the drawing a reader of the whole name gives (2026-11-18: 23 PubChem mixtures had shipped 'yttrium' for a neutral
# Y beside a carbanion, which OPSIN main reads as Y+).
@pytest.mark.parametrize("name, read", [
    ("acetate—dihydrogen (1/1)", "CC(=O)O.[H][H]"),                             # the acid atom neutralised
    ("methanide—palladium—dihydrogen (1/1/1)", "[CH3-].[Pd+].[H][H]"),          # the charge balanced onto Pd
    ("methanide—palladium(0)—dihydrogen (1/1/1)", "[CH3-].[Pd].[H][H]"),        # the oxidation number zero keeps it
    ("ethanol—dioxygen—dihydrogen (1/1/1)", "CCO.[O].[O].[H][H]"),              # 'dioxygen' inside an adduct
    ("benzenesulfinate—ethanol—dihydrogen (1/1/1)", "CCO.OS(=O)c1ccccc1.[H][H]"),  # an acid anion with no cation
])
def test_the_reader_reads_the_adduct_as_a_whole(name, read):
    assert H.read_back_with_hydrogen_words(name) == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(read)))


@pytest.mark.parametrize("name, drawn", [
    ("ethanol—dioxygen—hydron (1/1/1)", "[H+].CCO.O=O"),     # 'dioxygen' is two atoms inside an adduct
    ("yttrium—hydron (1/1)", "[Y].[H+]"),                    # the bare element takes its typical charge
])
def test_a_hydron_name_reads_as_opsin_reads_the_whole(name, drawn):
    assert H.read_back_with_hydrogen_words(name) is not None
    assert H.hydrogen_word_round_trip(name, drawn) is None


@pytest.mark.parametrize("name, read", [
    ("methanide—palladium—hydron (1/1/1)", "[CH3-].[Pd].[H+]"),        # the hydron balances: palladium stays neutral
    ("ethanol—nitrate—hydron (1/1/1)", "CCO.[O-][N+](=O)[O-].[H+]"),    # the whole reading protonates; same graph
    ("N,N,N-trimethylmethanaminium—dihydrogen (1/1)", "C[N+](C)(C)C.[H][H]"),   # a lone cation keeps its charge
])
def test_names_a_reader_of_the_whole_keeps(name, read):
    assert H.read_back_with_hydrogen_words(name) == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(read)))


@pytest.mark.parametrize("smiles, name", [
    ("[HH].[CH3-].[Pd]", "methanide—palladium(0)—dihydrogen (1/1/1)"),
    ("[HH].CC.[Y].[CH3-]", "methanide—ethane—yttrium(0)—dihydrogen (1/1/1/1)"),
    ("[HH].CCO.O=O", "ethanol—1,2-dioxaeth-1-ene—dihydrogen (1/1/1)"),   # the spelling OPSIN reads inside an adduct
    ("[HH].CC(=O)[O-]", "1-oxido-1-oxoethane—dihydrogen (1/1)"),   # 'acetate—dihydrogen (1/1)' reads as the acid
])
def test_a_neutral_metal_beside_ions_takes_its_zero_oxidation_number(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name, r
    if name:   # OPSIN main reads the dihydrogen words itself
        assert r["gate_outcome"] in ("reconstructed:hydrogen_words", "full_key_round_trip_verified"), r
        assert not r["is_pin"], r


@pytest.mark.parametrize("smiles, name", [
    ("[H+].[Pd].CCO", "ethanol—palladium(0)—hydron (1/1/1)"),
    ("[H+].[Cl-].[Cl-].[Y]", "chloride—yttrium(0)—hydron (2/1/1)"),
    ("[H+].[CH3-].[Pd]", "methanide—palladium—hydron (1/1/1)"),       # the charges balance: no zero needed
])
def test_a_bare_metal_beside_a_hydron_takes_its_zero_oxidation_number(smiles, name):
    """A reader that balances the charges of an adduct gives a bare metal word the charge of the ions beside it
    ('ethanol—palladium—hydron' reads Pd2+); the last producer of a mixture name cites the neutral atom drawn as
    'palladium(0)', as the adduct body does, the Blue Book)."""
    r = name_best_effort(smiles)
    assert (r["name"], r["tier"], r["is_pin"]) == (name, "systematic_verified", False), r
