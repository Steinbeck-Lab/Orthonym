"""A nuclide of the central atom of an oxoacid suffix stands before that suffix.

 "Names", (``the Blue Book Blue Book``): in a name of two or more words "the isotopic
descriptor is placed before the appropriate word or part of the word that includes the nuclide(s)". The printed
names of a nuclide in a suffix put it before the suffix, not before the whole name: 'benzene(13C)carboxylic acid
(PIN)' and 'benzene(13C)carbonitrile (PIN)' (:44254-:44255, under in " Location of nuclides on
positions not normally denoted by locants",:44221), 'cyclopentan-1-(18O)ol (PIN)' (:43744). The S of 'sulfonic
acid' is such a nuclide, so 'methane(34S)sulfonic acid' is the PIN and '(34S)methanesulfonic acid' is not. OPSIN 2.9.0
reads the descriptor before the acid and amide suffixes and before 'sulfonate'; it does not read it before the acyl
ending 'sulfonyl', so a name with a nuclide there is built with the descriptor at the front and is below the PIN.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _tiered_namer
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("[34S](=O)(=O)(O)C", "methane(34S)sulfonic acid"),
    ("C[34S](=O)(=O)N", "methane(34S)sulfonamide"),
    ("C[34S](=O)O", "methane(34S)sulfinic acid"),
    ("C[34S](=O)N", "methane(34S)sulfinamide"),
    ("C[34S](=O)(=O)OC", "methyl methane(34S)sulfonate"),
    ("O=[34S](=O)(O)C1CCCCC1", "cyclohexane(34S)sulfonic acid"),
    ("[34S](=O)(=O)(O)c1ccccc1", "benzene(34S)sulfonic acid"),
    ("C[77Se](=O)(=O)O", "methane(77Se)selenonic acid"),
    # a derivative that keeps the stem of the acid: the nuclide stands before the stem too
    ("C[34S](=O)(=O)NN", "methane(34S)sulfonohydrazide"),
    ("C[34S](=O)(=N)O", "methane(34S)sulfonimidic acid"),
    ("C[34S](=O)(=N)N", "methane(34S)sulfonimidamide"),
    ("c1ccccc1[34S](=O)(=O)NN", "benzene(34S)sulfonohydrazide"),
    ("C[34S](=O)(=O)N(C)C", "N,N-dimethylmethane(34S)sulfonamide"),
])
def test_the_nuclide_of_the_acid_centre_stands_before_the_suffix(smiles, expected):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"
    assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,best_effort_name", [
    ("C[34S](=O)(=O)Cl", "(34S)methanesulfonyl chloride"),
    ("C[34S](=O)(=O)c1ccccc1", "[(34S)methanesulfonyl]benzene"),
    ("C[34S](C)=O", "[(34S)methanesulfinyl]methane"),
    ("O[34S](=O)(=O)CC[34S](=O)(=O)O", "(34S2)ethane-1,2-disulfonic acid"),
])
def test_a_nuclide_at_a_place_opsin_cannot_read_is_below_the_pin(smiles, best_effort_name):
    """'methane(34S)sulfonyl chloride' is the PIN of the first row,:43800) and OPSIN 2.9.0 does not read
    it; a multiplied suffix cites the descriptor inside the multiplied part, 'cyclohexane-1,1-di[(14C)carboxylic
    acid] (PIN)',:43774). The front name round-trips and is kept below the PIN."""
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    wide = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert wide["name"] == best_effort_name and wide["tier"] == "systematic_verified"
    assert opsin_roundtrip_check(smiles, wide["name"])["passed"], wide["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("[13CH3]S(=O)(=O)O", "(13C)methanesulfonic acid"),          # a nuclide of the organyl group: before 'methane'
    ("[2H]C([2H])([2H])S(=O)(=O)O", "(2H3)methanesulfonic acid"),
    ("CS(=O)(=O)O[2H]", "methane(2H)sulfonic acid"),              # the hydrogen of the group: unchanged
])
def test_the_other_nuclides_keep_their_places(smiles, expected):
    assert Orthonym().name_tiered(smiles)["name"] == expected


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_a_ring_atom_is_no_centre_of_a_suffix_of_another_group():
    """The S of a thiolane 1,1-dioxide is a heteroatom of the ring name; the sulfonamide of the molecule is another group.
    The name is not lowered for a descriptor that is not before the stem 'sulfon' of the amide: the label is not on it."""
    smiles = "O=[34S]1(=O)CCCC1CS(=O)(=O)N"
    from orthonym.rules import isotope_oxoacid
    from rdkit import Chem
    assert not isotope_oxoacid.labels_a_centre(Chem.MolFromSmiles(smiles))
    assert isotope_oxoacid.labels_a_centre(Chem.MolFromSmiles("C[34S](=O)(=O)N"))
