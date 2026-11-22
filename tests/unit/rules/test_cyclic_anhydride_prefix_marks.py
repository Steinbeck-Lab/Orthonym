"""A compound prefix on a saturated cyclic anhydride dione is enclosed, and a locant tie goes to the first-cited prefix.

 (the Blue Book): a cyclic anhydride is named as a heterocyclic pseudoketone (method (1), the PIN,
'oxolane-2,5-dione'). "Parentheses" (:7230), (:7232): "Parentheses are used around compound...
and complex... prefixes". (g) (:3307): when there is a choice, low locants go to the prefix cited first in
alphanumerical order. rules.anhydrides._name_saturated_oxa_dione cited '2-methylpropyl' bare
('3-2-methylpropyloxolane-2,5-dione', labelled pin_verified because OPSIN reads it) and read one substituent
per ring atom.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)CC1CC(=O)OC1=O", "3-(2-methylpropyl)oxolane-2,5-dione"),
    ("O=C1CC(c2ccc(O)cc2)C(=O)O1", "3-(4-hydroxyphenyl)oxolane-2,5-dione"),
    ("ClCC1CC(=O)OC1=O", "3-(chloromethyl)oxolane-2,5-dione"),
    ("CC(C)CC1C(=O)OC(=O)C1c1ccc(O)cc1", "3-(4-hydroxyphenyl)-4-(2-methylpropyl)oxolane-2,5-dione"),
    ("CC(C)C[C@@H]1C(=O)OC(=O)[C@@H]1c1ccc(O)cc1",
     "(3S,4S)-3-(4-hydroxyphenyl)-4-(2-methylpropyl)oxolane-2,5-dione"),
    ("CCCCC/C=C/CC1CC(=O)OC1=O", "3-[(2E)-oct-2-en-1-yl]oxolane-2,5-dione"),
    # (g): 'ethyl' is cited first and takes locant 3
    ("CCC1C(=O)OC(=O)C1C", "3-ethyl-4-methyloxolane-2,5-dione"),
    # simple prefixes stay bare
    ("CC1CC(=O)OC1=O", "3-methyloxolane-2,5-dione"),
    ("CC(C)(C)C1CCC(=O)OC1=O", "3-tert-butyloxane-2,6-dione"),
    ("CCC1C(CC)C(=O)OC1=O", "3,4-diethyloxolane-2,5-dione"),
    # two substituents on one ring atom (the namer read only the first one)
    ("CC1(C)CC(=O)OC1=O", "3,3-dimethyloxolane-2,5-dione"),
    ("CCC1(C)CC(=O)OC1=O", "3-ethyl-3-methyloxolane-2,5-dione"),
    ("CC1(C)CC(=O)OC(=O)C1", "4,4-dimethyloxane-2,6-dione"),
])
def test_cyclic_anhydride_dione_prefixes(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", ["CC1(C(=O)OC)CC(=O)OC1=O", "O=C1CC(C(=O)OC)C(=O)O1"])
def test_dione_namer_leaves_a_senior_group_alone(smiles):
    # a cyclic anhydride named as a heterocycle is a pseudoketone class 16, the Blue Book): an ester on
    # the ring is senior, so the dione namer declines
    from rdkit import Chem
    from orthonym.rules.anhydrides import _holds_group_senior_to_ketone
    mol = Chem.MolFromSmiles(smiles)
    ester_atoms = [a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing()]
    assert _holds_group_senior_to_ketone(mol, ester_atoms)


@pytest.mark.parametrize("smiles,name", [
    # the ester class 9) is senior to a cyclic anhydride named as a heterocycle (class 16)
    ("CC1(C(=O)OC)CC(=O)OC1=O", "methyl 3-methyl-2,5-dioxooxolane-3-carboxylate"),
    ("O=C1CC(C(=O)OC)C(=O)O1", "methyl 2,5-dioxooxolane-3-carboxylate"),
    # an acyclic anhydride (class 8) stays senior to the ester
    ("CC(=O)OC(=O)c1ccc(C(=O)OC)cc1", "acetic 4-(methoxycarbonyl)benzoic anhydride"),
])
def test_ester_beside_a_cyclic_anhydride(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)

