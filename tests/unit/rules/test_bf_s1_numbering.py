"""Numbering of a bridged fused parent,,, and
the bridge prefixes,,,."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import numbering, prefixes
from orthonym.rules.bridged_fused_pin.selection import best_splits, ring_system


def _one(smiles):
    mol = Chem.MolFromSmiles(smiles)
    split = best_splits(mol, ring_system(mol))[0]   # symmetric readings: the first one
    names = [prefixes.bridge_prefix(mol, split, i).name for i in range(len(split.bridges))]
    return mol, split, names, numbering.numberings(mol, split, names)


def _best(nums):
    return min(nums, key=lambda n: (n.attachment_set, n.citation_order))


def test_the_parent_numberings_are_the_fixed_numbering_times_the_symmetry():
    # (a) (:3221) fixed numbering; naphthalene's skeleton has 4 automorphisms
    _, _, _, nums = _one("C1CC2CC1c1ccccc12")
    assert len(nums) == 4
    assert sorted({n.attachment_set for n in nums}) == [((1, ""), (4, "")), ((5, ""), (8, ""))]


def test_p25_4_4_bridge_atoms_start_at_the_higher_bridgehead():
    #:14403 "the numbering starts at the end of the chain... connected to the bridgehead
    # of the fused ring system having the highest locant" -> C9 is bonded to C4
    mol, split, _, nums = _one("C1CC2CCC1c1ccccc12")
    a2l = _best(nums).atom_to_locant
    atom9 = next(a for a, loc in a2l.items() if loc == 9)
    nbr_locs = {a2l[n.GetIdx()] for n in mol.GetAtomWithIdx(atom9).GetNeighbors()}
    assert nbr_locs == {4, 10}


def test_a_bridge_on_the_fusion_bond_starts_at_8a():
    mol, split, _, nums = _one("C1=CC23C=CC=CC2(C=C1)CC3")
    a2l = _best(nums).atom_to_locant
    assert _best(nums).attachment_set == ((4, "a"), (8, "a"))
    atom9 = next(a for a, loc in a2l.items() if loc == 9)
    assert {a2l[n.GetIdx()] for n in mol.GetAtomWithIdx(atom9).GetNeighbors()} == {"8a", 10}


def test_p25_4_5_2_the_bridge_on_the_higher_bridgeheads_is_numbered_first():
    #:14440 -- 1,4-epoxy-5,8-methano...: the methano carbon on 5,8 is C9, the O is C10
    mol, split, names, nums = _one("C1CC2CC1c1c2c2ccc1o2")
    a2l = _best(nums).atom_to_locant
    by_loc = {loc: mol.GetAtomWithIdx(a).GetSymbol() for a, loc in a2l.items()}
    assert (by_loc[9], by_loc[10]) == ("C", "O")


def test_p25_4_3_3_b_citation_order_gives_epoxy_the_low_pair():
    #:14231 "(b) lowest locants in the order of citation for the bridges";
    # ex.:14237 '1,4-epoxy-5,8-methanonaphthalene (PIN)'
    _, _, _, nums = _one("C1CC2CC1c1c2c2ccc1o2")
    best = _best(nums)
    assert best.bridge_locants == (("epoxy", (1, 4)), ("methano", (5, 8)))


@pytest.mark.parametrize("smiles,expected", [
    ("C1CC2CC1c1ccccc12", ["methano"]),
    ("C1CC2CCC1c1ccccc12", ["ethano"]),
    ("C1=CC23C=CC=CC2(C=C1)CC3", ["ethano"]),
    ("C1=CC2c3ccccc3C1c1ccccc12", ["etheno"]),
    ("C1=CC2OC1c1ccccc12", ["epoxy"]),
])
def test_bridge_prefixes(smiles, expected):
    _, _, names, _ = _one(smiles)
    assert names == expected


def test_general_nomenclature_bridge_prefixes_are_not_pin_forms():
    # (:14097): 'epithio' and 'epimino' are general nomenclature
    mol = Chem.MolFromSmiles("C12SC(C=C1)c1ccccc12")
    (split,) = best_splits(mol, ring_system(mol))
    bp = prefixes.bridge_prefix(mol, split, 0)
    assert (bp.name, bp.is_pin_form) == ("epithio", False)


def test_bridge_text_groups_identical_bridges_with_colons():
    # (:14173) and (:14181)
    assert prefixes.bridge_text([("methano", (1, 4)), ("methano", (5, 8))]) == "1,4:5,8-dimethano"
    assert prefixes.bridge_text([("epoxy", (1, 4)), ("ethano", ("4a", "8a"))]) == "1,4-epoxy-4a,8a-ethano"


def test_letter_locants_sort_after_their_number():
    # (:3193)
    assert sorted([5, "4a", 4, "8a", 8], key=numbering.loc_key) == [4, "4a", 5, 8, "8a"]


def test_a_bridge_from_a_fusion_atom_to_a_peripheral_atom():
    # The third bridge kind (beside 1,4- and 4a,8a-): from the four-bonded fusion atom 4a
    # to the peripheral C2 (BB analogues:14648 '2H,7H-4a,7-ethano-1-benzopyran (PIN)',
    #:14653 '1H-3a,7-ethanoazulene (PIN)'). (a) (:14225): {2,4a} is the lowest
    # of the four symmetric sets {2,4a}, {3,8a}, {6,8a}, {7,4a}.
    mol, split, _, nums = _one("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3")
    best = _best(nums)
    assert best.attachment_set == ((2, ""), (4, "a"))
    a2l = best.atom_to_locant
    atom9 = next(a for a, loc in a2l.items() if loc == 9)
    assert {a2l[n.GetIdx()] for n in mol.GetAtomWithIdx(atom9).GetNeighbors()} == {2, "4a"}
