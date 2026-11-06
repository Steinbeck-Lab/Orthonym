"""A macrocyclic lactone with a ketone and a double bond is numbered by its '-one' suffixes first.

 NUMBERING (the Blue Book), item "(c) principal characteristic groups and free valences (suffixes)"
(:3256), ranks before unsaturation; Lactones (:32084): "(1) as heterocyclic pseudoketones by adding
the suffix 'one', 'dione'... Method (1) gives preferred IUPAC names", so the lactone C=O and a ring ketone are
the two '-one' carbons of one suffix set. The numbering of the replacement name took the lactone atoms as the
whole set (symmetric between the two ring directions), so the ene locant decided and the ring was numbered
the other way round: '1-oxacyclotetradec-4-ene-2,4-dione' (the 4-ene collides with the 4-one; the oxo/ene
guard held it back and dev2000 d170 abstained).
"""
import pytest
from rdkit import Chem

from orthonym.rules.macrocycle_orientation import orient_macrocycle_for_replacement
from tests.support.read_back import same_key

pytestmark = pytest.mark.opsin_gate

D1226 = "CCC(O)C(C)C(O)C(C)C(O)C(C)C1C/C=C(/C)CC(C)CC(C)CC(C)C(=O)CC(=O)O1"
ANALOG = "O=C1CC(=O)CC(C)CC(C)CC=CCC(CC)O1"


def test_d1226_is_named(namer):
    row = namer.name_tiered(D1226)
    expected = ("(11Z)-5,7,9,11-tetramethyl-14-(3,5,7-trihydroxy-4,6-dimethylnonan-2-yl)"
                "-1-oxacyclotetradec-11-ene-2,4-dione")
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert same_key(D1226, expected)


@pytest.mark.parametrize("smiles,expected", [
    (ANALOG, "13-ethyl-6,8-dimethyl-1-oxacyclotridec-10-ene-2,4-dione"),
    ("O=C1CC(=O)CCC=CCCCCCCO1", "1-oxacyclotetradec-7-ene-2,4-dione"),      # unchanged
    ("O=C1CCCCCCC=CCCCCCO1", "1-oxacyclopentadec-9-en-2-one"),              # unchanged: a lactone alone
])
def test_dione_and_neighbours(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert same_key(smiles, expected)


def test_the_suffix_set_is_every_ring_carbonyl():
    # the orientation itself: the ene locant is that of the numbering that gives the dione {2,4}
    mol = Chem.MolFromSmiles(ANALOG)
    ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == 13)
    lactone = [a for a in ring if mol.GetAtomWithIdx(a).GetSymbol() == "C"
               and any(nb.GetSymbol() == "O" and nb.GetDegree() == 1 for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
    het, ene = orient_macrocycle_for_replacement(mol, ring, lactone[:1])
    assert het == [(1, "O")] and ene == [10]


def test_a_principal_group_outside_the_ring_leaves_the_numbering_alone():
    # an exocyclic acid: nothing of the principal group is in the ring, so the ring C=O stay prefixes
    mol = Chem.MolFromSmiles("OC(=O)CC1CC(=O)CC=CCCCCCCCO1")
    ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) >= 12)
    acid_c = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetDegree() == 3
              and any(nb.GetSymbol() == "O" and nb.GetTotalNumHs() == 1 for nb in a.GetNeighbors())]
    assert acid_c and acid_c[0] not in ring
    het_a, ene_a = orient_macrocycle_for_replacement(mol, ring, acid_c)
    het_b, ene_b = orient_macrocycle_for_replacement(mol, ring, None)
    assert (het_a, ene_a) == (het_b, ene_b)
