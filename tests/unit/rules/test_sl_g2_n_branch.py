"""SL lane G, G2: the group of a bridging amine follows the parent, and the chain on its nitrogen is one branch (pure).

 "Selection of the principal chain or senior ring system in secondary and tertiary amines"
(the Blue Book): "When a choice for parent hydride is possible between a ring and a chain, the
ring is preferred"; the nitrogen belongs to every parent it is bonded to. The perceived group keeps ONE
bearing carbon of that nitrogen (the one with most heavy neighbours, a tie by atom index): the producers
of the chosen parent read the group on that parent (``reanchor_bridging_amine``), and a chain on the
nitrogen that holds more than alkyl is one branch, a prefix of the ring parent
(``acyclic_branch_on_nitrogen``).
"""
import pytest
from rdkit import Chem

from orthonym.rules.n_branch import (
    acyclic_branch_on_nitrogen,
    amine_promotion,
    holds_amine_or_senior_group,
    is_saturated_chain_carbon,
    reanchor_bridging_amine,
)

pytestmark = pytest.mark.unit


def _setup(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = [a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing() and a.GetSymbol() == "C"]
    return mol, n, ring, chain


@pytest.mark.parametrize("smiles", ["CC(C)Nc1ccccc1", "c1ccccc1NC(C)C"])
def test_the_bearing_carbon_follows_the_parent(smiles):
    mol, n, ring, chain = _setup(smiles)
    ring_carbon = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetIdx() in ring)
    chain_carbon = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetIdx() not in ring)
    for group in ("secondary_amine", "tertiary_amine"):
        for held in (ring_carbon, chain_carbon):
            assert reanchor_bridging_amine(mol, group, [(n, held)], sorted(ring)) == [(n, ring_carbon)]
            assert reanchor_bridging_amine(mol, group, [(n, held)], chain) == [(n, chain_carbon)]


def test_a_group_that_is_not_a_bridging_amine_is_left_alone():
    mol, n, ring, chain = _setup("CC(C)Nc1ccccc1")
    pg = [(n, 1)]
    assert reanchor_bridging_amine(mol, "alcohol", pg, sorted(ring)) is pg
    assert reanchor_bridging_amine(mol, "secondary_amine", None, sorted(ring)) is None
    # a parent the nitrogen is not bonded to: the match stays
    assert reanchor_bridging_amine(mol, "secondary_amine", pg, [99]) == [(n, 1)]


def test_a_chain_branch_with_a_ring_of_its_own_is_one_branch():
    mol = Chem.MolFromSmiles("CCC(CSc1ccccc1)Nc1ccccc1")
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    aniline_ring = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.IsInRing()}
    ring_atoms = set(next(r for r in mol.GetRingInfo().AtomRings() if aniline_ring & set(r)))
    chain_c = next(nb for nb in mol.GetAtomWithIdx(n).GetNeighbors() if not nb.IsInRing())
    atoms, count = acyclic_branch_on_nitrogen(mol, chain_c, ring_atoms | {n})
    assert count == 0 and atoms == sorted(a.GetIdx() for a in mol.GetAtoms()
                                          if a.GetIdx() not in ring_atoms | {n})


@pytest.mark.parametrize("smiles,is_chain", [
    ("CC(C)Nc1ccccc1", True),                       # the isopropyl carbon
    ("CC(=O)Nc1ccccc1", False),                     # the carbon of an acyl group: an amide nitrogen
    ("c1ccccc1Nc1ccccc1", False),                   # an aryl carbon
    ("C1CCCCC1Nc1ccccc1", False),                   # a ring carbon
    ("CC(=N)Nc1ccccc1", False),                     # an imidoyl carbon
])
def test_only_a_saturated_chain_carbon_starts_a_branch(smiles, is_chain):
    mol, n, ring, chain = _setup(smiles)
    nb = [a for a in mol.GetAtomWithIdx(n).GetNeighbors()]
    assert any(is_saturated_chain_carbon(a) for a in nb) is is_chain
    if not is_chain:
        assert all(acyclic_branch_on_nitrogen(mol, a, set()) == (None, 0) for a in nb)


@pytest.mark.parametrize("smiles,held", [
    ("NCCNc1ccccc1", True),                          # one more amine
    ("CCN(CC)CCNc1ccccc1", True),                    # a tertiary amine
    ("OCCNc1ccccc1", True),                          # an alcohol: senior to the amine
    ("OOCCNc1ccccc1", True),                         # a hydroperoxide
    ("N#CCNc1ccccc1", True),                         # a nitrile
    ("OC(=O)CNc1ccccc1", True),                      # an acid
    ("O=CCNc1ccccc1", True),                         # an aldehyde
    ("CC(=O)CNc1ccccc1", True),                      # a ketone
    ("NNCCNc1ccccc1", True),                         # a hydrazine nitrogen
    ("CCCNc1ccccc1", False),                         # a plain alkyl
    ("CC(CSc1ccccc1)Nc1ccccc1", False),              # a sulfanyl chain with a ring of its own
    ("COCCNc1ccccc1", False),                        # an ether
    ("ClCCNc1ccccc1", False),                        # a halide
    ("O=[N+]([O-])CCNc1ccccc1", False),              # a nitro group is a prefix only
])
def test_a_branch_that_holds_an_amine_or_a_senior_group_is_no_prefix_of_an_amine_parent(smiles, held):
    """ (the Blue Book) and (:26375): such a branch decides the parent itself."""
    mol = Chem.MolFromSmiles(smiles)
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and any(
        nb.GetIsAromatic() for nb in a.GetNeighbors()))
    aryl_ring = set(next(r for r in mol.GetRingInfo().AtomRings() if any(
        nb.GetIdx() in r for nb in mol.GetAtomWithIdx(n).GetNeighbors())))
    branch = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in aryl_ring and a.GetIdx() != n]
    assert holds_amine_or_senior_group(mol, branch) is held
    assert bool(amine_promotion(mol, branch, "x")) is (not held)
    if not held:
        assert amine_promotion(mol, branch, "x") == {
            "amine_candidate": {"suffix_name": "amine", "n_substituents": ["x"]}}
    chain_c = next(nb for nb in mol.GetAtomWithIdx(n).GetNeighbors() if not nb.GetIsAromatic())
    aryl = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetIsAromatic())
    ring_atoms = set(next(r for r in mol.GetRingInfo().AtomRings() if aryl in r)) | {n}
    atoms, count = acyclic_branch_on_nitrogen(mol, chain_c, ring_atoms)
    assert (atoms is None) is held and count == 0
