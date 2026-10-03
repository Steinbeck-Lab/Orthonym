""" / selection of the fused ring system that is bridged."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin.selection import best_splits, fused_rings, ring_system


def _splits(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, best_splits(mol, ring_system(mol))


@pytest.mark.parametrize("smiles,parent,n_bridges,bridge_sizes", [
    ("C1=CC23C=CC=CC2(C=C1)CC3", "naphthalene", 1, [2]),          #:14249, bridge on the 4a-8a bond
    ("C1=CC23CCC2(C=C1)c1ccc3o1", "naphthalene", 2, [1, 2]),      #:14587
    ("c1ccc2c(c1)C1CCC2C2CCCCC12", "anthracene", 1, [2]),         #:19964
    ("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3", "naphthalene", 1, [1]),
    ("C[C@@]12CCC[C@@]3(C)[C@@H](C1)[C@@](O)(CO)CC[C@@]23C", "naphthalene", 1, [1]),
])
def test_the_bridged_parent_is_found(smiles, parent, n_bridges, bridge_sizes):
    _, splits = _splits(smiles)
    assert splits, splits
    assert {s.parent for s in splits} == {parent}
    assert all(sorted(len(b) for b in s.bridges) == bridge_sizes for s in splits)
    assert all(len(s.bridges) == n_bridges for s in splits)


def test_ethano_and_etheno_readings_are_both_offered_for_criterion_j():
    # 1,4-dihydro-1,4-ethanonaphthalene: excising -CH2-CH2- or -CH=CH- both leave a
    # naphthalene; (j) (:14395) is applied by build on n_double.
    _, splits = _splits("C1=CC2CCC1c1ccccc12")
    assert sorted(s.n_double for s in splits) == [3, 4]   # benzo 3 (+ C2=C3)


def test_a_metheno_reading_loses_to_the_divalent_methano_bridge():
    # 1,3-methanonaphthalene (:19863): the C2 ring atom read as a bridge would be
    # attached by a double bond (a polyvalent 'metheno'-type bridge); (e) prefers the
    # divalent -CH2- bridge.
    mol, splits = _splits("c1ccc2c3cc(cc2c1)C3")
    assert len(splits) == 1 and len(splits[0].bridges) == 1
    (bridge,) = splits[0].bridges
    assert mol.GetAtomWithIdx(bridge[0]).GetTotalNumHs() == 2


@pytest.mark.parametrize("smiles", [
    "C1Cc2cccc3cccc1c23",            # acenaphthene: the whole system is fused:14241)
    "C1c2cc3ccccc3cc21",             # 1H-cyclopropa[b]naphthalene: fused, not bridged
    "c1ccc2ccccc2c1",                # naphthalene
    "C1CC2CCC1C2",                   # norbornane: no fused parent
    "C1C2CC3CC1CC(C2)C3",            # adamantane
    "C1CC2CC1C2c1ccccc1",            # a phenyl substituent is another ring system
    "C1C2CCC1C1CC21",                # tricyclo[3.2.1.0^2,4]octane: no naphthalene skeleton
    "C1c2ccccc21",                   #:23718 bicyclo[4.1.0]hepta-1,3,5-triene (PIN): two rings
])
def test_no_bridged_naphthalene_or_anthracene(smiles):
    _, splits = _splits(smiles)
    assert splits is None


def test_fused_rings_rejects_a_bond_in_three_rings():
    # (d) (:14030): the ethano ring across the 4a-8a bond is a bridge, so the
    # whole ring system is not a fused ring system.
    mol = Chem.MolFromSmiles("C1=CC23C=CC=CC2(C=C1)CC3")
    assert fused_rings(mol, ring_system(mol)) is None
    naphthalene = {a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() < 4 or a.IsInRingSize(6)}
    assert fused_rings(mol, naphthalene - {10, 11}) is not None


def test_large_systems_are_declined_before_any_enumeration():
    # 24 ring atoms: more than an anthracene with two four-atom bridges
    mol = Chem.MolFromSmiles("C1CC2CC3CC4CC5CC6CC7CC8CC(C1)C2C3C4C5C6C7C8")
    assert best_splits(mol, ring_system(mol)) is None
