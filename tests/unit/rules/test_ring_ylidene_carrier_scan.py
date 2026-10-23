"""``ring_ylidene_carrier._ring_ylidene_bonds`` finds its bonds from the ring atoms of the fragment, not by
walking every bond of the molecule (a walk that cost the size of the molecule on each of the tens of thousands of
fragments a large peptide names). The reference below is the walk it replaced; both must give the same list, in
the same order, for any fragment and any set of ring atoms."""
import random

import pytest
from rdkit import Chem

from orthonym.rules import ring_ylidene_carrier as carrier


def _reference(mol, frag_set, ring_atoms, allow_stereo=False):
    found = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in frag_set or j not in frag_set or b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if (i in ring_atoms) == (j in ring_atoms):
            continue
        ring_c, chain_c = (mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j))
        if i not in ring_atoms:
            ring_c, chain_c = chain_c, ring_c
        if (ring_c.GetSymbol() != 'C' or chain_c.GetSymbol() != 'C' or ring_c.GetIsAromatic()
                or chain_c.GetIsAromatic() or chain_c.IsInRing() or b.GetIsAromatic()
                or (b.GetStereo() != Chem.BondStereo.STEREONONE and not allow_stereo)):
            continue
        found.append((chain_c.GetIdx(), ring_c.GetIdx()))
    return found


SMILES = [
    "C1CCCCC1=CC(=O)N",                                   # one plain ring-ylidene bond
    "CN\\1C2=CC=CC=C2S/C1=C\\C3=CC=[N+](C)C4=CC=CC=C34",   # a stereo-flagged one (the cyanine linker)
    "CC(C)=C1CCC(CC1)C(=O)N",                             # a ring-ylidene bond to a branched chain carbon
    "C=C1CCCCC1.C1CCCC1=CC",                              # two molecules, a ring-ylidene bond in each
    "c1ccccc1C=C",                                        # aromatic ring, no ring-ylidene bond
    "O=C1CCC(=CC=C2CCCC2)CC1",                            # two of them, one chain carbon between
    "CC=C(C)C1CCCC1=NO",                                  # a ring carbon double-bonded to nitrogen, not carbon
]


def _fragments(mol, rng):
    n = mol.GetNumAtoms()
    yield set(range(n))
    for _ in range(40):
        yield {i for i in range(n) if rng.random() < 0.6}


@pytest.mark.parametrize("smiles", SMILES)
@pytest.mark.parametrize("allow_stereo", [False, True])
def test_bonds_found_from_the_ring_atoms_equal_the_whole_molecule_walk(smiles, allow_stereo):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    rng = random.Random(20261023)
    ring_atoms_of_mol = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    hits = 0
    for frag_set in _fragments(mol, rng):
        # the ring atoms the producer is given (those of the fragment), and arbitrary sets of atoms: the
        # walk is plain set logic, so it must agree for any set
        for ring_atoms in (ring_atoms_of_mol & frag_set,
                           {i for i in range(mol.GetNumAtoms()) if rng.random() < 0.4}):
            want = _reference(mol, frag_set, ring_atoms, allow_stereo)
            got = carrier._ring_ylidene_bonds(mol, frag_set, ring_atoms, allow_stereo=allow_stereo)
            assert got == want
            hits += bool(want)
    if allow_stereo or "/" not in smiles:   # a stereo-flagged bond is skipped unless asked for
        assert hits, "the molecule must give at least one ring-ylidene bond, or the comparison proves nothing"


def test_the_stereo_flagged_bond_is_skipped_unless_asked_for():
    mol = Chem.MolFromSmiles("CN\\1C2=CC=CC=C2S/C1=C\\C3=CC=[N+](C)C4=CC=CC=C34")
    frag = set(range(mol.GetNumAtoms()))
    ring = {i for r in mol.GetRingInfo().AtomRings() for i in r}
    assert carrier._ring_ylidene_bonds(mol, frag, ring) == []
    assert len(carrier._ring_ylidene_bonds(mol, frag, ring, allow_stereo=True)) == 1
