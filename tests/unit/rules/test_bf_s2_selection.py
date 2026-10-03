""" selection of the fused ring system that is bridged, for every parent the
package names (slice S2): (a) rings, (b) skeletal atoms, (c) heteroatoms, (d) the senior
ring system, (e)-(h) divalent bridges; (two rings of five or more members)."""
from dataclasses import replace

import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build, selection
from orthonym.rules.bridged_fused_pin.selection import best_splits, ears, ring_system


def _splits(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, best_splits(mol, ring_system(mol))


@pytest.mark.parametrize("smiles,parent,bridge_sizes", [
    ("C1=CC2=CC3=CC=C(C3)C2=C1", "azulene", [1]),                    #:19855
    ("C1=CC23C=CCC2=CC(=C1)CC3", "azulene", [2]),                    #:14653
    ("c1cc2c3ccn2Cc1c3", "indole", [1]),                             #:14253
    ("C1=CC23C=CC(C=C2OC1)CC3", "1-benzopyran", [2]),                #:14648
    ("C1=CC23COCC2(C=C1)C3", "2-benzofuran", [1]),                   #:24653
    ("C1=CC2CCC1=C1Sc3ccccc3C=C12", "thioxanthene", [2]),            #:14193
    ("C1=C2CC(=C1)C1=C2OC2=C(O1)C1=CC=C2C1", "oxanthrene", [1, 1]),  #:14187
    ("C1=CC2=C3C=c4ccc(o4)=CC3=C1C2", "benzo[8]annulene", [1, 1]),   #:14189
    ("C1=CC2=C3C4=CCC(=C3C=C1C2)C4", "benzo[7]annulene", [1, 1]),    #:19831
    ("C1=CC2C3C=CC(C3)C2C1", "indene", [1]),                         # dicyclopentadiene
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2", "anthracene", [1, 1, 1]),  # three bridges
])
def test_the_bridged_parent_is_found(smiles, parent, bridge_sizes):
    _, splits = _splits(smiles)
    assert splits and {s.parent for s in splits} == {parent}, splits
    assert all(sorted(len(b) for b in s.bridges) == bridge_sizes for s in splits)


@pytest.mark.parametrize("smiles,parent,bridge_sizes", [
    # (b) (:14271) "include the maximum number of skeletal atoms", ex.:14277
    # (17 atoms rather than 16): the larger parent wins over naphthalene + a longer bridge
    ("C1CC2CCCC1c1ccccc12", "benzo[7]annulene", [2]),       # not 1,4-propanonaphthalene
    ("c1ccc2c(c1)CC1CCCC2C1", "benzo[8]annulene", [1]),     # not 1,3-propanonaphthalene
    ("C12=CC=C(C3=CC=CC=C13)C=CC=C2", "benzo[8]annulene", [2]),  # not 1,4-buta[1,3]dieno...
    ("C1=CC23C=CC=CC2(C=C1)CCCCC3", "benzo[7]annulene", [4]),     # not 4a,8a-pentano...
])
def test_criterion_b_prefers_the_larger_parent(smiles, parent, bridge_sizes):
    _, splits = _splits(smiles)
    assert splits and {s.parent for s in splits} == {parent}, splits
    assert all(sorted(len(b) for b in s.bridges) == bridge_sizes for s in splits)


@pytest.mark.parametrize("smiles,name", [
    ("C1CC2CCCC1c1ccccc12", "6,7,8,9-tetrahydro-5H-5,9-ethanobenzo[7]annulene"),
    # the 3-benzoxepine / 3-benzothiepine parent (11 atoms) with an etheno bridge, not
    # naphthalene (10 atoms) with a -CH2-X-CH2- bridge; OPSIN 2.9.0 reads both readings
    ("C1=CC2COCC1c1ccccc12", "1,2,4,5-tetrahydro-1,5-etheno-3-benzoxepine"),
    ("C1=CC2CSCC1c1ccccc12", "1,2,4,5-tetrahydro-1,5-etheno-3-benzothiepine"),
])
def test_criterion_b_names(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


def test_criterion_d_indole_is_senior_to_indolizine():
    #:14253 '1,5-methanoindole (PIN)': the indolizine reading of the same skeleton ties on
    # (a)-(c); (d) (:14289, "see ") decides by the heteroatom locants (N1 < N4), as
    # in:14329 ("the locant set '1,2,4'... is lower than the locant set '1,3,9'")
    mol = Chem.MolFromSmiles("c1cc2c3ccn2Cc1c3")
    system = ring_system(mol)
    unsat = selection.ring_unsaturation(mol, system)
    readings = []
    for chain in ears(mol, system):
        got = selection._candidate(mol, system, unsat, (chain,))
        if got is not None:
            split = got[2]
            readings.append(replace(split, parent=selection._parent_name(mol, set(split.residual))))
    assert sorted(s.parent for s in readings) == ["indole", "indolizine"]
    assert [s.parent for s in selection._senior(mol, readings)] == ["indole"]


def test_ears_are_whole_chains_between_branch_atoms():
    # (:14043): a divalent bridge is joined to the fused ring system at its two
    # ends only, so it is a whole chain of atoms with two ring-system neighbours
    mol = Chem.MolFromSmiles("C1CC2CCCC1c1ccccc12")
    sizes = sorted(len(c) for c in ears(mol, ring_system(mol)))
    assert sizes == [2, 3, 4]


@pytest.mark.parametrize("smiles", [
    "C1CC2C1C1CCC2C1",                         # tricyclo[4.2.1.0^2,5]nonane: every fused
                                               # residual has one ring of 5+:23710)
    "C1=CC=C2C(=C1)C1c3ccccc3C2c2ccccc21",     # triptycene: a ring bridge ([1,2]benzeno,
                                               #:14418) ranks first and is not spelled
    "c1ccc2ccccc2c1",                          # a fused ring system has no bridge
    "C1CC2CCC1C2",                             # norbornane: no fused parent
])
def test_no_bridged_fused_reading(smiles):
    _, splits = _splits(smiles)
    assert splits is None


def test_the_ethano_and_etheno_readings_are_both_offered_to_criterion_j():
    # 2H,7H-4a,7-ethano-1-benzopyran: the -CH2-CH2- and -CH=CH- links both leave a
    # 1-benzopyran; build applies (j) (:14395)
    _, splits = _splits("C1=CC23C=CC(C=C2OC1)CC3")
    assert sorted(s.n_double for s in splits) == [2, 3]


@pytest.mark.parametrize("smiles,parents", [
    # xanthene (three rings in a horizontal row) and 6H-dibenzo[b,d]pyran (two rows):
    # (b) (:19740, ex.:19759 'anthracene (PIN) > phenanthrene (PIN)') ranks
    # xanthene first, the heteroatom locants (O10, O5) would pick dibenzo[b,d]pyran
    ("c1ccc2c(c1)Cc1ccccc1O2.c1ccc2c(c1)COc1ccccc1-2", {"xanthene", "dibenzo[b,d]pyran"}),
    # acridine and benzo[g]quinoline: (b) ties; the fusion-descriptor criteria (c)-(e) decide
    ("c1ccc2nc3ccccc3cc2c1.c1ccc2cc3ncccc3cc2c1", {"acridine", "benzo[g]quinoline"}),
])
def test_criterion_d_declines_three_ring_parents_only_heteroatom_locants_separate(smiles, parents):
    # the heteroatom locant step is the book's (d) practice for two-ring parents
    # (:14327-:14329,:14253); it never pre-empts (b)
    mol = Chem.MolFromSmiles(smiles)
    splits = [selection.Split(frozenset(f), (), (), selection._parent_name(mol, set(f)), 0,
                              frozenset()) for f in Chem.GetMolFrags(mol)]
    assert {s.parent for s in splits} == parents
    assert selection._senior(mol, splits) is None


def test_criterion_d_fusion_descriptors_decide_only_a_tie(monkeypatch):
    # (:19737) "applied successively until no alternatives remain": (a)
    # (:19739; ex.:19751 'azulene (PIN) > naphthalene (PIN)') ranks cyclopenta[8]annulene
    # (rings 8, 5;:13974) above benzo[7]annulene (7, 6;:17032), so the fusion-descriptor
    # criteria (c), (d) (:19741-:19742) are never reached for this pair
    mol = Chem.MolFromSmiles("C1=CC=C2CC=CC=CC2=C1.C1C=CC2=C1C=CC=CC=C2")
    splits = [selection.Split(frozenset(f), (), (), selection._parent_name(mol, set(f)), 0,
                              frozenset()) for f in Chem.GetMolFrags(mol)]
    assert {s.parent for s in splits} == {"benzo[7]annulene", "cyclopenta[8]annulene"}
    assert [s.parent for s in selection._senior(mol, splits)] == ["cyclopenta[8]annulene"]
    # the same two names left tied by and (a), (b) need (c), (d): declined
    real = {s.parent: selection._seniority_key(mol, s) for s in splits}
    head = real["cyclopenta[8]annulene"][:2]
    tied = {"cyclopenta[8]annulene": head + ((1,),), "benzo[7]annulene": head + ((2,),)}
    monkeypatch.setattr(selection, "_seniority_key", lambda m, sp: tied[sp.parent])
    assert selection._senior(mol, splits) is None


def _exhaustive_competitors(mol, system):
    """Every connected atom set that is not a whole chain, removes two or three rings and
    leaves a residual ``_candidate`` accepts (all sizes; the test oracle)."""
    unsat = selection.ring_unsaturation(mol, system)
    adj = {a: {nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
               if nb.GetIdx() in system} for a in system}
    chains = set(ears(mol, system))
    frontier, every = {frozenset([a]) for a in system}, set()
    for _ in range(len(system) - 1):
        every |= frontier
        frontier = {s | {j} for s in frontier for a in s for j in adj[a] if j not in s}
    return adj, chains, unsat, {
        s for s in every if s not in chains and selection._rings_removed(s, adj) in (2, 3)
        and selection._candidate(mol, system, unsat, (s,)) is not None}


@pytest.mark.parametrize("smiles,sizes", [
    ("C1CCC2CCC34CCCCCCC3(CC2)CC(CC1)CC4", [9, 10]),   # crafted: competitors over 8 atoms
    ("C1CCCC23CCC4CCCCC(CCC2(CC1)CC4)C3", [8, 9]),
    ("C1=CC=C2C(=C1)C1c3ccccc3C2c2ccccc21", [6, 6, 6]),  # triptycene's [1,2]benzeno bridges
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2", []),   # three methano bridges, no rival
])
def test_competitors_are_found_whatever_their_size(smiles, sizes):
    # a reading that loses to a polyvalent, dependent or ring bridge is declined, so every
    # such competitor must be seen: _competitor_sets equals the exhaustive enumeration
    mol = Chem.MolFromSmiles(smiles)
    system = ring_system(mol)
    adj, chains, unsat, expected = _exhaustive_competitors(mol, system)
    got = {s for s, removes in selection._competitor_sets(system, adj, list(chains))
           if removes in (2, 3) and selection._candidate(mol, system, unsat, (s,)) is not None}
    assert got == expected
    assert sorted(len(s) for s in expected) == sizes
