import itertools
import random

import pytest
from rdkit import Chem

from orthonym.perception.mancude import (
    count_rings_at_least, fusion_pin_class, has_ortho_fused_pair, is_mancude_ring_system,
    mancude_fused_core, max_matching_size)


def _ring(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m, {a for r in m.GetRingInfo().AtomRings() for a in r}


def _brute_matching(nodes, adj):
    nodes = sorted(nodes)
    edges = sorted({(min(a, b), max(a, b)) for a in nodes for b in adj[a] if b in adj})
    best = 0
    for k in range(len(nodes) // 2, 0, -1):
        for combo in itertools.combinations(edges, k):
            used = [x for e in combo for x in e]
            if len(used) == len(set(used)):
                return k
    return best


@pytest.mark.unit
def test_edmonds_matches_brute_force_on_random_graphs():
    rng = random.Random(3)
    for _ in range(60):
        n = rng.randint(2, 11)
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    adj[i].add(j); adj[j].add(i)
        assert max_matching_size(adj.keys(), adj) == _brute_matching(adj.keys(), adj)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51", True),                 # corannulene
    ("C1C2C=CC1C=C2", False),                                  # norbornadiene: an augmenting path exists
    ("C1CC2CCC3CCCC4CCC(C1)C2C34", False),                     # perhydropyrene (S1)
    ("c1ccc2[nH]ccc2c1", True),                                # 1H-indole: N-H is indicated hydrogen
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2", False),     # WHOLE system: the CH2 bridges can augment
])
def test_is_mancude_whole_system(smiles, expected):
    m, ring = _ring(smiles)
    assert is_mancude_ring_system(m, ring) is expected


@pytest.mark.unit
def test_fused_core_of_trimethanoanthracene_is_mancude_anthracene():
    m, ring = _ring("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2")
    core = mancude_fused_core(m, ring)
    assert len(core) == 14 and is_mancude_ring_system(m, core) is True


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2", True),     # trimethanoanthracene (critic R4)
    ("C12=CC=C(C3=CC=4C5=CC=C(C4C=C13)C5)C2", True),           # 1,4:5,8-dimethanoanthracene
    ("c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51", True),                 # corannulene
    ("C12=CC=C(C3=CC=CC=C13)C2", True),                        # 1,4-methanonaphthalene (producer names it first)
    ("C1C2C=CC1C=C2", False),                                  # norbornadiene
    ("C1=CC2C=CC1C=C2", False),                                # barrelene
    ("C1CC2CCC3CCCC4CCC(C1)C2C34", False),                     # perhydropyrene (S1: not guarded here)
    ("C12=CC=CC=C(C=CC=C1)C2", False),                         # 1,6-methano[10]annulene (no ortho-fused pair)
    ("C1Cc2ccccc21", False),                                   # benzocyclobutene (one ring >= 5)
    ("C1C2CC3CC1CC(C2)C3", False),                             # adamantane
    # rev 2: ring O / pyrrole N-H are NOT bridge atoms (a review #6 exposed the first rule)
    ("COC(=O)CCc1oc2cc(C)cc(O)c2c(=O)c1[C@H](O)C(=O)OC", True),  # chromone ester (W02)
    ("c1ccc2[nH]ccc2c1", True),                                # 1H-indole
    ("C1CC2(C1)c1cccc3cccc2c13", True),                        # W01: ring sizes [4,4,6,6] -- a one-carbon peri bridge on naphthalene
    ("c1cc2cccc3ccc(c1)c23", True),                            # 1H-phenalene
])
def test_fusion_pin_class(smiles, expected):
    m, ring = _ring(smiles)
    assert fusion_pin_class(m, ring) is expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2", True),     # anthracene 6-rings share one bond
    ("C12=CC=CC=C(C=CC=C1)C2", False),                         # 1,6-methano[10]annulene: 7-rings share 2 bonds
    ("C1C2CC3CC1CC(C2)C3", False),                             # adamantane: 6-rings share 2 bonds
    ("C1Cc2ccccc21", False),                                   # benzocyclobutene: only one ring >= 5
])
def test_ortho_fused_pair(smiles, expected):
    m, ring = _ring(smiles)
    assert has_ortho_fused_pair(m, ring, 5) is expected


@pytest.mark.unit
def test_count_rings_at_least_benzocyclobutene():
    m, ring = _ring("C1Cc2ccccc21")
    assert count_rings_at_least(m, ring, 5) == 1
