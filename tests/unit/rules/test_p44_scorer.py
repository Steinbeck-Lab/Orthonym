# tests/unit/rules/test_p44_scorer.py
"""v25 G1: unified P-44 parent scorer (pool + comparator + selector)."""
import pytest
from rdkit import Chem

from orthonym.rules.p44_scorer import ParentCandidate, pool_candidates

pytestmark = pytest.mark.unit


def _ring_systems(mol):
    """Connected ring systems as sets of atom indices (SSSR union)."""
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ring = set(ring)
        merged = [s for s in systems if s & ring]
        for s in merged:
            systems.remove(s)
            ring |= s
        systems.append(ring)
    return systems


class TestPooling:
    def test_heptylbenzene_pools_ring_and_chain(self):
        mol = Chem.MolFromSmiles("CCCCCCCc1ccccc1")
        chain = [i for i in range(7)]  # the heptyl carbons (atom order of this SMILES)
        pool = pool_candidates(mol, _ring_systems(mol), chain, None, [])
        kinds = sorted(c.kind for c in pool)
        assert kinds == ["chain", "ring"]
        ring = [c for c in pool if c.kind == "ring"][0]
        assert len(ring.atoms) == 6
        assert ring.pg_count == 0

    def test_pg_counts_on_candidates(self):
        # 3-phenylpropan-1-ol: OH on the chain only
        mol = Chem.MolFromSmiles("OCCCc1ccccc1")
        match = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4][OX2H]"))
        chain = [1, 2, 3]  # the three chain carbons (verified atom order)
        pool = pool_candidates(mol, _ring_systems(mol), chain, "alcohol", list(match))
        chain_cand = [c for c in pool if c.kind == "chain"][0]
        ring_cand = [c for c in pool if c.kind == "ring"][0]
        assert chain_cand.pg_count == 1
        assert ring_cand.pg_count == 0

    def test_p514_hetero_chain_admitted_only_at_4_bridging_units(self):
        # tetraoxa chain on cyclohexane: 4 bridging O -> admitted
        # (count the O's: COCCOCCOCCOC... has exactly 4 bridging ethers)
        mol = Chem.MolFromSmiles("COCCOCCOCCOCC1CCCCC1")
        pool = pool_candidates(mol, _ring_systems(mol), [], None, [])
        assert any(c.kind == "chain" and len(c.atoms) >= 10 for c in pool)
        # butoxycyclohexane: 1 bridging O -> hetero chain NOT admitted
        mol2 = Chem.MolFromSmiles("CCCCOC1CCCCC1")
        pool2 = pool_candidates(mol2, _ring_systems(mol2), [0, 1, 2, 3], None, [])
        assert all(
            all(mol2.GetAtomWithIdx(i).GetSymbol() == "C" for i in c.atoms)
            for c in pool2 if c.kind == "chain"
        )
