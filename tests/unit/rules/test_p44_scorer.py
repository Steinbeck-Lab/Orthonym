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


from orthonym.rules.p44_scorer import compare_parent_candidates


def _pool(smiles, chain_idxs, pg=None, pg_smarts=None):
    mol = Chem.MolFromSmiles(smiles)
    matches = []
    if pg_smarts:
        matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts(pg_smarts)))
    pool = pool_candidates(mol, _ring_systems(mol), chain_idxs, pg, matches)
    return mol, pool, matches


def _best(mol, pool, pg=None, matches=None):
    import functools
    cmp = functools.cmp_to_key(
        lambda a, b: compare_parent_candidates(
            mol, a, b, principal_group=pg, principal_group_atoms=matches or []))
    return sorted(pool, key=cmp, reverse=True)[0]


class TestComparatorBlueBook:
    def test_p44_1_2_2_ring_senior_to_longer_chain(self):
        # BB line 19348: heptylbenzene (PIN) - ring senior regardless of size
        mol, pool, _ = _pool("CCCCCCCc1ccccc1", list(range(7)))
        assert _best(mol, pool).kind == "ring"

    def test_p44_1_2_2_ring_senior_despite_unsaturation(self):
        # BB line 19354: ethenylcyclohexane (PIN)
        mol, pool, _ = _pool("C=CC1CCCCC1", [0, 1])
        assert _best(mol, pool).kind == "ring"

    def test_p44_1_1_pg_on_chain_beats_ring(self):
        # 3-phenylpropan-1-ol: PCG (ol) on chain only -> chain parent
        mol, pool, m = _pool("OCCCc1ccccc1", [1, 2, 3],
                             pg="alcohol", pg_smarts="[CX4][OX2H]")
        assert _best(mol, pool, "alcohol", m).kind == "chain"

    def test_p44_1_2_hetero_chain_class_beats_carbocycle(self):
        # P-51.4-admitted tetraoxa chain (senior atom O, 4 bridging O)
        # vs all-C ring: P-44.1.2 class O > C -> chain parent.
        mol, pool, _ = _pool("COCCOCCOCCOCC1CCCCC1", [])
        assert _best(mol, pool).kind == "chain"

    def test_p44_1_2_heterocycle_still_beats_hetero_chain(self):
        # Ring contains N (rank above chain's O) -> P-44.1.2 keeps the
        # morpholine ring senior even against a P-51.4-admitted O-chain.
        mol = Chem.MolFromSmiles("COCCOCCOCCOCCN1CCOCC1")
        pool = pool_candidates(mol, _ring_systems(mol), [], None, [])
        assert _best(mol, pool).kind == "ring"

    def test_p44_2_1_ring_vs_ring_nitrogen_wins(self):
        # among rings, has-N (P-44.2.1.3) after heterocycle tie ->
        # pyridine ring senior to benzene ring.
        mol, pool, _ = _pool("c1ccc(-c2cccnc2)cc1", [])
        best = _best(mol, pool)
        best_syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in best.atoms}
        assert "N" in best_syms

    def test_p44_2_1_more_rings_wins(self):
        # naphthalene vs benzene: both carbocycles -> P-44.2.1.5 more rings.
        mol, pool, _ = _pool("c1ccc(-c2ccc3ccccc3c2)cc1", [])
        assert len(_best(mol, pool).atoms) == 10

    def test_p44_3_2_longer_chain_wins(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        a = ParentCandidate("chain", (0, 1, 2, 3, 4, 5), 0)
        b = ParentCandidate("chain", (0, 1, 2, 3), 0)
        assert compare_parent_candidates(mol, a, b) > 0

    def test_p44_4_1_1_more_multiple_bonds_wins(self):
        # 3-ethylhexa-1,5-diene skeleton: two same-length 6-chains through
        # C2; the one through both vinyls has 2 C=C, the ethyl-path has 1.
        # P-44.3 ties (same length, no heteroatoms) -> P-44.4.1.1 decides.
        mol = Chem.MolFromSmiles("C=CC(CC)CC=C")
        a = ParentCandidate("chain", (0, 1, 2, 5, 6, 7), 0)
        b = ParentCandidate("chain", (4, 3, 2, 5, 6, 7), 0)
        assert compare_parent_candidates(mol, a, b) > 0

    def test_deterministic_total_order(self):
        mol, pool, _ = _pool("CCCCCCCc1ccccc1", list(range(7)))
        import functools
        cmp = functools.cmp_to_key(
            lambda a, b: compare_parent_candidates(mol, a, b))
        assert sorted(pool, key=cmp) == sorted(sorted(pool, key=cmp), key=cmp)


from orthonym.rules.p44_scorer import select_parent_unified


class TestSelectParentUnified:
    def test_signature_matches_legacy(self):
        import inspect
        from orthonym.rules.parent_selection import select_parent
        assert (list(inspect.signature(select_parent_unified).parameters)
                == list(inspect.signature(select_parent).parameters))

    def test_heptylbenzene_ring_parent(self):
        mol = Chem.MolFromSmiles("CCCCCCCc1ccccc1")
        res = select_parent_unified(mol, _ring_systems(mol),
                                    list(range(7)), None, [])
        assert res.parent_type == "ring"
        assert len(res.parent_atoms) == 6

    def test_chain_pg_only_chain_parent_lists_ring_substituent(self):
        mol = Chem.MolFromSmiles("OCCCc1ccccc1")
        matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4][OX2H]")))
        res = select_parent_unified(mol, _ring_systems(mol), [1, 2, 3],
                                    "alcohol", matches)
        assert res.parent_type == "chain"
        assert len(res.substituent_rings) == 1

    def test_single_carbon_ring_attached_stays_ring(self):
        # benzaldehyde: PCG carbon bonded to ring -> ring parent (legacy rule)
        mol = Chem.MolFromSmiles("O=Cc1ccccc1")
        matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3H1]=O")))
        res = select_parent_unified(mol, _ring_systems(mol), [1],
                                    "aldehyde", matches)
        assert res.parent_type == "ring"

    def test_empty_chain_ring_parent(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        res = select_parent_unified(mol, _ring_systems(mol), [], None, [])
        assert res.parent_type == "ring"

    def test_reasoning_is_populated(self):
        mol = Chem.MolFromSmiles("CCCCCCCc1ccccc1")
        res = select_parent_unified(mol, _ring_systems(mol),
                                    list(range(7)), None, [])
        assert "P-44" in res.reasoning
