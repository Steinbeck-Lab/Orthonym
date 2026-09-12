from rdkit import Chem
from orthonym.assembly import memo
from orthonym.rules import ring_selection as rs


def test_ring_score_memo_hit_and_equal():
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1C1CCCCC1")
    ri = mol.GetRingInfo()
    systems = [set(r) for r in ri.AtomRings()]
    tok = memo.push_scope()
    try:
        s1 = [rs.ring_system_score(mol, s) for s in systems]
        s2 = [rs.ring_system_score(mol, s) for s in systems]
        assert s1 == s2
        cache = memo._cache_var.get()
        assert sum(1 for k in cache if k[0] == "ring_selection.score") == len(systems)
    finally:
        memo.pop_scope(tok)
    assert [rs._ring_system_score_impl(mol, s) for s in systems] == s1


def test_ring_score_without_scope_matches_impl():
    mol = Chem.MolFromSmiles("O=C1NC(=O)c2ccccc12")
    systems = [set(r) for r in mol.GetRingInfo().AtomRings()]
    assert memo._cache_var.get() is None
    assert [rs.ring_system_score(mol, s) for s in systems] == [rs._ring_system_score_impl(mol, s) for s in systems]
