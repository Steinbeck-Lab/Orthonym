"""Lever M (round 2): name_c_substituted_sugar runs its expensive body once per structure within a scope."""
from rdkit import Chem
from orthonym.assembly import memo
from orthonym.data import sugar_names as sn


def test_memo_within_scope(monkeypatch):
    calls = {"n": 0}
    orig = sn._name_c_substituted_sugar_impl
    def spy(mol, csmi):
        calls["n"] += 1
        return orig(mol, csmi)
    monkeypatch.setattr(sn, "_name_c_substituted_sugar_impl", spy)
    smi = "C[C@@]1(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H](CO)O1"     # a C-methyl hexopyranose (input to the C-substituted path)
    a = Chem.MolFromSmiles(smi); b = Chem.MolFromSmiles(smi)
    tok = memo.push_scope()
    try:
        ra = sn.name_c_substituted_sugar(a, smi); rb = sn.name_c_substituted_sugar(b, smi)
    finally:
        memo.pop_scope(tok)
    assert ra == rb
    assert calls["n"] == 1
    assert sn.name_c_substituted_sugar(Chem.MolFromSmiles(smi), smi) == ra     # no scope: computed directly, same answer
    assert calls["n"] == 2
