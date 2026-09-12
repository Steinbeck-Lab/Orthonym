"""Lever N (round 2): the von Baeyer main-ring search is computed once per (mol, ring system) in a scope,
returns the same value as the impl, and a memo hit charges the same perf-budget units."""
from rdkit import Chem
from orthonym.assembly import memo
from orthonym.assembly import fragment_naming as fn
from orthonym.rules.polycyclic import VonBaeyerAnalyzer


def _setup(smi):
    mol = Chem.MolFromSmiles(smi); an = VonBaeyerAnalyzer()
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    bh = an._find_all_bridgeheads(mol, ring_atoms)
    return mol, an, ring_atoms, bh


def test_memo_hit_and_equal_values(monkeypatch):
    calls = {"n": 0}
    orig = VonBaeyerAnalyzer._find_main_ring_impl
    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(VonBaeyerAnalyzer, "_find_main_ring_impl", spy)
    for smi in ("C1CC2CCC1C2", "C1C2CC3CC1CC(C2)C3", "C1CC2(C1)CCC2"):
        mol, an, ra, bh = _setup(smi)
        if not bh:
            continue
        tok = memo.push_scope()
        try:
            r1 = an._find_main_ring(mol, ra, bh)
            r2 = an._find_main_ring(mol, ra, bh)
            r3 = an._find_main_ring(mol, ra, bh, return_candidates=True)
        finally:
            memo.pop_scope(tok)
        assert r1 == r2 and (r3[0], r3[1]) == r1
        assert orig(an, mol, ra, bh) == r1 and orig(an, mol, ra, bh, return_candidates=True) == r3
    assert calls["n"] >= 1


def test_budget_units_replayed_on_hit():
    mol, an, ra, bh = _setup("C1C2CC3CC1CC(C2)C3")   # adamantane
    assert bh
    fn._fragment_guard.perf_budget = 10**9
    tok = memo.push_scope()
    try:
        an._find_main_ring(mol, ra, bh)
        spent_first = 10**9 - fn._fragment_guard.perf_budget
        fn._fragment_guard.perf_budget = 10**9
        an._find_main_ring(mol, ra, bh)          # memo hit
        spent_hit = 10**9 - fn._fragment_guard.perf_budget
    finally:
        memo.pop_scope(tok); fn._fragment_guard.perf_budget = None
    assert spent_first > 0 and spent_hit == spent_first
