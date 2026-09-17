"""R2(c): the DFS must enumerate the SAME paths in the SAME order from a precomputed
adjacency list as from live RDKit GetNeighbors calls (path order decides ties in
_best_disjoint_pair), and the expansion cap must fire at the same expansion."""
import pytest
from rdkit import Chem
from orthonym.rules import polycyclic as pc

CAGES = ["C1CC2CCC1C2", "C1CC2CC1CC2", "C12C3C4C1C5C2C3C45", "C1C2CC3CC1CC(C2)C3",
         "O=C1CC2CCC1(C)C2", "C1CC2(CC1)CCC2", "C1CCC2(C1)CC3CCC2C3"]

def _adj(mol):
    return {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}

@pytest.mark.parametrize("smi", CAGES)
def test_adjacency_list_gives_identical_ordered_paths(smi):
    mol = Chem.MolFromSmiles(smi); ri = mol.GetRingInfo()
    ring_atoms = {i for r in ri.AtomRings() for i in r}
    bhs = sorted(i for i in ring_atoms if sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring_atoms) >= 3)
    if len(bhs) < 2:
        pytest.skip(f"{smi} is spiro (1 shared atom, not a bridged system) -- no bridgehead pair to compare")
    adj = _adj(mol)
    for s in bhs:
        for e in bhs:
            if s == e: continue
            assert pc._find_all_simple_paths(mol, s, e, ring_atoms) == pc._find_all_simple_paths(mol, s, e, ring_atoms, adj=adj)
            sub = ring_atoms - {min(ring_atoms - {s, e})}
            assert pc._find_all_simple_paths(mol, s, e, sub) == pc._find_all_simple_paths(mol, s, e, sub, adj=adj)

def test_expansion_cap_fires_identically(monkeypatch):
    mol = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45"); ring_atoms = set(range(mol.GetNumAtoms()))
    monkeypatch.setattr(pc, "_MAX_DFS_EXPANSIONS", 7)
    a = pc._find_all_simple_paths(mol, 0, 5, ring_atoms)
    b = pc._find_all_simple_paths(mol, 0, 5, ring_atoms, adj=_adj(mol))
    assert a == b


def test_paths_memo_hits_within_scope_and_replays_units(monkeypatch):
    """R2(b): the same (mol, start, end, allowed) enumeration inside one naming
    scope runs the DFS once; the second call is served from the memo, charges the
    SAME perf-budget units (so any PerfBudgetExceeded trip is unchanged), and
    returns fresh list copies (callers filter/slice the result)."""
    from orthonym.assembly import memo
    from orthonym.assembly import fragment_naming as fn
    mol = Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3"); ring_atoms = set(range(mol.GetNumAtoms()))
    tok = memo.push_scope()
    try:
        fn._fragment_guard.perf_budget = 1_000_000
        a = pc._find_all_simple_paths(mol, 0, 5, ring_atoms); spent_a = 1_000_000 - fn._fragment_guard.perf_budget
        calls = []
        real = mol.GetAtomWithIdx
        monkeypatch.setattr(type(mol), "GetAtomWithIdx", lambda self, i: (calls.append(i), real(i))[1])
        fn._fragment_guard.perf_budget = 1_000_000
        b = pc._find_all_simple_paths(mol, 0, 5, ring_atoms); spent_b = 1_000_000 - fn._fragment_guard.perf_budget
        assert a == b and a is not b and a[0] is not b[0]     # equal, but copies
        assert calls == []                                    # no DFS ran
        assert spent_a == spent_b and spent_a > 0             # same budget trajectory
    finally:
        fn._fragment_guard.perf_budget = None
        memo.pop_scope(tok)


def test_main_ring_memo_is_shared_across_mol_objects_with_same_atom_order():
    """R2(a): the von Baeyer main-ring search is memoised by molecular STRUCTURE
    (canonical SMILES + atom output order + bonds + ring atoms + bridgeheads), so
    two DIFFERENT mol objects parsed from the same SMILES with the SAME atom order
    share the result inside one naming scope; a reparse with a DIFFERENT atom order
    recomputes because the atom indices (and therefore the answer) differ."""
    from orthonym.assembly import memo
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    # NB: the SMILES must NOT already be in canonical atom order, or the canonical
    # reparse below yields the SAME atom order and the memo correctly (not wrongly)
    # shares -- which would make the different-order assertion untestable. This
    # bridged bicyclic reorders on reparse (its input != canonical order).
    smi = "C2CC1CCCC1C2"
    m1, m2 = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(smi)
    ring_atoms = set(range(m1.GetNumAtoms()))
    bhs = {i for i in ring_atoms if m1.GetAtomWithIdx(i).GetDegree() >= 3}
    an = VonBaeyerAnalyzer()
    tok = memo.push_scope()
    try:
        r1 = an._find_main_ring(m1, ring_atoms, bhs)
        n_impl = []
        real = an._find_main_ring_impl
        an._find_main_ring_impl = lambda *a, **k: (n_impl.append(1), real(*a, **k))[1]
        r2 = an._find_main_ring(m2, ring_atoms, bhs)          # same structure, same atom order
        assert r1 == r2 and n_impl == []
        m3 = Chem.MolFromSmiles(Chem.MolToSmiles(m1))          # same molecule, DIFFERENT atom order
        ring3 = set(range(m3.GetNumAtoms())); bh3 = {i for i in ring3 if m3.GetAtomWithIdx(i).GetDegree() >= 3}
        an._find_main_ring(m3, ring3, bh3)
        assert n_impl == [1]                                   # recomputed: indices differ
    finally:
        memo.pop_scope(tok)
