"""Lever K (a performance pass): two distinct mol objects with the same indexed graph share one FG detection."""
import pytest
from rdkit import Chem
from orthonym.assembly import memo
from orthonym.perception import functional_groups as fg


@pytest.fixture(autouse=True)
def _isolate_process_cache():
    """These tests count calls to the impl, so they must not see entries another test left
    in the process-wide pure cache (perf lever A8, 2026-09-13)."""
    memo.clear_process_cache()
    yield
    memo.clear_process_cache()


def test_same_structure_different_objects_share_result(monkeypatch):
    calls = {"n": 0}
    orig = fg._detect_functional_groups_impl
    def spy(mol):
        calls["n"] += 1
        return orig(mol)
    monkeypatch.setattr(fg, "_detect_functional_groups_impl", spy)
    a = Chem.MolFromSmiles("CC(=O)NCC(=O)O"); b = Chem.MolFromSmiles("CC(=O)NCC(=O)O")
    tok = memo.push_scope()
    try:
        ra = fg.detect_functional_groups(a); rb = fg.detect_functional_groups(b)
    finally:
        memo.pop_scope(tok)
    assert ra == rb == orig(Chem.MolFromSmiles("CC(=O)NCC(=O)O"))
    if fg._FG_MODE != "verify":      # verify mode recomputes on every hit by design
        assert calls["n"] == 1


def test_different_atom_order_is_a_miss(monkeypatch):
    calls = {"n": 0}
    orig = fg._detect_functional_groups_impl
    def spy(mol):
        calls["n"] += 1
        return orig(mol)
    monkeypatch.setattr(fg, "_detect_functional_groups_impl", spy)
    a = Chem.MolFromSmiles("OC(=O)CNC(C)=O"); b = Chem.MolFromSmiles("CC(=O)NCC(=O)O")   # same molecule, other atom order
    tok = memo.push_scope()
    try:
        ra = fg.detect_functional_groups(a); rb = fg.detect_functional_groups(b)
    finally:
        memo.pop_scope(tok)
    assert ra != rb                       # index tuples differ, as they must
    if fg._FG_MODE != "verify":
        assert calls["n"] == 2


def test_no_scope_still_correct():
    m = Chem.MolFromSmiles("CCN")
    assert memo._cache_var.get() is None
    assert fg.detect_functional_groups(m) == fg._detect_functional_groups_impl(m)


def _bonds_added_in_reverse(smiles):
    """``smiles`` and the same molecule with the same atom order whose bonds were added in the reverse order."""
    a = Chem.MolFromSmiles(smiles)
    rw = Chem.RWMol()
    for at in a.GetAtoms():
        rw.AddAtom(Chem.Atom(at.GetAtomicNum()))
    for b in reversed(list(a.GetBonds())):
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return a, m


def test_another_neighbour_order_is_a_miss():
    """One canonical SMILES and one atom order, bonds added in another order: the matcher walks each atom's neighbours
    in that order, so a match lists its atoms in another order ('secondary_alcohol' (2, 1, 0, 3) and (2, 1, 3, 0)).
    The key carries the neighbour order, so each molecule gets its own value, whichever reached the process-wide entry
    first (``ORTHONYM_MEMO=verify`` found the other's value served on 12 of 360 panel rows)."""
    a, b = _bonds_added_in_reverse("CC(O)CC")
    assert Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
    assert a.GetProp("_smilesAtomOutputOrder") == b.GetProp("_smilesAtomOutputOrder")
    fresh_a, fresh_b = fg._detect_functional_groups_impl(a), fg._detect_functional_groups_impl(b)
    assert fresh_a != fresh_b
    tok = memo.push_scope()
    try:
        assert fg.detect_functional_groups(a) == fresh_a
        assert fg.detect_functional_groups(b) == fresh_b
    finally:
        memo.pop_scope(tok)
