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
