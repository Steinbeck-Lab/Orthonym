"""``fused_ring_selection.select_base_component`` is memoised for the naming call.

One naming asks for the base component of the same molecule and components again and again (22,965 of 23,205 calls on
the typical speed panel repeated a question of the same naming; 8% of its CPU). The memo keys on the molecule's graph
and the components' atom sets and stores their order, so a hit gives the caller its own component objects in the
order a fresh ranking gives.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import memo
from orthonym.rules import fused_ring_selection as frs

pytestmark = pytest.mark.unit

# quinoline: the benzene and pyridine rings as components; pyridine (N) is the base (.3 (a))
QUINOLINE = "c1ccc2ncccc2c1"


def _components(mol):
    return [set(r) for r in mol.GetRingInfo().AtomRings()]


def test_a_hit_gives_the_fresh_order_with_the_callers_own_sets(monkeypatch):
    mol = Chem.MolFromSmiles(QUINOLINE)
    fresh_base, fresh_rest = frs.select_base_component(mol, _components(mol))
    assert any(mol.GetAtomWithIdx(a).GetSymbol() == "N" for a in fresh_base)
    calls = []
    real = frs._ranked_atom_sets
    monkeypatch.setattr(frs, "_ranked_atom_sets", lambda m, s: calls.append(1) or real(m, s))
    tok = memo.push_scope()
    try:
        first = frs.select_base_component(mol, _components(mol))
        comps = list(reversed(_components(Chem.MolFromSmiles(QUINOLINE))))
        base, rest = frs.select_base_component(Chem.MolFromSmiles(QUINOLINE), comps)
    finally:
        memo.pop_scope(tok)
    assert (first[0], first[1]) == (fresh_base, fresh_rest) == (base, rest)
    assert base is comps[[i for i, c in enumerate(comps) if c == base][0]]
    if memo._MODE == "on":
        assert len(calls) == 1
