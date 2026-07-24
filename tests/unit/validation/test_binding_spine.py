"""v29 Phase 1: the recursion-safe name<->graph binding spine."""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import GeneralEngineResult, TokenBinding
from orthonym.validation import binding_spine as bs

pytestmark = pytest.mark.unit

ETHANOL = Chem.MolFromSmiles("CCO")  # 0 C, 1 C, 2 O


def _b(token, kind, atoms, **kw):
    return bs.SpineBinding(token=token, kind=kind,
                           atom_ids=frozenset(atoms), **kw)


def _spine(*roots):
    return bs.BindingSpine(roots=tuple(roots))


def test_subtree_atoms_unions_children():
    child = _b("methyl", bs.BindingKind.PREFIX, [7])
    parent = _b("phenyl", bs.BindingKind.PREFIX, [1, 2, 3, 4, 5, 6],
                children=(child,))
    assert parent.atom_ids == frozenset([1, 2, 3, 4, 5, 6])
    assert parent.subtree_atoms() == frozenset([1, 2, 3, 4, 5, 6, 7])


def test_walk_is_preorder_self_first():
    child = _b("methyl", bs.BindingKind.PREFIX, [7])
    parent = _b("phenyl", bs.BindingKind.PREFIX, [1], children=(child,))
    assert [x.token for x in _spine(parent).walk()] == ["phenyl", "methyl"]


def test_complete_partition_ok():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert p.ok, p.findings


def test_unbound_atom_is_error():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
    ), "ethane")
    assert not p.ok
    assert bs.ATOM_UNBOUND in p.codes()


def test_double_bound_atom_is_error():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [1, 2]),
    ), "ethan-1-ol")
    assert not p.ok
    assert bs.ATOM_DOUBLE_BOUND in p.codes()


def test_phantom_atom_is_error():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2, 99]),
    ), "ethan-1-ol")
    assert not p.ok
    assert bs.ATOM_PHANTOM in p.codes()


def test_child_atom_counts_toward_partition():
    # exclusive parent + child together cover the molecule -> ok
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1],
           children=(_b("ol", bs.BindingKind.SUFFIX, [2]),)),
    ), "ethan-1-ol")
    assert p.ok, p.findings


def test_child_overlapping_parent_is_double_bound():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1, 2],
           children=(_b("ol", bs.BindingKind.SUFFIX, [2]),)),
    ), "ethan-1-ol")
    assert not p.ok
    assert bs.ATOM_DOUBLE_BOUND in p.codes()


def test_from_token_bindings_adapts_legacy_flat_result():
    res = GeneralEngineResult(name="ethan-1-ol", bindings=(
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ))
    spine = bs.BindingSpine.from_token_bindings(res.bindings)
    assert [x.token for x in spine.walk()] == ["eth", "ol"]
    assert [x.kind for x in spine.walk()] == [bs.BindingKind.PARENT,
                                              bs.BindingKind.SUFFIX]
    assert all(x.children == () for x in spine.walk())
    assert bs.verify_spine(ETHANOL, spine, res.name).ok
