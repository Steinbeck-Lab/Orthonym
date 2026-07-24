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


def test_from_token_bindings_coerces_unmapped_role_and_reports_it():
    # An unknown legacy role must widen to PREFIX (the only role a
    # substituent-shaped token can safely be assumed to play) and the RAW
    # role string must survive into stats -- never silently absorbed.
    res = GeneralEngineResult(name="ethan-1-ol", bindings=(
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "linker"),
    ))
    spine = bs.BindingSpine.from_token_bindings(res.bindings)
    assert [x.kind for x in spine.walk()] == [bs.BindingKind.PARENT,
                                              bs.BindingKind.PREFIX]
    assert spine.legacy_role_coerced == ("linker",)
    proof = bs.verify_spine(ETHANOL, spine, res.name)
    assert proof.stats["legacy_role_coerced"] == ("linker",)


PROPANOL = Chem.MolFromSmiles("CCCO")
TOLUENE = Chem.MolFromSmiles("Cc1ccccc1")  # 0 = methyl C, 1-6 = ring


def _bond(mol, i, j):
    return mol.GetBondBetweenAtoms(i, j).GetIdx()


def test_p2_single_cross_bond_is_an_inferred_linkage():
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
    ), "methylbenzene")
    assert p.ok, p.findings
    assert p.stats["linkages_inferred"] == 1


def test_p2_declared_linkage_bond_is_claimed_not_inferred():
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0,
           bond_ids=frozenset([_bond(TOLUENE, 0, 1)])),
    ), "methylbenzene")
    assert p.ok, p.findings
    assert p.stats["linkages_inferred"] == 0


def test_p2_double_claimed_bond_is_error():
    bidx = _bond(TOLUENE, 0, 1)
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6],
           bond_ids=frozenset([bidx])),
        _b("methyl", bs.BindingKind.PREFIX, [0], bond_ids=frozenset([bidx])),
    ), "methylbenzene")
    assert not p.ok
    assert bs.BOND_DOUBLE_CLAIMED in p.codes()


def test_p2_two_undeclared_cross_bonds_are_ambiguous():
    # cyclohexane spelled as two disjoint 3-atom bindings: the two ring
    # closure bonds join the same pair -> the name cannot be saying that.
    mol = Chem.MolFromSmiles("C1CCCCC1")
    p = bs.verify_spine(mol, _spine(
        _b("prop", bs.BindingKind.PARENT, [0, 1, 2]),
        _b("propyl", bs.BindingKind.PREFIX, [3, 4, 5]),
    ), "propylpropane")
    assert not p.ok
    assert bs.BOND_AMBIGUOUS_LINKAGE in p.codes()


def test_p2_multivalent_suffix_on_one_atom_is_not_ambiguous():
    # 3-aminobutanoic acid, EXACTLY as the production producers bind it:
    # the single token "-oic acid" spells BOTH the C=O and the C-OH bond, and
    # both land on the same parent carbon. Two cross bonds, but no bridge and
    # nothing to disambiguate -- refusing this would refuse carboxylic acids.
    mol = Chem.MolFromSmiles("CC(N)CC(=O)O")  # 4 = acid C, 5 = O, 6 = O
    p = bs.verify_spine(mol, _spine(
        _b("but", bs.BindingKind.PARENT, [0, 1, 3, 4]),
        _b("amino", bs.BindingKind.PREFIX, [2]),
        _b("oic acid", bs.BindingKind.SUFFIX, [5, 6]),
    ), "3-aminobutanoic acid")
    assert p.ok, p.findings
    assert p.stats["linkages_inferred"] == 3


def test_p2_cycle_spanning_three_subtrees_is_ambiguous():
    # Cyclohexane spelled as THREE disjoint 2-atom bindings: only ONE
    # undeclared bond joins each pair, so per-pair counting sees nothing
    # wrong -- yet the three together close a ring no substituent says.
    mol = Chem.MolFromSmiles("C1CCCCC1")
    p = bs.verify_spine(mol, _spine(
        _b("a", bs.BindingKind.PARENT, [0, 1]),
        _b("b", bs.BindingKind.PREFIX, [2, 3]),
        _b("c", bs.BindingKind.PREFIX, [4, 5]),
    ), "nonsense")
    assert not p.ok
    assert bs.BOND_AMBIGUOUS_LINKAGE in p.codes()


def test_p2_declared_cross_bond_constrains_later_inference():
    # Same split cyclohexane, but one ring-closure bond IS declared. The
    # declared bond already joins the two halves, so the OTHER one closes a
    # cycle and must not be quietly inferred as an attachment.
    mol = Chem.MolFromSmiles("C1CCCCC1")
    p = bs.verify_spine(mol, _spine(
        _b("prop", bs.BindingKind.PARENT, [0, 1, 2],
           bond_ids=frozenset([_bond(mol, 2, 3)])),
        _b("propyl", bs.BindingKind.PREFIX, [3, 4, 5]),
    ), "propylpropane")
    assert not p.ok
    assert bs.BOND_AMBIGUOUS_LINKAGE in p.codes()


def test_p2_strict_mode_rejects_undeclared_linkage():
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0]),
    ), "methylbenzene", mode="strict")
    assert not p.ok
    assert bs.BOND_UNCLAIMED in p.codes()


def test_deepest_owner_is_the_nested_binding_not_an_ancestor():
    # a(0) -> b(1) -> c(2,3) over butane. Every ancestor's subtree contains
    # bond 2-3, but only c's own morpheme spells it, so c must win. Tested
    # directly because the winner is not otherwise observable: a bond claimed
    # by the wrong depth is still claimed exactly once.
    grandchild = _b("c", bs.BindingKind.PREFIX, [2, 3])
    child = _b("b", bs.BindingKind.PREFIX, [1], children=(grandchild,))
    root = _b("a", bs.BindingKind.PARENT, [0], children=(child,))
    nodes = bs._flatten(_spine(root))
    assert [n.binding.token for n in bs._deepest_owners(nodes, 2, 3)] == ["c"]
    assert [n.binding.token for n in bs._deepest_owners(nodes, 1, 2)] == ["b"]
    assert [n.binding.token for n in bs._deepest_owners(nodes, 0, 1)] == ["a"]
    assert bs._deepest_owners(nodes, 0, 9) == []


def test_deepest_owner_reports_an_impossible_tie_rather_than_guessing():
    # Two disjoint bindings at equal depth both containing both endpoints
    # cannot happen while P1 passes; if it does, both are returned so the
    # caller can refuse instead of silently picking one.
    left = _b("x", bs.BindingKind.PREFIX, [0, 1])
    right = _b("y", bs.BindingKind.PREFIX, [0, 1])
    nodes = bs._flatten(_spine(left, right))
    assert sorted(n.binding.token
                  for n in bs._deepest_owners(nodes, 0, 1)) == ["x", "y"]


def test_p2_nested_attachment_bond_is_internal_not_a_linkage():
    # SAME molecule and SAME atom claims as the sibling-roots case above,
    # but methyl nested INSIDE benzene: the attachment bond now lies within
    # one subtree, so it is claimed internally and nothing is inferred.
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6],
           children=(_b("methyl", bs.BindingKind.PREFIX, [0]),)),
    ), "methylbenzene")
    assert p.ok, p.findings
    assert p.stats["bonds_total"] == 7
    assert p.stats["bonds_claimed"] == 7
    assert p.stats["linkages_inferred"] == 0


def test_p2_every_bond_is_accounted_for():
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0]),
    ), "methylbenzene")
    assert p.ok, p.findings
    assert (p.stats["bonds_claimed"] + p.stats["linkages_inferred"]
            == p.stats["bonds_total"] == 7)


def test_p2_is_skipped_when_p1_fails():
    # Broken partition -> bond ownership is undecidable, so P2 must not
    # report derivative bond findings and must not claim to have run.
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
    ), "benzene")
    assert not p.ok
    assert p.codes() == (bs.ATOM_UNBOUND,)
    assert p.stats["p2_skipped"] is True
    assert p.stats["proofs"] == ("P1", "P3")
    assert "bonds_total" not in p.stats


def test_p2_ignores_bonds_to_explicit_hydrogens():
    # P1 partitions HEAVY atoms only, so P2 must scope to heavy-heavy bonds
    # or a mol carrying explicit Hs would manufacture unclaimed bonds.
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert p.ok, p.findings
    assert p.stats["bonds_total"] == 2
    assert p.stats["bonds_hydrogen"] == 6


def test_p2_unknown_declared_bond_id_claims_nothing():
    # A declared id that is not a heavy-heavy bond of this mol must not be
    # credited as a claim -- the real attachment bond stays a linkage.
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0],
           bond_ids=frozenset([99])),
    ), "methylbenzene")
    assert p.stats["bonds_declared_unknown"] == 1
    assert p.stats["linkages_inferred"] == 1


def test_p3_double_claimed_charge_is_error():
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1],
           charge_atom_ids=frozenset([2])),
        _b("olate", bs.BindingKind.SUFFIX, [2],
           charge_atom_ids=frozenset([2])),
    ), "ethanolate", allow_charged=True)
    assert not p.ok
    assert bs.CHARGE_DOUBLE_CLAIMED in p.codes()


def test_p3_unclaimed_charge_is_error():
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2]),
    ), "ethanolate", allow_charged=True)
    assert not p.ok
    assert bs.CHARGE_UNCLAIMED in p.codes()


def test_p3_claimed_charge_passes_under_allow_charged():
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2],
           charge_atom_ids=frozenset([2])),
    ), "ethanolate", allow_charged=True)
    assert p.ok, p.findings


def test_p3_net_charge_out_of_scope_is_e1_parity():
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2],
           charge_atom_ids=frozenset([2])),
    ), "ethanolate")  # allow_charged defaults False
    assert not p.ok
    assert bs.NET_CHARGE_OUT_OF_SCOPE in p.codes()
