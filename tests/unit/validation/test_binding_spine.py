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
    # P4 and P6 read only the name and the tokens, so they still run and are
    # often what explains the P1 failure; P5 needs P4's spans to be meaningful,
    # which a non-partitioning token set does not give.
    assert p.stats["p5_skipped"] is True
    # P7, like P4 and P6, reads each binding against the graph on its own and
    # needs nothing from the partition, so it still runs when P1 has failed.
    assert p.stats["proofs"] == ("P1", "P3", "P4", "P6", "P7")
    assert "bonds_total" not in p.stats
    assert "residue_runs" not in p.stats


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


def test_p3_no_charge_claims_anywhere_is_unverified_not_disproven():
    """A spine that declares NO charge claims at all is missing EVIDENCE, not
    contradicting the graph. The 12 legacy production producers structurally
    cannot populate charge_atom_ids, so calling this an error would make P3
    refuse every charged legacy-adapted spine even when the name is right."""
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2]),
    ), "ethanolate", allow_charged=True)
    assert bs.CHARGE_UNVERIFIED in p.codes()
    assert bs.CHARGE_UNCLAIMED not in p.codes()
    assert p.ok, p.findings          # unproven does not fail the audit
    assert p.stats["charge_claims_declared"] == 0


def test_p3_no_charge_claims_is_an_error_in_strict_mode():
    """strict leaves nothing unproven -- the same spine must fail there.

    Asserts the SEVERITY of the CHARGE_UNVERIFIED finding itself, not merely
    ``not p.ok``: strict mode also raises BOND_UNCLAIMED on this spine's
    undeclared linkage, so an ``ok``-only assertion would pass even if the
    charge severity never escalated (verified by mutation)."""
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2]),
    ), "ethanolate", mode="strict", allow_charged=True)
    charge_findings = [f for f in p.findings if f.code == bs.CHARGE_UNVERIFIED]
    assert len(charge_findings) == 1
    assert charge_findings[0].severity == "error"
    assert not p.ok


def test_p3_partial_charge_claims_leave_a_real_unclaimed_charge():
    """Once SOME binding declares a charge claim, an uncovered charged atom is
    a provable disagreement, not missing evidence -- CHARGE_UNCLAIMED, error."""
    mol = Chem.MolFromSmiles("[NH3+]CC[NH3+]")  # atoms 0 and 3 both charged
    charged = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]
    assert charged == [0, 3]
    p = bs.verify_spine(mol, _spine(
        _b("ethane", bs.BindingKind.PARENT, [1, 2]),
        _b("azanium", bs.BindingKind.SUFFIX, [0],
           charge_atom_ids=frozenset([0])),
        _b("amine", bs.BindingKind.SUFFIX, [3]),   # claims the atom, not its charge
    ), "ethane-1,2-diazanium", allow_charged=True)
    assert bs.CHARGE_UNCLAIMED in p.codes()
    assert bs.CHARGE_UNVERIFIED not in p.codes()
    assert not p.ok
    assert p.stats["charge_claims_declared"] == 1


def test_attachment_is_an_atom_of_the_fragment_itself():
    """Pins the semantics of SpineBinding.attachment against the reading that
    it names the PARENT-side atom. For toluene's methyl, attachment is atom 0
    (the methyl's own carbon); atom 1 (the ring carbon it bonds to) would be
    the other, wrong reading."""
    methyl = _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0)
    assert methyl.attachment in methyl.atom_ids
    ring = _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6])
    assert methyl.attachment not in ring.atom_ids


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


def test_p4_token_present_with_clean_boundary_passes():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert p.ok, p.findings


def test_p4_absent_token_is_error():
    p = bs.verify_spine(ETHANOL, _spine(
        _b("prop", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert not p.ok
    assert bs.TOKEN_ABSENT in p.codes() or bs.ARITY_MISMATCH in p.codes()


def test_p4_two_methyls_share_one_multiplied_span():
    mol = Chem.MolFromSmiles("CN(C)c1ccccc1")  # N,N-dimethylaniline skeleton
    p = bs.verify_spine(mol, _spine(
        _b("aniline", bs.BindingKind.PARENT, [1, 3, 4, 5, 6, 7, 8]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
        _b("methyl", bs.BindingKind.PREFIX, [2], attachment=2),
    ), "N,N-dimethylaniline")
    assert bs.MULTIPLICITY_MISMATCH not in p.codes(), p.findings
    assert bs.TOKEN_ABSENT not in p.codes()


def test_p4_three_bindings_against_a_di_multiplier_is_multiplicity_mismatch():
    mol = Chem.MolFromSmiles("CC(C)(C)c1ccccc1")
    p = bs.verify_spine(mol, _spine(
        _b("benzene", bs.BindingKind.PARENT, [4, 5, 6, 7, 8, 9]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
        _b("methyl", bs.BindingKind.PREFIX, [2], attachment=2),
        _b("methyl", bs.BindingKind.PREFIX, [3], attachment=3),
        _b("meth", bs.BindingKind.PARENT, [1]),
    ), "dimethylbenzene")
    assert bs.MULTIPLICITY_MISMATCH in p.codes()


def test_p4_substring_only_occurrence_is_error():
    """'eth' sits inside 'methane' with no morpheme boundary on its left.

    'm' is neither glue nor another binding's token, so no occurrence is
    boundary-valid while the raw substring is plainly there.
    """
    mol = Chem.MolFromSmiles("CC")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
    ), "methane")
    assert bs.TOKEN_SUBSTRING_ONLY in p.codes()
    assert bs.TOKEN_ABSENT not in p.codes()
    assert not p.ok


def test_p4_elision_vowel_glue_makes_phen_in_phenanthrene_undecidable_for_p4():
    """The limit of the boundary rule, pinned so it is not mistaken for a proof.

    'phen|anthrene' IS boundary-valid: '_RIGHT_GLUE' carries the elision
    vowels 'a'/'an', which it must, or every '...an-1-ol' would be refused.
    P4 therefore cannot call this one, and P6 cannot either (the oracle
    soundly refuses 'phen'). What DOES catch it is P5: 'threne' is name text
    no binding accounts for. Recorded as warn in audit, error in strict --
    never a silent pass.
    """
    mol = Chem.MolFromSmiles("CC")
    p = bs.verify_spine(mol, _spine(
        _b("phen", bs.BindingKind.PARENT, [0, 1]),
    ), "phenanthrene")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes()
    assert p.stats["residue_runs"] == ("threne",)
    assert bs.UNBOUND_MORPHEME in p.codes()


def test_p5_unexplained_morpheme_in_the_name_is_reported():
    """A whole functional suffix no binding claims, reported by P5.

    P1 says every ATOM is spelled; P5 asks the mirror question and is the only
    proof that sees a name carrying text for atoms that are not there.
    """
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
    ), "4-methylbenzene-1-carbaldehyde")
    assert bs.UNBOUND_MORPHEME in p.codes()
    assert p.stats["residue_runs"] == ("carbaldehyde",)
    assert any("carbaldehyde" in f.detail for f in p.findings)


def test_p5_strict_mode_escalates_unexplained_text_to_an_error():
    """Mirrors P3's CHARGE_UNVERIFIED escalation: audit warns, strict refuses.

    Asserts the SEVERITY of the UNBOUND_MORPHEME finding itself -- strict also
    raises BOND_UNCLAIMED on this spine's undeclared linkage, so an ``ok``-only
    assertion would pass even if the severity never escalated.
    """
    spine = _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
    )
    audit = bs.verify_spine(TOLUENE, spine, "4-methylbenzene-1-carbaldehyde")
    strict = bs.verify_spine(TOLUENE, spine, "4-methylbenzene-1-carbaldehyde",
                             mode="strict")
    assert [f.severity for f in audit.findings
            if f.code == bs.UNBOUND_MORPHEME] == ["warn"]
    assert [f.severity for f in strict.findings
            if f.code == bs.UNBOUND_MORPHEME] == ["error"]
    assert audit.ok          # unexplained text does not fail the audit


def test_p5_atom_bearing_right_glue_is_a_known_blind_spot():
    """Characterisation, NOT an endorsement: an unclaimed sulfonic acid is
    consumed as glue because '_RIGHT_GLUE' -- which the boundary rule needs --
    carries atom-bearing morphemes ('sulfon', 'ic'). Splitting the boundary
    lexicon from the residue lexicon would tighten this at the price of false
    errors on every unlisted connective, so the trade is deliberate. If a
    later phase splits them, this test fails and forces a conscious update.
    """
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
    ), "methylbenzenesulfonic acid")
    assert bs.UNBOUND_MORPHEME not in p.codes()
    assert p.stats["residue_runs"] == ()


def test_p6_headline_a_token_may_not_claim_atoms_it_does_not_spell():
    # The exact hole this phase closes: one token claiming the whole molecule.
    mol = Chem.MolFromSmiles("c1ccccc1C[As](=O)(O)O")  # 11 heavy atoms
    assert mol.GetNumHeavyAtoms() == 11
    p = bs.verify_spine(mol, _spine(
        _b("methylbenzene", bs.BindingKind.PARENT, range(11)),
    ), "methylbenzene")
    assert not p.ok
    assert bs.ARITY_MISMATCH in p.codes()


def test_p6_recursion_parent_claiming_child_atoms_is_caught():
    mol = Chem.MolFromSmiles("Cc1ccccc1C")  # xylene, 8 heavy atoms
    bad = bs.verify_spine(mol, _spine(
        _b("phenyl", bs.BindingKind.PREFIX, range(8)),
    ), "phenyl")
    assert bs.ARITY_MISMATCH in bad.codes()

    good = bs.verify_spine(mol, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6],
           children=(
               _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
               _b("methyl", bs.BindingKind.PREFIX, [7], attachment=7),
           )),
    ), "dimethylbenzene")
    assert bs.ARITY_MISMATCH not in good.codes(), good.findings


def test_p6_unverified_arity_is_info_not_failure():
    mol = Chem.MolFromSmiles("CC")
    p = bs.verify_spine(mol, _spine(
        _b("quuxyl", bs.BindingKind.PREFIX, [0, 1]),
    ), "quuxyl")
    assert bs.ARITY_UNVERIFIED in p.codes()
    assert all(f.severity != "error" for f in p.findings
               if f.code == bs.ARITY_UNVERIFIED)


def test_p4_a_non_letter_token_edge_is_itself_a_boundary():
    """A von Baeyer parent, the shape the real producers actually emit.

    Found by auditing 100 real emissions: 5 of 24 were refused
    TOKEN_SUBSTRING_ONLY because the token 'bicyclo[4.3.0]' is followed by
    'nonane', and 'n' happens to be missing from _RIGHT_GLUE (it is in
    _LEFT_GLUE). But the token ENDS IN ']' -- there is no letter run for it to
    bleed into, so no glue evidence is needed and refusing it was confidently
    wrong. Same on the left for a token opening with '(' or a locant.
    """
    mol = Chem.MolFromSmiles("C1CCC2CCCC2C1")   # hydrindane, 9 C
    p = bs.verify_spine(mol, _spine(
        _b("bicyclo[4.3.0]", bs.BindingKind.PARENT, range(9)),
    ), "bicyclo[4.3.0]nonane")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes(), p.findings
    assert p.stats["spans"] == ((0, 14, "bicyclo[4.3.0]"),)
    assert p.ok, p.findings
    # This molecule alone does not ISOLATE the clause: 'nonane' also happens to
    # start with the multiplier 'nona', which is right-glue in its own right.
    # Pin the clause on a tail that no lexicon can explain, or a mutation that
    # deletes it survives here (measured -- it did).
    assert bs._right_ok("bicyclo[4.3.0]quux", 14, "bicyclo[4.3.0]",
                        frozenset())


def test_p4_complex_multiplier_counts_through_its_enclosing_mark():
    """'bis(' -- the complex multiplicative prefix is separated from the group
    it multiplies by an enclosing mark (P-16.3.2), so the alphabetic run
    immediately before the token is EMPTY and a naive scan reads weight 1.
    Found on real data: every bis/tris/tetrakis prefix was refused
    MULTIPLICITY_MISMATCH.
    """
    mol = Chem.MolFromSmiles("ClCCNCCCl")
    p = bs.verify_spine(mol, _spine(
        _b("2-chloroethyl", bs.BindingKind.PREFIX, [0, 1, 2], attachment=2),
        _b("amine", bs.BindingKind.SUFFIX, [3]),
        _b("2-chloroethyl", bs.BindingKind.PREFIX, [4, 5, 6], attachment=4),
    ), "bis(2-chloroethyl)amine")
    assert bs.MULTIPLICITY_MISMATCH not in p.codes(), p.findings
    assert p.ok, p.findings


def test_p4_one_binding_against_a_di_multiplier_overshoots():
    """The other direction of the multiplicity check: the name says the token
    twice, only one binding claims it. Nothing atom-level sees this."""
    p = bs.verify_spine(TOLUENE, _spine(
        _b("benzene", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
    ), "dimethylbenzene")
    assert bs.MULTIPLICITY_MISMATCH in p.codes()
    assert not p.ok


def test_p4_shorter_token_blocked_by_a_longer_one_reports_overlap():
    """'meth' is a prefix of 'methyl'. Longest-first assignment stops the short
    token stealing the long one's span; the short one then reports the
    COLLISION rather than a multiplicity claim it cannot justify."""
    mol = Chem.MolFromSmiles("CC(C)(C)c1ccccc1")
    p = bs.verify_spine(mol, _spine(
        _b("benzene", bs.BindingKind.PARENT, [4, 5, 6, 7, 8, 9]),
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
        _b("methyl", bs.BindingKind.PREFIX, [2], attachment=2),
        _b("meth", bs.BindingKind.PARENT, [1]),
    ), "dimethylbenzene")
    assert bs.TOKEN_SPAN_OVERLAP in p.codes()
    assert bs.MULTIPLICITY_MISMATCH not in p.codes()


def test_p4_and_p6_skip_a_binding_with_no_text_of_its_own():
    """CHARGE/HYDRO kinds may carry no token; an empty string would otherwise
    match everywhere and score an arity of nothing."""
    mol = Chem.MolFromSmiles("CC[O-]")
    p = bs.verify_spine(mol, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("olate", bs.BindingKind.SUFFIX, [2], charge_atom_ids=frozenset([2])),
        _b("", bs.BindingKind.CHARGE, [], charge_atom_ids=frozenset()),
    ), "ethanolate", allow_charged=True)
    assert p.stats["empty_tokens"] == 1
    assert bs.TOKEN_ABSENT not in p.codes()
    assert p.stats["arity_confident"] + p.stats["arity_unverified"] == 2


def test_p4_a_suffix_after_a_saturation_ending_is_at_a_boundary():
    """The morpheme that most often precedes a suffix is the saturation or
    unsaturation ending -- 'propan|amide', 'hexan|-1-ol', 'octadec-9-yne|
    nitrile'. Those endings were listed only as things that may FOLLOW a
    token, so a suffix sitting after one was refused TOKEN_SUBSTRING_ONLY
    (measured on real emissions: 'octadec-9-ynenitrile').

    A morpheme boundary is evidenced by a KNOWN MORPHEME abutting it; which
    side of a token a morpheme usually sits on is not part of that evidence.
    Both sides therefore consult the same lexicon.
    """
    mol = Chem.MolFromSmiles("CCC#N")            # propanenitrile, 4 heavy
    p = bs.verify_spine(mol, _spine(
        _b("prop", bs.BindingKind.PARENT, [0, 1, 2]),
        _b("nitrile", bs.BindingKind.SUFFIX, [3]),
    ), "propanenitrile")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes(), p.findings
    assert p.ok, p.findings


def test_p4_multiplier_with_its_vowel_elided_still_counts():
    """P-16.3.3: a multiplying prefix drops its terminal vowel before a
    vowel-initial suffix -- 'butane-1,2,3,4-tetraol' is written '...-tetrol'.

    The elided form is the SAME morpheme, so it must serve both as boundary
    evidence (or the suffix after it is refused TOKEN_SUBSTRING_ONLY -- the
    last false error left on real data, on a '...-pentol') and as the
    occurrence's weight (or four -ol bindings against one written 'ol' are
    refused MULTIPLICITY_MISMATCH).
    """
    mol = Chem.MolFromSmiles("OCC(O)C(O)CO")     # butane-1,2,3,4-tetrol
    p = bs.verify_spine(mol, _spine(
        _b("butane", bs.BindingKind.PARENT, [1, 2, 4, 6]),
        _b("ol", bs.BindingKind.SUFFIX, [0]),
        _b("ol", bs.BindingKind.SUFFIX, [3]),
        _b("ol", bs.BindingKind.SUFFIX, [5]),
        _b("ol", bs.BindingKind.SUFFIX, [7]),
    ), "butane-1,2,3,4-tetrol")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes(), p.findings
    assert bs.MULTIPLICITY_MISMATCH not in p.codes(), p.findings
    assert p.ok, p.findings


def test_p4_a_multiplier_may_follow_a_token_as_well_as_precede_one():
    """'propane|diamide' -- the morpheme after the parent is 'di', a
    multiplier. Multipliers were filed as things that PRECEDE a token, so a
    side-specific lexicon refuses this while accepting the mirror image. The
    boundary rule must not care which side a morpheme usually sits on."""
    mol = Chem.MolFromSmiles("NC(=O)CC(=O)N")     # propanediamide, 7 heavy
    p = bs.verify_spine(mol, _spine(
        _b("propane", bs.BindingKind.PARENT, [1, 3, 4]),
        _b("amide", bs.BindingKind.SUFFIX, [0, 2]),
        _b("amide", bs.BindingKind.SUFFIX, [5, 6]),
    ), "propanediamide")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes(), p.findings
    assert p.ok, p.findings


def test_p4_a_group_may_need_several_occurrences_to_reach_its_count():
    """Two methyl bindings, written in two SEPARATE places rather than under
    one multiplier: 'methyl 2-methylpropanoate'. Taking only the first
    occurrence leaves the group short and invents a multiplicity fault."""
    mol = Chem.MolFromSmiles("COC(=O)C(C)C")      # methyl 2-methylpropanoate
    p = bs.verify_spine(mol, _spine(
        _b("methyl", bs.BindingKind.PREFIX, [0], attachment=0),
        _b("methyl", bs.BindingKind.PREFIX, [6], attachment=6),
        _b("prop", bs.BindingKind.PARENT, [2, 4, 5]),
        _b("oate", bs.BindingKind.SUFFIX, [1, 3]),
    ), "methyl 2-methylpropanoate")
    assert bs.MULTIPLICITY_MISMATCH not in p.codes(), p.findings
    assert len([s for s in p.stats["spans"] if s[2] == "methyl"]) == 2
    assert p.ok, p.findings


def test_occurrence_weight_reads_the_longest_multiplier_and_sees_past_brackets():
    """Directly, because the wrong reading is not otherwise observable without
    a twelve-binding fixture: 'dodeca' must read 12, not the 'deca' (10) it
    ends with, and 'bis(' must be seen through its enclosing mark."""
    assert bs._occurrence_weight("dodecaol", 6) == 12     # not deca = 10
    assert bs._occurrence_weight("undecaol", 6) == 11     # not deca = 10
    assert bs._occurrence_weight("bis(ethyl)", 4) == 2
    assert bs._occurrence_weight("4-methyl", 2) == 1      # locant, not a count
    assert bs._occurrence_weight("methyl", 0) == 1        # nothing precedes


def test_a_right_side_elision_vowel_is_not_evidence_of_a_left_boundary():
    """The boundary lexicons must be DIRECTIONAL, or P4 passes a fabrication.

    '_RIGHT_GLUE' has to carry the single-letter elision vowels ('e', 'a', 'o')
    or every '...an-1-ol' is refused. But a lone letter sitting before an
    ARBITRARY cut point evidences nothing, so if the same list is consulted on
    the left, the fabricated token 'than' anchors inside 'ethan-1-ol' -- its
    'evidence' being the 'e' in front of it -- and the whole spine passes with
    ok=True. Which side a morpheme is observed on is what makes it evidence.
    """
    p = bs.verify_spine(ETHANOL, _spine(
        _b("than", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert bs.TOKEN_SUBSTRING_ONLY in p.codes(), p.findings
    assert not p.ok


def test_the_boundary_lexicons_are_directional_in_both_directions():
    """Each side admits only the morphemes observed on THAT side.

    Tested on the helpers directly: a whole-spine fixture cannot isolate one
    side, and the two failure directions need different names to exhibit.
    """
    # A right-only morpheme ('e', an elision vowel) is not left evidence...
    assert not bs._left_ok("ethan-1-ol", 1, "than", frozenset({"than"}))
    # ...while on the right it must still hold, or 'phen|anthrene' and every
    # '...an-1-ol' would be refused.
    assert bs._right_ok("phenanthrene", 4, "phen", frozenset({"phen"}))
    # And the mirror: a left-only morpheme ('cyclo' opens a ring name, it never
    # trails one) is not right evidence.
    assert not bs._right_ok("benzenecyclohexane", 7, "benzene", frozenset())
    assert bs._left_ok("cyclohexane", 5, "hexane", frozenset())


def test_p6_a_spine_with_nothing_arity_confident_is_proof_unsubstantiated():
    """The residual P4 structurally cannot close, made explicit.

    'bu'+'tane' over butane: two fabricated tokens, each the other's boundary
    evidence. No boundary rule can separate 'bu|tane' from 'but|ane' -- both cut
    the string in the same two places -- so P4 rightly passes both, and the
    oracle refuses both as morphemes. The proof has therefore established
    NOTHING about what the name spells, and must say so rather than report a
    bare ok=True.
    """
    butane = Chem.MolFromSmiles("CCCC")
    p = bs.verify_spine(butane, _spine(
        _b("bu", bs.BindingKind.PARENT, [0, 1]),
        _b("tane", bs.BindingKind.PARENT, [2, 3]),
    ), "butane")
    assert bs.TOKEN_SUBSTRING_ONLY not in p.codes()      # P4 cannot call it
    assert p.stats["arity_confident"] == 0
    assert p.stats["arity_confident_atom_frac"] == 0.0
    assert bs.PROOF_UNSUBSTANTIATED in p.codes()
    assert [f.severity for f in p.findings
            if f.code == bs.PROOF_UNSUBSTANTIATED] == ["warn"]


def test_p6_proof_unsubstantiated_is_an_error_in_strict_mode():
    """Mirrors how P3 and P5 escalate: audit warns, strict refuses.

    Asserts the SEVERITY of the finding itself -- strict also raises
    BOND_UNCLAIMED on this spine's undeclared linkage, so an ``ok``-only
    assertion would pass even if the severity never escalated.
    """
    butane = Chem.MolFromSmiles("CCCC")
    spine = _spine(
        _b("bu", bs.BindingKind.PARENT, [0, 1]),
        _b("tane", bs.BindingKind.PARENT, [2, 3]),
    )
    strict = bs.verify_spine(butane, spine, "butane", mode="strict")
    assert [f.severity for f in strict.findings
            if f.code == bs.PROOF_UNSUBSTANTIATED] == ["error"]
    assert not strict.ok


def test_p6_a_corroborated_spine_is_not_reported_unsubstantiated():
    """The other direction: the residual must not fire on a normal spine.

    'eth' is a chain stem the oracle knows, so its 2 atoms ARE independently
    corroborated and the fraction is 1.0. A finding that fired here would make
    every correct emission look unproven.
    """
    p = bs.verify_spine(ETHANOL, _spine(
        _b("eth", bs.BindingKind.PARENT, [0, 1]),
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "ethan-1-ol")
    assert p.stats["arity_confident_atom_frac"] == 1.0
    assert p.stats["arity_confident_atoms"] == 3
    assert p.stats["arity_claimed_atoms"] == 3
    assert bs.PROOF_UNSUBSTANTIATED not in p.codes()
    assert p.ok, p.findings


def test_p6_partial_corroboration_does_not_raise_the_residual():
    """Zero is the line, and it is the ONLY line this phase can defend.

    One confident token among unconfident ones is not "mostly unproven" that a
    threshold could grade -- it is one independent statement the producer did
    not make, which is qualitatively more than none. Any cut above zero would
    be a tuning knob no measurement supports, so this pins that a partially
    corroborated spine stays quiet.
    """
    p = bs.verify_spine(ETHANOL, _spine(
        _b("quuxyl", bs.BindingKind.PREFIX, [0]),   # oracle refuses it
        _b("eth", bs.BindingKind.PARENT, [1]),      # 1 of 3 claimed atoms
        _b("ol", bs.BindingKind.SUFFIX, [2]),
    ), "quuxylethan-1-ol")
    assert p.stats["arity_unverified"] >= 1
    assert 0.0 < p.stats["arity_confident_atom_frac"] < 1.0
    assert bs.PROOF_UNSUBSTANTIATED not in p.codes()


def test_the_two_boundary_evidence_clauses_are_independently_load_bearing():
    """Pins the two clauses that the shared lexicon usually MASKS.

    Most real neighbours happen to be listed morphemes, so these two rarely
    decide anything -- but each is the only evidence available when the
    neighbour is not listed, which real names do produce (a von Baeyer token
    followed by an unlisted stem such as 'docosan-'). Tested on the helpers
    directly because a whole-spine fixture cannot isolate them.
    """
    # (a) the neighbour is ANOTHER BINDING'S TOKEN and nothing else
    both = frozenset({"quux", "benzene"})
    assert bs._left_ok("quuxbenzene", 4, "benzene", both)
    assert not bs._left_ok("quuxbenzene", 4, "benzene", frozenset({"benzene"}))
    # (b) the TOKEN'S OWN edge is not a letter, so it cannot bleed
    assert bs._right_ok("bicyclo[2.2.1]quux", 14, "bicyclo[2.2.1]", frozenset())
    assert not bs._right_ok("benzenequux", 7, "benzene", frozenset())
