"""Phase 168 Plan-02: triviality-controller unit tests.

Mirrors the Phase 167 HYG-04 class-per-bug discipline (analog:
tests/unit/assembly/test_diaryl_methyl_substituent.py). Tests construct ``NameTreeNode``
directly and exercise the per-Type dispatch + the frozen-dataclass invariants WITHOUT the
perception layer / OPSIN / candidate_pool wiring — the per-Type checks and ``_build_rewrite``
are pure functions of (node, seed_entry), so they are deterministic without recovery.

Pins the reviews-iter-1 fixes: #2 (conditional fragment_legacy reset via
``_replace_preserving_or_resetting_legacy``) and #3 (the hint-based path removed; zero
``atom_to_locant_hint`` references; 3-path recovery).

Source: 168-CONTEXT.md D-01/D-02/D-04/D-06/D-07/D-08; 168-REVIEWS.md #2 + #3.
"""

import dataclasses
import inspect

import pytest
from rdkit import Chem

from orthonym.assembly.name_tree import NameTreeNode, is_coarse_node
from orthonym.assembly import retained_substitution as rs
from orthonym.assembly.retained_substitution import (
    apply_triviality_controller,
    _build_rewrite,
    _recompute_multiplicative_prefix,
    _replace_preserving_or_resetting_legacy,
    _type_1_check,
    _type_2a_check,
    _type_2c_check,
    _type_3_check,
)
from orthonym.data.triviality_controller_seed import SEED_TABLE


def _entry(smiles):
    return SEED_TABLE[Chem.CanonSmiles(smiles)]


class TestStageAInvariants:
    """enabled=False is a no-op; coarse nodes pass through unchanged (D-04 + D-08)."""

    @pytest.mark.unit
    def test_enabled_false_returns_input_unchanged(self):
        n = NameTreeNode(parent_stem="benzene")
        out = apply_triviality_controller(n, None, None, enabled=False)
        assert out is n

    @pytest.mark.unit
    def test_coarse_node_returns_unchanged(self):
        n = NameTreeNode(parent_stem="benzenamine", fragment_legacy="benzenamine")
        assert is_coarse_node(n)
        out = apply_triviality_controller(n, None, None, enabled=True)
        assert out is n


class TestType1Branch:
    """Type 1 (P-15.1.8.1): unconditional swap on a canonical-SMILES match."""

    @pytest.mark.unit
    def test_type_1_check_always_true(self):
        assert _type_1_check(NameTreeNode(parent_stem="x"), _entry("c1ccoc1")) is True

    @pytest.mark.unit
    def test_build_rewrite_to_furan(self):
        entry = _entry("c1ccoc1")
        node = NameTreeNode(parent_stem="1-oxacyclopenta-2,4-diene", fragment_legacy="legacy")
        out = _build_rewrite(node, entry, ())
        assert out.parent_stem == "furan"
        assert out.fragment_legacy is None
        assert out.iupac_section_cite == entry.iupac_p_section


class TestType2aBranch:
    """Type 2a (P-15.1.8.2.1): principal-group-bound swap; refuses on PG mismatch (Pitfall 2)."""

    @pytest.mark.unit
    def test_phenol_swap_with_matching_pg(self):
        entry = _entry("Oc1ccccc1")
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), "primary_alcohol", entry) is True

    @pytest.mark.unit
    def test_phenol_refuses_swap_when_pg_mismatch(self):
        entry = _entry("Oc1ccccc1")
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), "carboxylic_acid", entry) is False

    @pytest.mark.unit
    def test_phenol_refuses_swap_when_pg_none(self):
        entry = _entry("Oc1ccccc1")
        assert _type_2a_check(NameTreeNode(parent_stem="benzenol"), None, entry) is False


class TestType2bBranch:
    """Type 2b (P-15.1.8.2.2): closed-list; bare reduces to unconditional."""

    @pytest.mark.unit
    def test_formic_acid_present(self):
        assert Chem.CanonSmiles("OC=O") in SEED_TABLE
        assert _entry("OC=O").substitution_type.value == "type_2b"

    @pytest.mark.unit
    def test_formic_acid_bare_is_allowed(self):
        # No prefixes => Type 2b reduces to Type 1 unconditional.
        from orthonym.assembly.retained_substitution import _type_2b_check
        entry = _entry("OC=O")
        assert _type_2b_check(NameTreeNode(parent_stem="methanoic acid"),
                              Chem.MolFromSmiles("OC=O"), entry, None) is True


class TestType2cBranch:
    """Type 2c (P-15.1.8.2.3): default-to-Type-3 when no locus override (anisole/hydroxylamine)."""

    @pytest.mark.unit
    def test_anisole_bare_swap_allowed(self):
        entry = _entry("COc1ccccc1")
        assert entry.locus_override_rule_id is None
        assert _type_2c_check(NameTreeNode(parent_stem="methoxybenzene"),
                              Chem.MolFromSmiles("COc1ccccc1"), entry) is True

    @pytest.mark.unit
    def test_anisole_refuses_swap_with_ring_substituent(self):
        entry = _entry("COc1ccccc1")
        child = NameTreeNode(parent_stem="methyl", locants=(2,))
        node = NameTreeNode(parent_stem="methoxybenzene", prefixes=(child,))
        assert _type_2c_check(node, Chem.MolFromSmiles("COc1ccccc1C"), entry) is False


class TestType3Branch:
    """Type 3 (P-15.1.8.3): bare-only + locant_context (toluene/xylene; Pitfall 3)."""

    @pytest.mark.unit
    def test_toluene_bare_swap_allowed(self):
        assert _type_3_check(NameTreeNode(parent_stem="methylbenzene"), _entry("Cc1ccccc1")) is True

    @pytest.mark.unit
    def test_toluene_refuses_swap_with_chlorine(self):
        child = NameTreeNode(parent_stem="chloro")
        node = NameTreeNode(parent_stem="methylbenzene", prefixes=(child,))
        assert _type_3_check(node, _entry("Cc1ccccc1")) is False

    @pytest.mark.unit
    def test_xylene_1_2_locant_context_match(self):
        entry = _entry("Cc1ccccc1C")
        assert entry.locant_context == (1, 2)
        # Bare node with matching locants passes; mismatched locants fail.
        assert _type_3_check(NameTreeNode(parent_stem="o-xylene-stem", locants=(1, 2)), entry) is True
        assert _type_3_check(NameTreeNode(parent_stem="o-xylene-stem", locants=(1, 3)), entry) is False


class TestFrozenDataclassInvariants:
    """Rewrites never mutate the input; parent-changing rewrite resets fragment_legacy;
    TRIV-02 multiplier is re-derived via the SSOT predicates (NOT statically None)."""

    @pytest.mark.unit
    def test_rewrite_does_not_mutate_input(self):
        entry = _entry("Oc1ccccc1")
        node = NameTreeNode(parent_stem="benzenol", fragment_legacy="legacy")
        _ = _build_rewrite(node, entry, ())
        assert node.parent_stem == "benzenol"  # original unchanged
        assert node.fragment_legacy == "legacy"

    @pytest.mark.unit
    def test_build_rewrite_resets_fragment_legacy(self):
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", fragment_legacy="x"), entry, ())
        assert out.fragment_legacy is None  # #2 / Pitfall 1

    @pytest.mark.unit
    def test_multiplier_none_stays_none(self):
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", multiplicative_prefix=None), entry, ())
        assert out.multiplicative_prefix is None

    @pytest.mark.unit
    def test_multiplier_recompute_simple_keeps_di(self):
        # phenol is a SIMPLE substituent name (no digits/hyphens) => di
        entry = _entry("Oc1ccccc1")
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "di"

    @pytest.mark.unit
    def test_multiplier_recompute_complex_flips_to_bis(self):
        # "1,2-xylene" is COMPLEX (digit + hyphen) => di flips to bis (TRIV-02 di<->bis feedback)
        entry = _entry("Cc1ccccc1C")
        out = _build_rewrite(NameTreeNode(parent_stem="x", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "bis"


class TestFragmentLegacyConditionalReset:
    """#2 FIX (reviews iter 1): _replace_preserving_or_resetting_legacy resets fragment_legacy
    when a child changed (the serializer short-circuit would otherwise discard the rewrite) and
    preserves it when nothing changed."""

    @pytest.mark.unit
    def test_no_child_change_preserves_fragment_legacy(self):
        c1 = NameTreeNode(parent_stem="bromo")
        node = NameTreeNode(parent_stem="benzene", prefixes=(c1,), fragment_legacy="X")
        # Pass the SAME (already-alphabetized single) prefixes => nothing changed => preserve.
        out = _replace_preserving_or_resetting_legacy(node, node.prefixes)
        assert out.fragment_legacy == "X"

    @pytest.mark.unit
    def test_child_changed_resets_fragment_legacy(self):
        c1 = NameTreeNode(parent_stem="bromo")
        node = NameTreeNode(parent_stem="benzene", prefixes=(c1,), fragment_legacy="X")
        new_child = NameTreeNode(parent_stem="chloro")  # a DIFFERENT child
        out = _replace_preserving_or_resetting_legacy(node, (new_child,))
        assert out.fragment_legacy is None  # child changed => reset (serializer-short-circuit closed)


class TestNoPathA:
    """#3 FIX (reviews iter 1): the hint-based recovery path was removed — the module source
    carries zero ``atom_to_locant_hint`` references; recovery is a 3-path B/C/D cascade."""

    @pytest.mark.unit
    def test_no_atom_to_locant_hint_reference(self):
        source = inspect.getsource(rs)
        assert "atom_to_locant_hint" not in source

    @pytest.mark.unit
    def test_recover_has_three_paths_not_path_a(self):
        source = inspect.getsource(rs)
        assert "Path B" in source and "Path C" in source and "Path D" in source
        assert "Path A" not in source
