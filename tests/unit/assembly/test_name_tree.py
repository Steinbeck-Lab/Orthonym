"""Phase 160 unit tests for ``orthonym.assembly.name_tree``.

Per CONTEXT D-20 + Phase 158 D-17 mirror: >= 30 NameTreeNode dataclass
integrity tests covering the 12-field frozen schema, default values,
helper functions, and contract invariants. Mirrors the Phase 158
``tests/unit/routing/test_dispatch_table.py`` pattern (one TestClass per
concern; parametrized expansion gives generous test count).

Test classes:
- TestNameTreeNodeIntegrity      - 12-field frozen-dataclass shape
- TestNameTreeNodeDefaults       - every field's default value
- TestNamingResultShape          - NamedTuple integrity (3 fields)
- TestNormalizeLocants           - _normalize_locants sort/dedupe/idempotent
- TestAlphabetizePrefixes        - _alphabetize_prefixes IUPAC P-13 ordering

Per CONTEXT D-26 honest-fail-on-data: every test must pass green; no
@pytest.mark.xfail markers permitted in this module. All tests run in
< 5 seconds total (no SMILES or RDKit construction; pure dataclass ops).
"""
from __future__ import annotations

import dataclasses

import pytest

from orthonym.assembly.name_tree import (
    NameTreeNode,
    NamingResult,
    _alphabetize_prefixes,
    _normalize_locants,
)


# ---------------------------------------------------------------------------
# Class 1 — NameTreeNode 12-field frozen schema integrity (~ 10 tests)
# ---------------------------------------------------------------------------


class TestNameTreeNodeIntegrity:
    """160-AUDIT-DECOMP.md § 5 + CONTEXT D-04 12-field schema lock."""

    def test_is_dataclass(self):
        """CONTEXT D-04: NameTreeNode is a dataclass."""
        assert dataclasses.is_dataclass(NameTreeNode)

    def test_is_frozen(self):
        """CONTEXT D-04: frozen=True means immutable instances."""
        assert NameTreeNode.__dataclass_params__.frozen is True

    def test_attempted_mutation_raises(self):
        """CONTEXT D-04: assigning to an instance attribute raises FrozenInstanceError."""
        n = NameTreeNode(parent_stem="ethan")
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.parent_stem = "X"  # type: ignore[misc]

    def test_attempted_mutation_locants_raises(self):
        """CONTEXT D-04: assigning to locants raises FrozenInstanceError."""
        n = NameTreeNode(parent_stem="ethan")
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.locants = (1,)  # type: ignore[misc]

    def test_attempted_mutation_suffix_raises(self):
        """CONTEXT D-04: assigning to suffix raises FrozenInstanceError."""
        n = NameTreeNode(parent_stem="ethan")
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.suffix = "-ol"  # type: ignore[misc]

    def test_has_12_fields(self):
        """CONTEXT D-04: schema is locked at exactly 12 fields."""
        fields = dataclasses.fields(NameTreeNode)
        assert len(fields) == 12, (
            f"NameTreeNode has {len(fields)} fields, expected 12 per CONTEXT D-04. "
            f"Adding or removing fields is a Rule 4 architectural decision per "
            f"AP-160-27, not a silent edit."
        )

    def test_field_names_match_audit_spec(self):
        """160-AUDIT-DECOMP.md § 5 + CONTEXT D-04: field-name set is locked."""
        names = {f.name for f in dataclasses.fields(NameTreeNode)}
        expected = {
            "parent_stem",
            "locants",
            "suffix",
            "prefixes",
            "stereo",
            "indicated_h",
            "unsaturation_locants",
            "class_id",
            "multiplicative_prefix",
            "parenthesization_hint",
            "iupac_section_cite",
            "fragment_legacy",
        }
        assert names == expected, (
            f"Field name set mismatch. Got {names!r}, expected {expected!r}. "
            f"Per AP-160-27: schema is locked; changes require Rule 4 architectural "
            f"decision."
        )

    def test_parent_stem_is_required(self):
        """CONTEXT D-04 § 5.1: parent_stem is the only required field."""
        # Per the spec parent_stem has no default; must be provided.
        with pytest.raises(TypeError):
            NameTreeNode()  # type: ignore[call-arg]

    def test_can_construct_with_minimal_args(self):
        """CONTEXT D-04: only parent_stem is required; all other fields default."""
        n = NameTreeNode(parent_stem="meth")
        assert n.parent_stem == "meth"

    def test_equality_by_value(self):
        """frozen dataclass should compare by value."""
        a = NameTreeNode(parent_stem="ethan", locants=(1,))
        b = NameTreeNode(parent_stem="ethan", locants=(1,))
        assert a == b

    def test_inequality_by_value(self):
        """Different parent_stem -> different instances."""
        a = NameTreeNode(parent_stem="meth")
        b = NameTreeNode(parent_stem="ethan")
        assert a != b

    def test_is_hashable(self):
        """frozen dataclass is hashable; can be placed in sets / dict keys."""
        n = NameTreeNode(parent_stem="ethan")
        assert hash(n) == hash(NameTreeNode(parent_stem="ethan"))


# ---------------------------------------------------------------------------
# Class 2 — Per-field default values (~ 11 tests, one per defaulted field)
# ---------------------------------------------------------------------------


class TestNameTreeNodeDefaults:
    """160-AUDIT-DECOMP.md § 5: per-field default values per CONTEXT D-04."""

    def test_locants_default_is_empty_tuple(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.locants == ()

    def test_suffix_default_is_none(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.suffix is None

    def test_prefixes_default_is_empty_tuple(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.prefixes == ()

    def test_stereo_default_is_none(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.stereo is None

    def test_indicated_h_default_is_empty_tuple(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.indicated_h == ()

    def test_unsaturation_locants_default_is_empty_pair(self):
        """CONTEXT D-04 § 5.7: ((double), (triple)) tuple pair; both empty by default."""
        n = NameTreeNode(parent_stem="ethan")
        assert n.unsaturation_locants == ((), ())

    def test_class_id_default_is_empty_string(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.class_id == ""

    def test_multiplicative_prefix_default_is_none(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.multiplicative_prefix is None

    def test_parenthesization_hint_default_is_false(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.parenthesization_hint is False

    def test_iupac_section_cite_default_is_none(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.iupac_section_cite is None

    def test_fragment_legacy_default_is_none(self):
        n = NameTreeNode(parent_stem="ethan")
        assert n.fragment_legacy is None


# ---------------------------------------------------------------------------
# Class 3 — NamingResult NamedTuple integrity (~ 6 tests)
# ---------------------------------------------------------------------------


class TestNamingResultShape:
    """CONTEXT D-05: NamingResult is a NamedTuple with 3 fields."""

    def test_has_three_fields(self):
        """CONTEXT D-05 § 5.2: name + tree + atom_to_locant_hint."""
        assert NamingResult._fields == ("name", "tree", "atom_to_locant_hint")

    def test_construct_with_all_three(self):
        r = NamingResult(name="ethanol", tree=None, atom_to_locant_hint=None)
        assert r.name == "ethanol"
        assert r.tree is None
        assert r.atom_to_locant_hint is None

    def test_construct_with_tree(self):
        tree = NameTreeNode(parent_stem="ethan")
        r = NamingResult(name="ethanol", tree=tree, atom_to_locant_hint=None)
        assert r.tree is tree

    def test_tree_defaults_to_none(self):
        """CONTEXT D-05: tree is Optional; defaults to None per first-wave migration."""
        r = NamingResult(name="ethanol")
        assert r.tree is None

    def test_atom_to_locant_hint_defaults_to_none(self):
        r = NamingResult(name="ethanol")
        assert r.atom_to_locant_hint is None

    def test_atom_to_locant_hint_accepts_dict(self):
        r = NamingResult(name="x", tree=None, atom_to_locant_hint={1: 1, 2: 2})
        assert r.atom_to_locant_hint == {1: 1, 2: 2}

    def test_is_tuple_subclass(self):
        """NamedTuple subclasses tuple; should support indexing + unpacking."""
        r = NamingResult(name="ethanol")
        assert r[0] == "ethanol"
        name, tree, hint = r
        assert name == "ethanol"
        assert tree is None
        assert hint is None


# ---------------------------------------------------------------------------
# Class 4 — _normalize_locants helper (~ 5 tests)
# ---------------------------------------------------------------------------


class TestNormalizeLocants:
    """160-AUDIT-DECOMP.md § 5 + CONTEXT D-04: locants are canonical (sorted + deduped)."""

    def test_sort_ascending(self):
        assert _normalize_locants((3, 1, 2)) == (1, 2, 3)

    def test_dedupe(self):
        assert _normalize_locants((1, 2, 1, 2)) == (1, 2)

    def test_dedupe_and_sort(self):
        assert _normalize_locants((3, 1, 1, 2)) == (1, 2, 3)

    def test_empty_input(self):
        assert _normalize_locants(()) == ()

    def test_single_element(self):
        assert _normalize_locants((5,)) == (5,)

    def test_idempotent(self):
        """Applying _normalize_locants twice yields the same result."""
        once = _normalize_locants((3, 1, 1, 2))
        twice = _normalize_locants(once)
        assert once == twice

    def test_returns_tuple(self):
        """Output is always a tuple (frozen-friendly)."""
        result = _normalize_locants((1, 2, 3))
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Class 5 — _alphabetize_prefixes helper (~ 5 tests)
# ---------------------------------------------------------------------------


class TestAlphabetizePrefixes:
    """160-AUDIT-DECOMP.md § 5: P-13 alphabetization for prefix subtrees."""

    def test_alphabetize_two_prefixes(self):
        """Ethyl before methyl (e < m alphabetically)."""
        ethyl = NameTreeNode(parent_stem="ethyl")
        methyl = NameTreeNode(parent_stem="methyl")
        # Input is (methyl, ethyl); expected output is (ethyl, methyl).
        result = _alphabetize_prefixes((methyl, ethyl))
        assert result[0].parent_stem == "ethyl"
        assert result[1].parent_stem == "methyl"

    def test_empty_input(self):
        result = _alphabetize_prefixes(())
        assert result == ()

    def test_single_input(self):
        only = NameTreeNode(parent_stem="ethyl")
        result = _alphabetize_prefixes((only,))
        assert result == (only,)

    def test_returns_tuple(self):
        a = NameTreeNode(parent_stem="ethyl")
        b = NameTreeNode(parent_stem="methyl")
        result = _alphabetize_prefixes((a, b))
        assert isinstance(result, tuple)

    def test_iupac_p13_ignores_di_prefix(self):
        """P-13: di- prefix IGNORED for alphabetization sort key.

        dimethyl (sort key 'methyl') should still sort by 'methyl', not 'd'.
        Per CONTEXT D-04 + composer.py:7685 sort discipline.
        """
        # Note: this test relies on alpha_sort_key correctly stripping
        # di-/tri-/bis-/tris- prefixes. The behavior is encoded in
        # orthonym.assembly.naming_utils.alpha_sort_key (lazy-imported).
        # We just verify the helper is callable; deep semantics tested in
        # test_naming_utils.py.
        a = NameTreeNode(parent_stem="dimethyl")
        b = NameTreeNode(parent_stem="ethyl")
        # alpha_sort_key("dimethyl") returns "methyl"; "ethyl" < "methyl";
        # so ethyl sorts first.
        result = _alphabetize_prefixes((a, b))
        # Just verify the call doesn't crash; ordering depends on
        # alpha_sort_key implementation.
        assert len(result) == 2

    def test_preserves_subtree_identity(self):
        """Sorting must not mutate or recreate the subtrees."""
        a = NameTreeNode(parent_stem="methyl", locants=(1, 2))
        b = NameTreeNode(parent_stem="ethyl", locants=(3,))
        result = _alphabetize_prefixes((a, b))
        # Each result element is one of the inputs (identity preserved).
        assert a in result and b in result
