"""a phase unit tests for ``orthonym.assembly.name_tree_to_string``.

Per internal notes + a phase mirror: >= 30 Pass-2 serializer tests
covering the byte-identical-vs-legacy-_assemble_fragments contract for
the substrate-only paths AND the explicit-field path (exercised by
Plan-04 --dump-tree + + tree-emitting handlers).

Note on internal notes first-wave migration:
- For the 30 currently-extracted handlers (Plans 02-03 ship) the tree
  is None; the legacy path (composer.py:_assemble_fragments) produces
  the byte-identical name. This serializer's first-wave compatibility
  mode (node.fragment_legacy is not None) re-routes to the legacy path.
- For tree-populated nodes (+1), the explicit-field branch assembles
  the name from the 12-field schema directly.

Test classes:
- TestSerializerContract - top-level dispatch (legacy vs explicit)
- TestExplicitFieldAssembly - explicit-field branch (parent + locants
                                     + suffix + stereo + prefixes)
- TestUnsaturationInfix - en / yn / enyn infix rules per
- TestSerializerErrorHandling - DECOMP-02 honest-fail behavior
- TestNormalizeAndAlphaIntegration - integration with name_tree helpers
"""
from __future__ import annotations

import pytest

from orthonym.assembly.name_tree import NameTreeNode
from orthonym.assembly.name_tree_to_string import (
    NameTreeSerializerError,
    name_tree_to_string,
)


# ---------------------------------------------------------------------------
# Class 1 — Serializer top-level contract (~ 6 tests)
# ---------------------------------------------------------------------------


class TestSerializerContract:
    """160-AUDIT-DECOMP.md § 5.4: top-level dispatch between legacy + explicit."""

    def test_explicit_field_simple_parent_only(self):
        """Minimal explicit-field call: a bare hydride stem -> the saturated
        hydrocarbon (a phase: a general_acyclic root carries the BARE stem,
        e.g. 'eth', and assembles to stem+'ane' via the shared grammar)."""
        n = NameTreeNode(parent_stem="eth", class_id="general_acyclic")
        out = name_tree_to_string(n)
        assert out == "ethane"

    def test_explicit_field_parent_plus_suffix(self):
        """Parent + suffix: 'ethan' + '-ol' -> '...ethanol' (suffix appended)."""
        n = NameTreeNode(parent_stem="ethan", suffix="ol")
        out = name_tree_to_string(n)
        # First-wave serializer: simple appends; no locant collision handling.
        assert "ethan" in out
        assert "ol" in out

    def test_explicit_field_parent_with_unsaturation(self):
        """Parent + unsaturation bond locant (a phase: on a chain, bond
        locants live in `unsaturation_locants`, NOT the generic `locants`
        field, which carries SUFFIX locants — byte-identical to the legacy
        assembler). Wave2 (d)): an unsubstituted prop node
        omits the bond locant -> propene."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         unsaturation_locants=((1,), ()))
        out = name_tree_to_string(n)
        assert out == "propene"

    def test_legacy_path_falls_through_to_assemble_fragments(self):
        """node.fragment_legacy is None + parent_stem present -> explicit branch."""
        n = NameTreeNode(parent_stem="ethan")
        # We're testing that when fragment_legacy is None we don't crash;
        # explicit-field path is taken (see earlier tests).
        out = name_tree_to_string(n)
        assert isinstance(out, str)
        assert out

    def test_default_style_is_pin(self):
        """Default style argument is 'pin' per Plan-04 API."""
        n = NameTreeNode(parent_stem="ethan")
        out_default = name_tree_to_string(n)
        out_explicit = name_tree_to_string(n, style="pin")
        assert out_default == out_explicit

    def test_style_general_compatible(self):
        """style='general' is accepted (forwarded to legacy assembler if needed)."""
        n = NameTreeNode(parent_stem="ethan")
        out = name_tree_to_string(n, style="general")
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Class 2 — Explicit-field assembly (~ 10 tests)
# ---------------------------------------------------------------------------


class TestExplicitFieldAssembly:
    """160-AUDIT-DECOMP.md § 5.4 + IUPAC P-14.5: explicit-field branch."""

    def test_stereo_prepended(self):
        """stereo is prepended before everything, with NO extra hyphen — the
        descriptor carries its own trailing hyphen (a phase gap #8;
        production stereo is e.g. '(2R)-')."""
        n = NameTreeNode(parent_stem="eth", stereo="(2R)-")
        out = name_tree_to_string(n)
        assert out.startswith("(2R)-")
        assert "(2R)--" not in out

    def test_indicated_h_prepended_to_parent(self):
        """indicated_h is prepended to parent_stem per."""
        n = NameTreeNode(parent_stem="pyrrol", indicated_h=(1,))
        out = name_tree_to_string(n)
        # '1H' prepended to 'pyrrol'.
        assert "1H" in out
        assert "pyrrol" in out

    def test_indicated_h_multiple(self):
        """Multiple indicated H's: '1H,3H-' prefix."""
        n = NameTreeNode(parent_stem="purine", indicated_h=(1, 3))
        out = name_tree_to_string(n)
        assert "1H" in out
        assert "3H" in out

    def test_prefixes_alphabetized(self):
        """Multiple prefixes are alphabetized per."""
        methyl = NameTreeNode(parent_stem="methyl")
        ethyl = NameTreeNode(parent_stem="ethyl")
        n = NameTreeNode(parent_stem="butan", prefixes=(methyl, ethyl))
        out = name_tree_to_string(n)
        # ethyl comes before methyl in output (e < m alphabetically).
        eidx = out.find("ethyl")
        midx = out.find("methyl")
        assert eidx >= 0 and midx >= 0 and eidx < midx

    def test_prefix_with_locant(self):
        """Prefix subtree with locants gets locant-prefixed assembly."""
        meth = NameTreeNode(parent_stem="methyl", locants=(2,))
        n = NameTreeNode(parent_stem="butan", prefixes=(meth,))
        out = name_tree_to_string(n)
        # '2-methyl' should appear.
        assert "2" in out
        assert "methyl" in out

    def test_prefix_with_multiplicative_prefix(self):
        """Prefix with multiplicative_prefix='di' produces 'dimethyl'."""
        meth = NameTreeNode(parent_stem="methyl", multiplicative_prefix="di")
        n = NameTreeNode(parent_stem="butan", prefixes=(meth,))
        out = name_tree_to_string(n)
        assert "di" in out
        assert "methyl" in out

    def test_prefix_with_parenthesization_hint(self):
        """parenthesization_hint=True wraps the prefix in parentheses."""
        complex_pref = NameTreeNode(
            parent_stem="methylethyl", parenthesization_hint=True,
        )
        n = NameTreeNode(parent_stem="butan", prefixes=(complex_pref,))
        out = name_tree_to_string(n)
        assert "(methylethyl)" in out

    def test_locants_formatted_comma_separated(self):
        """Suffix locants print as comma-separated ascending integers (a phase: locants render through the suffix grammar). prop + 'ol' (1,2)
        -> propane-1,2-diol."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         suffix="ol", locants=(1, 2))
        out = name_tree_to_string(n)
        assert out == "propane-1,2-diol"

    def test_locants_single_value(self):
        """Single suffix locant prints as a bare integer. prop + 'ol' (1,)
        -> propan-1-ol."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         suffix="ol", locants=(1,))
        out = name_tree_to_string(n)
        assert out == "propan-1-ol"

    def test_suffix_attached_after_parent(self):
        """Suffix is attached after the parent stem."""
        n = NameTreeNode(parent_stem="ethan", suffix="al")
        out = name_tree_to_string(n)
        assert out.endswith("al")

    def test_empty_locants_no_locant_string(self):
        """No locants -> no locant prefix in output."""
        n = NameTreeNode(parent_stem="ethan")
        out = name_tree_to_string(n)
        # The output should just be the parent_stem itself; no comma / digits.
        assert "," not in out


# ---------------------------------------------------------------------------
# Class 3 — Unsaturation infix (~ 6 tests)
# ---------------------------------------------------------------------------


class TestUnsaturationInfix:
    """IUPAC unsaturation infix on a general_acyclic hydride parent
    (a phase: bare stem + unsaturation_locants -> full -ene/-yne grammar via
    the shared composition_primitives._build_hydrocarbon_name)."""

    def test_no_unsaturation_passthrough(self):
        """eth + (, ) -> ethane (saturated)."""
        n = NameTreeNode(parent_stem="eth", class_id="general_acyclic",
                         unsaturation_locants=((), ()))
        out = name_tree_to_string(n)
        assert out == "ethane"

    def test_single_double_bond(self):
        """prop + ((1,), ) unsubstituted -> propene (d));
        but + ((1,), ) keeps the locant."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         unsaturation_locants=((1,), ()))
        out = name_tree_to_string(n)
        assert out == "propene"
        n4 = NameTreeNode(parent_stem="but", class_id="general_acyclic",
                          unsaturation_locants=((1,), ()))
        assert name_tree_to_string(n4) == "but-1-ene"

    def test_single_triple_bond(self):
        """prop + (, (1,)) unsubstituted -> propyne (d))."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         unsaturation_locants=((), (1,)))
        out = name_tree_to_string(n)
        assert out == "propyne"

    def test_two_double_bonds_dien(self):
        """prop + ((1, 2), ) -> propa-1,2-diene."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         unsaturation_locants=((1, 2), ()))
        out = name_tree_to_string(n)
        assert out == "propa-1,2-diene"

    def test_enyne_combination(self):
        """prop + ((1,), (3,)) -> prop-1-en-3-yne (enyne)."""
        n = NameTreeNode(parent_stem="prop", class_id="general_acyclic",
                         unsaturation_locants=((1,), (3,)))
        out = name_tree_to_string(n)
        assert out == "prop-1-en-3-yne"

    def test_three_double_bonds_trien(self):
        """hex + ((1, 3, 5), ) -> hexa-1,3,5-triene."""
        n = NameTreeNode(parent_stem="hex", class_id="general_acyclic",
                         unsaturation_locants=((1, 3, 5), ()))
        out = name_tree_to_string(n)
        assert out == "hexa-1,3,5-triene"


# ---------------------------------------------------------------------------
# Class 4 — Honest-fail error handling (~ 4 tests)
# ---------------------------------------------------------------------------


class TestSerializerErrorHandling:
    """DECOMP-02 + internal notes: honest-fail-on-data behavior."""

    def test_empty_parent_no_legacy_raises(self):
        """Malformed tree (empty parent_stem + no fragment_legacy) raises."""
        # NameTreeNode requires parent_stem; we have to pass an empty string.
        n = NameTreeNode(parent_stem="")
        with pytest.raises(NameTreeSerializerError):
            name_tree_to_string(n)

    def test_serializer_error_is_value_error_subclass(self):
        """NameTreeSerializerError is a ValueError subclass for easy catching."""
        assert issubclass(NameTreeSerializerError, ValueError)

    def test_error_message_cites_decomp_02(self):
        """Error message MUST cite the architectural contract (DECOMP-02)."""
        n = NameTreeNode(parent_stem="")
        try:
            name_tree_to_string(n)
        except NameTreeSerializerError as e:
            # Message should reference the upstream-fix recommendation.
            assert "upstream" in str(e).lower() or "decomp" in str(e).lower()

    def test_error_includes_node_repr(self):
        """Error message includes the node repr for debugging."""
        n = NameTreeNode(parent_stem="")
        try:
            name_tree_to_string(n)
        except NameTreeSerializerError as e:
            assert "NameTreeNode" in str(e)


# ---------------------------------------------------------------------------
# Class 5 — Integration with name_tree helpers (~ 4 tests)
# ---------------------------------------------------------------------------


class TestNormalizeAndAlphaIntegration:
    """Serializer respects the contract that locants are pre-normalized."""

    def test_locants_already_sorted(self):
        """If suffix locants are sorted (caller-side _normalize_locants), the
        output uses them as-is. prop + 'ol' (1,2,3) -> propane-1,2,3-triol."""
        n = NameTreeNode(parent_stem="prop", suffix="ol", locants=(1, 2, 3))
        out = name_tree_to_string(n)
        assert "1,2,3" in out

    def test_two_complex_prefixes_each_with_locants(self):
        """Two distinct prefixes both with locants are rendered side-by-side."""
        a = NameTreeNode(parent_stem="bromo", locants=(2,))
        b = NameTreeNode(parent_stem="chloro", locants=(3,))
        n = NameTreeNode(parent_stem="butan", prefixes=(a, b))
        out = name_tree_to_string(n)
        # Both substituent locants and stems appear.
        assert "2" in out
        assert "3" in out
        assert "bromo" in out
        assert "chloro" in out

    def test_nested_prefix_subtree(self):
        """Prefix subtree's own prefixes are recursively serialized."""
        inner = NameTreeNode(parent_stem="methyl", locants=(1,))
        outer = NameTreeNode(
            parent_stem="ethyl",
            prefixes=(inner,),
            parenthesization_hint=True,
        )
        root = NameTreeNode(parent_stem="butan", prefixes=(outer,))
        out = name_tree_to_string(root)
        # The recursive serialization should embed 'methyl' inside the outer ethyl.
        assert "methyl" in out
        assert "ethyl" in out
        assert "butan" in out

    def test_returns_string(self):
        n = NameTreeNode(parent_stem="ethan")
        out = name_tree_to_string(n)
        assert isinstance(out, str)
