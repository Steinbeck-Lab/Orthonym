"""Phase 165 Plan-01 Task-2 unit tests for ``name_tree_builder.fragments_to_tree``.

The deriver is a PURE transform: ``List[NameFragment] -> NameTreeNode``. It
mirrors the ``_assemble_fragments`` classification loop
(``_handler_shared.py:779-787``) and unsaturation unpacking (``:877-880``),
populating the structured ``NameTreeNode`` fields for SCORE-02 / Phase 166
while ALSO carrying ``fragment_legacy`` (the pre-assembled final string) so
``name_tree_to_string`` round-trips byte-identically (D-02 dual-carry).

Per the contributor guide / fix-methodology: NO ``@pytest.mark.xfail`` in this module.
"""
from __future__ import annotations

import copy

from orthonym.assembly.composer import NameFragment, _assemble_fragments
from orthonym.assembly.name_tree import NameTreeNode
from orthonym.assembly.name_tree_builder import fragments_to_tree
from orthonym.assembly.name_tree_to_string import name_tree_to_string


class TestFieldMapping:
    """Structured-field population (mirrors _assemble_fragments classification)."""

    def test_parent_only(self):
        frags = [NameFragment(text="butan", fragment_type="parent", locants=((), ()))]
        tree = fragments_to_tree(frags)
        assert isinstance(tree, NameTreeNode)
        assert tree.parent_stem == "butan"
        assert tree.prefixes == ()
        assert tree.suffix is None
        assert tree.stereo is None

    def test_parent_plus_suffix_locants_normalized(self):
        frags = [
            NameFragment(text="butan", fragment_type="parent", locants=((), ())),
            NameFragment(text="ol", fragment_type="suffix", locants=(3, 1, 1)),
        ]
        tree = fragments_to_tree(frags)
        assert tree.suffix == "ol"
        # _normalize_locants: sort + dedupe
        assert tree.locants == (1, 3)

    def test_multi_prefix_alphabetized(self):
        # Supplied in non-alphabetical order; _alphabetize_prefixes must reorder.
        frags = [
            NameFragment(text="methyl", fragment_type="prefix", locants=(2,), count=1),
            NameFragment(text="ethyl", fragment_type="prefix", locants=(3,), count=1),
            NameFragment(text="butan", fragment_type="parent", locants=((), ())),
        ]
        tree = fragments_to_tree(frags)
        assert [p.parent_stem for p in tree.prefixes] == ["ethyl", "methyl"]

    def test_parent_unsaturation_tuple(self):
        frags = [NameFragment(text="pent", fragment_type="parent", locants=((1,), (3,)))]
        tree = fragments_to_tree(frags)
        assert tree.unsaturation_locants == ((1,), (3,))

    def test_parent_unsaturation_double_only(self):
        frags = [NameFragment(text="but", fragment_type="parent", locants=((1,), ()))]
        tree = fragments_to_tree(frags)
        assert tree.unsaturation_locants == ((1,), ())

    def test_stereo_fragment(self):
        frags = [
            NameFragment(text="(2R)-", fragment_type="stereo"),
            NameFragment(text="butan", fragment_type="parent", locants=((), ())),
        ]
        tree = fragments_to_tree(frags)
        assert tree.stereo == "(2R)-"

    def test_prefix_count_gt_one_sets_multiplicative(self):
        frags = [
            NameFragment(text="chloro", fragment_type="prefix", locants=(1, 2), count=2),
            NameFragment(text="ethan", fragment_type="parent", locants=((), ())),
        ]
        tree = fragments_to_tree(frags)
        assert tree.prefixes[0].multiplicative_prefix == "di"

    def test_prefix_count_one_multiplicative_none(self):
        frags = [
            NameFragment(text="methyl", fragment_type="prefix", locants=(2,), count=1),
            NameFragment(text="butan", fragment_type="parent", locants=((), ())),
        ]
        tree = fragments_to_tree(frags)
        assert tree.prefixes[0].multiplicative_prefix is None

    def test_empty_fragment_list(self):
        tree = fragments_to_tree([])
        assert tree.parent_stem == ""
        assert tree.prefixes == ()
        assert tree.suffix is None

    def test_class_id_and_cite_threaded(self):
        frags = [NameFragment(text="butan", fragment_type="parent", locants=((), ()))]
        tree = fragments_to_tree(frags, class_id="general_acyclic", section_cite="P-31.1.4")
        assert tree.class_id == "general_acyclic"
        assert tree.iupac_section_cite == "P-31.1.4"


class TestPurity:
    """AP-160-15 / D-25: no mutation of inputs."""

    def test_purity_inputs_unchanged(self):
        frags = [
            NameFragment(text="methyl", fragment_type="prefix", locants=(2,), count=1),
            NameFragment(text="butan", fragment_type="parent", locants=((1,), ())),
            NameFragment(text="ol", fragment_type="suffix", locants=(1,), count=1),
            NameFragment(text="(2R)-", fragment_type="stereo"),
        ]
        before = copy.deepcopy(frags)
        fragments_to_tree(frags)
        assert len(frags) == len(before)
        for a, b in zip(frags, before):
            assert (a.text, a.locants, a.fragment_type, a.count) == (
                b.text, b.locants, b.fragment_type, b.count
            )


class TestParityWithAssembleFragments:
    """SC-1: name_tree_to_string(fragments_to_tree(f)) == _assemble_fragments(f, style)."""

    def _frag_lists(self):
        return [
            [NameFragment(text="butan", fragment_type="parent", locants=((), ())),
             NameFragment(text="ol", fragment_type="suffix", locants=(1,), count=1)],
            [NameFragment(text="ethyl", fragment_type="prefix", locants=(3,), count=1),
             NameFragment(text="methyl", fragment_type="prefix", locants=(2,), count=1),
             NameFragment(text="hexan", fragment_type="parent", locants=((), ())),
             NameFragment(text="ol", fragment_type="suffix", locants=(1,), count=1)],
            [NameFragment(text="(2Z)-", fragment_type="stereo"),
             NameFragment(text="but", fragment_type="parent", locants=((2,), ()))],
            [NameFragment(text="pent", fragment_type="parent", locants=((), ()))],
        ]

    def test_parity_byte_identical(self):
        for frags in self._frag_lists():
            tree = fragments_to_tree(frags)
            assert name_tree_to_string(tree, "pin") == _assemble_fragments(frags, "pin")

    def test_parity_style_independent(self):
        # _assemble_fragments is style-independent today; the str carrier returns
        # verbatim regardless of the style arg passed to name_tree_to_string.
        frags = self._frag_lists()[1]
        tree = fragments_to_tree(frags)
        assert name_tree_to_string(tree, "cas") == name_tree_to_string(tree, "pin")
