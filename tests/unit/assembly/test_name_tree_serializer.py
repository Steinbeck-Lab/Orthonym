"""Phase 179 (WSA-03) — per-gap byte-identity unit tests for the completed
``_assemble_explicit_fields`` serializer path.

These are JVM-free, pure-serializer unit tests: each test constructs a
``NameTreeNode`` with the EXACT structured-field shape that production
``fragments_to_tree`` builds for a ``general_acyclic`` molecule (verified by
probing ``name_with_tree`` on the corresponding SMILES), with
``fragment_legacy`` left ``None`` so the explicit-field branch runs (NOT the
``str`` short-circuit). Each asserts ``name_tree_to_string(node) == "<exact
legacy name>"`` — the byte-identical contract of CONTEXT D-02.

The root node MUST carry ``class_id="general_acyclic"`` (in
``SERIALIZER_PRODUCTION_CLASSES``) so the serializer applies the BARE-hydride
grammar (-ane/-ene/-yne + suffix infix); a complete-name node (retained /
organometallic) carries a different class_id and is passed through verbatim.
Prefix sub-nodes carry ``class_id=""`` (production shape — their parent_stem is
the full substituent text).

The expected strings are the production ``name_compound(smiles)`` outputs
(copied verbatim) so the tests are self-documenting and prove parity with the
legacy ``_assemble_fragments`` assembler.

Gap map (CONTEXT D-02 / 179-RESEARCH "8 Composition Gaps"):
  #2 prefix already-has-locant guard      -> 2-methylbutane (NOT 2-2-2-methylbut)
  #3 P-16.5.1.3.1 mononuclear enclosing    -> bromo(chloro)(fluoro)methane
  #5 full suffix grammar (multiplier/elide) -> butan-1-ol / pentanal / butane-1,4-diol
  #6 unsaturation infix (en/yn + ring)      -> hex-3-yne / cyclodecane
  #7 prefix->parent hyphenation             -> 1-chloropentane
  #8 stereo prepend (no extra hyphen)       -> (2S)-butan-2-ol (NOT (2S)--but-2-ol)
"""
from __future__ import annotations

import pytest

from orthonym.assembly.name_tree import NameTreeNode
from orthonym.assembly.name_tree_to_string import name_tree_to_string


def _ga(**kwargs) -> NameTreeNode:
    """A production-shaped general_acyclic ROOT node (bare hydride stem; uses
    the full hydride grammar). class_id places it in SERIALIZER_PRODUCTION_CLASSES."""
    return NameTreeNode(class_id="general_acyclic", **kwargs)


def _prefix(text, locants=()):
    """A production-shaped substituent prefix node (full text in parent_stem,
    no fragment_legacy, class_id="") — mirrors fragments_to_tree:102-108."""
    return NameTreeNode(parent_stem=text, locants=tuple(locants))


pytestmark = pytest.mark.unit


class TestSuffixGrammarGap5:
    """Gap #5: full suffix grammar via format_suffix_with_locants (P-16.7.1
    elision, multiplier, infix). Bare-stem root + bare suffix + suffix locants."""

    def test_butan_1_ol(self):
        # CCCCO -> butan-1-ol
        assert name_tree_to_string(_ga(parent_stem="but", suffix="ol", locants=(1,))) == "butan-1-ol"

    def test_pentanal(self):
        # CCCCC=O -> pentanal (aldehyde suffix, no locant)
        assert name_tree_to_string(_ga(parent_stem="pent", suffix="al", locants=())) == "pentanal"

    def test_pentan_2_one(self):
        # CC(=O)CCC -> pentan-2-one
        assert name_tree_to_string(_ga(parent_stem="pent", suffix="one", locants=(2,))) == "pentan-2-one"

    def test_butane_1_4_diol(self):
        # OCCCCO -> butane-1,4-diol (di- multiplier from 2 suffix locants,
        # 'e' retained before consonant-initial multiplied suffix)
        assert name_tree_to_string(_ga(parent_stem="but", suffix="ol", locants=(1, 4))) == "butane-1,4-diol"


class TestUnsaturationGap6:
    """Gap #6: unsaturation infix via _build_hydrocarbon_name / _build_unsaturation_infix."""

    def test_hex_3_yne(self):
        # CCC#CCC -> hex-3-yne (no-suffix triple-bond branch)
        assert name_tree_to_string(_ga(parent_stem="hex", unsaturation_locants=((), (3,)))) == "hex-3-yne"

    def test_cyclodecane(self):
        # C1CCCCCCCCC1 -> cyclodecane (saturated ring -> stem+'ane')
        assert name_tree_to_string(_ga(parent_stem="cyclodec")) == "cyclodecane"

    def test_but_3_en_1_ol_anchor(self):
        # C=CCCO -> but-3-en-1-ol (suffix + unsaturation infix)
        n = _ga(parent_stem="but", suffix="ol", locants=(1,), unsaturation_locants=((3,), ()))
        assert name_tree_to_string(n) == "but-3-en-1-ol"


class TestPrefixLocantGap2And7:
    """Gap #2 (already-has-locant guard) + gap #7 (prefix->parent hyphenation)."""

    def test_2_methylbutane(self):
        # CC(C)CC -> 2-methylbutane. Prefix parent_stem already carries '2-methyl'
        # AND locants=(2,): the guard must NOT re-prepend (no 2-2-2-methylbut).
        assert name_tree_to_string(_ga(parent_stem="but", prefixes=(_prefix("2-methyl", (2,)),))) == "2-methylbutane"

    def test_2_methylpropan_2_ol(self):
        # CC(C)(C)O -> 2-methylpropan-2-ol (prefix + suffix together)
        n = _ga(parent_stem="prop", suffix="ol", locants=(2,), prefixes=(_prefix("2-methyl", (2,)),))
        assert name_tree_to_string(n) == "2-methylpropan-2-ol"

    def test_1_chloropentane(self):
        # ClCCCCC -> 1-chloropentane. Prefix 'chloro' with locants=(1,) and NO
        # baked-in locant: the locant IS prepended -> '1-chloro' then joined.
        assert name_tree_to_string(_ga(parent_stem="pent", prefixes=(_prefix("chloro", (1,)),))) == "1-chloropentane"


class TestMononuclearEnclosingGap3:
    """Gap #3: P-16.5.1.3.1 mononuclear enclosing marks + the multiplier carve-out."""

    def test_bromo_chloro_fluoro_methane(self):
        # C(Br)(Cl)F -> bromo(chloro)(fluoro)methane. Mononuclear 'meth' with
        # >=2 simple prefixes: first bare, rest each enclosed.
        n = _ga(parent_stem="meth", prefixes=(_prefix("bromo"), _prefix("chloro"), _prefix("fluoro")))
        assert name_tree_to_string(n) == "bromo(chloro)(fluoro)methane"

    def test_bromodichlorofluoromethane_protect(self):
        # C(Br)(Cl)(Cl)F -> bromodichlorofluoromethane. A multiplied prefix
        # ('dichloro') disables enclosing for ALL (the common-PIN carve-out).
        n = _ga(parent_stem="meth", prefixes=(_prefix("bromo"), _prefix("dichloro"), _prefix("fluoro")))
        assert name_tree_to_string(n) == "bromodichlorofluoromethane"


class TestStereoPrependGap8:
    """Gap #8: stereo is prepended with NO extra hyphen (the descriptor already
    carries its own trailing hyphen, e.g. '(2S)-')."""

    def test_2S_butan_2_ol(self):
        # C[C@H](O)CC -> (2S)-butan-2-ol (NOT (2S)--but-2-ol)
        n = _ga(parent_stem="but", suffix="ol", locants=(2,), stereo="(2S)-")
        assert name_tree_to_string(n) == "(2S)-butan-2-ol"
