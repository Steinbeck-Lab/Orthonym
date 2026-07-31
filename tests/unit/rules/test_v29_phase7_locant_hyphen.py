"""v29 Phase 7 Task 4 -- the P-16.2.4.1(a) hyphen before a locant.

Defect class C4: `parent_to_prefix` glued a prefix word fragment straight onto a
stem that itself begins with a locant, producing OPSIN-unparseable fragments
such as `3-amino2,12-dimethyltetradecyl`.

Governing rule, quoted with its section heading:

  BB:6936, heading "P-16.2.4 Hyphens" --
    "P-16.2.4.1 Hyphens are used in substitutive names:
     (a) to separate locants from words or word fragments;
     Example: 2-chloro-2-methylpropane (PIN, P-61.3.1)"

  BB:2847, heading "P-14.3.1 Types of locants" --
    "Traditional types of locants are arabic numbers, for example, 1, 2, 3;
     primed locants, for example, 1', 1''', 2''; locants including a lower case
     Roman letter, for example, 3a, 3b; italicized Roman letters, for example,
     O, N, P; ..."
  -- so an italic element locant takes the hyphen exactly as an arabic one does.
  Confirmed verbatim by the PIN example under P-16.2.4.1(b), BB:6944:
    "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine (PIN, P-62.2.4.1.3)"

  The bounding counter-rule, BB:6968 "P-16.2.4.2": "No hyphen is placed after a
  numerical prefix cited in front of a compound substituent enclosed by
  parentheses, even if that substituent begins with locants" -- example
  "N,1-bis(4-chlorophenyl)methanimine (PIN)". A '(' is not a locant, so the
  predicate must decline it.

The producing sites were established by `sys.settrace` over the whole package,
not by grep: `substituent_naming.parent_to_prefix` returned the glued string
directly for both live witnesses (the multi-FG amine branch for one, the
locanted amine branch for the other).
"""
import pytest

from orthonym.assembly.substituent_naming import (
    _joined_prefix_parts,
    _prefix_stem_yl,
    _starts_with_locant,
    parent_to_prefix,
)

pytestmark = pytest.mark.unit


class TestStartsWithLocant:
    """P-14.3.1 -- which heads count as a cited locant set."""

    # (text, expected). Arabic and italic-element positives, then the
    # word-fragment / configurational-descriptor negatives that must NOT be
    # split (P-16.2.4.1(a) separates *locants*, not any leading letter).
    CASES = [
        ("2-methyl", True),
        ("2,12-dimethyltetradec", True),
        ("3a-methyl", True),
        ("N-methyl", True),
        ("O-methylhydroxyl", True),
        ("N1-(2-aminoethyl)-N2-methyleth", True),
        ("N2',N2'-dimethyleth", True),
        ("S-methyl", True),
        ("methyl", False),
        ("tetradec", False),
        ("tert-butyl", False),
        ("sec-butyl", False),
        ("beta-D-glucopyranosyl", False),
        ("D-gluco", False),
        ("R-methyl", False),
        ("(2E)-but-2-en", False),
        ("", False),
    ]

    def test_case_table_is_populated(self):
        # Guard against a vacuous parametrisation (v29 Phase 6 shipped one).
        assert len(self.CASES) == 17
        assert sum(1 for _, e in self.CASES if e) == 8
        assert sum(1 for _, e in self.CASES if not e) == 9

    @pytest.mark.parametrize("text,expected", CASES)
    def test_predicate(self, text, expected):
        assert _starts_with_locant(text) is expected


class TestPrefixStemYl:
    def test_arabic_locant_stem_is_separated(self):
        assert (_prefix_stem_yl("3-amino", "2,12-dimethyltetradec")
                == "3-amino-2,12-dimethyltetradecyl")

    def test_italic_element_locant_stem_is_separated(self):
        assert (_prefix_stem_yl("1,2-diamino", "N1-(2-aminoethyl)-N2-methyleth")
                == "1,2-diamino-N1-(2-aminoethyl)-N2-methylethyl")

    def test_italic_O_locant_stem_is_separated(self):
        # The third C4 witness fragment, at the primitive level. (Its molecule
        # no longer reaches this branch -- see TestWitnessThreeFailsClosed.)
        assert (_prefix_stem_yl("amino", "O-methylhydroxyl")
                == "amino-O-methylhydroxylyl")

    # --- negative controls: NO hyphen wanted -------------------------------
    def test_plain_stem_is_not_separated(self):
        assert _prefix_stem_yl("hydroxy", "tetradec") == "hydroxytetradecyl"

    def test_multiplier_before_word_fragment_is_not_separated(self):
        assert _prefix_stem_yl("2,5-di", "methylhex") == "2,5-dimethylhexyl"

    def test_p16_2_4_2_no_hyphen_before_an_enclosing_mark(self):
        # P-16.2.4.2: '(' is not a locant, so 'bis(' is never split.
        assert _prefix_stem_yl("bis", "(4-chlorophenyl)meth") == "bis(4-chlorophenyl)methyl"


class TestParentToPrefixWitnesses:
    """Whole-fragment pins on the two spy-proven producing branches."""

    def test_witness_one_locanted_amine_branch(self):
        # CCC(C)(CCCCCCCCC(C(C)C)N)C1=CC(=O)C(=CC1=O)C(C)(CC)CCCCCCCCC(C(C)C)N
        # was '3-amino2,12-dimethyltetradecyl'.
        assert (parent_to_prefix("2,12-dimethyltetradecan-3-amine", 14)
                == "3-amino-2,12-dimethyltetradecyl")

    def test_witness_two_multi_fg_amine_branch(self):
        # CNCCNCCNCC(=O)O was '1,2-diaminoN1-(2-aminoethyl)-N2-methylethyl'.
        assert (parent_to_prefix("N1-(2-aminoethyl)-N2-methylethane-1,2-diamine", 2)
                == "1,2-diamino-N1-(2-aminoethyl)-N2-methylethyl")

    def test_witness_three_fails_closed(self):
        # 'aminoO-methylhydroxyl' came from the unlocanted-amine branch on the
        # parent 'O-methylhydroxylamine'. P-29.2 (BB:15811) licenses '-yl' only
        # for a PARENT HYDRIDE, and a hydroxylamine is not one, so Task 3's
        # guard now declines the whole conversion -- the C4 hyphen is moot here
        # and the emitter abstains instead. Pinned so a later change to that
        # guard cannot silently resurrect the glued fragment.
        assert parent_to_prefix("O-methylhydroxylamine", 1) is None

    # --- negative controls: current output, verified before pinning --------
    @pytest.mark.parametrize("parent,length,expected", [
        ("tetradecan-1-ol", 14, "1-hydroxytetradecyl"),
        ("hexane-2,5-diamine", 6, "2,5-diaminohexyl"),
        ("butan-2-one", 4, "2-oxobutyl"),
        ("butanoic acid", 4, "3-carboxypropyl"),
        ("propan-2-ol", 3, "2-hydroxypropyl"),
    ])
    def test_undecorated_stems_keep_no_hyphen(self, parent, length, expected):
        assert parent_to_prefix(parent, length) == expected


class TestJoinedPrefixParts:
    """The shared predicate must not disturb the arabic answers."""

    def test_arabic_join_unchanged(self):
        assert _joined_prefix_parts(["amino", "2-carboxy"]) == ["amino", "-", "2-carboxy"]

    def test_closing_bracket_then_locant_unchanged(self):
        assert (_joined_prefix_parts(["2-(hydroxymethyl)", "5-oxo"])
                == ["2-(hydroxymethyl)", "-", "5-oxo"])

    def test_word_fragment_pair_is_not_split(self):
        assert _joined_prefix_parts(["chloro", "methyl"]) == ["chloro", "methyl"]

    def test_italic_element_locant_is_now_separated(self):
        # The additive half of the widening.
        assert _joined_prefix_parts(["amino", "N-methyl"]) == ["amino", "-", "N-methyl"]

    def test_first_part_never_gets_a_leading_hyphen(self):
        assert _joined_prefix_parts(["2-carboxy"]) == ["2-carboxy"]
