""" a phase Task 4 -- the (a) hyphen before a locant.

Defect class C4: `parent_to_prefix` glued a prefix word fragment straight onto a
stem that itself begins with a locant, producing OPSIN-unparseable fragments
such as `3-amino2,12-dimethyltetradecyl`.

Governing rule, quoted with its section heading:

  the Blue Book, heading " Hyphens" --
    " Hyphens are used in substitutive names:
     (a) to separate locants from words or word fragments;
     Example: 2-chloro-2-methylpropane (PIN, "

  the Blue Book, heading " Types of locants" --
    "Traditional types of locants are arabic numbers, for example, 1, 2, 3;
     primed locants, for example, 1', 1''', 2''; locants including a lower case
     Roman letter, for example, 3a, 3b; italicized Roman letters, for example,
     O, N, P;..."
  -- so an italic element locant takes the hyphen exactly as an arabic one does.
  Confirmed verbatim by the PIN example under (b), the Blue Book:
    "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine (PIN, "

  The bounding counter-rule, the Blue Book "": "No hyphen is placed after a
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
    ATTACH_LOCANT_UNKNOWN,
    _joined_prefix_parts,
    _prefix_stem_yl,
    _starts_with_locant,
    parent_to_prefix,
)

pytestmark = pytest.mark.unit


class TestStartsWithLocant:
    """ -- which heads count as a cited locant set."""

    # (text, expected). Arabic and italic-element positives, then the
    # word-fragment / configurational-descriptor negatives that must NOT be
    # split (a) separates *locants*, not any leading letter).
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
        ("β-D-glucopyranosyl", False),
        ("D-gluco", False),
        ("R-methyl", False),
        ("(2E)-but-2-en", False),
        ("", False),
    ]

    def test_case_table_is_populated(self):
        # Guard against a vacuous parametrisation (a phase shipped one).
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
        #: '(' is not a locant, so 'bis(' is never split.
        assert _prefix_stem_yl("bis", "(4-chlorophenyl)meth") == "bis(4-chlorophenyl)methyl"


class TestParentToPrefixWitnesses:
    """Whole-fragment pins on the two trace-proven producing branches."""

    # residue Task A: these two witnesses used to assert the STRING the
    # locanted-amine and multi-FG-amine branches produced. Those strings are
    # provably not general answers -- ``parent_to_prefix`` is handed only a name
    # and a carbon count, and that pair is NOT injective over fragments:
    # ``-CH2CH2CH2OH`` and ``-CH(OH)CH2CH3`` both cap to ``propan-1-ol`` with
    # count 3, yet OPSIN 2.9.0 makes the single old output ``1-hydroxypropyl``
    # EXACT for the second and a DIFFERENT MOLECULE for the first. Witness one
    # is the same failure: OPSIN round-trips ``(3-amino-2,12-dimethyltetradecyl)
    # benzene`` to a different molecule, while the structurally numbered
    # ``(12-amino-3,13-dimethyltetradecan-3-yl)benzene`` is EXACT
    #, the Blue Book: "The principal substituent chain has the lowest
    # locants for free valences of any kind").
    #
    # The C4 HYPHEN behaviour they were written to guard is preserved by pinning
    # it on ``_prefix_stem_yl`` -- the function that actually performs it -- so
    # this change updates the contract without dropping coverage.

    def test_witness_one_hyphen_is_still_inserted(self):
        assert (_prefix_stem_yl("3-amino", "2,12-dimethyltetradec")
                == "3-amino-2,12-dimethyltetradecyl")

    def test_witness_two_hyphen_is_still_inserted(self):
        assert (_prefix_stem_yl("1,2-diamino", "N1-(2-aminoethyl)-N2-methyleth")
                == "1,2-diamino-N1-(2-aminoethyl)-N2-methylethyl")

    @pytest.mark.parametrize("parent,length", [
        ("2,12-dimethyltetradecan-3-amine", 14),                    # R8.2
        ("N1-(2-aminoethyl)-N2-methylethane-1,2-diamine", 2),       # R4
    ])
    def test_witness_branches_now_fail_closed(self, parent, length):
        """Neither locant set is derivable from (name, count): decline."""
        assert parent_to_prefix(
            parent, length, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    def test_witness_three_fails_closed(self):
        # 'aminoO-methylhydroxyl' came from the unlocanted-amine branch on the
        # parent 'O-methylhydroxylamine'. (the Blue Book) licenses '-yl' only
        # for a PARENT HYDRIDE, and a hydroxylamine is not one, so Task 3's
        # guard now declines the whole conversion -- the C4 hyphen is moot here
        # and the emitter abstains instead. Pinned so a later change to that
        # guard cannot silently resurrect the glued fragment.
        assert parent_to_prefix("O-methylhydroxylamine", 1, attach_locant=ATTACH_LOCANT_UNKNOWN) is None

    # --- negative controls: the undecorated stems take no hyphen -----------
    # Re-pointed at _prefix_stem_yl for the same reason as the witnesses above:
    # the locants in these parent names belong to the CAPPED molecule, so
    # parent_to_prefix can no longer emit them. The hyphen rule under test is
    # unchanged and is still exercised on every stem.
    @pytest.mark.parametrize("prefix,stem,expected", [
        ("1-hydroxy", "tetradec", "1-hydroxytetradecyl"),
        ("2,5-diamino", "hex", "2,5-diaminohexyl"),
        ("2-oxo", "but", "2-oxobutyl"),
        ("3-carboxy", "prop", "3-carboxypropyl"),
        ("2-hydroxy", "prop", "2-hydroxypropyl"),
    ])
    def test_undecorated_stems_keep_no_hyphen(self, prefix, stem, expected):
        assert _prefix_stem_yl(prefix, stem) == expected

    @pytest.mark.parametrize("parent,length", [
        ("tetradecan-1-ol", 14),
        ("hexane-2,5-diamine", 6),
        ("butan-2-one", 4),
        ("butanoic acid", 4),
        ("propan-2-ol", 3),
    ])
    def test_those_parents_no_longer_borrow_a_locant(self, parent, length):
        assert parent_to_prefix(
            parent, length, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


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
