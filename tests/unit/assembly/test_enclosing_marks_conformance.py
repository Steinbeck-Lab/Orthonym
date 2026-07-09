"""P-16.5.4.1.1/.1.2/.1.3/.1.5 nesting-order conformance (Wave-2 P0c Task 6).

Names are the Blue Book's own examples (BlueBookV2.md:7463-7530).
"""
import pytest

from orthonym.assembly.naming_utils import (
    apply_enclosing_marks,
    compute_nesting_depth,
)


class TestP1654_1_1_IndicatedHydrogenIgnored:
    # BB 7465: "the parentheses of added indicated hydrogen atoms are ignored"
    # Example: 1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one
    def test_added_ih_parens_do_not_count(self):
        assert compute_nesting_depth("3,4-dihydroquinolin-1(2H)-yl") == 0

    def test_enclosure_stays_parentheses(self):
        assert apply_enclosing_marks("3,4-dihydroquinolin-1(2H)-yl", -1) == \
            "(3,4-dihydroquinolin-1(2H)-yl)"


class TestP1654_1_2_FusionSpiroAssemblyBracketsIgnored:
    # BB 7469 examples: bicyclo[2.2.1] (von Baeyer), [1,1'-biphenyl]
    # (ring assembly), dibenzo[b,d]furan (fusion).
    def test_von_baeyer_brackets_ignored(self):
        assert compute_nesting_depth(
            "7,7-dimethyl-2-oxobicyclo[2.2.1]heptan-1-yl") == 0

    def test_fusion_brackets_ignored(self):
        assert compute_nesting_depth("dibenzo[b,d]furan-1-yl") == 0

    def test_ring_assembly_brackets_with_primes_ignored(self):
        # BB: 10,10'-[[1,1'-biphenyl]-4,4'-diylbis(oxy)]di(decanoic acid)
        # -> the [1,1'-biphenyl] enclosure is NOT nesting-relevant; only the
        # (oxy) parens count, so effective depth is 1 (next mark: [ ]).
        assert compute_nesting_depth("[1,1'-biphenyl]-4,4'-diylbis(oxy)") == 1

    def test_ring_assembly_enclosure_gets_square_brackets(self):
        assert apply_enclosing_marks("[1,1'-biphenyl]-4,4'-diylbis(oxy)", -1) == \
            "[[1,1'-biphenyl]-4,4'-diylbis(oxy)]"


class TestP1654_1_3_CountedParentheses:
    # BB 7478: compound locants, stereo descriptors etc. ARE counted.
    def test_compound_locant_counts(self):
        # BB: 2-[bicyclo[6.6.1]pentadeca-8(15)-en-1-yl]ethan-1-ol
        assert compute_nesting_depth("bicyclo[6.6.1]pentadeca-8(15)-en-1-yl") == 1

    def test_stereo_descriptor_counts(self):
        # BB: 10-{[(3S)-1-phosphabicyclo[2.2.2]octan-3-yl]methyl}-...
        assert compute_nesting_depth("(3S)-1-phosphabicyclo[2.2.2]octan-3-yl") == 1

    def test_stereo_bearing_substituent_gets_square_brackets(self):
        assert apply_enclosing_marks(
            "(3S)-1-phosphabicyclo[2.2.2]octan-3-yl", -1) == \
            "[(3S)-1-phosphabicyclo[2.2.2]octan-3-yl]"


class TestP1654_1_5_ConsecutiveMarksEscalate:
    # BB 7509 worked example (the tetrahydroisoquinoline carboxylic acid):
    # inserting (2S) in front of a name that would otherwise take ( ) again
    # escalates to [ ].
    def test_stage2_escalation(self):
        inner = "(2S)-2-{[(2S)-1-ethoxy-1-oxo-4-phenylbutan-2-yl]amino}propanoyl"
        assert apply_enclosing_marks(inner, -1) == f"[{inner}]"

    def test_plain_paren_start_escalates(self):
        # A leading nesting-relevant '(' never doubles: ((...)) is forbidden.
        assert apply_enclosing_marks("(2-methylpropyl)amino", -1) == \
            "[(2-methylpropyl)amino]"
