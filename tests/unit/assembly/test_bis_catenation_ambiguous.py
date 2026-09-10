""" di/bis catenation disambiguation + bis-parentheses
(Wave-2 P0c Task 8). BB the Blue Book ('disulfanyl, -SSH, and
bis(sulfanyl), two -SH groups') and:7368 (parentheses enclose bis/tris terms).

: the catenated-chain prefixes themselves (disulfanyl -SSH,
trisulfanyl -SSSH, and the Se/Te analogues) are ALSO catenation-ambiguous when
multiplied -- 'didisulfanyl' reads as 'di' + 'disulfanyl' = a 4-sulfur chain --
so they take bis/tris too. BB verbatim: '3,4-bis(disulfanyl)benzamide (PIN)'
(the Blue Book) and 'bis(disulfanyl)... (not didisulfanyl)' (:7196).
"""
import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import (
    CATENATION_AMBIGUOUS_PREFIXES,
    format_substituent_prefix,
    get_multiplier_prefix,
)


class TestP351Multiplier:
    @pytest.mark.parametrize("name", sorted(CATENATION_AMBIGUOUS_PREFIXES))
    def test_ambiguous_prefixes_take_bis(self, name):
        assert get_multiplier_prefix(2, name) == "bis"
        assert get_multiplier_prefix(3, name) == "tris"

    def test_set_contents_pinned(self):
        # Each member collides with a catenated-hydride prefix:
        # disulfanyl -SSH, diselanyl -SeSeH, ditellanyl -TeTeH, diphosphanyl
        # -PH-PH2, diarsanyl, distibanyl, diazanyl -NH-NH2, dioxidanyl -OOH.
        # added the multi-atom chain prefixes themselves: two -S-S-H
        # groups are bis(disulfanyl) (the Blue Book, the Blue Book), because 'didisulfanyl'
        # would read as a catenation ('di' + 'disulfanyl' = -SSSS-);
        # the tri- forms (-S-S-S-H) collide the same way.
        assert CATENATION_AMBIGUOUS_PREFIXES == frozenset({
            "sulfanyl", "selanyl", "tellanyl", "phosphanyl",
            "arsanyl", "stibanyl", "azanyl", "oxidanyl",
            "disulfanyl", "diselanyl", "ditellanyl",
            "trisulfanyl", "triselanyl", "tritellanyl",
        })

    def test_unambiguous_simple_prefix_keeps_di(self):
        assert get_multiplier_prefix(2, "methyl") == "di"
        assert get_multiplier_prefix(2, "chloro") == "di"

    def test_count_one_unchanged(self):
        assert get_multiplier_prefix(1, "sulfanyl") == ""


class TestP165110Parentheses:
    def test_bis_term_is_parenthesized(self):
        #: 'Parentheses are used to enclose terms modified by
        # the numerical prefixes bis, tris, tetrakis'.
        assert format_substituent_prefix("sulfanyl", [3, 3], 2) == \
            "3,3-bis(sulfanyl)"

    def test_simple_di_not_parenthesized(self):
        assert format_substituent_prefix("methyl", [2, 2], 2) == "2,2-dimethyl"

    def test_existing_complex_bis_still_parenthesized(self):
        assert format_substituent_prefix("1-methylethyl", [4, 7], 2) == \
            "4,7-bis(1-methylethyl)"


class TestDisulfanylChainMultiplier:
    """: the catenated-chain prefixes take bis/tris, not di/tri."""

    @pytest.mark.parametrize("name", [
        "disulfanyl", "diselanyl", "ditellanyl",
        "trisulfanyl", "triselanyl", "tritellanyl",
    ])
    def test_chain_prefix_takes_bis(self, name):
        # 'didisulfanyl' would read as 'di' + 'disulfanyl' = a chain, so
        # two -S-S-H groups must be bis(disulfanyl) (the Blue Book 'not didisulfanyl').
        assert get_multiplier_prefix(2, name) == "bis"
        assert get_multiplier_prefix(3, name) == "tris"

    def test_bis_disulfanyl_end_to_end(self):
        # the Blue Book verbatim PIN. Was '3,4-didisulfanylbenzamide' before Task 6G.
        assert name_compound("NC(=O)c1ccc(SS)c(SS)c1") == \
            "3,4-bis(disulfanyl)benzamide"

    def test_single_disulfanyl_unchanged(self):
        # count == 1 takes no multiplier and the bare prefix stays bare.
        assert get_multiplier_prefix(1, "disulfanyl") == ""
        assert name_compound("c1ccc(SS)cc1") == "disulfanylbenzene"

    def test_compound_disulfanyl_prefix_unchanged(self):
        # A SUBSTITUTED disulfanyl prefix already took bis (via
        # is_substituted_substituent); Task 6G must not disturb it.
        assert get_multiplier_prefix(2, "methyldisulfanyl") == "bis"
        assert get_multiplier_prefix(2, "sulfanyldisulfanyl") == "bis"
