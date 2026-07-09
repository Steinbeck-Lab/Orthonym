"""P-35.1 di/bis catenation disambiguation + P-16.5.1.10 bis-parentheses
(Wave-2 P0c Task 8). BB BlueBookV2.md:17954 ('disulfanyl, -SSH, and
bis(sulfanyl), two -SH groups') and :7368 (parentheses enclose bis/tris terms).
"""
import pytest

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
        # Each member collides with a P-29.3.1 catenated-hydride prefix:
        # disulfanyl -SSH, diselanyl -SeSeH, ditellanyl -TeTeH, diphosphanyl
        # -PH-PH2, diarsanyl, distibanyl, diazanyl -NH-NH2, dioxidanyl -OOH.
        assert CATENATION_AMBIGUOUS_PREFIXES == frozenset({
            "sulfanyl", "selanyl", "tellanyl", "phosphanyl",
            "arsanyl", "stibanyl", "azanyl", "oxidanyl",
        })

    def test_unambiguous_simple_prefix_keeps_di(self):
        assert get_multiplier_prefix(2, "methyl") == "di"
        assert get_multiplier_prefix(2, "chloro") == "di"

    def test_count_one_unchanged(self):
        assert get_multiplier_prefix(1, "sulfanyl") == ""


class TestP165110Parentheses:
    def test_bis_term_is_parenthesized(self):
        # P-16.5.1.10: 'Parentheses are used to enclose terms modified by
        # the numerical prefixes bis, tris, tetrakis'.
        assert format_substituent_prefix("sulfanyl", [3, 3], 2) == \
            "3,3-bis(sulfanyl)"

    def test_simple_di_not_parenthesized(self):
        assert format_substituent_prefix("methyl", [2, 2], 2) == "2,2-dimethyl"

    def test_existing_complex_bis_still_parenthesized(self):
        assert format_substituent_prefix("1-methylethyl", [4, 7], 2) == \
            "4,7-bis(1-methylethyl)"
