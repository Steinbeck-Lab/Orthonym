"""P-14.3.5 locant ordering + P-45.5 name comparison (Wave-2 P0c Tasks 1-2-4).

BB P-14.3.5 (BlueBookV2.md:3191-3195): primes immediately after unprimed;
number+letter after the bare number; superscripts after letters; italic
Roman letters < Greek letters < numerals.
"""
import pytest

from orthonym.assembly.name_comparison import (
    locant_sort_key,
    compare_locant_strings,
    compare_locant_str_sets,
)


class TestLocantSortKeyP1435:
    """Every worked ordering from BB P-14.3.5 (line 3195)."""

    @pytest.mark.parametrize("lower,higher", [
        ("2", "2'"),        # primes after unprimed
        ("3", "3a"),        # letter-suffixed after bare number
        ("8a", "8b"),
        ("4'", "4a"),       # prime beats letter suffix
        ("4a", "4'a"),      # 4a before 4'a (BB: 4'a, not 4a')
        ("1^2", "1^3"),     # superscripts compare numerically
        ("1^4", "2'"),      # base number decides first (BB: 1^4 < 2')
        ("3a", "3a^1"),     # superscript after letters
        ("N", "alpha"),     # italic Roman < Greek
        ("alpha", "1"),     # Greek < numerals
        ("N", "1"),         # italic Roman < numerals
        ("N", "N'"),        # primed italic after unprimed
    ])
    def test_pairwise_order(self, lower, higher):
        assert locant_sort_key(lower) < locant_sort_key(higher)

    def test_lambda_token_base_number_drives_p1435(self):
        # For P-14.3.5 set comparison the λ mark rides along; base locant decides.
        assert locant_sort_key("1λ5") < locant_sort_key("2λ5")
        assert locant_sort_key("1lambda5") < locant_sort_key("2lambda5")

    def test_equal_tokens(self):
        assert locant_sort_key("4a") == locant_sort_key("4a")


class TestCompareLocantStrSets:
    def test_first_point_of_difference(self):
        # BB P-14.3.5 worked example: 1,1',2',1'',3'',1''' < 1,1',3',1'',2'',1'''
        a = ["1", "1'", "2'", "1''", "3''", "1'''"]
        b = ["1", "1'", "3'", "1''", "2''", "1'''"]
        assert compare_locant_str_sets(a, b) == -1
        assert compare_locant_str_sets(b, a) == 1

    def test_italic_and_greek_before_numerals(self):
        # BB: N,α,1,2 is lower than 1,2,4,6
        assert compare_locant_str_sets(["N", "alpha", "1", "2"],
                                       ["1", "2", "4", "6"]) == -1

    def test_shorter_set_wins_on_tied_prefix(self):
        assert compare_locant_str_sets(["2", "3"], ["2", "3", "5"]) == -1

    def test_identical_sets(self):
        assert compare_locant_str_sets(["2", "3"], ["3", "2"]) == 0

    def test_compare_locant_strings_scalar(self):
        assert compare_locant_strings("4'", "4a") == -1
        assert compare_locant_strings("4a", "4a") == 0


from orthonym.assembly.name_comparison import compare_names


class TestCompareNamesP455:
    """BB P-45.5 (BlueBookV2.md:22234ff) worked examples, string-level."""

    def test_bb_example_1_bromo_before_dibromo(self):
        # BB P-45.5 example (2): 'bromo' earlier alphabetically than 'dibromo'
        a = "2-bromo-4-chloro-N-(2,4-dibromophenyl)aniline"
        b = "2,4-dibromo-N-(2-bromo-4-chlorophenyl)aniline"
        assert compare_names(a, b) == -1
        assert compare_names(b, a) == 1

    def test_bb_example_4_difluoro_before_dinitro(self):
        # BB P-45.5 example (4) — the row's corroborating acyclic parent
        a = "4-(1,2-difluoropropyl)-5,6-dinitroheptanoic acid"
        b = "4-(1,2-dinitropropyl)-5,6-difluoroheptanoic acid"
        assert compare_names(a, b) == -1

    def test_numeric_locants_in_order_of_appearance(self):
        # BB line 6407: '1-chloroethoxy' precedes '2-chloroethoxy' when
        # letters are identical — numerals compared in APPEARANCE order.
        a = "4-{2-[2-(4-carboxyphenyl)-1-chloroethoxy]-1-chloroethyl}benzoic acid"
        b = "4-{2-[2-(4-carboxyphenyl)-2-chloroethoxy]-2-chloroethyl}benzoic acid"
        assert compare_names(a, b) == -1

    def test_letters_before_italic_letters(self):
        # Roman-letter tier decides before the fusion italic letter tier:
        # identical Roman letters, fusion letters 'f' < 'g' (BB P-14.5.3
        # naphtho[1,2-f]quinolin-2-yl preferred to naphtho[1,2-g]quinolin-1-yl).
        a = "naphtho[1,2-f]quinolin-2-yl"
        b = "naphtho[1,2-g]quinolin-1-yl"
        assert compare_names(a, b) == -1

    def test_equal_names(self):
        assert compare_names("hexan-1-ol", "hexan-1-ol") == 0


from orthonym.assembly.name_comparison import (
    parse_lambda_locant,
    compare_lambda_locant_sets,
)


class TestLambdaLocantsP4532:
    def test_parse(self):
        assert parse_lambda_locant("1λ5") == ("1", 5)
        assert parse_lambda_locant("1lambda5") == ("1", 5)
        assert parse_lambda_locant("2'λ4") == ("2'", 4)
        assert parse_lambda_locant("4a") is None

    def test_bb_p4532_example(self):
        # BB line 22204: '1λ5' ... is lower than '2λ5' — the 1λ5 candidate wins.
        assert compare_lambda_locant_sets(["1λ5"], ["2λ5"]) == -1
        assert compare_lambda_locant_sets(["2λ5"], ["1λ5"]) == 1

    def test_ascii_spelling_equivalent(self):
        assert compare_lambda_locant_sets(["1lambda5"], ["2lambda5"]) == -1

    def test_non_lambda_tokens_ignored(self):
        # Only λ-bearing locants participate in the P-45.3.2 tier.
        assert compare_lambda_locant_sets(["9", "1λ5"], ["2", "2λ5"]) == -1

    def test_no_lambda_on_either_side_ties(self):
        assert compare_lambda_locant_sets(["3", "4"], ["1", "2"]) == 0
