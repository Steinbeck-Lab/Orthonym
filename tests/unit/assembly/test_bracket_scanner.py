"""The enclosing-mark scanner of ``orthonym.assembly.naming_utils``.

One counter reads where the marks of a name open and close. Every table below is
the behaviour of one reading that the code base needs: the mark set a call counts
and what an unmatched close does are arguments, so the tables pin each
combination. Strings cover balanced, nested, mixed, unbalanced and empty names,
italic locants (``N``, ``S``, ``1H``), primes and quote characters.
"""

import pytest

from orthonym.assembly.naming_utils import (
    bracket_depths,
    enclosing_closer,
    enclosing_opener,
    is_fully_enclosed,
    matching_close,
    top_level_text,
    top_level_tokens,
)

pytestmark = pytest.mark.unit

ALL = "([{"


# ------------------------------------------------------------------ bracket_depths

class TestBracketDepths:
    def test_walk_reports_depth_before_and_after(self):
        """ ENCLOSING MARKS: an opener stands outside the group it opens, a closer ends the group it closes."""
        assert list(bracket_depths("a(b)")) == [
            (0, "a", 0, 0), (1, "(", 0, 1), (2, "b", 1, 1), (3, ")", 1, 0)]

    def test_nested_mixed_marks_share_one_counter(self):
        """ Multiple types of enclosing marks: {} nest in one sequence, so one depth counts all three types."""
        depths = [after for _i, _c, _b, after in bracket_depths("{[(a)]}")]
        assert depths == [1, 2, 3, 3, 2, 1, 0]

    @pytest.mark.parametrize("marks, text, expected_after", [
        ("(", "[a](b)", [0, 0, 0, 1, 1, 0]),
        ("[", "[a](b)", [1, 1, 0, 0, 0, 0]),
        ("([", "[a](b)", [1, 1, 0, 1, 1, 0]),
        ("{", "[a](b)", [0, 0, 0, 0, 0, 0]),
    ])
    def test_a_mark_that_is_not_counted_is_an_ordinary_character(self, marks, text, expected_after):
        """ ENCLOSING MARKS: parentheses, brackets and braces are separate types; a call counts the types it names."""
        assert [after for _i, _c, _b, after in bracket_depths(text, marks)] == expected_after

    def test_unmatched_close_goes_negative_without_clamp(self):
        """ ENCLOSING MARKS: an unbalanced name has a close with no opener; the depth is allowed below zero."""
        assert [(b, a) for _i, _c, b, a in bracket_depths(")a)")] == [(0, -1), (-1, -1), (-1, -2)]

    def test_unmatched_close_stays_at_zero_with_clamp(self):
        """ ENCLOSING MARKS: an unbalanced name read with the depth held at zero."""
        assert [(b, a) for _i, _c, b, a in bracket_depths(")a)", clamp=True)] == [(0, 0), (0, 0), (0, 0)]

    def test_start_counts_from_zero_at_the_start_index(self):
        """ ENCLOSING MARKS: a walk from a later index does not see the marks before it."""
        assert [(i, a) for i, _c, _b, a in bracket_depths("(ab)c", start=2)] == [(2, 0), (3, -1), (4, -1)]

    @pytest.mark.parametrize("text", ["", "ethanol", "2-(4-chlorophenyl)-1H-indole"])
    def test_every_character_is_yielded_once_in_order(self, text):
        """ ENCLOSING MARKS: the walk covers the whole name, italic locants and hyphens included."""
        assert [(i, c) for i, c, _b, _a in bracket_depths(text)] == list(enumerate(text))

    def test_start_beyond_the_end_yields_nothing(self):
        """ ENCLOSING MARKS: no characters, no depths."""
        assert list(bracket_depths("(a)", start=7)) == []


# ------------------------------------------------------------------ is_fully_enclosed

FULLY_ENCLOSED_ALL = [
    # one pair around everything
    ("(a)", True), ("[a]", True), ("{a}", True), ("()", True), ("[]", True),
    ("([a])", True), ("[(a)]", True), ("{[(a)]}", True), ("(a(b)c)", True),
    # the / compound-prefix shapes of real names
    ("(2-methylpropyl)", True), ("[bis(sulfanyl)]", True),
    ("(2R,3S)-3-hydroxybutan-2-yl", False),                  # italic-locant stereo prefix, then text
    ("[(2S)-butan-2-yl]", True), ("(1H)", True),
    ("(4′-cyano[1,1′-biphenyl]-4-yl)", True),         # primes inside
    ("(a'b)", True), ("(a\"b\")", True),
    # text outside the leading mark
    ("(oxan-2-yl)oxy", False), ("(benzylsulfanyl)methyl", False), ("(a)b", False), ("a(b)", False),
    ("(a)(b)", False), ("[a][b]", False), ("(a)-(b)", False),
    # the first character is not a mark
    ("", False), ("ethanol", False), ("'(a)'", False), ("\"(a)\"", False), (" (a)", False), ("N-(a)", False),
    # unbalanced
    ("(", False), (")", False), ("(a", False), ("a)", False), ("(a))", False), ("((a)", False), (")(", False),
    # the final close must be the opener's own closer
    ("(a]", False), ("[a)", False), ("{a)", False), ("(a}", False),
    # inner marks are counted, not type-checked
    ("([)(])", True), ("(a(b]c)", True),
    ("(a[b)c]d)", False),                                      # a close of another type brings the depth to 0 before the end
    ("(a(b)]c)", False),
]


class TestIsFullyEnclosed:
    @pytest.mark.parametrize("text, expected", FULLY_ENCLOSED_ALL)
    def test_all_three_mark_types(self, text, expected):
        """ ENCLOSING MARKS: a term is enclosed when one mark opens at its start and its own closer ends it."""
        assert is_fully_enclosed(text, ALL) is expected

    def test_default_marks_are_all_three_types(self):
        """ ENCLOSING MARKS: parentheses, brackets and braces are all enclosing marks."""
        assert all(is_fully_enclosed(t) is e for t, e in FULLY_ENCLOSED_ALL)

    @pytest.mark.parametrize("text, expected", [
        ("(a)", True), ("([a])", True), ("({a})", True), ("(a[b]c)", True),
        ("[a]", False), ("{a}", False), ("[(a)]", False), ("[bis(sulfanyl)]", False),
        ("(a)(b)", False), ("(a)b", False), ("", False), ("(", False), ("(a", False), ("(a))", False),
        ("(a]", False),                                         # the ] is an ordinary character, so the (never closes
        ("([)(])", False),                                      # the first (closes at the third character
        ("(a[b)c]d)", False),
        ("(a(b)c)", True), ("((a)b)", True), ("((a)(b))", True),
        ("(2-methylpropyl)", True), ("(naphthalen-2-yl)methyl", False),
    ])
    def test_parentheses_only(self, text, expected):
        """ Parentheses: read with  alone, a bracket or brace inside is an ordinary character."""
        assert is_fully_enclosed(text, "(") is expected

    @pytest.mark.parametrize("marks, text, expected", [
        ("[", "[a]", True), ("[", "(a)", False), ("[", "[(a)]", True), ("[", "[a](b)", False),
        ("{", "{a}", True), ("{", "[a]", False),
        ("([", "(a)", True), ("([", "[a]", True), ("([", "{a}", False), ("([", "([a])", True), ("([", "({a})", True),
        ("([", "(a]", False),
    ])
    def test_other_mark_sets(self, marks, text, expected):
        """ Brackets and Braces: each mark set counts only its own types."""
        assert is_fully_enclosed(text, marks) is expected

    def test_result_is_a_bool(self):
        """ ENCLOSING MARKS: callers test the result with ``is`` and ``not``."""
        assert is_fully_enclosed("(a)") is True
        assert is_fully_enclosed("a") is False


# ------------------------------------------------------------------ top_level_tokens

class TestTopLevelTokens:
    @pytest.mark.parametrize("text, expected", [
        ("", [""]),
        ("a", ["a"]),
        ("a-b-c", ["a", "b", "c"]),
        ("-a", ["", "a"]),
        ("a-", ["a", ""]),
        ("a--b", ["a", "", "b"]),
        ("2-(4-chlorophenyl)-1H-indole", ["2", "(4-chlorophenyl)", "1H", "indole"]),
        ("N-[(2S)-butan-2-yl]-N-methylacetamide", ["N", "[(2S)-butan-2-yl]", "N", "methylacetamide"]),
        ("{a-b}-c", ["{a-b}", "c"]),
        ("(a-[b-{c-d}])", ["(a-[b-{c-d}])"]),
        ("4′-cyano[1,1′-biphenyl]-4-yl", ["4′", "cyano[1,1′-biphenyl]", "4", "yl"]),
        ("'a-b'-c", ["'a", "b'", "c"]),                         # a quote is not a mark
        ("(a-b", ["(a-b"]),                                      # never closed: the rest stays inside
    ])
    def test_split_at_hyphens_outside_every_mark(self, text, expected):
        """ ENCLOSING MARKS: a hyphen inside a mark belongs to the enclosed term, one outside separates terms."""
        assert top_level_tokens(text, "-", ALL) == expected

    def test_unbalanced_close_hides_the_hyphens_after_it_without_clamp(self):
        """ ENCLOSING MARKS: an unmatched close drives the depth below zero, so a later hyphen is not top level."""
        assert top_level_tokens("a)-b", "-", ALL) == ["a)-b"]
        assert top_level_tokens("a))-b(-c", "-", ALL) == ["a))-b(-c"]

    def test_unbalanced_close_is_ignored_with_clamp(self):
        """ ENCLOSING MARKS: with the depth held at zero an unmatched close does not hide the hyphens after it."""
        assert top_level_tokens("a)-b", "-", ALL, clamp=True) == ["a)", "b"]

    @pytest.mark.parametrize("marks, text, expected", [
        ("(", "[a-b]-c", ["[a", "b]", "c"]),
        ("(", "(a-b)-c", ["(a-b)", "c"]),
        ("([", "{a-b}-c", ["{a", "b}", "c"]),
        ("([", "[a-b]-c", ["[a-b]", "c"]),
        ("[", "(a-b)-c", ["(a", "b)", "c"]),
    ])
    def test_a_mark_that_is_not_counted_does_not_hide_a_hyphen(self, marks, text, expected):
        """ ENCLOSING MARKS: the mark set of the call decides which marks enclose."""
        assert top_level_tokens(text, "-", marks) == expected

    def test_other_separators(self):
        """ ENCLOSING MARKS: the separator is any single character that is not a counted mark."""
        assert top_level_tokens("a,b(c,d),e", ",", ALL) == ["a", "b(c,d)", "e"]
        assert top_level_tokens("a[b", "[", "(") == ["a", "b"]    # [ is not counted by a (call, so it can separate

    def test_tokens_joined_with_the_separator_give_the_name_back(self):
        """ ENCLOSING MARKS: splitting loses no character."""
        for text in ("", "a-b", "(a-b)-c-[d-e]", "-", "--", "a)-b", "N-N-(a)"):
            assert "-".join(top_level_tokens(text, "-", ALL)) == text


# ------------------------------------------------------------------ top_level_text

class TestTopLevelText:
    @pytest.mark.parametrize("text, expected", [
        ("", ""),
        ("ethanol", "ethanol"),
        ("a(b)c", "ac"),
        ("(a)", ""),
        ("2-(4-chlorophenyl)ethyl[ph]x", "2-ethylx"),
        ("{[(a)b]c}d", "d"),
        ("(1H)-indole", "-indole"),
        ("1H-pyrrol-3-yl(a)", "1H-pyrrol-3-yl"),
        ("'a'(b)\"c\"", "'a'\"c\""),
        ("(a", ""),                                             # never closed: the rest is inside
    ])
    def test_characters_outside_every_mark(self, text, expected):
        """ ENCLOSING MARKS: what stands outside every mark of a name belongs to its parent."""
        assert top_level_text(text, ALL) == expected

    def test_unbalanced_without_clamp(self):
        """ ENCLOSING MARKS: an unmatched close lowers the depth, so a later term reads as enclosed."""
        assert top_level_text("a)b(c)d", ALL) == "ac"

    def test_unbalanced_with_clamp(self):
        """ ENCLOSING MARKS: with the depth held at zero an unmatched close leaves the text after it top level."""
        assert top_level_text("a)b(c)d", ALL, clamp=True) == "abd"

    @pytest.mark.parametrize("marks, text, expected", [
        ("(", "a[b]c(d)e", "a[b]ce"),
        ("[", "a[b]c(d)e", "ac(d)e"),
        ("([", "a[b]c(d)e", "ace"),
    ])
    def test_mark_set(self, marks, text, expected):
        """ ENCLOSING MARKS: only the counted types enclose; the others are ordinary characters and are kept."""
        assert top_level_text(text, marks) == expected


# ------------------------------------------------------------------ matching_close

class TestMatchingClose:
    @pytest.mark.parametrize("text, start, expected", [
        ("(a)", 0, 2),
        ("((a)b)", 0, 5),
        ("((a)b)", 1, 3),
        ("x(a)", 1, 3),
        ("[(a)]", 0, 4),
        ("{[(a)]}", 1, 5),
        ("(a]", 0, 2),                                          # the type of the close is not compared
        ("(a", 0, None),
        ("(", 0, None),
        ("", 0, None),
        ("a)", 0, None),                                        # the depth goes to -1 and never returns to 0 at a close
        (")(", 0, None),
        ("(a)b", 3, None),
        ("(a)", 7, None),
        ("(2-methylpropyl)oxy", 0, 15),
    ])
    def test_index_of_the_close_that_returns_the_depth_to_zero(self, text, start, expected):
        """ ENCLOSING MARKS: the close that matches an opener is found by counting the marks of every type."""
        assert matching_close(text, start, ALL) == expected

    @pytest.mark.parametrize("marks, text, start, expected", [
        ("(", "[(a)]", 0, 3),
        ("(", "(a]", 0, None),
        ("(", "$(a(b)c)d", 1, 7),
        ("[", "[a(b)]", 0, 5),
        ("([", "([a)]", 0, 4),
    ])
    def test_mark_set(self, marks, text, start, expected):
        """ ENCLOSING MARKS: the mark set of the call decides which marks are counted."""
        assert matching_close(text, start, marks) == expected


# ------------------------------------------------------------------ enclosing_opener / enclosing_closer

class TestEnclosingGroup:
    @pytest.mark.parametrize("text, pos, expected", [
        ("(a b)", 1, 4),
        ("(a (b) c)", 1, 8),
        ("(a (b) c)", 4, 5),
        ("(a (b) c)", 6, 8),
        ("ab", 0, None),
        ("a)", 0, 1),
        ("a)", 2, None),
        ("(a)", 0, None),                                       # position 0 is the opener: it stands outside its own group
        ("(a)", 1, 2),
        ("", 0, None),
    ])
    def test_close_of_the_group_that_holds_a_position(self, text, pos, expected):
        """ ENCLOSING MARKS: the close of the group around a position is the first close with no opener after it."""
        assert enclosing_closer(text, pos, ALL) == expected

    @pytest.mark.parametrize("text, pos, expected", [
        ("(a b)", 3, 0),
        ("(a (b) c)", 8, 0),
        ("(a (b) c)", 5, 3),
        ("(a (b) c)", 4, 3),
        ("(a (b) c)", 7, 0),
        ("ab", 2, None),
        ("(a)", 3, None),
        ("(a)", 0, None),
        ("", 0, None),
        ("(a)b", 99, None),                                      # a position past the end is the end
        ("(ab", 99, 0),
    ])
    def test_opener_of_the_group_that_holds_a_position(self, text, pos, expected):
        """ ENCLOSING MARKS: the opener of the group around a position is the nearest opener left unmatched before it."""
        assert enclosing_opener(text, pos, ALL) == expected

    def test_parentheses_only_ignore_brackets(self):
        """ Parentheses: a  call sees through a bracket pair around the position."""
        assert enclosing_opener("(x[a]b)", 4, "(") == 0
        assert enclosing_opener("(x[a]b)", 4, ALL) == 2
        assert enclosing_closer("(x[a]b)", 4, "(") == 6
        assert enclosing_closer("(x[a]b)", 4, ALL) == 4

    def test_the_two_ends_of_a_group_agree(self):
        """ ENCLOSING MARKS: the open and the close found from a position inside bracket the same group."""
        text = "a-(b-[c-{d}]-e)-f"
        for pos in range(len(text)):
            lo, hi = enclosing_opener(text, pos, ALL), enclosing_closer(text, pos, ALL)
            if lo is not None and hi is not None:
                assert matching_close(text, lo, ALL) == hi


# ------------------------------------------------------------------ the readings a site needs

class TestReadings:
    """Each reading below is built from the walk alone, with its own marks and clamp."""

    @staticmethod
    def max_depth(text):
        return max((a for _i, _c, b, a in bracket_depths(text, ALL, clamp=True) if a > b), default=0)

    @pytest.mark.parametrize("text, expected", [
        ("", 0), ("ethanol", 0), ("(a)", 1), ("(a)(b)", 1), ("(a[b]c)", 2), ("{[(a)]}", 3), ("(((a)))", 3),
        ("2-fluoro-4-({6-[methyl(prop-2-en-1-yl)amino]hexyl}oxy)phenyl", 4),
        (")(", 1), ("(a))((", 2), ("((a)", 2),
    ])
    def test_deepest_nesting(self, text, expected):
        """ Multiple types of enclosing marks: the nesting depth of a name is the highest level any mark reaches."""
        assert self.max_depth(text) == expected

    @staticmethod
    def last_top_level_close(text):
        return max((i for i, c, _b, a in bracket_depths(text, "(") if c == ")" and a == 0), default=-1)

    @pytest.mark.parametrize("text, expected", [
        ("", -1), ("ethanol", -1), ("(a)b", 2), ("3-(2-methoxyethyl)hexan-1-ol", 17),
        ("4-oxo-4-(prop-2-enoyloxy)but-2-enoic acid", 24), ("(a(b)c)d(e)f", 10), ("[a](b)", 5),
        (")(()", 3),                                             # without clamp the depth climbs back from -1 to 0 at the last)
        ("a))(b)", -1),
    ])
    def test_last_top_level_close_of_parentheses(self, text, expected):
        """ Parentheses: the parent of a name starts after its last parenthesis that returns the depth to zero."""
        assert self.last_top_level_close(text) == expected

    @staticmethod
    def ends_at_depth_zero(text):
        walk = list(bracket_depths(text, "("))
        return bool(walk) and walk[-1][3] == 0

    @pytest.mark.parametrize("text, expected", [
        ("(a)", True), ("(a)(b)", True), ("(a", False), ("a)", False), ("((a)b)", True), ("(a))", False),
        ("(a))(", True),                                         # the final depth is 0 although a close comes before its opener
    ])
    def test_final_depth_zero_differs_from_fully_enclosed(self, text, expected):
        """ Parentheses: a name that ends at depth 0 need not be one enclosed term ('(a)(b)' ends at 0, is not enclosed)."""
        assert self.ends_at_depth_zero(text) is expected
        assert is_fully_enclosed("(a)(b)", "(") is False

    @staticmethod
    def depth_at(text, off):
        walk = list(bracket_depths(text[:off], "(["))
        return walk[-1][3] if walk else 0

    @pytest.mark.parametrize("text, off, expected", [
        ("", 0, 0), ("(a)b", 0, 0), ("(a)b", 1, 1), ("(a)b", 3, 0), ("[(a)]b", 2, 2), ("{a}b", 2, 0),
        ("(1,1,2,2,2-pentafluoro(13C1)ethyl)", 24, 2), ("a)b", 2, -1),
    ])
    def test_depth_at_a_position_counts_parentheses_and_brackets(self, text, off, expected):
        """ ENCLOSING MARKS: the depth at an offset is the number of (and [ opened before it and not yet closed."""
        assert self.depth_at(text, off) == expected

    @staticmethod
    def strip_recursive(text):
        out, i = [], 0
        while i < len(text):
            if text.startswith("$(", i):
                end = matching_close(text, i + 1, "(")
                i = (len(text) if end is None else end) + 1
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    @pytest.mark.parametrize("text, expected", [
        ("[#6]", "[#6]"), ("[#6;$(C=O)]", "[#6;]"), ("[#6;!$(C(=O)O)]", "[#6;!]"), ("a$(b(c)d)e$(f", "ae"),
        ("$(", ""), ("[#6;$([OH])]", "[#6;]"),
    ])
    def test_group_removed_by_its_matching_parenthesis(self, text, expected):
        """ Parentheses: a parenthesised group ends at the parenthesis that matches its opener."""
        assert self.strip_recursive(text) == expected


# ------------------------------------------------------------------ arguments

class TestArguments:
    @pytest.mark.parametrize("marks", ["", "x", "<", ")", "(x", "()"])
    def test_marks_must_be_opening_marks(self, marks):
        """ ENCLOSING MARKS: the three enclosing mark types are given by their opening marks."""
        for call in (lambda: bracket_depths("(a)", marks), lambda: is_fully_enclosed("(a)", marks),
                     lambda: top_level_tokens("a-b", "-", marks), lambda: top_level_text("(a)", marks),
                     lambda: matching_close("(a)", 0, marks), lambda: enclosing_closer("(a)", 0, marks),
                     lambda: enclosing_opener("(a)", 1, marks)):
            with pytest.raises(ValueError):
                call()

    @pytest.mark.parametrize("sep", ["", "--", "(", ")", "[x"])
    def test_separator_is_one_character_that_is_not_a_counted_mark(self, sep):
        """ ENCLOSING MARKS: a separator cannot be one of the marks that is being counted."""
        with pytest.raises(ValueError):
            top_level_tokens("a-b", sep, ALL)

    def test_negative_positions_are_refused(self):
        """ ENCLOSING MARKS: a position is an index into the name."""
        for call in (lambda: bracket_depths("(a)", start=-1), lambda: matching_close("(a)", -1),
                     lambda: enclosing_closer("(a)", -1)):
            with pytest.raises(ValueError):
                call()

    def test_functions_are_deterministic_and_leave_the_name_alone(self):
        """ ENCLOSING MARKS: the same name always gives the same reading."""
        text = "N-[(2S)-butan-2-yl]-N-(2-methylpropyl)acetamide"
        first = (is_fully_enclosed(text), top_level_tokens(text, "-"), top_level_text(text), matching_close(text, 2))
        assert first == (is_fully_enclosed(text), top_level_tokens(text, "-"), top_level_text(text), matching_close(text, 2))
        assert text == "N-[(2S)-butan-2-yl]-N-(2-methylpropyl)acetamide"
