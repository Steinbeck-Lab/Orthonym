"""v29 P3-REGRESSION — a pre-enclosed FG prefix must not be enclosed a second time.

WHAT BROKE, AND WHY THE GATE WAS THE ONLY THING THAT SAW IT
----------------------------------------------------------
`064fefd5` added `is_substituted_substituent(prefix_form)` as a third disjunct to
the ENCLOSURE decision in `polyfunctional.format_fg_prefix`, so that enclosing
marks would stop being held by the multiplier WORD (the real I13 defect:
`1,3-bis(phosphonooxy)propan-2-ol` had collapsed to `1,3-diphosphonooxypropan-2-ol`).

That disjunct is correct for a BARE prefix and wrong for an ALREADY-ENCLOSED one,
because `is_substituted_substituent` answers the *multiplier* question and
deliberately looks THROUGH enclosing marks — `naming_utils.py`:

    # A fully enclosed token is a single component: decide on its interior
    # (`(2-chloroethyl)` is substituted, `(propan-2-yl)` is not).
    if _is_fully_enclosed(work):
        return is_substituted_substituent(work[1:-1], ...)

so `is_substituted_substituent('(dimethylamino)')` is `True`. Callers legitimately
pre-enclose (`polyfunctional.py:1693` passes `f"({_oxa}imino)"` verbatim), and the
result was a second enclosure plus a lost hyphen:

    methyl 4-(dimethylamino)-4-(ethylimino)butanoate      <- gold, and correct
    methyl 4-[(dimethylamino)]4-[(ethylimino)]butanoate   <- shipped at 64af6b2f

Four gold TARGET rows regressed (gate `target_passes` 1641 -> 1637). **Every one of
the four still round-trips through OPSIN cleanly**, which is exactly why no RT-based
check could see it: OPSIN re-parses the redundant brackets and the missing hyphen
and returns the right structure. This is the DEF-4/DEF-8 format class the
exact-match PIN gate exists for.

THE SECOND, INDEPENDENT DEFECT
------------------------------
`_join_prefixes` inserts the separating hyphen after a letter or `)` but not after
`]` or `}`. That is why the hyphen vanished above — and it is reachable without the
double-enclosure at all, on the escalated prefix the module's own comment
documents (`polyfunctional.py:407`, `'[(methylcarbamoyl)amino]'`, W2F-P6):

    _join_prefixes(['2-[(methylcarbamoyl)amino]', '4-methyl'])
        -> '2-[(methylcarbamoyl)amino]4-methyl'      (no hyphen)

Both are pinned here so neither can regress silently again.

BLUE BOOK
---------
`**P-16.5.1.1**` (BlueBookV2.md:7232), verbatim: "*Parentheses are used around
compound (see P-29.1.2) and complex (see P-29.1.3) prefixes; after the
multiplicative prefixes 'bis', 'tris', etc.*" — a requirement already SATISFIED by
the marks the caller supplied.

`### **P-16.5.4** Multiple types of enclosing marks` (:7444), verbatim at :7446:
"*When multiple types of enclosing marks are required, the nesting order is as
follows: {[({[( )]})]}, etc.*" — a further type is taken only when one is
REQUIRED, so a prefix already carrying its outermost marks does not take another.

(The section heading is "Multiple types of enclosing marks". An earlier draft of
this fix cited it as "Nesting order of enclosing marks", which is the heading it
does NOT have; corrected after reading the line. Chapter P-16 is OCR-mangled in
this file — `P"16.5.4`, `!` for spaces, `G` for hyphens — so a literal grep for
`P-16.5.4` finds NOTHING. The pattern that works is `P"16.5.4`, positive-controlled
by `grep -c 'P"16'` = 64 hits.)
"""

import pytest

from orthonym.assembly.naming_utils import _is_fully_enclosed
from orthonym.rules.polyfunctional import _join_prefixes, format_fg_prefix

pytestmark = pytest.mark.unit


# The exact prefix strings the four regressed gold rows flow through, each as the
# caller hands it over: already carrying its own outermost parentheses.
PRE_ENCLOSED_PREFIXES = [
    "(dimethylamino)",
    "(ethylimino)",
    "(3-hydroxypropoxy)",
    "(ethylsulfanyl)",
    "(hydroxyimino)",
]


class TestPreEnclosedPrefixKeepsOneLevel:
    """A prefix that arrives fully enclosed keeps exactly the marks it arrived with."""

    @pytest.mark.parametrize("prefix", PRE_ENCLOSED_PREFIXES)
    def test_locant_branch_does_not_double_enclose(self, prefix):
        assert format_fg_prefix(prefix, [4], 1) == f"4-{prefix}"

    @pytest.mark.parametrize("prefix", PRE_ENCLOSED_PREFIXES)
    def test_no_locant_branch_does_not_double_enclose(self, prefix):
        assert format_fg_prefix(prefix, [], 1) == prefix

    @pytest.mark.parametrize("prefix", PRE_ENCLOSED_PREFIXES)
    def test_multiplied_pre_enclosed_keeps_one_level(self, prefix):
        # count > 1 goes down a different return path; it must not double-enclose
        # either. `di`/`bis` choice is NOT under test here — only the marks.
        out = format_fg_prefix(prefix, [2, 4], 2)
        assert out.endswith(prefix), out
        assert "[(" not in out and ")]" not in out, out

    @pytest.mark.parametrize("prefix", PRE_ENCLOSED_PREFIXES)
    def test_the_prefixes_really_are_fully_enclosed(self, prefix):
        # Guards the test data itself: if these stopped being fully-enclosed the
        # assertions above would be vacuous rather than failing.
        assert _is_fully_enclosed(prefix) is True


class TestBarePrefixStillGetsItsMarks:
    """The I13 fix must survive: a BARE compound prefix still gets enclosed."""

    @pytest.mark.parametrize("bare,expected", [
        ("dimethylamino", "4-(dimethylamino)"),
        ("hydroxyimino", "4-(hydroxyimino)"),
        ("phosphonooxy", "4-(phosphonooxy)"),
        ("methoxymethyl", "4-(methoxymethyl)"),
        ("ethylsulfanyl", "4-(ethylsulfanyl)"),
    ])
    def test_bare_compound_prefix_is_enclosed(self, bare, expected):
        assert format_fg_prefix(bare, [4], 1) == expected

    @pytest.mark.parametrize("simple,expected", [
        ("hydroxy", "4-hydroxy"),
        ("oxo", "4-oxo"),
        ("amino", "4-amino"),
        ("methoxy", "4-methoxy"),
        ("nitroso", "4-nitroso"),
    ])
    def test_simple_prefix_stays_bare(self, simple, expected):
        assert format_fg_prefix(simple, [4], 1) == expected

    def test_partially_enclosed_prefix_still_escalates(self):
        # `(methylcarbamoyl)amino` carries inner marks but is NOT fully enclosed,
        # so it takes the next level out — the pre-existing W2F-P6 behaviour.
        assert (format_fg_prefix("(methylcarbamoyl)amino", [2], 1)
                == "2-[(methylcarbamoyl)amino]")

    def test_bis_on_bare_substituted_prefix_is_preserved(self):
        # The I13 keystone: enclosure must not be held by the multiplier word.
        assert format_fg_prefix("phosphonooxy", [1, 3], 2) == "1,3-bis(phosphonooxy)"


class TestJoinPrefixesHyphenatesAfterEveryEnclosingMark:
    """`_join_prefixes` must separate a locant-initial prefix from ANY closing mark."""

    @pytest.mark.parametrize("left,right,expected", [
        # the pre-existing, already-correct cases
        ("2-(methylsulfanyl)", "4-methyl", "2-(methylsulfanyl)-4-methyl"),
        ("2-amino", "4-methyl", "2-amino-4-methyl"),
        # `]` — reachable on the documented W2F-P6 escalated prefix
        ("2-[(methylcarbamoyl)amino]", "4-methyl", "2-[(methylcarbamoyl)amino]-4-methyl"),
        # `}` — the third nesting level
        ("2-{[(methylcarbamoyl)amino]methyl}", "4-methyl",
         "2-{[(methylcarbamoyl)amino]methyl}-4-methyl"),
    ])
    def test_hyphen_inserted_before_a_locant(self, left, right, expected):
        assert _join_prefixes([left, right]) == expected

    @pytest.mark.parametrize("left,right,expected", [
        # An italic-N locant is a locant-bearing term too — the module's own
        # comment says so ("like a numeric locant", HYG-04 site#2) — so it takes
        # the separator after `]`/`}` for exactly the same reason.
        ("2-(methylsulfanyl)", "N,N-dimethyl", "2-(methylsulfanyl)-N,N-dimethyl"),
        ("2-[(methylcarbamoyl)amino]", "N,N-dimethyl",
         "2-[(methylcarbamoyl)amino]-N,N-dimethyl"),
        ("2-{[(methylcarbamoyl)amino]methyl}", "N-methyl",
         "2-{[(methylcarbamoyl)amino]methyl}-N-methyl"),
    ])
    def test_hyphen_inserted_before_an_italic_n_locant(self, left, right, expected):
        assert _join_prefixes([left, right]) == expected

    def test_no_spurious_hyphen_before_a_letter(self):
        # The rule is letter/mark -> DIGIT. A letter-initial follower must not
        # gain a hyphen from this rule.
        assert _join_prefixes(["2-[(methylcarbamoyl)amino]", "methyl"]) == \
            "2-[(methylcarbamoyl)amino]methyl"
