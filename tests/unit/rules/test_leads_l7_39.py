"""Leads program L7 / 39 (REFUTED): the strict path's enclosing marks follow.

The two names pinned in tests/integration/test_ci_benchmark.py (ci-046, ci-056) were
recorded as "enclosing marks do not follow ". They do. This file re-derives the
marks of a name from its nesting TREE, independently of the engine (naming_utils._mark_levels
is not used), and checks the two names, the Blue Book's own examples, and that the checker
rejects a wrong mark.

 'Multiple types of enclosing marks' (the Blue Book),:7446: the nesting order
is {[({})]}, i.e. level 1 '(', 2 '[', 3 '{', 4 '(' and so on. (:7478): the
parentheses of stereodescriptors count. (:7509): "When the nesting order given in
 results in consecutive enclosing marks of the same level, the next level of enclosing
mark is used." Siblings keep their own level: '4-(6-{2-[(3-methylphenyl)methylidene]hydrazin-
1-yl}-2-[2-(pyridin-2-yl)ethoxy]pyrimidin-4-yl)morpholine (PIN)' (:19441; 'General
methodology':19420) has '{' (level 3) and '[' (level 2) side by side in a level-4 '('.

Scope of the checker: a name whose brackets are all enclosing marks (no ring-fusion, spiro,
von Baeyer, ring-assembly or isotope brackets, /.1.4, and no indicated-hydrogen
parentheses,.
"""
import pytest

_OPEN = "([{"
_CLOSE = ")]}"
_BY_LEVEL = "([{"  # level k -> _BY_LEVEL[(k - 1) % 3]; the order repeats ([ { ([ {


class _Group:
    def __init__(self, mark, start):
        self.mark = mark
        self.start = start
        self.end = None
        self.children = []


def _parse(name):
    root = _Group(None, -1)
    stack = [root]
    for i, ch in enumerate(name):
        if ch in _OPEN:
            g = _Group(ch, i)
            stack[-1].children.append(g)
            stack.append(g)
        elif ch in _CLOSE:
            g = stack.pop()
            assert g.mark is not None and _OPEN.index(g.mark) == _CLOSE.index(ch), (
                f"mismatched {g.mark!r}{ch!r} at {i} in {name!r}")
            g.end = i
    assert stack == [root], f"unbalanced marks in {name!r}"
    return root


def violations(name):
    """The groups of ``name`` whose mark is not the one requires."""
    bad = []

    def level_of(g):
        natural = 1 + max((level_of(c) for c in g.children), default=0)
        #: an opening (or closing) mark that stands directly next to a child's
        # mark of the same level takes the next level. The child's REQUIRED mark is compared,
        # so a wrongly marked child cannot hide a clash.
        adjacent = [c for c in g.children if c.start == g.start + 1 or c.end == g.end - 1]
        level = natural
        while any(_BY_LEVEL[(level - 1) % 3] == _BY_LEVEL[(c.level - 1) % 3]
                  for c in adjacent):
            level += 1
        g.level = level
        if g.mark != _BY_LEVEL[(level - 1) % 3]:
            bad.append((g.mark, name[g.start:g.start + 24]))
        return level

    root = _parse(name)
    for c in root.children:
        level_of(c)
    return bad


# The two pinned strict-path names and the Blue Book's own names (ring brackets absent).
STRICT_NAMES = [
    # ci-046
    "O-{[(2R)-2-{[(11Z,14Z)-icosa-11,14-dienoyl]oxy}-3-{[(9Z,12Z,15Z)-octadeca-9,12,15-"
    "trienoyl]oxy}propoxy]hydroxyphosphoryl}-L-serine",
    # ci-056
    "N-(2-(4-amino-4-carboxybutanamido)-3-{[1-(2-hydroxyphenyl)-3-oxopropyl]sulfanyl}"
    "propanoyl)glycine",
]
BLUE_BOOK_NAMES = [
    #,:7509ff (with stereochemistry; ring bracket-free)
    "(3S)-2-[(2S)-2-{[(2S)-1-ethoxy-1-oxo-4-phenylbutan-2-yl]amino}propanoyl]-1,2,3,4-"
    "tetrahydroisoquinoline-3-carboxylic acid",
    #,:7509ff (without stereochemistry)
    "2-{2-[(1-ethoxy-1-oxo-4-phenylbutan-2-yl)amino]propanoyl}-1,2,3,4-"
    "tetrahydroisoquinoline-3-carboxylic acid",
    #,:19441 (sibling levels)
    "4-(6-{2-[(3-methylphenyl)methylidene]hydrazin-1-yl}-2-[2-(pyridin-2-yl)ethoxy]"
    "pyrimidin-4-yl)morpholine",
    #:7446
    "2,2′-({2-[(carboxymethyl)(2-hydroxyethyl)amino]ethyl}azanediyl)diacetic acid",
]


@pytest.mark.unit
class TestEnclosingMarksFollowP16_5_4:
    @pytest.mark.parametrize("name", BLUE_BOOK_NAMES)
    def test_checker_accepts_the_blue_book_names(self, name):
        assert violations(name) == []

    @pytest.mark.parametrize("name", STRICT_NAMES)
    def test_strict_path_names_follow_the_nesting_order(self, name):
        assert violations(name) == []

    def test_checker_rejects_a_wrong_mark(self):
        # the propanoyl group of ci-056 is level 4 '('; '[' there is wrong
        wrong = (
            "N-[2-(4-amino-4-carboxybutanamido)-3-{[1-(2-hydroxyphenyl)-3-oxopropyl]sulfanyl}"
            "propanoyl]glycine"
        )
        assert violations(wrong) != []

    def test_checker_rejects_consecutive_parentheses(self):
        # ci-046's propoxy group written '((2R)-...': consecutive marks of the same level
        wrong = STRICT_NAMES[0].replace("{[(2R)-2-", "{((2R)-2-").replace(
            "propoxy]hydroxyphosphoryl}", "propoxy)hydroxyphosphoryl}")
        assert violations(wrong) != []
