"""The enclosing marks of the second and further simple prefix of a mononuclear parent.

 "Parentheses are placed also around prefixes denoting simple substituent groups qualified by
locants" (the Blue Book), (:7272): "For mononuclear parent hydrides with two or more
substituents the first cited substituent never has enclosing marks unless it includes a locant. The second and further substituents
are each enclosed with parentheses even for simple substituents. When the simple substituent groups are
accompanied by multiplicative prefixes such as 'di' and 'tri', the multiplicative prefixes are not included in
the parentheses." Its examples: 'ethyldi(methyl)phosphane (PIN)' (:7290), 'ethyldi(propan-2-yl)silane (PIN)'.
So the second cited group of two phenyls is 'di(phenyl)', never '(diphenyl)'.
"""
from __future__ import annotations

from .composition_primitives import _MULTIPLIER_PREFIXES


def _simple_prefixes() -> frozenset:
    """The simple substituent names the multiplied form is read for: the lexicon of the spelling reader (its
    own copy of the book's closed list, ``validation.spelling.units``) and the unbranched alkyl groups."""
    from ..validation.spelling.units import SIMPLE, STEMS
    return frozenset(SIMPLE) | frozenset(stem + 'yl' for stem in STEMS)


def enclose_second_prefix(part: str, depth: int = -1) -> str:
    """``part`` (a prefix cited after another, without a locant of its own) with its enclosing marks
    (:func:`naming_utils.apply_enclosing_marks`), a multiplier of a simple group left outside them:
    ``'diphenyl'`` -> ``'di(phenyl)'``. A name that is itself one simple group ('diazenyl', 'disilyl'), a
    compound one ('dimethylamino') and any other text are enclosed whole, as before."""
    from .naming_utils import apply_enclosing_marks
    simple = _simple_prefixes()
    if part not in simple:
        for mult in _MULTIPLIER_PREFIXES:
            if part.startswith(mult) and part[len(mult):] in simple:
                return mult + apply_enclosing_marks(part[len(mult):], depth)
    return apply_enclosing_marks(part, depth)
