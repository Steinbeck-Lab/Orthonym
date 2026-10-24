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


def mononuclear_group_token(count, name, *, first, enclosed=False):
    """One group of the substituents on a mononuclear centre, cited with its multiplier: the decision of
     (:7272) for ONE group, made here for every writer that cites the groups of a centre.

    ``first`` is the group cited first: it takes only the marks its own name needs,
    ``enclose_if_compound``). Every later group is enclosed, the mark escalating (-> [ -> { over an inner
    pair and none added to a group the caller cites already enclosed. ``enclosed`` names a
    group the caller knows to be compound (a ligand 'acetyloxy', an 'imino' group) and which takes marks
    as the first group too. The multiplier and its hyphen come from ``multiplied_component``: a simple
    group is 'di(methyl)', a compound one 'bis(cyclohexylmethyl)', and 'di-tert-butyl' keeps
    its hyphen (d))."""
    from .naming_utils import (
        _is_fully_enclosed,
        apply_enclosing_marks,
        enclose_if_compound,
        multiplied_component,
    )
    marked = enclose_if_compound(name)
    if marked == name and (enclosed or not first) and not _is_fully_enclosed(name):
        marked = apply_enclosing_marks(name, -1)
    return multiplied_component(count, name, marked)


def mononuclear_substituent_block(names, *, max_count=None, enclosed=frozenset()):
    """The citation of the substituent names on ONE mononuclear centre (a parent hydride, or the substituent
    group derived from it), (:7272): "For mononuclear parent hydrides with two or more
    substituents the first cited substituent never has enclosing marks unless it includes a locant. The
    second and further substituents are each enclosed with parentheses even for simple substituents. When the
    simple substituent groups are accompanied by multiplicative prefixes such as 'di' and 'tri', the
    multiplicative prefixes are not included in the parentheses."

    THE ONE implementation of that sentence for the mononuclear hydrides (silane, germane, stannane, plumbane,
    borane, phosphane, arsane, stibane, the -ium cations on them) and for the groups named on them (silyl,
    germyl, boranyl, phosphanyl, arsanyl,...): the parent ('ethyldi(methyl)phosphane (PIN)',:7290) and the
    prefix ('4-[ethyl(methyl)phosphanyl]-1H-imidazole (PIN)',:39228; '3-[amino(methyl)silyl]...cyclopentan-1-ol
    (PIN)',:3545) take the same block, so 'dimethyl(propyl)silyl' and 'ethyldi(methyl)silyl' are spelled as
    'dimethyl(propyl)silane' and 'ethyldi(methyl)silane' are. The parents that cite no locant for their
    groups and that the engine spells the same way (the a(ba)n chains of, diazene, nitramide) take
    it too.

    Order is the / citation order (``prefix_citation_sort_key``: the letters, then the prefix's
    own locants, so 'tert-butyl' stands at 'b'; and the lambda convention after them). Each group is
    ``mononuclear_group_token``.

    ``names`` are the bare names of the substituents, one per bond; a name repeated is one multiplied group.
    ``max_count`` bounds the multiplicity a caller can spell (None: no bound); a larger one gives None.
    ``enclosed``: the names that take marks as the first group too (see ``mononuclear_group_token``).

    RULING (coordinator, 2026-10-24) for a multiplied later group of a silyl or germyl GROUP. The book
    contradicts itself: 'aminodichlorosilyl (preselected prefix)' (:36385), '(tert-butylperoxy)dimethylsilyl
    propanoate (PIN)' (:23389) and '(acetyldimethylsilyl)methoxy' (:3553) cite it bare, while the rule's text
    (:7272, maintained with an erratum in the online edition), its parents ('ethyldi(methyl)phosphane',
    'tert-butyldi(methyl)phosphane':16286, 'chlorodi(methyl)borane':25866) and the groups
    'tert-butoxytri(phenyl)-lambda5-phosphanyl (PIN)' (:40442) and 'methyldi(phenyl)phosphaniumyl' (:42468)
    enclose it. The rule text is followed.
    """
    from collections import Counter

    from .naming_utils import prefix_citation_sort_key
    counts = Counter(names)
    parts = []
    for i, name in enumerate(sorted(counts, key=prefix_citation_sort_key)):
        count = counts[name]
        if max_count is not None and count > max_count:
            return None
        parts.append(mononuclear_group_token(count, name, first=(i == 0), enclosed=name in enclosed))
    return ''.join(parts)


def mononuclear_group_name(names, stem, *, max_count=None):
    """The substituent group named on a mononuclear hydride (``stem``: 'silyl', 'germyl',...) with the
    substituents ``names`` on its centre: the block of:func:`mononuclear_substituent_block` and the stem,
    carrying the writer's record (``prefix_derivation``). A group with substituents on it is a substituted
    group, so it takes the derived multipliers 'bis', 'tris' (c), the Blue Book;
     (a)) and the marks of a compound prefix,:7232), whatever its text: the enclosed
    second group ('dimethyl(propyl)silyl') is not a text the multiplier predicate reads as substituted.
    A group with no substituent is the bare stem. None when the block is None."""
    if not names:
        return stem
    block = mononuclear_substituent_block(names, max_count=max_count)
    if block is None:
        return None
    from .prefix_derivation import built
    return built(f"{block}{stem}", substituted=True, enclosed=True)
