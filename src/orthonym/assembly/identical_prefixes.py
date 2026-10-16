"""Identical substituent prefixes of one scope are cited once, with a multiplying prefix.

 "MULTIPLICATIVE PREFIXES 'DI', 'TRI', ETC. vs. 'BIS', 'TRIS', ETC." (the Blue Book),
 (:7027): "Multiplicative prefixes... are the principal method for describing a
multiplicity of identical features of a structure in chemical names." The locants of the group
are cited together whatever their kind: "Lowest set of locants" (:3189), "Italic capital
and lower-case letter locants are lower than Greek letter locants, which, in turn, are lower than
numerals" (:3195), so the italic N is cited before the numerals, and (:3448) "multiplicative
prefixes, if necessary, are then inserted and do not alter the alphabetical order already
established". Preferred names that cite them so: ``N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)``
(:32879), ``N,N,2-trimethyl-3-{...}propanamide (PIN)`` (:21624), ``N,N,N,1-tetramethylquinolin-1-ium-
3-aminium (PIN)`` (:42213).

A producer that cites the prefixes of the parent from several sources (the characteristic-group
loop, the substituent walk, the N-substituents of an amine or an amide) used to put the
sources side by side, so ``2-methyl`` and ``N-methyl`` stood in one name and the parent read
``2-(4-methoxyphenyl)-2-methyl-N-methylpropanamide``. This module is the one place that
combines them. Its input is either the parts of the prefixes (name and locants,
:func:`shared_prefix_entries`) or the rendered prefixes (:func:`group_identical_prefixes`);
a rendered prefix is read back into its parts and used only when the engine's own writer of a
multiplied component, ``naming_utils.multiplied_component``, writes the very same text from
them, so a prefix the reader does not fully understand is left as it was. Nothing is cut out
of a finished name: the combined prefix is written again from its parts, and the set of
(locant, substituent name) pairs of the input is the set of the output.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.multipliers import COMPLEX_MULTIPLIERS, SIMPLE_MULTIPLIERS

#: one locant of a rendered prefix: a numeral with its letter and primes ('2', '3a', "2'"), or an
#: italic element locant with its index and primes ('N', "N'", 'N1', 'O', 'S', 'P')
_LOC = r"(?:[0-9]+[a-z]?'*|[NOSP][0-9]*'*)"
_UNIT = re.compile(rf"^({_LOC}(?:,{_LOC})*)-(.+)$", re.S)

#: every multiplying word, longest first so 'tetrakis' is tried before 'tetra'
_MULTIPLIER_WORDS = tuple(sorted(
    {*SIMPLE_MULTIPLIERS.values(), *COMPLEX_MULTIPLIERS.values()}, key=lambda w: (-len(w), w)))


def cited_locants(locants: Sequence) -> List[str]:
    """``locants`` as strings in the order they are cited in a group,
    the Blue Book: italic letters before numerals; ``prefix_locant_order_key``)."""
    from .composition_primitives import prefix_locant_order_key
    return [str(loc) for loc in sorted(locants, key=prefix_locant_order_key)]


def _mult_free_candidates(rest: str):
    """The ``(name, marked)`` readings of the text after the locants: ``name`` is the bare
    substituent name, ``marked`` the same with the enclosing marks it is cited with."""
    from .naming_utils import is_fully_enclosed
    bodies = [rest]
    for word in _MULTIPLIER_WORDS:
        if rest.startswith(word) and len(rest) > len(word):
            tail = rest[len(word):]
            bodies.append(tail)
            if tail.startswith("-"):
                bodies.append(tail[1:])
    for body in bodies:
        if body and is_fully_enclosed(body, "([{"):
            yield body[1:-1], body          # the marks belong to the citation, not to the name
        else:
            yield body, body


class _Unit:
    """One rendered prefix read into its parts."""

    __slots__ = ("text", "locants", "name", "marked")

    def __init__(self, text: str, locants: List[str], name: str, marked: str):
        self.text, self.locants, self.name, self.marked = text, locants, name, marked


def read_prefix(text: str) -> Optional[_Unit]:
    """``text`` ('N-methyl', '2,3-dihydroxy', '2-(4-methoxyphenyl)') read into its locants and
    substituent; ``None`` unless ``multiplied_component`` writes exactly ``text`` from them.
    Two readings that disagree on the name are no reading."""
    from .naming_utils import multiplied_component
    m = _UNIT.match(text)
    if m is None:
        return None
    locants = m.group(1).split(",")
    rest = m.group(2)
    found = {}
    for name, marked in _mult_free_candidates(rest):
        if name and multiplied_component(len(locants), name, marked) == rest:
            found.setdefault((str(name), str(marked)), None)
    if len(found) != 1:
        return None
    (name, marked), = found
    return _Unit(text, locants, name, marked)


def write_prefix(name: str, marked: str, locants: Sequence) -> str:
    """The prefix that cites ``locants`` (as they are cited in a group) in front of the
    substituent ``name`` written with the marks ``marked``."""
    from .naming_utils import multiplied_component
    cited = cited_locants(locants)
    return f"{','.join(cited)}-{multiplied_component(len(cited), name, marked)}"


def group_identical_prefixes(prefixes: Sequence[str], *, sort: bool = True) -> List[str]:
    """The rendered prefixes of one parent with every substituent that is cited twice or more
    combined into one multiplied prefix, locants N-first (``['2-methyl', 'N-methyl']`` gives
    ``['N,2-dimethyl']``), in the order of when ``sort`` (the order the callers give a
    list of rendered prefixes, ``prefix_citation_sort_key`` with ``parent_locants``).

    A prefix the reader does not understand, one without locants, and a substituent whose two
    citations are the very same text (one group an earlier step listed twice, not two
    substituents) are left alone."""
    from .composition_primitives import identical_prefixes_grouped
    units = [read_prefix(p) for p in prefixes] if identical_prefixes_grouped() else [None] * len(prefixes)
    members: Dict[Tuple[str, str], List[int]] = {}
    for i, unit in enumerate(units):
        if unit is not None:
            members.setdefault((unit.name, unit.marked), []).append(i)
    out = list(prefixes)
    merged = []
    consumed = set()
    for (name, marked), idx in members.items():
        if len(idx) < 2 or len({prefixes[i] for i in idx}) < len(idx):
            continue
        consumed.update(idx)
        merged.append(write_prefix(name, marked, [loc for i in idx for loc in units[i].locants]))
    if consumed:
        out = [p for i, p in enumerate(prefixes) if i not in consumed] + merged
    if sort:
        from .naming_utils import prefix_citation_sort_key
        out.sort(key=lambda p: prefix_citation_sort_key(p, parent_locants=True))
    return out


def shared_prefix_groups(ring_groups: Dict[str, List], n_groups: Dict[str, List], tag=str):
    """The substituents the ring and the nitrogens of an amine both carry, as multiplied prefixes.

    ``ring_groups`` maps a substituent name to its locants on the ring and ``n_groups`` maps a
    name to the locants of the nitrogens that carry it (``tag`` turns one of them into the text
    that is cited: 'N', or 'N1' from the superscript 1). A name in both is cited once, the italic
    N locants first (``4-methylaniline`` with a methyl on the nitrogen, ``N,4-dimethylaniline``);
    the grouping is ``composition_primitives.combine_identical_prefix_groups``.

    Returns ``(ring_groups, n_groups, entries)``: the two inputs without the shared names and
    ``entries``, the ``(alphanumerical key, rendered prefix)`` pairs of the shared ones."""
    from .composition_primitives import combine_identical_prefix_groups, identical_prefixes_grouped
    from .naming_utils import alpha_sort_key, format_substituent_prefix
    shared = [name for name in n_groups if name in ring_groups]
    if not shared or not identical_prefixes_grouped():
        return ring_groups, n_groups, []
    groups = combine_identical_prefix_groups(
        [(name, [tag(loc) for loc in n_groups[name]]) for name in shared]
        + [(name, list(ring_groups[name])) for name in shared])
    entries = [(alpha_sort_key(name), format_substituent_prefix(name, locants, len(locants)))
               for name, locants in groups]
    return ({k: v for k, v in ring_groups.items() if k not in shared},
            {k: v for k, v in n_groups.items() if k not in shared}, entries)


def shared_prefix_entries(ring_groups: Dict[str, List[int]], n_names: Sequence[str]):
    """:func:`shared_prefix_groups` for an aniline, whose nitrogen carries the substituents
    ``n_names`` (one name per substituent, a name that occurs twice sits twice on the
    nitrogen). Returns ``(ring_groups, n_names, entries)``."""
    counted: Dict[str, List[str]] = {}
    for name in n_names:
        counted.setdefault(name, []).append("N")
    ring_rest, n_rest, entries = shared_prefix_groups(ring_groups, counted)
    return ring_rest, [n for n in n_names if n in n_rest], entries


# ----------------------------------------------------------------------------------------------
# A prefix block written in front of the name of a parent that cites prefixes of its own
# ----------------------------------------------------------------------------------------------

#: a chain amide parent, with what is glued to its front: 'dimethylbutanamide' is the prefix
#: 'dimethyl' and the parent 'butanamide'
_CHAIN_AMIDE = re.compile(
    r"^(?P<head>.*?)(?P<parent>(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec"
    r"|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)anamide)$")
_LOCANT_SET = re.compile(rf"^{_LOC}(?:,{_LOC})*$")


def _units_of(tokens: List[str]) -> Optional[List[str]]:
    """The prefixes of a list of hyphen-separated tokens that is a run of (locants, body)
    pairs; ``None`` for any other shape."""
    if len(tokens) % 2:
        return None
    out = []
    for locants, body in zip(tokens[0::2], tokens[1::2]):
        if not _LOCANT_SET.match(locants) or not body:
            return None
        out.append(f"{locants}-{body}")
    return out


def split_prefix_zone(name: str) -> Optional[Tuple[List[str], str]]:
    """``(prefixes, parent)`` of a chain-amide name whose prefixes all cite locants:
    ``'2-{...}-3,3-dimethylbutanamide'`` gives ``(['2-{...}', '3,3-dimethyl'], 'butanamide')``.
    ``None`` for a name of any other shape, or when a prefix is not one the engine's writer of
    a multiplied component writes from its parts (``read_prefix``)."""
    from .naming_utils import matching_close, top_level_tokens
    tokens = top_level_tokens(name, "-", "([{")
    if len(tokens) < 2 or len(tokens) % 2:
        return None
    locants, body = tokens[-2], tokens[-1]
    units = _units_of(tokens[:-2])
    if units is None or not _LOCANT_SET.match(locants):
        return None
    glued = _CHAIN_AMIDE.match(body)
    if glued is not None:
        # a simple prefix with the parent glued to it: 'dimethylbutanamide'
        if not glued.group("head"):
            return None
        last, parent = f"{locants}-{glued.group('head')}", glued.group("parent")
    else:
        # an enclosed prefix with the parent glued behind its closing mark
        mult = next((w for w in _MULTIPLIER_WORDS if body[len(w):len(w) + 1] in ("(", "[", "{")
                     and body.startswith(w)), "")
        close = matching_close(body, len(mult)) if body[len(mult):len(mult) + 1] in "([{" else None
        tail = _CHAIN_AMIDE.match(body[close + 1:]) if close is not None else None
        if tail is None or tail.group("head"):
            return None
        last, parent = f"{locants}-{body[:close + 1]}", tail.group("parent")
    units = units + [last]
    if any(read_prefix(u) is None for u in units):
        return None
    return units, parent


def _pairs(units: Sequence[str]) -> Optional[List[tuple]]:
    """The sorted (locant, name, marks) triples of rendered prefixes."""
    out = []
    for text in units:
        unit = read_prefix(text)
        if unit is None:
            return None
        out.extend((loc, unit.name, unit.marked) for loc in unit.locants)
    return sorted(out)


def merge_n_block_into_name(n_block: str, name: str) -> Optional[str]:
    """``name`` (a chain amide that cites prefixes of its own) with the N-substituents of
    ``n_block`` ('N-methyl', 'N,N-diethyl', 'N-ethyl-N-methyl') cited with them: where a
    substituent is cited in both, the group is one multiplied prefix, and the prefixes are in
    the order of ('N-methyl' in front of '3,3-dimethylbutanamide' gives
    'N,3,3-trimethylbutanamide'). ``None`` when no substituent is shared, or when either text
    is not read with certainty; the caller then writes the block in front as before.

    The result is audited: the (locant, substituent) pairs of the two texts are exactly those of
    the result."""
    from .composition_primitives import (
        _join_prefix_to_name,
        _join_prefixes,
        identical_prefixes_grouped,
    )
    from .naming_utils import top_level_tokens
    if not identical_prefixes_grouped():
        return None
    zone = split_prefix_zone(name)
    n_units = _units_of(top_level_tokens(n_block, "-", "([{"))
    if zone is None or n_units is None:
        return None
    units, parent = zone
    before = _pairs(n_units + units)
    if before is None:
        return None
    grouped = group_identical_prefixes(n_units + units)
    if len(grouped) == len(n_units) + len(units) or _pairs(grouped) != before:
        return None
    return _join_prefix_to_name(_join_prefixes(grouped), parent)
