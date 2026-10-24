"""Assembly of a phane name from a numbered skeleton,,,.

the Blue Book, each line read with ``sed -n``: (:14922) the amplification prefixes are cited in
decreasing seniority of ring; (:14926) 'di', 'tri',... or 'bis', 'tris',... multiply
them; (:14937) and (:14947) the superatom locants of amplificants with one attachment
set are cited together, in front of the attachment locants in parentheses; (:15192) "First, the
parent phane hydrocarbon is named and then the heteroatoms are denoted by means of nondetachable 'a'
prefixes cited in front of the name so created. Locants for the heteroatoms are assigned according to the
numbering of the simplified parent skeleton"; (:16489) "In names, the ending 'ene' always
precedes 'yne', with elision of the final letter 'e' in 'ene'"; (a) (:7595) the terminal 'e' of the
parent hydride or of 'ene' and 'yne' goes before a suffix or 'en' ending that starts with 'a', 'e', 'i', 'o',
'u' or 'y'.

Format: ``[prefixes-][a-prefixes-]<amplification block><skeleton>[endings][suffix]``.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from .numbering import (
    Numbered,
    Placed,
    UndecidableTiesError,
    best_numberings,
    render_locant,
    skeleton_heteroatoms,
)
from .skeleton import PhaneSkeleton, SkeletonKind, bridges_between, spiro_rings
from .substituents import Plan

_LOG = logging.getLogger(__name__)
_VOWELS = "aeiouy"


def _loc(locant: Tuple[int, str]) -> str:
    """(4, "a") is 4a."""
    return f"{locant[0]}{locant[1]}"


def amplification_block(placed: Tuple[Placed, ...]) -> Optional[str]:
    """,,,: ``1,7(1),3,5(1,4)-tetrabenzena``."""
    from ..phane import _multiplied_amplificant
    types = {}
    for p in placed:
        types.setdefault((p.amp.hydride.score, p.amp.hydride.key), []).append(p)
    parts = []
    for (_, _), items in sorted(types.items()):
        groups = {}
        for p in items:
            groups.setdefault(p.cited, []).append(p.locant)
        text = ",".join(",".join(map(str, sorted(locs))) + "(" + ",".join(_loc(x) for x in cited) + ")"
                        for cited, locs in sorted(groups.items(), key=lambda kv: min(kv[1])))
        multiplied = _multiplied_amplificant(items[0].amp.hydride.prefix, len(items))
        if multiplied is None:
            return None
        parts.append(f"{text}-{multiplied}")
    return "-".join(parts)


def skeleton_word(skel: PhaneSkeleton) -> Optional[str]:
    """ (:14847): the structure prefix, the number of nodes (superatoms included) and 'phane'."""
    from ..phane import SkeletonClass, _simplified_skeletal_name
    if skel.kind is SkeletonKind.CYCLE:
        return _simplified_skeletal_name(len(skel.nodes), SkeletonClass.MONOCYCLIC)
    if skel.kind is SkeletonKind.PATH:
        return _simplified_skeletal_name(len(skel.nodes), SkeletonClass.ACYCLIC)
    from ..phane import _phane_node_multiplier
    adj = {node: list(nbs) for node, nbs in skel.adj.items()}
    if skel.kind is SkeletonKind.BICYCLO:
        # (:14878); the descriptor of: the interior lengths of the bridges, longest first
        heads = sorted(n for n in skel.nodes if len(adj[n]) == 3)
        bridges = bridges_between(adj, heads[0], heads[1])
        lengths = ".".join(str(n) for n in sorted((len(b) for b in bridges), reverse=True))
        return f"bicyclo[{lengths}]{_phane_node_multiplier(len(skel.nodes))}phane"
    if skel.kind is SkeletonKind.SPIRO:
        # (:14884), (:9961): the numbers of nodes linked to the spiro node in each ring, ascending
        (spiro,) = [n for n in skel.nodes if len(adj[n]) == 4]
        lengths = ".".join(str(n) for n in sorted(len(r) for r in spiro_rings(adj, spiro)))
        return f"spiro[{lengths}]{_phane_node_multiplier(len(skel.nodes))}phane"
    return None


def _locs(locants: Sequence) -> str:
    return ",".join(map(str, locants))


def _mult(locants: Sequence) -> Optional[str]:
    """di, tri,... for a set of locants; nothing for one or none."""
    from ...assembly.naming_utils import SIMPLE_MULTIPLIERS
    return "" if len(locants) <= 1 else SIMPLE_MULTIPLIERS.get(len(locants))


def endings(base: str, enes: Sequence[int], ynes: Sequence[int]) -> Optional[str]:
    """``phane`` -> ``phan-2-ene``, ``phane-2,8-diene``, ``phan-4-en-2-yne``, ``phane-5,7-dien-9-yne``.

     (a) (:7595): the terminal 'e' goes only before an ending that starts with 'a', 'e', 'i', 'o',
    'u' or 'y': 'en' and 'yn' take it away from 'phane', 'dien' and 'diyne' do not."""
    if not (enes or ynes):
        return base
    me, my = _mult(enes), _mult(ynes)
    if me is None or my is None:
        return None
    stem = base if (me if enes else my) else base[:-1]
    if enes and ynes:
        return stem + f"-{_locs(enes)}-{me}en-{_locs(ynes)}-{my}yne"
    if enes:
        return stem + f"-{_locs(enes)}-{me}ene"
    return stem + f"-{_locs(ynes)}-{my}yne"


def with_suffix(ending: str, locants: Sequence[str], core: str) -> Optional[str]:
    """ (a): the terminal 'e' goes before a suffix that starts with a vowel; the multiplier is part
    of the suffix ('dione' starts with 'd')."""
    multiplier = _mult(locants)
    if multiplier is None:
        return None
    word = multiplier + core
    stem = ending[:-1] if word[0] in _VOWELS and ending.endswith("e") else ending
    return f"{stem}-{_locs(locants)}-{word}"


def prefix_block(plan: Plan, numbered: Numbered, skel: PhaneSkeleton) -> Optional[str]:
    """The detachable prefixes in alphanumerical order, each name with the locants of its atoms in the order
    of (never contracted,,:15113)."""
    from ...assembly.general_engine import _alpha_key, _mult_prefix
    from .numbering import atom_locant
    groups = {}
    for prefix in plan.prefixes:
        groups.setdefault(prefix.name, []).append(atom_locant(skel, numbered.pos, numbered.chosen, prefix.atom))
    parts: List[str] = []
    for name in sorted(groups, key=lambda n: (_alpha_key(n), n)):
        text = _mult_prefix(len(groups[name]), name)
        if text is None:
            return None
        parts.append(",".join(render_locant(loc) for loc in sorted(groups[name])) + "-" + text)
    return "-".join(parts)


def build_name(skel: PhaneSkeleton, numbered: Numbered, plan: Optional[Plan] = None) -> Optional[str]:
    """The name of ``skel`` under one numbering, or None when a part of it is not built."""
    from .numbering import atom_locant
    plan = plan or Plan((), ())
    if numbered.wraps:
        return None                 # a multiple bond between the last and the first node is cited '1(n)': not built
    block = amplification_block(numbered.placed)
    word = skeleton_word(skel)
    if block is None or word is None:
        return None
    name = endings(block + word, numbered.enes, numbered.ynes)
    if name is None:
        return None
    if plan.suffixes:
        locants = sorted(atom_locant(skel, numbered.pos, numbered.chosen, s.atom) for s in plan.suffixes)
        name = with_suffix(name, [render_locant(loc) for loc in locants], plan.suffixes[0].core)
        if name is None:
            return None
    hetero = skeleton_heteroatoms(skel, numbered.pos)
    if hetero:
        from ..phane import _apply_skeletal_replacement
        name = _apply_skeletal_replacement(hetero, name)
    if plan.prefixes:
        prefixes = prefix_block(plan, numbered, skel)
        if prefixes is None:
            return None
        name = f"{prefixes}-{name}"
    return name


def phane_candidate(skel: PhaneSkeleton, plan: Optional[Plan] = None) -> Optional[Tuple[tuple, str]]:
    """``(rank, name)``: the name under the numbering with the lowest key (of numberings with equal keys the
    lowest name; a tie that gives different names means a rule is missing, and is logged) and what ranks
    skeletons that tie on the nodes: (:21604) the maximum number of substituents cited as prefixes,
     (:21698) their lower locants, (:21791) in their order of citation."""
    try:
        numberings = best_numberings(skel, plan)
    except UndecidableTiesError:
        return None
    names = {build_name(skel, numbered, plan): numbered for numbered in numberings}
    names.pop(None, None)
    if not names:
        return None
    if len(names) > 1:
        _LOG.warning("phane numberings with one key give different names: %s", sorted(names))
    name = min(names)
    key = names[name].key
    return (-len(plan.prefixes) if plan else 0, key[-2], key[-1]), name


def phane_name(skel: PhaneSkeleton, plan: Optional[Plan] = None) -> Optional[str]:
    """The name of ``skel`` (see:func:`phane_candidate`), or None."""
    got = phane_candidate(skel, plan)
    return None if got is None else got[1]
