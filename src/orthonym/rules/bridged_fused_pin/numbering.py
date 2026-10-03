"""Numbering of a bridged fused ring system: the fused parent's own numberings, the
bridge attachment rules and the bridge-atom locants.

 (the Blue Book): "The fused ring system to be bridged is numbered in
the usual way." A fixed numbering (a),:3221) leaves the parent's symmetry as
the only freedom: every numbering is the fixed one composed with an automorphism of the
parent skeleton. (:14225,:14231) then chooses "(a) the lowest set of locants
for all the bridge attachment points considered as a set" and "(b) lowest locants in the
order of citation for the bridges". (:14403) numbers the bridge atoms
"continuing from the highest locant of the fused ring system", starting "at the end of
the chain... connected to the bridgehead of the fused ring system having the highest
locant"; (:14440) numbers first "the bridge attached to the bridgehead with the
higher locant at the first point of difference" and (:14450) numbers bridges
on the same bridgeheads "in accordance with their order of citation".
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .selection import Split


def loc_key(loc: Any) -> Tuple[int, str]:
    """Order of locants: 4 < 4a < 5, the Blue Book)."""
    if isinstance(loc, int):
        return (loc, "")
    if isinstance(loc, tuple):
        return (int(loc[0]), str(loc[1]))
    text = str(loc)
    digits = "".join(c for c in text if c.isdigit())
    return (int(digits), text[len(digits):])


def locant_tuple(locs) -> Tuple[Tuple[int, str], ...]:
    return tuple(sorted(loc_key(x) for x in locs))


@dataclass(frozen=True)
class Numbering:
    """One candidate numbering: every ring-system atom (bridge atoms included) to its
    locant, plus the keys and the bridge citation it implies."""
    atom_to_locant: Dict[int, Any]
    attachment_set: Tuple[Tuple[int, str], ...]
    citation_order: Tuple[Tuple[int, str], ...]
    bridge_locants: Tuple[Tuple[str, Tuple[Any, Any]], ...]


def _parent_numberings(mol, split: Split) -> List[Dict[int, Any]]:
    """Every IUPAC numbering of the residual: its fixed numbering (a)) composed
    with each element-aware automorphism of the residual skeleton (``parents.fused_parent``)."""
    from .parents import fused_parent
    got = fused_parent(mol, set(split.residual))
    return [dict(n) for n in got.numberings] if got is not None else []


def _citation(prefixes: Sequence[str], heads: Sequence[Tuple[Any, Any]],
              ordered: Sequence[bool]):
    """ (:14173) citation order: alphabetical by bridge prefix (parentheses
    ignored); (:14181) the locant pair of a symmetric bridge in numerical
    order, (:14201) that of an unsymmetric or composite bridge "in the order
    expressed or implied by the name of the bridge". Returns [(prefix, (first, second))]."""
    pairs = [(p, tuple(h) if o else tuple(sorted(h, key=loc_key)))
             for p, h, o in zip(prefixes, heads, ordered)]
    return sorted(pairs, key=lambda e: (e[0].strip("()"), [loc_key(x) for x in e[1]]))


def numberings(mol, split: Split, prefixes: Sequence[str],
               first: Optional[Sequence[Optional[int]]] = None) -> List[Numbering]:
    """Every candidate numbering of the bridged system, bridge atoms included. ``first[i]``
    is the bridgehead of bridge i whose locant is cited first (None: numerical order)."""
    first = list(first) if first is not None else [None] * len(split.bridges)
    out = []
    for ring_loc in _parent_numberings(mol, split):
        heads = []
        for (a, b), f in zip(split.bridgeheads, first):
            heads.append((ring_loc[b], ring_loc[a]) if f == b else (ring_loc[a], ring_loc[b]))
        nxt = max(loc_key(v)[0] for v in ring_loc.values()) + 1
        # /: the bridge on the higher bridgeheads first; the
        # same bridgeheads in citation order.
        # (a stable sort keeps the citation order among equal bridgeheads)
        order = sorted(range(len(split.bridges)), key=lambda i: prefixes[i].strip("()"))
        order = sorted(order, reverse=True,
                       key=lambda i: sorted((loc_key(v) for v in heads[i]), reverse=True))
        full = dict(ring_loc)
        for i in order:
            chain = list(split.bridges[i])
            bh0, bh1 = split.bridgeheads[i]
            #: start at the end bonded to the higher-numbered bridgehead.
            if loc_key(ring_loc[bh1]) > loc_key(ring_loc[bh0]):
                chain.reverse()
            for a in chain:
                full[a] = nxt
                nxt += 1
        cited = _citation(prefixes, heads, [f is not None for f in first])
        out.append(Numbering(
            atom_to_locant=full,
            attachment_set=locant_tuple(x for h in heads for x in h),
            citation_order=tuple(loc_key(x) for _, pair in cited for x in pair),
            bridge_locants=tuple(cited)))
    return out
