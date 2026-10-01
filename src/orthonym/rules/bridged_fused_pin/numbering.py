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

from rdkit import Chem

from ...perception.automorphisms import skeleton_automorphisms
from .selection import Split


def loc_key(loc: Any) -> Tuple[int, str]:
    """Order of locants: 4 < 4a < 5, the Blue Book)."""
    if isinstance(loc, int):
        return (loc, "")
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
    """Every IUPAC numbering of the residual: its fixed numbering (a)) read off
    one match of the parent, composed with each automorphism of the residual skeleton."""
    from ...data.polycyclic_data import POLYCYCLIC_DATA
    entry = POLYCYCLIC_DATA[split.parent]
    fixed = entry["iupac_numbering"]
    cmol = Chem.MolFromSmiles(entry["canonical_smiles"])
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    query = Chem.AdjustQueryProperties(cmol, params)
    residual = set(split.residual)
    base = None
    for match in mol.GetSubstructMatches(query, uniquify=False, useChirality=False):
        if set(match) == residual:
            base = {match[c]: loc for c, loc in fixed.items()}
            break
    if base is None or set(base) != residual:
        return []
    auts, exhaustive = skeleton_automorphisms(mol, residual, element_blind=False, cap=64)
    if not exhaustive:
        return []
    out, seen = [], set()
    for sigma in auts:
        numb = {a: base[sigma[a]] for a in residual}
        sig = tuple(sorted(numb.items(), key=lambda kv: kv[0]))
        if sig not in seen:
            seen.add(sig)
            out.append(numb)
    return out


def _citation(prefixes: Sequence[str], heads: Sequence[Tuple[Any, Any]]):
    """ (:14173) citation order: alphabetical by bridge prefix;
    (:14181) each locant pair in numerical order. Returns [(prefix, (lo, hi)),...]."""
    pairs = [(p, tuple(sorted(h, key=loc_key))) for p, h in zip(prefixes, heads)]
    return sorted(pairs, key=lambda e: (e[0], [loc_key(x) for x in e[1]]))


def numberings(mol, split: Split, prefixes: Sequence[str]) -> List[Numbering]:
    """Every candidate numbering of the bridged system, bridge atoms included."""
    out = []
    for ring_loc in _parent_numberings(mol, split):
        heads = [(ring_loc[a], ring_loc[b]) for a, b in split.bridgeheads]
        nxt = max(loc_key(v)[0] for v in ring_loc.values()) + 1
        # /: the bridge on the higher bridgeheads first; the
        # same bridgeheads in citation order.
        # (a stable sort keeps the citation order among equal bridgeheads)
        order = sorted(range(len(split.bridges)), key=lambda i: prefixes[i])
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
        cited = _citation(prefixes, heads)
        out.append(Numbering(
            atom_to_locant=full,
            attachment_set=locant_tuple(x for h in heads for x in h),
            citation_order=tuple(loc_key(x) for _, pair in cited for x in pair),
            bridge_locants=tuple(cited)))
    return out
