"""The preferred name of a linear polyanhydride with three or more links: method (1).

 (the Blue Book) "Linear polyanhydrides are named by one of the following methods: (1) by selecting
the preferred dicarboxylic acid and citing the adjoining acid groups one of which will be substituted using the
principles of substitutive nomenclature... Method (1) generates preferred IUPAC names." The name is a dianhydride
,:32380): the preferred dicarboxylic acid and its two adjoining acids, each adjoining acid carrying the
rest of the chain on its side as an acyloxy prefix, "cited in their order of occurrence in the structure beginning
with the end acid group lower in alphabetical order": 'acetic butanedioic 4-oxo-4-(propanoyloxy)butanoic
dianhydride' (:32402, the book cites the prefixes of that word out of alphanumerical order; its later examples,
'4-[4-(acetyloxy)-4-oxobutanoic] 2-methylbutanedioic 1-propanoic dianhydride (PIN)':32413, keep the order).

The preferred dicarboxylic acid is chosen as orders acid parents, as far as this module decides it: a
carboxylic acid before a sulfonic one ('acetic butanedioic [(acetyloxy)sulfonyl]methanesulfonic dianhydride (PIN)',
:32415), a ring before a chain, the longer chain between the two groups, then the residue with more atoms
('2-methylbutanedioic',:32413). Two residues this order cannot tell apart that give different words leave the name
unbuilt (no guess at a PIN); two equal residues give the name whose first word is lower in alphabetical order
(the book's choice at:32402). Every name is offered only when its acids read back to the input
(``validation.anhydride_acids``).
"""
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem


def _chain(mol, links):
    """``(residues in chain order, carbonyls of each, bridges)`` of a linear polyanhydride, else None."""
    from .anhydrides import _bfs_side_atoms
    bridges = {bridge for _, bridge, _ in links}
    carbonyls = sorted({c for first, _, second in links for c in (first, second)})
    frag_of = {c: frozenset(_bfs_side_atoms(mol, c, bridges)) for c in carbonyls}
    residues: Dict[frozenset, List[int]] = {}
    for carbonyl, frag in frag_of.items():
        residues.setdefault(frag, []).append(carbonyl)
    adjacent = {frag: [] for frag in residues}
    for first, _, second in links:
        adjacent[frag_of[first]].append(frag_of[second])
        adjacent[frag_of[second]].append(frag_of[first])
    ends = [frag for frag in residues if len(adjacent[frag]) == 1]
    if len(ends) != 2 or any(len(a) > 2 for a in adjacent.values()):
        return None
    order, previous, current = [], None, ends[0]
    while current is not None:
        order.append(current)
        following = [f for f in adjacent[current] if f != previous and f not in order]
        previous, current = current, (following[0] if following else None)
    if len(order) != len(residues):
        return None
    return order, residues, frag_of, links


def _side_word(mol, chain, center: int, side: Sequence[int]):
    """The acid word of the adjoining residue ``side[0]`` with the residues beyond it as its acyloxy prefix."""
    from .anhydrides import _residue_acid_name
    order, residues, frag_of, links = chain
    frags = {order[k] for k in side}
    keep = set().union(*frags)
    near = None
    for first, bridge, second in links:
        ends = {frag_of[first], frag_of[second]}
        if ends <= frags:
            keep.add(bridge)
        elif ends == {order[side[0]], order[center]}:
            near = first if frag_of[first] == order[side[0]] else second
    return None if near is None else _residue_acid_name(mol, keep, [near])


def _rank(mol, frag, carbonyls) -> Tuple[int, int, int, int]:
    """The order of a candidate dicarboxylic acid residue, lower first (module docstring)."""
    not_carbon = sum(1 for c in carbonyls if mol.GetAtomWithIdx(c).GetSymbol() != 'C')
    ring = 0 if any(mol.GetAtomWithIdx(a).IsInRing() for a in frag) else 1
    path = Chem.GetShortestPath(mol, carbonyls[0], carbonyls[1]) if len(carbonyls) == 2 else ()
    return not_carbon, ring, -len(path), -len(frag)


def method_one_words(mol, links) -> Optional[List[str]]:
    """The acid words of the method (1) name, in their order (module docstring), or None."""
    from ..assembly.naming_utils import alpha_sort_key
    from .anhydrides import _multiplied_acid_word, _residue_acid_name
    chain = _chain(mol, links)
    if chain is None:
        return None
    order, residues, _frag_of, _links = chain
    best = []
    for i in range(1, len(order) - 1):
        center = _residue_acid_name(mol, order[i], residues[order[i]])
        left = _side_word(mol, chain, i, range(i - 1, -1, -1))
        right = _side_word(mol, chain, i, range(i + 1, len(order)))
        if center is None or left is None or right is None:
            continue
        words = [left, center, right]
        if alpha_sort_key(right[0]) < alpha_sort_key(left[0]):
            words.reverse()
        best.append((_rank(mol, order[i], residues[order[i]]), center[0], words))
    if not best:
        return None
    top = min(key for key, _c, _w in best)
    chosen = [(c, w) for key, c, w in best if key == top]
    if len({c for c, _w in chosen}) > 1:
        return None                     # two different acids this order cannot rank: no guess at the PIN
    words = min((w for _c, w in chosen), key=lambda w: [alpha_sort_key(s) for s, _ in w])
    cited, counts = [], {}
    for stem, simple in words:
        if stem not in counts:
            cited.append((stem, simple))
            counts[stem] = 0
        counts[stem] += 1
    return [_multiplied_acid_word(stem, counts[stem], simple) for stem, simple in cited]


def record_one_link_name(mol, name: str) -> None:
    """Label ``name`` below the PIN when ``mol`` has two or more anhydride links: a name of one class word cites the
    further links in an acid word, and the PIN is the method (1) dianhydride, the Blue Book)."""
    from ..metrics.provenance import record_non_pin_label
    from ..perception.smarts_cache import compiled
    from .anhydrides import ANHYDRIDE_SMARTS
    pattern = compiled(ANHYDRIDE_SMARTS)
    if pattern is not None and len(mol.GetSubstructMatches(pattern, uniquify=True)) >= 2:
        record_non_pin_label(name, rule="P-65.7.6.2", detail="a polyanhydride named through one of its links")
