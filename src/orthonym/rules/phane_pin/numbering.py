"""Numbering of a simplified phane skeleton and of its amplificants.

The key compares two numberings term by term, lower first,:3189). Each part is a rule of the
Blue Book (the Blue Book, each line read with ``sed -n``):

* K1 (:14965) "the numbering that gives the lowest set of locants for the superatoms is
  selected";
* K2 (:14989) "the lower available superatom locant is assigned to an amplificant that
  appears earlier in the seniority of rings and ring systems (see ";
* K3 (:15043), (:15091), (:19707) "The senior cyclic phane
  system has the lower set of attachment locants when compared term by term in their order of
  citation in the name";
* K4 (:14953), (:15003), (:15033): the same locants in the order in
  which they are cited, the first cited being adjacent to the lower skeleton locant of its
  neighbours;
* K5 (:15232): the set of locants of the skeleton heteroatoms, whatever the element;
* K6 (:15246) "Lowest locants are assigned to heteroatoms considered in the order of their
  seniority: O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl";
* K6b (b) (:3246): the locants of the indicated hydrogen of the amplificants, before the principal groups;
* K7 (c) (:3256): the locants of the principal characteristic groups (heteroatoms of the parent
  come first,:3225: "heteroatoms in chains are now considered to be a part of the parent hydride; as
  such, they have seniority over suffixes for numbering");
* K8 (e)(i) (:3289): the hydro prefixes of the amplificants and the 'ene' and 'yne' endings as one set, by
  their primary locants first, then by the composite locants;
  (e)(ii) (:3290) and (:16489): "Locants as low as possible are given to multiple
  bonds as a set, even though this may at times give 'yne' endings lower locants than 'ene' endings. If
  a choice remains, preference for low locants is given to the double bonds";
* K9 (:15115): the locants of the detachable prefixes as one set, first by their primary
  locants and, if these are identical, by the complete composite locants (f),:3301);
* K10 (g) (:3307) "lowest locants for the substituent cited first as a prefix in the name";
* K11 (j) (:3346): the CIP descriptors Z, R before E, S at the lower locants (``stereo.rank_key``), those of the
  skeleton and those inside the prefixes, each at the locant of the atom its prefix is attached to;
  (h) and (i), the valence states and the isotopes, take no part in a name built here;
* K12 the book states no rule for the prefixes cited after the first, so the comparison of K10 goes on in
  the order of citation (ASSUMED reading: it is what leaves one name for one structure). It stands after
  K11: a decision that (j) takes is never taken by the citation order of the later prefixes.

Numbering of an amplificant is that of its hydride,:14967): of the valid numberings
(``AmplificantHydride.numberings``) the ones that give its indicated hydrogen the lowest locants stay (the indicated
hydrogen belongs to the parent name, ``indicated_numberings``), then the one with the lowest attachment locants is
taken; among those, the one that gives the suffix groups and then the prefixes the lowest composite locants.

A locant is ``(primary, number, letter)``: ``(4, 0, "")`` for a skeletal atom or a superatom, ``(1, 4, "")``
for atom 4 of the amplificant at superatom 1 (written ``1^4``,,:15111).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from rdkit import Chem

from .amplificants import HETERO_ORDER, hydro_split, indicated_numberings
from .skeleton import Amplificant, Node, PhaneSkeleton, SkeletonKind, bridges_between, spiro_rings
from .stereo import prefix_codes, stereo_atoms, units_of
from .stereo import rank_key as stereo_rank_key
from .substituents import Plan

_Locant = Tuple[int, str]
LocantKey = Tuple[int, int, str]


@dataclass(frozen=True)
class Placed:
    """An amplificant at skeleton locant ``locant``, its attachment locants as they are cited, and every
    numbering of its hydride that gives them (they differ only where a later part of the key tells)."""

    locant: int
    amp: Amplificant
    cited: Tuple[_Locant, ...]
    tied: Tuple[Mapping[int, _Locant], ...]


@dataclass(frozen=True)
class Numbered:
    pos: Dict[Node, int]
    placed: Tuple[Placed, ...]
    key: tuple
    chosen: Dict[int, Mapping[int, _Locant]]     # amplificant k -> the numbering of its atoms that the name uses
    enes: Tuple[int, ...] = ()
    ynes: Tuple[int, ...] = ()
    wraps: bool = False                          # a multiple bond joins the last node to the first


def render_locant(locant: LocantKey) -> str:
    primary, number, letter = locant
    return str(primary) if number == 0 else f"{primary}^{number}{letter}"


def skeleton_numberings(skel: PhaneSkeleton) -> Iterator[Dict[Node, int]]:
    """Every numbering of the skeleton: for a cycle each node as the first and both directions
     leaves them all), for a path the two directions."""
    nodes, n = skel.nodes, len(skel.nodes)
    if skel.kind is SkeletonKind.PATH:
        yield {node: k + 1 for k, node in enumerate(nodes)}
        yield {node: n - k for k, node in enumerate(nodes)}
        return
    if skel.kind is SkeletonKind.BICYCLO:
        yield from _von_baeyer_numberings(skel)
        return
    if skel.kind is SkeletonKind.SPIRO:
        yield from _spiro_numberings(skel)
        return
    for start in range(n):
        for step in (1, -1):
            yield {nodes[(start + step * k) % n]: k + 1 for k in range(n)}


def _von_baeyer_numberings(skel: PhaneSkeleton) -> Iterator[Dict[Node, int]]:
    """ (:9589) "The bicyclic ring system is numbered starting with one of the bridgeheads and proceeding
    first along the longer segment of the main ring to the second bridgehead, then back to the first bridgehead
    along the unnumbered segment of the main ring. Numbering is completed by numbering the main bridge beginning
    with the atom next to the first bridgehead." The main ring holds the two longest bridges; bridges of equal
    length may change places."""
    adj = {node: list(nbs) for node, nbs in skel.adj.items()}
    heads = sorted(n for n in skel.nodes if len(adj[n]) == 3)
    for start, other in ((heads[0], heads[1]), (heads[1], heads[0])):
        bridges = bridges_between(adj, start, other)
        for first, second, third in itertools.permutations(bridges):
            if not len(first) >= len(second) >= len(third):
                continue
            order = [start] + first + [other] + second[::-1] + third
            yield {node: k + 1 for k, node in enumerate(order)}


def _spiro_numberings(skel: PhaneSkeleton) -> Iterator[Dict[Node, int]]:
    """ (:9961) "Numbering starts in the smaller ring": at an atom next to the spiro atom, round that ring,
    through the spiro atom, and round the second ring."""
    adj = {node: list(nbs) for node, nbs in skel.adj.items()}
    (spiro,) = [n for n in skel.nodes if len(adj[n]) == 4]
    one, two = spiro_rings(adj, spiro)
    for first, second in ((one, two), (two, one)):
        if len(first) > len(second):
            continue
        for a in (first, first[::-1]):
            for b in (second, second[::-1]):
                order = a + [spiro] + b
                yield {node: k + 1 for k, node in enumerate(order)}


def best_attachment(skel: PhaneSkeleton, amp: Amplificant, neighbours: List[Node]):
    """,,: the numbering of the amplificant with the lowest attachment set, among the
    numberings its parent name allows (the lowest indicated hydrogen first, ``indicated_numberings``), cited in the order
    of the skeleton locants of its neighbours; the tied numberings are kept."""
    atoms = [skel.edge[(("amp", amp.k), nb)][0] for nb in neighbours]
    best = None
    for numbering in indicated_numberings(amp.hydride):
        cited = tuple(numbering[a] for a in atoms)
        key = (tuple(sorted(cited)), cited)
        if best is None or key < best[0]:
            best = (key, cited, [numbering])
        elif key == best[0]:
            best[2].append(numbering)
    return best


def skeleton_heteroatoms(skel: PhaneSkeleton, pos: Dict[Node, int]) -> Dict[int, str]:
    """Skeleton locant -> element symbol of every acyclic skeletal heteroatom."""
    found = {}
    for node, locant in pos.items():
        if node[0] == "atom":
            symbol = skel.mol.GetAtomWithIdx(node[1]).GetSymbol()
            if symbol != "C":
                found[locant] = symbol
    return found


def _heteroatom_key(skel: PhaneSkeleton, pos: Dict[Node, int]) -> tuple:
    """K5 and K6."""
    hetero = skeleton_heteroatoms(skel, pos)
    k5 = tuple(sorted(hetero))
    k6 = tuple(tuple(sorted(loc for loc, sym in hetero.items() if sym == e)) for e in HETERO_ORDER)
    return k5, k6


def multiple_bonds(skel: PhaneSkeleton, pos: Dict[Node, int]) -> Tuple[List[int], List[int], bool]:
    """Locants of the double and of the triple bonds between acyclic skeletal atoms (the lower locant of the
    bond) and whether one of them joins the last node to the first, which cites as ``1(n)``."""
    enes: List[int] = []
    ynes: List[int] = []
    wraps = False
    n = len(skel.nodes)
    for (u, v), (_, _, bond_type) in skel.edge.items():
        if pos[u] >= pos[v] or bond_type == Chem.BondType.SINGLE:
            continue
        if pos[v] - pos[u] != 1:
            wraps = wraps or (pos[u] == 1 and pos[v] == n)
        (enes if bond_type == Chem.BondType.DOUBLE else ynes).append(pos[u])
    return sorted(enes), sorted(ynes), wraps


#: the most combinations of tied amplificant numberings that are compared; beyond it the name is not built
_MAX_TIED_COMBINATIONS = 4096


class UndecidableTiesError(Exception):
    """Too many ways to give the amplificants their locants for one name to be chosen by comparison."""


def _options(plan: Optional[Plan], placed: Placed, units=()) -> List[Mapping[int, _Locant]]:
    """The numberings of an amplificant that tie on everything the amplificant alone decides: the lowest
    composite locants for its suffix groups, then for its prefixes (K7, K9: the primary locants are the
    same for every one of them), one per distinct way of giving those groups and its stereogenic atoms
    their locants. Which of them the name uses is decided by K10 to K12, over the whole skeleton."""
    if plan is None or len(placed.tied) == 1:
        return [placed.tied[0]]
    suffix_atoms = [s.atom for s in plan.suffixes if s.atom in placed.amp.atoms]
    prefix_atoms = [p.atom for p in plan.prefixes if p.atom in placed.amp.atoms]
    unit_atoms = [a for a in stereo_atoms(units) if a in placed.amp.atoms]
    if not suffix_atoms and not prefix_atoms and not unit_atoms and not placed.amp.hydride.saturated:
        return [placed.tied[0]]

    def sets(t):
        indicated, hydro = hydro_split(placed.amp.hydride, t)
        return (sorted(t[a] for a in indicated), sorted(t[a] for a in suffix_atoms),
                sorted(t[a] for a in hydro), sorted(t[a] for a in prefix_atoms))
    best = min(sets(t) for t in placed.tied)
    atoms = sorted(set(suffix_atoms) | set(prefix_atoms) | set(unit_atoms))
    distinct: Dict[tuple, Mapping[int, _Locant]] = {}
    for t in placed.tied:
        if sets(t) == best:
            distinct.setdefault(tuple(t[a] for a in atoms), t)
    return list(distinct.values())


def atom_locant(skel: PhaneSkeleton, pos: Dict[Node, int], chosen: Mapping[int, Mapping[int, _Locant]],
                atom: int) -> LocantKey:
    node = skel.node_of(atom)
    if node[0] == "atom":
        return (pos[node], 0, "")
    number, letter = chosen[node[1]][atom]
    return (pos[node], number, letter)


def citation_key(name: str) -> tuple:
    """The order in which prefix names are cited: alphanumerical, descriptors ignored); names that differ only in
    their descriptors by the CIP rank of the descriptors, Z before E and R before S (j): the prefix cited first takes
    the lower locant, ``cip_descriptor_rank_key``); then the string."""
    from ...assembly.general_engine import _alpha_key
    from ...assembly.naming_utils import cip_descriptor_rank_key
    return (_alpha_key(name), cip_descriptor_rank_key(name), name)


def _prefix_sequence(skel: PhaneSkeleton, pos: Dict[Node, int], chosen: Mapping[int, Mapping[int, _Locant]],
                     plan: Optional[Plan]) -> tuple:
    """K10: for each prefix name in the order of citation, the locants of its atoms."""
    if plan is None or not plan.prefixes:
        return ()
    groups: Dict[str, list] = {}
    for p in plan.prefixes:
        groups.setdefault(p.name, []).append(atom_locant(skel, pos, chosen, p.atom))
    return tuple(tuple(sorted(groups[name])) for name in sorted(groups, key=citation_key))


def _prefix_descriptors(skel: PhaneSkeleton, pos: Dict[Node, int], chosen: Mapping[int, Mapping[int, _Locant]],
                        plan: Optional[Plan]) -> list:
    """K11 for the prefixes: ``(locant, code)`` of every descriptor inside a prefix, at the locant of its attachment atom."""
    if plan is None:
        return []
    return [(atom_locant(skel, pos, chosen, p.atom), code) for p in plan.prefixes for code in prefix_codes(p.name)]


def numbering_key(skel: PhaneSkeleton, pos: Dict[Node, int], plan: Optional[Plan] = None) -> Numbered:
    placed = []
    for node in sorted(pos, key=pos.get):
        if node[0] == "amp":
            amp = skel.amplificants[node[1]]
            neighbours = sorted(skel.adj[node], key=pos.get)
            key, cited, tied = best_attachment(skel, amp, neighbours)
            placed.append((Placed(pos[node], amp, cited, tuple(tied)), key))
    scores = [p.amp.hydride.score for p, _ in placed]
    classes = sorted(set(scores))
    k1 = tuple(p.locant for p, _ in placed)
    k2 = tuple(scores)
    k3 = tuple(tuple(key[0] for p, key in placed if p.amp.hydride.score == c) for c in classes)
    k4 = tuple(tuple(p.cited for p, _ in placed if p.amp.hydride.score == c) for c in classes)
    units = (units_of(skel) or ()) if plan is not None else ()
    options = [_options(plan, p, units) for p, _ in placed]
    combinations = 1
    for o in options:
        combinations *= len(o)
    if combinations > _MAX_TIED_COMBINATIONS:
        raise UndecidableTiesError(combinations)
    best, chosen = None, None
    for combo in itertools.product(*options):
        candidate = {p.amp.k: numbering for (p, _), numbering in zip(placed, combo)}
        sequence = _prefix_sequence(skel, pos, candidate, plan)
        ranking = (sequence[:1], stereo_rank_key(skel, pos, candidate, units, _prefix_descriptors(skel, pos, candidate, plan)),
                   sequence[1:])
        if best is None or ranking < best:
            best, chosen = ranking, candidate
    enes, ynes, wraps = multiple_bonds(skel, pos)
    suffixes = tuple(sorted(atom_locant(skel, pos, chosen, s.atom) for s in plan.suffixes)) if plan else ()
    prefixes = sorted(atom_locant(skel, pos, chosen, p.atom) for p in plan.prefixes) if plan else []
    indicated, hydro = [], []
    for p, _ in placed:
        ind, hyd = hydro_split(p.amp.hydride, chosen[p.amp.k])
        indicated += [atom_locant(skel, pos, chosen, a) for a in ind]
        hydro += [atom_locant(skel, pos, chosen, a) for a in hyd]
    k6b = tuple(sorted(indicated))
    k7 = suffixes
    multiple = [(n, 0, "") for n in enes + ynes]
    together = hydro + multiple
    # as for the prefixes (K9): the primary locants of the set first, then the composite locants (the example of
    #:49696 numbers the pyridine ring so that the 'ene' has locant 2 and not 11, though the hydro prefixes of the
    # other numbering have lower superscripts)
    k8 = ((tuple(sorted(loc[0] for loc in together)), tuple(sorted(together))), tuple(sorted(multiple)),
          tuple((n, 0, "") for n in enes))
    k9 = (tuple(loc[0] for loc in prefixes), tuple(prefixes))
    key = (k1, k2, k3, k4) + _heteroatom_key(skel, pos) + (k6b, k7, k8, k9) + best
    return Numbered(pos, tuple(p for p, _ in placed), key, chosen, tuple(enes), tuple(ynes), wraps)


def best_numberings(skel: PhaneSkeleton, plan: Optional[Plan] = None) -> List[Numbered]:
    """Every numbering whose key is the lowest (a tie is decided by the name string, in ``naming``)."""
    from ...assembly.fragment_naming import spend_perf_work
    best, found = None, []
    for pos in skeleton_numberings(skel):
        spend_perf_work(1)
        numbered = numbering_key(skel, pos, plan)
        if best is None or numbered.key < best:
            best, found = numbered.key, []
        if numbered.key == best:
            found.append(numbered)
    return found
