"""A branched ring assembly of identical benzene rings, named by substituting into its longest unbranched assembly.

``## **** BRANCHED RING ASSEMBLIES OF IDENTICAL CYCLIC SYSTEMS`` (the Blue Book): "Preferred IUPAC
names are formed by substituting into the longest unbranched assembly. The names of substituent groups are formed
in accord with the methods described in. If necessary, the criteria for selecting the principal chain are
applied, namely, the longest chain, the maximum number of substituents, the lowest locants for substituents
considered together, then alphanumerical order." The assembly itself is ``### **** DEFINITIONS`` (:15540,
:15542): cyclic systems "directly joined to each other by single or double bonds... when the number of such direct
ring junctions is one less than the number of cyclic systems involved", a tree of rings; ``****`` (:15675)
names three to six benzene rings 'terphenyl'... 'sexiphenyl'; the examples of print the primed names of
general nomenclature next to the PINs ('5'-phenyl-1,1':3',1''-terphenyl', '5''-([1,1'-biphenyl]-3-yl)-1,1':3',1'':
3'',1''':3''',1''''-quinquephenyl', '3',4',5',6'-tetraphenyl-1,1':2',1''-terphenyl'), and ``****`` (:15651)
allows "serially primed locants" in general nomenclature. The PIN spells the junctions with composite locants
('2^5-phenyl-1^1,2^1:2^3,3^1-terphenyl'), which this module does not write; a chain of more than six rings is named
by phane nomenclature in the PIN (``****``,:15705), which ``phane_pin`` builds where OPSIN reads it.

``rules.ring_assemblies.detect_ring_assembly`` takes only an unbranched assembly (every ring system joined to at
most two others). A tree with a ring joined to three or more others is not one of its assemblies, and the namers
that consume its result (the prefix builders of ring fragments) name every ring system of the assembly alike. The
caller that names the whole molecule asks:func:`detect_assembly_or_branched` instead: the result of the first, else
that of:func:`detect_branched_ring_assembly`, which

* takes the tree of identical benzene rings whose other atoms are neutral carbon atoms and halogens (no group that
  could be a principal characteristic group, so that the chain is chosen by the four criteria above alone),
* chooses the chain among the longest paths of the tree by those four criteria, each read as the book reads it: the
  number of rings of the chain, the number of substituents (every bond from a chain ring to an atom outside the chain
  rings), the lowest locants of the junctions and then of the substituents considered together, the lowest locant of
  the substituent cited first in alphanumerical order,
* numbers each ring of the chain from the junction to the ring before it, "Lowest possible locants must be
  used to denote the positions of attachment"; the junction to the previous ring is cited first), the other junction
  of a middle ring at its lowest locant, the walking direction that is left decided by the substituents,
* names the chain and its substituents with ``name_ring_assembly`` (a branch that holds rings is a compound prefix
  named by ``assembly.substituent_enumerator.name_substituent``), and returns the result only when OPSIN 2.9.0 reads
  that name back to the full InChIKey of the molecule, as ``detect_hydro_ring_assembly`` does.

A molecule it does not take (a ring that is not benzene, a heteroatom, a charge, a stereocentre, more candidate
chains than:data:`MAX_CHAIN_CANDIDATES`, more tied candidates than:data:`MAX_TIED_CHAINS`, more branches to name
than:data:`MAX_NAMED_BRANCHES`, a name that is not built or not read back) is declined, and keeps the classification it had.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

__all__ = ["MAX_CHAIN_CANDIDATES", "MAX_NAMED_BRANCHES", "MAX_TIED_CHAINS", "detect_assembly_or_branched", "detect_branched_ring_assembly"]

#: Ring systems beyond which a tree is not examined.
MAX_SYSTEMS = 200
#: Longest paths of the ring tree beyond which the chain is not chosen (declined).
MAX_CHAIN_CANDIDATES = 4000
#: Candidate chains that tie on the numbers of the first three criteria and are not equivalent under a symmetry of the
#: molecule, beyond which the chain is not chosen (declined): each is named to read the fourth.
MAX_TIED_CHAINS = 8
#: Substituents named to read the fourth criterion, beyond which the chain is not chosen (declined): a branch that holds
#: rings is named by the general substituent namer, which takes seconds for a large one.
MAX_NAMED_BRANCHES = 40

_HALOGENS = frozenset({9, 17, 35, 53})
_RING = 6


def detect_assembly_or_branched(mol, ring_systems: List[Set[int]]) -> Optional[Dict]:
    """``detect_ring_assembly`` of ``ring_systems``, else:func:`detect_branched_ring_assembly`. An unbranched assembly
    of three or more benzene rings that carries substituents keeps the result of ``detect_ring_assembly`` when its name
    is built and read back, and gets the numbering of this module otherwise (:func:`_renumbered`)."""
    from .ring_assemblies import detect_ring_assembly
    info = detect_ring_assembly(mol, ring_systems)
    if info is None:
        return detect_branched_ring_assembly(mol, ring_systems)
    return _renumbered(mol, ring_systems, info) or info


def _plain_atoms(mol, ring_atoms: Set[int]) -> bool:
    """One component; every atom neutral, unlabelled and with no radical electron; a ring atom is carbon, any other atom
    is carbon or a halogen; no stereo is specified."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return False
        number = atom.GetAtomicNum()
        if number != 6 and (atom.GetIdx() in ring_atoms or number not in _HALOGENS):
            return False
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return False
    return all(b.GetStereo() == Chem.BondStereo.STEREONONE for b in mol.GetBonds())


def _benzene_rings(mol, ring_systems: List[Set[int]]) -> Optional[Dict[int, List[int]]]:
    """``{system index: its atoms in ring order}`` when every ring system is one benzene ring, else None."""
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
    by_atoms = {frozenset(r): r for r in rings}
    cycles: Dict[int, List[int]] = {}
    for k, system in enumerate(ring_systems):
        ring = by_atoms.get(frozenset(system))
        if ring is None or len(ring) != _RING or len(system) != _RING:
            return None
        if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            return None
        cycles[k] = list(ring)
    return cycles if len(rings) == len(ring_systems) else None


def _ring_tree(mol, ring_systems: List[Set[int]], branched: bool = True):
    """``(adjacency, junction atoms)`` of the tree the ring systems form, every junction a single bond, else None; with
    ``branched`` the tree must have a ring joined to three or more others, without it no ring may be. ``junction[(i, j)]``
    is the atom of ``i`` bonded to ``j``."""
    from .ring_assemblies import _check_all_connected, _find_inter_system_bonds
    n = len(ring_systems)
    connections = _find_inter_system_bonds(mol, ring_systems)
    if len(connections) != n - 1 or not _check_all_connected(n, connections):
        return None
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    junction: Dict[Tuple[int, int], int] = {}
    for a1, a2, s1, s2 in connections:
        if mol.GetBondBetweenAtoms(a1, a2).GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
            return None
        if (s1, s2) in junction:
            return None
        adjacency[s1].append(s2)
        adjacency[s2].append(s1)
        junction[(s1, s2)], junction[(s2, s1)] = a1, a2
    if (max(len(v) for v in adjacency.values()) >= 3) != branched:
        return None
    return adjacency, junction


def _longest_paths(adjacency: Dict[int, List[int]]) -> Optional[List[List[int]]]:
    """Every path of the tree with the most vertices, leaf to leaf (each once, in the order of the lower leaf), or None
    when there are more than:data:`MAX_CHAIN_CANDIDATES`."""
    leaves = sorted(i for i, v in adjacency.items() if len(v) == 1)

    def walk(source):
        parent, dist, queue = {source: None}, {source: 0}, deque([source])
        while queue:
            x = queue.popleft()
            for y in adjacency[x]:
                if y not in parent:
                    parent[y], dist[y] = x, dist[x] + 1
                    queue.append(y)
        return parent, dist

    _, from_first = walk(leaves[0])
    far = max(from_first, key=lambda k: (from_first[k], -k))
    diameter = max(walk(far)[1].values())
    paths: List[List[int]] = []
    for u in leaves:
        parent, dist = walk(u)
        for v in leaves:
            if v > u and dist[v] == diameter:
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                paths.append(path[::-1])
                if len(paths) > MAX_CHAIN_CANDIDATES:
                    return None
    return paths


class _Chains:
    """The candidate chains of one molecule: their numberings and the names of the substituents they carry."""

    def __init__(self, mol, ring_systems, cycles, junction):
        self.mol, self.systems, self.cycles, self.junction = mol, ring_systems, cycles, junction
        self.colour = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        self._names: Dict[Tuple[int, int], Optional[str]] = {}

    def branch(self, ring_atom: int, first: int) -> List[int]:
        """The atoms beyond the bond ``ring_atom``-``first``, ``first`` included."""
        seen, stack = set(), [first]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(n.GetIdx() for n in self.mol.GetAtomWithIdx(x).GetNeighbors()
                         if n.GetIdx() != ring_atom and n.GetIdx() not in seen)
        return sorted(seen)

    def name(self, ring_atom: int, first: int) -> Optional[str]:
        """The prefix of the substituent that hangs from ``ring_atom`` by its atom ``first``."""
        key = (ring_atom, first)
        if key not in self._names:
            if len(self._names) >= MAX_NAMED_BRANCHES:
                raise ValueError("too many branches to name")
            from .ring_assemblies import _name_substituent
            self._names[key] = _name_substituent(self.mol, self.branch(ring_atom, first), first)
        return self._names[key]

    def substituents(self, ring: int, chain_atoms: Set[int]) -> List[Tuple[int, int]]:
        """``(ring atom, first atom of the substituent)`` of the substituents of system ``ring``."""
        found = []
        for a in self.cycles[ring]:
            for n in self.mol.GetAtomWithIdx(a).GetNeighbors():
                if n.GetIdx() not in chain_atoms:
                    found.append((a, n.GetIdx()))
        return found

    def _ring_numbering(self, order: List[int], k: int, chain_atoms: Set[int]):
        """The numbering of the ring at position ``k`` of the chain ``order`` (atom -> locant, the junction to the
        previous ring 1), with the junction to the next ring, and the substituents, in their lowest locants."""
        ring = order[k]
        previous = self.junction[(ring, order[k - 1])] if k else None
        following = self.junction[(ring, order[k + 1])] if k + 1 < len(order) else None
        anchor = previous if previous is not None else following
        cycle = self.cycles[ring]
        start = cycle.index(anchor)
        options = [{cycle[(start + d * i) % _RING]: i + 1 for i in range(_RING)} for d in (1, -1)]
        if previous is not None and following is not None:      # the other junction at its lowest locant
            lowest = min(o[following] for o in options)
            options = [o for o in options if o[following] == lowest]
        subs = self.substituents(ring, chain_atoms)
        if len(options) > 1 and subs:
            options = self._lowest_substituents(options, subs)
        return options[0], subs

    def _lowest_substituents(self, options, subs):
        """The numberings left by the junctions, reduced to those giving the substituents the lowest locants
        together, then the lowest locant to the substituent cited first in alphanumerical order."""
        from ..assembly.naming_utils import alpha_sort_key
        keyed = [(sorted(o[a] for a, _ in subs), o) for o in options]
        lowest = min(k for k, _ in keyed)
        tied = [o for k, o in keyed if k == lowest]
        if len(tied) > 1:
            def alphabetical(o):
                return [o[a] for a, first in sorted(subs, key=lambda s: (alpha_sort_key(self.name(*s) or ""), o[s[0]]))]
            tied.sort(key=alphabetical)
        return tied

    def score(self, order: List[int]):
        """``(key, numbering, substituents)`` of the chain ``order`` read from its first ring, or None. ``key`` is
        what the first three criteria compare: fewer substituents last, the junction locants, the substituent locants."""
        chain_atoms = set().union(*(self.systems[i] for i in order))
        numbering, junctions, locants, substituents = [], [], [], []
        for k in range(len(order)):
            number, subs = self._ring_numbering(order, k, chain_atoms)
            numbering.append(number)
            if 0 < k < len(order) - 1:
                junctions.append(number[self.junction[(order[k], order[k + 1])]])
            locants.extend((number[a], k) for a, _ in subs)
            substituents.append(subs)
        return (-len(locants), tuple(junctions), tuple(sorted(locants))), numbering, substituents

    def colours(self, numbering) -> tuple:
        """The colour (symmetry class) of every atom of every ring of the chain in the order of its locants: two chains
        with the same sequence are one chain under a symmetry of the molecule."""
        return tuple(tuple(self.colour[a] for a in sorted(num, key=num.get)) for num in numbering)


def _choose_chain(chains: _Chains, paths: List[List[int]]):
    """The chain the four criteria of choose, as ``(order, numbering)``, or None."""
    from ..assembly.naming_utils import alpha_sort_key
    scored = []
    for path in paths:
        for order in (path, path[::-1]):
            scored.append((chains.score(order), order))
    best = min(item[0][0] for item in scored)
    distinct: Dict[tuple, tuple] = {}
    for (key, numbering, subs), order in scored:
        if key == best:
            distinct.setdefault(chains.colours(numbering), (order, numbering, subs))
    if len(distinct) > MAX_TIED_CHAINS:
        return None

    def alphabetical(item):
        _, numbering, subs = item
        named = [(numbering[k][a], k, chains.name(a, first)) for k, ring in enumerate(subs) for a, first in ring]
        if any(name is None for _, _, name in named):
            return None
        return [(loc, k) for loc, k, name in sorted(named, key=lambda t: (alpha_sort_key(t[2]), t[0], t[1]))]

    items = sorted(distinct.items())
    if len(items) > 1:
        keyed = [(alphabetical(v), c, v) for c, v in items]
        if any(a is None for a, _, _ in keyed):
            return None
        keyed.sort(key=lambda t: (t[0], t[1]))
        items = [(c, v) for _, c, v in keyed]
    order, numbering, _ = items[0][1]
    return order, numbering


def _certified_info(mol, ring_systems, cycles, adjacency, junction) -> Optional[Dict]:
    """The assembly info of the chain of the tree ``adjacency`` that the criteria of choose, with the numbering of its
    rings and the name of the molecule, when that name is built and OPSIN 2.9.0 reads it back to the molecule; else None."""
    paths = _longest_paths(adjacency)
    if not paths:
        return None
    try:
        chosen = _choose_chain(_Chains(mol, ring_systems, cycles, junction), paths)
    except Exception:  # noqa: BLE001 - a molecule this reader cannot number is declined
        return None
    if chosen is None:
        return None
    order, numbering = chosen
    info = {
        'ring_systems': [set(ring_systems[i]) for i in order],
        'connections': [(junction[(order[k], order[k + 1])], junction[(order[k + 1], order[k])], k, k + 1)
                        for k in range(len(order) - 1)],
        'count': len(order),
        'ring_type': 'carbocyclic',
        'double_bond_junction': False,
        'mancude_tautomer': False,
        'branched': True,
        'numbering': numbering,
    }
    from .ring_assemblies import _hydro_certified, name_ring_assembly
    try:
        name = name_ring_assembly(mol, info, None)
    except Exception:  # noqa: BLE001 - a name that cannot be built is no offer
        return None
    if not name or not _hydro_certified(mol, name):
        return None
    info['name'] = name
    return info


def _benzene_tree(mol, ring_systems: List[Set[int]], branched: bool):
    """``(cycles, adjacency, junction)`` when the ring systems of ``mol`` are benzene rings that form a tree of the
    kind asked for and the atoms are plain, else None."""
    n = len(ring_systems)
    if n > MAX_SYSTEMS or not _plain_atoms(mol, set().union(*ring_systems)):
        return None
    cycles = _benzene_rings(mol, ring_systems)
    tree = _ring_tree(mol, ring_systems, branched) if cycles is not None else None
    return None if tree is None else (cycles, *tree)


def _renumbered(mol, ring_systems: List[Set[int]], info: Dict) -> Optional[Dict]:
    """For an unbranched assembly of three or more benzene rings with substituents, whose name ``name_ring_assembly``
    does not build or OPSIN does not read back (its locants of the substituents of a middle ring are read from a numbering
    other than the one of the junctions), the info of the same chain numbered by this module, with its name; else None."""
    if info.get('count', 0) < 3 or info.get('replacement') or info.get('hydro') or info.get('mancude_tautomer'):
        return None
    if info.get('double_bond_junction') or sum(len(s) for s in ring_systems) == mol.GetNumAtoms():
        return None
    parts = _benzene_tree(mol, ring_systems, False)
    if parts is None:
        return None
    from .ring_assemblies import _hydro_certified, name_ring_assembly
    try:
        name = name_ring_assembly(mol, info, None)
    except Exception:  # noqa: BLE001 - a name that cannot be built is no name
        name = None
    if name and _hydro_certified(mol, name):
        return None
    return _certified_info(mol, ring_systems, *parts)


def detect_branched_ring_assembly(mol, ring_systems: List[Set[int]]) -> Optional[Dict]:
    """The assembly info of the longest unbranched assembly of a branched ring assembly of identical benzene rings
    (see the module docstring), with the numbering of its rings and the name of the molecule, or None.

    The keys are those of ``detect_ring_assembly`` for the chain, with ``branched`` True, ``numbering`` (one dict
    atom -> locant per ring of the chain) and ``name``: ``name_ring_assembly`` returns that name."""
    if len(ring_systems) < 4:
        return None
    parts = _benzene_tree(mol, ring_systems, True)
    return None if parts is None else _certified_info(mol, ring_systems, *parts)
