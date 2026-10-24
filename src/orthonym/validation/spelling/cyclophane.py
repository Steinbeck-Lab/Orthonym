""" (1): the structures that are cyclophanes, read from the structure alone.

The structure test behind the cyclic phane check and the cyclic phane producers. RDKit only; no
engine naming code and no OPSIN, so that a check built on it is independent of the names it
reads. Rules (the Blue Book, each line read with ``sed -n``):

* "Preferred IUPAC names in phane nomenclature" (:23824); (:23826), item (1)
  (:23828): "cyclophanes are cyclic phane structures containing one or more rings or ring
  systems, at least one ring or ring system of which must be a mancude system attached to
  adjacent atoms or chains at nonadjacent ring positions".
* (:23831): "When the conditions given in for cyclic phane systems are not
  fulfilled, names of fused ring systems, bridged fused systems, or von Baeyer systems are
  preferred IUPAC names."
* "Mancude systems attached to adjacent atoms of an alicyclic large ring." (:23833):
  such systems are fused or bridged fused, "A cyclophane name is not allowed" (the dodecahydro
  benzo[14]annulene example).

Reading of the rule: the rings of up to eight atoms that share an atom are the rings and ring
systems of the skeleton (a larger ring is the macrocycle, not a component); each such system is
one node, every other atom is one node. A cyclophane has a cycle in that node graph, and on the
cycle a mancude system is entered and left at two atoms that are not bonded to each other.
Mancude "Indicated hydrogen",:3557: "a mancude ring or ring system, i.e., one that
contains the maximum number of noncumulative double bonds") is read on the system alone: its
hydride (every bond to an outside atom replaced by hydrogen) has every atom aromatic, or is a
monocycle of up to eight atoms with the maximum number of noncumulative double bonds. RDKit leaves
the furans of a conjugated macrocycle non-aromatic; the isolated ring is read instead. The
porphyrin core is a retained parent (:51723 "porphyrin (fundamental parent structure)"), and the
tetrapyrrole macrocycles of its class (the reduced ones, the corrole, and those with nitrogen at the
meso positions: porphyrazine, phthalocyanine) are read as that class, ASSUMED: the book names none of
the last three. A structure with such a macrocycle and a separate cyclophane is judged on the cyclophane.

The book states no size limit for a cyclophane (1),:23828;,:14800-:14801 "atoms and/or
saturated or unsaturated chains as alternate components of a large ring"). The test judges a structure only
where the book prints a phane PIN for it: its macrocycle has at least ten atoms (the smallest macrocycle of
the book's cyclophane PIN rows is the ten-atom ring of the [2.2]metacyclophane name,:15021). Every other
structure is unjudged, so a check built on the test abstains on it: a macrocycle of nine atoms or fewer
([5]metacyclophane, [6]metacyclophane, [1.1]metacyclophane, which the rings of up to eight atoms swallow or
which fall under the minimum), a mancude system that needs indicated hydrogen (1H-indene, 9H-fluorene) or is
partly hydrogenated (a tetralin), a ring system that RDKit cannot read as a hydride, and the tetrapyrrole
class. A structure of these kinds keeps whatever name and label the engine gives it. A macrocycle of ten atoms
or more closed across one polycyclic system by a chain is judged like any other: (:15033) prints
'1(3,10)-fluoranthenacyclononaphane (I) (PIN)' (:15039), a phane PIN with one polycyclic amplificant and no other ring.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from .linear_phane import ring_system_hydride

#: a ring of at most this many atoms belongs to a ring system of the skeleton; a larger one is the macrocycle
_SMALL_RING = 8

#: the smallest macrocycle of the book's cyclophane PIN rows (:15021); a structure whose macrocycles are smaller is unjudged
_MIN_MACROCYCLE = 10

#: O, S, Se and Te with bonding number two carry no ring double bond
_DIVALENT = frozenset({"O", "S", "Se", "Te"})

#: the tetrapyrrole macrocycle: four pyrroles (nitrogen, two alpha and two beta atoms) joined at their alpha
#: atoms by meso atoms that are carbon or nitrogen (porphyrin, chlorin, porphyrazine, phthalocyanine), or by
#: three meso atoms and one direct bond (corrole); bonds of any order, so a metal complex and a reduced
#: ring read alike
_MESO = "[#6,#7]"
_PYRROLE = "[#6]2~[#6]~[#6]~[#6](~[#7]~2)"
_TETRAPYRROLE_MACROCYCLES = tuple(Chem.MolFromSmarts(text) for text in (
    f"{_MESO}1~{_PYRROLE}~{_MESO}~{_PYRROLE}~{_MESO}~{_PYRROLE}~{_MESO}~[#6]2~[#6]~[#6]~[#6]~1~[#7]~2",
    f"{_MESO}1~{_PYRROLE}~{_MESO}~{_PYRROLE}~{_MESO}~[#6]2~[#6]~[#6]~[#6](~[#7]~2)~[#6]2~[#6]~[#6]~[#6]~1~[#7]~2",
))


def _merge(rings: Sequence[Set[int]], min_shared: int) -> List[FrozenSet[int]]:
    """The atoms of each group of rings joined (transitively) by ``min_shared`` shared atoms or more."""
    systems: List[Set[int]] = []
    for ring in rings:
        merged = [s for s in systems if len(s & ring) >= min_shared]
        grown = set(ring).union(*merged)
        systems = [s for s in systems if not any(s is m for m in merged)] + [grown]
    return sorted((frozenset(s) for s in systems), key=min)


def _cycle_blocks(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    """Edge index -> number of its block, for every edge of a multigraph that lies on a cycle. The blocks are
    the biconnected components: two cycles that share one node only are two blocks. A bridge is in no block;
    a loop is a block of its own and so is each pair of parallel edges."""
    blocks: Dict[int, int] = {}
    count = 0
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for i, (u, v) in enumerate(edges):
        if u == v:
            blocks[i] = count
            count += 1
        else:
            adj[u].append((v, i))
            adj[v].append((u, i))
    disc = [-1] * n_nodes
    low = [0] * n_nodes
    clock = 0
    pending: List[int] = []
    for root in range(n_nodes):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = clock
        clock += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, via, it = stack[-1]
            for v, i in it:
                if i == via:
                    continue
                if disc[v] == -1:
                    pending.append(i)
                    disc[v] = low[v] = clock
                    clock += 1
                    stack.append((v, i, iter(adj[v])))
                    break
                if disc[v] < disc[u]:
                    pending.append(i)
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[u])
                    if low[u] >= disc[parent]:
                        block = []
                        while True:
                            block.append(pending.pop())
                            if block[-1] == via:
                                break
                        if len(block) > 1:
                            blocks.update({e: count for e in block})
                            count += 1
    return blocks


def _max_ring_double_bonds(mol: Chem.Mol, ring: Sequence[int]) -> int:
    """The maximum number of noncumulative double bonds of the monocycle ``ring`` (atoms in cycle order):
    the largest set of ring bonds no two of which share an atom, a bond counting only when both its
    atoms can carry a ring double bond. An O, S, Se or Te atom of bonding number two cannot."""
    n = len(ring)
    can = []
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        can.append(not (atom.GetSymbol() in _DIVALENT and atom.GetTotalValence() == 2))
    positions = [p for p in range(n) if can[p] and can[(p + 1) % n]]
    best = 0
    for mask in range(1 << len(positions)):
        used: Set[int] = set()
        size = 0
        for k, p in enumerate(positions):
            if not mask >> k & 1:
                continue
            a, b = p, (p + 1) % n
            if a in used or b in used:
                size = -1
                break
            used.update((a, b))
            size += 1
        best = max(best, size)
    return best


def is_mancude_system(mol: Chem.Mol, atoms: FrozenSet[int]) -> bool:
    """ (1) 'a mancude system': the ring system ``atoms`` on its own, as its hydride."""
    got = ring_system_hydride(mol, atoms)
    if got is None:
        return False
    hydride = got[0]
    if all(a.GetIsAromatic() for a in hydride.GetAtoms()):
        return True
    rings = hydride.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) > _SMALL_RING or len(rings[0]) != hydride.GetNumAtoms():
        return False
    doubles = sum(1 for b in hydride.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    return doubles == _max_ring_double_bonds(hydride, list(rings[0]))


def small_ring_systems(mol: Chem.Mol) -> List[FrozenSet[int]]:
    """The ring systems that are nodes of a phane skeleton: the rings of up to eight atoms, joined when
    they share an atom. A larger ring is the macrocycle, not a component."""
    return _merge([set(r) for r in mol.GetRingInfo().AtomRings() if len(r) <= _SMALL_RING], 1)


def _skeleton_graph(mol: Chem.Mol, systems: Sequence[FrozenSet[int]]):
    """Nodes (a ring system, or an atom outside the systems) and the edges between them; ``edges[i]`` is
    ``(node, node, atom, atom)``. A bond inside a system that closes no small ring is a loop."""
    node: Dict[int, int] = {}
    for k, system in enumerate(systems):
        for a in system:
            node[a] = k
    n_sys = len(systems)
    small_ring_bonds: Set[int] = set()
    for ring in mol.GetRingInfo().BondRings():
        if len(ring) <= _SMALL_RING:
            small_ring_bonds.update(ring)
    edges = []
    for bond in mol.GetBonds():
        a, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        u, v = node.get(a, n_sys + a), node.get(c, n_sys + c)
        if u == v and bond.GetIdx() in small_ring_bonds:
            continue
        edges.append((u, v, a, c))
    return n_sys, edges


def _holding_system(ring_systems: Sequence[FrozenSet[int]], atom: int) -> Optional[FrozenSet[int]]:
    for system in ring_systems:
        if atom in system:
            return system
    return None


def p52_cyclophane_systems(mol: Optional[Chem.Mol]) -> List[FrozenSet[int]]:
    """The atoms of each ring system that holds a cyclophane skeleton (1)), ordered by lowest
    atom index. A ring system is a group of rings that share two atoms or more (the macrocycle shares
    the atoms between the attachment atoms of a ring with it), so the whole skeleton of [2.2]paracyclophane
    is one system; rings that share one atom (spiro) are kept apart. A system that holds a tetrapyrrole
    macrocycle is left out, and so is one whose macrocycles have fewer than ten atoms (see the module text:
    unjudged). Empty when ``mol`` is no cyclophane or is unjudged, and for a molecule without ring."""
    if mol is None or mol.GetRingInfo().NumRings() < 2:
        return []
    systems = small_ring_systems(mol)
    if not systems:
        return []
    n_sys, edges = _skeleton_graph(mol, systems)
    if not edges:
        return []
    blocks = _cycle_blocks(n_sys + mol.GetNumAtoms(), [(u, v) for u, v, _, _ in edges])
    if not blocks:
        return []
    # the attachment atoms of a node are compared within one cycle: two ortho annulations on different
    # macrocycles do not make a ring attached at nonadjacent positions,:23833)
    attached: Dict[Tuple[int, int], Set[int]] = {}
    for i, block in blocks.items():
        u, v, a, c = edges[i]
        if u < n_sys:
            attached.setdefault((u, block), set()).add(a)
        if v < n_sys:
            attached.setdefault((v, block), set()).add(c)
    holders: Set[FrozenSet[int]] = set()
    atom_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    ring_systems = _merge(atom_rings, 2)
    mancude: Dict[int, bool] = {}
    for (k, _), atoms in sorted(attached.items()):
        spread = sorted(atoms)
        if not any(mol.GetBondBetweenAtoms(x, y) is None
                   for i, x in enumerate(spread) for y in spread[i + 1:]):
            continue
        if k not in mancude:
            mancude[k] = is_mancude_system(mol, systems[k])
        if not mancude[k]:
            continue
        holder = _holding_system(ring_systems, spread[0])
        if holder is not None and any(len(r) >= _MIN_MACROCYCLE and r <= holder for r in atom_rings):
            holders.add(holder)
    if holders:
        core = _tetrapyrrole_atoms(mol)
        holders = {h for h in holders if not h & core}
    return sorted(holders, key=min)


def _tetrapyrrole_atoms(mol: Chem.Mol) -> Set[int]:
    """The atoms of every tetrapyrrole macrocycle of ``mol``."""
    core: Set[int] = set()
    for pattern in _TETRAPYRROLE_MACROCYCLES:
        for match in mol.GetSubstructMatches(pattern):
            core.update(match)
    return core


_PHANE_NAME = re.compile(r"(?<!phos)phan(?:e|-)")


def is_phane_name(name: Optional[str]) -> bool:
    """True when ``name`` has a phane parent hydride, cyclic or linear (``cyclohexaphane``,
    ``heptaphan-7^4-yl``); ``phosphane`` and its substituted forms do not count. Whether it is the right
    phane name is for the round trip and the read-back to say."""
    return bool(name) and _PHANE_NAME.search(name) is not None


def is_p52_cyclophane(mol: Optional[Chem.Mol]) -> bool:
    """True when ``mol`` has a cyclophane skeleton in the sense of (1)."""
    return bool(p52_cyclophane_systems(mol))
