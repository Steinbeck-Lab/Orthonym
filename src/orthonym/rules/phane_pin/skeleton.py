"""The simplified skeleton of a phane,.

the Blue Book, each line read with ``sed -n``:

* (:14847): the skeleton is named by the number of its nodes, superatoms included, and
  the term 'phane'; every ring or ring system of the skeleton is one superatom and every
  other skeletal atom is one node.
* (:14902) allows a mancude ring or ring system, a bridged fused ring system, a
  saturated monocycle, a von Baeyer hydrocarbon and a spiro alkane as the parent of an
  amplificant; (a)(2) (:14910) disallows "spiro ring systems with at least one fused
  ring system or polycycloalkane ring system".

The nodes are those of ``validation.spelling.cyclophane``: the rings of up to eight atoms that share
an atom form one node, every other atom is one node. A skeleton is read from the structure only.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem

from ...validation.spelling.cyclophane import small_ring_systems
from ..skeletal_replacement import REPLACEMENT_TERMS
from .amplificants import HETERO_ORDER, AmplificantHydride, amplificant_hydride

Node = Tuple[str, int]      # ("amp", k) for the k-th amplificant, ("atom", i) for an acyclic skeletal atom

class SkeletonKind(enum.Enum):
    CYCLE = "cycle"
    PATH = "path"
    BICYCLO = "bicyclo"
    SPIRO = "spiro"


@dataclass(frozen=True)
class Amplificant:
    k: int
    atoms: FrozenSet[int]
    hydride: AmplificantHydride


@dataclass(frozen=True, eq=False)
class PhaneSkeleton:
    """``nodes`` follow the walk of the skeleton; ``adj[node]`` are its neighbours; ``edge[(u, v)]`` is
    ``(atom of u, atom of v, bond type)`` of the bond between the nodes, for both orders; ``other`` are
    the atoms outside the skeleton (substituents and groups of a suffix)."""

    mol: Chem.Mol
    kind: SkeletonKind
    nodes: Tuple[Node, ...]
    adj: Dict[Node, Tuple[Node, ...]]
    edge: Dict[Tuple[Node, Node], Tuple[int, int, Chem.BondType]]
    amplificants: Tuple[Amplificant, ...]
    other: FrozenSet[int]

    def node_of(self, atom: int) -> Optional[Node]:
        """The node of a skeletal atom (an amplificant for a ring atom), None for an atom outside the skeleton."""
        found = self._owner().get(atom)
        return found

    def _owner(self) -> Dict[int, Node]:
        cached = self.__dict__.get("_owner_cache")
        if cached is None:
            cached = {a: ("amp", amp.k) for amp in self.amplificants for a in amp.atoms}
            cached.update({n[1]: n for n in self.nodes if n[0] == "atom"})
            object.__setattr__(self, "_owner_cache", cached)
        return cached


def chain_atom_ok(atom: Chem.Atom) -> bool:
    """An acyclic skeletal atom a phane name can express: carbon, or a heteroatom with a nondetachable 'a'
    prefix,:15192) in its standard valence and without a charge. lists Al, Ga, In and
    Tl, which have no 'a' prefix in the replacement table: they are not built."""
    symbol = atom.GetSymbol()
    if symbol == "C":
        return True
    return (symbol in REPLACEMENT_TERMS and symbol in HETERO_ORDER and not atom.GetFormalCharge()
            and atom.GetTotalValence() == Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()))


def _graph(mol: Chem.Mol, systems: List[FrozenSet[int]]):
    """Nodes, the bonds between nodes as ``(u, v, atom of u, atom of v, bond type)`` and the nodes with a
    bond inside them that closes no small ring."""
    node_of: Dict[int, Node] = {}
    for k, system in enumerate(systems):
        for a in system:
            node_of[a] = ("amp", k)
    for atom in mol.GetAtoms():
        node_of.setdefault(atom.GetIdx(), ("atom", atom.GetIdx()))
    small_ring_bonds: Set[int] = set()
    for ring in mol.GetRingInfo().BondRings():
        if len(ring) <= 8:
            small_ring_bonds.update(ring)
    bonds, loops = [], set()
    for bond in mol.GetBonds():
        a, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        u, v = node_of[a], node_of[c]
        if u != v:
            bonds.append((u, v, a, c, bond.GetBondType()))
        elif bond.GetIdx() not in small_ring_bonds:
            loops.add(u)
    return sorted(set(node_of.values())), bonds, loops


def _core(nodes: List[Node], bonds) -> Set[Node]:
    """What stays of the node graph when the nodes of degree one or none are deleted again and again."""
    degree: Dict[Node, int] = {n: 0 for n in nodes}
    around: Dict[Node, List[Node]] = {n: [] for n in nodes}
    for u, v, *_ in bonds:
        degree[u] += 1
        degree[v] += 1
        around[u].append(v)
        around[v].append(u)
    alive = set(nodes)
    queue = [n for n in nodes if degree[n] <= 1]
    while queue:
        n = queue.pop()
        if n not in alive:
            continue
        alive.discard(n)
        for m in around[n]:
            if m in alive:
                degree[m] -= 1
                if degree[m] <= 1:
                    queue.append(m)
    return alive


def perceive_cyclic(mol: Chem.Mol) -> Optional[PhaneSkeleton]:
    """The skeleton of ``mol`` when its simplified graph is one simple cycle (a monocyclic phane skeleton), a
    bicycle of two cycles that share a path (von Baeyer, or two cycles that share an acyclic atom
    (monospiro,; else None. None as well when a node is not an allowed amplificant, an amplificant
    is attached twice to one atom or by a multiple bond, or an acyclic skeletal atom is not one a phane name
    expresses."""
    systems = small_ring_systems(mol)
    nodes, bonds, loops = _graph(mol, systems)
    core = _core(nodes, bonds)
    if len(core) < 3 or loops & core:
        return None
    core_bonds = [b for b in bonds if b[0] in core and b[1] in core]
    adj: Dict[Node, List[Node]] = {n: [] for n in core}
    edge: Dict[Tuple[Node, Node], Tuple[int, int, Chem.BondType]] = {}
    for u, v, a, c, bond_type in core_bonds:
        if (u, v) in edge:
            return None            # two bonds between two nodes
        if u[0] == "atom" and v[0] == "atom" and mol.GetBondBetweenAtoms(a, c).GetIsAromatic():
            return None            # an aromatic bond between acyclic skeletal atoms has no ene or yne locant
        edge[(u, v)] = (a, c, bond_type)
        edge[(v, u)] = (c, a, bond_type)
        adj[u].append(v)
        adj[v].append(u)
    degrees = sorted(len(adj[n]) for n in core)
    if degrees == [2] * len(core):
        kind = SkeletonKind.CYCLE
        order = _walk(adj, min(core))
        if order is None or len(order) != len(core):
            return None
    elif degrees == [2] * (len(core) - 2) + [3, 3]:
        # two cycles that share a path of nodes (a von Baeyer bicycle), not two rings joined by a chain
        kind = SkeletonKind.BICYCLO
        order = sorted(core)
        if bicyclo_bridges(adj, core) is None:
            return None
    elif degrees == [2] * (len(core) - 1) + [4]:
        # two cycles that share one node; the node is an acyclic atom (the book's:19659)
        kind = SkeletonKind.SPIRO
        order = sorted(core)
        (spiro,) = [n for n in core if len(adj[n]) == 4]
        if spiro[0] != "atom" or spiro_rings(adj, spiro) is None:
            return None
    else:
        return None                  # three rings or more: not built
    for u, v in edge:
        bond_type = edge[(u, v)][2]
        if bond_type not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return None
        if (u[0] == "amp" or v[0] == "amp") and bond_type != Chem.BondType.SINGLE:
            return None
    rename: Dict[Node, Node] = {}
    amplificants: List[Amplificant] = []
    for node in order:
        if node[0] == "amp":
            atoms = systems[node[1]]
            if len({edge[(node, m)][0] for m in adj[node]}) != len(adj[node]):
                return None            # two bonds of the amplificant leave from one atom
            hydride = amplificant_hydride(mol, atoms)
            if hydride is None:
                return None
            rename[node] = ("amp", len(amplificants))
            amplificants.append(Amplificant(len(amplificants), atoms, hydride))
        elif chain_atom_ok(mol.GetAtomWithIdx(node[1])):
            rename[node] = node
        else:
            return None
    inside = {a for amp in amplificants for a in amp.atoms} | {n[1] for n in order if n[0] == "atom"}
    return PhaneSkeleton(
        mol, kind, tuple(rename[n] for n in order),
        {rename[n]: tuple(rename[m] for m in adj[n]) for n in order},
        {(rename[u], rename[v]): data for (u, v), data in edge.items()},
        tuple(amplificants), frozenset(range(mol.GetNumAtoms())) - inside)


def bridges_between(adj: Dict[Node, List[Node]], start: Node, other: Node) -> Optional[List[List[Node]]]:
    """The interior nodes of the bridges that leave ``start`` and reach ``other``, each listed from ``start``;
    None when a bridge does not reach ``other`` (a ring on a bridgehead, a chain that branches)."""
    bridges: List[List[Node]] = []
    for first in sorted(adj[start]):
        interior: List[Node] = []
        previous, current = start, first
        while current != other:
            interior.append(current)
            following = [m for m in adj[current] if m != previous]
            if len(following) != 1 or current == start:
                return None
            previous, current = current, following[0]
        bridges.append(interior)
    return bridges


def bicyclo_bridges(adj: Dict[Node, List[Node]], core) -> Optional[Tuple[Node, Node, List[List[Node]]]]:
    """The two bridgeheads (the nodes of degree three) and the interior nodes of the three bridges between
    them, each listed from the first bridgehead; None when the graph is not a bicycle (a ring joined to another by
    a chain, two rings on a bridgehead)."""
    start, other = sorted(n for n in core if len(adj[n]) == 3)
    bridges = bridges_between(adj, start, other)
    return None if bridges is None else (start, other, bridges)


def spiro_rings(adj: Dict[Node, List[Node]], spiro: Node) -> Optional[List[List[Node]]]:
    """The two rings of a monospiro graph without the spiro node, each as its nodes in order from one end
    (both ends are bonded to the spiro node); None when the graph is not two rings on one node."""
    rings: List[List[Node]] = []
    used = set()
    for first in sorted(adj[spiro]):
        if first in used:
            continue
        ring = [first]
        previous, current = spiro, first
        while True:
            following = [m for m in adj[current] if m != previous]
            if len(following) != 1:
                return None
            if following[0] == spiro:
                break
            ring.append(following[0])
            previous, current = current, following[0]
        used.update((ring[0], ring[-1]))
        rings.append(ring)
    return rings if len(rings) == 2 and len(used) == 4 else None


def _walk(adj: Dict[Node, List[Node]], start: Node) -> Optional[List[Node]]:
    """The nodes of a simple cycle in the order of a walk from ``start`` towards its lower neighbour."""
    order = [start]
    previous, current = start, min(adj[start])
    while current != start:
        order.append(current)
        following = [m for m in adj[current] if m != previous]
        if len(following) != 1 or len(order) > len(adj):
            return None
        previous, current = current, following[0]
    return order
