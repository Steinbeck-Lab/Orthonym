"""The ring graph of one ring system, for the spiro analyses of a molecule that has other rings, and the
ring bonds of a spiro atom that is not an sp3 carbon.

``spiro._ring_atom_graph`` joins every two ring atoms that share a bond, a ring bond or not, over the
whole molecule. A ring joined to a spiro system by a single bond (a biaryl link, a boron-bearing ring) is
then a part of one of its components, and a ring system elsewhere in the molecule is a third component,
so ``find_monospiro_separation_atom`` finds no clean two-way split. Inside ``ring_cluster_scope`` the
graph holds only the atoms of the named ring system and the ring bonds between them; outside it the graph
is what it always was, so no PIN-tier analysis changes. A spiro atom has exactly four ring bonds there; a
hypervalent one (a phosphorane or a sulfurane that joins a monocycle to a bicycle it also belongs to, 3 + 2
ring bonds) has more, and inside the scope it is a spiro atom too (``spiro_ring_bond_count``).
"""
import contextlib
import contextvars
import re
from typing import Dict, Optional, Set, Tuple

from ..perception.molcache import bonds_of

_CLUSTER: "contextvars.ContextVar[Optional[frozenset]]" = contextvars.ContextVar(
    "orthonym_ring_cluster", default=None)


@contextlib.contextmanager
def ring_cluster_scope(cluster):
    """Within the body ``ring_atom_graph`` is limited to the ring atoms ``cluster`` and their ring bonds."""
    token = _CLUSTER.set(frozenset(cluster) if cluster is not None else None)
    try:
        yield
    finally:
        _CLUSTER.reset(token)


def ring_atom_graph(mol) -> Tuple[Set[int], Dict[int, Set[int]]]:
    """Ring-atom set and the adjacency between ring atoms: every bond between two ring atoms, or,
    inside ``ring_cluster_scope``, the ring bonds among the atoms of the cluster."""
    cluster = _CLUSTER.get()
    ring_atoms = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    if cluster is not None:
        ring_atoms &= cluster
    adj: Dict[int, Set[int]] = {a: set() for a in ring_atoms}
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_atoms and j in ring_atoms and (cluster is None or bond.IsInRing()):
            adj[i].add(j)
            adj[j].add(i)
    return ring_atoms, adj


def spiro_ring_bond_count(n: int) -> bool:
    """True iff an atom with ``n`` ring bonds can be the spiro atom of a monospiro system: exactly four,
    or, inside ``ring_cluster_scope``, four or more (the atom must still split the ring system into
    two parts, which the caller checks)."""
    return n >= 4 if _CLUSTER.get() is not None else n == 4


def ring_cluster_scope_active() -> bool:
    """True inside ``ring_cluster_scope``: the component-name spiro of one ring system of a molecule
    (the general engine's fallback), not the spiro PIN of the molecule's whole ring system."""
    return _CLUSTER.get() is not None


def is_carbon_von_baeyer_name(name: str) -> bool:
    """True iff ``name`` is a von Baeyer hydrocarbon name that carries no 'a' prefix: ``bicyclo[2.2.1]``
    (what the spiro component namer has always cited), and inside ``ring_cluster_scope`` also
    ``tricyclo[...]`` and the polycycles above it, whose heteroatoms are then cited before 'spiro'
    . Outside the scope only ``bicyclo`` names count, so no PIN name changes."""
    if name.startswith('bicyclo'):
        return True
    return _CLUSTER.get() is not None and re.match(r'^[a-z]+cyclo\[', name) is not None
