"""The two arms of a bridge atom, read from the graph: a cheap necessary condition for the arms to be the same.

The multiplicative namers that cut a molecule at a bridge atom (``rules.multiplicative``) ask whether the two
arms are one structure, by the canonical SMILES of the two pieces. Cutting costs a copy of the molecule, a
sanitization and two SMILES, and most bridges of a large molecule (every ether, thioether and secondary amine,
every N-H of a peptide) have arms of unlike size. The count below settles those from the graph.
"""
from typing import Sequence

from ..perception.molcache import atoms_of, neighbors_of


def _arm(neighbours, bridge_idx: int, start: int) -> set:
    """The atoms reached from ``start`` without crossing ``bridge_idx``."""
    seen = {bridge_idx, start}
    stack = [start]
    while stack:
        for nxt in neighbours[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    seen.discard(bridge_idx)
    return seen


def arm_sizes_differ(mol, bridge_idx: int, nbr_indices: Sequence[int]) -> bool:
    """True when the two arms of a bridge atom hold unlike numbers of atoms that are not hydrogen, so that the
    canonical SMILES of the two pieces cannot be the same string.

    ``nbr_indices`` are the two neighbours of ``bridge_idx`` that start the arms. Two pieces with one
    canonical SMILES have the same atoms but for hydrogens (a SMILES read drops a plain explicit hydrogen and
    no other atom), so unlike counts of the atoms that are not hydrogen settle it. The answer is False, "not
    decided", when the arms hold equally many atoms, which includes a bridge atom in a ring (the two
    neighbours then start one arm); the caller then cuts the molecule as before.
    """
    neighbours = neighbors_of(mol)
    atoms = atoms_of(mol)
    return (sum(1 for i in _arm(neighbours, bridge_idx, nbr_indices[0]) if atoms[i].GetAtomicNum() != 1)
            != sum(1 for i in _arm(neighbours, bridge_idx, nbr_indices[1]) if atoms[i].GetAtomicNum() != 1))
