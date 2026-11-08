"""The principal characteristic group of a fused ring parent cited as its suffix in the general engine.

The general engine's fused tail (``general_engine._emit_fused_ring``) named a fusable ring system by its fusion name only
when the molecule had no suffix group; with one ('naphthalene-2-carboxylic acid') it fell back to the von Baeyer polyene
'bicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene-3-carboxylic acid'. (the Blue Book): "Fusion nomenclature gives
preferred IUPAC names only to compounds having at least two rings of at least five or more members"; the suffix is cited
after the parent with its locants, which come first after indicated hydrogen (c), principal characteristic
groups). This module reads the simple suffix groups on the ring atoms: '-ol', '-amine', '-thiol' on a ring atom, and the
appended '-carboxylic acid', '-carbaldehyde', '-carbonitrile', '-carboxamide' of a carbon on a ring atom.
"""
from typing import List, Optional, Set, Tuple

_INLINE = {'ol': 'O', 'amine': 'N', 'thiol': 'S'}
_APPENDED = ('carboxylic acid', 'carbaldehyde', 'carbonitrile', 'carboxamide')


def fused_suffix_plan(mol, features, ring_set) -> Optional[Tuple[str, List[int], Set[int]]]:
    """``(suffix core, ring carrier atom per group, the group atoms)`` for the principal group of ``features`` on the ring
    system ``ring_set``, or None when a group is not one of the simple shapes above."""
    pg = getattr(features, 'principal_group', None)
    matches = list(getattr(features, 'principal_group_atoms', None) or [])
    if not pg or not matches:
        return None
    from ..rules.seniority import get_suffix
    core = get_suffix(pg, is_ring=True)
    if core not in _INLINE and core not in _APPENDED:
        return None
    carriers: List[int] = []
    group: Set[int] = set()
    seen = set()
    branch_counts = {}
    for match in matches:
        key = tuple(sorted(match))
        if key in seen:
            continue
        seen.add(key)
        off = [i for i in match if i not in ring_set]
        heads_on = [i for i in off if mol.GetAtomWithIdx(i).GetSymbol() == (_INLINE.get(core) or 'C')]
        if not _touches_ring(mol, heads_on, ring_set):
            # a group on a side chain: a prefix of this parent while no branch holds more such groups than the ring
            # does most principal groups, then a ring before a chain)
            comp = frozenset(_branch_of(mol, off[0], ring_set)) if off else frozenset()   # (the O of -CH2OH is off the ring)
            branch_counts[comp] = branch_counts.get(comp, 0) + 1
            continue
        if core in _INLINE:
            heads = [i for i in off if mol.GetAtomWithIdx(i).GetSymbol() == _INLINE[core]]
            if len(heads) != 1:
                return None
            x = mol.GetAtomWithIdx(heads[0])
            ring_nbs = [nb.GetIdx() for nb in x.GetNeighbors() if nb.GetIdx() in ring_set]
            if x.GetDegree() != 1 or len(ring_nbs) != 1 or x.GetFormalCharge():
                return None
            carriers.append(ring_nbs[0])
            group.add(heads[0])
            continue
        heads = [i for i in off if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        if len(heads) != 1:
            return None
        c = mol.GetAtomWithIdx(heads[0])
        ring_nbs = [nb.GetIdx() for nb in c.GetNeighbors() if nb.GetIdx() in ring_set]
        rest = [nb for nb in c.GetNeighbors() if nb.GetIdx() not in ring_set]
        if len(ring_nbs) != 1 or any(nb.GetIdx() not in match or nb.GetDegree() != 1 for nb in rest):
            return None
        carriers.append(ring_nbs[0])
        group.add(heads[0])
        group.update(nb.GetIdx() for nb in rest)
    if not carriers or len(set(carriers)) != len(carriers) or any(n > len(carriers) for n in branch_counts.values()):
        return None
    return core, carriers, group


def _touches_ring(mol, atoms, ring_set) -> bool:
    """True when one of ``atoms`` is bonded to the ring system."""
    return any(nb.GetIdx() in ring_set for i in atoms for nb in mol.GetAtomWithIdx(i).GetNeighbors())


def _branch_of(mol, start: int, ring_set) -> Set[int]:
    """The atoms outside the ring system reached from ``start``."""
    seen, stack = set(), [start]
    while stack:
        i = stack.pop()
        if i in seen or i in ring_set:
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
    return seen
