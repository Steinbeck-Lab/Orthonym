"""The nitrogen of a ring that is a substituent of a ring parent.

A nitrogen bonded to the parent ring and to two carbons of a ring of its own is the free valence
of that ring: 'piperidin-1-yl' "General methodology for naming substituent groups",
the heteromonocyclic parent hydride with the free valence at its nitrogen, locant 1), '1H-pyrrol-1-yl',
'9H-carbazol-9-yl'. It is no amino group whose two substituents are chains: the carbons around such a
ring are all-carbon and walk as an 'alkyl' group of the carbons of the ring (``-N1CCCCC1`` read as
'dipentylamino', ``-n1cccc1`` as 'dibutylamino'), so the name described another molecule and the
round trip refused it.
"""
from typing import Iterable, Set


def n_in_substituent_ring(mol, n_idx: int, parent_atoms: Iterable[int]) -> bool:
    """True when the atom ``n_idx`` is a member of a ring that shares no atom with ``parent_atoms``
    (the ring system of the parent): the atom is the attachment atom of a ring substituent."""
    parent: Set[int] = set(parent_atoms)
    return any(n_idx in ring and parent.isdisjoint(ring)
               for ring in mol.GetRingInfo().AtomRings())


#: the largest ring system, and the largest ring in it, whose substituent the benzene writers offer to
#: the substituent namer whatever the size of the rest of the fragment: the fused and spiro
#: hydrocarbons and heterocycles of the retained, fusion and spiro names (pyrene 16, spirobi[fluorene]
#: 25). A macrocycle (the 16-membered ring of porphyrin) or a larger system is a von Baeyer name when
#: it is a prefix, not a ring-yl of the writers below.
MAX_RING_YL_SYSTEM = 40
MAX_RING_YL_RING = 8


def ring_yl_in_scope(mol, atom_idx: int, fragment_atoms: Iterable[int]) -> bool:
    """True when ``atom_idx`` is a ring atom of a ring system (ring-bond-connected: fused, spiro and
    bridged rings count, a bond to another ring does not) of at most ``MAX_RING_YL_SYSTEM`` atoms
    whose rings have at most ``MAX_RING_YL_RING`` members, and the whole fragment bonded to the parent
    has at most that many atoms: a larger fragment is several ring systems joined by bonds (a branched
    polyphenylene), which the writers of the ring-yl names do not take."""
    if not mol.GetAtomWithIdx(atom_idx).IsInRing() or len(set(fragment_atoms)) > MAX_RING_YL_SYSTEM:
        return False
    group = _ring_group(mol, atom_idx)
    return len(group) <= MAX_RING_YL_SYSTEM and all(
        len(ring) <= MAX_RING_YL_RING for ring in mol.GetRingInfo().AtomRings() if group.issuperset(ring))


def ring_yl_branch_in_scope(mol, fragment_atoms: Iterable[int], parent_atoms: Iterable[int]) -> bool:
    """``ring_yl_in_scope`` for a fragment bonded to the parent through a chain: the fragment holds a ring, every
    ring system in it is within the bounds, the whole fragment has at most ``MAX_RING_YL_SYSTEM`` atoms, and
    the parent ring system bears the most principal characteristic groups of the molecule, so no
    other ring system could be the parent: '2-[(decahydronaphthalen-1-yl)methyl]benzene-1,3-diol'). The size limit
    of the writers guards against naming a ring as the parent of a molecule whose parent is another ring;
    the groups tell which ring that is."""
    fragment = set(fragment_atoms)
    if len(fragment) > MAX_RING_YL_SYSTEM:
        return False
    rings = [r for r in mol.GetRingInfo().AtomRings() if fragment.issuperset(r)]
    if not rings or not all(len(_ring_group(mol, ring[0])) <= MAX_RING_YL_SYSTEM and len(ring) <= MAX_RING_YL_RING
                            for ring in rings):
        return False
    from ..perception.functional_groups import detect_functional_groups
    from ..perception.rings import get_ring_systems
    from .parent_selection import _count_pg_on_ring
    from .seniority import get_principal_group
    pg, atoms = get_principal_group(mol, detect_functional_groups(mol))
    parent = set(parent_atoms)
    if not pg or not atoms:
        return False
    held = _count_pg_on_ring(mol, parent, atoms, pg)
    return held > 0 and all(
        _count_pg_on_ring(mol, set(rs), atoms, pg) <= held
        for rs in get_ring_systems(mol) if not (set(rs) & parent))


def _ring_group(mol, atom_idx: int):
    seen = {atom_idx}
    stack = [atom_idx]
    while stack:
        current = stack.pop()
        for bond in mol.GetAtomWithIdx(current).GetBonds():
            if not bond.IsInRing():
                continue
            other = bond.GetOtherAtomIdx(current)
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return seen
