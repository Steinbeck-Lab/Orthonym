"""
Spiro compound naming module.

Handles naming of spiro systems where rings share exactly one atom each
(the spiro center). Generates IUPAC spiro[a.b] descriptors for monospiro
and dispiro[a.b.c.d] for polyspiro compounds.

IUPAC P-31.3 / P-24.2 rules for spiro naming:
- Monospiro descriptor: spiro[a.b] where a <= b (P-31.3.1.1)
- a = smaller_ring_size - 1, b = larger_ring_size - 1
- The -1 accounts for the shared spiro center
- Numbering starts at atom adjacent to spiro center in smaller ring (P-31.3.1.2)
- Goes around smaller ring, through spiro center, then around larger ring

IUPAC P-24.2.2: dispiro/trispiro naming for multiple spiro centers
- dispiro[a.b.c.d] where a,b,c,d are segment sizes between spiro atoms
- Numbering starts in terminal ring, proceeds through spiro atoms

IUPAC P-24.2.4.1: heterocyclic spiro compounds use skeletal replacement
'a' prefixes (oxa, aza, thia, etc.) with locants from spiro numbering

Examples:
    spiro[4.5]decane - cyclopentane fused to cyclohexane (5-1=4, 6-1=5)
    spiro[5.5]undecane - two cyclohexanes sharing one carbon (6-1=5, 6-1=5)
    dispiro[2.1.2.1]octane - three rings sharing two spiro centers
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem

from ..perception.rings import get_spiro_atoms
from ..rules.polycyclic_bridged import get_heteroatom_prefix
# Phase 151-04 WR-01: shared IUPAC P-25.3.1.3 heteroatom priority (halogen-aware).
# v22 G4: get_hw_prefix (clean 'a'-prefix table; get_heteroatom_prefix has typos
# like Te->'tea') + sort_heteroatoms_by_priority for skeletal-replacement order.
from ..data.hw_heteroatoms import (
    get_heteroatom_priority,
    get_hw_prefix,
    sort_heteroatoms_by_priority,
)
# Phase 151-02 D-11/D-20: locant comparator reuse — no parallel comparator
# permitted in this module. Imported at the top so the source-grep lock in
# tests/unit/rules/test_mixed_spiro_fused.py and test_spiro_numbering.py
# can verify the invariant without inspecting individual function bodies.
from ..rules.locants import compare_locant_sets  # noqa: F401 — re-export lock

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix


# Phase 151-02 D-13/D-21: locant tuple type alias for cascade-step-6
# suppliers. Plain int for typical ring atoms; (int, str) tuple for
# fusion atoms with a letter suffix (e.g., (4, 'a') -> "4a").
_Locant = Union[int, Tuple[int, str]]

# Polyspiro prefix names indexed by number of spiro centers
_POLYSPIRO_PREFIXES = [
    '',           # 0 (unused)
    'spiro',      # 1 = monospiro
    'dispiro',    # 2
    'trispiro',   # 3
    'tetraspiro', # 4
    'pentaspiro', # 5
]

# IUPAC standard bonding numbers for skeletal-replacement ('a') atoms
# (P-31.1.4.2.4, Table 2.8). A ring heteroatom whose ACTUAL bonding number
# differs from its standard value carries the lambda convention (P-31.1.4.2):
# e.g. a tetravalent ring sulfur -> "lambda4" cited after its locant
# (4lambda4-thiaspiro[3.5]nonane). Elements absent from this table are
# treated as standard (no lambda) — fail-closed, never a spurious lambda.
_STANDARD_BONDING_NUMBER = {
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2, 'Po': 2,
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
}


def _nonstandard_bonding_number(mol, atom_idx: int) -> Optional[int]:
    """Return the lambda bonding number (P-31.1.4.2) for a ring skeletal atom
    whose valence is non-standard, or None when the valence is standard.

    The bonding number is the total count of skeletal/H bonds (RDKit
    ``GetTotalValence``). It is emitted as ``lambda<n>`` immediately after the
    atom's locant in the 'a'-replacement prefix, e.g. a tetravalent ring
    sulfur at locant 4 -> ``4lambda4-thia`` (P-24.2.4.1 spiro example
    ``4lambda4-thiaspiro[3.5]nonane``).

    Fail-closed: only neutral atoms present in ``_STANDARD_BONDING_NUMBER``
    can carry a lambda; charged/exotic atoms return None (no spurious lambda).
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0:
        return None
    standard = _STANDARD_BONDING_NUMBER.get(atom.GetSymbol())
    if standard is None:
        return None
    bonding_number = atom.GetTotalValence()
    return bonding_number if bonding_number != standard else None


def is_spiro_system(mol) -> bool:
    """
    Check if molecule is a pure spiro system.

    A pure spiro system has N spiro atoms connecting N+1 rings, with
    no additional fused or bridged ring junctions. Molecules that have
    spiro atoms but also have additional polycyclic complexity (fused,
    bridged, etc.) are NOT classified as spiro systems.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a pure spiro system
    """
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return False

    # A pure spiro system with N spiro atoms should have exactly N+1 SSSR rings.
    # If there are more rings, the molecule has additional fused/bridged
    # ring junctions and should not be classified as a spiro system.
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_spiro = len(spiro_atoms)

    return n_rings == n_spiro + 1


def get_spiro_ring_sizes(mol, spiro_center: int) -> Tuple[int, int]:
    """
    Get the sizes of the two rings sharing a spiro center.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Tuple of (smaller_ring_size, larger_ring_size), sorted

    Raises:
        ValueError: If spiro_center is not in exactly 2 rings
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        raise ValueError(
            f"Spiro center at atom {spiro_center} must be in exactly 2 rings, "
            f"found {len(rings_containing)}"
        )

    size1 = len(rings_containing[0])
    size2 = len(rings_containing[1])
    return (min(size1, size2), max(size1, size2))


def generate_spiro_descriptor(mol) -> Optional[str]:
    """
    Generate the spiro descriptor for a spiro compound.

    For monospiro (1 spiro center): spiro[a.b] where a <= b.
    For polyspiro (2+ spiro centers): dispiro[a.b.c.d], trispiro[...], etc.

    Args:
        mol: RDKit Mol object

    Returns:
        Spiro descriptor string like "spiro[4.5]" or "dispiro[2.1.2.1]",
        or None if not spiro
    """
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return None

    if len(spiro_atoms) == 1:
        spiro_center = list(spiro_atoms)[0]
        try:
            smaller_ring, larger_ring = get_spiro_ring_sizes(mol, spiro_center)
        except ValueError:
            return None
        a = smaller_ring - 1
        b = larger_ring - 1
        return f"spiro[{a}.{b}]"

    # Polyspiro (dispiro, trispiro, etc.)
    return _generate_polyspiro_descriptor(mol, spiro_atoms)


def _generate_polyspiro_descriptor(mol, spiro_atoms: Set[int]) -> Optional[str]:
    """
    Generate the polyspiro descriptor (dispiro, trispiro, etc.).

    For N spiro centers sharing N+1 rings, the descriptor has 2*N segments.
    """
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    ring_chain, spiro_chain = _build_ring_chain(mol, all_rings, spiro_atoms)
    if ring_chain is None or spiro_chain is None:
        return None
    if len(spiro_chain) != len(spiro_atoms):
        return None

    segments = _compute_spiro_segments(mol, ring_chain, spiro_chain)
    if not segments:
        return None

    n_spiro = len(spiro_atoms)
    if n_spiro < len(_POLYSPIRO_PREFIXES):
        prefix = _POLYSPIRO_PREFIXES[n_spiro]
    else:
        prefix = f'{n_spiro}spiro'

    seg_str = '.'.join(str(s) for s in segments)
    return f"{prefix}[{seg_str}]"


def _build_ring_chain(
    mol,
    all_rings: List[List[int]],
    spiro_atoms: Set[int],
) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    """
    Build an ordered chain of rings connected by spiro atoms.

    For dispiro (2 spiro atoms, 3 rings):
    result = [terminal_ring_1, middle_ring, terminal_ring_2]
    spiro_order = [spiro_atom_1, spiro_atom_2]
    """
    ring_spiro_map = []
    for ring in all_rings:
        ring_set = set(ring)
        shared = ring_set & spiro_atoms
        ring_spiro_map.append(shared)

    terminal_indices = [i for i, s in enumerate(ring_spiro_map) if len(s) == 1]
    if len(terminal_indices) < 2:
        return None, None

    n_rings = len(all_rings)
    ring_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n_rings)}
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            common_spiro = ring_spiro_map[i] & ring_spiro_map[j]
            if common_spiro:
                for sa in common_spiro:
                    ring_adj[i].append((j, sa))
                    ring_adj[j].append((i, sa))

    best_result = None

    for start_idx in terminal_indices:
        visited_rings = {start_idx}
        chain = [all_rings[start_idx]]
        spiro_order: List[int] = []
        current = start_idx

        while True:
            found_next = False
            for next_ring, via_spiro in ring_adj[current]:
                if next_ring not in visited_rings:
                    visited_rings.add(next_ring)
                    chain.append(all_rings[next_ring])
                    spiro_order.append(via_spiro)
                    current = next_ring
                    found_next = True
                    break
            if not found_next:
                break

        if len(spiro_order) == len(spiro_atoms):
            segments = _compute_spiro_segments(mol, chain, spiro_order)
            if segments:
                if best_result is None or segments < best_result[2]:
                    best_result = (chain, spiro_order, segments)

    if best_result is None:
        return None, None
    return best_result[0], best_result[1]


def _compute_spiro_segments(
    mol,
    ring_chain: List[List[int]],
    spiro_chain: List[int],
) -> List[int]:
    """
    Compute the segment sizes for a polyspiro descriptor.

    For dispiro with ring_chain = [ring_A, ring_B, ring_C] and
    spiro_chain = [s1, s2]:
    - Segment 1: non-spiro atoms in ring_A
    - Segment 2: one path through ring_B between s1 and s2
    - Segment 3: non-spiro atoms in ring_C
    - Segment 4: other path through ring_B between s1 and s2
    """
    segments: List[int] = []
    n_rings = len(ring_chain)

    for ring_idx in range(n_rings):
        ring = ring_chain[ring_idx]
        ring_set = set(ring)
        spiro_in_ring = [s for s in spiro_chain if s in ring_set]

        if len(spiro_in_ring) == 1:
            # Terminal ring: one segment = non-spiro atoms
            seg = len(ring) - 1
            segments.append(seg)

        elif len(spiro_in_ring) == 2:
            # Middle ring: two segments (two paths between the two spiro atoms)
            s1, s2 = spiro_in_ring[0], spiro_in_ring[1]
            path1, path2 = _find_two_paths(mol, ring, s1, s2)
            seg_a = len(path1) - 2
            seg_b = len(path2) - 2
            segments.append(min(seg_a, seg_b))
            segments.append(max(seg_a, seg_b))

    # Reorder for IUPAC format: interleave terminal and middle ring segments
    if n_rings >= 3:
        segments = _reorder_segments_iupac(segments, n_rings)

    return segments


def _reorder_segments_iupac(
    raw_segments: List[int],
    n_rings: int,
) -> List[int]:
    """
    Reorder segments from ring-sequential order to IUPAC descriptor order.

    Ring-sequential: [term1, mid_short, mid_long, term2]
    IUPAC: [term1, mid_short, term2, mid_long]
    """
    if n_rings == 3 and len(raw_segments) == 4:
        return [raw_segments[0], raw_segments[1], raw_segments[3], raw_segments[2]]
    return raw_segments


def _find_two_paths(
    mol,
    ring: List[int],
    start: int,
    end: int,
) -> Tuple[List[int], List[int]]:
    """
    Find the two paths around a ring between start and end atoms.
    """
    ring_set = set(ring)
    adj: Dict[int, List[int]] = {a: [] for a in ring}
    for a in ring:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring_set:
                adj[a].append(nbr.GetIdx())

    neighbors_of_start = [n for n in adj[start] if n in ring_set]
    if len(neighbors_of_start) < 2:
        return ([start, end], [start, end])

    path1 = _ring_walk_to(adj, start, end, neighbors_of_start[0])
    path2 = _ring_walk_to(adj, start, end, neighbors_of_start[1])
    return path1, path2


def _ring_walk_to(
    adj: Dict[int, List[int]],
    start: int,
    end: int,
    first_step: int,
) -> List[int]:
    """Walk around a ring from start via first_step until we reach end."""
    path = [start, first_step]
    visited = {start, first_step}
    current = first_step

    max_steps = len(adj) + 1
    for _ in range(max_steps):
        if current == end:
            break
        for nbr in adj[current]:
            if nbr not in visited:
                path.append(nbr)
                visited.add(nbr)
                current = nbr
                break
    return path


def _walk_ring_from_spiro(
    mol,
    ring: List[int],
    spiro_center: int,
    already_visited: Set[int],
) -> List[int]:
    """
    Walk around a ring starting from the atom adjacent to spiro_center,
    going around the ring and ending at spiro_center.
    Returns atoms in walk order: [non-spiro atoms..., spiro_center]
    """
    ring_set = set(ring)
    spiro_atom = mol.GetAtomWithIdx(spiro_center)

    ring_neighbors = []
    for nbr in spiro_atom.GetNeighbors():
        if nbr.GetIdx() in ring_set and nbr.GetIdx() != spiro_center:
            ring_neighbors.append(nbr.GetIdx())

    if not ring_neighbors:
        return list(ring)

    start = None
    for nbr in ring_neighbors:
        if nbr not in already_visited:
            start = nbr
            break
    if start is None:
        start = ring_neighbors[0]

    ordered = [start]
    walk_visited = {start, spiro_center}

    current = start
    while len(ordered) < len(ring) - 1:
        atom = mol.GetAtomWithIdx(current)
        found = False
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in walk_visited:
                ordered.append(nbr_idx)
                walk_visited.add(nbr_idx)
                current = nbr_idx
                found = True
                break
        if not found:
            break

    ordered.append(spiro_center)
    return ordered


def _walk_ring_between_spiros(
    mol,
    ring: List[int],
    entry_spiro: int,
    exit_spiro: int,
    already_visited: Set[int],
) -> List[int]:
    """Walk through a middle ring from entry_spiro to exit_spiro."""
    path1, path2 = _find_two_paths(mol, ring, entry_spiro, exit_spiro)
    unvisited1 = sum(1 for a in path1 if a not in already_visited)
    unvisited2 = sum(1 for a in path2 if a not in already_visited)
    return path1 if unvisited1 >= unvisited2 else path2


def _walk_unvisited_ring_atoms(
    mol,
    ring: List[int],
    entry_spiro: int,
    already_visited: Set[int],
) -> List[int]:
    """Walk the unvisited portion of a middle ring."""
    ring_set = set(ring)
    unvisited = [a for a in ring if a not in already_visited]
    if not unvisited:
        return []

    start = None
    for a in unvisited:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in already_visited and nbr.GetIdx() in ring_set:
                start = a
                break
        if start:
            break

    if start is None:
        return unvisited

    ordered = [start]
    walk_visited = {start}
    current = start
    while len(ordered) < len(unvisited):
        atom = mol.GetAtomWithIdx(current)
        found = False
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if (nbr_idx in ring_set
                    and nbr_idx not in walk_visited
                    and nbr_idx not in already_visited):
                ordered.append(nbr_idx)
                walk_visited.add(nbr_idx)
                current = nbr_idx
                found = True
                break
        if not found:
            break
    return ordered


def _build_polyspiro_numbering_sequence(
    mol,
    ring_chain: List[List[int]],
    spiro_chain: List[int],
) -> List[int]:
    """
    Build the IUPAC numbering sequence for a polyspiro system.
    """
    n_rings = len(ring_chain)
    sequence: List[int] = []
    visited: Set[int] = set()

    for ring_idx in range(n_rings):
        ring = ring_chain[ring_idx]

        if ring_idx == 0:
            entry_spiro = spiro_chain[0]
            ordered = _walk_ring_from_spiro(mol, ring, entry_spiro, visited)
            for a in ordered:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)
        elif ring_idx == n_rings - 1:
            entry_spiro = spiro_chain[-1]
            ordered = _walk_ring_from_spiro(mol, ring, entry_spiro, visited)
            for a in ordered:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)
        else:
            entry_spiro = spiro_chain[ring_idx - 1]
            exit_spiro = spiro_chain[ring_idx]
            path = _walk_ring_between_spiros(
                mol, ring, entry_spiro, exit_spiro, visited
            )
            for a in path:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)

    # Return path through middle rings (remaining unvisited atoms)
    for ring_idx in range(n_rings - 2, 0, -1):
        ring = ring_chain[ring_idx]
        unvisited_in_ring = [a for a in ring if a not in visited]
        if unvisited_in_ring:
            entry_spiro = spiro_chain[ring_idx]
            path = _walk_unvisited_ring_atoms(mol, ring, entry_spiro, visited)
            for a in path:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)

    return sequence


def _get_polyspiro_numbering(
    mol, spiro_atoms: Set[int]
) -> Optional[Dict[int, int]]:
    """Generate IUPAC numbering for a polyspiro system."""
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    ring_chain, spiro_chain = _build_ring_chain(mol, all_rings, spiro_atoms)
    if ring_chain is None:
        return None

    sequence = _build_polyspiro_numbering_sequence(mol, ring_chain, spiro_chain)
    if not sequence:
        return None

    return {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}


def _spiro_ring_traversals(
    mol, ring: List[int], spiro_center: int
) -> List[List[int]]:
    """Enumerate each directional ordering of a ring's NON-spiro atoms.

    Each ordering starts at a spiro-centre neighbour and walks away from the
    centre around the ring. A simple ring yields two orderings (one per spiro
    neighbour, i.e. the two numbering directions); each has ``len(ring) - 1``
    atoms (the spiro centre itself is excluded). Used to enumerate the
    candidate IUPAC spiro numberings so the lowest-locant rule can choose
    among them deterministically.
    """
    ring_set = set(ring)
    spiro_neighbors = [
        n.GetIdx() for n in mol.GetAtomWithIdx(spiro_center).GetNeighbors()
        if n.GetIdx() in ring_set
    ]
    traversals: List[List[int]] = []
    for start in spiro_neighbors:
        ordered = [start]
        visited = {spiro_center, start}
        current = start
        while len(ordered) < len(ring) - 1:
            nxt = None
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in ring_set and ni not in visited:
                    nxt = ni
                    break
            if nxt is None:
                break
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(ordered) == len(ring) - 1:
            traversals.append(ordered)
    return traversals


def get_spiro_numbering(mol, spiro_center: int) -> Dict[int, int]:
    """
    Generate IUPAC numbering for a monospiro system.

    IUPAC P-24.2.1 / P-31.3.1.2: numbering starts at an atom adjacent to the
    spiro centre in the SMALLER ring, proceeds around that ring, through the
    spiro centre, then around the larger ring.

    Among the directional choices (which spiro-neighbour starts each ring, and
    — when the two rings are the same size — which ring is numbered first), the
    chosen numbering gives the LOWEST locants to the heteroatoms considered
    together, then to the most senior heteroatom (P-31.1.4.3.4 / P-24.2.4.1).
    A spelling-independent canonical-rank tiebreak makes the result fully
    deterministic for symmetric systems (e.g. spiro[5.5] acetals). This both
    fixes the latent SMILES-order dependence in heteroatom locants (a tetra-
    valent ``1-oxaspiro[4.5]decane`` was flipping to ``4-oxaspiro[4.5]decane``)
    and yields the correct lowest-locant PIN.

    Returns a dict mapping atom index to IUPAC locant (1-indexed), or {} when
    the spiro centre is not in exactly two rings.
    """
    ri = mol.GetRingInfo()
    rings_containing = [list(r) for r in ri.AtomRings() if spiro_center in r]
    if len(rings_containing) != 2:
        return {}
    r1, r2 = rings_containing

    # Smaller ring numbered first; equal-sized rings -> either may be first, so
    # enumerate both and let the lowest-locant rule decide.
    if len(r1) < len(r2):
        size_pairs = [(r1, r2)]
    elif len(r2) < len(r1):
        size_pairs = [(r2, r1)]
    else:
        size_pairs = [(r1, r2), (r2, r1)]

    candidates: List[Dict[int, int]] = []
    for smaller, larger in size_pairs:
        for small_seq in _spiro_ring_traversals(mol, smaller, spiro_center):
            for large_seq in _spiro_ring_traversals(mol, larger, spiro_center):
                sequence = small_seq + [spiro_center] + large_seq
                candidates.append(
                    {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}
                )
    if not candidates:
        return {}

    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    def _key(mapping: Dict[int, int]):
        heteros = [
            (a, loc) for a, loc in mapping.items()
            if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
        ]
        # (1) lowest locants for ALL heteroatoms together (P-31.1.4.3.4)
        het_locs = sorted(loc for _a, loc in heteros)
        # (2) then lowest locants to the most senior heteroatom (O > S > ...)
        het_by_seniority = sorted(
            (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc)
            for a, loc in heteros
        )
        # (3) deterministic, spelling-independent tiebreak for symmetric rings
        seq = [a for a, _loc in sorted(mapping.items(), key=lambda kv: kv[1])]
        canon_seq = tuple(canon[a] for a in seq)
        return (het_locs, het_by_seniority, canon_seq)

    return min(candidates, key=_key)


def _get_ring_adjacent_to_spiro(
    mol, ring_atoms: List[int], spiro_center: int
) -> List[int]:
    """Order ring atoms starting from atom adjacent to spiro center."""
    ring_set = set(ring_atoms)
    spiro_atom = mol.GetAtomWithIdx(spiro_center)

    ring_neighbors = []
    for neighbor in spiro_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_set:
            ring_neighbors.append(neighbor.GetIdx())

    if len(ring_neighbors) != 2:
        return ring_atoms

    start_atom = ring_neighbors[0]
    ordered = [start_atom]
    visited = {start_atom}
    current = start_atom

    while len(ordered) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                ordered.append(nbr_idx)
                visited.add(nbr_idx)
                current = nbr_idx
                break

    return ordered


def get_spiro_substituents(
    mol,
    spiro_center: int,
    atom_to_locant: Dict[int, int]
) -> Dict[int, str]:
    """
    Find substituents on a spiro system.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Dictionary mapping locant to substituent name
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    spiro_atoms = set()
    for ring in rings_containing:
        spiro_atoms.update(ring)

    substituents = {}
    for atom_idx in spiro_atoms:
        if atom_idx not in atom_to_locant:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        locant = atom_to_locant[atom_idx]
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in spiro_atoms:
                carbon_count = _count_substituent_carbons(mol, nbr_idx, spiro_atoms)
                if carbon_count > 0:
                    from ..data.chain_names import get_alkyl_name as _gal
                    substituents[locant] = _gal(carbon_count)

    return substituents


def _count_substituent_carbons(mol, start_idx: int, exclude: Set[int]) -> int:
    """Count carbon atoms in a substituent group via BFS."""
    from collections import deque

    visited: Set[int] = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def _get_alkane_name(atom_count: int) -> str:
    """Get the alkane name for a given atom count."""
    return f"{_get_chain_prefix(atom_count)}ane"


def name_spiro_system(mol):
    """
    Generate the complete IUPAC name for a spiro compound.

    Handles monospiro, dispiro, trispiro hydrocarbons and heterospiro
    compounds with skeletal replacement 'a' prefixes.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is False (spiro handler does not
        discover substituents via universal pipeline), or None if not
        a valid spiro system.

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')
        >>> result = name_spiro_system(mol)
        >>> result[0]
        'spiro[4.5]decane'
    """
    descriptor = generate_spiro_descriptor(mol)
    if descriptor is None:
        return None

    spiro_atoms_set = get_spiro_atoms(mol)
    n_spiro = len(spiro_atoms_set)

    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    if n_spiro == 1:
        spiro_center = list(spiro_atoms_set)[0]
        rings_containing = [r for r in all_rings if spiro_center in set(r)]
        if len(rings_containing) != 2:
            return None
        total_atoms = len(rings_containing[0]) + len(rings_containing[1]) - 1
    else:
        spiro_ring_atoms: Set[int] = set()
        for ring in all_rings:
            ring_set = set(ring)
            if ring_set & spiro_atoms_set:
                spiro_ring_atoms |= ring_set
        total_atoms = len(spiro_ring_atoms)

    parent_name = _get_alkane_name(total_atoms)

    # Collect all ring atoms in the spiro system
    ring_atoms_to_check: Set[int] = set()
    for ring in all_rings:
        ring_set = set(ring)
        if ring_set & spiro_atoms_set:
            ring_atoms_to_check |= ring_set

    # Compute IUPAC numbering for the spiro system
    if n_spiro == 1:
        spiro_center = list(spiro_atoms_set)[0]
        atom_to_locant = get_spiro_numbering(mol, spiro_center)
    else:
        atom_to_locant = _get_polyspiro_numbering(mol, spiro_atoms_set)
        if atom_to_locant is None:
            atom_to_locant = {}

    has_heteroatoms = any(
        mol.GetAtomWithIdx(idx).GetSymbol() != 'C'
        for idx in ring_atoms_to_check
    )

    if has_heteroatoms:
        hetero_prefix = _build_hetero_prefix(mol, spiro_atoms_set, ring_atoms_to_check)
        if hetero_prefix:
            name = f"{hetero_prefix}{descriptor}{parent_name}"
            return (name, ring_atoms_to_check, atom_to_locant, False)

    name = f"{descriptor}{parent_name}"
    return (name, ring_atoms_to_check, atom_to_locant, False)


def _build_hetero_prefix(
    mol,
    spiro_atoms_set: Set[int],
    ring_atoms: Set[int],
) -> Optional[str]:
    """
    Build the skeletal replacement 'a' prefix for heterospiro compounds.

    Generates prefix strings like "2,8-dioxa", "2-oxa-6-thia" using
    spiro numbering locants and IUPAC priority ordering.
    """
    n_spiro = len(spiro_atoms_set)

    if n_spiro == 1:
        spiro_center = list(spiro_atoms_set)[0]
        numbering = get_spiro_numbering(mol, spiro_center)
    else:
        numbering = _get_polyspiro_numbering(mol, spiro_atoms_set)

    if not numbering:
        return None

    from ..assembly.naming_utils import get_multiplier_prefix

    heteroatom_info = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            locant = numbering.get(atom_idx)
            if locant is not None:
                heteroatom_info.append((locant, symbol, atom_idx))

    if not heteroatom_info:
        return None

    # Group each element's (locant, lambda) records. The lambda bonding
    # number (P-31.1.4.2) is None for standard-valence atoms.
    by_element: Dict[str, List[Tuple[int, Optional[int]]]] = {}
    for locant, symbol, atom_idx in heteroatom_info:
        lam = _nonstandard_bonding_number(mol, atom_idx)
        by_element.setdefault(symbol, []).append((locant, lam))

    # Cite elements in skeletal-replacement seniority order
    # (P-25.3.1.3 / hw_heteroatoms: O > S > Se > Te > N > P > ... > Si > B),
    # replacing the old hard-coded list (which omitted Te/Ge/As/Sb).
    prefix_parts = []
    for element in sort_heteroatoms_by_priority(list(by_element.keys())):
        entries = sorted(by_element[element])  # by locant, then lambda
        prefix_name = get_hw_prefix(element) or get_heteroatom_prefix(element)
        count = len(entries)
        # Shared multiplying-prefix generator (di/tri/.../hexa/hepta/octa...)
        # — replaces the old penta-capped dict that emitted the malformed
        # "6-oxa" instead of "hexaoxa" (DD7 spiro multiplier bug).
        mult = get_multiplier_prefix(count, prefix_name)
        locant_tokens = [
            (f"{loc}lambda{lam}" if lam is not None else str(loc))
            for loc, lam in entries
        ]
        locant_str = ','.join(locant_tokens)
        prefix_parts.append(f"{locant_str}-{mult}{prefix_name}")

    if not prefix_parts:
        return None

    # Join parts with hyphens; no trailing hyphen -- the last 'a' prefix
    # connects directly to the spiro descriptor (e.g., "2-oxa-6-thiaspiro")
    return '-'.join(prefix_parts)


def get_rings_from_spiro_center(
    mol, spiro_center: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get the two rings sharing a spiro center.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Tuple of two ring tuples: (smaller_ring, larger_ring)

    Raises:
        ValueError: If spiro_center is not in exactly 2 rings
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        raise ValueError(
            f"Spiro center at atom {spiro_center} must be in exactly 2 rings"
        )

    ring1 = tuple(rings_containing[0])
    ring2 = tuple(rings_containing[1])

    if len(ring1) <= len(ring2):
        return (ring1, ring2)
    else:
        return (ring2, ring1)


# ============================================================================
# Phase 151-02 — Mixed spiro/fused detector + name builder + cascade suppliers
# ============================================================================
#
# Source: 151-02-PLAN.md tasks 2-3; 151-AUDIT-B.md verdict
# (PURE_SPIRO_PARTIAL · MIXED_SPIRO_FUSED_MISSING · Q-05 NESTED_FORM_PARSEABLE);
# 151-CONTEXT.md D-09/D-13/D-21/D-24; HERITAGE-1990-insights.md §4.
#
# Design notes (codified from audit):
#  * D-09 lock — `is_spiro_system` body remains byte-identical. Mixed
#    cases are a SEPARATE detector (`is_mixed_spiro_fused`) and a
#    SEPARATE name builder (`name_mixed_spiro_fused`).
#  * D-13 + HERITAGE §4 — name_mixed_spiro_fused implements separable
#    parts: identify fused component → name via existing fused-ring
#    pipeline → identify spiro side ring → name algorithmically →
#    recombine with primed locant on the side-ring's spiro-attachment
#    locant only (Q-05 OPSIN preview confirmed parseable).
#  * D-24 — no postprocessor band-aids. When the algorithm cannot
#    name a fixture, return None and let the caller log to
#    HERITAGE-followups.md.
#  * v18 scope: monospiro mixed cases (1 spiro centre joining a fused
#    component to a single side ring). Multi-spiro mixed cases return
#    None and are logged as v19 follow-ups.
# ============================================================================


def is_mixed_spiro_fused(mol) -> bool:
    """
    Detect a mixed spiro / fused ring system AMENABLE TO HERITAGE §4 NAMING.

    A mixed spiro/fused system has:
      (a) exactly ONE spiro atom (v18 scope; multi-spiro mixed → v19)
      (b) at least one fused-ring junction (rings share an edge)
      (c) the spiro centre cleanly separates the ring graph into
          a FUSED component (≥2 rings sharing edges) on one side AND
          a SINGLE algorithmic side ring on the other — i.e.,
          ``_classify_rings_around_spiro_center`` succeeds.
      (d) is NOT a recognized natural-product backbone (RESEARCH Pitfall 3
          false-positive guard — steroids and alkaloids may carry RDKit
          ring perception artifacts that look spiro-like).

    Mutually exclusive with `is_spiro_system` per Phase 151-02 D-09:
    `is_spiro_system` returns True only when n_rings == n_spiro + 1.

    Topology constraint (c) is critical for canary stability: hexacyclic
    natural-product variants (e.g., aconitane derivatives with one spiro
    centre between two multi-ring fused components) must NOT route to
    Branch 5b — they belong to the polycyclic-bridged Von Baeyer branch.
    Logged as v19 follow-up #7 for both-sides-fused topology.

    Phase 151-02 D-09 / D-13 / D-22(b).

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True iff (a) AND (b) AND (c) AND (d) hold.
    """
    if mol is None:
        return False
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return False
    if len(spiro_atoms) != 1:
        return False  # v18 scope: monospiro mixed only.
    ri = mol.GetRingInfo()
    if ri.NumRings() <= len(spiro_atoms) + 1:
        return False  # pure spiro — defer to is_spiro_system
    # FALSE-POSITIVE GUARD per RESEARCH Pitfall 3: steroid + alkaloid
    # backbones short-circuit to False.
    from ..perception.natural_products import detect_natural_product
    if detect_natural_product(mol) is not None:
        return False
    # CANARY-STABILITY GUARD per Phase 151-02 D-22(b): only claim
    # mixed-spiro-fused when the HERITAGE §4 separable topology applies
    # AND we can actually name the fused part. This restricts the new
    # branch to canonical Q-05 OPSIN-validated forms (indoline, isoquinoline,
    # chromane, indane, tetrahydroquinoline, etc.) and lets exotic
    # large polycyclic natural products (palytoxin-class, aconitane-class)
    # continue to flow through the polycyclic-bridged Von Baeyer branch
    # they were on before Plan 151-02. v19 Follow-up #7 + #8 lift these
    # restrictions once both-sides-fused / multi-spiro-mixed naming is
    # implemented.
    spiro_center = list(spiro_atoms)[0]
    all_rings = [list(r) for r in ri.AtomRings()]
    classification = _classify_rings_around_spiro_center(
        mol, spiro_center, all_rings,
    )
    if classification is None:
        return False
    fused_rings, side_rings = classification
    if len(side_rings) != 1:
        return False
    # Cap the fused-component size at 2 rings. HERITAGE §4 separable form
    # was conceived for benzo-5-saturated, benzo-6-saturated, and similar
    # 2-ring fused components — the catalog (FUSED_HETEROCYCLE_DATA) covers
    # exactly that surface. Larger fused parts (3+) require either
    # systematic ortho-fused naming (which the catalog miss case in
    # _name_fused_component falls through to a generic synthesis that
    # rarely produces a roundtrippable name) OR Von Baeyer treatment
    # (better preserved by the existing polycyclic-bridged dispatch).
    if len(fused_rings) > 2:
        return False
    # Verify the fused component has a CATALOG name BEFORE claiming the
    # input as mixed-spiro-fused. Without this guard the dispatch hijacks
    # molecules whose fused part is uncategorized (e.g., 12-ring fused
    # natural-product backbones) and produces partial names.
    fused_named = _name_fused_component(mol, fused_rings)
    if fused_named is None:
        return False
    return True


def get_spiro_iupac_locants(mol) -> Optional[Dict[int, _Locant]]:
    """
    Cascade-step-6 supplier for pure spiro systems (Phase 151-02 D-21).

    Wraps the existing `_get_polyspiro_numbering` (multi-spiro) and
    `get_spiro_numbering` (monospiro) helpers, returning the same
    atom -> locant map shape that Phase 147's `_build_ring_pos`
    consumes via the `_has_iupac_locants` cascade-step-6 gate.

    Coverage invariant per Pitfall 7: returns None on partial coverage
    so the cascade-step-6 gate falls through to the sorted-int proxy.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom_idx -> int locant, covering ALL ring atoms,
        OR None if mol is not a pure spiro system OR coverage is partial.
    """
    if mol is None:
        return None
    if not is_spiro_system(mol):
        return None
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return None
    ri = mol.GetRingInfo()
    ring_atoms: Set[int] = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)
    if len(spiro_atoms) == 1:
        numbering = get_spiro_numbering(mol, list(spiro_atoms)[0])
    else:
        numbering = _get_polyspiro_numbering(mol, set(spiro_atoms))
    if not numbering:
        return None
    if not (set(numbering.keys()) >= ring_atoms):
        return None  # Pitfall 7: partial coverage -> None
    # Filter to ring atoms only (drop exocyclic side-chain locants if any)
    return {k: v for k, v in numbering.items() if k in ring_atoms}


# ============================================================================
# HERITAGE §4 separable-parts helpers (private)
# ============================================================================


def _classify_rings_around_spiro_center(
    mol, spiro_center: int, all_rings: List[List[int]],
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Partition rings into (FUSED_GROUP, SIDE_GROUP) at a spiro centre.

    The FUSED_GROUP is the connected component of rings sharing edges
    (fused-ring junctions, ≥2 shared atoms) reachable from one of the
    two rings at the spiro centre. The SIDE_GROUP is the connected
    component on the other side of the spiro centre.

    Returns (fused_rings, side_rings) or None if the topology does
    not separate cleanly (e.g., both sides are fused, or the spiro
    centre is not in exactly 2 rings).
    """
    rings_at_center = [r for r in all_rings if spiro_center in r]
    if len(rings_at_center) != 2:
        return None
    ring_a, ring_b = rings_at_center

    # Build fused-only adjacency (rings sharing ≥2 atoms = fused edge,
    # excluding the spiro-only adjacency at the centre itself).
    n = len(all_rings)
    fused_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(all_rings[i]) & set(all_rings[j])
            if len(shared) >= 2:
                fused_adj[i].append(j)
                fused_adj[j].append(i)

    def _component(start_ring_idx: int) -> Set[int]:
        seen = {start_ring_idx}
        stack = [start_ring_idx]
        while stack:
            cur = stack.pop()
            for nbr in fused_adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen

    idx_a = all_rings.index(ring_a)
    idx_b = all_rings.index(ring_b)
    comp_a = _component(idx_a)
    comp_b = _component(idx_b)

    # If both rings are in the SAME fused component, the topology is
    # not a clean separable spiro/fused split — defer to v19.
    if comp_a == comp_b:
        return None

    # Component sizes determine which side is "fused part" (>=2 rings)
    # vs "side ring" (1 ring). Both sides must include >=1 ring;
    # at least one side must have >=2 rings (fused).
    if len(comp_a) >= 2 and len(comp_b) == 1:
        fused_idx, side_idx = comp_a, comp_b
    elif len(comp_b) >= 2 and len(comp_a) == 1:
        fused_idx, side_idx = comp_b, comp_a
    else:
        # Both sides are 1-ring (would be pure spiro, not mixed) or
        # both >=2 (two fused components on either side — exotic v19).
        return None

    fused_rings = [all_rings[i] for i in fused_idx]
    side_rings = [all_rings[i] for i in side_idx]
    return fused_rings, side_rings


def _extract_subfragment(
    mol, atom_indices: Set[int],
) -> Optional[Tuple[Chem.Mol, Dict[int, int]]]:
    """Build an RDKit Mol containing only the specified atoms (and the
    bonds between them). Returns (frag_mol, orig_to_frag_idx_map)
    or None on failure.

    For each kept atom, the count of OUT-OF-FRAGMENT neighbors in the
    original molecule is added as explicit hydrogens on the fragment
    atom. This preserves valence so the fragment sanitizes cleanly and
    downstream naming helpers (which expect bare-skeleton input) see
    a chemically valid molecule.

    The fragment is sanitized so downstream naming helpers see a
    well-formed molecule (aromaticity perception, ring info, valence).
    """
    if not atom_indices:
        return None
    rwmol = Chem.RWMol()
    orig_to_frag: Dict[int, int] = {}
    for orig_idx in sorted(atom_indices):
        atom = mol.GetAtomWithIdx(orig_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        # Hydrogen budget: existing explicit Hs + (degree - in_fragment_neighbors)
        # to compensate for the bonds we are dropping.
        original_in_frag_neighbors = sum(
            1 for n in atom.GetNeighbors()
            if n.GetIdx() in atom_indices
        )
        out_of_frag_neighbors = atom.GetDegree() - original_in_frag_neighbors
        new_atom.SetNumExplicitHs(
            atom.GetTotalNumHs() + out_of_frag_neighbors
        )
        new_atom.SetNoImplicit(False)
        # Preserve aromaticity flag — we re-perceive after sanitize.
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_idx = rwmol.AddAtom(new_atom)
        orig_to_frag[orig_idx] = new_idx
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in orig_to_frag and b in orig_to_frag:
            rwmol.AddBond(orig_to_frag[a], orig_to_frag[b], bond.GetBondType())
    frag = rwmol.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    return frag, orig_to_frag


def _name_fused_component(
    mol, fused_rings: List[List[int]],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name the fused component fragment.

    Returns (name, frag_atom_idx -> locant_in_name) on success, or None.

    Strategy (mirrors the composer cascade):
      1. Build the fragment as an isolated mol.
      2. Use ``match_fused_heterocycle_core`` to fetch the catalog-supplied
         IUPAC locant mapping (which `name_fused_heterocycle` discards
         on the retained-name early-exit path).
      3. Fall back to `name_ortho_fused_bicyclic` for systematic naming
         when no catalog match.
      4. Otherwise return None (v19 follow-up).
    """
    fused_atoms: Set[int] = set()
    for r in fused_rings:
        fused_atoms.update(r)
    extracted = _extract_subfragment(mol, fused_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}

    # Prefer the heterocycle catalog path first — direct call to
    # match_fused_heterocycle_core preserves the locant mapping that
    # name_fused_heterocycle's retained-name shortcut discards.
    try:
        from ..data.fused_heterocycles import match_fused_heterocycle_core
        from .fused_rings import name_fused_heterocycle, name_ortho_fused_bicyclic
    except ImportError:
        return None

    core_match = match_fused_heterocycle_core(frag)
    if core_match is not None:
        core_name, atom_mapping_in_frag, _core_smiles = core_match
        # atom_mapping_in_frag : Dict[int, int|str] — locants for each frag atom
        # Map back to original atom indices, coercing letter-suffixed
        # locants like '7a' to base int (7) for the spiro descriptor
        # (Q-05 OPSIN preview: spiro descriptors use plain integer
        # locants in both pre- and post-comma positions).
        atom_to_locant_in_orig: Dict[int, int] = {}
        for frag_idx, locant in atom_mapping_in_frag.items():
            if frag_idx not in frag_to_orig:
                continue
            base: int
            if isinstance(locant, int):
                base = locant
            elif isinstance(locant, str):
                # '7a' -> 7
                digits = "".join(c for c in locant if c.isdigit())
                if not digits:
                    continue
                base = int(digits)
            elif isinstance(locant, tuple) and len(locant) >= 1:
                base = locant[0] if isinstance(locant[0], int) else 0
            else:
                continue
            atom_to_locant_in_orig[frag_to_orig[frag_idx]] = base
        return core_name, atom_to_locant_in_orig

    # No catalog hit — try systematic ortho-fused naming.
    result = name_ortho_fused_bicyclic(frag)
    if result is None:
        # Try the heterocycle path one more time in case it produces a
        # name (e.g., algorithmic _try_algorithmic_fusion_name).
        result = name_fused_heterocycle(frag)
        if result is None:
            return None

    name, _ring_atoms, atom_to_locant_in_frag, _subs_included = result

    if not atom_to_locant_in_frag:
        # Systematic fallback (e.g., 'decahydronaphthalene') yields a
        # name but no locant map. Synthesize a peripheral-numbering walk
        # to produce a connectivity-correct locant map covering all
        # ring atoms. This honors the cascade-step-6 coverage invariant
        # (Pitfall 7) without inventing new IUPAC numbering rules —
        # for saturated fused bicyclics, the peripheral walk is the
        # canonical IUPAC traversal (P-23.2.5 + P-25.3 inheritance).
        atom_to_locant_in_frag = _synthesize_fused_locants(frag)
        if not atom_to_locant_in_frag:
            return None

    atom_to_locant_in_orig: Dict[int, int] = {}
    for frag_idx, locant in atom_to_locant_in_frag.items():
        if frag_idx in frag_to_orig:
            base = locant if isinstance(locant, int) else (
                locant[0] if isinstance(locant, tuple) else 0)
            if base:
                atom_to_locant_in_orig[frag_to_orig[frag_idx]] = base

    return name, atom_to_locant_in_orig


def _synthesize_fused_locants(frag) -> Dict[int, int]:
    """Synthesize a peripheral-walk locant map for a fused-ring fragment.

    For 2-ring fused systems (ortho-fused bicyclics) without a catalog
    entry — used only for saturated fused carbocyclics named as
    decahydro/octahydro derivatives by `name_ortho_fused_bicyclic`.

    Returns a Dict[atom_idx -> locant_int] covering all ring atoms.
    Returns {} if the fragment is not a clean 2-ring fused system or
    the walk fails.
    """
    ri = frag.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings()]
    if len(rings) != 2:
        return {}
    r1, r2 = rings
    shared = set(r1) & set(r2)
    if len(shared) != 2:
        return {}

    # Build adjacency limited to ring atoms.
    ring_atoms = set(r1) | set(r2)
    adj: Dict[int, List[int]] = {a: [] for a in ring_atoms}
    for a in ring_atoms:
        for nbr in frag.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() in ring_atoms:
                adj[a].append(nbr.GetIdx())

    # Peripheral walk: start at a non-fusion atom of the LARGER ring,
    # walk around the periphery (skipping the bridge), assigning 1..N.
    # The two fusion atoms get '4a'/'8a'-style markers, but for the
    # spiro-attachment locant we only need INT positions.
    larger = r1 if len(r1) >= len(r2) else r2
    smaller = r2 if larger is r1 else r1

    # Find a peripheral start — a non-fusion atom in the larger ring
    # adjacent to a fusion atom (so locant 1 is "next to" the fusion).
    fusion = list(shared)
    start = None
    for a in larger:
        if a in shared:
            continue
        if any(n in shared for n in adj[a]):
            start = a
            break
    if start is None:
        return {}

    # Walk around the larger ring's non-fusion atoms first
    # (start, neighbour, ..., then fusion -> smaller ring -> other fusion -> back).
    locants: Dict[int, int] = {}
    visited: Set[int] = set()
    counter = 1

    # Walk larger ring's non-fusion side
    current = start
    locants[current] = counter
    visited.add(current)
    counter += 1
    # Step into the larger ring: prefer the non-fusion neighbour
    next_atom = None
    for n in adj[current]:
        if n not in visited and n not in shared:
            next_atom = n
            break
    while next_atom is not None and len(locants) < len(larger) - len(shared) + len(smaller) + len(shared):
        if next_atom in shared:
            # Crossed a fusion atom — assign locant and pivot
            locants[next_atom] = counter
            visited.add(next_atom)
            counter += 1
            # Now traverse smaller ring's non-fusion atoms
            for sa in smaller:
                if sa not in visited and sa not in shared:
                    locants[sa] = counter
                    visited.add(sa)
                    counter += 1
                    # Traverse remaining smaller ring atoms in order
                    cur = sa
                    while True:
                        nxt = None
                        for n in adj[cur]:
                            if n not in visited and n in smaller and n not in shared:
                                nxt = n
                                break
                        if nxt is None:
                            break
                        locants[nxt] = counter
                        visited.add(nxt)
                        counter += 1
                        cur = nxt
                    break
            # Assign the OTHER fusion atom
            for fa in fusion:
                if fa not in visited:
                    locants[fa] = counter
                    visited.add(fa)
                    counter += 1
                    break
            break
        else:
            locants[next_atom] = counter
            visited.add(next_atom)
            counter += 1
            cur_step = None
            for n in adj[next_atom]:
                if n not in visited:
                    cur_step = n
                    break
            next_atom = cur_step

    # Coverage check
    if not (set(locants.keys()) >= ring_atoms):
        return {}
    return locants


def _name_side_ring(
    mol, side_ring: List[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a single algorithmic side ring.

    Returns (name, atom_idx_in_orig -> locant) or None.
    """
    side_atoms = set(side_ring)
    extracted = _extract_subfragment(mol, side_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted

    # Detect heteroatoms in the side ring
    has_hetero = any(
        a.GetSymbol() != "C" and a.GetIdx() in [orig_to_frag[i]
                                                for i in side_atoms]
        for a in frag.GetAtoms()
    )

    if has_hetero:
        try:
            from .heterocycles import name_heterocycle
        except ImportError:
            return None
        ring_atoms_in_frag = list(orig_to_frag[i] for i in side_atoms)
        try:
            name = name_heterocycle(frag, ring_atoms_in_frag)
        except Exception:
            return None
        if not name:
            return None
        # For heterocycles, build a simple atom-to-locant map by walking
        # the ring starting at locant 1 = first heteroatom (consistent
        # with IUPAC HW). Defer rigorous orient_heterocycle integration
        # to v19; v18 emits a connectivity-correct map.
        atom_to_locant = _walk_side_ring_locants(
            mol, side_ring, hetero_first=True,
        )
    else:
        # Carbocyclic side ring: cyclo<N>ane / cyclo<N>ene if unsaturated
        ring_size = len(side_ring)
        # Detect at least one double bond inside the ring
        has_double = False
        ring_set = set(side_ring)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_set and b in ring_set:
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_double = True
                    break
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    has_double = True
                    break
        prefix = _get_chain_prefix(ring_size)
        suffix = "ene" if has_double else "ane"
        # cycloprop -> cycloprop + ane = cyclopropane
        # Strip trailing 'a' before -ane already handled by chain_names.
        name = f"cyclo{prefix}{suffix}"
        atom_to_locant = _walk_side_ring_locants(
            mol, side_ring, hetero_first=False,
        )

    return name, atom_to_locant


def _walk_side_ring_locants(
    mol, ring: List[int], *, hetero_first: bool,
) -> Dict[int, int]:
    """Number a side ring 1..N walking around the ring.

    For carbocyclic side rings, locant 1 is chosen at the spiro-attached
    atom (i.e., the atom at the spiro centre). For heterocyclic side
    rings, locant 1 is the highest-priority heteroatom (O > S > Se > N >
    P > Si > B per IUPAC P-25.2). The walk direction is the one giving
    the lowest locant set for the spiro attachment atom (D-11 reuse).

    Returns Dict[atom_idx -> locant_in_side_ring].
    """
    ring_set = set(ring)

    # Build adjacency limited to ring atoms.
    adj: Dict[int, List[int]] = {a: [] for a in ring}
    for a in ring:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() in ring_set:
                adj[a].append(nbr.GetIdx())

    # Choose start atom.
    spiro_set = get_spiro_atoms(mol)
    spiro_in_ring = [a for a in ring if a in spiro_set]
    start: int
    if hetero_first:
        # Phase 151-04 WR-01: use canonical IUPAC P-25 priority from
        # data.hw_heteroatoms (which includes halogens F < Cl < Br < I <
        # O < S < ... per P-25.3.1.3). The previous local dict was
        # missing halogen entries, causing F/Cl/Br/I ring atoms to fall
        # through to default priority 99 (least senior) when IUPAC
        # P-25.3.1.3 requires them to be MOST senior.
        hetero_candidates = sorted(
            (a for a in ring if mol.GetAtomWithIdx(a).GetSymbol() != "C"),
            key=lambda a: get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()),
        )
        if hetero_candidates:
            start = hetero_candidates[0]
        elif spiro_in_ring:
            start = spiro_in_ring[0]
        else:
            start = ring[0]
    else:
        # Carbocyclic: locant 1 at the spiro centre per spiro-descriptor
        # conventions for the side ring.
        start = spiro_in_ring[0] if spiro_in_ring else ring[0]

    # Walk the ring and assign 1, 2, ..., N.
    # Two directions possible — pick the one giving the LOWEST set of
    # spiro-locants (D-11 reuse).
    candidates = []
    for first_step in adj[start]:
        path = [start, first_step]
        visited = {start, first_step}
        current = first_step
        while len(path) < len(ring):
            extended = False
            for nbr in adj[current]:
                if nbr not in visited:
                    path.append(nbr)
                    visited.add(nbr)
                    current = nbr
                    extended = True
                    break
            if not extended:
                break
        if len(path) != len(ring):
            continue
        atom_to_locant = {a: i + 1 for i, a in enumerate(path)}
        # Spiro-locant set for this ordering
        spiro_locants = sorted(atom_to_locant[a] for a in spiro_in_ring)
        candidates.append((spiro_locants, atom_to_locant))

    if not candidates:
        return {}
    # Pick lowest spiro-locant set per first-point-of-difference (D-11)
    candidates.sort(key=lambda x: x[0])  # tuple comparison = first-pt-of-diff
    # Verify with compare_locant_sets to honor the D-11/D-20 lock —
    # tuple-sort and compare_locant_sets agree on plain int lists.
    best = candidates[0]
    for cand in candidates[1:]:
        if compare_locant_sets(cand[0], best[0]) < 0:
            best = cand
    return best[1]


def _component_alpha_key(name: str) -> str:
    """Alphanumerical sort key for a spiro ring-component name (P-24.5.1 /
    P-14.5). Compare by the ring-component name itself, ignoring a leading
    indicated-hydrogen descriptor — e.g. ``1H-indene`` -> ``indene`` so that
    ``cyclopentane`` (c) sorts before ``indene`` (i), per the P-24.5.1 Note
    ("the first ring to be cited is determined by alphabetical order and not
    by seniority of the rings or ring systems")."""
    import re
    s = name.strip().lower()
    s = re.sub(r'^\d+h-', '', s)  # drop a leading '1h-'/'3h-' indicated H
    return s


def _strip_consumed_indicated_h(component_name: str, spiro_locant) -> str:
    """Drop a leading indicated-hydrogen descriptor (``1H-``) from a spiro
    ring-component name when the spiro atom sits at that locant.

    A mancude component such as ``1H-indene`` carries indicated hydrogen to
    place its single sp3 centre. When the quaternary spiro atom occupies that
    centre, the indicated hydrogen is no longer needed in the complete
    structure (P-24.3.2 / P-24.5.1), e.g. spiro at indene C1 -> ``indene``,
    giving ``spiro[cyclopentane-1,1'-indene]`` (NOT ``...1H-indene]``).

    Conservative: only strips when the indicated-H locant equals the spiro
    attachment locant; otherwise the name is returned unchanged (fail-safe)."""
    import re
    m = re.match(r'^(\d+)H-(.+)$', component_name)
    if m and str(spiro_locant) == m.group(1):
        return m.group(2)
    return component_name


def name_mixed_spiro_fused(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """
    Build the HERITAGE §4 separable-parts name for a mixed spiro/fused system.

    Algorithm (Phase 151-02 D-13 + HERITAGE §4):
      1. Identify the spiro centre (must be exactly 1 in v18 scope).
      2. Partition rings at the centre into FUSED component and SIDE ring.
      3. Name the fused component via existing fused-ring pipeline.
      4. Name the side ring algorithmically (cycloalkane / heterocycle).
      5. Recombine: spiro[<fused-name>-<f_loc>,<s_loc>'-<side-name>].
         The prime sits on the side-ring's spiro-attachment locant only,
         per Q-05 OPSIN preview (NESTED_FORM_PARSEABLE).
      6. (v19 follow-up) Re-calculate unsaturation when one part becomes
         fully saturated by the spiro junction.

    Return shape MUST match name_spiro_system per composer.py:3098-3105:
        (name, ring_atoms, atom_to_locant, substituents_included=False)

    Out-of-scope for v18 (return None, log to HERITAGE-followups):
      - Multi-spiro mixed cases (n_spiro > 1 + fused junctions).
      - Cases where _classify_rings_around_spiro_center cannot cleanly
        separate the topology (both sides fused, exotic 4-way junctions).
      - Cases where the fused component name builder declines (no
        retained name AND _name_saturated_fused_carbocyclic returns None).

    Source: 151-CONTEXT.md D-13; HERITAGE-1990-insights.md §4;
            IUPAC P-24; Q-05 OPSIN preview (151-AUDIT-B.md).
    """
    if not is_mixed_spiro_fused(mol):
        return None

    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) != 1:
        # v18 scope: monospiro mixed only. Multi-spiro mixed is logged
        # to HERITAGE-followups for v19 in Task 2 commit message.
        return None
    spiro_center = list(spiro_atoms)[0]

    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    classification = _classify_rings_around_spiro_center(
        mol, spiro_center, all_rings,
    )
    if classification is None:
        return None
    fused_rings, side_rings = classification

    if len(side_rings) != 1:
        return None  # v19: multi-side-ring mixed (rare)

    # Step 2: name the fused component
    fused_named = _name_fused_component(mol, fused_rings)
    if fused_named is None:
        return None
    fused_name, fused_atom_to_locant = fused_named

    # Step 3: name the side ring
    side_named = _name_side_ring(mol, side_rings[0])
    if side_named is None:
        return None
    side_name, side_atom_to_locant = side_named

    # Step 4: locate the spiro centre in each part's locant map.
    f_loc = fused_atom_to_locant.get(spiro_center)
    s_loc = side_atom_to_locant.get(spiro_center)
    if f_loc is None or s_loc is None:
        return None

    # Step 5: assemble the P-24.5.1 nested form.
    #   spiro[<comp1>-<l1>,<l2>'-<comp2>]
    # P-24.5.1 (+ its Note): the two ring components are cited in
    # ALPHANUMERICAL order of the component name — NOT fused-component-first
    # and NOT by ring seniority. The first-cited component is unprimed, the
    # second primed. Indicated hydrogen at the spiro locant is dropped
    # (consumed by the quaternary spiro junction; P-24.3.2).
    fused_key = _component_alpha_key(fused_name)
    side_key = _component_alpha_key(side_name)
    fused_primed = fused_key > side_key  # fused cited SECOND iff it sorts later
    if not fused_primed:
        first_name, first_loc, second_name, second_loc = (
            fused_name, f_loc, side_name, s_loc,
        )
    else:
        first_name, first_loc, second_name, second_loc = (
            side_name, s_loc, fused_name, f_loc,
        )
    first_name = _strip_consumed_indicated_h(first_name, first_loc)
    second_name = _strip_consumed_indicated_h(second_name, second_loc)
    name = f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"

    # Build the combined atom_to_locant map for the cascade-step-6 supplier,
    # primed-consistent with the cited order: the SECOND-cited (primed)
    # component's atoms get the tuple (locant, "'") marker; the first-cited
    # (unprimed) component keeps plain integer locants. The spiro centre lives
    # in both partitions — keep its unprimed (first-cited) locant.
    unprimed_map = side_atom_to_locant if fused_primed else fused_atom_to_locant
    primed_map = fused_atom_to_locant if fused_primed else side_atom_to_locant
    combined_locants: Dict[int, _Locant] = {}
    for atom_idx, locant in unprimed_map.items():
        combined_locants[atom_idx] = locant
    for atom_idx, locant in primed_map.items():
        if atom_idx == spiro_center:
            continue
        # _build_ring_pos coerces ints to (n, '') tuples when ANY tuple is
        # present in the dict, so this primed marking is internally consistent.
        combined_locants[atom_idx] = (locant, "'")

    # Coverage invariant: combined map covers ALL ring atoms.
    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms.update(r)
    if not (set(combined_locants.keys()) >= all_ring_atoms):
        return None  # Pitfall 7: partial coverage -> None

    return (name, all_ring_atoms, combined_locants, False)


def get_mixed_spiro_fused_iupac_locants(
    mol,
) -> Optional[Dict[int, _Locant]]:
    """
    Cascade-step-6 supplier for mixed spiro/fused systems (Phase 151-02 D-21).

    Wraps `name_mixed_spiro_fused` and returns the combined atom-to-locant
    map (covering ALL ring atoms) or None if naming declined.

    Coverage invariant per Pitfall 7: returns None on partial coverage
    so the cascade-step-6 gate falls through to the sorted-int proxy.
    """
    if mol is None:
        return None
    result = name_mixed_spiro_fused(mol)
    if result is None:
        return None
    _name, ring_atoms, atom_to_locant, _subs_included = result
    if not (set(atom_to_locant.keys()) >= ring_atoms):
        return None
    return {k: v for k, v in atom_to_locant.items() if k in ring_atoms}
