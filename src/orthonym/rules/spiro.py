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

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..perception.rings import get_spiro_atoms
from ..rules.polycyclic_bridged import get_heteroatom_prefix

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix

# Polyspiro prefix names indexed by number of spiro centers
_POLYSPIRO_PREFIXES = [
    '',           # 0 (unused)
    'spiro',      # 1 = monospiro
    'dispiro',    # 2
    'trispiro',   # 3
    'tetraspiro', # 4
    'pentaspiro', # 5
]


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


def get_spiro_numbering(mol, spiro_center: int) -> Dict[int, int]:
    """
    Generate IUPAC numbering for atoms in a spiro system.

    IUPAC P-31.3.1.2 rules for spiro numbering:
    1. Start at atom adjacent to spiro center in the SMALLER ring
    2. Number around the smaller ring
    3. The spiro center gets the next number
    4. Continue around the larger ring

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Dictionary mapping atom index to IUPAC locant (1-indexed)
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        return {}

    ring1 = list(rings_containing[0])
    ring2 = list(rings_containing[1])

    if len(ring1) <= len(ring2):
        smaller_ring = ring1
        larger_ring = ring2
    else:
        smaller_ring = ring2
        larger_ring = ring1

    spiro_pos_small = smaller_ring.index(spiro_center)
    spiro_pos_large = larger_ring.index(spiro_center)

    reordered_small = (
        smaller_ring[spiro_pos_small + 1:]
        + smaller_ring[:spiro_pos_small + 1]
    )
    reordered_large = (
        larger_ring[spiro_pos_large + 1:]
        + larger_ring[:spiro_pos_large]
    )

    sequence = reordered_small[:-1]
    sequence.append(spiro_center)
    sequence.extend(reordered_large)

    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}
    return atom_to_locant


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


def name_spiro_system(mol) -> Optional[str]:
    """
    Generate the complete IUPAC name for a spiro compound.

    Handles monospiro, dispiro, trispiro hydrocarbons and heterospiro
    compounds with skeletal replacement 'a' prefixes.

    Args:
        mol: RDKit Mol object

    Returns:
        Complete spiro name, or None if not a valid spiro system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')
        >>> name_spiro_system(mol)
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

    has_heteroatoms = any(
        mol.GetAtomWithIdx(idx).GetSymbol() != 'C'
        for idx in ring_atoms_to_check
    )

    if has_heteroatoms:
        hetero_prefix = _build_hetero_prefix(mol, spiro_atoms_set, ring_atoms_to_check)
        if hetero_prefix:
            return f"{hetero_prefix}{descriptor}{parent_name}"

    return f"{descriptor}{parent_name}"


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

    heteroatom_info = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            locant = numbering.get(atom_idx)
            if locant is not None:
                heteroatom_info.append((locant, symbol))

    if not heteroatom_info:
        return None

    # IUPAC priority order: O > S > Se > N > P > Si > B
    priority_order = ['O', 'S', 'Se', 'N', 'P', 'Si', 'B']

    by_element: Dict[str, List[int]] = {}
    for locant, symbol in heteroatom_info:
        by_element.setdefault(symbol, []).append(locant)

    mult_names = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta'}
    prefix_parts = []
    for element in priority_order:
        if element in by_element:
            locants = sorted(by_element[element])
            prefix_name = get_heteroatom_prefix(element)
            count = len(locants)
            mult = mult_names.get(count, f'{count}-')
            locant_str = ','.join(str(loc) for loc in locants)
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
