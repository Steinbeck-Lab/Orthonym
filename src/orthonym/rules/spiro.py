"""
Spiro compound naming module.

Handles naming of spiro systems where two rings share exactly one atom
(the spiro center). Generates IUPAC spiro[a.b] descriptors.

IUPAC rules for spiro naming:
- Spiro descriptor: spiro[a.b] where a <= b
- a = smaller_ring_size - 1, b = larger_ring_size - 1
- The -1 accounts for the shared spiro center
- Numbering starts at atom adjacent to spiro center in smaller ring
- Goes around smaller ring, through spiro center, then around larger ring

Examples:
    spiro[4.5]decane - cyclopentane fused to cyclohexane (5-1=4, 6-1=5)
    spiro[5.5]undecane - two cyclohexanes sharing one carbon (6-1=5, 6-1=5)
"""

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..perception.rings import get_spiro_atoms


# Chain length prefixes for alkane parent names
CHAIN_PREFIXES = {
    1: "meth", 2: "eth", 3: "prop", 4: "but", 5: "pent",
    6: "hex", 7: "hept", 8: "oct", 9: "non", 10: "dec",
    11: "undec", 12: "dodec", 13: "tridec", 14: "tetradec",
    15: "pentadec", 16: "hexadec", 17: "heptadec", 18: "octadec",
    19: "nonadec", 20: "icos",
}


def is_spiro_system(mol) -> bool:
    """
    Check if molecule contains a spiro system.

    For simple spiro: exactly 1 spiro atom and 2 rings.
    For dispiro/trispiro: multiple spiro atoms (future support).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has at least one spiro center

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')  # spiro[4.5]decane
        >>> is_spiro_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        >>> is_spiro_system(mol)
        False
    """
    spiro_atoms = get_spiro_atoms(mol)

    if not spiro_atoms:
        return False

    # For simple monospiro: exactly 1 spiro atom
    # The get_spiro_atoms function already validates that each spiro atom
    # is shared by exactly 2 rings
    return len(spiro_atoms) >= 1


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

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')  # spiro[4.5]decane
        >>> spiro_atoms = list(get_spiro_atoms(mol))
        >>> get_spiro_ring_sizes(mol, spiro_atoms[0])
        (5, 6)
    """
    ri = mol.GetRingInfo()

    # Find all rings containing the spiro center
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        raise ValueError(
            f"Spiro center at atom {spiro_center} must be in exactly 2 rings, "
            f"found {len(rings_containing)}"
        )

    size1 = len(rings_containing[0])
    size2 = len(rings_containing[1])

    # Return sorted (smaller first)
    return (min(size1, size2), max(size1, size2))


def generate_spiro_descriptor(mol) -> Optional[str]:
    """
    Generate the spiro[a.b] descriptor for a spiro compound.

    For simple monospiro compounds (one spiro center), returns spiro[a.b]
    where a and b are (ring_size - 1) for each ring, with a <= b.

    Args:
        mol: RDKit Mol object

    Returns:
        Spiro descriptor string like "spiro[4.5]", or None if not spiro

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')  # spiro[4.5]decane
        >>> generate_spiro_descriptor(mol)
        'spiro[4.5]'
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCCC2')  # spiro[5.5]undecane
        >>> generate_spiro_descriptor(mol)
        'spiro[5.5]'
    """
    spiro_atoms = get_spiro_atoms(mol)

    if not spiro_atoms:
        return None

    # For now, handle only simple monospiro (one spiro center)
    # Dispiro/trispiro support can be added later
    if len(spiro_atoms) != 1:
        return None  # TODO: Handle dispiro, trispiro

    spiro_center = list(spiro_atoms)[0]

    try:
        smaller_ring, larger_ring = get_spiro_ring_sizes(mol, spiro_center)
    except ValueError:
        return None

    # Descriptor values are ring_size - 1 (excluding double-counted spiro center)
    a = smaller_ring - 1
    b = larger_ring - 1

    return f"spiro[{a}.{b}]"


def get_spiro_numbering(mol, spiro_center: int) -> Dict[int, int]:
    """
    Generate IUPAC numbering for atoms in a spiro system.

    IUPAC rules for spiro numbering:
    1. Start at atom adjacent to spiro center in the SMALLER ring
    2. Number around the smaller ring
    3. The spiro center gets the next number
    4. Continue around the larger ring

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Dictionary mapping atom index to IUPAC locant (1-indexed)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')  # spiro[4.5]decane
        >>> spiro_atoms = list(get_spiro_atoms(mol))
        >>> numbering = get_spiro_numbering(mol, spiro_atoms[0])
        >>> len(numbering) == 10  # 10 atoms total
        True
    """
    ri = mol.GetRingInfo()

    # Get the two rings
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        return {}

    ring1 = list(rings_containing[0])
    ring2 = list(rings_containing[1])

    # Determine which is smaller
    if len(ring1) <= len(ring2):
        smaller_ring = ring1
        larger_ring = ring2
    else:
        smaller_ring = ring2
        larger_ring = ring1

    # Find position of spiro center in each ring
    spiro_pos_small = smaller_ring.index(spiro_center)
    spiro_pos_large = larger_ring.index(spiro_center)

    # Reorder smaller ring to put spiro center at the end
    # We want to start at the atom AFTER the spiro center (going around the ring)
    reordered_small = (
        smaller_ring[spiro_pos_small + 1:] +
        smaller_ring[:spiro_pos_small + 1]
    )

    # Reorder larger ring to put spiro center at the start
    # (we continue from spiro center into larger ring)
    reordered_large = (
        larger_ring[spiro_pos_large + 1:] +
        larger_ring[:spiro_pos_large]  # Exclude spiro center (already numbered)
    )

    # Combine: smaller ring (excluding spiro) + spiro center + larger ring (excluding spiro)
    # The spiro center is at position len(smaller_ring) - 1 in smaller_ring
    # Actually: start at atom after spiro in smaller, go around smaller,
    # spiro center is numbered, then continue around larger

    # Build the numbering sequence
    # smaller ring atoms (excluding spiro center, starting from atom after spiro)
    sequence = reordered_small[:-1]  # All except the spiro center at the end
    sequence.append(spiro_center)     # Add spiro center
    sequence.extend(reordered_large)  # Add larger ring atoms (excluding spiro)

    # Create atom_to_locant mapping (1-indexed)
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}

    return atom_to_locant


def _get_ring_adjacent_to_spiro(mol, ring_atoms: List[int], spiro_center: int) -> List[int]:
    """
    Order ring atoms starting from atom adjacent to spiro center.

    Finds the two neighbors of spiro center that are in this ring,
    then returns the ring in order starting from one of them.

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices in the ring
        spiro_center: Atom index of the spiro center

    Returns:
        Ring atoms in order starting from atom adjacent to spiro center
    """
    ring_set = set(ring_atoms)
    spiro_atom = mol.GetAtomWithIdx(spiro_center)

    # Find neighbors of spiro center that are in this ring
    ring_neighbors = []
    for neighbor in spiro_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_set:
            ring_neighbors.append(neighbor.GetIdx())

    if len(ring_neighbors) != 2:
        return ring_atoms  # Unexpected, return as-is

    # Start from the first ring neighbor and walk around the ring
    # Choose the neighbor that gives the correct direction
    start_atom = ring_neighbors[0]

    # Build ordered ring starting from start_atom
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

    Note:
        For simple unsubstituted spiro compounds, returns empty dict.
        Substituent naming will be implemented for Phase 6.
    """
    ri = mol.GetRingInfo()

    # Get all atoms in the spiro system (both rings)
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

        # Find neighbors not in the spiro system
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in spiro_atoms:
                # This is a substituent
                # For now, just count carbons for simple alkyl naming
                carbon_count = _count_substituent_carbons(mol, nbr_idx, spiro_atoms)
                if carbon_count > 0 and carbon_count <= 10:
                    ALKYL_NAMES = {
                        1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl",
                        5: "pentyl", 6: "hexyl", 7: "heptyl", 8: "octyl",
                        9: "nonyl", 10: "decyl",
                    }
                    substituents[locant] = ALKYL_NAMES.get(carbon_count, f"{carbon_count}C")

    return substituents


def _count_substituent_carbons(mol, start_idx: int, exclude: Set[int]) -> int:
    """Count carbon atoms in a substituent group via BFS."""
    from collections import deque

    visited = set()
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
    """
    Get the alkane name for a given atom count.

    Args:
        atom_count: Total number of atoms (carbons) in the spiro system

    Returns:
        Alkane parent name (e.g., "decane", "undecane")
    """
    if atom_count in CHAIN_PREFIXES:
        return f"{CHAIN_PREFIXES[atom_count]}ane"
    else:
        # For counts not in our table, use numerical fallback
        return f"{atom_count}Cane"


def name_spiro_system(mol) -> Optional[str]:
    """
    Generate the complete IUPAC name for a spiro compound.

    For simple monospiro hydrocarbons, returns spiro[a.b]alkane
    (e.g., "spiro[4.5]decane", "spiro[5.5]undecane").

    Args:
        mol: RDKit Mol object

    Returns:
        Complete spiro name, or None if not a valid spiro system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')  # spiro[4.5]decane
        >>> name_spiro_system(mol)
        'spiro[4.5]decane'
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCCC2')  # spiro[5.5]undecane
        >>> name_spiro_system(mol)
        'spiro[5.5]undecane'
    """
    descriptor = generate_spiro_descriptor(mol)

    if descriptor is None:
        return None

    spiro_atoms = get_spiro_atoms(mol)

    if len(spiro_atoms) != 1:
        return None  # Only simple monospiro for now

    spiro_center = list(spiro_atoms)[0]

    # Get the two rings
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        return None

    # Count total carbons in the spiro system
    # This is (ring1_size + ring2_size - 1) because spiro center is shared
    ring1_size = len(rings_containing[0])
    ring2_size = len(rings_containing[1])
    total_atoms = ring1_size + ring2_size - 1  # -1 for shared spiro center

    # Get the alkane parent name
    parent_name = _get_alkane_name(total_atoms)

    # For heterospiro, we would add heteroatom prefixes here
    # For now, just handle hydrocarbon spiro compounds

    # Check for heteroatoms in the spiro system
    all_ring_atoms = set(rings_containing[0]) | set(rings_containing[1])
    has_heteroatoms = False
    for atom_idx in all_ring_atoms:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() != 'C':
            has_heteroatoms = True
            break

    if has_heteroatoms:
        # TODO: Handle heterospiro naming
        # For now, return just the carbocyclic name
        pass

    return f"{descriptor}{parent_name}"


def get_rings_from_spiro_center(mol, spiro_center: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get the two rings sharing a spiro center.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Tuple of two ring tuples: (smaller_ring, larger_ring)
        Each ring is a tuple of atom indices

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

    # Return sorted by size (smaller first)
    if len(ring1) <= len(ring2):
        return (ring1, ring2)
    else:
        return (ring2, ring1)
