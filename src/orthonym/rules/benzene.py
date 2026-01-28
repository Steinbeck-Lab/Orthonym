"""
Benzene naming rules according to IUPAC 2013 (Blue Book).

Handles:
- Monosubstituted benzenes (chlorobenzene, nitrobenzene)
- Polysubstituted benzenes with numeric locants (1,4-dimethylbenzene)
- Benzene ring orientation for lowest locants
- Alphabetical ordering of substituents

IUPAC 2013 PIN Rules:
- Numeric locants are REQUIRED (not ortho/meta/para)
- Toluene is retained ONLY for unsubstituted methylbenzene
- Substituted methylbenzene uses "methylbenzene" (not "toluene")
- Position 1 assigned to give lowest locants via first-point-of-difference
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from rdkit import Chem

from ..assembly.naming_utils import (
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
)


# Mapping from substituent atom symbol/pattern to prefix name
# Key: (symbol, hybridization/bond_info) or simple symbol
# Value: prefix name
SUBSTITUENT_PREFIXES = {
    # Halogens
    "F": "fluoro",
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo",
    # Common groups - these are detected by the get_benzene_substituents function
    # based on the substituent structure
}

# Alkyl group names (by carbon count)
ALKYL_NAMES = {
    1: "methyl",
    2: "ethyl",
    3: "propyl",
    4: "butyl",
    5: "pentyl",
    6: "hexyl",
    7: "heptyl",
    8: "octyl",
    9: "nonyl",
    10: "decyl",
}


def is_benzene_ring(mol, ring_atoms: Tuple[int, ...]) -> bool:
    """
    Check if a ring is a benzene ring (6-membered aromatic carbocycle).

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        True if ring is benzene (6 aromatic carbons)
    """
    if len(ring_atoms) != 6:
        return False

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False
        # Must be aromatic
        if not atom.GetIsAromatic():
            return False

    return True


def get_benzene_ring(mol) -> Optional[Tuple[int, ...]]:
    """
    Find the benzene ring in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices in the benzene ring, or None if not found
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if is_benzene_ring(mol, ring):
            return ring
    return None


def get_benzene_substituents(mol, ring_atoms: Tuple[int, ...]) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to each benzene carbon.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict mapping ring atom index to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)
    """
    ring_set = set(ring_atoms)
    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip ring atoms
            if nbr_idx in ring_set:
                continue

            # Identify the substituent
            sub_info = _identify_substituent(mol, nbr_idx, ring_set)
            if sub_info:
                substituents[ring_idx].append(sub_info)

    return dict(substituents)


def _identify_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent starting from an atom attached to the ring.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens - single atom substituents
    if symbol in SUBSTITUENT_PREFIXES:
        return {
            'name': SUBSTITUENT_PREFIXES[symbol],
            'atoms': [start_idx]
        }

    # Nitrogen-based groups
    if symbol == 'N':
        return _identify_nitrogen_group(mol, start_idx, ring_atoms)

    # Oxygen-based groups
    if symbol == 'O':
        return _identify_oxygen_group(mol, start_idx, ring_atoms)

    # Carbon-based groups (alkyl)
    if symbol == 'C':
        return _identify_alkyl_group(mol, start_idx, ring_atoms)

    return None


def _identify_nitrogen_group(mol, n_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent (amino, nitro, etc.)."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            # Nitro group
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    # Simple amino (-NH2)
    h_count = n_atom.GetTotalNumHs()
    if h_count == 2 and len(neighbors) == 0:
        return {'name': 'amino', 'atoms': [n_idx]}

    return None


def _identify_oxygen_group(mol, o_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent (hydroxy, methoxy, etc.)."""
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    # Methoxy (-OCH3)
    if len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        c_atom = neighbors[0]
        # Check if it's methyl (3 hydrogens, no other heavy atoms)
        if c_atom.GetTotalNumHs() == 3:
            c_neighbors = [n for n in c_atom.GetNeighbors() if n.GetIdx() != o_idx]
            if not c_neighbors:
                return {'name': 'methoxy', 'atoms': [o_idx, c_atom.GetIdx()]}

    return None


def _identify_alkyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent.

    Uses BFS to find all connected carbons and counts total carbons
    to determine alkyl name.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = [start_idx]
    carbon_count = 0
    all_atoms = []

    while queue:
        current_idx = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        else:
            # Non-carbon in the chain - this is not a simple alkyl
            # For Phase 2, we'll handle more complex substituents
            pass

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains heteroatom - not a simple alkyl
            # Check for vinyl (ethenyl) - C=C attached to ring
            if _is_vinyl_group(mol, start_idx, ring_atoms):
                return {'name': 'ethenyl', 'atoms': all_atoms}
            return None

    # Check for branched alkyl (isopropyl, tert-butyl, etc.)
    name = _get_alkyl_name(mol, start_idx, carbon_count, ring_atoms)
    if name:
        return {'name': name, 'atoms': all_atoms}

    return None


def _is_vinyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> bool:
    """Check if the group is a vinyl (ethenyl) group."""
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check for C=C where start is attached to ring
    for neighbor in start_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        bond = mol.GetBondBetweenAtoms(start_idx, neighbor.GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            if neighbor.GetSymbol() == 'C':
                return True

    return False


def _get_alkyl_name(mol, start_idx: int, carbon_count: int, ring_atoms: Set[int]) -> Optional[str]:
    """
    Get the name for an alkyl substituent.

    For now, handles simple linear alkyls. Branched alkyls will be
    implemented in later phases.
    """
    if carbon_count in ALKYL_NAMES:
        # TODO: Check for branching (isopropyl vs propyl, etc.)
        return ALKYL_NAMES[carbon_count]

    return None


def orient_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Orient benzene ring to give lowest locants to substituents.

    IUPAC 2013 Rules:
    1. For monosubstituted: substituent position is 1
    2. For polysubstituted: use first-point-of-difference
    3. For identical substituents: minimize locant set
    4. For different substituents: position 1 at alphabetically first
       (only if it gives equivalent locant sets)

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring (ordered)
        substituents: Dict from get_benzene_substituents

    Returns:
        List of ring atom indices reordered so position 1 is first
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)  # Should be 6

    # Get substituted positions (indices in ring_list)
    substituted_indices = []
    for i, atom_idx in enumerate(ring_list):
        if atom_idx in substituents:
            substituted_indices.append(i)

    if not substituted_indices:
        # Unsubstituted benzene - any orientation is fine
        return ring_list

    if len(substituted_indices) == 1:
        # Monosubstituted - that position becomes 1
        start_pos = substituted_indices[0]
        return _rotate_list(ring_list, start_pos)

    # Polysubstituted - try all starting positions and both directions
    best_orientation = None
    best_locants = None

    for start_pos in range(n):
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            # Build oriented ring
            oriented = _build_oriented_ring(ring_list, start_pos, direction)

            # Calculate locants for this orientation
            locants = _calculate_locants(oriented, substituents)

            # Compare with best
            if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
                best_locants = locants
                best_orientation = oriented

    return best_orientation


def _rotate_list(lst: List, start: int) -> List:
    """Rotate list so element at index start becomes first."""
    return lst[start:] + lst[:start]


def _build_oriented_ring(
    ring_list: List[int],
    start_pos: int,
    direction: int
) -> List[int]:
    """
    Build an oriented version of the ring.

    Args:
        ring_list: Original list of ring atom indices
        start_pos: Index to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos first, going in specified direction
    """
    n = len(ring_list)
    oriented = []

    for i in range(n):
        idx = (start_pos + i * direction) % n
        oriented.append(ring_list[idx])

    return oriented


def _calculate_locants(
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Calculate locant set for a given orientation.

    Args:
        oriented_ring: Ring atoms in order (position 0 = locant 1)
        substituents: Dict from get_benzene_substituents

    Returns:
        Sorted list of locants (1-indexed positions)
    """
    locants = []
    for i, atom_idx in enumerate(oriented_ring):
        if atom_idx in substituents:
            locants.append(i + 1)  # Locants are 1-indexed

    return sorted(locants)


def _compare_locant_sets(a: List[int], b: List[int]) -> int:
    """
    Compare two locant sets using first-point-of-difference.

    Returns:
        < 0 if a is preferred (lower)
        > 0 if b is preferred
        0 if equal
    """
    for i in range(max(len(a), len(b))):
        val_a = a[i] if i < len(a) else float('inf')
        val_b = b[i] if i < len(b) else float('inf')

        if val_a < val_b:
            return -1
        if val_a > val_b:
            return 1

    return 0


def name_substituted_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> str:
    """
    Generate systematic name for substituted benzene.

    IUPAC 2013 PIN Rules:
    - Use numeric locants (not ortho/meta/para)
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-benzene

    Args:
        mol: RDKit Mol object
        ring_atoms: Original ring atom tuple
        oriented_ring: Oriented ring from orient_benzene
        substituents: Dict from get_benzene_substituents

    Returns:
        IUPAC name string (e.g., "1,4-dimethylbenzene")
    """
    # Build locant-to-substituent mapping
    # oriented_ring[0] = position 1, oriented_ring[1] = position 2, etc.
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Group substituents by name
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for atom_idx in oriented_ring:
        if atom_idx not in substituents:
            continue

        for sub_info in substituents[atom_idx]:
            name = sub_info['name']
            locant = atom_to_locant[atom_idx]
            substituent_groups[name].append(locant)

    # Sort locants within each group
    for name in substituent_groups:
        substituent_groups[name].sort()

    # Build prefix strings, sorted alphabetically by substituent name
    prefixes = []
    for name in sorted(substituent_groups.keys(), key=alpha_sort_key):
        locants = substituent_groups[name]
        count = len(locants)

        # Format prefix
        prefix_str = format_substituent_prefix(name, locants, count)
        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_benzene_prefixes(prefixes)

    # Build final name
    return f"{prefix_part}benzene"


def _join_benzene_prefixes(prefixes: List[str]) -> str:
    """
    Join benzene substituent prefixes with proper hyphenation.

    When one prefix ends with a letter and the next starts with a digit,
    a hyphen is needed.

    Args:
        prefixes: List of formatted prefix strings

    Returns:
        Joined prefix string
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return prefixes[0]

    result = prefixes[0]
    for i in range(1, len(prefixes)):
        current = prefixes[i]

        # Check if we need a hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]

            if last_char.isalpha() and first_char.isdigit():
                result += "-"

        result += current

    return result


def name_benzene_derivative(mol) -> Optional[str]:
    """
    Generate name for a benzene derivative.

    This is the main entry point for benzene naming.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not a benzene derivative
    """
    # Find benzene ring
    ring_atoms = get_benzene_ring(mol)
    if ring_atoms is None:
        return None

    # Get substituents
    substituents = get_benzene_substituents(mol, ring_atoms)

    if not substituents:
        # Unsubstituted benzene - should be caught by retained names
        return "benzene"

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Generate systematic name
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)
