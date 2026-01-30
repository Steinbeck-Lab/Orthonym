"""
Bicyclo compound naming according to IUPAC 2013 nomenclature.

Implements bicyclo[x.y.z] descriptor generation for bridged bicyclic hydrocarbons.

IUPAC Reference: Blue Book 2013, P-23.2 (Bridged bicyclic hydrocarbons)

The bicyclo descriptor format is bicyclo[x.y.z] where:
- x, y, z are the number of atoms in each bridge BETWEEN the bridgeheads
  (i.e., path length - 2, excluding the bridgehead atoms)
- Values are sorted in descending order: x >= y >= z
- Total ring atoms = x + y + z + 2 (the +2 accounts for bridgehead atoms)

Examples:
- Norbornane: bicyclo[2.2.1]heptane (bridges: 2, 2, 1; total: 7 carbons)
- Bicyclo[2.2.2]octane (bridges: 2, 2, 2; total: 8 carbons)
"""

from typing import List, Optional, Set, Tuple
from collections import deque
from rdkit import Chem

from ..perception.rings import get_bridgehead_atoms, get_spiro_atoms


# ============================================================================
# Alkane Parent Names (shared with naming_utils but kept local for clarity)
# ============================================================================

_ALKANE_NAMES = {
    1: "methane",
    2: "ethane",
    3: "propane",
    4: "butane",
    5: "pentane",
    6: "hexane",
    7: "heptane",
    8: "octane",
    9: "nonane",
    10: "decane",
    11: "undecane",
    12: "dodecane",
}


def _get_alkane_name(carbon_count: int) -> str:
    """Get the alkane parent name for a carbon count."""
    if carbon_count in _ALKANE_NAMES:
        return _ALKANE_NAMES[carbon_count]
    return f"C{carbon_count}H{2*carbon_count+2}"  # Fallback for large systems


# ============================================================================
# Bridgehead Detection
# ============================================================================

def find_true_bridgeheads(mol) -> Set[int]:
    """
    Find the true bridgehead atoms in a bicyclic system.

    For a simple bicyclic system, bridgehead atoms are:
    1. In 2 or more rings
    2. Have 3 neighbors all within the ring system

    This is more specific than get_bridgehead_atoms() which returns
    all atoms in multiple rings.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices that are true bridgeheads

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> find_true_bridgeheads(mol)
        {2, 5}  # or similar indices for the bridgehead carbons
    """
    ri = mol.GetRingInfo()

    # Count ring membership for each atom
    atom_ring_count = {}
    for ring in ri.AtomRings():
        for idx in ring:
            atom_ring_count[idx] = atom_ring_count.get(idx, 0) + 1

    # Get all ring atoms
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    bridgeheads = set()

    for idx in range(mol.GetNumAtoms()):
        # Must be in multiple rings
        if atom_ring_count.get(idx, 0) < 2:
            continue

        # Must have 3 neighbors
        atom = mol.GetAtomWithIdx(idx)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) != 3:
            continue

        # All neighbors must be in the ring system
        if all(n in ring_atoms for n in neighbors):
            bridgeheads.add(idx)

    return bridgeheads


# ============================================================================
# System Detection
# ============================================================================

def is_bicyclo_system(mol) -> bool:
    """
    Check if a molecule is a simple bicyclo system.

    A bicyclo system has:
    - Exactly 2 true bridgehead atoms
    - No spiro centers
    - At least 2 rings

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a bicyclo system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> is_bicyclo_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        >>> is_bicyclo_system(mol)
        False
    """
    ri = mol.GetRingInfo()

    # Must have at least 2 rings
    if ri.NumRings() < 2:
        return False

    # Must not be spiro
    if get_spiro_atoms(mol):
        return False

    # Must have exactly 2 true bridgeheads
    bridgeheads = find_true_bridgeheads(mol)
    if len(bridgeheads) != 2:
        return False

    return True


# ============================================================================
# Bridge Path Finding
# ============================================================================

def find_bridge_paths(mol, bridgehead1: int, bridgehead2: int) -> List[List[int]]:
    """
    Find all paths between two bridgehead atoms.

    Uses BFS to find all simple paths between the bridgeheads.
    For a bicyclo system, there should be exactly 3 paths.

    Args:
        mol: RDKit Mol object
        bridgehead1: Index of first bridgehead atom
        bridgehead2: Index of second bridgehead atom

    Returns:
        List of paths, where each path is a list of atom indices
        including both bridgeheads

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> bridgeheads = list(find_true_bridgeheads(mol))
        >>> paths = find_bridge_paths(mol, bridgeheads[0], bridgeheads[1])
        >>> len(paths)
        3
    """
    # Get all ring atoms to constrain search
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    all_paths = []

    # BFS to find all paths
    # Each state is (current_atom, path_so_far)
    queue = deque([(bridgehead1, [bridgehead1])])

    while queue:
        current, path = queue.popleft()

        if current == bridgehead2 and len(path) > 1:
            all_paths.append(path)
            continue

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            # Skip if already in path (except target)
            if n_idx in path and n_idx != bridgehead2:
                continue

            # Only follow ring atoms
            if n_idx not in ring_atoms:
                continue

            # If this is the target, add to path and save
            if n_idx == bridgehead2:
                queue.append((n_idx, path + [n_idx]))
            # Otherwise, continue exploring (but don't go back to start)
            elif n_idx != bridgehead1:
                queue.append((n_idx, path + [n_idx]))

    return all_paths


# ============================================================================
# Bridge Length Calculation
# ============================================================================

def get_bridge_lengths(mol, bridgehead1: int, bridgehead2: int) -> List[int]:
    """
    Calculate the bridge lengths between two bridgehead atoms.

    Bridge length = number of atoms BETWEEN bridgeheads (path length - 2).
    Returns lengths sorted in descending order: x >= y >= z.

    Args:
        mol: RDKit Mol object
        bridgehead1: Index of first bridgehead atom
        bridgehead2: Index of second bridgehead atom

    Returns:
        List of bridge lengths sorted descending

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> bridgeheads = list(find_true_bridgeheads(mol))
        >>> get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])
        [2, 2, 1]  # bicyclo[2.2.1]
    """
    paths = find_bridge_paths(mol, bridgehead1, bridgehead2)

    # Bridge length = path length - 2 (exclude both bridgeheads)
    lengths = [len(path) - 2 for path in paths]

    # Sort descending
    lengths.sort(reverse=True)

    return lengths


# ============================================================================
# Descriptor Generation
# ============================================================================

def generate_bicyclo_descriptor(mol) -> Optional[str]:
    """
    Generate the bicyclo[x.y.z] descriptor for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Descriptor string like "bicyclo[2.2.1]", or None if not a bicyclo system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> generate_bicyclo_descriptor(mol)
        'bicyclo[2.2.1]'
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1CC2')  # bicyclo[2.2.2]octane
        >>> generate_bicyclo_descriptor(mol)
        'bicyclo[2.2.2]'
    """
    if not is_bicyclo_system(mol):
        return None

    bridgeheads = list(find_true_bridgeheads(mol))
    if len(bridgeheads) != 2:
        return None

    lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])

    if len(lengths) != 3:
        return None

    return f"bicyclo[{lengths[0]}.{lengths[1]}.{lengths[2]}]"


# ============================================================================
# Full Naming
# ============================================================================

def name_bicyclo_system(mol) -> Optional[str]:
    """
    Generate the full IUPAC name for a bicyclo system.

    Checks for retained names first, then generates systematic name.

    Args:
        mol: RDKit Mol object

    Returns:
        Full IUPAC name like "bicyclo[2.2.1]heptane" or "norbornane",
        or None if not a bicyclo system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> name_bicyclo_system(mol)
        'norbornane'  # or 'bicyclo[2.2.1]heptane' depending on retained name preference
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        >>> name_bicyclo_system(mol)
        'bicyclo[2.2.2]octane'
    """
    # Import here to avoid circular imports
    from ..data.bicyclo_systems import get_retained_bicyclo_name

    # Check for retained name first
    canonical = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    retained = get_retained_bicyclo_name(canonical)
    if retained:
        return retained

    # Generate descriptor
    descriptor = generate_bicyclo_descriptor(mol)
    if not descriptor:
        return None

    # Count total ring carbons
    # For simple bicyclics, count atoms in the ring system
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Count only carbon atoms in ring
    carbon_count = sum(
        1 for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Check for heteroatoms in ring
    heteroatoms = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    if heteroatoms:
        # Heterobicyclic - needs aza/oxa prefix
        # For now, return systematic name with descriptor only
        # Full hetero naming is more complex and deferred
        total_ring = len(ring_atoms)
        parent_name = _get_alkane_name(total_ring)
        return f"{descriptor}{parent_name}"

    # All-carbon bicyclic
    parent_name = _get_alkane_name(carbon_count)
    return f"{descriptor}{parent_name}"


def get_bicyclo_ring_atoms(mol) -> Optional[Set[int]]:
    """
    Get all atoms in the bicyclo ring system.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices in the ring system, or None if not bicyclo

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> get_bicyclo_ring_atoms(mol)
        {0, 1, 2, 3, 4, 5, 6}
    """
    if not is_bicyclo_system(mol):
        return None

    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    return ring_atoms
