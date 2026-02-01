"""
Partial saturation detection and prefix generation for fused ring systems.

This module handles IUPAC 2013 hydro prefixes for partially saturated
fused heterocycles and polycyclic compounds:
- dihydro- (2 H added)
- tetrahydro- (4 H added)
- hexahydro- (6 H added)
- octahydro- (8 H added)
- decahydro- (10 H added)
- dodecahydro- (12 H added)
- perhydro- (fully saturated, no locants needed)

IUPAC 2013 Blue Book P-31.1.1:
"Prefixes 'dihydro', 'tetrahydro', etc. indicate the addition of hydrogen
to specified positions of an otherwise unsaturated parent structure."

Key Rules:
1. Always include locants for partial saturation (2,3-dihydro, not just dihydro)
2. Perhydro means ALL ring atoms saturated - no locants used
3. Saturation prefix comes LAST before parent name, AFTER substituents
4. Order: [substituents]-[saturation prefix]-[indicated H]-[parent]

Reference: IUPAC 2013 Blue Book P-31.1.1
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from rdkit import Chem


# Saturation prefix mapping based on number of added hydrogens
# Each sp3 carbon in ring adds 2 hydrogens vs aromatic parent
SATURATION_PREFIXES: Dict[int, str] = {
    2: 'dihydro',
    4: 'tetrahydro',
    6: 'hexahydro',
    8: 'octahydro',
    10: 'decahydro',
    12: 'dodecahydro',
    14: 'tetradecahydro',
    16: 'hexadecahydro',
}


def detect_partial_saturation(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation by comparing molecule to aromatic parent.

    Counts sp3-hybridized atoms in the ring system that would be sp2/aromatic
    in the parent structure. The IUPAC hydro prefix (dihydro-, tetrahydro-, etc.)
    indicates the number of hydrogen atoms added, which corresponds to
    2 * (number of sp3 ring atoms).

    For fused ring systems:
    - tetrahydroquinoline: 4 positions saturated = 4 sp3 atoms in reduced ring
    - indoline (2,3-dihydroindole): 2 sp3 atoms at positions 2,3
    - decahydronaphthalene (decalin): all 10 ring atoms sp3 = perhydro

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the fully aromatic parent structure

    Returns:
        Dict with saturation info, or None if no saturation detected:
        - 'sp3_count': Number of sp3 atoms in ring system
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'prefix': Saturation prefix string ('dihydro', 'tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'is_perhydro': True if fully saturated

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> result = detect_partial_saturation(mol, 'c1ccc2ncccc2c1')
        >>> result['prefix']
        'hexahydro'  # 3 sp3 carbons * 2 = 6H
    """
    if mol is None:
        return None

    if aromatic_parent_smiles is None:
        return None

    parent = Chem.MolFromSmiles(aromatic_parent_smiles)
    if parent is None:
        return None

    # Get ring atoms in the molecule
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Count aromatic atoms in the parent
    parent_aromatic_count = sum(1 for atom in parent.GetAtoms() if atom.GetIsAromatic())

    # Find sp3-hybridized atoms in the ring system
    # These are the "saturated positions" where hydrogen was added
    sp3_indices = []

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Count sp3 atoms (saturated positions)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)

    sp3_count = len(sp3_indices)

    # No saturation detected
    if sp3_count == 0:
        return None

    # Calculate hydrogen count: each sp3 atom in ring adds ~2H vs aromatic
    hydrogen_count = sp3_count * 2

    # Check for perhydro (fully saturated)
    # When all ring atoms are sp3
    is_perhydro = sp3_count >= parent_aromatic_count

    # Get prefix
    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    return {
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'is_perhydro': is_perhydro,
        'parent_aromatic_count': parent_aromatic_count,
    }


def get_saturation_prefix(hydrogen_count: int) -> Optional[str]:
    """
    Get the IUPAC saturation prefix for a given hydrogen count.

    Args:
        hydrogen_count: Number of added hydrogens (typically 2, 4, 6, 8, 10, 12)

    Returns:
        Saturation prefix string, or None if not a standard count

    Examples:
        >>> get_saturation_prefix(2)
        'dihydro'
        >>> get_saturation_prefix(4)
        'tetrahydro'
        >>> get_saturation_prefix(10)
        'decahydro'
    """
    return SATURATION_PREFIXES.get(hydrogen_count)


def get_saturation_locants(
    mol: Chem.Mol,
    sp3_atom_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    Converts atom indices to IUPAC locants using the provided mapping,
    then sorts them according to IUPAC rules (lowest locant set).

    Args:
        mol: RDKit molecule
        sp3_atom_indices: List of atom indices that are sp3 (saturated)
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix

    Examples:
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, ...}
        >>> get_saturation_locants(mol, [0, 1, 2, 3], atom_to_locant)
        [1, 2, 3, 4]
    """
    locants = []

    for idx in sp3_atom_indices:
        locant = atom_to_locant.get(idx)
        if locant is not None:
            locants.append(locant)

    # Sort locants using IUPAC rules
    return _sort_locants(locants)


def _sort_locants(locants: List[Union[int, str]]) -> List[Union[int, str]]:
    """
    Sort locants according to IUPAC rules.

    Handles mixed int/str locants (e.g., 1, 2, '3a', '4a').
    Numeric locants come first in ascending order, then fusion locants.

    Args:
        locants: List of locants (int or str like '3a', '7a')

    Returns:
        Sorted list of locants
    """
    def _locant_sort_key(loc: Union[int, str]) -> Tuple[int, str]:
        """Sort key: (numeric_part, alpha_suffix)."""
        if isinstance(loc, int):
            return (loc, '')
        elif isinstance(loc, str):
            # Parse '3a' -> (3, 'a'), '7a' -> (7, 'a')
            if loc and loc[-1].isalpha():
                try:
                    return (int(loc[:-1]), loc[-1])
                except ValueError:
                    return (999, loc)
            try:
                return (int(loc), '')
            except ValueError:
                return (999, loc)
        return (999, str(loc))

    return sorted(locants, key=_locant_sort_key)


def format_saturation_prefix(
    prefix: str,
    locants: Optional[List[Union[int, str]]] = None
) -> str:
    """
    Format saturation prefix with locants for IUPAC name.

    IUPAC rules:
    - Perhydro: no locants (perhydro-)
    - All others: locants required (2,3-dihydro-, 1,2,3,4-tetrahydro-)

    Args:
        prefix: Saturation prefix ('dihydro', 'tetrahydro', 'perhydro', etc.)
        locants: Optional list of locants (not used for perhydro)

    Returns:
        Formatted prefix string ready for name assembly

    Examples:
        >>> format_saturation_prefix('perhydro')
        'perhydro'
        >>> format_saturation_prefix('tetrahydro', [1, 2, 3, 4])
        '1,2,3,4-tetrahydro'
        >>> format_saturation_prefix('dihydro', [2, 3])
        '2,3-dihydro'
    """
    # Perhydro never has locants
    if prefix == 'perhydro':
        return 'perhydro'

    # All other saturation prefixes MUST have locants per IUPAC
    if locants:
        sorted_locants = _sort_locants(locants)
        locant_str = ','.join(str(loc) for loc in sorted_locants)
        return f"{locant_str}-{prefix}"

    # Fallback: return prefix without locants (not ideal, but better than nothing)
    return prefix


def get_saturated_position_locants(
    mol: Chem.Mol,
    saturated_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    This is an alias for get_saturation_locants for clearer naming.

    Args:
        mol: RDKit molecule
        saturated_indices: List of atom indices that are saturated
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix
    """
    return get_saturation_locants(mol, saturated_indices, atom_to_locant)


def analyze_saturation_for_naming(
    mol: Chem.Mol,
    aromatic_parent_smiles: str,
    atom_to_locant: Dict[int, Union[int, str]]
) -> Optional[str]:
    """
    Complete analysis for saturation prefix generation in naming.

    This is the main entry point for the composer module. It combines
    detection, locant assignment, and formatting into a single call.

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the aromatic parent structure
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Formatted saturation prefix string, or None if no saturation

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> parent = 'c1ccc2ncccc2c1'  # quinoline
        >>> atom_to_locant = {...}  # IUPAC locant mapping
        >>> analyze_saturation_for_naming(mol, parent, atom_to_locant)
        '1,2,3,4-tetrahydro'
    """
    # Detect saturation
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        return None

    prefix = result['prefix']
    saturated_indices = result['saturated_indices']
    is_perhydro = result['is_perhydro']

    if is_perhydro:
        # Perhydro: no locants needed
        return 'perhydro'

    # Get locants for saturated positions
    locants = get_saturated_position_locants(mol, saturated_indices, atom_to_locant)

    if not locants:
        # No locants available - return prefix only (suboptimal)
        return prefix

    return format_saturation_prefix(prefix, locants)


def is_fully_saturated(mol: Chem.Mol, aromatic_parent_smiles: str) -> bool:
    """
    Check if a molecule is fully saturated (perhydro) relative to parent.

    Args:
        mol: RDKit molecule to check
        aromatic_parent_smiles: SMILES of the aromatic parent

    Returns:
        True if the molecule is fully saturated (perhydro)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCCC2CCCCC12')  # decalin (perhydronaphthalene)
        >>> is_fully_saturated(mol, 'c1ccc2ccccc2c1')  # naphthalene
        True
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        # No saturation info - could be already fully aromatic or no match
        return False
    return result.get('is_perhydro', False)


def count_ring_sp3_atoms(mol: Chem.Mol) -> int:
    """
    Count sp3-hybridized atoms in ring systems.

    Utility function for saturation analysis.

    Args:
        mol: RDKit molecule

    Returns:
        Number of sp3 atoms in rings
    """
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    count = 0
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            count += 1

    return count


def get_ring_saturation_level(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> str:
    """
    Get a human-readable saturation level description.

    Args:
        mol: RDKit molecule
        aromatic_parent_smiles: SMILES of aromatic parent

    Returns:
        Description string: 'aromatic', 'partially saturated', or 'fully saturated'
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)

    if result is None:
        return 'aromatic'
    elif result.get('is_perhydro', False):
        return 'fully saturated'
    else:
        return 'partially saturated'


# =============================================================================
# Carbocyclic Partial Saturation (PAH systems)
# =============================================================================


def detect_carbocyclic_partial_saturation(
    mol: Chem.Mol,
    fused_ring_atoms: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation in carbocyclic fused systems.

    This function identifies partially saturated polycyclic aromatic hydrocarbons
    like tetrahydronaphthalene, dihydroanthracene, etc. It analyzes the fused
    ring system to detect if it's a partially saturated version of a known
    aromatic parent (naphthalene, anthracene, phenanthrene).

    The detection works by:
    1. Checking all ring atoms are carbons (pure carbocycle)
    2. Counting aromatic vs sp3 atoms
    3. Matching the ring system size and structure to known parents
    4. Computing the saturation prefix based on sp3 count

    IUPAC 2013 Blue Book P-31.1.1:
    - tetrahydronaphthalene: 4 sp3 atoms = tetrahydro prefix
    - dihydronaphthalene: 2 sp3 atoms = dihydro prefix
    - decahydronaphthalene (decalin): all sp3 = perhydro or decahydro

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        Dict with saturation info, or None if not a recognized partially
        saturated carbocycle:
        - 'parent_name': Name of aromatic parent ('naphthalene', etc.)
        - 'parent_smiles': SMILES of aromatic parent
        - 'prefix': Saturation prefix string ('tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'sp3_count': Number of sp3 atoms
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'is_perhydro': True if fully saturated
        - 'atom_to_locant': Mapping from atom index to IUPAC locant (if available)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')  # tetrahydronaphthalene
        >>> ri = mol.GetRingInfo()
        >>> ring_atoms = set()
        >>> for ring in ri.AtomRings():
        ...     ring_atoms.update(ring)
        >>> result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        >>> result['prefix']
        'tetrahydro'
        >>> result['parent_name']
        'naphthalene'
    """
    from ..data.partial_saturation_refs import get_reference_smiles

    if mol is None or not fused_ring_atoms:
        return None

    # Check if all ring atoms are carbons (carbocyclic)
    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            # Has heteroatom - not a pure carbocycle
            return None

    # Count sp3 and aromatic atoms in the fused ring system
    sp3_indices = []
    aromatic_indices = []

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)
        elif atom.GetIsAromatic():
            aromatic_indices.append(idx)

    sp3_count = len(sp3_indices)
    aromatic_count = len(aromatic_indices)
    total_ring_atoms = len(fused_ring_atoms)

    # No sp3 atoms = fully aromatic, not partially saturated
    if sp3_count == 0:
        return None

    # Identify parent based on ring system characteristics
    parent_name = None
    parent_smiles = None
    parent_aromatic_count = 0

    # Naphthalene-type: 10 ring atoms total
    if total_ring_atoms == 10:
        parent_name = 'naphthalene'
        parent_smiles = get_reference_smiles('naphthalene')
        parent_aromatic_count = 10

    # Anthracene/phenanthrene-type: 14 ring atoms total
    elif total_ring_atoms == 14:
        # Could be anthracene or phenanthrene - use anthracene as default
        parent_name = 'anthracene'
        parent_smiles = get_reference_smiles('anthracene')
        parent_aromatic_count = 14

    else:
        # Unknown fused system size
        return None

    if parent_smiles is None:
        return None

    # Calculate hydrogen count
    # In fused aromatic systems, each aromatic C has 1 H.
    # When a C goes from sp2 (aromatic) to sp3 (saturated), it gains 1 H.
    # So hydrogen_count = sp3_count (not sp3_count * 2).
    # Example: tetrahydronaphthalene has 4 sp3 C = 4 extra H = tetrahydro
    hydrogen_count = sp3_count

    # Check for perhydro (fully saturated)
    is_perhydro = sp3_count >= parent_aromatic_count

    # Determine prefix
    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    # Build atom-to-locant mapping for the fused system
    # For naphthalene-type systems, use standard IUPAC numbering
    atom_to_locant = _build_naphthalene_type_locants(mol, fused_ring_atoms, sp3_indices)

    return {
        'parent_name': parent_name,
        'parent_smiles': parent_smiles,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'is_perhydro': is_perhydro,
        'atom_to_locant': atom_to_locant,
    }


def _build_naphthalene_type_locants(
    mol: Chem.Mol,
    ring_atoms: Set[int],
    sp3_indices: List[int]
) -> Dict[int, int]:
    """
    Build IUPAC locant mapping for naphthalene-type fused systems.

    For tetrahydronaphthalene, IUPAC numbering is:
    - Positions 1-4: saturated ring (the sp3 atoms)
    - Positions 4a, 5-8, 8a: aromatic ring

    For simplicity, we assign:
    - sp3 atoms get locants 1, 2, 3, 4 (in order around the saturated ring)
    - Aromatic atoms get locants 5, 6, 7, 8 (in order)

    Args:
        mol: RDKit molecule
        ring_atoms: Set of atom indices in the fused system
        sp3_indices: List of sp3 atom indices

    Returns:
        Dict mapping atom index to IUPAC locant (1-indexed)
    """
    from collections import defaultdict

    if not sp3_indices:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Get ring info
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Find the ring containing sp3 atoms (the saturated ring)
    saturated_ring = None
    for ring in atom_rings:
        ring_set = set(ring)
        if not (ring_set & ring_atoms):
            continue
        ring_sp3 = ring_set & set(sp3_indices)
        if len(ring_sp3) >= 2:
            saturated_ring = ring
            break

    if saturated_ring is None:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Find fusion atoms (shared between rings)
    atom_ring_count = defaultdict(int)
    for ring in atom_rings:
        ring_set = set(ring)
        if ring_set & ring_atoms:
            for idx in ring:
                if idx in ring_atoms:
                    atom_ring_count[idx] += 1

    fusion_atoms = {idx for idx, count in atom_ring_count.items() if count > 1}

    # Build ordered traversal around the saturated ring
    # Start from an sp3 atom that's adjacent to a fusion atom
    sp3_set = set(sp3_indices)

    # Find starting sp3 atom adjacent to fusion
    start_atom = None
    for idx in sp3_indices:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in fusion_atoms:
                start_atom = idx
                break
        if start_atom:
            break

    if start_atom is None:
        start_atom = sp3_indices[0]

    # Traverse the sp3 atoms in order (simple chain traversal)
    visited = set()
    sp3_order = []
    current = start_atom

    while len(sp3_order) < len(sp3_indices):
        if current in visited:
            break
        visited.add(current)
        sp3_order.append(current)

        # Find next unvisited sp3 neighbor
        atom = mol.GetAtomWithIdx(current)
        found_next = False
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in sp3_set and nbr_idx not in visited:
                current = nbr_idx
                found_next = True
                break

        if not found_next:
            # No more sp3 neighbors, break
            break

    # Add any missed sp3 atoms
    for idx in sp3_indices:
        if idx not in visited:
            sp3_order.append(idx)

    # Build locant mapping: sp3 atoms get locants 1,2,3,4
    atom_to_locant = {}
    for i, idx in enumerate(sp3_order):
        atom_to_locant[idx] = i + 1

    # Non-sp3 atoms get higher locants (5, 6, 7, 8, ...)
    non_sp3 = [idx for idx in ring_atoms if idx not in sp3_set]
    next_locant = len(sp3_order) + 1
    for idx in sorted(non_sp3):  # Simple ordering for now
        atom_to_locant[idx] = next_locant
        next_locant += 1

    return atom_to_locant


def is_tetrahydronaphthalene(mol: Chem.Mol, fused_ring_atoms: Set[int]) -> bool:
    """
    Check if fused system is tetrahydronaphthalene-type.

    Tetrahydronaphthalene has:
    - 10 ring atoms total
    - 4 sp3 carbons (saturated ring)
    - 6 aromatic carbons (benzene ring)
    - All carbons (no heteroatoms)

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        True if the system is tetrahydronaphthalene-type

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        >>> ri = mol.GetRingInfo()
        >>> ring_atoms = set()
        >>> for ring in ri.AtomRings():
        ...     ring_atoms.update(ring)
        >>> is_tetrahydronaphthalene(mol, ring_atoms)
        True
    """
    if len(fused_ring_atoms) != 10:
        return False

    sp3_count = 0
    aromatic_count = 0

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False

        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_count += 1
        elif atom.GetIsAromatic():
            aromatic_count += 1

    # Tetrahydronaphthalene: 4 sp3 + 6 aromatic
    return sp3_count == 4 and aromatic_count == 6
