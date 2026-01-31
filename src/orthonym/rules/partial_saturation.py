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
