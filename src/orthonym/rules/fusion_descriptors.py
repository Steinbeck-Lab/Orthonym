"""
Fusion descriptor generation for systematic fusion names.

Generates IUPAC fusion descriptors for naming fused ring systems when no
retained name exists. Used for systematic names like benzo[a]anthracene,
naphtho[2,1-b]furan, etc.

IUPAC 2013 Fusion Descriptor Rules:
- Letter locants (a, b, c...) designate edges of the PARENT ring
- Edge 'a' is between atoms 1-2, edge 'b' is between 2-3, etc.
- Numerical locants indicate which atoms of the CHILD ring are fused
- Format: child[child_locants-letter]parent (e.g., benzo[a]anthracene)
- IUPAC 2013: 'o' in fusion prefix is NOT elided before vowels

Reference: IUPAC 2013 Blue Book, Section P-25 (Fused Ring Systems)
"""

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..data.polycyclic_data import (
    get_polycyclic_by_smiles,
    get_polycyclic_by_name,
    POLYCYCLIC_DATA,
)
from ..data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
)
from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    is_heterocyclic,
    is_aromatic_ring,
)


# Letters for edge designation (a=first edge, b=second, etc.)
EDGE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


# Mapping from ring names to their fusion prefix forms
# Fusion prefixes follow specific rules:
# - Drop final 'ene' -> 'o' (benzene -> benzo)
# - Drop final 'an' or 'ane' -> 'o' (furan -> furo)
# - Drop final 'ole' -> 'olo' (pyrrole -> pyrrolo)
# - Drop final 'ine' -> 'ino' (pyridine -> pyridino)
# etc.
FUSION_PREFIXES: Dict[str, str] = {
    # Aromatic carbocycles
    'benzene': 'benzo',
    'naphthalene': 'naphtho',
    'anthracene': 'anthra',
    'phenanthrene': 'phenanthro',
    'pyrene': 'pyreno',
    'fluorene': 'fluoreno',

    # 5-membered heterocycles
    'furan': 'furo',
    'thiophene': 'thieno',
    'pyrrole': 'pyrrolo',
    'imidazole': 'imidazo',
    'pyrazole': 'pyrazolo',
    'oxazole': 'oxazolo',
    'isoxazole': 'isoxazolo',
    'thiazole': 'thiazolo',
    'isothiazole': 'isothiazolo',
    'triazole': 'triazolo',
    'tetrazole': 'tetrazolo',

    # 6-membered heterocycles
    'pyridine': 'pyrido',
    'pyrimidine': 'pyrimido',
    'pyrazine': 'pyrazino',
    'pyridazine': 'pyridazino',
    'triazine': 'triazino',

    # Fused heterocycles (for multi-fused systems)
    '1H-indole': 'indolo',
    'indole': 'indolo',
    'quinoline': 'quinolino',
    'isoquinoline': 'isoquinolino',
    '1H-benzimidazole': 'benzimidazo',
    'benzimidazole': 'benzimidazo',
    'benzofuran': 'benzofuro',
    '1-benzofuran': 'benzofuro',
    'benzothiophene': 'benzothieno',
    '1-benzothiophene': 'benzothieno',
    '9H-purine': 'purino',
    'purine': 'purino',
}


def get_fusion_edge(parent_ring: List[int], atom1: int, atom2: int) -> int:
    """
    Find which edge (bond) in the parent ring is shared.

    An edge is defined by two adjacent atoms in the ring. Edge numbering
    follows ring atom order: edge 0 is between atoms at positions 0-1,
    edge 1 is between positions 1-2, etc.

    Args:
        parent_ring: List of atom indices in the parent ring (ordered)
        atom1: First shared atom index
        atom2: Second shared atom index

    Returns:
        0-indexed edge number, or -1 if atoms are not adjacent in ring

    Examples:
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 0, 1)
        0
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 1, 2)
        1
        >>> get_fusion_edge([0, 1, 2, 3, 4, 5], 5, 0)  # wraparound
        5
    """
    ring_size = len(parent_ring)

    # Find positions of both atoms in the ring
    try:
        pos1 = parent_ring.index(atom1)
        pos2 = parent_ring.index(atom2)
    except ValueError:
        return -1  # Atoms not in ring

    # Check if atoms are adjacent (including wraparound)
    diff = abs(pos1 - pos2)

    if diff == 1:
        # Adjacent in sequence
        return min(pos1, pos2)
    elif diff == ring_size - 1:
        # Adjacent via wraparound (last-first)
        return ring_size - 1
    else:
        return -1  # Not adjacent


def get_fusion_letter(parent_ring: List[int], shared_atoms: Tuple[int, int]) -> str:
    """
    Get the fusion letter locant for the parent ring edge.

    Converts edge index to letter: edge 0 -> 'a', edge 1 -> 'b', etc.

    Args:
        parent_ring: List of atom indices in the parent ring
        shared_atoms: Tuple of (atom1, atom2) shared between rings

    Returns:
        Fusion letter ('a', 'b', etc.) or empty string if edge not found

    Examples:
        >>> get_fusion_letter([0, 1, 2, 3, 4, 5], (0, 1))
        'a'
        >>> get_fusion_letter([0, 1, 2, 3, 4, 5], (3, 4))
        'd'
    """
    edge_idx = get_fusion_edge(parent_ring, shared_atoms[0], shared_atoms[1])

    if edge_idx < 0 or edge_idx >= len(EDGE_LETTERS):
        return ''

    return EDGE_LETTERS[edge_idx]


def get_child_locants(
    child_ring: List[int],
    shared_atoms: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Find positions of shared atoms in child ring numbering.

    Returns positions as 1-indexed locants (IUPAC convention).

    Args:
        child_ring: List of atom indices in the child ring
        shared_atoms: Tuple of (atom1, atom2) shared between rings

    Returns:
        Tuple of (lower_locant, higher_locant), 1-indexed
        Returns (0, 0) if atoms not found

    Examples:
        >>> get_child_locants([0, 1, 2, 3, 4], (3, 4))
        (4, 5)
        >>> get_child_locants([6, 7, 8, 9, 10], (6, 7))
        (1, 2)
    """
    try:
        # Find 0-indexed positions
        pos1 = child_ring.index(shared_atoms[0])
        pos2 = child_ring.index(shared_atoms[1])
    except ValueError:
        return (0, 0)

    # Convert to 1-indexed locants
    loc1 = pos1 + 1
    loc2 = pos2 + 1

    # Return in sorted order (lower first)
    if loc1 <= loc2:
        return (loc1, loc2)
    else:
        return (loc2, loc1)


def generate_fusion_descriptor(
    parent_ring: List[int],
    child_ring: List[int],
    shared_atoms: Set[int]
) -> str:
    """
    Generate the fusion descriptor [num,num-letter] format.

    Combines child locants and parent fusion letter into the standard
    IUPAC fusion descriptor format.

    Args:
        parent_ring: List of atom indices in the parent ring
        child_ring: List of atom indices in the child ring
        shared_atoms: Set of atom indices shared between rings

    Returns:
        Fusion descriptor string like "[4,5-d]" or "[a]" for simple cases
        Returns empty string if descriptor cannot be generated

    Examples:
        >>> generate_fusion_descriptor([0,1,2,3,4,5], [6,7,8,9,10], {0,1})
        '[1,2-a]'
    """
    if len(shared_atoms) != 2:
        return ''

    # Convert set to tuple
    atoms_tuple = tuple(sorted(shared_atoms))

    # Get parent fusion letter
    fusion_letter = get_fusion_letter(parent_ring, atoms_tuple)
    if not fusion_letter:
        return ''

    # Get child locants
    child_locs = get_child_locants(child_ring, atoms_tuple)
    if child_locs == (0, 0):
        return ''

    # Build descriptor
    # For simple cases (benzene fused to parent), may use just [a]
    # For more complex, use [num,num-letter]
    loc1, loc2 = child_locs

    return f"[{loc1},{loc2}-{fusion_letter}]"


def get_fusion_prefix(ring_name: str) -> str:
    """
    Convert ring name to its fusion prefix form.

    Uses lookup table for known rings, then applies general rules
    for unknown rings.

    Args:
        ring_name: The name of the ring (e.g., 'benzene', 'furan')

    Returns:
        Fusion prefix form (e.g., 'benzo', 'furo')

    Examples:
        >>> get_fusion_prefix('benzene')
        'benzo'
        >>> get_fusion_prefix('naphthalene')
        'naphtho'
        >>> get_fusion_prefix('furan')
        'furo'
        >>> get_fusion_prefix('pyrrole')
        'pyrrolo'
    """
    # Check lookup table first
    if ring_name in FUSION_PREFIXES:
        return FUSION_PREFIXES[ring_name]

    # Apply general rules for unknown rings
    name = ring_name.lower()

    # Rule: -ene -> -o (benzene -> benzo)
    if name.endswith('ene'):
        return name[:-3] + 'o'

    # Rule: -an -> -o (furan -> furo, pyran -> pyro)
    if name.endswith('an'):
        return name[:-2] + 'o'

    # Rule: -ane -> -o (thiane -> thio)
    if name.endswith('ane'):
        return name[:-3] + 'o'

    # Rule: -ole -> -olo (pyrrole -> pyrrolo, imidazole -> imidazolo)
    if name.endswith('ole'):
        return name[:-1] + 'o'

    # Rule: -ine -> -ino (pyridine -> pyridino)
    if name.endswith('ine'):
        return name[:-1] + 'o'

    # Default: add 'o' suffix
    return name + 'o'


def build_systematic_fusion_name(
    parent_name: str,
    child_name: str,
    descriptor: str
) -> str:
    """
    Assemble the systematic fusion name from components.

    Format: {fusion_prefix}{descriptor}{parent}

    IUPAC 2013 Note: The 'o' in fusion prefixes (benzo, naphtho) is
    NOT elided before vowels (unlike some older conventions).

    Args:
        parent_name: Name of the parent ring system
        child_name: Name of the child (fused) ring
        descriptor: Fusion descriptor (e.g., '[a]', '[2,1-b]')

    Returns:
        Complete systematic fusion name

    Examples:
        >>> build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        'benzo[a]anthracene'
        >>> build_systematic_fusion_name('furan', 'naphthalene', '[2,1-b]')
        'naphtho[2,1-b]furan'
    """
    # Get fusion prefix for the child ring
    prefix = get_fusion_prefix(child_name)

    # Assemble: prefix + descriptor + parent
    # Note: IUPAC 2013 does NOT elide 'o' before vowels
    return f"{prefix}{descriptor}{parent_name}"


def identify_parent_and_child(
    mol,
    ring_a: Set[int],
    ring_b: Set[int]
) -> Tuple[str, str, List[int], List[int]]:
    """
    Determine which ring is parent and which is child for fusion naming.

    Parent selection criteria (IUPAC 2013):
    1. Larger ring system (more atoms) is parent
    2. Nitrogen-containing heterocycle takes priority as parent
    3. More senior heteroatom (O > S > N) may affect selection
    4. Ring with more fusion positions (more edges) is preferred

    Args:
        mol: RDKit Mol object
        ring_a: Set of atom indices in first ring
        ring_b: Set of atom indices in second ring

    Returns:
        Tuple of (parent_name, child_name, parent_ring_list, child_ring_list)
        Returns ('', '', [], []) if rings cannot be identified
    """
    # Convert sets to lists for ordered operations
    ring_a_list = list(ring_a)
    ring_b_list = list(ring_b)

    # Get ring properties
    size_a = len(ring_a)
    size_b = len(ring_b)

    is_hetero_a = is_heterocyclic(mol, ring_a_list)
    is_hetero_b = is_heterocyclic(mol, ring_b_list)

    is_aromatic_a = is_aromatic_ring(mol, ring_a_list)
    is_aromatic_b = is_aromatic_ring(mol, ring_b_list)

    # Determine parent by size first
    parent_ring: List[int]
    child_ring: List[int]

    if size_a > size_b:
        parent_ring = ring_a_list
        child_ring = ring_b_list
    elif size_b > size_a:
        parent_ring = ring_b_list
        child_ring = ring_a_list
    else:
        # Same size - use heteroatom criteria
        # For heterocyclic + carbocyclic fusion, heterocycle is typically parent
        if is_hetero_a and not is_hetero_b:
            parent_ring = ring_a_list
            child_ring = ring_b_list
        elif is_hetero_b and not is_hetero_a:
            parent_ring = ring_b_list
            child_ring = ring_a_list
        else:
            # Default: first ring is parent (arbitrary but consistent)
            parent_ring = ring_a_list
            child_ring = ring_b_list

    # Get names for the rings
    parent_name = _identify_ring_name(mol, parent_ring)
    child_name = _identify_ring_name(mol, child_ring)

    return (parent_name, child_name, parent_ring, child_ring)


def _identify_ring_name(mol, ring_atoms: List[int]) -> str:
    """
    Identify the name of a ring based on its structure.

    Checks against known ring systems (polycyclics, heterocycles).

    Args:
        mol: RDKit Mol object
        ring_atoms: List of atom indices in the ring

    Returns:
        Ring name or empty string if unknown
    """
    ring_size = len(ring_atoms)
    is_hetero = is_heterocyclic(mol, ring_atoms)
    is_aromatic = is_aromatic_ring(mol, ring_atoms)

    # Check for 6-membered aromatic carbocycle (benzene)
    if ring_size == 6 and is_aromatic and not is_hetero:
        return 'benzene'

    # Check for common heterocycles
    if is_hetero and ring_size == 5:
        # Get heteroatom types
        heteroatoms = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if symbol != 'C':
                heteroatoms.append(symbol)

        if heteroatoms == ['O']:
            return 'furan'
        elif heteroatoms == ['S']:
            return 'thiophene'
        elif heteroatoms == ['N']:
            return 'pyrrole'
        elif sorted(heteroatoms) == ['N', 'N']:
            return 'pyrazole'
        elif sorted(heteroatoms) == ['N', 'O']:
            return 'oxazole'
        elif sorted(heteroatoms) == ['N', 'S']:
            return 'thiazole'

    if is_hetero and ring_size == 6:
        heteroatoms = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if symbol != 'C':
                heteroatoms.append(symbol)

        if heteroatoms == ['N']:
            return 'pyridine'
        elif sorted(heteroatoms) == ['N', 'N']:
            return 'pyrimidine'  # or pyrazine/pyridazine - simplified
        elif heteroatoms == ['O']:
            return 'pyran'

    # Default: use generic names based on size
    if not is_hetero:
        if ring_size == 5:
            return 'cyclopentene' if not is_aromatic else 'cyclopentadiene'
        elif ring_size == 6:
            return 'cyclohexene' if not is_aromatic else 'benzene'
        elif ring_size == 7:
            return 'cycloheptene'

    return ''


def generate_systematic_name_for_fused_pair(
    mol,
    ring1: List[int],
    ring2: List[int],
    shared_atoms: Set[int]
) -> Optional[str]:
    """
    Generate systematic fusion name for a pair of fused rings.

    This is the main entry point for generating fusion names when
    no retained name exists.

    Args:
        mol: RDKit Mol object
        ring1: List of atom indices in first ring
        ring2: List of atom indices in second ring
        shared_atoms: Set of atom indices shared between rings

    Returns:
        Systematic fusion name, or None if name cannot be generated

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')  # anthracene
        >>> ri = mol.GetRingInfo()
        >>> # Would generate 'benzo[a]naphthalene' for benzene fused to naphthalene
    """
    if len(shared_atoms) != 2:
        return None

    # Identify parent and child
    parent_name, child_name, parent_ring, child_ring = identify_parent_and_child(
        mol, set(ring1), set(ring2)
    )

    if not parent_name or not child_name:
        return None

    # Generate fusion descriptor
    descriptor = generate_fusion_descriptor(parent_ring, child_ring, shared_atoms)
    if not descriptor:
        return None

    # Build the systematic name
    return build_systematic_fusion_name(parent_name, child_name, descriptor)
