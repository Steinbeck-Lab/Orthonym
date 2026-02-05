"""
Ring-as-substituent naming according to IUPAC 2013 (Blue Book).

Implements IUPAC P-61.5: Standard substituent names for rings when they
become substituents on a chain parent structure.

Examples:
- benzene -> phenyl (4-phenylbutanoic acid)
- cyclohexane -> cyclohexyl (4-cyclohexylbutanoic acid)
- naphthalene -> naphthyl (position-specific: 1-naphthyl, 2-naphthyl)
- pyridine -> pyridyl (position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl)
"""

from typing import Dict, List, Optional, Set, Tuple

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix
_CHAIN_PREFIXES = {i: _get_chain_prefix(i) for i in range(3, 21)}


# IUPAC P-61.5: Standard substituent names for rings
# Maps ring system name to substituent prefix name
RING_SUBSTITUENT_NAMES: Dict[str, str] = {
    # Carbocyclic aromatic
    'benzene': 'phenyl',
    'naphthalene': 'naphthyl',  # Position-specific: 1-naphthyl, 2-naphthyl
    'anthracene': 'anthryl',
    'phenanthrene': 'phenanthryl',

    # Carbocyclic saturated (cycloalkanes)
    'cyclopropane': 'cyclopropyl',
    'cyclobutane': 'cyclobutyl',
    'cyclopentane': 'cyclopentyl',
    'cyclohexane': 'cyclohexyl',
    'cycloheptane': 'cycloheptyl',
    'cyclooctane': 'cyclooctyl',

    # Heterocyclic aromatic
    'pyridine': 'pyridyl',  # Position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl
    'furan': 'furyl',
    'thiophene': 'thienyl',
    'pyrrole': 'pyrrolyl',
    'imidazole': 'imidazolyl',
    'pyrimidine': 'pyrimidinyl',
    'pyrazine': 'pyrazinyl',

    # Heterocyclic saturated
    'tetrahydrofuran': 'tetrahydrofuryl',
    'pyrrolidine': 'pyrrolidinyl',
    'piperidine': 'piperidinyl',
    'morpholine': 'morpholinyl',
    'piperazine': 'piperazinyl',
}

# Rings that need position-specific names based on attachment point
# Maps ring name -> {ring_position: substituent_name}
POSITION_SPECIFIC_RINGS: Dict[str, Dict[int, str]] = {
    'naphthalene': {
        1: '1-naphthyl',
        2: '2-naphthyl',
    },
    'pyridine': {
        2: '2-pyridyl',
        3: '3-pyridyl',
        4: '4-pyridyl',
    },
}


def identify_ring_system(mol, ring_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Identify ring system name from molecular structure.

    Examines the ring atoms to determine what type of ring system it is
    based on size, aromaticity, and heteroatom composition.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        Ring system name string (e.g., 'benzene', 'cyclohexane', 'pyridine'),
        or None if the ring cannot be identified
    """
    ring_size = len(ring_atoms)
    ring_set = set(ring_atoms)

    # Check aromaticity of all ring atoms
    is_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )

    # Get heteroatom list (symbols of non-carbon atoms)
    heteroatoms = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append(symbol)

    # Sort for consistent comparison
    heteroatoms.sort()

    # === 6-membered rings ===
    if ring_size == 6:
        if is_aromatic:
            if not heteroatoms:
                return 'benzene'
            elif heteroatoms == ['N']:
                return 'pyridine'
            elif heteroatoms == ['N', 'N']:
                # Could be pyrimidine, pyrazine, pyridazine
                # For now, return pyrimidine as common case
                return 'pyrimidine'
        else:
            # Saturated 6-membered
            if not heteroatoms:
                return 'cyclohexane'
            elif heteroatoms == ['O']:
                return 'tetrahydropyran'
            elif heteroatoms == ['N']:
                return 'piperidine'
            elif heteroatoms == ['N', 'O']:
                return 'morpholine'
            elif heteroatoms == ['N', 'N']:
                return 'piperazine'

    # === 5-membered rings ===
    elif ring_size == 5:
        if is_aromatic:
            if not heteroatoms:
                # Aromatic 5-membered all carbon = cyclopentadienyl anion
                # Usually not encountered, but handle gracefully
                return 'cyclopentadiene'
            elif heteroatoms == ['O']:
                return 'furan'
            elif heteroatoms == ['S']:
                return 'thiophene'
            elif heteroatoms == ['N']:
                return 'pyrrole'
            elif heteroatoms == ['N', 'N']:
                return 'imidazole'
        else:
            # Saturated 5-membered
            if not heteroatoms:
                return 'cyclopentane'
            elif heteroatoms == ['O']:
                return 'tetrahydrofuran'
            elif heteroatoms == ['N']:
                return 'pyrrolidine'

    # === 3-membered rings ===
    elif ring_size == 3:
        if not heteroatoms:
            return 'cyclopropane'
        elif heteroatoms == ['O']:
            return 'oxirane'
        elif heteroatoms == ['N']:
            return 'aziridine'

    # === 4-membered rings ===
    elif ring_size == 4:
        if not heteroatoms:
            return 'cyclobutane'
        elif heteroatoms == ['O']:
            return 'oxetane'
        elif heteroatoms == ['N']:
            return 'azetidine'

    # === 7-membered rings ===
    elif ring_size == 7:
        if not heteroatoms:
            if is_aromatic:
                return 'cycloheptatriene'
            else:
                return 'cycloheptane'

    # === 8-membered and larger ===
    elif ring_size == 8 and not heteroatoms:
        return 'cyclooctane'

    # Fallback: generic cycloalkane for all-carbon saturated rings
    if not heteroatoms and not is_aromatic:
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}ane'

    return None


def get_ring_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_point: Optional[int] = None
) -> str:
    """
    Get the substituent name for a ring when it becomes a substituent on a chain.

    This is the main entry point for ring-as-substituent naming.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_point: Optional ring atom index where the ring attaches to chain.
                         Used for position-specific names (e.g., 2-pyridyl vs 4-pyridyl).

    Returns:
        Substituent name string (e.g., 'phenyl', 'cyclohexyl', '2-pyridyl')
    """
    # Identify the ring system
    ring_name = identify_ring_system(mol, ring_atoms)

    if ring_name is None:
        # Unknown ring - generate generic cycloXyl name
        ring_size = len(ring_atoms)
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}yl'

    # Check for position-specific name
    if ring_name in POSITION_SPECIFIC_RINGS and attachment_point is not None:
        # Determine the position in the ring
        ring_position = _get_ring_position_for_attachment(
            mol, ring_atoms, attachment_point, ring_name
        )
        if ring_position in POSITION_SPECIFIC_RINGS[ring_name]:
            return POSITION_SPECIFIC_RINGS[ring_name][ring_position]

    # Look up standard substituent name
    if ring_name in RING_SUBSTITUENT_NAMES:
        return RING_SUBSTITUENT_NAMES[ring_name]

    # Fallback: convert ring name to substituent form
    # Generally: remove 'e' and add 'yl' (benzene -> benzyl, but benzene -> phenyl is special)
    if ring_name.endswith('ane'):
        return ring_name[:-1] + 'yl'  # cyclohexane -> cyclohexanyl (but we have cyclohexyl in dict)
    elif ring_name.endswith('ene'):
        return ring_name[:-1] + 'yl'  # cyclohexene -> cyclohexenyl
    elif ring_name.endswith('ine'):
        return ring_name[:-1] + 'yl'  # pyridine -> pyridinyl

    return ring_name + 'yl'


def _get_ring_position_for_attachment(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
    ring_name: str
) -> Optional[int]:
    """
    Determine the ring position number for an attachment point.

    For position-specific substituents like pyridine (2-pyridyl, 3-pyridyl, 4-pyridyl),
    we need to determine which position the attachment is at based on IUPAC numbering.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_atom: The ring atom that connects to the chain
        ring_name: Name of the ring system

    Returns:
        Ring position number (1-indexed), or None if cannot determine
    """
    # For pyridine: N is at position 1, so we number relative to N
    if ring_name == 'pyridine':
        # Find the nitrogen
        n_idx = None
        for idx in ring_atoms:
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'N':
                n_idx = idx
                break

        if n_idx is None:
            return None

        # Build ring path from N
        ring_path = _build_ring_path_from_start(mol, ring_atoms, n_idx)

        # Find position of attachment atom
        try:
            pos = ring_path.index(attachment_atom)
            return pos + 1  # 1-indexed
        except ValueError:
            return None

    # For naphthalene: uses standard IUPAC peripheral numbering
    # Position 1 is adjacent to fusion, position 2 is farther
    # This is complex; for now, return based on simple heuristics
    if ring_name == 'naphthalene':
        # Simplified: check if atom is alpha (1,4,5,8) or beta (2,3,6,7)
        # This would require more sophisticated analysis
        # For now, return None to use generic naphthyl
        return None

    return None


def _build_ring_path_from_start(
    mol,
    ring_atoms: Tuple[int, ...],
    start_idx: int
) -> List[int]:
    """
    Build an ordered path around the ring starting from a given atom.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        start_idx: Starting atom index

    Returns:
        List of atom indices in order around the ring
    """
    ring_set = set(ring_atoms)
    path = [start_idx]
    visited = {start_idx}

    current = start_idx
    while len(path) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)

        # Find next ring neighbor not yet visited
        next_idx = None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                next_idx = nbr_idx
                break

        if next_idx is None:
            break

        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx

    return path


def get_ring_attachment_locant(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int],
    atom_to_locant: Dict[int, int]
) -> int:
    """
    Find which chain position the ring is attached to.

    The ring connects to the chain via a bond between a ring atom and a chain atom.
    This function finds that chain atom and returns its locant.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain
        atom_to_locant: Mapping from chain atom index to locant (1-indexed)

    Returns:
        Locant (1-indexed position) where the ring attaches to the chain

    Raises:
        ValueError: If no connection found between ring and chain
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    # Find the bond connecting ring to chain
    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Check if neighbor is on the chain
            if nbr_idx in chain_set:
                # Found the connection
                if nbr_idx in atom_to_locant:
                    return atom_to_locant[nbr_idx]

    # If no direct connection found, raise error
    raise ValueError(
        "Could not find connection between ring and chain. "
        f"Ring atoms: {ring_atoms}, Chain atoms: {chain_atoms}"
    )


def get_ring_attachment_atom(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int]
) -> Optional[int]:
    """
    Find the ring atom that attaches to the chain.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain

    Returns:
        Ring atom index that connects to chain, or None if not found
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            if nbr_idx in chain_set:
                return ring_idx

    return None
