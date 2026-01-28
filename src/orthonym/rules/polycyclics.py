"""
Polycyclic aromatic hydrocarbon (PAH) naming rules.

Handles identification and naming of common polycyclic aromatics:
- naphthalene, anthracene, phenanthrene, pyrene
- fluorene, acenaphthene, acenaphthylene, chrysene

IUPAC 2013 Rules (Blue Book Section P-25):
- PAH numbering is FIXED by IUPAC standard
- Use retained names as parent
- Substituents are named with their IUPAC locant position
- Numbering is NOT reoriented based on substituents (unlike benzene)

Key difference from benzene:
- Benzene substituent numbering uses lowest locants
- PAH numbering is fixed to the standard IUPAC orientation
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from rdkit import Chem

from ..data.polycyclic_data import (
    POLYCYCLIC_DATA,
    get_polycyclic_by_smiles,
    is_polycyclic_aromatic,
)
from ..assembly.naming_utils import (
    format_substituent_prefix,
    alpha_sort_key,
    get_multiplier_prefix,
)


def identify_polycyclic(mol) -> Optional[str]:
    """
    Identify if a molecule is a (possibly substituted) polycyclic aromatic.

    This function checks if the molecule's core structure matches a known PAH.
    It uses substructure matching to find PAH cores in substituted molecules.

    Args:
        mol: RDKit Mol object

    Returns:
        PAH name if identified, None otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> identify_polycyclic(mol)
        'naphthalene'
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> identify_polycyclic(mol)
        'naphthalene'
    """
    # First, try exact canonical match (unsubstituted PAH)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    if is_polycyclic_aromatic(canonical_smiles):
        result = get_polycyclic_by_smiles(canonical_smiles)
        if result:
            return result['name']

    # Try substructure matching for substituted PAHs
    # Check from largest to smallest to find the best (largest) match
    pah_by_size = sorted(
        POLYCYCLIC_DATA.items(),
        key=lambda x: x[1]['num_atoms'],
        reverse=True
    )

    for pah_name, pah_data in pah_by_size:
        smarts = pah_data['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Verify that all atoms in the match are aromatic or expected sp3
            # (for fluorene, acenaphthene which have methylene bridges)
            match_atoms = set(matches[0])

            # Count how many atoms in the molecule are in the core
            # vs how many are substituents
            total_atoms = mol.GetNumAtoms()
            core_atoms = len(match_atoms)

            # If the core matches and we have substituents, this is the PAH
            if core_atoms == pah_data['num_atoms']:
                return pah_name

    return None


def get_polycyclic_core_atoms(mol, pah_name: str) -> Optional[Set[int]]:
    """
    Get the atom indices that form the PAH core.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Set of atom indices forming the PAH core, or None if no match
    """
    if pah_name not in POLYCYCLIC_DATA:
        return None

    pah_data = POLYCYCLIC_DATA[pah_name]
    smarts = pah_data['smarts']
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return set(matches[0])

    return None


def get_polycyclic_substituents(mol, pah_name: str) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to a polycyclic aromatic core.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Dict mapping IUPAC locant (1-indexed) to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)

    Note:
        Position mapping for naphthalene is based on IUPAC numbering:
        - Atoms are numbered 1-8 around the periphery
        - Fusion carbons (4a, 8a) are not substituent positions
    """
    core_atoms = get_polycyclic_core_atoms(mol, pah_name)
    if core_atoms is None:
        return {}

    pah_data = POLYCYCLIC_DATA.get(pah_name, {})
    allowed_positions = pah_data.get('substituent_positions', [])

    # Get the SMARTS match for atom ordering
    smarts = pah_data.get('smarts', '')
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return {}

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return {}

    match_atoms = list(matches[0])

    # Build atom index -> IUPAC position mapping
    # This requires understanding the SMARTS match order vs IUPAC numbering
    # For naphthalene c1ccc2ccccc2c1:
    #   SMARTS traverses: c1 c c c2 c c c c c2 c1
    #   IUPAC positions:  1  2 3 4  5 6 7 8 4a 8a (where 4a=4.5 and 8a=8.5 are fusions)
    # But the match order depends on which atom starts first in the SMILES

    atom_to_position = _map_pah_atoms_to_iupac(mol, pah_name, match_atoms)

    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the core
            if nbr_idx in core_atoms:
                continue

            # Identify the substituent
            sub_info = _identify_pah_substituent(mol, nbr_idx, core_atoms)
            if sub_info:
                # Get the IUPAC position for this attachment point
                position = atom_to_position.get(atom_idx)

                # Skip fusion carbons (non-integer positions like 4.5, 8.5)
                if position is None or not isinstance(position, int):
                    continue

                # Only include if it's an allowed substituent position
                if position in allowed_positions or not allowed_positions:
                    substituents[position].append(sub_info)

    return dict(substituents)


def _map_pah_atoms_to_iupac(mol, pah_name: str, match_atoms: List[int]) -> Dict[int, int]:
    """
    Map PAH atom indices to IUPAC position numbers.

    For naphthalene, IUPAC numbering is:
        8  1
       /  \ /
      7    2
      |    |
      6    3
       \  / \
        5  4

    Positions 1, 4, 5, 8 are "alpha" (adjacent to fusion carbons 4a/8a).
    Positions 2, 3, 6, 7 are "beta" (NOT adjacent to fusion carbons).

    Key insight: The alpha/beta classification of an atom is a STRUCTURAL
    property that doesn't change with numbering. A methyl at an alpha position
    can only have locants 1, 4, 5, or 8. A methyl at a beta position can
    only have locants 2, 3, 6, or 7.

    The algorithm:
    1. Identify fusion atoms and classify peripheral atoms as alpha/beta
    2. Build the peripheral ring order by traversing
    3. For naphthalene, assign positions 1,2,3,4 to one ring and 5,6,7,8 to other
    4. Apply lowest-locant rule: positions 1,2 should have substituents
       before positions 8,7 (or 4,5 before 3,6)

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH
        match_atoms: Atom indices from SMARTS match

    Returns:
        Dict mapping atom index to IUPAC position (int for peripheral atoms)
    """
    from collections import defaultdict

    core_set = set(match_atoms)

    # Get ring info to identify fusion atoms
    ri = mol.GetRingInfo()

    # Find atoms shared by multiple rings (fusion atoms)
    atom_ring_count = defaultdict(int)
    for ring in ri.AtomRings():
        for atom_idx in ring:
            if atom_idx in core_set:
                atom_ring_count[atom_idx] += 1

    fusion_atoms = {idx for idx, count in atom_ring_count.items() if count > 1}
    peripheral_atoms = [idx for idx in match_atoms if idx not in fusion_atoms]

    if not peripheral_atoms:
        return {}

    # Build adjacency for atoms in the PAH core
    adjacency = defaultdict(set)
    for atom_idx in match_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in core_set:
                adjacency[atom_idx].add(nbr_idx)

    # Classify peripheral atoms as alpha (adjacent to fusion) or beta
    alpha_atoms = set()
    beta_atoms = set()
    for atom_idx in peripheral_atoms:
        if adjacency[atom_idx] & fusion_atoms:
            alpha_atoms.add(atom_idx)
        else:
            beta_atoms.add(atom_idx)

    # Find substituent atoms (atoms with non-core neighbors)
    substituted_atoms = set()
    for atom_idx in peripheral_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in core_set:
                substituted_atoms.add(atom_idx)
                break

    # For naphthalene, we need to assign:
    # - Alpha atoms get positions 1, 4, 5, 8 (corners)
    # - Beta atoms get positions 2, 3, 6, 7 (sides)
    # The LOWEST locant for an alpha substituent is 1
    # The LOWEST locant for a beta substituent is 2

    # Build the peripheral ring order
    peripheral_order = _build_peripheral_order(
        peripheral_atoms[0], peripheral_atoms, fusion_atoms, adjacency
    )

    if not peripheral_order or len(peripheral_order) != len(peripheral_atoms):
        # Fallback
        return {atom_idx: i + 1 for i, atom_idx in enumerate(peripheral_atoms)}

    # For naphthalene (8 peripheral atoms), the pattern alternates:
    # Position 1 (alpha) - Position 2 (beta) - Position 3 (beta) - Position 4 (alpha) -
    # Position 5 (alpha) - Position 6 (beta) - Position 7 (beta) - Position 8 (alpha)
    #
    # So in the traversal order, if we start from an alpha atom:
    # alpha - beta - beta - alpha - alpha - beta - beta - alpha
    #   1      2      3      4       5      6      7       8

    # Find the best starting point to minimize locants
    n = len(peripheral_order)
    best_mapping = None
    best_locants = None

    # For naphthalene, valid starting points must be alpha atoms (positions 1, 4, 5, 8)
    # and the traversal direction must give alpha-beta-beta-alpha pattern
    for start_idx in range(n):
        start_atom = peripheral_order[start_idx]
        if start_atom not in alpha_atoms:
            continue  # Position 1 must be alpha

        for direction in [1, -1]:
            # Check if this orientation gives valid alpha/beta pattern
            valid = True
            mapping = {}
            expected_alpha = [0, 3, 4, 7]  # Positions 1,4,5,8 (0-indexed: 0,3,4,7)

            for i in range(n):
                actual_idx = (start_idx + i * direction) % n
                atom_idx = peripheral_order[actual_idx]
                mapping[atom_idx] = i + 1

                # Check alpha/beta consistency
                is_alpha = atom_idx in alpha_atoms
                should_be_alpha = i in expected_alpha
                if is_alpha != should_be_alpha:
                    valid = False
                    break

            if not valid:
                continue

            # Get locants for substituted positions
            locants = sorted([mapping[pos] for pos in substituted_atoms if pos in mapping])

            # Compare with best using first-point-of-difference
            if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
                best_locants = locants
                best_mapping = mapping

    return best_mapping if best_mapping else {atom_idx: i + 1 for i, atom_idx in enumerate(peripheral_atoms)}


def _build_peripheral_order(
    start: int,
    peripheral_atoms: List[int],
    fusion_atoms: Set[int],
    adjacency: Dict[int, Set[int]]
) -> List[int]:
    """
    Build an ordered list of peripheral atoms by traversing around the ring system.

    For naphthalene, this produces the 8 peripheral atoms in order around the
    outer edge, skipping the fusion atoms.
    """
    peripheral_set = set(peripheral_atoms)
    order = []
    visited = set()
    current = start

    while len(order) < len(peripheral_atoms):
        if current in visited:
            break

        visited.add(current)
        order.append(current)

        # Find next unvisited peripheral neighbor
        found_next = False
        for nbr in adjacency[current]:
            if nbr in peripheral_set and nbr not in visited:
                current = nbr
                found_next = True
                break

        if not found_next:
            # Need to go through a fusion atom
            for nbr in adjacency[current]:
                if nbr in fusion_atoms:
                    for next_nbr in adjacency[nbr]:
                        if next_nbr in peripheral_set and next_nbr not in visited:
                            current = next_nbr
                            found_next = True
                            break
                    if found_next:
                        break

        if not found_next:
            break

    return order


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


def _identify_pah_substituent(mol, start_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent attached to the PAH core.

    Similar to benzene substituent identification but excludes core atoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the core
        core_atoms: Set of atom indices forming the PAH core

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens - single atom substituents
    halogen_names = {
        'F': 'fluoro',
        'Cl': 'chloro',
        'Br': 'bromo',
        'I': 'iodo',
    }
    if symbol in halogen_names:
        return {
            'name': halogen_names[symbol],
            'atoms': [start_idx]
        }

    # Carbon-based groups (alkyl)
    if symbol == 'C':
        return _identify_pah_alkyl_group(mol, start_idx, core_atoms)

    # Nitrogen groups
    if symbol == 'N':
        return _identify_pah_nitrogen_group(mol, start_idx, core_atoms)

    # Oxygen groups
    if symbol == 'O':
        return _identify_pah_oxygen_group(mol, start_idx, core_atoms)

    return None


def _identify_pah_alkyl_group(mol, start_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent on PAH.

    Uses BFS to find all connected carbons and determines name from carbon count.
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

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in core_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains heteroatom - not a simple alkyl
            return None

    # Get alkyl name from carbon count
    alkyl_names = {
        1: 'methyl',
        2: 'ethyl',
        3: 'propyl',
        4: 'butyl',
        5: 'pentyl',
        6: 'hexyl',
        7: 'heptyl',
        8: 'octyl',
        9: 'nonyl',
        10: 'decyl',
    }

    if carbon_count in alkyl_names:
        return {
            'name': alkyl_names[carbon_count],
            'atoms': all_atoms
        }

    return None


def _identify_pah_nitrogen_group(mol, n_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent on PAH."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding core)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in core_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    # Simple amino (-NH2)
    h_count = n_atom.GetTotalNumHs()
    if h_count == 2 and len(neighbors) == 0:
        return {'name': 'amino', 'atoms': [n_idx]}

    return None


def _identify_pah_oxygen_group(mol, o_idx: int, core_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent on PAH."""
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding core)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in core_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    return None


def name_substituted_polycyclic(
    mol,
    pah_name: str,
    substituents: Dict[int, List[Dict]]
) -> str:
    """
    Generate IUPAC name for a substituted polycyclic aromatic.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH parent (e.g., 'naphthalene')
        substituents: Dict from get_polycyclic_substituents

    Returns:
        IUPAC name string (e.g., '2-methylnaphthalene')

    IUPAC Rules:
    - Locants are FIXED to IUPAC standard numbering
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-parent
    """
    if not substituents:
        return pah_name

    # Group substituents by name
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for position, sub_list in substituents.items():
        for sub_info in sub_list:
            name = sub_info['name']
            substituent_groups[name].append(position)

    # Sort locants within each group
    for name in substituent_groups:
        substituent_groups[name].sort()

    # Count total substituents
    total_substituents = sum(len(locs) for locs in substituent_groups.values())

    # For monosubstituted PAHs, we still include the locant (unlike benzene)
    # because PAH positions are not equivalent
    is_monosubstituted = total_substituents == 1

    # Build prefix strings, sorted alphabetically by substituent name
    prefixes = []
    for name in sorted(substituent_groups.keys(), key=alpha_sort_key):
        locants = substituent_groups[name]
        count = len(locants)
        prefix_str = format_substituent_prefix(name, locants, count)
        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_pah_prefixes(prefixes)

    # Build final name
    return f"{prefix_part}{pah_name}"


def _join_pah_prefixes(prefixes: List[str]) -> str:
    """
    Join PAH substituent prefixes with proper hyphenation.

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
