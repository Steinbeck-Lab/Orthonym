"""
Fused ring system detection and naming.

Handles:
- Classification of fused ring systems (ortho-fused, ortho-peri-fused, bridged-fused)
- Naming of fused heterocycles with retained names priority
- Substituent detection and locant assignment for fused systems
- N-substitution handling for fused heterocycles

IUPAC 2013 Rules for fused systems:
- Ortho-fused: rings share exactly one bond (2 atoms)
- Ortho-peri-fused: at least one ring shares atoms with 3+ other rings
- ALWAYS check retained names FIRST before systematic naming
- Tautomer locants (1H-, 2H-, 9H-) must be preserved in names
- N-substitution uses N-locant format (N-methyl, not 1-methyl)

Reference: IUPAC 2013 Blue Book, Section P-25 (Fused Ring Systems)
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from rdkit import Chem

from ..data.fused_heterocycles import (
    get_fused_heterocycle_name,
    match_fused_heterocycle_core,
    FUSED_HETEROCYCLE_DATA,
)
from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    is_aromatic_ring,
    is_heterocyclic,
)
from ..assembly.naming_utils import (
    get_alkyl_name,
    alpha_sort_key,
    get_multiplier_prefix,
)


# Simple multiplicative prefixes for substituent naming
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def get_shared_atoms(mol, ring1: Tuple[int, ...], ring2: Tuple[int, ...]) -> Set[int]:
    """
    Get atoms shared between two rings.

    Args:
        mol: RDKit Mol object
        ring1: Tuple of atom indices in first ring
        ring2: Tuple of atom indices in second ring

    Returns:
        Set of atom indices shared by both rings

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> ri = mol.GetRingInfo()
        >>> rings = ri.AtomRings()
        >>> shared = get_shared_atoms(mol, rings[0], rings[1])
        >>> len(shared)  # 2 atoms shared in ortho-fused system
        2
    """
    set1 = set(ring1)
    set2 = set(ring2)
    return set1 & set2


def classify_fused_system(mol) -> str:
    """
    Classify a fused ring system by its fusion type.

    Classification:
    - 'ortho-fused': All ring pairs share exactly 2 atoms (one edge)
    - 'ortho-peri-fused': At least one ring shares atoms with 3+ other rings
    - 'bridged-fused': Bridges exist across fused system (like norbornane)
    - 'not-fused': Rings share 0-1 atoms (isolated or spiro)

    Args:
        mol: RDKit Mol object

    Returns:
        Classification string

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> classify_fused_system(mol)
        'ortho-fused'
        >>> mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
        >>> classify_fused_system(mol)
        'ortho-peri-fused'
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return 'not-fused'

    # Track how many rings each ring shares atoms with
    # and how many atoms each pair of rings shares
    ring_connection_count = defaultdict(int)  # ring_idx -> count of connected rings

    # Track atoms shared between each pair of rings
    fused_pairs = []  # List of (ring_i, ring_j, shared_count)

    for i, ring1 in enumerate(atom_rings):
        for j, ring2 in enumerate(atom_rings):
            if i >= j:
                continue

            shared = get_shared_atoms(mol, ring1, ring2)
            shared_count = len(shared)

            if shared_count >= 2:
                # These rings are fused (share at least one edge)
                fused_pairs.append((i, j, shared_count))
                ring_connection_count[i] += 1
                ring_connection_count[j] += 1
            elif shared_count == 1:
                # Spiro connection - only one shared atom
                pass  # Don't count as fused

    if not fused_pairs:
        return 'not-fused'

    # Check for ortho-peri-fused: any ring connected to 3+ other rings
    for ring_idx, count in ring_connection_count.items():
        if count >= 3:
            return 'ortho-peri-fused'

    # Check for bridged-fused: any pair shares more than 2 atoms
    # (indicating a bridge across the ring system)
    for i, j, shared_count in fused_pairs:
        if shared_count > 2:
            # More than 2 shared atoms suggests bridging
            return 'bridged-fused'

    # All fused pairs share exactly 2 atoms - ortho-fused
    return 'ortho-fused'


def is_fused_bicyclic(mol) -> bool:
    """
    Check if molecule is a fused bicyclic system (exactly 2 rings sharing one edge).

    This distinguishes fused bicyclics from bridged bicyclics (like norbornane)
    which have bridgehead atoms shared by more than 2 rings conceptually.

    Args:
        mol: RDKit Mol object

    Returns:
        True if exactly 2 rings sharing exactly 2 atoms

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> is_fused_bicyclic(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')  # carbazole (tricyclic)
        >>> is_fused_bicyclic(mol)
        False
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) != 2:
        return False

    ring1, ring2 = atom_rings[0], atom_rings[1]
    shared = get_shared_atoms(mol, ring1, ring2)

    # Exactly 2 shared atoms = one shared edge = ortho-fused bicyclic
    return len(shared) == 2


def name_fused_heterocycle(mol) -> Optional[str]:
    """
    Generate IUPAC name for a fused heterocycle.

    Naming priority:
    1. Check retained names FIRST (indole, quinoline, carbazole, etc.)
    2. Add tautomer locant if present (1H-indole)
    3. For substituted: find substituents and add prefixes
    4. Handle N-substitution specially (N-methyl, not 1-methyl)

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not a recognized fused heterocycle

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> name_fused_heterocycle(mol)
        '1H-indole'
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        >>> name_fused_heterocycle(mol)
        '5-methyl-1H-indole'
    """
    if mol is None:
        return None

    # First try exact match for unsubstituted fused heterocycle
    result = get_fused_heterocycle_name(mol)
    if result:
        name, tautomer_locant = result
        return name  # Name already includes tautomer locant if present

    # Try substructure matching for substituted fused heterocycles
    core_result = match_fused_heterocycle_core(mol)
    if core_result is None:
        return None

    core_name, atom_mapping, _core_smiles = core_result

    # Find substituents on the core
    substituents = get_fused_heterocycle_substituents(mol, atom_mapping)

    if not substituents:
        return core_name

    # Build the substituted name
    return _assemble_fused_heterocycle_name(mol, core_name, substituents, atom_mapping)


def get_fused_heterocycle_substituents(
    mol,
    core_match: Dict[int, Any]
) -> Dict:
    """
    Find substituents on a fused heterocycle core.

    Identifies atoms not in the core match as potential substituents,
    maps them to IUPAC locants using core numbering, and tracks
    N-substitution separately.

    Args:
        mol: RDKit Mol object
        core_match: Dict mapping mol atom indices to IUPAC locants (int or str like '3a')

    Returns:
        Dict with:
        - 'c_substituents': Dict[str, List[int]] - C-substituent name -> locants
        - 'n_substituents': Dict[str, int] - N-substituent name -> count
        - 'other': List[Dict] - Other substituents (halogens, etc.)

    Examples:
        >>> mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        >>> core_match = match_fused_heterocycle_core(mol)[1]
        >>> subs = get_fused_heterocycle_substituents(mol, core_match)
        >>> 'methyl' in subs['c_substituents']
        True
    """
    core_atoms = set(core_match.keys())

    result = {
        'c_substituents': defaultdict(list),  # name -> list of locants
        'n_substituents': defaultdict(int),   # name -> count
        'oxo_substituents': [],   # list of locants for =O (suffix: -one)
        'amino_substituents': [], # list of locants for -NH2 (suffix: -amine)
        'other': [],  # For halogens, etc.
    }

    # Find which core atoms have substituents
    for core_atom_idx, locant in core_match.items():
        core_atom = mol.GetAtomWithIdx(core_atom_idx)
        is_nitrogen = core_atom.GetSymbol() == 'N'

        for neighbor in core_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the core
            if nbr_idx in core_atoms:
                continue

            # Identify the substituent
            sub_info = _identify_fused_substituent(mol, nbr_idx, core_atoms)
            if sub_info is None:
                continue

            sub_name = sub_info.get('name')
            sub_type = sub_info.get('type', 'alkyl')

            if sub_type == 'alkyl':
                if is_nitrogen:
                    # N-substitution
                    result['n_substituents'][sub_name] += 1
                else:
                    # C-substitution
                    result['c_substituents'][sub_name].append(locant)
            elif sub_type == 'oxo':
                # Oxo group (=O) - use suffix form (-one)
                result['oxo_substituents'].append(locant)
            elif sub_type == 'functional' and sub_name == 'amino':
                # Amino group (-NH2) - use suffix form (-amine)
                result['amino_substituents'].append(locant)
            else:
                # Halogen or other
                sub_info['locant'] = locant
                sub_info['is_on_nitrogen'] = is_nitrogen
                result['other'].append(sub_info)

    # Sort C-substituent locants (handle mixed int/str like 5, '3a', '7a')
    def _locant_sort_key(loc):
        """Sort key for IUPAC locants - handles int (5) and str ('3a')."""
        if isinstance(loc, str):
            # Parse '3a' -> (3, 'a'), '7a' -> (7, 'a')
            if loc and loc[-1].isalpha():
                return (int(loc[:-1]), loc[-1])
            return (int(loc), '')
        return (loc, '')

    for name in result['c_substituents']:
        result['c_substituents'][name].sort(key=_locant_sort_key)

    # Sort oxo and amino locants
    result['oxo_substituents'].sort(key=_locant_sort_key)
    result['amino_substituents'].sort(key=_locant_sort_key)

    return dict(result)


def _identify_fused_substituent(
    mol,
    start_idx: int,
    excluded: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Identify a substituent attached to a fused ring core.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        excluded: Set of atom indices to exclude (core atoms)

    Returns:
        Dict with 'name', 'type', 'atoms', or None if unrecognized
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens
    halogen_names = {
        'F': 'fluoro',
        'Cl': 'chloro',
        'Br': 'bromo',
        'I': 'iodo',
    }
    if symbol in halogen_names:
        return {
            'name': halogen_names[symbol],
            'type': 'halogen',
            'atoms': [start_idx]
        }

    # Carbon-based (alkyl) groups
    if symbol == 'C':
        return _identify_alkyl_substituent(mol, start_idx, excluded)

    # Oxygen groups (oxo C=O, hydroxyl -OH)
    if symbol == 'O':
        h_count = start_atom.GetTotalNumHs()
        neighbors_outside_core = [n for n in start_atom.GetNeighbors() if n.GetIdx() not in excluded]

        # Check for oxo group (=O double-bonded to ring carbon)
        if h_count == 0 and len(neighbors_outside_core) == 0:
            # Oxo group: O with no H, double-bonded to core carbon
            for bond in start_atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(start_idx)
                if other_idx in excluded:  # Bond to core atom
                    other_atom = mol.GetAtomWithIdx(other_idx)
                    if other_atom.GetSymbol() == 'C' and bond.GetBondTypeAsDouble() == 2.0:
                        return {
                            'name': 'oxo',
                            'type': 'oxo',  # Special type for suffix handling
                            'atoms': [start_idx],
                            'bond_type': 'double'
                        }

        # Hydroxyl group (-OH)
        if h_count == 1 and len(neighbors_outside_core) == 0:
            return {
                'name': 'hydroxy',
                'type': 'functional',
                'atoms': [start_idx]
            }

    # Nitrogen groups (amino, etc.)
    if symbol == 'N':
        h_count = start_atom.GetTotalNumHs()
        neighbors = [n for n in start_atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 2 and len(neighbors) == 0:
            return {
                'name': 'amino',
                'type': 'functional',
                'atoms': [start_idx]
            }

    return None


def _identify_alkyl_substituent(
    mol,
    start_idx: int,
    excluded: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Identify an alkyl substituent using BFS.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index
        excluded: Set of atom indices to exclude

    Returns:
        Dict with 'name', 'type', 'atoms', or None
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = [start_idx]
    all_atoms = []
    carbon_count = 0

    while queue:
        current_idx = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if pure alkyl (only C and H)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            return None  # Contains heteroatom, not simple alkyl

    if carbon_count == 0:
        return None

    # Get alkyl name
    try:
        alkyl_name = get_alkyl_name(carbon_count)
        return {
            'name': alkyl_name,
            'type': 'alkyl',
            'atoms': all_atoms
        }
    except ValueError:
        return None


def _assemble_fused_heterocycle_name(
    mol,
    core_name: str,
    substituents: Dict,
    atom_mapping: Dict[int, int]
) -> str:
    """
    Assemble the complete name for a substituted fused heterocycle.

    Args:
        mol: RDKit Mol object
        core_name: Base name (e.g., '1H-indole')
        substituents: Dict from get_fused_heterocycle_substituents
        atom_mapping: Dict mapping mol atom indices to IUPAC locants

    Returns:
        Complete IUPAC name string
    """
    prefix_parts = []

    # Handle N-substituents
    for name, count in substituents['n_substituents'].items():
        prefix = _format_n_prefix(name, count)
        prefix_parts.append((prefix, name))

    # Handle C-substituents
    for name, locants in substituents['c_substituents'].items():
        count = len(locants)
        prefix = _format_c_prefix(name, locants, count)
        prefix_parts.append((prefix, name))

    # Handle other substituents (halogens, etc.)
    other_groups = defaultdict(list)
    for sub in substituents['other']:
        name = sub['name']
        locant = sub['locant']
        other_groups[name].append(locant)

    for name, locants in other_groups.items():
        locants.sort()
        count = len(locants)
        prefix = _format_c_prefix(name, locants, count)
        prefix_parts.append((prefix, name))

    # Sort alphabetically by base name
    prefix_parts.sort(key=lambda x: alpha_sort_key(x[1]))

    # Join prefixes
    prefix_str = _join_fused_prefixes([p[0] for p in prefix_parts])

    return f"{prefix_str}{core_name}"


def _format_n_prefix(name: str, count: int) -> str:
    """Format an N-substituent prefix (N-methyl, N,N-dimethyl)."""
    if count == 1:
        return f"N-{name}-"
    else:
        n_locants = ",".join(["N"] * count)
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{n_locants}-{multiplier}{name}-"


def _format_c_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a C-substituent prefix with numeric locants."""
    locant_str = ",".join(str(loc) for loc in locants)
    if count == 1:
        return f"{locant_str}-{name}-"
    else:
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{locant_str}-{multiplier}{name}-"


def _join_fused_prefixes(prefixes: List[str]) -> str:
    """Join fused ring substituent prefixes."""
    if not prefixes:
        return ""

    # Each prefix already ends with '-', just concatenate
    result = ""
    for prefix in prefixes:
        # Remove trailing hyphen if present, we'll manage hyphens ourselves
        p = prefix.rstrip('-')
        if result:
            # Check if hyphen needed between prefixes
            last_char = result[-1]
            first_char = p[0]
            if last_char.isalpha() and first_char.isdigit():
                result += "-"
            elif last_char.isalpha() and first_char == 'N':
                result += "-"
        result += p

    # Add final hyphen before parent name
    return result + "-"


def name_ortho_fused_bicyclic(mol) -> Optional[str]:
    """
    Generate name for an ortho-fused bicyclic system without a retained name.

    This is a fallback for carbocyclic ortho-fused systems not covered by
    polycyclic_data. Most common fused systems should have retained names.

    For systematic naming, uses fusion descriptors like benzo[x]parent.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not an ortho-fused bicyclic

    Examples:
        >>> # For systems without retained names, would generate systematic names
        >>> # Most common ones (naphthalene, indole) have retained names
    """
    if not is_fused_bicyclic(mol):
        return None

    # First check if it's a known fused heterocycle
    heterocycle_name = name_fused_heterocycle(mol)
    if heterocycle_name:
        return heterocycle_name

    # For carbocyclic ortho-fused systems, check polycyclic data
    # (naphthalene, etc.) - this is handled by polycyclics module
    # Return None to indicate this module doesn't handle it
    return None


def is_fused_aromatic_system(mol) -> bool:
    """
    Check if molecule contains a fused aromatic ring system.

    A fused aromatic system has 2+ aromatic rings sharing edges.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused aromatic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        >>> is_fused_aromatic_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        >>> is_fused_aromatic_system(mol)
        False
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Find aromatic rings
    aromatic_rings = [ring for ring in atom_rings if is_aromatic_ring(mol, ring)]

    if len(aromatic_rings) < 2:
        return False

    # Check if any pair of aromatic rings are fused (share 2+ atoms)
    for i, ring1 in enumerate(aromatic_rings):
        for j, ring2 in enumerate(aromatic_rings):
            if i >= j:
                continue
            shared = get_shared_atoms(mol, ring1, ring2)
            if len(shared) >= 2:
                return True

    return False


def is_fused_heterocyclic_system(mol) -> bool:
    """
    Check if molecule contains a fused heterocyclic ring system.

    A fused heterocyclic system has at least one heterocyclic ring
    fused with another ring.

    Args:
        mol: RDKit Mol object

    Returns:
        True if fused heterocyclic system detected

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        >>> is_fused_heterocyclic_system(mol)
        True
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return False

    # Find heterocyclic rings
    heterocyclic_rings = []
    for ring in atom_rings:
        if is_heterocyclic(mol, ring):
            heterocyclic_rings.append(ring)

    if not heterocyclic_rings:
        return False

    # Check if any heterocyclic ring is fused with another ring
    for hetero_ring in heterocyclic_rings:
        for other_ring in atom_rings:
            if other_ring == hetero_ring:
                continue
            shared = get_shared_atoms(mol, hetero_ring, other_ring)
            if len(shared) >= 2:
                return True

    return False
