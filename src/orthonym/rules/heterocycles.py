"""
Heterocycle detection, classification, ring numbering, and naming according to IUPAC 2013.

Handles:
- Heterocycle classification (ring size, aromaticity, heteroatom types)
- Ring numbering starting at highest-priority heteroatom
- Direction selection to minimize locants for other heteroatoms
- First-point-of-difference rule for direction ties
- Retained name lookup for common heterocycles
- Hantzsch-Widman (HW) systematic name generation

IUPAC 2013 Rules for heterocycle numbering:
- Position 1 goes to highest-priority heteroatom (O > S > Se > Te > N > P > ...)
- Numbering direction chosen to give lowest locants to other heteroatoms
- First-point-of-difference comparison (not sum of locants)

Naming priority:
1. Check retained names FIRST (pyridine, furan, morpholine, etc.)
2. Fall back to HW systematic naming if no retained name

Reference: IUPAC 2013 Blue Book, Section P-22 (Heterocycles)
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

from rdkit import Chem

from ..data.hw_heteroatoms import HETEROATOM_PRIORITY, get_heteroatom_priority, get_hw_prefix
from ..data.hw_stems import get_hw_stem
from ..data.retained_names import get_retained_name
from ..perception.rings import (
    get_ring_heteroatoms,
    is_aromatic_ring,
    is_saturated_ring,
)


# Simple multiplicative prefixes for HW naming
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def classify_heterocycle(mol, ring_atoms) -> Dict:
    """
    Classify a heterocyclic ring by its structural features.

    Returns detailed information about the ring for naming purposes:
    - Ring size (number of atoms)
    - Heteroatom identity and positions
    - Aromaticity
    - Saturation status
    - Dominant heteroatom (highest priority)

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Dict with:
        - ring_size: int (3-10 typically)
        - heteroatoms: List[Tuple[int, str]] - (atom_idx, element_symbol)
        - is_aromatic: bool
        - is_saturated: bool (no double bonds within ring)
        - num_heteroatoms: int
        - dominant_heteroatom: str (highest priority element in ring)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> info = classify_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])
        >>> info['ring_size']
        6
        >>> info['is_aromatic']
        True
        >>> info['dominant_heteroatom']
        'N'
    """
    ring_list = list(ring_atoms)
    ring_size = len(ring_list)

    # Get heteroatoms with their atom indices (not ring positions)
    # get_ring_heteroatoms returns (position_in_ring, element) but we need atom indices
    heteroatoms = []
    for idx in ring_list:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    # Determine aromaticity and saturation
    # Note: is_saturated_ring checks for explicit double bonds, but aromatic
    # rings have delocalized bonding. If a ring is aromatic, it's NOT saturated.
    is_arom = is_aromatic_ring(mol, ring_atoms)
    is_sat = is_saturated_ring(mol, ring_atoms) and not is_arom

    # Find dominant (highest priority) heteroatom
    dominant = None
    if heteroatoms:
        # Sort by priority (lower number = higher priority)
        sorted_heteroatoms = sorted(
            heteroatoms,
            key=lambda x: get_heteroatom_priority(x[1])
        )
        dominant = sorted_heteroatoms[0][1]

    return {
        'ring_size': ring_size,
        'heteroatoms': heteroatoms,
        'is_aromatic': is_arom,
        'is_saturated': is_sat,
        'num_heteroatoms': len(heteroatoms),
        'dominant_heteroatom': dominant,
    }


def number_heterocycle_ring(
    mol,
    ring_atoms,
    heteroatoms: Optional[List[Tuple[int, str]]] = None
) -> List[int]:
    """
    Reorder ring atoms for IUPAC numbering of heterocycles.

    Position 1 is assigned to the highest-priority heteroatom.
    Numbering direction is chosen to give lowest locants to other heteroatoms,
    using first-point-of-difference comparison.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        heteroatoms: Optional list of (atom_idx, element) tuples.
                     If None, will be computed from ring_atoms.

    Returns:
        List of ring atom indices reordered for IUPAC numbering.
        Position 1 (index 0) is the highest-priority heteroatom.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented = number_heterocycle_ring(mol, ring)
        >>> # N atom should be at position 1 (index 0)
        >>> mol.GetAtomWithIdx(oriented[0]).GetSymbol()
        'N'
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)

    if n == 0:
        return []

    # Get heteroatoms if not provided
    if heteroatoms is None:
        heteroatoms = []
        for idx in ring_list:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if symbol != 'C':
                heteroatoms.append((idx, symbol))

    if not heteroatoms:
        # No heteroatoms - not a true heterocycle, return as-is
        return ring_list

    # Find highest-priority heteroatom for position 1
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: get_heteroatom_priority(x[1])
    )
    start_idx = sorted_heteroatoms[0][0]  # Atom index of highest priority

    # Find position of start atom in ring list
    start_pos = ring_list.index(start_idx)

    # If only one heteroatom, direction doesn't matter
    if len(heteroatoms) == 1:
        # Rotate so start atom is first, arbitrary direction
        return _rotate_ring(ring_list, start_pos, 1)

    # Multiple heteroatoms - try both directions and pick lowest locants
    cw = _rotate_ring(ring_list, start_pos, 1)   # Clockwise
    ccw = _rotate_ring(ring_list, start_pos, -1)  # Counterclockwise

    # Get locants for OTHER heteroatoms (not position 1) in each direction
    cw_locants = _get_other_heteroatom_locants(cw, heteroatoms, start_idx)
    ccw_locants = _get_other_heteroatom_locants(ccw, heteroatoms, start_idx)

    # Compare using first-point-of-difference
    comparison = _compare_locant_sets(cw_locants, ccw_locants)

    if comparison <= 0:
        return cw
    else:
        return ccw


def orient_heterocycle(mol, ring_atoms) -> Tuple[List[int], Dict[int, int]]:
    """
    Convenience function combining classification and numbering.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Tuple of:
        - oriented_ring: List of atom indices in IUPAC numbering order
        - atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented, mapping = orient_heterocycle(mol, ring)
        >>> # N is at locant 1
        >>> n_idx = [i for i in oriented if mol.GetAtomWithIdx(i).GetSymbol() == 'N'][0]
        >>> mapping[n_idx]
        1
    """
    # Classify to get heteroatoms
    info = classify_heterocycle(mol, ring_atoms)
    heteroatoms = info['heteroatoms']

    # Number the ring
    oriented = number_heterocycle_ring(mol, ring_atoms, heteroatoms)

    # Build atom-to-locant mapping
    atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(oriented, 1)}

    return oriented, atom_to_locant


def orient_heterocycle_with_substituents(
    mol,
    ring_atoms,
    substituent_positions: Optional[Set[int]] = None
) -> Tuple[List[int], Dict[int, int]]:
    """
    Orient heterocycle considering both heteroatoms AND substituent positions.

    IUPAC Rule: For substituted heterocycles, after fixing the heteroatom at
    position 1, choose the numbering direction that gives lowest locants to:
    1. Other heteroatoms (if any)
    2. Substituents (if heteroatom locants are equal)

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        substituent_positions: Set of ring atom indices that have substituents.
                               If None, will be detected from the molecule.

    Returns:
        Tuple of:
        - oriented_ring: List of atom indices in IUPAC numbering order
        - atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Examples:
        >>> mol = Chem.MolFromSmiles('Cc1ccccn1')  # 2-methylpyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> # Find which ring atom has the methyl substituent
        >>> sub_positions = {idx for idx in ring if _has_substituent(mol, idx, ring)}
        >>> oriented, mapping = orient_heterocycle_with_substituents(mol, ring, sub_positions)
        >>> # Should give methyl the lowest possible locant (2, not 6)
    """
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)
    n = len(ring_list)

    if n == 0:
        return [], {}

    # Detect substituent positions if not provided
    if substituent_positions is None:
        substituent_positions = set()
        for idx in ring_list:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    substituent_positions.add(idx)
                    break

    # Get heteroatoms
    heteroatoms = []
    for idx in ring_list:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    if not heteroatoms:
        # No heteroatoms - return as-is
        atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(ring_list, 1)}
        return ring_list, atom_to_locant

    # Find highest-priority heteroatom for position 1
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: get_heteroatom_priority(x[1])
    )
    start_idx = sorted_heteroatoms[0][0]
    start_pos = ring_list.index(start_idx)

    # Try both directions
    cw = _rotate_ring(ring_list, start_pos, 1)
    ccw = _rotate_ring(ring_list, start_pos, -1)

    # Build position lookups
    cw_map = {atom_idx: pos for pos, atom_idx in enumerate(cw, 1)}
    ccw_map = {atom_idx: pos for pos, atom_idx in enumerate(ccw, 1)}

    # Compare by heteroatom locants first (excluding position 1)
    cw_hetero_locants = sorted([cw_map[idx] for idx, _ in heteroatoms if idx != start_idx])
    ccw_hetero_locants = sorted([ccw_map[idx] for idx, _ in heteroatoms if idx != start_idx])

    hetero_comparison = _compare_locant_sets(cw_hetero_locants, ccw_hetero_locants)

    if hetero_comparison < 0:
        oriented = cw
    elif hetero_comparison > 0:
        oriented = ccw
    else:
        # Heteroatom locants are equal - compare substituent locants
        cw_sub_locants = sorted([cw_map[idx] for idx in substituent_positions if idx in cw_map])
        ccw_sub_locants = sorted([ccw_map[idx] for idx in substituent_positions if idx in ccw_map])

        sub_comparison = _compare_locant_sets(cw_sub_locants, ccw_sub_locants)

        if sub_comparison <= 0:
            oriented = cw
        else:
            oriented = ccw

    atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(oriented, 1)}
    return oriented, atom_to_locant


def get_heteroatom_locants(oriented_ring: List[int], mol) -> List[Tuple[int, str]]:
    """
    Get locants and element symbols for all heteroatoms in an oriented ring.

    Args:
        oriented_ring: Ring atoms already in IUPAC numbering order
                       (position 1 at index 0)
        mol: RDKit Mol object

    Returns:
        List of (locant, element_symbol) tuples for each heteroatom,
        sorted by locant.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ncnc1')  # pyrimidine-like
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented, _ = orient_heterocycle(mol, ring)
        >>> locants = get_heteroatom_locants(oriented, mol)
        >>> # Returns something like [(1, 'N'), (3, 'N')]
    """
    result = []

    for locant, atom_idx in enumerate(oriented_ring, 1):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            result.append((locant, symbol))

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotate_ring(ring_list: List[int], start_pos: int, direction: int) -> List[int]:
    """
    Rotate a ring so that the atom at start_pos becomes first.

    Args:
        ring_list: List of atom indices
        start_pos: Index in ring_list to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos atom first
    """
    n = len(ring_list)
    if n == 0:
        return []

    result = []
    for i in range(n):
        idx = (start_pos + i * direction) % n
        result.append(ring_list[idx])

    return result


def _get_other_heteroatom_locants(
    oriented_ring: List[int],
    heteroatoms: List[Tuple[int, str]],
    start_idx: int
) -> List[int]:
    """
    Get locants for heteroatoms other than the one at position 1.

    Args:
        oriented_ring: Ring atoms in numbering order (position 1 at index 0)
        heteroatoms: List of (atom_idx, element) tuples
        start_idx: Atom index of the heteroatom at position 1 (to exclude)

    Returns:
        Sorted list of locants for other heteroatoms
    """
    # Build position lookup
    pos_map = {atom_idx: locant for locant, atom_idx in enumerate(oriented_ring, 1)}

    locants = []
    for atom_idx, _ in heteroatoms:
        if atom_idx != start_idx and atom_idx in pos_map:
            locants.append(pos_map[atom_idx])

    return sorted(locants)


def _compare_locant_sets(set_a: List[int], set_b: List[int]) -> int:
    """
    Compare two locant sets using first-point-of-difference rule.

    Returns:
        -1 if set_a is preferred (lower at first difference)
         0 if sets are equal
         1 if set_b is preferred
    """
    a_sorted = sorted(set_a)
    b_sorted = sorted(set_b)

    for a, b in zip(a_sorted, b_sorted):
        if a < b:
            return -1
        if a > b:
            return 1

    # All compared elements equal - shorter set wins
    if len(a_sorted) < len(b_sorted):
        return -1
    if len(a_sorted) > len(b_sorted):
        return 1

    return 0


# ---------------------------------------------------------------------------
# Heterocycle naming functions
# ---------------------------------------------------------------------------


def get_ring_canonical_smiles(mol, ring_atoms) -> str:
    """
    Extract a ring as canonical SMILES for retained name lookup.

    Creates a new molecule containing only the ring atoms and their bonds,
    then returns the canonical SMILES representation.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Canonical SMILES string for the ring

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> get_ring_canonical_smiles(mol, ring)
        'c1ccncc1'
    """
    ring_set = set(ring_atoms)

    # Use MolFragmentToSmiles to extract just the ring
    # This handles aromaticity correctly
    ring_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(ring_atoms))

    # Canonicalize the SMILES
    ring_mol = Chem.MolFromSmiles(ring_smiles)
    if ring_mol:
        return Chem.MolToSmiles(ring_mol, canonical=True)

    return ring_smiles


def build_hw_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool,
    is_aromatic: bool
) -> str:
    """
    Build Hantzsch-Widman systematic name for a heterocycle.

    Assembles a systematic HW name from heteroatom prefixes and ring stem:
    1. Group heteroatoms by element
    2. Order by IUPAC priority (O > S > N > ...)
    3. Add multipliers and locants for multiple same heteroatoms
    4. Get stem based on ring size and saturation
    5. Apply 'a' elision (drop terminal 'a' before vowel stem)

    Args:
        heteroatoms: List of (locant, element) tuples for heteroatoms in numbered order
        ring_size: Ring size (3-10)
        is_saturated: True if fully saturated
        is_aromatic: True if aromatic (overrides is_saturated for stem selection)

    Returns:
        HW systematic name (e.g., 'oxolane', '1,3-dioxolane', 'azine')

    Examples:
        >>> build_hw_name([(1, 'O')], 5, True, False)
        'oxolane'
        >>> build_hw_name([(1, 'O'), (3, 'O')], 5, True, False)
        '1,3-dioxolane'
        >>> build_hw_name([(1, 'N')], 6, False, True)
        'azine'
    """
    if not heteroatoms:
        return ""

    # Group heteroatoms by element: {element: [locants]}
    element_locants: Dict[str, List[int]] = {}
    for locant, elem in heteroatoms:
        if elem not in element_locants:
            element_locants[elem] = []
        element_locants[elem].append(locant)

    # Build prefix parts, ordered by IUPAC priority
    prefix_parts = []
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    for elem in elements_by_priority:
        hw_prefix = get_hw_prefix(elem)
        if not hw_prefix:
            continue

        locants = sorted(element_locants[elem])
        count = len(locants)

        if count > 1:
            # Multiple same heteroatoms: add locants and multiplier
            locant_str = ','.join(str(loc) for loc in locants)
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f"{locant_str}-{multiplier}{hw_prefix}")
        else:
            # Single heteroatom: just the prefix
            prefix_parts.append(hw_prefix)

    prefix = ''.join(prefix_parts)

    # Determine saturation for stem lookup
    # Aromatic = unsaturated for HW naming purposes
    saturated_for_stem = is_saturated and not is_aromatic

    # Get the dominant heteroatom (highest priority) for stem selection
    dominant_elem = elements_by_priority[0] if elements_by_priority else 'O'

    # Get stem based on ring size, saturation, and dominant heteroatom
    stem = get_hw_stem(ring_size, saturated_for_stem, dominant_elem)
    if not stem:
        stem = ""

    # Apply 'a' elision: drop terminal 'a' before vowel stem
    if prefix.endswith('a') and stem and stem[0] in 'aeiou':
        prefix = prefix[:-1]

    return prefix + stem


def name_heterocycle(mol, ring_atoms) -> str:
    """
    Generate IUPAC name for a heterocyclic ring.

    Naming priority:
    1. Check retained names FIRST (pyridine, furan, morpholine, etc.)
    2. For rings 3-10: Hantzsch-Widman systematic naming
    3. For rings > 10: replacement ("a") nomenclature on cycloalkane parent

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        IUPAC name for the heterocycle

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> name_heterocycle(mol, ring)
        'pyridine'
        >>> mol2 = Chem.MolFromSmiles('C1CO1')  # oxirane
        >>> ring2 = mol2.GetRingInfo().AtomRings()[0]
        >>> name_heterocycle(mol2, ring2)
        'oxirane'
    """
    # Get ring canonical SMILES for retained name lookup
    ring_smiles = get_ring_canonical_smiles(mol, ring_atoms)

    # Check retained names FIRST
    retained = get_retained_name(ring_smiles)
    if retained:
        return retained

    # Fall back to systematic naming
    info = classify_heterocycle(mol, ring_atoms)
    oriented, _ = orient_heterocycle(mol, ring_atoms)

    # Get heteroatom locants from oriented ring
    heteroatom_locants = get_heteroatom_locants(oriented, mol)

    ring_size = info['ring_size']

    # For rings > 10: use replacement nomenclature (cycloXXXane parent)
    if ring_size > 10:
        return _build_replacement_name(
            heteroatom_locants,
            ring_size,
            info['is_saturated']
        )

    return build_hw_name(
        heteroatom_locants,
        ring_size,
        info['is_saturated'],
        info['is_aromatic']
    )


def _build_replacement_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool
) -> str:
    """
    Build replacement ("a") nomenclature name for macrocyclic heterocycles.

    For rings larger than 10 atoms, IUPAC uses "a" nomenclature:
    replacement prefixes (oxa, aza, thia) + cycloalkane parent name.

    Example: 1,4,7,10-tetraoxacyclododecane (12-crown-4)

    Args:
        heteroatoms: List of (locant, element) tuples
        ring_size: Number of atoms in the ring (> 10)
        is_saturated: True if fully saturated

    Returns:
        IUPAC replacement name (e.g., '1,4,7,10-tetraoxacyclododecane')
    """
    from ..data.chain_names import get_chain_prefix

    if not heteroatoms:
        # No heteroatoms: just cycloalkane (shouldn't reach here)
        prefix = get_chain_prefix(ring_size)
        return f"cyclo{prefix}ane"

    # Build replacement prefix (same logic as HW prefix builder)
    element_locants: Dict[str, List[int]] = {}
    for locant, elem in heteroatoms:
        if elem not in element_locants:
            element_locants[elem] = []
        element_locants[elem].append(locant)

    # Order by IUPAC priority (O > S > N > ...)
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    prefix_parts = []
    for elem in elements_by_priority:
        hw_prefix = get_hw_prefix(elem)
        if not hw_prefix:
            continue

        locants = sorted(element_locants[elem])
        count = len(locants)

        locant_str = ','.join(str(loc) for loc in locants)
        if count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f"{locant_str}-{multiplier}{hw_prefix}")
        else:
            prefix_parts.append(f"{locant_str}-{hw_prefix}")

    replacement_prefix = ''.join(prefix_parts)

    # Build parent ring name
    chain_prefix = get_chain_prefix(ring_size)
    if is_saturated:
        parent = f"cyclo{chain_prefix}ane"
    else:
        parent = f"cyclo{chain_prefix}ene"

    # Apply 'a' elision: drop terminal 'a' before 'cyclo' (which starts with 'c')
    # No elision needed since 'cyclo' starts with consonant

    return f"{replacement_prefix}{parent}"


# ---------------------------------------------------------------------------
# Substituted heterocycle naming (HETERO-09)
# ---------------------------------------------------------------------------


def get_heterocycle_substituents(
    mol,
    ring_atoms,
    oriented_ring: List[int],
    atom_to_locant: Dict[int, int]
) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to a heterocyclic ring.

    For each ring atom that has neighbors not in the ring, identifies the
    substituent and tracks whether it's attached to a nitrogen (N-substitution).
    Also classifies substituents as ring or alkyl to correctly name ring
    substituents (piperidinyl, phenyl) instead of counting carbons (pentyl, hexyl).

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        oriented_ring: Ring atoms in IUPAC numbering order
        atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Returns:
        Dict mapping locant to list of substituent info dicts:
        {
            locant: [
                {
                    'atoms': List[int],  # Atom indices in substituent
                    'is_on_nitrogen': bool,  # True if attached to N
                    'carbon_count': int,  # Number of C atoms (for alkyl naming)
                    'connecting_atom': int,  # Ring atom the sub attaches to
                    'is_ring': bool,  # True if substituent is a ring
                    'ring_name': str,  # Ring substituent name (if is_ring)
                }
            ]
        }

    Examples:
        >>> mol = Chem.MolFromSmiles('CN1CCCC1')  # N-methylpyrrolidine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented, atom_to_loc = orient_heterocycle(mol, ring)
        >>> subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_loc)
        >>> # N-methyl should be at locant 1 (N position) with is_on_nitrogen=True
    """
    from ..perception.chains import classify_substituent

    ring_set = set(ring_atoms)
    substituents: Dict[int, List[Dict]] = {}

    for ring_atom_idx in oriented_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        # Check if this ring atom is a nitrogen
        is_nitrogen = ring_atom.GetSymbol() == 'N'

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the ring
            if nbr_idx in ring_set:
                continue

            # BFS to find full substituent
            sub_atoms = _bfs_substituent(mol, nbr_idx, ring_set)

            # Count carbon atoms for alkyl naming
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Detect non-carbon functional substituents (amino, hydroxy, nitro, etc.)
            hetero_sub_name = None
            if carbon_count == 0:
                # Check for functional groups that have no carbons
                hetero_sub_name = _identify_hetero_substituent(mol, sub_atoms, ring_set)
                if hetero_sub_name is None:
                    continue  # Truly empty (just implicit H)

            # Classify substituent as ring or alkyl
            classification = classify_substituent(mol, sub_atoms, ring_set)
            is_ring = classification['type'] == 'ring'
            ring_name = classification['name'] if is_ring else None

            sub_info = {
                'atoms': sub_atoms,
                'is_on_nitrogen': is_nitrogen,
                'carbon_count': carbon_count,
                'connecting_atom': ring_atom_idx,
                'is_ring': is_ring,
                'ring_name': ring_name,
            }
            if hetero_sub_name:
                sub_info['hetero_name'] = hetero_sub_name

            if locant not in substituents:
                substituents[locant] = []
            substituents[locant].append(sub_info)

    return substituents


def _identify_hetero_substituent(mol, sub_atoms, ring_set) -> Optional[str]:
    """
    Identify a non-carbon functional substituent (amino, hydroxy, nitro, etc.).

    Args:
        mol: RDKit Mol object
        sub_atoms: List of atom indices in the substituent
        ring_set: Set of ring atom indices

    Returns:
        Substituent prefix name or None if not recognized
    """
    if not sub_atoms:
        return None

    first_atom = mol.GetAtomWithIdx(sub_atoms[0])
    symbol = first_atom.GetSymbol()
    h_count = first_atom.GetTotalNumHs()

    # Amino (-NH2)
    if symbol == 'N' and h_count == 2:
        return 'amino'

    # Hydroxy (-OH)
    if symbol == 'O' and h_count == 1:
        return 'hydroxy'

    # Nitro (-NO2)
    if symbol == 'N' and first_atom.GetFormalCharge() == 1:
        o_count = sum(1 for idx in sub_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == 'O')
        if o_count == 2:
            return 'nitro'

    # Sulfanyl (-SH)
    if symbol == 'S' and h_count == 1:
        return 'sulfanyl'

    # Oxo (=O) - check if double-bonded to ring carbon
    if symbol == 'O' and h_count == 0 and len(sub_atoms) == 1:
        for bond in first_atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(sub_atoms[0])
            if other_idx in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
                return 'oxo'

    return None


def _bfs_substituent(mol, start_idx: int, excluded: Set[int]) -> List[int]:
    """
    BFS to find all atoms in a substituent.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        excluded: Set of atom indices to exclude (ring atoms)

    Returns:
        List of atom indices in the substituent
    """
    visited = set()
    queue = [start_idx]
    result = []

    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in visited or atom_idx in excluded:
            continue

        visited.add(atom_idx)
        result.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                queue.append(nbr_idx)

    return result


def name_substituted_heterocycle(
    mol,
    ring_atoms,
    parent_name: str,
    substituents: Dict[int, List[Dict]],
    atom_to_locant: Dict[int, int]
) -> str:
    """
    Assemble complete name for a substituted heterocycle.

    N-substituted groups use N-locant format (N-methyl, N,N-dimethyl).
    C-substituted groups use numeric locants (2-methyl, 3-ethyl).
    Ring substituents use proper ring names (piperidinyl, phenyl) not carbon counts.
    All prefixes are sorted alphabetically (ignoring N-, numbers, multipliers).
    Stereochemistry descriptors (R/S) are added as a prefix if present.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        parent_name: Base heterocycle name (e.g., 'pyrrolidine', 'pyridine')
        substituents: Dict from get_heterocycle_substituents
        atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Returns:
        Complete IUPAC name (e.g., 'N-methylpyrrolidine', '3-methylpyridine',
        '(2S)-N-methyl-2-propylpiperidine')

    Examples:
        >>> mol = Chem.MolFromSmiles('CN1CCCC1')  # N-methylpyrrolidine
        >>> # ... get ring, orient, get substituents ...
        >>> name_substituted_heterocycle(mol, ring, 'pyrrolidine', subs, atom_to_loc)
        'N-methylpyrrolidine'
    """
    from ..assembly.naming_utils import (
        get_alkyl_name,
        alpha_sort_key,
        get_multiplier_prefix,
    )
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from rdkit.Chem import rdCIPLabeler

    # Ensure CIP labels are assigned
    rdCIPLabeler.AssignCIPLabels(mol)

    # Collect stereodescriptors using the heterocycle locant mapping
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    if not substituents:
        # No substituents but may have stereo
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            return f"{stereo_prefix}{parent_name}"
        return parent_name

    # Group substituents by name and N/C classification
    # Key: (substituent_name, is_on_nitrogen) -> list of locants
    n_groups: Dict[str, List[int]] = {}  # N-substituents: name -> locants
    c_groups: Dict[str, List[int]] = {}  # C-substituents: name -> locants

    for locant, sub_list in substituents.items():
        for sub_info in sub_list:
            is_on_n = sub_info['is_on_nitrogen']

            # Check if this is a ring substituent
            is_ring = sub_info.get('is_ring', False)
            ring_name = sub_info.get('ring_name')

            if is_ring and ring_name:
                # Use ring substituent name (piperidinyl, phenyl, etc.)
                sub_name = ring_name
            elif 'hetero_name' in sub_info:
                # Non-carbon functional substituent (amino, hydroxy, etc.)
                sub_name = sub_info['hetero_name']
            else:
                # Get alkyl name from carbon count
                carbon_count = sub_info['carbon_count']
                try:
                    sub_name = get_alkyl_name(carbon_count)
                except ValueError:
                    # Unsupported carbon count (> 10), skip
                    continue

            if is_on_n:
                if sub_name not in n_groups:
                    n_groups[sub_name] = []
                n_groups[sub_name].append(locant)
            else:
                if sub_name not in c_groups:
                    c_groups[sub_name] = []
                c_groups[sub_name].append(locant)

    # Build prefix parts
    prefix_parts = []

    # Format N-substituent prefixes
    for name, locants in n_groups.items():
        count = len(locants)
        prefix = _format_n_substituent(name, count)
        prefix_parts.append((prefix, name))

    # Format C-substituent prefixes
    for name, locants in c_groups.items():
        count = len(locants)
        prefix = _format_c_substituent(name, sorted(locants), count)
        prefix_parts.append((prefix, name))

    # Sort alphabetically by the base substituent name
    # (ignoring N-, locants, multipliers)
    prefix_parts.sort(key=lambda x: alpha_sort_key(x[1]))

    # Join prefixes
    prefix_str = ""
    for i, (prefix, _) in enumerate(prefix_parts):
        if i == 0:
            prefix_str = prefix
        else:
            # Add hyphen between prefixes if needed
            if prefix_str[-1].isalpha() and prefix[0].isdigit():
                prefix_str += "-"
            elif prefix_str[-1].isalpha() and prefix[0] == 'N':
                prefix_str += "-"
            prefix_str += prefix

    # Assemble with stereo prefix if present
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        return f"{stereo_prefix}{prefix_str}{parent_name}"

    return f"{prefix_str}{parent_name}"


def _format_n_substituent(name: str, count: int) -> str:
    """
    Format an N-substituent prefix.

    Single: N-methyl
    Multiple same: N,N-dimethyl
    """
    if count == 1:
        return f"N-{name}"
    else:
        # N,N-dimethyl, N,N,N-trimethyl, etc.
        n_locants = ",".join(["N"] * count)
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{n_locants}-{multiplier}{name}"


def _format_c_substituent(name: str, locants: List[int], count: int) -> str:
    """
    Format a C-substituent prefix with numeric locants.

    Single: 3-methyl
    Multiple same: 2,4-dimethyl
    """
    locant_str = ",".join(str(loc) for loc in locants)
    if count == 1:
        return f"{locant_str}-{name}"
    else:
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{locant_str}-{multiplier}{name}"


def get_saturation_prefix(
    mol,
    ring_atoms,
    is_aromatic_parent: bool
) -> str:
    """
    Get saturation prefix for partially saturated heterocycles (HETERO-08).

    For saturated versions of aromatic parents, returns prefixes like:
    - dihydro- (2 additional H atoms)
    - tetrahydro- (4 additional H atoms)

    Note: This is a placeholder - full implementation requires comparing
    to the aromatic parent structure.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        is_aromatic_parent: True if the parent is aromatic (e.g., furan vs THF)

    Returns:
        Saturation prefix string (e.g., 'tetrahydro-') or empty string

    Examples:
        >>> # Tetrahydrofuran is "oxolane" (retained name), but could also be
        >>> # named "tetrahydrofuran" relating it to furan
        >>> # This function helps determine the saturation prefix when needed
    """
    # For now, this is a simplified implementation
    # Full HETERO-08 support would need to compare H counts between
    # aromatic and saturated forms
    info = classify_heterocycle(mol, ring_atoms)

    if is_aromatic_parent and info['is_saturated']:
        # Calculate hydrogenation level
        # For now, return placeholder - retained names handle common cases
        # like tetrahydrofuran, tetrahydropyran, etc.
        pass

    return ""
