"""
Heterocycle detection, classification, and ring numbering according to IUPAC 2013.

Handles:
- Heterocycle classification (ring size, aromaticity, heteroatom types)
- Ring numbering starting at highest-priority heteroatom
- Direction selection to minimize locants for other heteroatoms
- First-point-of-difference rule for direction ties

IUPAC 2013 Rules for heterocycle numbering:
- Position 1 goes to highest-priority heteroatom (O > S > Se > Te > N > P > ...)
- Numbering direction chosen to give lowest locants to other heteroatoms
- First-point-of-difference comparison (not sum of locants)

Reference: IUPAC 2013 Blue Book, Section P-22 (Heterocycles)
"""

from typing import Dict, List, Tuple, Optional, Set

from rdkit import Chem

from ..data.hw_heteroatoms import HETEROATOM_PRIORITY, get_heteroatom_priority
from ..perception.rings import (
    get_ring_heteroatoms,
    is_aromatic_ring,
    is_saturated_ring,
)


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
    is_arom = is_aromatic_ring(mol, ring_atoms)
    is_sat = is_saturated_ring(mol, ring_atoms)

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
