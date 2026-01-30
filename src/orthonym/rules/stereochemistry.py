"""
Stereochemistry rules - IUPAC stereodescriptor collection and formatting.

This module handles the mapping from RDKit stereochemistry (indexed by atom
indices) to IUPAC nomenclature (indexed by locants on the principal chain/ring).

Key functions:
- collect_stereodescriptors: Get R/S and E/Z descriptors with IUPAC locants
- format_stereodescriptor_string: Format as "(2R,3S)-" prefix
- get_double_bond_locant: Get lower locant for E/Z double bond

IMPORTANT: This module requires atom_to_locant mapping to be provided by the
caller (computed during chain/ring classification). It does NOT fall back to
atom indices as those are NOT valid IUPAC locants.
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


def collect_stereodescriptors(
    mol,
    atom_to_locant: Dict[int, int]
) -> List[Tuple[int, str]]:
    """
    Collect all stereodescriptors from a molecule using IUPAC locants.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        atom_to_locant: Mapping from atom index to IUPAC locant number.
                       Only atoms in this mapping are considered (principal
                       chain/ring atoms).

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.
        cip_code is 'R', 'S', 'r', 's' (lowercase for pseudoasymmetric),
        or 'E', 'Z' for double bonds.

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H](O)CC')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}  # chain oriented
        >>> collect_stereodescriptors(mol, atom_to_locant)
        [(3, 'R')]
    """
    # Ensure stereochemistry is assigned
    rdCIPLabeler.AssignCIPLabels(mol)

    descriptors: List[Tuple[int, str]] = []

    # Collect R/S stereocenters (atom-based)
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            atom_idx = atom.GetIdx()
            # Only include atoms on principal chain/ring
            if atom_idx in atom_to_locant:
                locant = atom_to_locant[atom_idx]
                cip_code = atom.GetProp('_CIPCode')  # 'R', 'S', 'r', or 's'
                descriptors.append((locant, cip_code))

    # Collect E/Z double bonds (bond-based)
    for bond in mol.GetBonds():
        if bond.HasProp('_CIPCode'):
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Both atoms must be in mapping
            if begin_idx in atom_to_locant and end_idx in atom_to_locant:
                locant_a = atom_to_locant[begin_idx]
                locant_b = atom_to_locant[end_idx]
                # IUPAC: use lower locant for bond position
                locant = min(locant_a, locant_b)
                cip_code = bond.GetProp('_CIPCode')  # 'E' or 'Z'
                descriptors.append((locant, cip_code))

    # Sort by locant ascending
    descriptors.sort(key=lambda x: x[0])

    return descriptors


def get_double_bond_locant(
    bond,
    atom_to_locant: Dict[int, int]
) -> Optional[int]:
    """
    Get the IUPAC locant for a double bond.

    Per IUPAC convention, the locant is the lower of the two atom locants.

    Args:
        bond: RDKit Bond object
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Lower locant of the two bond atoms, or None if either atom is
        not in the mapping (not on principal chain/ring).

    Example:
        >>> mol = Chem.MolFromSmiles('C/C=C/C')
        >>> bond = mol.GetBondWithIdx(1)  # the C=C bond
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        >>> get_double_bond_locant(bond, atom_to_locant)
        2
    """
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    # Both atoms must be in mapping
    if begin_idx not in atom_to_locant or end_idx not in atom_to_locant:
        return None

    locant_a = atom_to_locant[begin_idx]
    locant_b = atom_to_locant[end_idx]

    return min(locant_a, locant_b)


def format_stereodescriptor_string(
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format stereodescriptors as an IUPAC name prefix.

    STEREO-04: Multiple descriptors in single block with comma separation.

    Args:
        descriptors: List of (locant, cip_code) tuples, sorted by locant.

    Returns:
        Formatted string like "(2R)-", "(2R,3S)-", "(2E,3R,5Z)-"
        Returns empty string if no descriptors.

    Examples:
        >>> format_stereodescriptor_string([])
        ''
        >>> format_stereodescriptor_string([(2, 'R')])
        '(2R)-'
        >>> format_stereodescriptor_string([(2, 'R'), (3, 'S')])
        '(2R,3S)-'
        >>> format_stereodescriptor_string([(2, 'E'), (3, 'R'), (5, 'Z')])
        '(2E,3R,5Z)-'
        >>> format_stereodescriptor_string([(2, 'r'), (3, 's')])
        '(2r,3s)-'
    """
    if not descriptors:
        return ""

    # Build comma-separated list of "locantCIP"
    parts = [f"{locant}{cip}" for locant, cip in descriptors]

    return f"({','.join(parts)})-"


def collect_ring_stereodescriptors(
    mol,
    ring_atom_to_locant: Dict[int, int]
) -> List[Tuple[int, str]]:
    """
    Collect stereodescriptors for ring compounds.

    Same as collect_stereodescriptors but specifically for ring naming,
    where only atoms in the ring are considered.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        ring_atom_to_locant: Mapping from ring atom indices to IUPAC locants.
                            Only atoms that are keys in this dict are included.

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.

    Note:
        For rings, double bond E/Z is less common (ring strain), but
        we still handle it for completeness.
    """
    # Reuse the main function - it already handles the filtering correctly
    return collect_stereodescriptors(mol, ring_atom_to_locant)
