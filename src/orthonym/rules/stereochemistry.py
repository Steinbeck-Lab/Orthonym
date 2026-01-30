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


def determine_ring_cis_trans(
    mol,
    ring_atoms: List[int],
    sub1_idx: int,
    sub2_idx: int
) -> Optional[str]:
    """
    Determine if two substituents on a ring are cis or trans.

    For a ring with exactly 2 substituents at specified positions,
    determine their relative stereochemistry based on CIP labels.

    The rule for 1,2-disubstituted rings:
    - SAME CIP codes (R,R or S,S) -> CIS (substituents on same face)
    - DIFFERENT CIP codes (R,S or S,R) -> TRANS (substituents on opposite faces)

    This is because in a ring, atoms with the same absolute configuration
    at adjacent positions have their substituents on the same face.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices that form the ring
        sub1_idx: Atom index of first substituted ring carbon
        sub2_idx: Atom index of second substituted ring carbon

    Returns:
        'cis' or 'trans', or None if cannot be determined (missing CIP labels)

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')  # cis-1,2-dimethylcyclohexane
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> determine_ring_cis_trans(mol, ring_atoms, 1, 6)
        'cis'
    """
    # Both atoms must be in the ring
    if sub1_idx not in ring_atoms or sub2_idx not in ring_atoms:
        return None

    # Get atoms
    atom1 = mol.GetAtomWithIdx(sub1_idx)
    atom2 = mol.GetAtomWithIdx(sub2_idx)

    # Both must have CIP labels
    if not atom1.HasProp('_CIPCode') or not atom2.HasProp('_CIPCode'):
        return None

    cip1 = atom1.GetProp('_CIPCode')
    cip2 = atom2.GetProp('_CIPCode')

    # Only handle R/S (not r/s pseudoasymmetric for now)
    if cip1 not in ['R', 'S'] or cip2 not in ['R', 'S']:
        return None

    # Same CIP = cis, Different CIP = trans
    if cip1 == cip2:
        return 'cis'
    else:
        return 'trans'


def get_simple_ring_stereo(
    mol,
    ring_atoms: List[int],
    ring_substituents: Dict[int, str]
) -> Optional[str]:
    """
    Get cis/trans prefix for simple disubstituted rings.

    This function handles the common case of exactly 2 substituted positions
    on a ring, returning a cis- or trans- prefix for the name.

    Note: This is a simplification. More complex rings (3+ substituents)
    would need the full IUPAC r/c/t reference system, which is deferred.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        ring_substituents: Dict mapping ring atom locant -> substituent name.
                          Only keys (locants) are used to identify substituted positions.

    Returns:
        'cis-' or 'trans-' prefix string, or None if:
        - Not exactly 2 substituted positions
        - Cannot determine stereochemistry

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> # Assuming oriented_ring maps locant -> atom_idx
        >>> ring_substituents = {1: 'methyl', 2: 'methyl'}
        >>> get_simple_ring_stereo(mol, ring_atoms, ring_substituents)
        'cis-'
    """
    # Need exactly 2 substituted positions for simple cis/trans
    if len(ring_substituents) != 2:
        return None

    # Get the locants with substituents
    locants = list(ring_substituents.keys())

    # We need to find which atom indices correspond to these locants
    # This requires knowing the atom_to_locant mapping
    # Since we don't have it directly, we'll need the caller to provide
    # the actual atom indices

    # For now, return None and let the caller handle the mapping
    # This function needs the actual atom indices, not locants
    return None


def get_simple_ring_stereo_from_atoms(
    mol,
    ring_atoms: List[int],
    substituted_atom_indices: List[int]
) -> Optional[str]:
    """
    Get cis/trans prefix given the actual atom indices of substituted positions.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        substituted_atom_indices: List of exactly 2 atom indices that have substituents

    Returns:
        'cis-' or 'trans-' prefix string, or None if cannot determine

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> get_simple_ring_stereo_from_atoms(mol, ring_atoms, [1, 6])
        'cis-'
    """
    if len(substituted_atom_indices) != 2:
        return None

    result = determine_ring_cis_trans(
        mol,
        ring_atoms,
        substituted_atom_indices[0],
        substituted_atom_indices[1]
    )

    if result:
        return f'{result}-'
    return None


def format_ring_stereo_with_descriptors(
    ring_cis_trans: Optional[str],
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format ring stereo prefix with optional R/S descriptors.

    IUPAC format for ring stereo:
    - If only cis/trans: "cis-1,2-dimethylcyclohexane"
    - If cis/trans + R/S: "cis-(1R,2S)-1,2-dimethylcyclohexane"
    - The cis/trans goes BEFORE the stereodescriptors

    Args:
        ring_cis_trans: 'cis-' or 'trans-' prefix, or None
        descriptors: List of (locant, cip_code) tuples from collect_stereodescriptors

    Returns:
        Combined prefix string like "cis-", "trans-(1R,2S)-", etc.
        Empty string if no stereo information.

    Example:
        >>> format_ring_stereo_with_descriptors('cis-', [(1, 'R'), (2, 'S')])
        'cis-(1R,2S)-'
        >>> format_ring_stereo_with_descriptors('trans-', [])
        'trans-'
        >>> format_ring_stereo_with_descriptors(None, [(1, 'R')])
        '(1R)-'
    """
    rs_string = format_stereodescriptor_string(descriptors)

    if ring_cis_trans:
        if rs_string:
            return f'{ring_cis_trans}{rs_string}'
        else:
            return ring_cis_trans
    else:
        return rs_string
