"""
Stereochemistry perception and CIP assignment.

CRITICAL: Always use rdCIPLabeler.AssignCIPLabels(), not the legacy
Chem.AssignStereochemistry() which fails on complex molecules.
"""

from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


_CIP_ASSIGNED_PROP = '_Orthonym_CIPAssigned'


def assign_stereochemistry(mol) -> None:
    """
    Assign CIP stereochemistry labels to a molecule (idempotent guard).

    Uses a private marker property to track whether rdCIPLabeler has
    already been called on this mol object.  This is more reliable than
    checking for _CIPCode because RDKit's MolFromSmiles() automatically
    sets atom _CIPCode from @/@@ notation, but does NOT set bond _CIPCode
    for E/Z -- so an atom-based check would short-circuit and skip the
    bond labels.

    The authoritative call site in namer.py:_perceive() sets the marker
    after calling rdCIPLabeler.  Handler modules call this function for
    safety (e.g., natural_products runs BEFORE _perceive()).

    Args:
        mol: RDKit Mol object (modified in place)
    """
    if mol.HasProp(_CIP_ASSIGNED_PROP):
        return

    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        # Fallback to legacy for very simple molecules
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    mol.SetProp(_CIP_ASSIGNED_PROP, '1')


def get_stereocenters(mol) -> List[Dict]:
    """
    Get all stereocenters with their CIP labels.
    
    Args:
        mol: RDKit Mol object (stereochemistry should be assigned first)
        
    Returns:
        List of dicts with keys:
        - idx: atom index
        - cip: 'R' or 'S'
        - symbol: atom element symbol
        - neighbors: list of neighbor atom indices
    """
    # Ensure stereochemistry is assigned
    assign_stereochemistry(mol)
    
    centers = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            cip_code = atom.GetProp('_CIPCode')
            # Preserve CIP code as-is: uppercase R/S for normal stereocenters,
            # lowercase r/s for pseudoasymmetric centers per IUPAC P-92.1.4.2.
            centers.append({
                'idx': atom.GetIdx(),
                'cip': cip_code,
                'symbol': atom.GetSymbol(),
                'neighbors': [n.GetIdx() for n in atom.GetNeighbors()],
            })
    
    return centers


def get_double_bond_stereo(mol) -> List[Dict]:
    """
    Get E/Z configuration of double bonds.

    Uses the _CIPCode property set by rdCIPLabeler as the sole source
    of E/Z labels. BondStereo fallback was removed in Phase 92-03.

    Args:
        mol: RDKit Mol object (stereochemistry should be assigned via
             rdCIPLabeler.AssignCIPLabels BEFORE calling this function)

    Returns:
        List of dicts with keys:
        - idx: bond index
        - stereo: 'E' or 'Z'
        - atoms: (begin_atom_idx, end_atom_idx)
    """
    stereo_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if bond.HasProp('_CIPCode'):
                cip_code = bond.GetProp('_CIPCode')
                if cip_code in ('E', 'Z'):
                    stereo_bonds.append({
                        'idx': bond.GetIdx(),
                        'stereo': cip_code,
                        'atoms': (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    })

    return stereo_bonds


def has_stereochemistry(mol) -> bool:
    """
    Check if molecule has any defined stereochemistry.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if molecule has stereocenters or double bond stereo
    """
    assign_stereochemistry(mol)
    
    # Check for atom stereocenters
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            return True
    
    # Check for defined double bond stereochemistry via _CIPCode
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if bond.HasProp('_CIPCode') and bond.GetProp('_CIPCode') in ('E', 'Z'):
                return True
    
    return False


def get_stereodescriptor_string(
    mol,
    locant_map: Optional[Dict[int, int]] = None
) -> str:
    """
    Generate stereodescriptor string for name prefix.
    
    Format: (2R,3S)-... or (E)-... or (2R,3S,5E)-...
    
    Args:
        mol: RDKit Mol object
        locant_map: Optional mapping from atom index to locant number
                    If None, uses atom indices as locants
        
    Returns:
        Stereodescriptor string (empty if no stereochemistry)
    """
    assign_stereochemistry(mol)
    
    descriptors = []
    
    # Collect atom stereocenters
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            idx = atom.GetIdx()
            cip = atom.GetProp('_CIPCode')
            # Preserve CIP code as-is: R/S for normal, r/s for pseudoasymmetric
            # per IUPAC P-92.1.4.2.

            if locant_map and idx in locant_map:
                locant = locant_map[idx]
            else:
                locant = idx + 1  # 1-indexed fallback
            
            descriptors.append((locant, f"{locant}{cip}"))
    
    # Collect double bond stereochemistry via _CIPCode
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            if bond.HasProp('_CIPCode'):
                cip_code = bond.GetProp('_CIPCode')
                if cip_code in ('E', 'Z'):
                    begin_idx = bond.GetBeginAtomIdx()
                    if locant_map and begin_idx in locant_map:
                        locant = locant_map[begin_idx]
                    else:
                        locant = begin_idx + 1
                    descriptors.append((locant, f"{locant}{cip_code}"))
    
    if not descriptors:
        return ""
    
    # Sort by locant and join
    descriptors.sort(key=lambda x: x[0])
    descriptor_strings = [d[1] for d in descriptors]
    
    return f"({','.join(descriptor_strings)})-"


def count_stereocenters(mol) -> int:
    """
    Count the number of stereocenters in a molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Number of defined stereocenters
    """
    return len(get_stereocenters(mol))


def count_double_bond_stereo(mol) -> int:
    """
    Count the number of double bonds with defined E/Z stereochemistry.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Number of E/Z defined double bonds
    """
    return len(get_double_bond_stereo(mol))


def is_chiral(mol) -> bool:
    """
    Check if molecule has any chiral centers.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if molecule has at least one defined stereocenter
    """
    return count_stereocenters(mol) > 0


def get_undefined_stereocenters(mol) -> List[int]:
    """
    Find potential stereocenters without defined stereochemistry.

    Args:
        mol: RDKit Mol object

    Returns:
        List of atom indices that are potential stereocenters
        but don't have defined R/S configuration
    """
    # Get atoms flagged as potential stereocenters
    Chem.AssignStereochemistry(mol, cleanIt=False, force=False, flagPossibleStereoCenters=True)

    undefined = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            if not atom.HasProp('_CIPCode'):
                undefined.append(atom.GetIdx())

    return undefined


# =============================================================================
# Axial Chirality Detection (IUPAC P-93.5)
# =============================================================================

def detect_axial_chirality(mol) -> List[Dict]:
    """
    Detect allene and atropisomer axial chirality in a molecule.

    Identifies two types of axial chirality encoded in the input:
    1. Atropisomers: bonds with STEREOATROPCW or STEREOATROPCCW stereo
    2. Allenes: atoms with CHI_ALLENE chiral tag on central C of C=C=C

    Only detects chirality that is explicitly encoded in the molecular
    representation. Does NOT attempt to infer chirality where the input
    is silent.

    CIP mapping (IUPAC P-93.5.3):
        RDKit CIP P -> Ra (clockwise in elongated tetrahedron)
        RDKit CIP M -> Sa (anticlockwise in elongated tetrahedron)

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts with keys:
        - type: 'allene' or 'atropisomer'
        - idx: atom index (allene) or bond index (atropisomer)
        - cip: 'Ra' or 'Sa' (or None if undetermined)
        - locant_atom: atom index to use for IUPAC locant mapping
    """
    # Ensure CIP labels are assigned (needed for atropisomer P/M)
    assign_stereochemistry(mol)

    results = []

    # 1. Atropisomers: check bonds for STEREOATROPCW/STEREOATROPCCW
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOATROPCW,
                      Chem.BondStereo.STEREOATROPCCW):
            # RDKit CIP: P -> Ra, M -> Sa
            cip = None
            if bond.HasProp('_CIPCode'):
                rdkit_cip = bond.GetProp('_CIPCode')
                if rdkit_cip == 'P':
                    cip = 'Ra'
                elif rdkit_cip == 'M':
                    cip = 'Sa'
            results.append({
                'type': 'atropisomer',
                'idx': bond.GetIdx(),
                'cip': cip,
                'locant_atom': bond.GetBeginAtomIdx(),
            })

    # 2. Allenes: find atoms with CHI_ALLENE chiral tag
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_ALLENE:
            # rdCIPLabeler does not assign CIP to allene atoms in RDKit 2025.09
            # Use manual CIP determination
            cip = _manual_allene_cip(mol, atom.GetIdx())
            results.append({
                'type': 'allene',
                'idx': atom.GetIdx(),
                'cip': cip,
                'locant_atom': atom.GetIdx(),
            })

    return results


def _manual_allene_cip(mol, central_idx: int) -> Optional[str]:
    """
    Determine Ra/Sa for an allene by comparing terminal substituent priorities.

    For an allene C1=C=C2, view along the C=C=C axis. The allene is treated
    as an elongated tetrahedron with 4 substituents (2 on each terminal carbon).
    If the arrangement from highest-priority-near to highest-priority-far
    is clockwise: Ra. If counterclockwise: Sa.

    Uses RDKit canonical atom ranks as a proxy for CIP priority ordering.

    Args:
        mol: RDKit Mol object
        central_idx: Atom index of the central allene carbon

    Returns:
        'Ra', 'Sa', or None (if achiral -- fewer than 4 distinct groups)
    """
    central_atom = mol.GetAtomWithIdx(central_idx)

    # Get the two double-bond neighbors of the central allene C
    terminal_atoms = []
    for bond in central_atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            other_idx = bond.GetOtherAtomIdx(central_idx)
            terminal_atoms.append(other_idx)

    if len(terminal_atoms) != 2:
        return None

    # Add explicit Hs for priority analysis
    mol_h = Chem.AddHs(mol)
    ranks = Chem.CanonicalRankAtoms(mol_h)

    term_a_idx, term_b_idx = terminal_atoms

    # Get substituents on each terminal atom (excluding the allene central C)
    def _get_terminal_substituents(atom_idx):
        """Get substituent atom indices and their ranks for a terminal atom."""
        atom = mol_h.GetAtomWithIdx(atom_idx)
        subs = []
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx != central_idx:
                subs.append((nbr_idx, ranks[nbr_idx]))
        # Sort by rank descending (higher rank = higher CIP priority proxy)
        subs.sort(key=lambda x: x[1], reverse=True)
        return subs

    subs_a = _get_terminal_substituents(term_a_idx)
    subs_b = _get_terminal_substituents(term_b_idx)

    # Check achirality: if both substituents on a terminal are chemically
    # equivalent (same atomic number), the allene is achiral at that terminal.
    # We cannot use canonical ranks for this check because CanonicalRankAtoms
    # differentiates even chemically equivalent atoms (e.g., two H atoms
    # on the same carbon get different ranks).
    def _terminal_is_achiral(subs_list):
        """Check if a terminal's substituents are chemically equivalent."""
        if len(subs_list) < 2:
            return False
        atom_a = mol_h.GetAtomWithIdx(subs_list[0][0])
        atom_b = mol_h.GetAtomWithIdx(subs_list[1][0])
        return atom_a.GetAtomicNum() == atom_b.GetAtomicNum()

    if _terminal_is_achiral(subs_a) or _terminal_is_achiral(subs_b):
        return None  # Achiral allene

    # Now determine chirality using the elongated tetrahedron model.
    # The CHI_ALLENE tag encodes the enantiomer via the neighbor ordering
    # in the atom's neighbor list.

    # Priority ordering: for each terminal, get the high-priority sub index
    high_a = subs_a[0][0] if subs_a else None
    high_b = subs_b[0][0] if subs_b else None

    if high_a is None or high_b is None:
        return None

    # Get the neighbor order as stored in the molecule for the central atom
    central_atom_h = mol_h.GetAtomWithIdx(central_idx)
    nbr_list = [n.GetIdx() for n in central_atom_h.GetNeighbors()]

    # The central allene C has exactly 2 neighbors (the two terminal Cs)
    if len(nbr_list) != 2:
        return None

    rank_high_a = ranks[high_a]
    rank_high_b = ranks[high_b]

    # The terminal atom that appears first in the central atom's neighbor
    # list defines the "near" end for chirality determination.
    first_terminal = nbr_list[0]

    # Determine sense based on the elongated tetrahedron model:
    # View along the allene axis from the near terminal to the far terminal.
    # If high-priority-near to high-priority-far is clockwise: Ra
    # If counterclockwise: Sa
    if first_terminal == term_a_idx:
        near_high_rank = rank_high_a
        far_high_rank = rank_high_b
    else:
        near_high_rank = rank_high_b
        far_high_rank = rank_high_a

    if near_high_rank > far_high_rank:
        return 'Ra'
    elif near_high_rank < far_high_rank:
        return 'Sa'
    else:
        return None  # Identical priorities -- achiral
