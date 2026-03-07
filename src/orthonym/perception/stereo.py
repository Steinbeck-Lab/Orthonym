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

    Uses the _CIPCode property set by rdCIPLabeler as the primary source
    (more reliable than BondStereo for complex molecules), with fallback
    to BondStereo enum values.

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
