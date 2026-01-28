"""
Stereochemistry perception and CIP assignment.

CRITICAL: Always use rdCIPLabeler.AssignCIPLabels(), not the legacy
Chem.AssignStereochemistry() which fails on complex molecules.
"""

from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


def assign_stereochemistry(mol) -> None:
    """
    Assign CIP stereochemistry labels to a molecule.
    
    Uses the new accurate CIP algorithm (rdCIPLabeler) introduced
    in RDKit 2022.09. Falls back to legacy for simple cases only.
    
    Args:
        mol: RDKit Mol object (modified in place)
    """
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        # Fallback to legacy for very simple molecules
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


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
            centers.append({
                'idx': atom.GetIdx(),
                'cip': atom.GetProp('_CIPCode'),
                'symbol': atom.GetSymbol(),
                'neighbors': [n.GetIdx() for n in atom.GetNeighbors()],
            })
    
    return centers


def get_double_bond_stereo(mol) -> List[Dict]:
    """
    Get E/Z configuration of double bonds.
    
    Args:
        mol: RDKit Mol object (stereochemistry should be assigned)
        
    Returns:
        List of dicts with keys:
        - idx: bond index
        - stereo: 'E' or 'Z'
        - atoms: (begin_atom_idx, end_atom_idx)
    """
    assign_stereochemistry(mol)
    
    stereo_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.BondStereo.STEREOE:
                stereo_bonds.append({
                    'idx': bond.GetIdx(),
                    'stereo': 'E',
                    'atoms': (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                })
            elif stereo == Chem.BondStereo.STEREOZ:
                stereo_bonds.append({
                    'idx': bond.GetIdx(),
                    'stereo': 'Z',
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
    
    # Check for defined double bond stereochemistry
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
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
            
            if locant_map and idx in locant_map:
                locant = locant_map[idx]
            else:
                locant = idx + 1  # 1-indexed fallback
            
            descriptors.append((locant, f"{locant}{cip}"))
    
    # Collect double bond stereochemistry
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.BondStereo.STEREOE:
                begin_idx = bond.GetBeginAtomIdx()
                if locant_map and begin_idx in locant_map:
                    locant = locant_map[begin_idx]
                else:
                    locant = begin_idx + 1
                descriptors.append((locant, f"{locant}E"))
            elif stereo == Chem.BondStereo.STEREOZ:
                begin_idx = bond.GetBeginAtomIdx()
                if locant_map and begin_idx in locant_map:
                    locant = locant_map[begin_idx]
                else:
                    locant = begin_idx + 1
                descriptors.append((locant, f"{locant}Z"))
    
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
