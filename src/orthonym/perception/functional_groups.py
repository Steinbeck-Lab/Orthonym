"""
Functional group detection using SMARTS patterns.

Groups are ordered by IUPAC seniority (highest priority first).
The principal group (highest seniority) becomes the suffix;
all others become prefixes.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from rdkit import Chem


# SMARTS patterns ordered by IUPAC seniority (P-41 to P-43)
# First match = highest priority = principal group
FUNCTIONAL_GROUP_SMARTS = {
    # === ACIDS (highest priority) ===
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "sulfinic_acid": "[SX3](=O)[OX2H1]",
    "phosphonic_acid": "[PX4](=O)([OX2H1])[OX2H1]",
    
    # === ACID DERIVATIVES ===
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    "thioester": "[CX3](=O)[SX2][#6]",
    "acid_chloride": "[CX3](=O)[Cl]",
    "acid_bromide": "[CX3](=O)[Br]",
    "acid_fluoride": "[CX3](=O)[F]",
    
    # === NITROGEN ACID DERIVATIVES ===
    "primary_amide": "[CX3](=O)[NX3H2]",
    "secondary_amide": "[CX3](=O)[NX3H1][#6]",
    "tertiary_amide": "[CX3](=O)[NX3]([#6])[#6]",
    "hydrazide": "[CX3](=O)[NX3][NX3]",
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    
    # === NITRILES ===
    "nitrile": "[CX2]#[NX1]",
    "isocyanide": "[#6][NX2]#[CX1]",
    
    # === CARBONYLS ===
    # Aldehyde: carbonyl with H and bonded to C (not N/O)
    # [CX3H1](=O) matches the carbonyl, [#6] ensures attached to carbon
    # This excludes amides where C is bonded to N
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "thioaldehyde": "[CX3H1](=S)",
    "thioketone": "[#6][CX3](=S)[#6]",
    
    # === ALCOHOLS AND ANALOGS ===
    "primary_alcohol": "[OX2H][CX4H2]",
    "secondary_alcohol": "[OX2H][CX4H1]([#6])[#6]",
    "tertiary_alcohol": "[OX2H][CX4]([#6])([#6])[#6]",
    "phenol": "[OX2H][cX3]",
    "enol": "[OX2H][CX3]=[CX3]",
    "thiol": "[SX2H][#6]",
    "selenol": "[SeX2H]",
    
    # === HYDROPEROXIDES ===
    "hydroperoxide": "[OX2H][OX2][#6]",
    "peroxide": "[#6][OX2][OX2][#6]",
    
    # === AMINES ===
    "primary_amine": "[NX3H2][CX4]",
    "secondary_amine": "[NX3H1]([CX4])[CX4]",
    "tertiary_amine": "[NX3]([CX4])([CX4])[CX4]",
    "aromatic_amine": "[NX3H2][cX3]",
    
    # === IMINES ===
    "imine": "[CX3]=[NX2H]",
    "oxime": "[CX3]=[NX2][OX2H]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    
    # === ETHERS (no suffix - substitutive naming) ===
    "ether": "[OX2]([CX4])[CX4]",
    "vinyl_ether": "[OX2]([#6])[CX3]=[CX3]",
    "aromatic_ether": "[OX2]([#6])[cX3]",
    "thioether": "[SX2]([#6])[#6]",
    
    # === UNSATURATION ===
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    
    # === HALOGENS (always prefixes) ===
    "fluoro": "[FX1][#6]",
    "chloro": "[ClX1][#6]",
    "bromo": "[BrX1][#6]",
    "iodo": "[IX1][#6]",
    
    # === OTHER ===
    "nitro": "[NX3+](=O)[O-]",
    "nitroso": "[NX2]=[OX1]",
    "azido": "[NX1]=[NX2+]=[NX1-]",
}


def detect_functional_groups(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Detect all functional groups in a molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary mapping functional group names to lists of atom index tuples.
        Each tuple contains the indices of atoms in one instance of that group.
        
    Example:
        >>> mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        >>> groups = detect_functional_groups(mol)
        >>> "carboxylic_acid" in groups
        True
        >>> len(groups["carboxylic_acid"])
        1
    """
    results = defaultdict(list)
    
    for fg_name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        
        matches = mol.GetSubstructMatches(pattern, uniquify=True)
        for match in matches:
            results[fg_name].append(match)
    
    return dict(results)


def has_functional_group(mol, fg_name: str) -> bool:
    """
    Check if molecule contains a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group (must be in FUNCTIONAL_GROUP_SMARTS)
        
    Returns:
        True if functional group is present
    """
    if fg_name not in FUNCTIONAL_GROUP_SMARTS:
        return False
    
    smarts = FUNCTIONAL_GROUP_SMARTS[fg_name]
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return False
    
    return mol.HasSubstructMatch(pattern)


def get_functional_group_atoms(mol, fg_name: str) -> List[Tuple[int, ...]]:
    """
    Get atom indices for all instances of a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group
        
    Returns:
        List of tuples of atom indices
    """
    if fg_name not in FUNCTIONAL_GROUP_SMARTS:
        return []
    
    smarts = FUNCTIONAL_GROUP_SMARTS[fg_name]
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return []
    
    return list(mol.GetSubstructMatches(pattern, uniquify=True))


def count_functional_groups(mol) -> Dict[str, int]:
    """
    Count occurrences of each functional group.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary mapping functional group names to counts
    """
    groups = detect_functional_groups(mol)
    return {name: len(matches) for name, matches in groups.items()}


def get_all_functional_group_atoms(mol) -> set:
    """
    Get all atom indices that are part of any functional group.
    
    Useful for identifying which atoms are "special" vs backbone.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Set of atom indices
    """
    all_atoms = set()
    groups = detect_functional_groups(mol)
    
    for matches in groups.values():
        for match in matches:
            all_atoms.update(match)
    
    return all_atoms
