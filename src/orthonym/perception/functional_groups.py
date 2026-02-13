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
    # Thiocarboxylic acids (IUPAC P-65.3) -- rank just below carboxylic acid
    "thioic_S_acid": "[CX3](=O)[SX2H1]",    # R-C(=O)-SH -> thioic S-acid
    "thioic_O_acid": "[CX3](=S)[OX2H1]",    # R-C(=S)-OH -> thioic O-acid
    "dithioic_acid": "[CX3](=S)[SX2H1]",    # R-C(=S)-SH -> dithioic acid
    # Carbamic acid (IUPAC P-65.2.3): N-C(=O)-OH (free acid, not ester)
    "carbamic_acid": "[NX3][CX3](=O)[OX2H1]",  # R2N-C(=O)-OH -> carbamic acid
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "sulfinic_acid": "[SX3](=O)[OX2H1]",
    "phosphonic_acid": "[PX4](=O)([OX2H1])[OX2H1]",
    # Phosphinic acid: R2P(=O)(OH) - two C attached to P
    "phosphinic_acid": "[PX4](=O)([OX2H1])([#6])[#6]",
    
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

    # === SULFONAMIDES ===
    "primary_sulfonamide": "[SX4](=O)(=O)[NX3H2]",
    "secondary_sulfonamide": "[SX4](=O)(=O)[NX3H1][#6]",
    "tertiary_sulfonamide": "[SX4](=O)(=O)[NX3]([#6])[#6]",
    
    # === CARBAMATES (must check before esters -- N-C(=O)-O is more specific) ===
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",

    # === UREA (must check before amides -- N-C(=O)-N is more specific) ===
    "urea": "[NX3][CX3](=O)[NX3]",

    # === GUANIDINE (must check before imines -- N-C(=N)-N is more specific) ===
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",

    # === ISOCYANATES/ISOTHIOCYANATES (cumulated double bonds) ===
    "isocyanate": "[#6][NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[#6][NX2]=[CX2]=[SX1]",

    # === N-OXIDES ===
    "n_oxide_aromatic": "[n+][O-]",
    "n_oxide_aliphatic": "[NX4+]([#6])([#6])([#6])[O-]",

    # === BORONIC ACIDS ===
    "boronic_acid": "[#6][BX3]([OX2H])([OX2H])",

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

    # === PHOSPHORUS COMPOUNDS (check more specific first) ===
    # Phosphate esters (C-O-P bonds, not C-P bonds) - check before phosphine oxide
    "phosphate_triester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2][#6]",
    "phosphate_diester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2H1]",
    "phosphate_monoester": "[PX4](=O)([OX2][#6])([OX2H1])[OX2H1]",
    # Phosphine oxide: R3P=O - three C attached to P(V)
    "phosphine_oxide": "[PX4](=O)([#6])([#6])[#6]",
    # Phosphines (P(III)) - check last as parent hydride
    "tertiary_phosphine": "[PX3]([#6])([#6])[#6]",
    "secondary_phosphine": "[PX3H1]([#6])[#6]",
    "primary_phosphine": "[PX3H2][#6]",

    # === SULFUR OXIDATION STATES (check more specific first) ===
    # Sulfone: S with 2 =O and 2 C neighbors (R-SO2-R')
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    # Sulfoxide: S with 1 =O and 2 C neighbors (R-SO-R')
    "sulfoxide": "[SX3](=[OX1])([#6])[#6]",
    
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
    
    # Post-processing: remove generic FG matches that overlap with more-specific FGs
    results = _resolve_fg_collisions(results)

    return dict(results)


def _resolve_fg_collisions(results):
    """Remove generic FG matches that overlap with more-specific FGs.

    Collision rules:
    - urea atoms should NOT also be detected as primary_amide/secondary_amide/tertiary_amide
    - guanidine atoms should NOT also be detected as imine
    - carbamate atoms should NOT also be detected as ester or amide (primary/secondary/tertiary)
    - isocyanate/isothiocyanate atoms should NOT also be detected as nitrile or primary_amide
    """
    for fg_specific, fg_generic_list in [
        ('urea', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('guanidine', ['imine']),
        ('carbamate', ['ester', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('isocyanate', ['nitrile', 'primary_amide']),
        ('isothiocyanate', ['nitrile', 'primary_amide']),
        # Carbamic acid: N-C(=O)-OH must NOT also match carboxylic_acid or amide
        ('carbamic_acid', ['carboxylic_acid', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        # Thiocarboxylic acids: SH in C(=O)SH or C(=S)SH must NOT match thiol
        # C(=O)SH must NOT match thioester either (C(=O)S is substructure of both)
        ('thioic_S_acid', ['thiol', 'thioester']),
        ('dithioic_acid', ['thiol', 'thioketone']),
        # C(=S)OH should not collide with carboxylic_acid (different SMARTS: =S vs =O)
        # but suppress thioketone matches on the C=S carbon
        ('thioic_O_acid', ['thioketone']),
    ]:
        if fg_specific in results:
            specific_atoms = set()
            for match in results[fg_specific]:
                specific_atoms.update(match)

            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    # Remove generic matches where ANY atom overlaps with specific
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if not any(atom in specific_atoms for atom in m)
                    ]
                    # Clean up empty lists
                    if not results[fg_generic]:
                        del results[fg_generic]

    # Phosphate specificity: more-specific phosphate esters suppress
    # less-specific phosphate esters for the same P atom.
    # NOTE: phosphonic_acid is NOT suppressed -- it remains as principal group
    # candidate for molecules with C-O-P(=O)(OH)2 because the naming pipeline
    # uses its "phosphono" prefix. Full phosphate principal group naming is
    # deferred to a future phase.
    _phosphate_suppress = [
        ('phosphate_triester', ['phosphate_diester', 'phosphate_monoester']),
        ('phosphate_diester', ['phosphate_monoester']),
    ]
    for fg_specific, fg_generic_list in _phosphate_suppress:
        if fg_specific in results:
            # Collect P atom indices from specific matches (P is always first atom in SMARTS)
            specific_p_atoms = {m[0] for m in results[fg_specific]}
            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if m[0] not in specific_p_atoms
                    ]
                    if not results[fg_generic]:
                        del results[fg_generic]

    return results


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
