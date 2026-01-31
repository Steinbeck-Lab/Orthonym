"""
Functional group seniority rules for principal group selection.

Based on IUPAC 2013 Blue Book P-41 to P-43.
The principal group (highest seniority) becomes the suffix;
all other groups become prefixes.
"""

from typing import Optional, Tuple, List, Dict

# Functional group seniority order (highest priority first)
# Groups in this list can be expressed as suffixes when principal
SENIORITY_ORDER = [
    # Acids (highest priority)
    "carboxylic_acid",
    "sulfonic_acid",
    "sulfinic_acid",
    "phosphonic_acid",
    
    # Acid derivatives
    "anhydride",
    "ester",
    "thioester",
    "acid_chloride",
    "acid_bromide",
    "acid_fluoride",
    
    # Nitrogen acid derivatives
    "primary_amide",
    "secondary_amide",
    "tertiary_amide",
    "hydrazide",
    "imide",
    
    # Nitriles
    "nitrile",
    "isocyanide",
    
    # Carbonyls
    "aldehyde",
    "ketone",
    "thioaldehyde",
    "thioketone",
    
    # Alcohols and analogs
    "primary_alcohol",
    "secondary_alcohol",
    "tertiary_alcohol",
    "phenol",
    "enol",
    "thiol",
    "selenol",
    
    # Hydroperoxides
    "hydroperoxide",
    
    # Amines
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "aromatic_amine",
    
    # Imines
    "imine",
    "oxime",
    "hydrazone",

    # Sulfur oxidation states (functional class naming, lower seniority than amines)
    "sulfoxide",
    "sulfone",
    "thioether",  # Also called sulfide
]

# Suffix forms for principal groups
# Format: (chain_terminal_suffix, ring_attached_suffix)
SUFFIX_FORMS = {
    "carboxylic_acid": ("oic acid", "carboxylic acid"),
    "sulfonic_acid": ("sulfonic acid", "sulfonic acid"),
    "sulfinic_acid": ("sulfinic acid", "sulfinic acid"),
    "phosphonic_acid": ("phosphonic acid", "phosphonic acid"),
    "anhydride": ("oic anhydride", "carboxylic anhydride"),
    "ester": ("oate", "carboxylate"),
    "acid_chloride": ("oyl chloride", "carbonyl chloride"),
    "acid_bromide": ("oyl bromide", "carbonyl bromide"),
    "acid_fluoride": ("oyl fluoride", "carbonyl fluoride"),
    "primary_amide": ("amide", "carboxamide"),
    "secondary_amide": ("amide", "carboxamide"),
    "tertiary_amide": ("amide", "carboxamide"),
    "nitrile": ("nitrile", "carbonitrile"),
    "aldehyde": ("al", "carbaldehyde"),
    "ketone": ("one", "one"),
    "thioaldehyde": ("thial", "carbothialdehyde"),
    "thioketone": ("thione", "thione"),
    "primary_alcohol": ("ol", "ol"),
    "secondary_alcohol": ("ol", "ol"),
    "tertiary_alcohol": ("ol", "ol"),
    "phenol": ("ol", "ol"),
    "enol": ("ol", "ol"),
    "thiol": ("thiol", "thiol"),
    "selenol": ("selenol", "selenol"),
    "primary_amine": ("amine", "amine"),
    "secondary_amine": ("amine", "amine"),
    "tertiary_amine": ("amine", "amine"),
    "aromatic_amine": ("amine", "amine"),
    "imine": ("imine", "imine"),
}

# Prefix forms for non-principal groups
PREFIX_FORMS = {
    "carboxylic_acid": "carboxy",
    "sulfonic_acid": "sulfo",
    "sulfinic_acid": "sulfino",
    "aldehyde": "oxo",  # or "formyl" for terminal
    "ketone": "oxo",
    "primary_alcohol": "hydroxy",
    "secondary_alcohol": "hydroxy",
    "tertiary_alcohol": "hydroxy",
    "phenol": "hydroxy",
    "enol": "hydroxy",
    "thiol": "sulfanyl",
    "selenol": "selanyl",
    "primary_amine": "amino",
    "secondary_amine": "amino",
    "tertiary_amine": "amino",
    "aromatic_amine": "amino",
    "imine": "imino",
    "oxime": "hydroxyimino",
    "nitrile": "cyano",
    "isocyanide": "isocyano",
    # Halogens (always prefixes)
    "fluoro": "fluoro",
    "chloro": "chloro",
    "bromo": "bromo",
    "iodo": "iodo",
    # Other
    "nitro": "nitro",
    "nitroso": "nitroso",
    "azido": "azido",
    # Ethers and thioethers
    "ether": None,  # Named by substitution: methoxy, ethoxy, etc.
    "thioether": None,
    # Sulfur oxidation states (functional class naming)
    "sulfoxide": None,  # Named by functional class (dimethyl sulfoxide)
    "sulfone": None,    # Named by functional class (dimethyl sulfone)
}


def get_principal_group(
    mol,
    functional_groups: Dict[str, List[tuple]]
) -> Tuple[Optional[str], List[tuple]]:
    """
    Determine the principal characteristic group.
    
    The principal group is the highest-seniority functional group
    that will be expressed as a suffix in the name.
    
    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()
        
    Returns:
        Tuple of (group_name, list_of_atom_index_tuples)
        Returns (None, []) if no suffix-capable group found
    """
    for fg_name in SENIORITY_ORDER:
        if fg_name in functional_groups and functional_groups[fg_name]:
            return fg_name, functional_groups[fg_name]
    
    return None, []


def get_suffix(fg_name: str, is_ring: bool = False) -> Optional[str]:
    """
    Get the suffix form for a functional group.
    
    Args:
        fg_name: Name of functional group
        is_ring: True if the group is attached to a ring (not terminal on chain)
        
    Returns:
        Suffix string, or None if group has no suffix form
    """
    if fg_name not in SUFFIX_FORMS:
        return None
    
    chain_suffix, ring_suffix = SUFFIX_FORMS[fg_name]
    return ring_suffix if is_ring else chain_suffix


def get_prefix(fg_name: str) -> Optional[str]:
    """
    Get the prefix form for a functional group.
    
    Used when the group is not the principal group.
    
    Args:
        fg_name: Name of functional group
        
    Returns:
        Prefix string, or None if group has no standard prefix form
    """
    return PREFIX_FORMS.get(fg_name)


def is_suffix_group(fg_name: str) -> bool:
    """
    Check if a functional group can be expressed as a suffix.
    
    Args:
        fg_name: Name of functional group
        
    Returns:
        True if group can be a suffix (is in seniority order)
    """
    return fg_name in SENIORITY_ORDER


def compare_seniority(fg1: str, fg2: str) -> int:
    """
    Compare seniority of two functional groups.
    
    Args:
        fg1, fg2: Names of functional groups
        
    Returns:
        -1 if fg1 is higher seniority
         0 if equal seniority (or both not in list)
         1 if fg2 is higher seniority
    """
    try:
        idx1 = SENIORITY_ORDER.index(fg1)
    except ValueError:
        idx1 = len(SENIORITY_ORDER)  # Not in list = lowest priority
    
    try:
        idx2 = SENIORITY_ORDER.index(fg2)
    except ValueError:
        idx2 = len(SENIORITY_ORDER)
    
    if idx1 < idx2:
        return -1  # fg1 is higher (lower index = higher priority)
    elif idx1 > idx2:
        return 1   # fg2 is higher
    return 0       # Equal


def get_all_prefix_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that should be named as prefixes.
    
    This includes all groups except the principal group.
    
    Args:
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of the principal group (or None)
        
    Returns:
        Dict of functional group names to their atom indices,
        excluding the principal group
    """
    prefix_groups = {}
    
    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if matches:
            prefix_groups[fg_name] = matches
    
    return prefix_groups
