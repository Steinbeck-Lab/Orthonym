"""
Retained names and data for bicyclic ring systems.

Provides lookup tables for bicyclic compounds that have retained
(trivial) names preferred over systematic bicyclo[x.y.z] nomenclature.

IUPAC Reference: Blue Book 2013, P-23.2 (Bridged bicyclic hydrocarbons)

The bicyclo descriptor format is bicyclo[x.y.z] where:
- x, y, z are the number of atoms in each bridge (excluding bridgeheads)
- Values are sorted descending: x >= y >= z
- Total carbons = x + y + z + 2 (the +2 accounts for bridgehead carbons)
"""

from typing import Optional, Dict


# ============================================================================
# Retained Names for Bicyclic Systems
# ============================================================================

# Mapping from canonical SMILES to retained names
# Keys MUST be in RDKit canonical SMILES format
BICYCLO_RETAINED_NAMES: Dict[str, str] = {
    # === Bicyclo[2.2.1]heptane (norbornane) ===
    # Most important bridged bicyclic - common in natural products
    "C1CC2CCC1C2": "norbornane",

    # === Bicyclo[2.2.1]hept-2-ene (norbornene) ===
    # Unsaturated norbornane, common in polymer chemistry
    "C1=CC2CCC1C2": "norbornene",

    # === Bicyclo[1.1.0]butane ===
    # Smallest bicyclic hydrocarbon (4 carbons)
    # Note: Strained system, used in drug design
    "C1C2CC12": "bicyclo[1.1.0]butane",

    # === Bornane (RING-04): 1,7,7-trimethylbicyclo[2.2.1]heptane ===
    # Terpene scaffold retained name per IUPAC P-31.1.3.4
    "CC12CCC(CC1)C2(C)C": "bornane",

    # === Pinane (RING-04): 2,6,6-trimethylbicyclo[3.1.1]heptane ===
    # Monoterpene scaffold retained name per IUPAC P-31.1.3.4
    "CC1CCC2CC1C2(C)C": "pinane",

    # === Heterobicyclics with retained names ===
    # Quinuclidine: 1-azabicyclo[2.2.2]octane
    # Important in alkaloid chemistry
    "C1CC2CCC1CN2": "quinuclidine",
}


# Mapping from descriptor to retained name (for systems named by descriptor)
# These are NOT true retained names but are commonly used
BICYCLO_DESCRIPTOR_NAMES: Dict[str, str] = {
    "bicyclo[2.2.1]": "bicyclo[2.2.1]",
    "bicyclo[2.2.2]": "bicyclo[2.2.2]",
    "bicyclo[3.2.1]": "bicyclo[3.2.1]",
    "bicyclo[1.1.0]": "bicyclo[1.1.0]",
    "bicyclo[1.1.1]": "bicyclo[1.1.1]",
    "bicyclo[2.1.0]": "bicyclo[2.1.0]",
}


# ============================================================================
# Lookup Functions
# ============================================================================

def get_retained_bicyclo_name(canonical_smiles: str) -> Optional[str]:
    """
    Get the retained name for a bicyclic compound if one exists.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        Retained name if exists, None otherwise

    Examples:
        >>> get_retained_bicyclo_name("C1CC2CCC1C2")
        'norbornane'
        >>> get_retained_bicyclo_name("C1CC2CCC1CN2")
        'quinuclidine'
        >>> get_retained_bicyclo_name("C1CC2CCC1CC2")  # bicyclo[2.2.2]octane
        None
    """
    return BICYCLO_RETAINED_NAMES.get(canonical_smiles)


def get_bicyclo_by_descriptor(descriptor: str) -> Optional[str]:
    """
    Get the standard descriptor format for a bicyclo system.

    Validates that the descriptor is in the correct format.

    Args:
        descriptor: Bicyclo descriptor string (e.g., "bicyclo[2.2.1]")

    Returns:
        Normalized descriptor if valid, None otherwise

    Examples:
        >>> get_bicyclo_by_descriptor("bicyclo[2.2.1]")
        'bicyclo[2.2.1]'
        >>> get_bicyclo_by_descriptor("invalid")
        None
    """
    return BICYCLO_DESCRIPTOR_NAMES.get(descriptor)


def is_retained_bicyclo(canonical_smiles: str) -> bool:
    """
    Check if a bicyclic compound has a retained name.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        True if compound has a retained name

    Examples:
        >>> is_retained_bicyclo("C1CC2CCC1C2")  # norbornane
        True
        >>> is_retained_bicyclo("C1CC2CCC1CC2")  # bicyclo[2.2.2]octane
        False
    """
    return canonical_smiles in BICYCLO_RETAINED_NAMES
