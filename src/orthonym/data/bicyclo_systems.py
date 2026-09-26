"""
Retained names and data for bicyclic ring systems.

Provides lookup tables for bicyclic compounds that have retained
(trivial) names preferred over systematic bicyclo[x.y.z] nomenclature.

IUPAC Reference: Blue Book 2013, (Bridged bicyclic hydrocarbons)

The bicyclo descriptor format is bicyclo[x.y.z] where:
- x, y, z are the number of atoms in each bridge (excluding bridgeheads)
- Values are sorted descending: x >= y >= z
- Total carbons = x + y + z + 2 (the +2 accounts for bridgehead carbons)
"""

from typing import Dict, Optional

# ============================================================================
# Retained Names for Bicyclic Systems
# ============================================================================

# Mapping from canonical SMILES to retained names
# Keys MUST be in RDKit canonical SMILES format
BICYCLO_RETAINED_NAMES: Dict[str, str] = {
    # NO norbornane / norbornene rows. "RETAINED NAMES FOR VON BAEYER
    # PARENT HYDRIDES" (the Blue Book; sentence:9881): "The retained names
    # adamantane and cubane are used in general nomenclature and as preferred
    # IUPAC names." No other von Baeyer hydride keeps a retained PIN. The Blue
    # Book's only 'norbornane' is the subtractive '10-norbornane' of bornane in
    # Table 1.2, next to 'bicyclo[2.2.1]heptane (PIN)' (:2037-2038). The rows
    # "C1CC2CCC1C2": "norbornane", "C1=CC2CCC1C2": "norbornene"
    # made the PIN tier emit the non-PIN trivial name (gold row
    # DD7-bridged-protect-norbornane). With them gone the von Baeyer namer
    # emits 'bicyclo[2.2.1]heptane' / 'bicyclo[2.2.1]hept-2-ene'.

    # === Bicyclo[1.1.0]butane ===
    # Smallest bicyclic hydrocarbon (4 carbons)
    # Note: Strained system, used in drug design
    "C1C2CC12": "bicyclo[1.1.0]butane",

    # === Bornane : 1,7,7-trimethylbicyclo[2.2.1]heptane ===
    # Terpene scaffold retained name per IUPAC
    "CC12CCC(CC1)C2(C)C": "bornane",

    # === Pinane : 2,6,6-trimethylbicyclo[3.1.1]heptane ===
    # Monoterpene scaffold retained name per IUPAC
    "CC1CCC2CC1C2(C)C": "pinane",

    # === Heterobicyclics with retained names ===
    # PA1 R6 /: the row `"C1CC2CCC1CN2": "quinuclidine"` was DELETED here, for
    # two independent reasons, either of which is sufficient.
    #
    # (1) WRONG STRUCTURE. That SMILES is not quinuclidine. Both strings are
    # already RDKit-canonical, so this was not a harmless dead-key typo but a
    # canonical key pointing at a different molecule:
    # C1CC2CCC1CN2 InChIKey KPUSZZFAYGWAHZ 2-azabicyclo[2.2.2]octane
    # (isoquinuclidine)
    # C1CN2CCC1CC2 InChIKey SBYHFKPVCBCYGV quinuclidine (1-aza)
    # Confirmed by building 2-azabicyclo[2.2.2]octane independently from
    # `C1CC2CCC1NC2`, which reproduces KPUSZZFAYGWAHZ. Both are C7H13N
    # isomers, which is why the error survived review. Contrary to the audit
    # that found it, the key was NOT unreachable: `get_retained_bicyclo_name`
    # is live via rules/bicyclo.py and rules/polycyclic.py, and
    # `C1CC2CCC1CN2` emitted the wrong name "quinuclidine" until this
    # deletion. Same bug class as the `cubane` note below.
    #
    # (2) NON-PIN. Even re-keyed to the correct SMILES the row must not exist:
    # the Blue Book "The name quinuclidine is retained for general
    # nomenclature only"; the Blue Book "quinuclidine 1-azabicyclo[2.2.2]octane
    # (PIN)". This table is NOT gated by the iupac_2013_pin_list.json deny
    # set, so an entry here would bypass that governance and re-introduce the
    # non-PIN headline the deny was added to stop. The von Baeyer namer
    # already emits the PIN `1-azabicyclo[2.2.2]octane`, and the trivial name
    # stays reachable under --trivial via GENERAL_RETAINED_NAMES.

    # === Decalin (bicyclo[4.4.0]decane / decahydronaphthalene) ===
    # Retained name per IUPAC 2013
    # OPSIN RT verified 2026-03-08
    "C1CCC2CCCCC2C1": "decalin",

    # === Adamantane: tricyclo[3.3.1.1(3,7)]decane retained name per IUPAC ===
    "C1C2CC3CC1CC(C2)C3": "adamantane",

    # === Cubane: pentacyclo[4.2.0.0(2,5).0(3,8).0(4,7)]octane; retained name AND PIN
    # per IUPAC / Blue Book line 9881 ("adamantane and cubane... as
    # preferred IUPAC names"). a phase: rekeyed from the stale C12C3C4C1C1C3C2C41
    # (InChIKey BOLISNSTKUABPW -- a DIFFERENT (CH)8 cage isomer, NOT cubane) to the true
    # cubane canonical SMILES (InChIKey TXWRERCHRDBNLG, what OPSIN emits for both "cubane"
    # and the systematic pentacyclo name). The old key mislabelled the wrong isomer as
    # cubane and left true cubane named systematically. OPSIN-RT verified. ===
    "C12C3C4C1C1C2C3C41": "cubane",
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
        >>> get_retained_bicyclo_name("C1C2CC3CC1CC(C2)C3")
        'adamantane'
        >>> get_retained_bicyclo_name("C1CC2CCC1C2") # bicyclo[2.2.1]heptane,
        None
        >>> get_retained_bicyclo_name("C1CC2CCC1CN2") # isoquinuclidine, PA1 R6
        None
        >>> get_retained_bicyclo_name("C1CC2CCC1CC2") # bicyclo[2.2.2]octane
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
        >>> is_retained_bicyclo("C1C2CC3CC1CC(C2)C3") # adamantane
        True
        >>> is_retained_bicyclo("C1CC2CCC1C2") # bicyclo[2.2.1]heptane,
        False
        >>> is_retained_bicyclo("C1CC2CCC1CC2") # bicyclo[2.2.2]octane
        False
    """
    return canonical_smiles in BICYCLO_RETAINED_NAMES
