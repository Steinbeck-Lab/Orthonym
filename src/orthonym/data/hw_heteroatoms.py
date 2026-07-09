"""
Hantzsch-Widman heteroatom prefix data for systematic heterocycle naming.

The Hantzsch-Widman (HW) system uses 'a' term prefixes to denote heteroatoms
in systematic ring nomenclature. These prefixes replace the standard carbon
skeleton terminology when heteroatoms are present in the ring.

Prefix naming convention: Element symbol -> 'a' term
  - Oxygen (O) -> oxa
  - Nitrogen (N) -> aza
  - Sulfur (S) -> thia
  - etc.

Priority ordering: Used to determine which heteroatom gets position 1 in
ring numbering when multiple different heteroatoms are present. Lower
priority number = higher priority = gets position 1 or lower locant.

IUPAC priority (Group 16 > Group 15 > Group 14 > Group 13):
  O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B

Reference: IUPAC 2013 Blue Book, Table 2.3 and Section P-22.2.2.1
Source: Derived from OPSIN hwHeteroAtoms.xml
"""

from typing import Dict, Optional


# Hantzsch-Widman 'a' term prefixes for heteroatoms
# Maps element symbol to the standard HW prefix
HW_PREFIXES: Dict[str, str] = {
    # Group 17 (halogens — only reachable in lambda-convention rings,
    # P-22.2.7.1: 1lambda3-iodinane; standard-valence halogens cannot be
    # skeletal ring atoms)
    'F': 'fluora',
    'Cl': 'chlora',
    'Br': 'broma',
    'I': 'ioda',

    # Group 16 (Chalcogens)
    'O': 'oxa',
    'S': 'thia',
    'Se': 'selena',
    'Te': 'tellura',

    # Group 15 (Pnictogens)
    'N': 'aza',
    'P': 'phospha',
    'As': 'arsa',
    'Sb': 'stiba',
    'Bi': 'bisma',

    # Group 14
    'Si': 'sila',
    'Ge': 'germa',
    'Sn': 'stanna',
    'Pb': 'plumba',

    # Group 13
    'B': 'bora',

    # Additional elements (less common)
    'Hg': 'mercura',
}


# Heteroatom priority for ring numbering
# Lower number = higher priority (gets position 1)
# IUPAC order: F > Cl > Br > I > O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B
# Phase 151-04 WR-01: halogens included per IUPAC P-25.3.1.3 (skeletal
# replacement nomenclature priority). Negative values keep them strictly
# more senior than O without renumbering downstream callers that depend
# on the relative ordering of O..B established before halogens were added.
HETEROATOM_PRIORITY: Dict[str, int] = {
    # Group 17 (halogens — most senior per IUPAC P-25.3.1.3)
    'F': -4,
    'Cl': -3,
    'Br': -2,
    'I': -1,

    # Group 16 (highest priority among non-halogens)
    'O': 1,
    'S': 2,
    'Se': 3,
    'Te': 4,

    # Group 15
    'N': 5,
    'P': 6,
    'As': 7,
    'Sb': 8,
    'Bi': 9,

    # Group 14
    'Si': 10,
    'Ge': 11,
    'Sn': 12,
    'Pb': 13,

    # Group 13 (lowest priority)
    'B': 14,
    'Al': 15,
    'Ga': 16,

    # Additional
    'Hg': 17,
}


def get_hw_prefix(element: str) -> Optional[str]:
    """
    Get the Hantzsch-Widman 'a' term prefix for a heteroatom.

    Args:
        element: Element symbol (e.g., 'O', 'N', 'S')

    Returns:
        HW prefix string (e.g., 'oxa', 'aza', 'thia'), or None if not found

    Examples:
        >>> get_hw_prefix('O')
        'oxa'
        >>> get_hw_prefix('N')
        'aza'
        >>> get_hw_prefix('C')  # Carbon has no HW prefix
        None
    """
    return HW_PREFIXES.get(element)


def get_heteroatom_priority(element: str) -> int:
    """
    Get the IUPAC priority for a heteroatom (for ring numbering).

    Lower priority number = higher priority = gets position 1 or lower locant.
    Unknown elements return 999 (lowest priority).

    Args:
        element: Element symbol (e.g., 'O', 'N', 'S')

    Returns:
        Priority integer (1 = highest priority)

    Examples:
        >>> get_heteroatom_priority('O')
        1
        >>> get_heteroatom_priority('N')
        5
        >>> get_heteroatom_priority('O') < get_heteroatom_priority('N')
        True
        >>> get_heteroatom_priority('X')  # Unknown element
        999
    """
    return HETEROATOM_PRIORITY.get(element, 999)


def compare_heteroatom_priority(element1: str, element2: str) -> int:
    """
    Compare two heteroatoms by IUPAC priority.

    Args:
        element1: First element symbol
        element2: Second element symbol

    Returns:
        -1 if element1 has higher priority (lower number)
         0 if same priority
         1 if element2 has higher priority

    Examples:
        >>> compare_heteroatom_priority('O', 'N')
        -1  # O has higher priority
        >>> compare_heteroatom_priority('N', 'O')
        1   # O has higher priority
        >>> compare_heteroatom_priority('O', 'O')
        0   # Same priority
    """
    p1 = get_heteroatom_priority(element1)
    p2 = get_heteroatom_priority(element2)

    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    else:
        return 0


def sort_heteroatoms_by_priority(elements: list) -> list:
    """
    Sort heteroatom element symbols by IUPAC priority (highest first).

    Args:
        elements: List of element symbols

    Returns:
        List sorted by priority (O before N before Si, etc.)

    Examples:
        >>> sort_heteroatoms_by_priority(['N', 'O', 'S'])
        ['O', 'S', 'N']
    """
    return sorted(elements, key=get_heteroatom_priority)
