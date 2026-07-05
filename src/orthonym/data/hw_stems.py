"""
Hantzsch-Widman stem suffix data for systematic heterocycle naming.

The Hantzsch-Widman (HW) system uses stem suffixes to indicate ring size
and saturation state. The suffix is appended after the heteroatom prefixes
to form the complete heterocycle name.

Suffix structure:
  [heteroatom prefix] + [stem suffix] = heterocycle name
  e.g., oxa + ole = oxole (unsaturated 5-membered O-heterocycle)
        aza + iridine = aziridine (saturated 3-membered N-heterocycle)

Special cases for 6-membered rings:
  - Saturated rings with O, S, Se, Te, Bi, Hg use '-ane' (oxane, thiane)
  - Saturated rings with N, Si, Ge, Sn, Pb, B, P use '-inane' (azinane)

The 'n_saturated' variants use '-idine' suffixes for N-containing saturated
rings (e.g., pyrrolidine = 5-membered saturated N-ring, olidine stem).
Note: Many N-saturated heterocycles have retained names (piperidine, morpholine).

Reference: IUPAC 2013 Blue Book, Table 2.2 and Section P-22.2.2.1
Source: Derived from OPSIN hwSuffixes.xml
"""

from typing import Dict, Optional, Set, Any


# Hantzsch-Widman stem suffixes by ring size
# Keys: ring size (3-10)
# Values: dict with saturation variants
#   - 'unsaturated': Maximum unsaturation for ring size
#   - 'saturated': Standard saturated suffix (or 'saturated_os' / 'saturated_n' for 6-membered)
#   - 'n_saturated': Alternative suffix for N-containing saturated rings (optional)
HW_STEMS: Dict[int, Dict[str, str]] = {
    3: {
        'unsaturated': 'irene',      # e.g., oxirene (non-N or mixed heteroatoms)
        # Wave2 T2d (P-22.2.2.1.5.1): a 3-membered MANCUDE ring whose only
        # heteroatoms are nitrogen uses 'irine' (1H-/2H-azirine, diazirine),
        # not 'irene'. Selected in build_hw_name (needs the full heteroatom
        # set, which get_hw_stem's single-element signature cannot express).
        'n_unsaturated': 'irine',
        'saturated': 'irane',        # e.g., oxirane, aziridine
        'n_saturated': 'iridine',    # e.g., aziridine (N-containing)
    },
    4: {
        'unsaturated': 'ete',        # e.g., oxete, azete
        'saturated': 'etane',        # e.g., oxetane, azetidine
        'n_saturated': 'etidine',    # e.g., azetidine (N-containing)
    },
    5: {
        'unsaturated': 'ole',        # e.g., oxole, azole, thiole
        'saturated': 'olane',        # e.g., oxolane (THF), thiolane
        'n_saturated': 'olidine',    # e.g., pyrrolidine (aza-olidine)
    },
    6: {
        'unsaturated': 'ine',        # e.g., azine (pyridine); oxine overridden by retained "2H-pyran"
        'saturated_os': 'ane',       # For O, S, Se, Te, Bi, Hg: oxane, thiane
        'saturated_n': 'inane',      # For N, Si, Ge, Sn, Pb, B, P: azinane
    },
    7: {
        'unsaturated': 'epine',      # e.g., oxepine, azepine
        'saturated': 'epane',        # e.g., oxepane, azepane
    },
    8: {
        'unsaturated': 'ocine',      # e.g., oxocine, azocine
        'saturated': 'ocane',        # e.g., oxocane, azocane
    },
    9: {
        'unsaturated': 'onine',      # e.g., oxonine, azonine
        'saturated': 'onane',        # e.g., oxonane, azonane
    },
    10: {
        'unsaturated': 'ecine',      # e.g., oxecine, azecine
        'saturated': 'ecane',        # e.g., oxecane, azecane
    },
}


# Heteroatoms that use '-ane' for saturated 6-membered rings
# (instead of '-inane')
HETEROATOMS_USE_ANE: Set[str] = {'O', 'S', 'Se', 'Te', 'Bi', 'Hg'}


# Heteroatoms that use '-inane' for saturated 6-membered rings
HETEROATOMS_USE_INANE: Set[str] = {'N', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'P', 'As', 'Sb'}


def get_hw_stem(ring_size: int, is_saturated: bool, heteroatom: str = 'O') -> Optional[str]:
    """
    Get the Hantzsch-Widman stem suffix for a heterocycle.

    Args:
        ring_size: Number of atoms in the ring (3-10)
        is_saturated: True for saturated, False for unsaturated
        heteroatom: Principal heteroatom symbol (used for 6-membered ring
                    suffix selection: O/S use 'ane', N/P/Si use 'inane')

    Returns:
        HW stem suffix string, or None if ring size not supported

    Examples:
        >>> get_hw_stem(5, False)  # Unsaturated 5-ring
        'ole'
        >>> get_hw_stem(5, True)   # Saturated 5-ring
        'olane'
        >>> get_hw_stem(6, True, 'O')  # Saturated 6-ring with O
        'ane'
        >>> get_hw_stem(6, True, 'N')  # Saturated 6-ring with N
        'inane'
        >>> get_hw_stem(3, True, 'N')  # Saturated 3-ring with N
        'iridine'
    """
    if ring_size not in HW_STEMS:
        return None

    stems = HW_STEMS[ring_size]

    if not is_saturated:
        return stems.get('unsaturated')

    # Saturated case - handle special cases
    if ring_size == 6:
        # 6-membered rings have different suffixes based on heteroatom
        if heteroatom in HETEROATOMS_USE_ANE:
            return stems.get('saturated_os')
        else:
            return stems.get('saturated_n')

    # For 3, 4, 5-membered saturated rings with nitrogen
    # Check if N-specific variant exists and is requested
    if heteroatom == 'N' and 'n_saturated' in stems:
        return stems.get('n_saturated')

    # Default saturated suffix
    return stems.get('saturated')


def get_unsaturated_stem(ring_size: int) -> Optional[str]:
    """
    Get the unsaturated HW stem suffix for a ring size.

    Args:
        ring_size: Number of atoms in the ring (3-10)

    Returns:
        Unsaturated stem suffix, or None if not supported

    Examples:
        >>> get_unsaturated_stem(5)
        'ole'
        >>> get_unsaturated_stem(6)
        'ine'
    """
    if ring_size not in HW_STEMS:
        return None
    return HW_STEMS[ring_size].get('unsaturated')


def get_saturated_stem(ring_size: int, heteroatom: str = 'C') -> Optional[str]:
    """
    Get the saturated HW stem suffix for a ring size.

    Args:
        ring_size: Number of atoms in the ring (3-10)
        heteroatom: Principal heteroatom (affects 6-membered ring suffix)

    Returns:
        Saturated stem suffix, or None if not supported

    Examples:
        >>> get_saturated_stem(5)
        'olane'
        >>> get_saturated_stem(6, 'O')
        'ane'
        >>> get_saturated_stem(6, 'N')
        'inane'
    """
    return get_hw_stem(ring_size, is_saturated=True, heteroatom=heteroatom)


def is_supported_ring_size(ring_size: int) -> bool:
    """
    Check if a ring size is supported by HW nomenclature.

    Args:
        ring_size: Number of atoms in the ring

    Returns:
        True if ring size 3-10 (HW supported range)

    Examples:
        >>> is_supported_ring_size(5)
        True
        >>> is_supported_ring_size(11)
        False
    """
    return ring_size in HW_STEMS


def get_supported_ring_sizes() -> list:
    """
    Get list of ring sizes supported by HW nomenclature.

    Returns:
        List of integers [3, 4, 5, 6, 7, 8, 9, 10]
    """
    return sorted(HW_STEMS.keys())
