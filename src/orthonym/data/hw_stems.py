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

from typing import Any, Dict, Optional, Set, Tuple


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
        # P-22.2.2.1.3 / Table 2.7: the 6-membered stem ending depends on the
        # heteroatom CLASS present in the ring:
        #   6A (O, S, Se, Te, Bi): unsaturated 'ine',   saturated 'ane'
        #   6B (N, Si, Ge, Sn, Pb): unsaturated 'ine',  saturated 'inane'
        #   6C (F, Cl, Br, I, P, As, Sb, B, Al, Ga, In, Tl):
        #                           unsaturated 'inine', saturated 'inane'
        # Precedence for stem selection when classes mix: 6C > 6B > 6A.
        'unsaturated': 'ine',        # 6A/6B unsaturated: azine (pyridine); oxine overridden by retained "2H-pyran"
        'unsaturated_6c': 'inine',   # 6C unsaturated: 1,4-oxaphosphinine, 1,3,5-triphosphinine
        'saturated_os': 'ane',       # 6A: O, S, Se, Te, Bi, Hg -> oxane, thiane
        'saturated_n': 'inane',      # 6B/6C: N, Si, Ge, Sn, Pb, B, P, As, Sb -> azinane
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
# (6B ∪ 6C: N, Si, Ge, Sn, Pb  +  P, As, Sb, B, Al, Ga, In, Tl, and halogens)
HETEROATOMS_USE_INANE: Set[str] = {'N', 'Si', 'Ge', 'Sn', 'Pb',
                                   'P', 'As', 'Sb', 'B', 'Al', 'Ga', 'In', 'Tl',
                                   'F', 'Cl', 'Br', 'I'}


# P-22.2.2.1.3 / Table 2.7 class 6C: these heteroatoms give a 6-membered
# UNSATURATED (mancude) ring the '-inine' ending (not '-ine'). Saturated form
# is still '-inane' (shared with 6B).
#
# NOTE: these two flat sets are kept for backwards compatibility with callers
# that have only a single heteroatom. They encode a "which group is this ONE
# atom in" question, which is NOT the rule for a ring with several kinds of
# heteroatom -- see SIX_RING_CITATION_SENIORITY below.
HETEROATOMS_USE_ININE: Set[str] = {'F', 'Cl', 'Br', 'I',
                                   'P', 'As', 'Sb', 'B', 'Al', 'Ga', 'In', 'Tl'}


# P-22.2.2.1.3 (BlueBookV2.md:8284): "If two or more kinds of heteroatoms occur
# in the same name, their order of citation follows the sequence: F, Cl, Br, I,
# O, S, Se, Te, N, P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl."
#
# Most senior first, so the LEAST senior heteroatom of a ring is the one
# occurring LAST in this sequence -- equivalently the one whose 'a' prefix is
# written immediately before the stem, which is how P-22.2.2.1.6 phrases it.
SIX_RING_CITATION_SENIORITY: Tuple[str, ...] = (
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As',
    'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl',
)

# Table 2.5 (BlueBookV2.md:8259-8261). The three groups partition exactly the 22
# elements of SIX_RING_CITATION_SENIORITY (5 + 5 + 12), which is the consistency
# check that the two tables describe the same element set.
#
# Bi is group A, NOT group C -- it sits in the sequence between Sb and Si, so a
# ring pairing As or Sb with Bi takes the group-A stem.
SIX_RING_GROUP_A: Set[str] = {'O', 'S', 'Se', 'Te', 'Bi'}
SIX_RING_GROUP_B: Set[str] = {'N', 'Si', 'Ge', 'Sn', 'Pb'}
SIX_RING_GROUP_C: Set[str] = {'F', 'Cl', 'Br', 'I', 'P', 'As', 'Sb',
                              'B', 'Al', 'Ga', 'In', 'Tl'}


def least_senior_six_ring_heteroatom(ring_heteroatoms: Set[str]) -> Optional[str]:
    """The heteroatom that selects a six-membered ring's stem, or None.

    P-22.2.2.1.6 (BlueBookV2.md:8411), heading "Selecting Hantzsch-Widman names
    for six-membered rings": "The stem for six-membered rings depends on the
    least senior heteroatom in the ring, i.e., the heteroatom whose name directly
    precedes the stem. ... The stem is selected in accordance with the group to
    which the least senior heteroatom belongs."

    Returns None when the ring carries no heteroatom that Table 2.5 classifies
    (Hg/Zn/Cd are in no group and in no citation sequence). Callers must then
    fall back rather than treat an unlisted element as least senior -- promoting
    one would silently change a listed atom's stem.
    """
    listed = [e for e in ring_heteroatoms if e in SIX_RING_CITATION_SENIORITY]
    if not listed:
        return None
    return max(listed, key=SIX_RING_CITATION_SENIORITY.index)


def get_hw_stem(ring_size: int, is_saturated: bool, heteroatom: str = 'O',
                ring_heteroatoms: Optional[Set[str]] = None) -> Optional[str]:
    """
    Get the Hantzsch-Widman stem suffix for a heterocycle.

    Args:
        ring_size: Number of atoms in the ring (3-10)
        is_saturated: True for saturated, False for unsaturated
        heteroatom: Principal heteroatom symbol (used for 6-membered ring
                    suffix selection: O/S use 'ane', N/P/Si use 'inane')
        ring_heteroatoms: OPTIONAL set of ALL heteroatom symbols in the ring.
                    P-22.2.2.1.3 / Table 2.7 class 6C (P, As, Sb, B, halogens,
                    ...) gives an UNSATURATED 6-ring the '-inine' ending; that
                    depends on whether ANY 6C atom is present, not on the single
                    dominant heteroatom, so callers pass the full set. When
                    omitted, the single `heteroatom` is treated as the ring's
                    only heteroatom (backwards-compatible).

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
        >>> get_hw_stem(6, False, 'O', {'O', 'P'})  # unsaturated O+P 6-ring
        'inine'
        >>> get_hw_stem(3, True, 'N')  # Saturated 3-ring with N
        'iridine'
    """
    if ring_size not in HW_STEMS:
        return None

    stems = HW_STEMS[ring_size]

    # P-22.2.2.1.6 (BlueBookV2.md:8411): a SIX-membered ring's stem is selected
    # by the group of its LEAST SENIOR heteroatom -- not by the most senior one,
    # and not by "any group-C atom present". The two older heuristics agree with
    # the rule whenever the least senior heteroatom shares a group with the most
    # senior (which is why 1,3-oxazinane and 1,3-oxaselenane were already right)
    # and disagree otherwise: O+As gave 'oxarsane' instead of 1,3-oxarsinane
    # (PIN, :8455), and As+Bi would give the group-C stem though Bi, the least
    # senior, is group A.
    if ring_size == 6:
        check_set = ring_heteroatoms if ring_heteroatoms is not None else {heteroatom}
        least = least_senior_six_ring_heteroatom(check_set)
        if least is not None:
            if is_saturated:
                return stems.get('saturated_os' if least in SIX_RING_GROUP_A
                                 else 'saturated_n')
            return stems.get('unsaturated_6c' if least in SIX_RING_GROUP_C
                             else 'unsaturated')
        # No Table 2.5-classified heteroatom (Hg/Zn/Cd): fall through to the
        # single-atom sets rather than reclassify an unlisted element.

    if not is_saturated:
        if ring_size == 6 and (
                (ring_heteroatoms if ring_heteroatoms is not None
                 else {heteroatom}) & HETEROATOMS_USE_ININE):
            return stems.get('unsaturated_6c')
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
