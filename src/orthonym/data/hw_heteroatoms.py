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

IUPAC priority, Table 2.4's own "decreasing order of seniority" [BBv2:8236]:
  F > Cl > Br > I > O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb
    > B > Al > Ga > In > Tl

Reference: IUPAC 2013 Blue Book, **Table 2.4** (P-22.2.2.1.1) [BBv2:8234-8250].
    Earlier revisions of this docstring cited "Table 2.3"; that table is the
    retained-name morpholine entry, not the Hantzsch-Widman prefix table.
    Corrected against the book.
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

    # Group 13. Al/Ga/In/Tl are the four elements P-22.2.2 [BBv2:8218] ADDED to
    # the recommended Hantzsch-Widman system. Their Table 2.4 spellings are
    # transcribed verbatim from [BBv2:8244-8248] and TWO of them deliberately
    # differ from Table 1.5: the book prints ``aluminium | 3 | aluma`` with the
    # parenthetical "(not alumina)" and ``indium | 3 | indiga`` with "(not
    # inda)", under footnote 1 "Compare with Table 1.5" [BBv2:8250]. Ga and Tl
    # are spelled the same in both tables. The divergence is a Blue Book
    # requirement -- a caller in a Table-1.5 context (von Baeyer, spiro, chain,
    # ring > 10) must NOT read these; it reads
    # ``rules/ring_replacement.HETEROATOM_PREFIXES``.
    'B': 'bora',
    'Al': 'aluma',
    'Ga': 'galla',
    'In': 'indiga',
    'Tl': 'thalla',

    # NOTE: mercury is deliberately ABSENT. P-22.2.2 [BBv2:8218]: "The elements
    # aluminium, gallium, indium, and thallium are now included in the recommended
    # Hantzsch-Widman system and mercury has been deleted." Hg appears in neither
    # Table 1.5 nor Table 2.4 (only in the Appendix 1 *seniority* list), so
    # offering ``mercura`` here spelled a prefix the replacement system does not
    # have. Organomercury is named by P-69 organometallic nomenclature, which
    # keeps its own ``data/organometallics.METALLACYCLE_A_PREFIX`` -- that is the
    # legitimate home of Hg/Zn/Cd and is untouched by this removal.
    #
    # Al/Ga/In/Tl -- which P-22.2.2 ADDED to Hantzsch-Widman in the same sentence
    # -- are now listed above, per element against [BBv2:8244-8248].
}


# Heteroatom priority for ring numbering
# Lower number = higher priority (gets position 1)
# Table 2.4 order [BBv2:8236]: F > Cl > Br > I > O > S > Se > Te > N > P > As
#   > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl. Identical, element for
#   element, to P-23.3.1's von Baeyer citation order [BBv2:9765] -- which is why
#   the Table-1.5 spiro sites that sort with this table have never disagreed with
#   ``ring_replacement.HETEROATOM_PREFIXES``'s ranks. Only the SPELLINGS diverge
#   (aluma/indiga), and those come from HW_PREFIXES above, never from here.
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

    # Group 13 (lowest priority). In and Tl complete Table 2.4's order, whose
    # last four entries are Al > Ga > In > Tl [BBv2:8244-8248]. They were MISSING
    # while B..Ga were present, so ``get_heteroatom_priority`` fell through to 999
    # for both -- which made every In/Tl pair a TIE. ``sort_heteroatoms_by_priority``
    # is a stable sort over a list built by iterating a ``set`` of ring atoms, so a
    # tie there produced a citation order that depended on set iteration: a
    # nondeterministic name. Reachable as soon as either element can be spelled at
    # all, which it now can (Table 2.4 above, and Table 1.5 in ``ring_replacement``).
    'B': 14,
    'Al': 15,
    'Ga': 16,
    'In': 17,
    'Tl': 18,

    # NOT a replacement-seniority position: Hg has no prefix in Table 1.5 OR
    # Table 2.4 (P-22.2.2 deleted it), so no name can ever cite it from here and
    # this rank is unreachable in every replacement path. Kept only so the value
    # stays defined for the P-69 organometallic consumers, and moved past In/Tl so
    # those two could take their real Table 2.4 positions. Do not read it as
    # sanctioning ``mercura`` in replacement nomenclature -- see HW_PREFIXES.
    'Hg': 19,
}


def get_hw_prefix(element: str) -> Optional[str]:
    """
    Get the Hantzsch-Widman 'a' term prefix for a heteroatom.

    Args:
        element: Element symbol (e.g., 'O', 'N', 'S')

    Returns:
        HW prefix string (e.g., 'oxa', 'aza', 'thia'), or None if not found.

    ``None`` means the Hantzsch-Widman system has no prefix for this element, so
    **the caller must refuse** -- it must not skip the atom and emit the ring stem
    anyway. Skipping is what produced ``inane`` / ``epane`` / ``olane`` for
    aluminium rings: the heteroatom vanished from the name while the HW stem still
    counted it toward the ring size.

    This is Table 2.4 (P-22.2.2.1.1), the Hantzsch-Widman context only. General
    skeletal replacement -- von Baeyer, spiro, chains, rings > 10 -- uses
    Table 1.5 via ``rules/ring_replacement.HETEROATOM_PREFIXES``, which spells Al
    and In differently on purpose (``alumina``/``inda`` vs ``aluma``/``indiga``,
    [BBv2:8245] printing ``aluma`` with an explicit "(not alumina)"). The two
    tables must not be merged.

    Examples:
        >>> get_hw_prefix('O')
        'oxa'
        >>> get_hw_prefix('N')
        'aza'
        >>> get_hw_prefix('C')  # Carbon has no HW prefix
        None
        >>> get_hw_prefix('Hg')  # deleted from HW by P-22.2.2
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
