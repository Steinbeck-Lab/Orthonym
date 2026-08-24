"""
Locant validation module for IUPAC nomenclature.

Provides standalone validation functions that detect and resolve locant
conflicts between suffix groups, prefix substituents, and stereo descriptors
on ring and chain parent structures.

This module is pure data validation -- no RDKit dependency, no side effects.
It will be wired into the composer pipeline by Plan 17-03.

Key problem addressed:
    32 of 70 OPSIN parse failures are "unphysical valency" errors caused by
    suffix and prefix locants colliding on the same atom (e.g., ketone at C-3
    and chloro at C-3 on cyclohexanone producing an impossible valence).

Reference: IUPAC 2013 Blue Book, P-14.4, P-31.1
"""

from typing import Any, List, Optional, Tuple


def validate_suffix_locants(
    locants: List[int],
    parent_size: int,
    count: int,
) -> Tuple[List[int], int]:
    """
    Validate suffix locants against parent structure capacity.

    Filters out:
    - Locants <= 0 (not valid IUPAC; numbering is 1-indexed)
    - Locants > parent_size (cannot exist on the parent structure)

    Adjusts *count* to match the number of remaining valid locants.

    Args:
        locants: Suffix locant positions (e.g., [1, 3] for propane-1,3-diol).
        parent_size: Number of atoms in the parent ring or chain.
        count: Original multiplier count (e.g., 2 for "di", 3 for "tri").

    Returns:
        (filtered_locants, adjusted_count) -- both aligned to valid positions.

    Examples:
        >>> validate_suffix_locants([1, 3], parent_size=6, count=2)
        ([1, 3], 2)
        >>> validate_suffix_locants([1, 3, 8], parent_size=6, count=3)
        ([1, 3], 2)
        >>> validate_suffix_locants([0, 2, 4], parent_size=6, count=3)
        ([2, 4], 2)
    """
    filtered = [loc for loc in locants if 0 < loc <= parent_size]
    # When locants list is empty (terminal groups like diacids, dialdehydes),
    # preserve the original count. Only adjust count when we actually
    # had locants to filter.
    if not locants:
        return filtered, count
    return filtered, len(filtered)


def detect_locant_collisions(
    suffix_locants: List[int],
    prefix_locant_groups: List[List[int]],
    parent_type: str = "ring",
    parent_size: int = 0,
) -> List[Tuple[int, int]]:
    """
    Detect collisions between suffix locants and prefix locant groups.

    A collision occurs when a suffix locant (functional group position) and
    a prefix locant (substituent position) share the same numeric value on
    a **ring** system.  On chains, suffix and prefix atoms at the same
    locant number can coexist (different chemistry), so chain systems are
    skipped entirely.

    Args:
        suffix_locants: Locant positions for the principal characteristic
            group suffix (e.g., [1] for cyclohexan-1-one).
        prefix_locant_groups: List of locant lists, one per prefix substituent
            group (e.g., [[3], [5]] for 3-chloro-5-methyl-).
        parent_type: ``"ring"`` or ``"chain"``.  Chain systems skip detection.
        parent_size: Number of atoms in the parent (currently informational;
            reserved for future resolution strategies).

    Returns:
        List of ``(prefix_group_index, colliding_locant)`` tuples.
        Empty list if no collisions or if parent is a chain.

    Examples:
        >>> detect_locant_collisions([3], [[3]], parent_type="ring", parent_size=6)
        [(0, 3)]
        >>> detect_locant_collisions([1], [[3]], parent_type="chain", parent_size=5)
        []
    """
    if parent_type != "ring":
        return []

    if not suffix_locants or not prefix_locant_groups:
        return []

    suffix_set = set(suffix_locants)
    collisions: List[Tuple[int, int]] = []

    for group_idx, prefix_locants in enumerate(prefix_locant_groups):
        for loc in prefix_locants:
            if loc in suffix_set:
                collisions.append((group_idx, loc))

    return collisions


def validate_stereo_locants(
    stereo_descriptors: List[Tuple[Any, str]],
    parent_size: int,
) -> List[Tuple[Any, str]]:
    """
    Filter stereo descriptors to remove entries with invalid locants.

    Removes entries where:
    - Locant is 0 (not valid IUPAC)
    - Locant is negative (integer check)
    - Locant (if integer) exceeds parent_size

    String locants like ``'4a'`` are kept if their numeric base
    (leading digits) parses to a value <= parent_size.  This covers
    fused ring systems where positional labels include letter suffixes.

    Args:
        stereo_descriptors: List of ``(locant, descriptor)`` tuples where
            *locant* is ``int`` or ``str`` and *descriptor* is e.g.
            ``'R'``, ``'S'``, ``'E'``, ``'Z'``.
        parent_size: Number of atoms in the parent structure.

    Returns:
        Filtered list preserving original order.

    Examples:
        >>> validate_stereo_locants([(2, 'R'), (4, 'S')], parent_size=6)
        [(2, 'R'), (4, 'S')]
        >>> validate_stereo_locants([(0, 'R'), (3, 'S'), (99, 'R')], parent_size=6)
        [(3, 'S')]
    """
    result: List[Tuple[Any, str]] = []

    for locant, descriptor in stereo_descriptors:
        if _is_valid_locant(locant, parent_size):
            result.append((locant, descriptor))

    return result


def reconcile_multiplier_count(
    count: int,
    locants: List[int],
) -> int:
    """
    Ensure multiplier count matches locant count.

    OPSIN requires exact agreement between the multiplier prefix
    (di, tri, tetra) and the number of locants cited.  This function
    forces them to agree by using ``len(locants)`` as the canonical
    count.

    If *locants* is empty, the original *count* is preserved because
    some naming contexts use implicit locants (e.g., terminal acids
    on chains where locant-1 is omitted).

    Args:
        count: Current multiplier count.
        locants: Locant positions that were actually generated.

    Returns:
        Reconciled count.

    Examples:
        >>> reconcile_multiplier_count(count=3, locants=[1, 3])
        2
        >>> reconcile_multiplier_count(count=2, locants=[])
        2
    """
    if not locants:
        return count
    return len(locants)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_valid_locant(locant: Any, parent_size: int) -> bool:
    """
    Check whether a single locant value is valid for a parent of given size.

    Handles:
    - ``int`` locants: must be > 0 and <= parent_size
    - ``str`` locants (e.g., ``'4a'``): extract leading digits, validate
      the numeric portion against parent_size
    - Other types: rejected

    Returns:
        ``True`` if valid, ``False`` otherwise.
    """
    if isinstance(locant, int):
        return 0 < locant <= parent_size

    if isinstance(locant, str):
        base_num = _extract_base_number(locant)
        if base_num is not None:
            return 0 < base_num <= parent_size
        return False

    # v36-A2: a PRIMED multi-component locant (n, "'"/"''") — the 2nd component
    # of a spiro/fused name (e.g. (11, "'") for the tricyclo side of
    # spiro[oxolane-2,12'-tricyclo…]). It is numbered by its OWN component's
    # namer, so it must NOT be range-checked against parent_size, which reflects
    # only the UNPRIMED component's size. Accept when its base number is a
    # positive int (fail-closed on 0 / non-positive / malformed). 0-wrong is
    # still guaranteed downstream: a mis-attributed descriptor RT-fails → abstain.
    if isinstance(locant, tuple):
        base = locant[0] if locant and isinstance(locant[0], int) else None
        return base is not None and base > 0

    # Float, None, or other unexpected type
    return False


def _extract_base_number(locant_str: str) -> Optional[int]:
    """
    Extract the leading integer from a locant string like '4a' or '10b'.

    Returns:
        The integer portion, or ``None`` if no leading digits found.

    Examples:
        >>> _extract_base_number('4a')
        4
        >>> _extract_base_number('10b')
        10
        >>> _extract_base_number('abc')  # no leading digits
    """
    digits = []
    for ch in locant_str:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    if digits:
        return int("".join(digits))
    return None
