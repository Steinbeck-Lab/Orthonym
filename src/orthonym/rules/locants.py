"""
Locant assignment utilities for IUPAC nomenclature.

Locants map atom positions in the principal chain to 1-indexed IUPAC locant
numbers. This module provides:

1. build_atom_to_locant: Convert atom indices to locant mapping
2. compare_locant_sets: First-point-of-difference comparison
3. orient_chain: Apply IUPAC 2013 chain orientation criteria
4. get_functional_group_locants: Resolve FG atoms to chain locants

IUPAC 2013 chain orientation criteria (applied in order):
    a. Lowest locants for principal characteristic group
    b. Lowest locants for multiple bonds (as a set)
    c. Lowest locants for double bonds (if tie with triple bonds)
    d. Lowest locants for substituents (detachable prefixes)

Reference: IUPAC 2013 Blue Book, P-14.4, P-14.6, P-14.7
"""

from typing import Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem

# Phase 147 D-01/D-02: locant type system extension.
# Locants may be plain ints (e.g. 4) or (int_base, str_suffix) tuples for
# fusion atoms (e.g. '4a' -> (4, 'a')). The empty string '' sorts
# lexicographically before any letter, so (4, '') < (4, 'a') < (5, ''),
# matching the IUPAC convention that locant 4 is "lower" than 4a.
_Locant = Union[int, Tuple[int, str]]


def _assert_homogeneous_locants(locants: List[_Locant]) -> None:
    """Raise ValueError if ``locants`` contains a mix of int and tuple types.

    Per Phase 146 D-19: after coercion, a locant list must be uniformly
    int OR uniformly tuple. Mixed types indicate a caller bug (e.g.,
    ring_info populated only partially) and would cause Python's
    ``sorted()`` / ``min()`` to raise ``TypeError`` on mixed int/tuple
    comparison.

    Empty lists are vacuously homogeneous and return silently.

    Args:
        locants: List of int or (int, str) tuple locants.

    Raises:
        ValueError: if ``locants`` contains both int and tuple values.
                    The message includes the first offending int and the
                    first offending tuple to aid debugging.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.7
            (locant set comparison semantics)
    Source: Phase 146 D-19 (locant type safety lock-in)
    """
    has_int = any(isinstance(x, int) for x in locants)
    has_tuple = any(isinstance(x, tuple) for x in locants)
    if has_int and has_tuple:
        first_int = next(x for x in locants if isinstance(x, int))
        first_tuple = next(x for x in locants if isinstance(x, tuple))
        raise ValueError(
            f"Locant list contains mixed int and tuple types: "
            f"{first_int!r} and {first_tuple!r}. "
            f"Coerce all to tuples or all to ints before passing to "
            f"compare_locant_sets."
        )


def build_atom_to_locant(principal_chain: List[int]) -> Dict[int, int]:
    """
    Create a mapping from RDKit atom indices to 1-indexed IUPAC locants.

    The locant numbering follows the order of the principal chain: the first
    atom in the chain gets locant 1, the second gets locant 2, etc.

    Args:
        principal_chain: Ordered list of atom indices forming the principal chain.
                         The ordering must already reflect the correct numbering
                         direction (see orient_chain).

    Returns:
        Dict mapping atom_idx -> locant (1-indexed).
        Empty dict if principal_chain is empty.

    Examples:
        >>> build_atom_to_locant([5, 3, 1, 0])
        {5: 1, 3: 2, 1: 3, 0: 4}
        >>> build_atom_to_locant([])
        {}
        >>> build_atom_to_locant([0])
        {0: 1}
    """
    return {atom_idx: locant for locant, atom_idx in enumerate(principal_chain, 1)}


def compare_locant_sets(
    set_a: List[_Locant],
    set_b: List[_Locant],
) -> int:
    """
    Compare two locant sets using IUPAC first-point-of-difference rule.

    This is NOT the sum-of-locants method. Both sets are sorted ascending,
    then compared term-by-term. The set with the lower value at the first
    point of difference is preferred.

    If all compared elements are equal, the shorter set wins (fewer locants
    needed means simpler name). If completely identical, returns 0.

    Phase 147 extension (D-01, D-02): also accepts ``List[Tuple[int, str]]``
    with first-point-of-difference semantics for fusion atoms. When one
    list contains tuples and the other contains ints, all ints are coerced
    to ``(n, '')`` tuples internally — empty string sorts before any
    letter, preserving the IUPAC convention that locant ``4`` is "lower"
    than locant ``4a``. Truly heterogeneous lists are rejected by
    ``_assert_homogeneous_locants`` (raises ``ValueError``).

    Args:
        set_a: First locant set (unsorted or sorted). Elements are int
               or ``(int, str)`` tuples; mixed allowed only if all ints
               can be coerced via the ``(n, '')`` padding.
        set_b: Second locant set (same type contract as set_a).

    Returns:
        -1 if set_a is preferred (lower at first difference)
         0 if sets are equal
         1 if set_b is preferred (lower at first difference)

    Examples:
        >>> compare_locant_sets([2, 3, 5], [3, 4, 6])
        -1
        >>> compare_locant_sets([2, 4, 5], [2, 3, 5])
        1
        >>> compare_locant_sets([2, 3], [2, 3])
        0
        >>> compare_locant_sets([(4, ''), (5, '')], [(4, 'a'), (5, '')])
        -1
        >>> compare_locant_sets([(4, 'a'), (5, '')], [(4, 'b'), (5, '')])
        -1

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2, P-14.7
    Source: Phase 146 D-19 (locant type safety); Phase 147 D-01/D-02
    """
    # Phase 147: tuple-coercion entry path. If either list contains a
    # tuple locant, coerce all ints to (n, '') tuples in BOTH lists so
    # Python's sorted()/comparison operators stay type-safe. Pure-int
    # lists fall through to the back-compat fast path unchanged.
    any_tuple = (
        any(isinstance(x, tuple) for x in set_a)
        or any(isinstance(x, tuple) for x in set_b)
    )
    if any_tuple:
        a_coerced = [(x, '') if isinstance(x, int) else x for x in set_a]
        b_coerced = [(x, '') if isinstance(x, int) else x for x in set_b]
        _assert_homogeneous_locants(a_coerced)
        _assert_homogeneous_locants(b_coerced)
        a_sorted = sorted(a_coerced)
        b_sorted = sorted(b_coerced)
    else:
        a_sorted = sorted(set_a)
        b_sorted = sorted(set_b)

    for a, b in zip(a_sorted, b_sorted):
        if a < b:
            return -1  # set_a preferred
        if a > b:
            return 1   # set_b preferred

    # All compared elements equal -- shorter set wins
    if len(a_sorted) < len(b_sorted):
        return -1
    if len(a_sorted) > len(b_sorted):
        return 1

    return 0  # Identical


def orient_chain(
    chain: List[int],
    mol,
    principal_group_atoms: Set[int],
    double_bonds: List[Tuple[int, int]],
    triple_bonds: List[Tuple[int, int]],
    substituent_positions: Optional[Dict[int, List]] = None,
) -> List[int]:
    """
    Orient the principal chain to produce the lowest IUPAC locant set.

    Applies IUPAC 2013 criteria in strict order:
        a. Lowest locants for principal characteristic group
        b. Lowest locants for multiple bonds (double + triple, as a set)
        c. Lowest locants for double bonds specifically
        d. Lowest locants for substituents (detachable prefixes)
        e. Lowest locant for alphabetically first substituent (P-14.4(g))

    For pure hydrocarbons (no principal group), criterion (a) is skipped.

    Args:
        chain: Ordered list of atom indices forming the candidate chain.
               This function evaluates forward vs reversed ordering.
        mol: RDKit Mol object (used to detect bonds on the chain).
        principal_group_atoms: Set of atom indices belonging to the principal
                                characteristic group (may be empty).
        double_bonds: List of (atom_idx, atom_idx) tuples for C=C bonds.
        triple_bonds: List of (atom_idx, atom_idx) tuples for C#C bonds.
        substituent_positions: Optional dict mapping chain atom index to list
                                of substituent groups. If None, criterion (d)
                                is skipped.

    Returns:
        The chain in the preferred orientation (may be the same list or
        reversed).
    """
    if len(chain) <= 1:
        return chain

    forward = chain
    reverse = list(reversed(chain))

    # Build locant maps for both directions
    fwd_map = build_atom_to_locant(forward)
    rev_map = build_atom_to_locant(reverse)

    chain_set = set(chain)

    # --- Criterion (a): Lowest locants for principal characteristic group ---
    if principal_group_atoms:
        pg_on_chain = principal_group_atoms & chain_set
        if not pg_on_chain:
            # P-14.4(a) / P-31.1.4: a principal group whose defining atoms are NOT
            # chain atoms (-SO3H / -PO3H2: the S/P and its O's hang OFF a chain
            # carbon) is located by the CHAIN CARBON that bears it. Without this,
            # criterion (a) was skipped for such groups and a mere substituent stole
            # C1 ('1-hydroxyethanesulfonic acid' instead of the PIN
            # '2-hydroxyethanesulfonic acid'). Anchor on the attachment carbon(s) so
            # the principal group still drives the lowest-locant rule. (Groups whose
            # match includes the chain carbon -- acids/ketones/alcohols/amines -- keep
            # a non-empty pg_on_chain above and are unaffected: byte-identical.)
            pg_on_chain = {
                a for a in chain
                if any(nb.GetIdx() in principal_group_atoms
                       for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            }
        if pg_on_chain:
            fwd_locants = sorted(fwd_map[a] for a in pg_on_chain)
            rev_locants = sorted(rev_map[a] for a in pg_on_chain)
            result = compare_locant_sets(fwd_locants, rev_locants)
            if result == -1:
                return forward
            if result == 1:
                return reverse
            # result == 0 -> tie, continue to next criterion

    # --- Criterion (b): Lowest locants for multiple bonds (combined set) ---
    # Must compute bond locant atoms separately for forward and reverse chains,
    # since the "lower position" atom of each bond depends on chain direction
    all_bonds = double_bonds + triple_bonds
    fwd_bond_atoms = _get_bond_locant_atoms(forward, all_bonds)
    rev_bond_atoms = _get_bond_locant_atoms(reverse, all_bonds)
    if fwd_bond_atoms or rev_bond_atoms:
        fwd_locants = sorted(fwd_map[a] for a in fwd_bond_atoms)
        rev_locants = sorted(rev_map[a] for a in rev_bond_atoms)
        result = compare_locant_sets(fwd_locants, rev_locants)
        if result == -1:
            return forward
        if result == 1:
            return reverse

    # --- Criterion (c): Lowest locants for double bonds specifically ---
    fwd_double_atoms = _get_bond_locant_atoms(forward, double_bonds)
    rev_double_atoms = _get_bond_locant_atoms(reverse, double_bonds)
    if fwd_double_atoms or rev_double_atoms:
        fwd_locants = sorted(fwd_map[a] for a in fwd_double_atoms)
        rev_locants = sorted(rev_map[a] for a in rev_double_atoms)
        result = compare_locant_sets(fwd_locants, rev_locants)
        if result == -1:
            return forward
        if result == 1:
            return reverse

    # --- Criterion (d): Lowest locants for substituents ---
    if substituent_positions:
        # Substituent positions are already keyed by atom index
        sub_atoms = set(substituent_positions.keys()) & chain_set
        if sub_atoms:
            fwd_locants = sorted(fwd_map[a] for a in sub_atoms)
            rev_locants = sorted(rev_map[a] for a in sub_atoms)
            result = compare_locant_sets(fwd_locants, rev_locants)
            if result == -1:
                return forward
            if result == 1:
                return reverse

    # --- Criterion (e): Lowest locant for alphabetically first substituent ---
    # P-14.4(g): when substituent locant sets are identical in both directions,
    # prefer the orientation giving the lowest locant to the first-cited prefix
    # (alphabetically first substituent).
    if substituent_positions:
        sub_atoms = set(substituent_positions.keys()) & chain_set
        if len(sub_atoms) >= 2:  # Need 2+ substituent positions for this to matter
            fwd_alpha = _alphabetical_tiebreaker(forward, fwd_map, substituent_positions, mol)
            rev_alpha = _alphabetical_tiebreaker(reverse, rev_map, substituent_positions, mol)
            if fwd_alpha < rev_alpha:
                return forward
            if rev_alpha < fwd_alpha:
                return reverse

    # All criteria tied -- return forward (arbitrary but deterministic)
    return forward


def get_functional_group_locants(
    chain: List[int],
    fg_atom_tuples: List[Tuple[int, ...]],
    atom_to_locant: Dict[int, int],
    mol=None,
) -> List[int]:
    """
    Resolve functional group SMARTS match atoms to chain locants.

    Given FG match tuples from SMARTS pattern matching, find the
    locant-defining atom for each match on the principal chain.

    The locant-defining atom is the carbon on the chain that bears the
    functional group -- e.g. the carbonyl carbon for ketones, the carbon
    bearing -OH for alcohols, the carboxyl carbon for acids.

    When an RDKit mol is provided, the function selects the on-chain
    carbon with the most bonds to heteroatoms (O, N, S, etc.) within the
    match.  This correctly handles SMARTS patterns where a neighbor carbon
    appears before the key carbon in the match tuple (e.g. the ketone
    pattern ``[#6][CX3](=O)[#6]``).

    Without a mol, falls back to picking the first on-chain atom.

    Args:
        chain: Ordered principal chain atom indices.
        fg_atom_tuples: List of atom index tuples from SMARTS matching.
                        Each tuple contains indices of atoms in one FG instance.
        atom_to_locant: Mapping from atom index to locant (from build_atom_to_locant).
        mol: Optional RDKit Mol object.  When provided, used to score
             candidate carbon atoms by their heteroatom connectivity.

    Returns:
        Sorted list of locants where the functional group attaches to the chain.
        Empty list if no FG atoms are on the chain.
    """
    chain_set = set(chain)
    locants = []

    for match_tuple in fg_atom_tuples:
        best_idx = _pick_locant_atom(match_tuple, chain_set, mol)
        if best_idx is not None and best_idx in atom_to_locant:
            locants.append(atom_to_locant[best_idx])

    return sorted(locants)


def _pick_locant_atom(
    match_tuple: Tuple[int, ...],
    chain_set: Set[int],
    mol=None,
) -> Optional[int]:
    """
    Pick the locant-defining atom from a SMARTS match tuple.

    Strategy:
        1. Collect all carbon atoms in the match that lie on the chain.
        2. If *mol* is available, score each candidate by the number of
           bonds it has to heteroatoms (non-C, non-H) within the same
           match tuple.  The atom with the highest score is the
           functional-group carbon.
        3. On tie (or when mol is unavailable), return the first
           candidate in match-tuple order.

    Returns:
        Atom index of the locant-defining atom, or ``None`` if no atom
        in the match is on the chain.
    """
    if mol is None:
        # Fallback: first atom on chain
        for atom_idx in match_tuple:
            if atom_idx in chain_set:
                return atom_idx
        return None

    match_set = set(match_tuple)
    candidates: List[int] = []
    for atom_idx in match_tuple:
        if atom_idx in chain_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 6:  # carbon
                candidates.append(atom_idx)

    if not candidates:
        # No carbon on chain -- fall back to any atom on chain
        for atom_idx in match_tuple:
            if atom_idx in chain_set:
                return atom_idx
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Score: count bonds to non-carbon atoms within the match
    def _hetero_score(idx: int) -> int:
        atom = mol.GetAtomWithIdx(idx)
        score = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in match_set and nbr.GetAtomicNum() != 6:
                score += 1
        return score

    # Pick highest score; on tie, preserve match-tuple order
    best = max(candidates, key=_hetero_score)
    return best


def get_bond_locants(
    chain: List[int],
    bonds: List[Tuple[int, int]],
    atom_to_locant: Dict[int, int],
) -> List[int]:
    """
    Get locants for bonds (double or triple) within the chain.

    For each bond (atom_a, atom_b), both atoms must be in the chain.
    The locant is the LOWER of the two atom locants (per IUPAC convention,
    a bond between positions i and i+1 is cited by locant i).

    Args:
        chain: Ordered list of atom indices in the principal chain.
        bonds: List of (atom_idx_a, atom_idx_b) tuples for bonds.
        atom_to_locant: Mapping from atom index to locant (1-indexed).

    Returns:
        Sorted list of locants for the bonds.

    Examples:
        >>> chain = [0, 1, 2, 3]
        >>> bonds = [(1, 2)]  # bond between atoms 1 and 2
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        >>> get_bond_locants(chain, bonds, atom_to_locant)
        [2]
    """
    bond_atoms = _get_bond_locant_atoms(chain, bonds)
    locants = [atom_to_locant[atom_idx] for atom_idx in bond_atoms
               if atom_idx in atom_to_locant]
    return sorted(locants)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _alphabetical_tiebreaker(
    chain: List[int],
    atom_to_locant: Dict[int, int],
    substituent_positions: Dict[int, List],
    mol,
) -> tuple:
    """Build a comparison key for the alphabetical tiebreaker (criterion e).

    For each substituent attachment point on the chain, determine the
    substituent name (based on carbon count for simple alkyl groups) and
    its locant. Return a tuple of (alpha_sort_key, locant) pairs sorted
    by alpha_sort_key first, so that lexicographic comparison of tuples
    selects the orientation giving the lowest locant to the
    alphabetically first substituent.

    Args:
        chain: Ordered chain atom indices.
        atom_to_locant: Mapping from atom index to 1-indexed locant.
        substituent_positions: Dict mapping chain atom index to list of
            substituent atom groups.
        mol: RDKit Mol object.

    Returns:
        Tuple for lexicographic comparison. Lower = preferred.
    """
    from ..assembly.naming_utils import alpha_sort_key, get_alkyl_name

    chain_set = set(chain)
    entries = []

    for atom_idx, sub_groups in substituent_positions.items():
        if atom_idx not in chain_set:
            continue
        locant = atom_to_locant.get(atom_idx)
        if locant is None:
            continue

        # Count carbon atoms in the first substituent group at this position
        # (sufficient for the alphabetical comparison of simple alkyls)
        for sub_atoms in sub_groups:
            name = None
            # WS-A task 9 (P-14.5.2): a RING substituent must be compared by
            # its REAL cited prefix name, not a name fabricated from its
            # carbon count (phenyl is NOT 'hexyl', thiophen-2-yl is NOT
            # 'butyl' — the fabricated keys inverted the orientation of
            # 1-phenyl-4-(thiophen-2-yl)butane-1,4-dione). Derive the
            # attachment as the fragment atom bonded to the chain atom
            # (sub_atoms comes from a set — position 0 is arbitrary).
            if sub_atoms and any(
                    mol.GetAtomWithIdx(a).IsInRing() for a in sub_atoms):
                try:
                    attach_idx = next(
                        (a for a in sub_atoms
                         if any(nbr.GetIdx() == atom_idx
                                for nbr in mol.GetAtomWithIdx(a).GetNeighbors())),
                        None,
                    )
                    if attach_idx is not None:
                        from ..assembly.substituent_enumerator import (
                            name_substituent,
                        )
                        name = name_substituent(mol, list(sub_atoms), attach_idx)
                except Exception:
                    name = None
            if not name:
                carbon_count = sum(
                    1 for a in sub_atoms
                    if mol.GetAtomWithIdx(a).GetSymbol() == "C"
                )
                if carbon_count > 0:
                    name = get_alkyl_name(carbon_count)
                else:
                    # Non-carbon substituent (e.g., halogen): use atom symbol
                    if sub_atoms:
                        name = mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol().lower()
                    else:
                        name = "zzz"
            entries.append((alpha_sort_key(name), locant))

    # Sort by alpha key first, then by locant
    entries.sort()
    # Return as flat tuple for lexicographic comparison:
    # the first entry's (alpha_key, locant) dominates
    return tuple(item for pair in entries for item in pair)


def _get_bond_locant_atoms(
    chain: List[int],
    bonds: List[Tuple[int, int]],
) -> Set[int]:
    """
    Find atoms on the chain that are part of specified bonds.

    For IUPAC locant purposes, a multiple bond between atoms at positions i
    and i+1 in the chain is cited by the lower locant (position i). This
    function returns the set of lower-numbered atoms (by chain position) for
    each bond that lies entirely on the chain.

    Args:
        chain: Ordered list of atom indices in the principal chain.
        bonds: List of (atom_idx_a, atom_idx_b) tuples for bonds.

    Returns:
        Set of atom indices (the lower-position atom of each on-chain bond).
    """
    chain_set = set(chain)
    # Build position lookup: atom_idx -> chain position (0-indexed)
    pos = {atom_idx: i for i, atom_idx in enumerate(chain)}
    result = set()

    for a, b in bonds:
        if a in chain_set and b in chain_set:
            # Return the atom with the lower chain position
            if pos[a] < pos[b]:
                result.add(a)
            else:
                result.add(b)

    return result
