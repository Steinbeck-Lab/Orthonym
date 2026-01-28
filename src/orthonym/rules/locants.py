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

from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem


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


def compare_locant_sets(set_a: List[int], set_b: List[int]) -> int:
    """
    Compare two locant sets using IUPAC first-point-of-difference rule.

    This is NOT the sum-of-locants method. Both sets are sorted ascending,
    then compared term-by-term. The set with the lower value at the first
    point of difference is preferred.

    If all compared elements are equal, the shorter set wins (fewer locants
    needed means simpler name). If completely identical, returns 0.

    Args:
        set_a: First locant set (unsorted or sorted).
        set_b: Second locant set (unsorted or sorted).

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
    """
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
    multiple_bond_atoms = _get_bond_locant_atoms(chain, double_bonds + triple_bonds)
    if multiple_bond_atoms:
        fwd_locants = sorted(fwd_map[a] for a in multiple_bond_atoms)
        rev_locants = sorted(rev_map[a] for a in multiple_bond_atoms)
        result = compare_locant_sets(fwd_locants, rev_locants)
        if result == -1:
            return forward
        if result == 1:
            return reverse

    # --- Criterion (c): Lowest locants for double bonds specifically ---
    double_bond_atoms = _get_bond_locant_atoms(chain, double_bonds)
    if double_bond_atoms:
        fwd_locants = sorted(fwd_map[a] for a in double_bond_atoms)
        rev_locants = sorted(rev_map[a] for a in double_bond_atoms)
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

    # All criteria tied -- return forward (arbitrary but deterministic)
    return forward


def get_functional_group_locants(
    chain: List[int],
    fg_atom_tuples: List[Tuple[int, ...]],
    atom_to_locant: Dict[int, int],
) -> List[int]:
    """
    Resolve functional group SMARTS match atoms to chain locants.

    Given FG match tuples from SMARTS pattern matching, find which atom in
    each match lies on the principal chain and return its locant.

    Conventions:
        - For alcohols: the carbon bearing -OH is the locant (not the oxygen)
        - For ketones: the carbonyl carbon is the locant
        - For acids/aldehydes: always locant 1 (terminal carbon)
        - General rule: the first atom in the match tuple that is on the chain
          is the locant-defining atom

    Args:
        chain: Ordered principal chain atom indices.
        fg_atom_tuples: List of atom index tuples from SMARTS matching.
                        Each tuple contains indices of atoms in one FG instance.
        atom_to_locant: Mapping from atom index to locant (from build_atom_to_locant).

    Returns:
        Sorted list of locants where the functional group attaches to the chain.
        Empty list if no FG atoms are on the chain.
    """
    chain_set = set(chain)
    locants = []

    for match_tuple in fg_atom_tuples:
        # Find the first atom in the match that is on the principal chain.
        # SMARTS patterns typically list the key carbon first (e.g., the
        # carbonyl carbon for ketones, the carbon bearing OH for alcohols).
        for atom_idx in match_tuple:
            if atom_idx in chain_set and atom_idx in atom_to_locant:
                locants.append(atom_to_locant[atom_idx])
                break

    return sorted(locants)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
