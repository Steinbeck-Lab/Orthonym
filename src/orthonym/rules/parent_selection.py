"""
Parent selection logic for ring vs chain compounds (IUPAC P-44.1).

This module implements the IUPAC 2013 rules for selecting between a ring
and a chain as the parent structure in organic compound naming.

Key rule (P-44.1): "The principal characteristic group cited as suffix
must be attached to the principal chain or ring system."

Key rule (P-52.2.8): "When the ring and the chain contain the same number
of skeletal atoms in the ring or chain, the ring system is always preferred
as the principal chain."

This means:
- If -COOH is on the chain, chain MUST be parent
- If -COOH is directly on the ring, ring MUST be parent
- For hydrocarbons (no FG), rings have seniority over chains (P-44.1.2.2)
- When FG count is tied, P-44.1 cascade (chain length > multiple bonds) applied before P-52.2.8 ring default
- When multiple ring systems exist, the most senior one is the parent (P-44.2)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from .locants import compare_locant_sets
from .ring_selection import ring_system_score

logger = logging.getLogger(__name__)


@dataclass
class ParentSelectionResult:
    """Result of parent structure selection.

    Attributes:
        parent_type: 'ring' or 'chain'
        parent_atoms: Atom indices of the parent structure
        substituent_rings: Rings that become substituents (when chain is parent)
        reasoning: Explanation for debugging
    """
    parent_type: str  # 'ring' or 'chain'
    parent_atoms: List[int]
    substituent_rings: List[Tuple[int, ...]]  # Rings that become substituents
    reasoning: str  # For debugging


def is_principal_group_on_ring(
    mol,
    ring_atoms: Set[int],
    principal_group_atoms: List[tuple]
) -> bool:
    """
    Check if the principal functional group is directly attached to the ring.

    For carboxylic acid: the carbonyl carbon must be bonded to a ring atom.
    For alcohol: the carbon bearing -OH must be a ring atom.

    The key distinction is:
    - "on ring" = FG attachment point is bonded to a ring carbon
    - NOT just anywhere connected through chain to ring

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the ring
        principal_group_atoms: List of tuples, each tuple is a SMARTS match

    Returns:
        True if principal group is directly attached to ring
    """
    if not principal_group_atoms:
        return False

    ring_atoms_set = set(ring_atoms)

    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue

        # The attachment point is typically the first atom in SMARTS match
        # For [CX3](=O)[OX2H1] (carboxylic acid): atom 0 is carbonyl C
        # For [CX4][OX2H1] (alcohol): atom 0 is C bearing OH
        attachment_atom = pg_atoms[0]

        # Check if attachment atom is directly bonded to a ring atom
        atom = mol.GetAtomWithIdx(attachment_atom)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in ring_atoms_set:
                return True

    return False


def is_principal_group_on_chain(
    mol,
    chain_atoms: List[int],
    principal_group_atoms: List[tuple]
) -> bool:
    """
    Check if the principal functional group is on the chain.

    For carboxylic acid: the carbonyl carbon must be in chain_atoms.
    For alcohol: the carbon bearing -OH must be in chain_atoms.

    Args:
        mol: RDKit Mol object
        chain_atoms: List of atom indices in the principal chain
        principal_group_atoms: List of tuples, each tuple is a SMARTS match

    Returns:
        True if principal group is on the chain
    """
    if not principal_group_atoms or not chain_atoms:
        return False

    chain_set = set(chain_atoms)

    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue

        # The attachment point is typically the first atom in SMARTS match
        # For [CX3](=O)[OX2H1] (carboxylic acid): atom 0 is carbonyl C
        # For [CX4][OX2H1] (alcohol): atom 0 is C bearing OH
        attachment_atom = pg_atoms[0]

        # Check if attachment atom is in the chain
        if attachment_atom in chain_set:
            return True

        # Also check if attachment atom is bonded to chain
        # (for cases where FG is terminal, e.g., -CH2-COOH)
        atom = mol.GetAtomWithIdx(attachment_atom)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in chain_set:
                # The attachment point is bonded to chain
                return True

    return False


def _count_multiple_bonds(mol, atom_set: Set[int]) -> int:
    """Count double + triple bonds where both atoms are in atom_set.

    Args:
        mol: RDKit Mol object
        atom_set: Set of atom indices to consider

    Returns:
        Number of double or triple bonds within atom_set
    """
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE or bt == Chem.BondType.TRIPLE:
                count += 1
    return count


def _count_double_bonds(mol, atom_set: Set[int]) -> int:
    """Count double bonds (not triple) where both atoms are in atom_set."""
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                count += 1
    return count


def _get_largest_individual_ring_size(mol, ring_system_atoms: Set[int]) -> int:
    """Get size of largest individual SSSR ring in a ring system.

    Per P-52.2.8, ring-vs-chain comparison uses individual ring size,
    not total fused system atom count.

    Args:
        mol: RDKit Mol object
        ring_system_atoms: Set of atom indices in the ring system

    Returns:
        Size of the largest individual ring, or total atoms if no rings found
    """
    ri = mol.GetRingInfo()
    max_size = 0
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(ring_system_atoms):
            max_size = max(max_size, len(ring))
    return max_size if max_size > 0 else len(ring_system_atoms)


def _count_substituents_on_atoms(mol, atom_set: Set[int]) -> int:
    """Count non-H substituents attached to atoms in atom_set but not in it."""
    count = 0
    for idx in atom_set:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in atom_set and nbr.GetSymbol() != 'H':
                count += 1
    return count


# ============================================================================
# Composable P-44.1 Comparators (ring vs chain)
# ============================================================================
# Each returns: 1 (chain wins), -1 (ring wins), 0 (tie)


def _compare_chain_length(chain_len: int, ring_size: int) -> int:
    """P-44.1(c): Maximum chain length / ring skeletal atoms."""
    if chain_len > ring_size:
        return 1
    elif ring_size > chain_len:
        return -1
    return 0


def _compare_multiple_bonds(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(d): Maximum number of multiple bonds (double + triple)."""
    chain_mult = _count_multiple_bonds(mol, chain_set)
    ring_mult = _count_multiple_bonds(mol, ring_set)
    if chain_mult > ring_mult:
        return 1
    elif ring_mult > chain_mult:
        return -1
    return 0


def _compare_double_bonds(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(e): Maximum number of double bonds."""
    chain_db = _count_double_bonds(mol, chain_set)
    ring_db = _count_double_bonds(mol, ring_set)
    if chain_db > ring_db:
        return 1
    elif ring_db > chain_db:
        return -1
    return 0


def _compare_pg_locants(
    mol, chain: List[int], ring_set: Set[int],
    principal_group_atoms: List[tuple]
) -> int:
    """P-44.1(f): Lowest locants for principal groups.

    Compares the locant sets for principal characteristic group attachment
    points on the chain versus on the ring, using first-point-of-difference
    comparison (IUPAC P-14.7).

    Locants are 1-indexed IUPAC-style positions (per P-14.7):
    - Chain: position along the chain list (atom at index 0 -> locant 1).
    - Ring: sorted atom indices mapped to 1-indexed positions. This is a
      positional proxy consistent with ring numbering convention used
      throughout the P-44.1 cascade. True IUPAC ring numbering (P-14.7)
      follows ring perception rules, but for ring-vs-chain comparison
      the consistent positional convention produces equivalent results.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system
        principal_group_atoms: List of tuples of atom indices from SMARTS matches

    Returns:
        1 if chain has lower PG locants (chain wins)
        -1 if ring has lower PG locants (ring wins)
        0 if tied or neither has PG locants
    """
    chain_set = set(chain)

    # Build 1-indexed position maps (IUPAC P-14.7: locants start at 1)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_sorted = sorted(ring_set)
    ring_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(ring_sorted)}

    # Get PG locants on chain (1-indexed IUPAC locants)
    chain_pg_locants = []
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        attachment = pg_atoms[0]
        if attachment in chain_set:
            chain_pg_locants.append(chain_pos[attachment])
        else:
            atom = mol.GetAtomWithIdx(attachment)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in chain_set:
                    chain_pg_locants.append(chain_pos[nbr.GetIdx()])
                    break

    # Get PG locants on ring (1-indexed positional proxy)
    ring_pg_locants = []
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        attachment = pg_atoms[0]
        if attachment in ring_set:
            ring_pg_locants.append(ring_pos[attachment])
        else:
            atom = mol.GetAtomWithIdx(attachment)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_set:
                    ring_pg_locants.append(ring_pos[nbr.GetIdx()])
                    break

    if not chain_pg_locants and not ring_pg_locants:
        return 0

    # Use compare_locant_sets for first-point-of-difference comparison
    # compare_locant_sets returns: -1 (a preferred), 0 (tie), 1 (b preferred)
    # chain=a, ring=b: -1 -> return 1 (chain wins), 1 -> return -1 (ring wins)
    cmp = compare_locant_sets(chain_pg_locants, ring_pg_locants)
    return -cmp


def _compare_multiple_bond_locants(
    mol, chain: List[int], ring_set: Set[int]
) -> int:
    """P-44.1(g): Lowest locants for multiple bonds.

    Compares the locant sets for double and triple bonds on the chain
    versus on the ring, using first-point-of-difference (IUPAC P-14.7).

    For chain: bond between chain positions i and i+1 gets locant i+1
    (1-indexed, using the lower position per IUPAC convention).
    For ring: atoms are sorted by index and mapped to 1-indexed positions.
    Bond locant is the lower position of the two bonded atoms.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system

    Returns:
        1 if chain has lower bond locants (chain wins)
        -1 if ring has lower bond locants (ring wins)
        0 if tied or neither has multiple bonds
    """
    chain_set = set(chain)

    # Build position maps (1-indexed)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_sorted = sorted(ring_set)
    ring_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(ring_sorted)}

    # Collect bond locants for chain and ring
    chain_bond_locants = []
    ring_bond_locants = []

    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt != Chem.BondType.DOUBLE and bt != Chem.BondType.TRIPLE:
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        # Check if bond is on chain (both atoms in chain)
        if a in chain_set and b in chain_set:
            locant = min(chain_pos[a], chain_pos[b])
            chain_bond_locants.append(locant)

        # Check if bond is on ring (both atoms in ring)
        if a in ring_set and b in ring_set:
            locant = min(ring_pos[a], ring_pos[b])
            ring_bond_locants.append(locant)

    if not chain_bond_locants and not ring_bond_locants:
        return 0

    # Use compare_locant_sets for first-point-of-difference comparison
    # compare_locant_sets returns: -1 (a preferred), 0 (tie), 1 (b preferred)
    # We pass chain as a, ring as b:
    #   -1 (chain preferred) -> return 1 (chain wins)
    #    1 (ring preferred)  -> return -1 (ring wins)
    #    0                   -> return 0
    cmp = compare_locant_sets(chain_bond_locants, ring_bond_locants)
    return -cmp


def _compare_substituent_locants(
    mol, chain: List[int], ring_set: Set[int]
) -> int:
    """P-44.1(i): Lowest locants for substituents (detachable prefixes).

    Compares the locant sets for substituent attachment points on the chain
    versus on the ring, using first-point-of-difference (IUPAC P-14.7).

    A substituent is any non-hydrogen atom bonded to a chain/ring atom
    but NOT itself part of the chain/ring.

    For chain: substituent at chain position i gets locant i+1 (1-indexed).
    For ring: atoms sorted by index, mapped to 1-indexed positions.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system

    Returns:
        1 if chain has lower substituent locants (chain wins)
        -1 if ring has lower substituent locants (ring wins)
        0 if tied or neither has substituents
    """
    chain_set = set(chain)

    # Build position maps (1-indexed)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_sorted = sorted(ring_set)
    ring_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(ring_sorted)}

    # Collect substituent locants for chain
    chain_sub_locants = []
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in chain_set and nbr.GetSymbol() != 'H':
                # This chain atom has an external substituent
                chain_sub_locants.append(chain_pos[idx])
                break  # Only count the position once, even with multiple substituents

    # Collect substituent locants for ring
    ring_sub_locants = []
    for idx in ring_sorted:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_set and nbr.GetSymbol() != 'H':
                ring_sub_locants.append(ring_pos[idx])
                break

    if not chain_sub_locants and not ring_sub_locants:
        return 0

    # compare_locant_sets: -1 (a preferred), 0 (tie), 1 (b preferred)
    # chain=a, ring=b: -1 -> return 1 (chain wins), 1 -> return -1 (ring wins)
    cmp = compare_locant_sets(chain_sub_locants, ring_sub_locants)
    return -cmp


def _compare_substituent_count(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(h): Maximum number of substituents."""
    chain_subs = _count_substituents_on_atoms(mol, chain_set)
    ring_subs = _count_substituents_on_atoms(mol, ring_set)
    if chain_subs > ring_subs:
        return 1
    elif ring_subs > chain_subs:
        return -1
    return 0


def select_parent(
    mol,
    ring_systems: List[set],
    principal_chain: List[int],
    principal_group: Optional[str],
    principal_group_atoms: List[tuple]
) -> ParentSelectionResult:
    """
    Select parent structure per IUPAC P-44.1.

    Determines whether the ring or chain should be the parent structure
    based on where the principal characteristic group is located.

    Decision logic:
    1. If PG on chain and NOT on ring -> chain is parent
    2. If PG on ring and NOT on chain -> ring is parent
    3. If PG on both -> compare by count, tiebreak by ring seniority
    4. If no PG (hydrocarbon) -> ring has seniority (P-44.1.2.2)

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices for each ring
        principal_chain: Pre-computed principal chain (from namer.py)
        principal_group: Name of principal functional group (or None)
        principal_group_atoms: List of tuples of atom indices

    Returns:
        ParentSelectionResult with decision and metadata
    """
    # Convert ring systems to tuple format for result
    all_ring_atoms: Set[int] = set()
    for ring in ring_systems:
        all_ring_atoms.update(ring)

    substituent_ring_tuples = [tuple(sorted(ring)) for ring in ring_systems]

    # Handle empty chain case
    if not principal_chain:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning="No chain provided - ring is parent"
        )

    # Handle single carbon "chain" - this is just a substituent
    if len(principal_chain) == 1:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning="Single carbon chain - ring is parent"
        )

    # No principal group (hydrocarbon) - ring has seniority (P-44.1.2.2)
    if not principal_group or not principal_group_atoms:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning="No principal group - ring has seniority (P-44.1.2.2)"
        )

    # Check where principal group is located
    pg_on_ring = is_principal_group_on_ring(mol, all_ring_atoms, principal_group_atoms)
    pg_on_chain = is_principal_group_on_chain(mol, principal_chain, principal_group_atoms)

    # Decision logic per IUPAC P-44.1

    # P-44.1(a): PG on chain only -> chain MUST be parent
    if pg_on_chain and not pg_on_ring:
        return ParentSelectionResult(
            parent_type='chain',
            parent_atoms=principal_chain,
            substituent_rings=substituent_ring_tuples,
            reasoning=f"P-44.1(a): principal group ({principal_group}) is on chain only"
        )

    # P-44.1(a): PG on ring only -> ring is parent
    if pg_on_ring and not pg_on_chain:
        best_ring, other_rings = _select_best_ring_system(
            mol, ring_systems, principal_group_atoms
        )
        other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(best_ring)),
            substituent_rings=other_ring_tuples,
            reasoning=f"P-44.1(a): principal group ({principal_group}) is on ring only"
        )

    if pg_on_ring and pg_on_chain:
        # Ester-specific: acyl (C=O) side determines the parent acid
        if principal_group == "ester":
            ester_result = _compare_ester_parent(
                mol, ring_systems, principal_chain, principal_group_atoms,
                all_ring_atoms, substituent_ring_tuples
            )
            if ester_result is not None:
                return ester_result
            # Tie: fall through to general cascade

        # P-44.1(b): Compare PG count on ring vs chain
        pg_count_on_ring = _count_pg_on_ring(mol, all_ring_atoms, principal_group_atoms)
        pg_count_on_chain = _count_pg_on_chain(mol, principal_chain, principal_group_atoms)

        if pg_count_on_chain > pg_count_on_ring:
            # P-44.1(b): Chain has more PGs -> chain wins
            return ParentSelectionResult(
                parent_type='chain',
                parent_atoms=principal_chain,
                substituent_rings=substituent_ring_tuples,
                reasoning=f"P-44.1(b): more {principal_group} on chain ({pg_count_on_chain}) than ring ({pg_count_on_ring})"
            )
        elif pg_count_on_ring > pg_count_on_chain:
            # P-44.1(b): Ring has more PGs -> ring wins
            best_ring, other_rings = _select_best_ring_system(
                mol, ring_systems, principal_group_atoms
            )
            other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(best_ring)),
                substituent_rings=other_ring_tuples,
                reasoning=f"P-44.1(b): more {principal_group} on ring ({pg_count_on_ring}) than chain ({pg_count_on_chain})"
            )

        # PG counts tied -> apply P-44.1 cascade criteria (c) through (i)
        best_ring, other_rings = _select_best_ring_system(
            mol, ring_systems, principal_group_atoms
        )
        # P-52.2.8: Use largest individual ring size, not total fused system
        ring_size = _get_largest_individual_ring_size(mol, best_ring)
        chain_len = len(principal_chain)
        chain_set = set(principal_chain)

        # P-44.1 cascade: criteria (c) through (i)
        cascade_result = _compare_chain_length(chain_len, ring_size)        # P-44.1(c)
        if cascade_result == 0:
            cascade_result = _compare_multiple_bonds(mol, chain_set, best_ring)  # P-44.1(d)
        if cascade_result == 0:
            cascade_result = _compare_double_bonds(mol, chain_set, best_ring)    # P-44.1(e)
        if cascade_result == 0:
            cascade_result = _compare_pg_locants(
                mol, principal_chain, best_ring, principal_group_atoms)           # P-44.1(f)
        if cascade_result == 0:
            cascade_result = _compare_multiple_bond_locants(
                mol, principal_chain, best_ring)                                 # P-44.1(g)
        if cascade_result == 0:
            cascade_result = _compare_substituent_count(mol, chain_set, best_ring)  # P-44.1(h)
        if cascade_result == 0:
            cascade_result = _compare_substituent_locants(
                mol, principal_chain, best_ring)                                 # P-44.1(i)

        if cascade_result > 0:
            # Chain wins
            other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
            all_sub_rings = [tuple(sorted(best_ring))] + other_ring_tuples
            return ParentSelectionResult(
                parent_type='chain',
                parent_atoms=principal_chain,
                substituent_rings=all_sub_rings,
                reasoning=f"P-44.1 cascade: chain wins (len={chain_len}, ring={ring_size})"
            )
        elif cascade_result < 0:
            # Ring wins
            other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(best_ring)),
                substituent_rings=other_ring_tuples,
                reasoning=f"P-44.1 cascade: ring wins (ring={ring_size}, chain={chain_len})"
            )

        # All criteria tied -> P-52.2.8: ring wins as final tiebreaker
        other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(best_ring)),
            substituent_rings=other_ring_tuples,
            reasoning=f"P-52.2.8: all P-44.1 criteria tied - ring wins"
        )

    # Neither on ring nor chain (shouldn't happen in well-formed molecules)
    return ParentSelectionResult(
        parent_type='ring',
        parent_atoms=list(sorted(all_ring_atoms)),
        substituent_rings=[],
        reasoning="Principal group location unclear - defaulting to ring"
    )


def _count_pg_on_ring(
    mol,
    ring_atoms: Set[int],
    principal_group_atoms: List[tuple]
) -> int:
    """Count how many distinct principal group instances are on the ring.

    PSEL-01: Deduplicates by match tuple to avoid counting the same
    physical FG instance multiple times from overlapping SMARTS matches.
    """
    seen_matches = set()
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        key = tuple(sorted(pg_atoms))
        if key in seen_matches:
            continue
        seen_matches.add(key)
        attachment = pg_atoms[0]
        atom = mol.GetAtomWithIdx(attachment)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in ring_atoms:
                count += 1
                break
    return count


def _count_pg_on_chain(
    mol,
    chain_atoms: List[int],
    principal_group_atoms: List[tuple]
) -> int:
    """Count how many distinct principal group instances are on the chain.

    PSEL-01: Deduplicates by match tuple to avoid counting the same
    physical FG instance multiple times from overlapping SMARTS matches.
    """
    chain_set = set(chain_atoms)
    seen_matches = set()
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        key = tuple(sorted(pg_atoms))
        if key in seen_matches:
            continue
        seen_matches.add(key)
        attachment = pg_atoms[0]
        if attachment in chain_set:
            count += 1
            continue
        atom = mol.GetAtomWithIdx(attachment)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in chain_set:
                count += 1
                break
    return count


def _select_best_ring_system(
    mol,
    ring_systems: List[set],
    principal_group_atoms: Optional[List[tuple]] = None
) -> Tuple[set, List[set]]:
    """Select the most senior ring system from multiple candidates.

    When a molecule has disconnected ring systems (e.g., pyridine + cyclohexane),
    only the most senior ring system should be the parent. Others become
    substituents.

    Uses ring_system_score() from ring_selection.py (P-44.2) to compare.
    Tiebreaker: prefer the ring system with more principal group attachment points.

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices for each ring system
        principal_group_atoms: Optional list of FG atom tuples for tiebreaking

    Returns:
        Tuple of (best_ring_system_atoms, other_ring_systems)
    """
    if len(ring_systems) <= 1:
        return (ring_systems[0] if ring_systems else set(), [])

    # Score each ring system via P-44.2 criteria
    scored = []
    for i, system in enumerate(ring_systems):
        score = ring_system_score(mol, system)

        # Tiebreaker: count principal group attachment points on this ring system
        # Weight 2 for PG atom directly IN the ring (e.g., ring ketone C=O where C is in ring)
        # Weight 1 for PG atom bonded TO a ring atom (e.g., -COOH where C(=O) is bonded to ring C)
        # This ensures direct-on-ring PG wins over adjacent-to-ring PG (PRNT-05)
        pg_attachments = 0
        if principal_group_atoms:
            for pg_atoms in principal_group_atoms:
                if not pg_atoms:
                    continue
                attachment = pg_atoms[0]
                # Direct: PG atom itself is a ring atom (e.g., ring ketone C=O where C is in ring)
                if attachment in system:
                    pg_attachments += 2
                    continue
                # Adjacent: PG atom bonded to a ring atom (e.g., -COOH where C(=O) is bonded to ring C)
                atom = mol.GetAtomWithIdx(attachment)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in system:
                        pg_attachments += 1
                        break

        # Append negative pg_attachments so min() prefers more attachments
        scored.append((score, -pg_attachments, i))

    scored.sort()
    best_idx = scored[0][2]

    best_system = ring_systems[best_idx]
    other_systems = [ring_systems[i] for i in range(len(ring_systems)) if i != best_idx]

    return (best_system, other_systems)


def _ring_system_has_nitrogen(mol, ring_atoms: Set[int]) -> bool:
    """Check if any atom in the ring system is nitrogen."""
    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7:
            return True
    return False


def _compare_ester_parent(
    mol, ring_systems, principal_chain, principal_group_atoms,
    all_ring_atoms, substituent_ring_tuples
) -> Optional[ParentSelectionResult]:
    """Ester-specific parent determination: acyl (C=O) side is parent acid.

    For esters, the acyl carbon location determines the parent:
    - Acyl C bonded to ring -> ring provides the acid name
    - Acyl C bonded to chain -> chain provides the acid name

    Returns None if tied (fall through to general cascade).
    """
    acyl_ring_count = 0
    acyl_chain_count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms or len(pg_atoms) < 1:
            continue
        acyl_c = pg_atoms[0]  # carbonyl C
        atom = mol.GetAtomWithIdx(acyl_c)
        acyl_on_ring = acyl_c in all_ring_atoms
        if not acyl_on_ring:
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in all_ring_atoms:
                    acyl_on_ring = True
                    break
        if acyl_on_ring:
            acyl_ring_count += 1
        else:
            acyl_chain_count += 1

    if acyl_ring_count > acyl_chain_count:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning=f"Ester acyl on ring ({acyl_ring_count}) > chain ({acyl_chain_count})"
        )
    elif acyl_chain_count > acyl_ring_count:
        return ParentSelectionResult(
            parent_type='chain',
            parent_atoms=principal_chain,
            substituent_rings=substituent_ring_tuples,
            reasoning=f"Ester acyl on chain ({acyl_chain_count}) > ring ({acyl_ring_count})"
        )
    return None  # Tied: fall through to general cascade
