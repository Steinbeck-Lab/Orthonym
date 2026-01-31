"""
Parent selection logic for ring vs chain compounds (IUPAC P-44.1).

This module implements the IUPAC 2013 rules for selecting between a ring
and a chain as the parent structure in organic compound naming.

Key rule (P-44.1): "The principal characteristic group cited as suffix
must be attached to the principal chain or ring system."

This means:
- If -COOH is on the chain, chain MUST be parent
- If -COOH is directly on the ring, ring MUST be parent
- For hydrocarbons (no FG), rings have seniority over chains (P-44.1.2.2)
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


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
    if pg_on_chain and not pg_on_ring:
        # Principal group is on chain only - chain MUST be parent
        return ParentSelectionResult(
            parent_type='chain',
            parent_atoms=principal_chain,
            substituent_rings=substituent_ring_tuples,
            reasoning=f"Principal group ({principal_group}) is on chain only"
        )

    if pg_on_ring and not pg_on_chain:
        # Principal group is on ring only - ring is parent
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning=f"Principal group ({principal_group}) is on ring only"
        )

    if pg_on_ring and pg_on_chain:
        # Principal group is on both - need to compare
        # Count PG occurrences on each
        pg_count_on_ring = _count_pg_on_ring(mol, all_ring_atoms, principal_group_atoms)
        pg_count_on_chain = _count_pg_on_chain(mol, principal_chain, principal_group_atoms)

        if pg_count_on_chain > pg_count_on_ring:
            return ParentSelectionResult(
                parent_type='chain',
                parent_atoms=principal_chain,
                substituent_rings=substituent_ring_tuples,
                reasoning=f"More {principal_group} groups on chain ({pg_count_on_chain}) than ring ({pg_count_on_ring})"
            )
        else:
            # Ring wins on tie or if more on ring
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(all_ring_atoms)),
                substituent_rings=[],
                reasoning=f"Ring wins: {pg_count_on_ring} {principal_group} on ring vs {pg_count_on_chain} on chain"
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
    """Count how many principal group instances are on the ring."""
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
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
    """Count how many principal group instances are on the chain."""
    chain_set = set(chain_atoms)
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
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
