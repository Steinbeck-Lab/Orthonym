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

    Used by P-44.1 cascade as second tiebreaker (after chain length)
    to compare multiple-bond counts between chain and ring candidates.

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
        # Override: When chain is very short (< 3 atoms) and the molecule
        # has exactly ONE connected ring system that is large (>= 13 atoms),
        # selecting the tiny chain as parent drops the entire ring system.
        # Per P-52.2.8 spirit and P-44.2.1(e) "greater skeletal atoms",
        # prefer ring. The FG becomes a prefix instead of suffix.
        #
        # Guard: only for single-ring-system molecules (VB, fused systems).
        # Multi-ring-system molecules are not affected.
        chain_len = len(principal_chain)
        if (chain_len < 3
                and len(ring_systems) == 1
                and len(ring_systems[0]) >= 13):
            best_ring = ring_systems[0]
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(best_ring)),
                substituent_rings=[],
                reasoning=f"P-52.2.8 override: tiny chain ({chain_len}) vs large ring ({len(best_ring)}) - ring preferred"
            )

        # Principal group is on chain only - chain MUST be parent
        return ParentSelectionResult(
            parent_type='chain',
            parent_atoms=principal_chain,
            substituent_rings=substituent_ring_tuples,
            reasoning=f"Principal group ({principal_group}) is on chain only"
        )

    if pg_on_ring and not pg_on_chain:
        # Principal group is on ring only - ring is parent
        # Select best ring system if multiple exist (P-44.2)
        best_ring, other_rings = _select_best_ring_system(
            mol, ring_systems, principal_group_atoms
        )
        other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(best_ring)),
            substituent_rings=other_ring_tuples,
            reasoning=f"Principal group ({principal_group}) is on ring only"
        )

    if pg_on_ring and pg_on_chain:
        # Special handling for esters when PG is on both ring and chain.
        # For esters, the acyl (C=O) side determines the parent:
        #   - If acyl C is bonded to ring -> ring is acid parent
        #     (e.g., methyl benzoate: ring provides "benzoate")
        #   - If acyl C is bonded to chain -> chain is acid parent
        #     (e.g., phenyl butanoate: chain provides "butanoate")
        # When multiple esters exist, count acyl-on-ring vs acyl-on-chain.
        if principal_group == "ester":
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
            # Decide based on which side has more acyl groups
            if acyl_ring_count > acyl_chain_count:
                return ParentSelectionResult(
                    parent_type='ring',
                    parent_atoms=list(sorted(all_ring_atoms)),
                    substituent_rings=[],
                    reasoning=f"Ester acyl on ring ({acyl_ring_count}) > chain ({acyl_chain_count}) - ring is parent"
                )
            elif acyl_chain_count > acyl_ring_count:
                return ParentSelectionResult(
                    parent_type='chain',
                    parent_atoms=principal_chain,
                    substituent_rings=substituent_ring_tuples,
                    reasoning=f"Ester acyl on chain ({acyl_chain_count}) > ring ({acyl_ring_count}) - chain is parent"
                )
            # Tie: fall through to general count-based comparison below

        # General case: Principal group is on both - compare by count
        pg_count_on_ring = _count_pg_on_ring(mol, all_ring_atoms, principal_group_atoms)
        pg_count_on_chain = _count_pg_on_chain(mol, principal_chain, principal_group_atoms)

        if pg_count_on_chain > pg_count_on_ring:
            # Enhancement 4 (P-52.2.8): When chain is very short (< 3 atoms)
            # but the ring is large (5+ atoms), prefer ring even when chain
            # has more FGs -- the short chain is essentially a substituent.
            # Only override if FG is NOT exclusively on chain (already handled
            # above by the pg_on_chain and not pg_on_ring branch).
            if len(principal_chain) < 3 and len(all_ring_atoms) >= 5:
                best_ring, other_rings = _select_best_ring_system(
                    mol, ring_systems, principal_group_atoms
                )
                other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
                return ParentSelectionResult(
                    parent_type='ring',
                    parent_atoms=list(sorted(best_ring)),
                    substituent_rings=other_ring_tuples,
                    reasoning=f"P-52.2.8 short chain ({len(principal_chain)} atoms) vs large ring ({len(all_ring_atoms)} atoms) - ring preferred"
                )

            # Enhancement 3 (P-44.1 step 2): Senior atom consideration.
            # If the ring contains nitrogen and the chain does not, and the
            # FG count difference is only 1, the ring's senior atom provides
            # additional signal favoring ring as parent. This is a tiebreaker
            # supplement when counts are close.
            if (pg_count_on_chain - pg_count_on_ring == 1
                    and _ring_system_has_nitrogen(mol, all_ring_atoms)):
                # Check if chain has nitrogen
                chain_has_n = any(
                    mol.GetAtomWithIdx(idx).GetAtomicNum() == 7
                    for idx in principal_chain
                )
                if not chain_has_n:
                    best_ring, other_rings = _select_best_ring_system(
                        mol, ring_systems, principal_group_atoms
                    )
                    other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
                    return ParentSelectionResult(
                        parent_type='ring',
                        parent_atoms=list(sorted(best_ring)),
                        substituent_rings=other_ring_tuples,
                        reasoning=f"P-44.1 senior atom: N-containing ring preferred over chain ({pg_count_on_chain} vs {pg_count_on_ring} {principal_group})"
                    )

            # INST-03/INST-04: Audit chain-length and multiple-bond data
            # when chain wins by PG count (behavior unchanged)
            if logger.isEnabledFor(logging.DEBUG):
                chain_set = set(principal_chain)
                chain_mult = _count_multiple_bonds(mol, chain_set)
                ring_mult = _count_multiple_bonds(mol, all_ring_atoms)
                logger.debug(
                    "P44_AUDIT: chain_wins_pg_count chain_pg=%d ring_pg=%d "
                    "chain_len=%d ring_size=%d "
                    "chain_mult=%d ring_mult=%d",
                    pg_count_on_chain, pg_count_on_ring,
                    len(principal_chain), len(all_ring_atoms),
                    chain_mult, ring_mult,
                )

            return ParentSelectionResult(
                parent_type='chain',
                parent_atoms=principal_chain,
                substituent_rings=substituent_ring_tuples,
                reasoning=f"More {principal_group} groups on chain ({pg_count_on_chain}) than ring ({pg_count_on_ring})"
            )
        elif pg_count_on_ring > pg_count_on_chain:
            # Ring has strictly more PGs -> ring wins by count
            best_ring, other_rings = _select_best_ring_system(
                mol, ring_systems, principal_group_atoms
            )
            other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(best_ring)),
                substituent_rings=other_ring_tuples,
                reasoning=f"Ring wins by PG count: {pg_count_on_ring} > {pg_count_on_chain} {principal_group}"
            )
        else:
            # PG counts tied -> P-44.1 cascade: length > multiple bonds > P-52.2.8
            chain_len = len(principal_chain)

            # Compare chain length vs BEST ring system size (not all_ring_atoms union)
            best_ring, other_rings = _select_best_ring_system(
                mol, ring_systems, principal_group_atoms
            )
            ring_size = len(best_ring)

            if chain_len > ring_size:
                # PRNT-01: Chain is longer -> chain wins
                other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
                # Include the best ring in substituent_rings too
                all_sub_rings = [tuple(sorted(best_ring))] + other_ring_tuples
                return ParentSelectionResult(
                    parent_type='chain',
                    parent_atoms=principal_chain,
                    substituent_rings=all_sub_rings,
                    reasoning=f"P-44.1 chain-length: chain ({chain_len}) > ring ({ring_size}), "
                              f"PG tied at {pg_count_on_ring} {principal_group}"
                )

            if ring_size > chain_len:
                # Ring is larger -> ring wins
                other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
                return ParentSelectionResult(
                    parent_type='ring',
                    parent_atoms=list(sorted(best_ring)),
                    substituent_rings=other_ring_tuples,
                    reasoning=f"P-44.1 ring-size: ring ({ring_size}) > chain ({chain_len}), "
                              f"PG tied at {pg_count_on_ring} {principal_group}"
                )

            # Length also tied -> PRNT-02: compare multiple bonds
            chain_set = set(principal_chain)
            chain_mult = _count_multiple_bonds(mol, chain_set)
            ring_mult = _count_multiple_bonds(mol, best_ring)

            if chain_mult > ring_mult:
                other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
                all_sub_rings = [tuple(sorted(best_ring))] + other_ring_tuples
                return ParentSelectionResult(
                    parent_type='chain',
                    parent_atoms=principal_chain,
                    substituent_rings=all_sub_rings,
                    reasoning=f"P-44.1 multiple-bonds: chain ({chain_mult}) > ring ({ring_mult}), "
                              f"PG and length tied"
                )

            # All criteria tied -> P-52.2.8 final tiebreaker: ring wins
            other_ring_tuples = [tuple(sorted(r)) for r in other_rings]
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(best_ring)),
                substituent_rings=other_ring_tuples,
                reasoning=f"P-52.2.8 ring wins: PG={pg_count_on_ring}, length={ring_size}, "
                          f"bonds: ring={ring_mult} chain={chain_mult} - all tied"
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
    """Check if any atom in the ring system is nitrogen.

    Used for P-44.1 step 2 senior atom tiebreaking: a ring containing
    nitrogen is preferred over one without (and over a chain without).

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the ring system

    Returns:
        True if ring system contains at least one nitrogen atom
    """
    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7:  # Nitrogen
            return True
    return False
