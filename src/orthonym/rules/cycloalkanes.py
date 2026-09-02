"""
Cycloalkane and cycloalkene naming rules according to IUPAC 2013 (Blue Book).

Handles:
- Ring numbering and orientation for lowest locants
- Cycloalkene double bond positioning (C1-C2)
- Ring substituent detection
- Ring vs chain parent selection

IUPAC 2013 Rules:
- Cycloalkenes: Double bond is at C1-C2 position
- Mono-cycloalkenes: locant is omitted (cyclohexene, not cyclohex-1-ene)
- Cycloalkadienes: locants required (cyclohexa-1,3-diene)
- Substituent locants: use first-point-of-difference rule
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from rdkit import Chem

from ..assembly.naming_utils import alpha_sort_key, get_alkyl_name as _canonical_get_alkyl_name
from .locants import compare_locant_sets as _compare_locant_sets  # IM-02



def get_ring_double_bonds(mol, ring_atoms: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    Get all double bonds within a ring.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices defining the ring

    Returns:
        List of (atom_idx1, atom_idx2) tuples for each double bond in the ring
    """
    ring_set = set(ring_atoms)
    double_bonds = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in ring_set and end_idx in ring_set:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds.append((begin_idx, end_idx))

    return double_bonds


def is_mancude_monocyclic_hydrocarbon(mol, ring_atoms: Tuple[int, ...]) -> bool:
    """P-22.1.2(b) / P-25.3.2.1.1 detection: a mancude monocyclic hydrocarbon
    that RDKit marks aromatic but which is NOT benzene is named as the
    cyclo-polyene (cyclodeca-1,3,5,7,9-pentaene), never as an [n]annulene
    component prefix.

    Fail-closed scope: exactly one ring, all-carbon, every ring atom RDKit-
    aromatic, size >= 7 (benzene size 6 keeps its retained name), and every
    ring atom a bare CH (unsubstituted mancude hydrocarbon). Returns True only
    when the ring should be re-routed through the cycloalkene (polyene) path.
    """
    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return False
    ring = list(ring_atoms)
    if len(ring) < 7:
        return False
    ring_set = set(ring)
    for i in ring:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'C':
            return False
        if not a.GetIsAromatic():
            return False
        # every ring atom must be a bare CH (no exocyclic heavy neighbours)
        heavy_ext = [n for n in a.GetNeighbors()
                     if n.GetIdx() not in ring_set and n.GetAtomicNum() > 1]
        if heavy_ext:
            return False
    return True


def get_ring_substituents(mol, ring_atoms: Tuple[int, ...]) -> Dict[int, List[List[int]]]:
    """
    Find substituents attached to ring atoms.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices defining the ring

    Returns:
        Dict mapping ring atom index to list of substituent atom lists.
        Each substituent is a list of atom indices (found via BFS).
    """
    from ..perception.rings import get_containing_ring_system

    # Use the complete ring system as BFS boundary (IUPAC P-25.3)
    # Prevents walking into fused/bridged partner rings
    ring_set = set(get_containing_ring_system(mol, ring_atoms))
    substituents: Dict[int, List[List[int]]] = defaultdict(list)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the ring
            if nbr_idx in ring_set:
                continue

            # BFS to find full substituent
            sub_atoms = _bfs_substituent(mol, nbr_idx, ring_set)
            substituents[ring_idx].append(sub_atoms)

    return dict(substituents)


def _bfs_substituent(mol, start_idx: int, exclude_atoms: Set[int]) -> List[int]:
    """
    BFS to find all atoms in a substituent.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index
        exclude_atoms: Atoms to exclude (e.g., ring atoms)

    Returns:
        List of atom indices in the substituent
    """
    visited = {start_idx}
    queue = deque([start_idx])
    atoms = []

    while queue:
        current_idx = queue.popleft()
        atoms.append(current_idx)

        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return atoms


def orient_cycloalkane(
    mol,
    ring_atoms: Tuple[int, ...],
    substituent_positions: Dict[int, List[List[int]]],
    principal_group_atoms: Optional[Set[int]] = None
) -> List[int]:
    """
    Orient a cycloalkane ring to give lowest locants to substituents.

    For cycloalkanes without double bonds:
    - Single substituent: that carbon is position 1
    - Multiple substituents: apply first-point-of-difference rule

    When the principal characteristic group sits on ring atoms (expressed as
    a suffix: -ol, -one, -amine,...), P-31.1.4 numbering applies instead:
    the suffix anchor takes the lowest locant before any detachable prefix.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        substituent_positions: Dict mapping ring atom index to substituent lists
        principal_group_atoms: Optional set of ring atom indices bearing the
            principal characteristic group (e.g., the ring C of C-OH / C=O /
            C-NH2). Exocyclic-carbon suffixes (-carbaldehyde, -carboxylic
            acid) must NOT be included — they do not seize ring numbering.

    Returns:
        List of ring atom indices reordered so position 1 is first
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)

    # Get substituted positions (indices in ring_list)
    substituted_atom_indices = set(substituent_positions.keys())

    # --- P-31.1.4: principal group on ring -> suffix-locant priority ---
    if principal_group_atoms:
        pg_set = {a for a in principal_group_atoms if a in ring_list}
        if pg_set:
            return _orient_cycloalkane_with_pg(
                mol, ring_list, substituent_positions, pg_set
            )

    if not substituted_atom_indices:
        # Unsubstituted - any orientation is fine
        return ring_list

    if len(substituted_atom_indices) == 1:
        # Single substituent - that position becomes 1
        sub_atom = next(iter(substituted_atom_indices))
        start_pos = ring_list.index(sub_atom)
        return _rotate_list(ring_list, start_pos)

    # Multiple substituents - try all starting positions and both directions
    candidates = []

    for start_pos in range(n):
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            oriented = _build_oriented_ring(ring_list, start_pos, direction)

            # Calculate locants for this orientation
            locants = []
            for i, atom_idx in enumerate(oriented):
                if atom_idx in substituted_atom_indices:
                    locants.append(i + 1)  # Locants are 1-indexed

            locants.sort()

            # Get substituent name at position 1 for alphabetic tie-breaking
            pos1_atom = oriented[0]
            pos1_sub_name = None
            if pos1_atom in substituent_positions and substituent_positions[pos1_atom]:
                # Get substituent name (by carbon count for alkyl, recursive for branched)
                sub_atoms = substituent_positions[pos1_atom][0]
                carbon_count = sum(
                    1 for idx in sub_atoms
                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                )
                # Try recursive naming for branched subs
                all_c_h = all(
                    mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                    for i in sub_atoms
                )
                if all_c_h and len(sub_atoms) > 1:
                    from ..assembly.substituent_naming import name_substituent_fragment
                    pos1_sub_name = name_substituent_fragment(
                        mol, sub_atoms, sub_atoms[0], list(ring_atoms)
                    )
                if pos1_sub_name is None:
                    pos1_sub_name = _get_alkyl_name(carbon_count)

            candidates.append((oriented, locants, pos1_sub_name))

    # Find best locant set
    best_locants = None
    for _, locants, _ in candidates:
        if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
            best_locants = locants

    # Filter to candidates with best locant set
    best_candidates = [
        (oriented, pos1_sub) for oriented, locants, pos1_sub in candidates
        if locants == best_locants
    ]

    if len(best_candidates) == 1:
        return best_candidates[0][0]

    # Alphabetic tie-breaking
    def sort_key(item):
        oriented, pos1_sub = item
        if pos1_sub is None:
            return 'zzzzz'
        return alpha_sort_key(pos1_sub)

    best_candidates.sort(key=sort_key)
    return best_candidates[0][0]


def _orient_cycloalkane_with_pg(
    mol,
    ring_list: List[int],
    substituent_positions: Dict[int, List[List[int]]],
    pg_set: Set[int]
) -> List[int]:
    """
    P-31.1.4 numbering for a saturated ring whose principal characteristic
    group sits on ring atoms (suffix expression: -ol, -one, -amine,...).

    Tier order, first-decision-wins (P-31.1.4.2.4 / P-31.1.4.3.4):
      (c) lowest locants to the suffix anchor atoms,
      (f) lowest locants to the detachable-prefix-only set — the suffix
          expression itself (the heteroatom-rooted substituent list at a PG
          position) is NOT a prefix and is excluded from this set,
      (g) lowest locant to the prefix cited first in alphanumerical order.
    """
    n = len(ring_list)
    candidates = []

    for start_pos in range(n):
        for direction in (1, -1):
            oriented = _build_oriented_ring(ring_list, start_pos, direction)

            # (c) suffix-anchor locants
            pg_locants = sorted(oriented.index(a) + 1 for a in pg_set)

            # (f) prefix-only entries: at a PG position, the heteroatom-rooted
            # list IS the suffix (C-OH / C=O / C-NH2) and does not count;
            # carbon-rooted lists there are genuine prefixes and still count.
            prefix_entries = []  # (locant, sub_atoms)
            for i, atom_idx in enumerate(oriented):
                for sub_atoms in substituent_positions.get(atom_idx, ()):
                    if (atom_idx in pg_set and sub_atoms
                            and mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol() != 'C'):
                        continue
                    prefix_entries.append((i + 1, sub_atoms))
            prefix_locants = sorted(loc for loc, _ in prefix_entries)

            candidates.append((oriented, pg_locants, prefix_locants, prefix_entries))

    best = candidates[0]
    for cand in candidates[1:]:
        cmp = _compare_locant_sets(cand[1], best[1])
        if cmp == 0:
            cmp = _compare_locant_sets(cand[2], best[2])
        if cmp < 0:
            best = cand

    tied = [
        cand for cand in candidates
        if _compare_locant_sets(cand[1], best[1]) == 0
        and _compare_locant_sets(cand[2], best[2]) == 0
    ]
    if len(tied) == 1:
        return tied[0][0]

    # (g) cite each candidate's prefixes in alphabetical order and compare
    # the (name, locant) vectors — the first-cited name's locant decides.
    def alpha_citation_key(cand):
        entries = []
        for locant, sub_atoms in cand[3]:
            name = _prefix_name_for_sort(mol, sub_atoms, ring_list)
            entries.append((alpha_sort_key(name) if name else 'zzzzz', locant))
        entries.sort()
        return entries

    tied.sort(key=alpha_citation_key)
    return tied[0][0]


def ring_double_bond_locant(pos1: int, pos2: int, n: int) -> int:
    """Locant of a ring double bond between two oriented positions (0-based).

    Ring bonds connect consecutive positions; the locant is the lower
    position + 1 — EXCEPT the ring-closure bond (positions 0 and n-1), whose
    locant is n (a 6-ring bond between C6 and C1 is the 6-ene bond, never
    1-ene). The naive ``min(pos)+1`` called the closure bond "1", which let
    the orientation comparator pick a direction whose emitted ``1-ene``
    described a DIFFERENT structure (canary rt75_0019 RT True->False).
    """
    lo, hi = (pos1, pos2) if pos1 < pos2 else (pos2, pos1)
    if lo == 0 and hi == n - 1:
        return n
    return lo + 1


def _prefix_name_for_sort(mol, sub_atoms: List[int], ring_list: List[int]) -> Optional[str]:
    """Best-effort prefix name for alphabetic tie-breaking (tier (g)).

    Mirrors the legacy pos1 naming: recursive fragment naming for branched
    all-C/H substituents, straight alkyl name by carbon count otherwise.
    Returns None when no name can be derived (sorts last).
    """
    if not sub_atoms:
        return None
    carbon_count = sum(
        1 for idx in sub_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )
    all_c_h = all(
        mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
        for i in sub_atoms
    )
    name = None
    if all_c_h and len(sub_atoms) > 1:
        from ..assembly.substituent_naming import name_substituent_fragment
        name = name_substituent_fragment(mol, sub_atoms, sub_atoms[0], ring_list)
    if name is None and all_c_h:
        name = _get_alkyl_name(carbon_count)
    return name


def orient_cycloalkene(
    mol,
    ring_atoms: Tuple[int, ...],
    double_bond_atoms: List[Tuple[int, int]],
    substituent_positions: Optional[Dict[int, List[List[int]]]] = None,
    principal_group_atoms: Optional[Set[int]] = None
) -> List[int]:
    """
    Orient a cycloalkene ring for IUPAC naming.

    IUPAC 2013 rules (P-31.1.3.4):
    - When a principal characteristic group is on the ring, it receives
      the lowest possible locant (ideally 1).
    - Double bond locant is secondary to principal group locant.
    - When no principal group is on the ring, double bond is at C1-C2.
    - Direction is chosen to give lowest locants to other substituents.
    - For mono-cycloalkenes, the locant is omitted in the name.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        double_bond_atoms: List of (atom1, atom2) tuples for double bonds
        substituent_positions: Optional dict mapping ring atom index to substituent lists
        principal_group_atoms: Optional set of ring atom indices bearing the principal
            characteristic group (e.g., C bearing =O for ketone)

    Returns:
        List of ring atom indices reordered for IUPAC naming
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)

    if not double_bond_atoms:
        # No double bonds - just return as-is (shouldn't happen for cycloalkene)
        return ring_list

    # Get substituent atom indices
    sub_atom_indices = set(substituent_positions.keys()) if substituent_positions else set()

    # --- Path A: Principal group on ring -> PG gets lowest locant ---
    if principal_group_atoms:
        candidates = []

        for start_idx in range(n):
            for direction in [1, -1]:
                oriented = _build_oriented_ring(ring_list, start_idx, direction)

                # Calculate principal group locants
                pg_locants = sorted(
                    oriented.index(atom) + 1
                    for atom in principal_group_atoms
                    if atom in oriented
                )

                # Calculate double bond locants (wrap-aware: the closure
                # bond positions (0, n-1) is locant n, not 1)
                db_locants = []
                for a1, a2 in double_bond_atoms:
                    if a1 in oriented and a2 in oriented:
                        pos1 = oriented.index(a1)
                        pos2 = oriented.index(a2)
                        db_locants.append(ring_double_bond_locant(pos1, pos2, n))
                db_locants.sort()

                # Calculate substituent locants
                sub_locants = sorted(
                    i + 1 for i, atom in enumerate(oriented)
                    if atom in sub_atom_indices
                )

                candidates.append((oriented, pg_locants, db_locants, sub_locants))

        # Sort: lowest PG locants, then lowest DB locants, then lowest sub locants
        def sort_key(item):
            _, pg, db, sub = item
            return (pg, db, sub)

        candidates.sort(key=sort_key)
        return candidates[0][0]

    # --- Path B: No principal group on ring -> double bond at C1-C2 ---
    # Get positions of double bond atoms in the ring
    db_atoms_set = set()
    for a1, a2 in double_bond_atoms:
        db_atoms_set.add(a1)
        db_atoms_set.add(a2)

    # Try all orientations where a double bond starts at position 1
    candidates = []

    for db_a1, db_a2 in double_bond_atoms:
        # For each double bond, try starting with either atom as position 1
        for start_atom in [db_a1, db_a2]:
            # The other atom of the double bond should be position 2
            other_atom = db_a2 if start_atom == db_a1 else db_a1

            start_pos = ring_list.index(start_atom)

            # Determine direction: other_atom should be at position 2 (index 1)
            # Check both directions
            for direction in [1, -1]:
                oriented = _build_oriented_ring(ring_list, start_pos, direction)

                # Check if other_atom is at position 2
                if oriented[1] != other_atom:
                    continue

                # Calculate locants for substituents
                sub_locants = []
                for i, atom_idx in enumerate(oriented):
                    if atom_idx in sub_atom_indices:
                        sub_locants.append(i + 1)

                sub_locants.sort()

                # Calculate locants for all double bonds (wrap-aware)
                db_locants = []
                for a1, a2 in double_bond_atoms:
                    pos1 = oriented.index(a1)
                    pos2 = oriented.index(a2)
                    db_locants.append(ring_double_bond_locant(pos1, pos2, n))

                db_locants.sort()

                candidates.append((oriented, db_locants, sub_locants))

    if not candidates:
        # Fallback - shouldn't happen
        return ring_list

    # First criterion: lowest double bond locants
    best_db_locants = None
    for _, db_locants, _ in candidates:
        if best_db_locants is None or _compare_locant_sets(db_locants, best_db_locants) < 0:
            best_db_locants = db_locants

    # Filter by best double bond locants
    filtered = [
        (oriented, sub_locants) for oriented, db_locants, sub_locants in candidates
        if db_locants == best_db_locants
    ]

    if len(filtered) == 1:
        return filtered[0][0]

    # Second criterion: lowest substituent locants
    best_sub_locants = None
    for _, sub_locants in filtered:
        if best_sub_locants is None or _compare_locant_sets(sub_locants, best_sub_locants) < 0:
            best_sub_locants = sub_locants

    # Return first orientation with best locants
    for oriented, sub_locants in filtered:
        if sub_locants == best_sub_locants:
            return oriented

    return filtered[0][0]


def select_ring_or_chain_parent(
    mol,
    rings: List[Tuple[int, ...]],
    chain: List[int],
    principal_group: Optional[str] = None
) -> str:
    """
    Determine whether ring or chain should be the parent structure.

    IUPAC 2013 Method 1 (PIN):
    1. Principal characteristic group location: If principal FG is on chain
       but not ring, chain wins. If on ring but not chain, ring wins.
    2. Same class (both carbon): Ring has seniority over chain
    3. Exception: Chain is parent if it's significantly longer than ring

    For pure hydrocarbons (no FG):
    - Ring is parent when ring carbon count >= chain carbon count
    - Chain is parent when significantly longer than ring

    Args:
        mol: RDKit Mol object
        rings: List of ring atom tuples
        chain: List of chain atom indices
        principal_group: Name of principal functional group, if any

    Returns:
        'ring' or 'chain'
    """
    if not rings:
        return 'chain'

    if not chain:
        return 'ring'

    # Get the largest ring
    largest_ring = max(rings, key=len)
    ring_size = len(largest_ring)
    chain_length = len(chain)

    # For pure hydrocarbons or when FG is not a deciding factor:
    # Ring is parent if ring_size >= chain_length
    # This follows IUPAC convention that rings are preferred as parent

    # Exception: If chain is much longer than ring (>2x), chain wins
    # This handles cases like decylcyclopropane vs cyclopropyldecan-X-yl
    if chain_length > ring_size * 2:
        return 'chain'

    # Default: Ring is parent (IUPAC preference)
    return 'ring'


def _rotate_list(lst: List, start: int) -> List:
    """Rotate list so element at index start becomes first."""
    return lst[start:] + lst[:start]


def _build_oriented_ring(
    ring_list: List[int],
    start_pos: int,
    direction: int
) -> List[int]:
    """
    Build an oriented version of the ring.

    Args:
        ring_list: Original list of ring atom indices
        start_pos: Index to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos first, going in specified direction
    """
    n = len(ring_list)
    oriented = []

    for i in range(n):
        idx = (start_pos + i * direction) % n
        oriented.append(ring_list[idx])

    return oriented


def _get_alkyl_name(carbon_count: int) -> Optional[str]:
    """Get alkyl substituent name from carbon count.

    Delegates to the canonical get_alkyl_name() in naming_utils.
    Returns None for invalid or unknown counts.
    """
    try:
        return _canonical_get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        return None


def get_substituent_name(mol, sub_atoms: List[int], ring_atoms: set = None) -> Optional[str]:
    """
    Get the name of a substituent from its atom list.

    Args:
        mol: RDKit Mol object
        sub_atoms: List of atom indices in the substituent
        ring_atoms: Optional set of ring atom indices (for recursive naming)

    Returns:
        Substituent name (e.g., 'methyl', 'ethyl', 'isopropyl') or None
    """
    # Count carbons
    carbon_count = sum(
        1 for idx in sub_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Check for heteroatoms
    has_heteroatom = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in sub_atoms
    )

    if has_heteroatom:
        # Complex substituent - handle in later phases
        return None

    # Try recursive naming (handles retained names like isopropyl + branched subs)
    if len(sub_atoms) > 1 and ring_atoms is not None:
        from ..assembly.substituent_naming import name_substituent_fragment
        attach_idx = sub_atoms[0]
        rec_name = name_substituent_fragment(
            mol, sub_atoms, attach_idx, list(ring_atoms)
        )
        if rec_name:
            return rec_name

    return _get_alkyl_name(carbon_count)
