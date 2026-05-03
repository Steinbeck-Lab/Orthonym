"""
Skeletal replacement ("a") nomenclature for chains with embedded heteroatoms.

Implements IUPAC 2013 P-15.4 replacement nomenclature where heteroatoms
embedded in a carbon chain backbone are named using replacement terms
(oxa, aza, thia, etc.) rather than substitutive prefixes (methoxy, amino, etc.).

Examples:
    COCCOC  -> 2,5-dioxahexane   (not 1,2-dimethoxyethane)
    CCNCCC  -> 3-azahexane        (not N-ethylpropan-1-amine)
    CCOCCOCC -> 3,6-dioxaoctane   (not butoxyethane)

Scope: Chain-only (acyclic). Rings <= 10 atoms are handled by
Hantzsch-Widman naming in the heterocycles module.

References:
    IUPAC 2013 Blue Book, P-15.4 (Replacement nomenclature)
    IUPAC 2013 Blue Book, P-15.4.3.1 (Order of citation of replacement terms)
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rdkit import Chem

from ..data.chain_names import get_chain_prefix
from ..assembly.naming_utils import SIMPLE_MULTIPLIERS


# ============================================================================
# Replacement term table (IUPAC P-15.4, Table 2.3)
# ============================================================================

REPLACEMENT_TERMS: Dict[str, str] = {
    'O': 'oxa',
    'S': 'thia',
    'Se': 'selena',
    'Te': 'tellura',
    'N': 'aza',
    'P': 'phospha',
    'As': 'arsa',
    'Sb': 'stiba',
    'Bi': 'bisma',
    'Si': 'sila',
    'Ge': 'germa',
    'Sn': 'stanna',
    'Pb': 'plumba',
    'B': 'bora',
}


def _qualifies_for_pin_skeletal_replacement(
    backbone: List[int], mol,
) -> Tuple[bool, str]:
    """Phase 154.A D-03: lock PIN trigger to strict IUPAC P-15.4.1.2.

    Three accept branches per IUPAC Blue Book P-15.4.1.2:
      (a) >= 4 same-kind embedded heteroatoms in the chain backbone
      (b) >= 3 mixed-kind embedded heteroatoms (>= 2 distinct elements)
      (c) substitutive expression would require >= 5 prefix units
          ("undue complexity"; conservative threshold for v18 -- equivalent
          to >= 5 embedded heteroatoms total regardless of kind diversity)

    Falls through to the legacy gate-5 semantics: a single embedded heteroatom
    in a backbone of length < 6 is REJECTED (substitutive form preferred:
    methoxymethane / ethoxyethane / etc.). A single embedded heteroatom in a
    backbone of length >= 6 is ACCEPTED (substitutive form becomes awkward).

    Two-heteroatom cases that do not trip branches (a/b/c) are ACCEPTED
    (preserves the legacy "diether and similar" behavior for compounds like
    3,6-dioxaoctan-1-ol).

    Args:
        backbone: ordered list of atom indices forming the principal chain
                  (output of _find_replacement_chain).
        mol: RDKit Mol object.

    Returns:
        (True, branch_label) where branch_label in
            {">=4-same-kind", ">=3-mixed-kind", "undue-complexity",
             "single-hetero-long-chain", "two-hetero-substitutive-equivalent"}
        (False, reason) where reason in
            {"no-heteroatoms", "single-hetero-short-chain"}.

    Source: IUPAC Blue Book 2013 P-15.4.1.2.
    Source: 154-CONTEXT.md D-03; 154-AUDIT-A.md gap inventory; 154-RESEARCH.md §3.2.
    """
    from collections import Counter
    embedded = []
    for i, atom_idx in enumerate(backbone):
        if i == 0 or i == len(backbone) - 1:
            continue  # skip terminal positions
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            embedded.append(symbol)

    if not embedded:
        return (False, "no-heteroatoms")

    counts = Counter(embedded)
    same_kind_max = max(counts.values())
    distinct_kinds = len(counts)
    total = sum(counts.values())

    # Branch (a): >= 4 same-kind heteroatoms
    if same_kind_max >= 4:
        return (True, ">=4-same-kind")

    # Branch (b): >= 3 mixed-kind heteroatoms
    if distinct_kinds >= 2 and total >= 3:
        return (True, ">=3-mixed-kind")

    # Branch (c): "undue complexity" -- conservative >= 5 total threshold
    if total >= 5:
        return (True, "undue-complexity")

    # Fall-through: single heteroatom -- preserve legacy gate-5 semantics.
    if total == 1 and len(backbone) < 6:
        return (False, "single-hetero-short-chain")

    # Single heteroatom in a long chain (>= 6): accept (legacy gate-5 behavior).
    if total == 1:
        return (True, "single-hetero-long-chain")

    # Two heteroatoms not covered by branches (a/b/c): accept (preserves
    # current behavior for diethers, etc. -- e.g., 3,6-dioxaoctan-1-ol).
    return (True, "two-hetero-substitutive-equivalent")


# Phase 154.A D-05: terminal -amine / -thiol support DEFERRED to v19.
# 154-AUDIT-A.md §4 corpus tally: amine eligible=2, thiol eligible=0
# (threshold 5); both below threshold => v19 follow-up
# IM-154-D05-amine / IM-154-D05-thiol.
# Source: 154-CONTEXT.md D-05; 154-AUDIT-A.md §4.


# IUPAC P-15.4.3.1: Order of citation for replacement terms
# When different heteroatom groups have the same lowest locant,
# alphabetical order of the replacement term breaks the tie.
# The seniority order from the IUPAC table (high to low):
# O > S > Se > Te > N > P > As > Si > Ge > Sn > Pb > B
# But citation order in the name is by ascending locant, then alphabetical.

# Functional groups that take priority over replacement naming.
# If ANY of these SMARTS match, do NOT use skeletal replacement.
_PRIORITY_FG_SMARTS = [
    '[CX3](=O)[OX2H1]',    # Carboxylic acid
    '[CX3](=O)[OX1-]',     # Carboxylate
    '[CX3H1](=O)',          # Aldehyde
    '[CX3](=O)[#6]',        # Ketone (C=O bonded to two carbons)
    '[CX3](=O)[OX2][#6]',  # Ester
    '[CX3](=O)[NX3]',      # Amide
    '[CX3](=O)[FX1,ClX1,BrX1,IX1]',  # Acid halide
    '[C]#[N]',             # Nitrile
    '[NX3][CX3](=[NX1])',  # Amidine
    '[SX2H]',              # Thiol
    '[NX2]=[CX2]=[OX1]',  # Isocyanate (N=C=O)
    '[NX2]=[CX2]=[SX1]',  # Isothiocyanate (N=C=S)
]

# Pre-compile the SMARTS patterns
_PRIORITY_FG_PATTERNS = []
for sma in _PRIORITY_FG_SMARTS:
    pat = Chem.MolFromSmarts(sma)
    if pat is not None:
        _PRIORITY_FG_PATTERNS.append(pat)


def try_skeletal_replacement_name(mol: Chem.Mol) -> Optional[str]:
    """Try to name a molecule using skeletal replacement nomenclature.

    Returns an IUPAC replacement name if the molecule is an acyclic chain
    with embedded heteroatoms suitable for replacement naming. Returns None
    if the molecule does not qualify (cyclic, has priority functional groups,
    too few heteroatoms, etc.).

    Supports terminal alcohol (-OH) suffix integration:
        OCCOCCOCC -> 3,6-dioxaoctan-1-ol

    Args:
        mol: RDKit molecule object (already parsed from SMILES).

    Returns:
        Replacement name string (e.g., '2,5-dioxahexane') or None.
    """
    if mol is None:
        return None

    # ----------------------------------------------------------------
    # Gate 1: Rings gate (IUPAC P-15.4 / P-22.1.3)
    # Chain replacement requires no rings, EXCEPT for large heterocyclic
    # rings (>= 7 members with heteroatoms) where skeletal replacement
    # naming may be simpler than substitutive naming.
    # Small rings (<= 6 members) are handled by Hantzsch-Widman naming.
    # ----------------------------------------------------------------
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        # Check if ALL rings are large (>= 7) and contain heteroatoms
        all_rings_large_hetero = True
        for ring in ring_info.AtomRings():
            if len(ring) < 7:
                all_rings_large_hetero = False
                break
            # Check ring has at least one heteroatom
            has_hetero = any(
                mol.GetAtomWithIdx(idx).GetSymbol() in REPLACEMENT_TERMS
                for idx in ring
            )
            if not has_hetero:
                all_rings_large_hetero = False
                break
        if not all_rings_large_hetero:
            return None
        # For large heterocyclic rings, try cyclic replacement naming
        # per IUPAC P-22.1.3. Keep the "no priority FGs" gate.
        for pat in _PRIORITY_FG_PATTERNS:
            if mol.HasSubstructMatch(pat):
                return None
        return _try_cyclic_replacement_name(mol, ring_info)

    # ----------------------------------------------------------------
    # Gate 2: No priority functional groups
    # ----------------------------------------------------------------
    for pat in _PRIORITY_FG_PATTERNS:
        if mol.HasSubstructMatch(pat):
            return None

    # ----------------------------------------------------------------
    # Gate 3: Check terminal functional groups
    # Terminal OH is allowed (suffix integration). Other terminal FGs
    # (NH2, SH) cause fallback to substitutive naming.
    # ----------------------------------------------------------------
    terminal_oh_info = _detect_terminal_oh(mol)
    has_other_terminal_fg = _has_terminal_functional_group(mol, exclude_oh=True)

    if has_other_terminal_fg:
        return None

    # If terminal OH detected, check that it's only OH (no NH2/SH combo)
    # and that there are still enough embedded heteroatoms for replacement naming
    has_terminal_oh = terminal_oh_info is not None

    # ----------------------------------------------------------------
    # Find the longest chain backbone including heteroatoms
    # ----------------------------------------------------------------
    backbone = _find_replacement_chain(mol)
    if backbone is None:
        return None

    # ----------------------------------------------------------------
    # For terminal OH: strip the OH oxygen from the backbone.
    # The terminal O-H is NOT a chain atom -- it's a functional suffix.
    # The chain consists only of C and embedded heteroatoms.
    # OCCOCCOCC backbone: O-C-C-O-C-C-O-C-C -> strip terminal O
    #   -> chain = C-C-O-C-C-O-C-C (8 atoms = octane)
    # ----------------------------------------------------------------
    if has_terminal_oh:
        oh_idx = terminal_oh_info['oh_idx']
        if backbone[0] == oh_idx:
            backbone = backbone[1:]
        elif backbone[-1] == oh_idx:
            backbone = backbone[:-1]
        # After stripping, verify backbone is still valid
        if backbone is None or len(backbone) < 3:
            return None

    # ----------------------------------------------------------------
    # Gate 4: All heavy atoms must be on the backbone (no substituents)
    # Branched molecules with substituents off the replacement chain
    # are not handled. Only unbranched replacement chains.
    # For terminal OH: the OH oxygen is excluded from backbone count,
    # so add 1 to expected atom count.
    # ----------------------------------------------------------------
    expected_atoms = len(backbone) + (1 if has_terminal_oh else 0)
    if expected_atoms != mol.GetNumAtoms():
        return None

    # ----------------------------------------------------------------
    # Gate 5: Check heteroatom count and chain length thresholds
    # ----------------------------------------------------------------
    embedded_heteroatoms = []
    for i, atom_idx in enumerate(backbone):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS and i > 0 and i < len(backbone) - 1:
            embedded_heteroatoms.append((i, symbol))

    if len(embedded_heteroatoms) == 0:
        return None

    # ----------------------------------------------------------------
    # Gate 5 (Phase 154.A D-03): strict IUPAC P-15.4.1.2 PIN trigger.
    # Replaces the legacy single-hetero chain-len < 6 reject with explicit
    # branch labels. Rationale string is for debug logging + 154-AUDIT-A.md
    # evidence trail.
    # ----------------------------------------------------------------
    qualifies, _rationale = _qualifies_for_pin_skeletal_replacement(backbone, mol)
    if not qualifies:
        return None
    # NOTE: _rationale (">=4-same-kind", ">=3-mixed-kind", "undue-complexity",
    # "single-hetero-long-chain", "two-hetero-substitutive-equivalent") is
    # currently unused but available for debug logging via:
    # logger.debug("skeletal_replacement: trigger_branch=%s smiles=%s",
    #              _rationale, Chem.MolToSmiles(mol))

    # ----------------------------------------------------------------
    # Number the chain: for -ol suffix, the OH end gets locant 1.
    # For plain replacement chains, give lowest locants to heteroatoms.
    # ----------------------------------------------------------------
    if has_terminal_oh:
        backbone = _orient_oh_end_first(
            backbone, mol, terminal_oh_info['carbon_idx']
        )
    else:
        backbone = _orient_for_lowest_locants(backbone, mol)

    # Rebuild heteroatom positions after reorientation
    heteroatom_positions = []
    for i, atom_idx in enumerate(backbone):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS and i > 0 and i < len(backbone) - 1:
            # Locants are 1-based
            heteroatom_positions.append((i + 1, symbol))

    if not heteroatom_positions:
        return None

    # ----------------------------------------------------------------
    # Determine suffix info for terminal OH
    # ----------------------------------------------------------------
    suffix = None
    if has_terminal_oh:
        # The OH-bearing carbon should be at position 0 (locant 1) after orient
        suffix = ('ol', 1)

    # ----------------------------------------------------------------
    # Build the replacement name
    # ----------------------------------------------------------------
    return _build_replacement_name(len(backbone), heteroatom_positions, suffix=suffix)


def _detect_terminal_oh(mol: Chem.Mol) -> Optional[Dict]:
    """Detect a terminal alcohol (-OH) suitable for replacement name suffix.

    A terminal OH is an oxygen with 1 H, bonded to exactly 1 heavy neighbor
    (a carbon), where that carbon is at the end of the chain (degree <= 2
    in the heavy-atom graph, meaning it has at most one other heavy neighbor).

    Args:
        mol: RDKit molecule object.

    Returns:
        Dict with 'oh_idx' (oxygen atom index) and 'carbon_idx' (bearing
        carbon index), or None if no suitable terminal OH found.
    """
    terminal_ohs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'O':
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
        if len(heavy_neighbors) != 1:
            continue
        carbon = heavy_neighbors[0]
        if carbon.GetSymbol() != 'C':
            continue
        # Check that the carbon is a chain terminal (degree 1 or 2 in heavy graph)
        carbon_heavy_nbrs = [n for n in carbon.GetNeighbors() if n.GetSymbol() != 'H']
        # The carbon should have at most 2 heavy neighbors: the OH oxygen + one chain atom
        if len(carbon_heavy_nbrs) <= 2:
            terminal_ohs.append({
                'oh_idx': atom.GetIdx(),
                'carbon_idx': carbon.GetIdx(),
            })

    # Only support single terminal OH for now
    if len(terminal_ohs) == 1:
        return terminal_ohs[0]

    return None


def _has_terminal_functional_group(
    mol: Chem.Mol, exclude_oh: bool = False
) -> bool:
    """Check if molecule has terminal functional groups (OH, NH2, SH, etc.).

    Terminal means an atom at degree 1 (or H-bearing heteroatom at chain end)
    that would normally take a functional group suffix.

    Args:
        mol: RDKit molecule object.
        exclude_oh: If True, ignore terminal OH groups (for suffix integration).

    Returns:
        True if terminal functional groups are present.
    """
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # Skip carbons and hydrogens
        if symbol in ('C', 'H'):
            continue

        if symbol not in REPLACEMENT_TERMS:
            continue

        # Check if this heteroatom is terminal (has H atoms indicating
        # a functional group: -OH, -NH2, -SH)
        num_h = atom.GetTotalNumHs()
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']

        if symbol == 'O' and num_h >= 1:
            if exclude_oh:
                continue  # Skip OH when checking for non-OH terminal FGs
            return True
        if symbol == 'N' and num_h >= 2 and len(heavy_neighbors) <= 1:
            # Terminal NH2 (primary amine at chain end)
            return True
        if symbol == 'S' and num_h >= 1:
            # Terminal SH (thiol)
            return True

    return False


def _find_replacement_chain(mol: Chem.Mol) -> Optional[List[int]]:
    """Find the longest chain backbone including heteroatoms.

    For acyclic molecules, finds the longest simple path between
    terminal atoms (degree 1 in heavy-atom graph).

    Args:
        mol: RDKit molecule object.

    Returns:
        List of atom indices forming the backbone, or None if no valid
        backbone found.
    """
    # Build adjacency list for heavy atoms only
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 3:
        return None

    adj: Dict[int, List[int]] = defaultdict(list)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        adj[a1].append(a2)
        adj[a2].append(a1)

    # Find terminal atoms (degree 1 in heavy-atom graph)
    terminals = [idx for idx in range(num_atoms) if len(adj[idx]) == 1]

    if len(terminals) < 2:
        # No clear chain endpoints -- not a chain molecule
        return None

    # For acyclic molecules, find the longest path using BFS from each terminal.
    # In a tree (acyclic graph), the longest path can be found by:
    # 1. BFS from any node to find the farthest node
    # 2. BFS from that farthest node to find the actual longest path
    # But since we need the actual path, we use DFS enumeration between terminal pairs.

    # Optimization: For trees, use double-BFS to find diameter endpoints
    # Step 1: BFS from first terminal to find farthest node
    farthest, _ = _bfs_farthest(adj, terminals[0], num_atoms)
    # Step 2: BFS from farthest to find the other end of the diameter
    other_end, _ = _bfs_farthest(adj, farthest, num_atoms)

    # Now find the actual path between farthest and other_end using BFS
    path = _find_path_bfs(adj, farthest, other_end, num_atoms)

    if path is None or len(path) < 3:
        return None

    return path


def _bfs_farthest(
    adj: Dict[int, List[int]], start: int, num_atoms: int
) -> Tuple[int, int]:
    """BFS from start, return (farthest_node, distance).

    Args:
        adj: Adjacency list.
        start: Starting atom index.
        num_atoms: Total number of atoms.

    Returns:
        Tuple of (farthest atom index, distance to it).
    """
    visited = [False] * num_atoms
    visited[start] = True
    queue = [(start, 0)]
    farthest = start
    max_dist = 0

    head = 0
    while head < len(queue):
        node, dist = queue[head]
        head += 1
        if dist > max_dist:
            max_dist = dist
            farthest = node
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))

    return farthest, max_dist


def _find_path_bfs(
    adj: Dict[int, List[int]], start: int, end: int, num_atoms: int
) -> Optional[List[int]]:
    """Find the path between start and end using BFS (for trees, this is unique).

    Args:
        adj: Adjacency list.
        start: Starting atom index.
        end: Ending atom index.
        num_atoms: Total number of atoms.

    Returns:
        List of atom indices from start to end, or None.
    """
    visited = [False] * num_atoms
    parent = [-1] * num_atoms
    visited[start] = True
    queue = [start]

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if node == end:
            # Reconstruct path
            path = []
            current = end
            while current != -1:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)

    return None


def _orient_oh_end_first(
    backbone: List[int], mol: Chem.Mol, carbon_idx: int
) -> List[int]:
    """Orient backbone so the carbon bearing the terminal OH gets locant 1.

    For replacement chains with terminal -ol suffix, the principal group
    (OH) must receive the lowest possible locant per IUPAC P-14.7.

    The OH oxygen has already been stripped from the backbone. This function
    ensures the carbon that was bonded to the OH is at position 0 (locant 1).

    Args:
        backbone: List of atom indices forming the backbone (OH oxygen excluded).
        mol: RDKit molecule object.
        carbon_idx: Atom index of the carbon bearing the -OH group.

    Returns:
        Reoriented backbone list with OH-bearing carbon at position 0.
    """
    if backbone[0] == carbon_idx:
        return backbone
    elif backbone[-1] == carbon_idx:
        return list(reversed(backbone))
    else:
        # Carbon not at either end -- shouldn't happen for unbranched chain
        # Fall back to lowest heteroatom locants
        return _orient_for_lowest_locants(backbone, mol)


def _orient_for_lowest_locants(
    backbone: List[int], mol: Chem.Mol
) -> List[int]:
    """Orient the backbone chain to give lowest locants to heteroatoms.

    Tries both directions and picks the one where the heteroatom locant
    set is numerically lower at the first point of difference.

    Args:
        backbone: List of atom indices forming the backbone.
        mol: RDKit molecule object.

    Returns:
        Reoriented backbone list.
    """
    forward = backbone
    reverse = list(reversed(backbone))

    forward_locants = _get_heteroatom_locants(forward, mol)
    reverse_locants = _get_heteroatom_locants(reverse, mol)

    # Compare locant sets at first point of difference
    for f_loc, r_loc in zip(forward_locants, reverse_locants):
        if f_loc < r_loc:
            return forward
        elif r_loc < f_loc:
            return reverse

    # If equal, return forward (arbitrary but deterministic)
    return forward


def _get_heteroatom_locants(backbone: List[int], mol: Chem.Mol) -> List[int]:
    """Get sorted locants of embedded heteroatoms in a backbone.

    Args:
        backbone: List of atom indices.
        mol: RDKit molecule object.

    Returns:
        Sorted list of 1-based locants for embedded heteroatoms.
    """
    locants = []
    for i, atom_idx in enumerate(backbone):
        if i == 0 or i == len(backbone) - 1:
            continue  # Skip terminal atoms
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() in REPLACEMENT_TERMS:
            locants.append(i + 1)  # 1-based
    return sorted(locants)


def _build_replacement_name(
    chain_length: int,
    heteroatom_positions: List[Tuple[int, str]],
    suffix: Optional[Tuple[str, int]] = None,
) -> str:
    """Build the skeletal replacement name from chain length and heteroatom info.

    Follows IUPAC P-15.4.3.1: replacement terms are cited in ascending
    locant order. When different elements share the same lowest locant
    (rare), alphabetical order of the replacement term breaks the tie.

    Args:
        chain_length: Total number of atoms in the backbone.
        heteroatom_positions: List of (locant, element_symbol) tuples.
        suffix: Optional (suffix_name, locant) tuple for terminal FG,
                e.g., ('ol', 1) for terminal alcohol.

    Returns:
        Complete replacement name string.
    """
    # Get the chain prefix (hex, oct, non, etc.)
    chain_prefix = get_chain_prefix(chain_length)

    # Group heteroatoms by element symbol
    element_groups: Dict[str, List[int]] = defaultdict(list)
    for locant, symbol in heteroatom_positions:
        element_groups[symbol].append(locant)

    # Sort each group's locants
    for symbol in element_groups:
        element_groups[symbol].sort()

    # Sort groups: by lowest locant, then alphabetically by replacement term
    sorted_groups = sorted(
        element_groups.items(),
        key=lambda item: (min(item[1]), REPLACEMENT_TERMS[item[0]])
    )

    # Build replacement term parts
    parts = []
    for symbol, locants in sorted_groups:
        term = REPLACEMENT_TERMS[symbol]
        locant_str = ','.join(str(loc) for loc in locants)
        count = len(locants)

        if count == 1:
            multiplier = ''
        elif count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            # Fallback for very large counts (unlikely for replacement)
            multiplier = SIMPLE_MULTIPLIERS.get(count, f'{count}')

        parts.append(f'{locant_str}-{multiplier}{term}')

    # Join parts with hyphens
    replacement_prefix = '-'.join(parts)

    if suffix is not None:
        # Build name with functional group suffix
        # e.g., "3,6-dioxaoctan-1-ol"
        suffix_name, suffix_locant = suffix
        # Vowel elision: remove terminal 'e' before suffix starting with vowel
        # "octane" -> "octan" before "-1-ol"
        stem = f'{chain_prefix}an'
        if suffix_name.startswith(('a', 'e', 'i', 'o', 'u', 'y')):
            # "an" already drops the 'e' from "ane"
            pass
        else:
            stem = f'{chain_prefix}ane'
        return f'{replacement_prefix}{stem}-{suffix_locant}-{suffix_name}'
    else:
        # Plain replacement name: "3,6-dioxaoctane"
        return f'{replacement_prefix}{chain_prefix}ane'


def _try_cyclic_replacement_name(mol: Chem.Mol, ring_info) -> Optional[str]:
    """Try cyclic skeletal replacement naming for large heterocyclic rings.

    Per IUPAC P-22.1.3, large heterocyclic rings (>= 7 members) can use
    replacement nomenclature (oxa-/aza-/thia- prefixes on cycloalkane parent).
    Example: 1,4-dioxacyclononane for a 9-membered ring with 2 oxygens.

    Only applies when:
    - Single ring with >= 7 members
    - Ring contains heteroatoms from REPLACEMENT_TERMS
    - No substituents off the ring (all heavy atoms are ring atoms)
    - No priority functional groups (checked before calling this function)

    Args:
        mol: RDKit molecule object.
        ring_info: RDKit RingInfo object.

    Returns:
        Replacement name string or None if not applicable.
    """
    rings = ring_info.AtomRings()
    if len(rings) != 1:
        return None  # Only handle single-ring molecules for now

    ring = rings[0]
    ring_size = len(ring)

    # Gate: all heavy atoms must be in the ring (no substituents)
    if mol.GetNumAtoms() != ring_size:
        return None

    # Collect heteroatom positions in the ring
    ring_atoms = list(ring)
    heteroatoms = []
    for i, atom_idx in enumerate(ring_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol in REPLACEMENT_TERMS:
            heteroatoms.append((i, symbol))

    if not heteroatoms:
        return None

    # Check bond types: detect saturated vs unsaturated vs aromatic
    all_single = True
    has_aromatic = False
    for i in range(ring_size):
        a1 = ring_atoms[i]
        a2 = ring_atoms[(i + 1) % ring_size]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond:
            btype = bond.GetBondTypeAsDouble()
            if btype == 1.5:
                has_aromatic = True
                break
            elif btype != 1.0:
                all_single = False

    # Aromatic rings use Hantzsch-Widman or retained names, not replacement
    if has_aromatic:
        return None

    # Orient the ring to give lowest locants to heteroatoms.
    # Try all rotations and both directions; pick the one giving the
    # lowest heteroatom locant set at first point of difference.
    # For unsaturated rings, double bond locants serve as tiebreaker
    # after heteroatom locants (per IUPAC P-31.1.3.4).
    best_key = None
    best_positions = None
    best_ordered = None

    for start in range(ring_size):
        for direction in [1, -1]:
            # Build the ordered ring from this starting point
            ordered = []
            for step in range(ring_size):
                idx = (start + step * direction) % ring_size
                ordered.append(ring_atoms[idx])

            # Compute heteroatom locants (1-based)
            positions = []
            for i, atom_idx in enumerate(ordered):
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                if symbol in REPLACEMENT_TERMS:
                    positions.append((i + 1, symbol))

            hetero_locant_set = sorted(pos[0] for pos in positions)

            # Compute double bond locants for tiebreaker
            db_locants = []
            if not all_single:
                for i in range(ring_size):
                    a1 = ordered[i]
                    a2 = ordered[(i + 1) % ring_size]
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        # Locant of double bond = lower-numbered atom (1-based)
                        db_locants.append(i + 1)
                db_locants.sort()

            # Combined comparison key: heteroatom locants first, then DB locants
            comparison_key = (hetero_locant_set, db_locants)

            if best_key is None or comparison_key < best_key:
                best_key = comparison_key
                best_positions = positions
                best_ordered = ordered

    if not best_positions:
        return None

    # Build the cyclic replacement name using cyclo- prefix
    chain_prefix = get_chain_prefix(ring_size)

    # Group heteroatoms by element symbol
    element_groups: Dict[str, List[int]] = defaultdict(list)
    for locant, symbol in best_positions:
        element_groups[symbol].append(locant)

    for symbol in element_groups:
        element_groups[symbol].sort()

    # Sort groups: by lowest locant, then alphabetically by replacement term
    sorted_groups = sorted(
        element_groups.items(),
        key=lambda item: (min(item[1]), REPLACEMENT_TERMS[item[0]])
    )

    parts = []
    for symbol, locants in sorted_groups:
        term = REPLACEMENT_TERMS[symbol]
        locant_str = ','.join(str(loc) for loc in locants)
        count = len(locants)

        if count == 1:
            multiplier = ''
        elif count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(count, f'{count}')

        parts.append(f'{locant_str}-{multiplier}{term}')

    replacement_prefix = '-'.join(parts)

    if all_single:
        return f'{replacement_prefix}cyclo{chain_prefix}ane'

    # Unsaturated large heterocyclic rings: build name with -ene/-adiene/-atriene
    # Compute double bond locants from the best orientation
    db_locants = []
    for i in range(ring_size):
        a1 = best_ordered[i]
        a2 = best_ordered[(i + 1) % ring_size]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond and bond.GetBondTypeAsDouble() == 2.0:
            db_locants.append(i + 1)
    db_locants.sort()

    if not db_locants:
        # No double bonds found despite all_single being False (shouldn't happen)
        return f'{replacement_prefix}cyclo{chain_prefix}ane'

    return _build_unsaturated_cyclic_name(
        replacement_prefix, chain_prefix, db_locants
    )


def _build_unsaturated_cyclic_name(
    replacement_prefix: str,
    chain_prefix: str,
    double_bond_locants: List[int],
) -> str:
    """Build cyclic replacement name with unsaturation suffix.

    Constructs names like '1-oxacyclohept-4,5-diene' from the replacement
    prefix, chain size prefix, and double bond locant positions.

    Handles vowel elision: when chain_prefix ends in 'a' and the suffix
    starts with a vowel, the trailing 'a' is dropped (e.g., 'octa' + 'ene'
    becomes 'octene', not 'octaene'). Since get_chain_prefix() returns
    stems without trailing 'a' (e.g., 'oct', 'dec'), and we build the
    suffix directly, no special elision is needed for most cases.

    For single double bond: '-{locant}-ene'
    For 2 double bonds: '-{loc1},{loc2}-diene'  (with linking 'a')
    For 3 double bonds: '-{loc1},{loc2},{loc3}-triene' (with linking 'a')

    Args:
        replacement_prefix: Heteroatom prefix (e.g., '1-oxa').
        chain_prefix: Ring size prefix from get_chain_prefix() (e.g., 'hept').
        double_bond_locants: Sorted list of 1-based locant positions for
            double bonds.

    Returns:
        Complete unsaturated cyclic replacement name string.
    """
    count = len(double_bond_locants)
    locant_str = ','.join(str(loc) for loc in double_bond_locants)

    if count == 1:
        suffix = 'ene'
    else:
        # For multiple double bonds: multiplier + 'ene'
        # 2 DB = "diene", 3 DB = "triene", 4 DB = "tetraene", etc.
        if count in SIMPLE_MULTIPLIERS:
            multiplier = SIMPLE_MULTIPLIERS[count]
        else:
            multiplier = str(count)
        suffix = f'{multiplier}ene'

    # Build stem: cyclo + chain_prefix
    # get_chain_prefix() returns e.g., 'hept', 'oct', 'dec' (no trailing 'a')
    # IUPAC convention for unsaturation:
    # - Single ene: stem without linking vowel -> cyclohept-2-ene
    # - Multiple ene: stem with linking 'a' -> cyclohepta-2,4-diene
    # The linking 'a' goes on the stem when the suffix starts with a
    # consonant (d in diene, t in triene).
    stem = chain_prefix
    if count > 1:
        # Add linking vowel 'a' to chain prefix for multi-ene
        # "hept" -> "hepta", "oct" -> "octa", "dec" -> "deca"
        stem = chain_prefix + 'a'

    return f'{replacement_prefix}cyclo{stem}-{locant_str}-{suffix}'
