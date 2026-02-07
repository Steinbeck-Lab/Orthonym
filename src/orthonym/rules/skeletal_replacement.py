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
    'Si': 'sila',
    'B': 'bora',
}

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
    # Gate 1: No rings allowed (chain replacement only)
    # ----------------------------------------------------------------
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        return None

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

    # For single heteroatom: only apply if chain is long enough (>= 6)
    # Short/moderate chains (COC=3, COCC=4, CCOCC=5) use substitutive naming
    # (methoxymethane, ethoxyethane, etc.). Replacement names preferred for
    # longer chains where substitutive names become awkward.
    if len(embedded_heteroatoms) == 1 and len(backbone) < 6:
        return None

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
