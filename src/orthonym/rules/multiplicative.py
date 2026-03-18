"""Multiplicative nomenclature for symmetric bridge-linked molecules (IUPAC P-51.3).

Detects molecules with two or more identical parent structures connected by a
polyvalent linking group (bridge) and produces multiplicative names like
"4,4'-methylenedianiline" or "4,4'-oxydibenzoic acid".

Three conditions for multiplicative naming (IUPAC P-51.3):
    1. Two or more identical parent structures (verified by canonical SMILES)
    2. Connected by a polyvalent linking group (the bridge atom(s))
    3. Parent has a principal characteristic group (suffix-forming FG)

Uses the recursion depth guard from fragment_naming.py to prevent infinite
loops when naming fragment halves.
"""

from typing import Optional, List, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import RWMol

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS


# ---------------------------------------------------------------------------
# Saturation/modification prefixes that must not be preceded by "di"
# ---------------------------------------------------------------------------
# When a parent name starts with one of these prefixes, multiplicative naming
# would produce unparseable concatenation (e.g., "ditetrahydropyran").
# In such cases, fall through to substitutive naming instead.

SATURATION_PREFIXES = [
    'tetrahydro', 'dihydro', 'hexahydro', 'perhydro', 'octahydro',
    'decahydro', 'dodecahydro',
]


# ---------------------------------------------------------------------------
# Bridge definitions: atom pattern -> IUPAC bridge name
# ---------------------------------------------------------------------------

# Single-atom bridge names
_SINGLE_ATOM_BRIDGES: Dict[str, str] = {
    "O": "oxy",
    "S": "thio",
    "NH": "imino",
    "CH2": "methylene",
}

# Two-atom bridge patterns
_TWO_ATOM_BRIDGES = [
    # (element1, element2, h_count1, h_count2, bridge_name)
    ("C", "C", 2, 2, "ethylene"),     # CH2-CH2
    ("C", "C", 1, 1, "vinylene"),     # CH=CH (rare)
]

# Multi-atom bridge names: (element, H_count, ring_neighbor_count) -> bridge_name
# These handle star-topology bridges where 3+ identical parents radiate from
# a single central atom (IUPAC P-51.3.3).
_MULTI_BRIDGE_NAMES: Dict[Tuple[str, int, int], str] = {
    ("N", 0, 3): "nitrilo",           # N connecting 3 rings (trivalent)
    ("C", 1, 3): "methylidyne",       # CH connecting 3 rings (trivalent)
    ("C", 0, 4): "methanetetrayl",    # C connecting 4 rings (tetravalent)
}


# Retained parent names for common ring+FG combinations
# Maps canonical SMILES -> (retained_name, fg_locant)
# fg_locant is the IUPAC position of the principal group
_RETAINED_PARENT_NAMES: Dict[str, Tuple[str, int]] = {
    "Nc1ccccc1": ("aniline", 1),           # 4-aminophenyl = aniline
    "OC(=O)c1ccccc1": ("benzoic acid", 1), # benzoic acid
    "O=C(O)c1ccccc1": ("benzoic acid", 1), # alternate SMILES
    "Oc1ccccc1": ("phenol", 1),            # phenol
}


def name_multiplicative(mol) -> Optional[str]:
    """Detect and name multiplicative nomenclature cases.

    Args:
        mol: RDKit Mol object.

    Returns:
        Multiplicative IUPAC name if applicable, None otherwise.
    """
    if mol is None:
        return None

    # Quick reject: need at least 2 ring systems
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return None

    # Collect ring atom indices
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)

    # --- Try single-atom bridges first ---
    result = _try_single_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try two-atom bridges ---
    result = _try_two_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try multi-atom bridges (3+ units, star topology) ---
    result = _try_multi_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    return None


def _try_single_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find single-atom bridges between identical ring systems."""
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]

        # Single-atom bridge: exactly 2 heavy neighbors, both in rings
        if len(heavy_neighbors) != 2:
            continue
        if not all(n.GetIdx() in ring_atoms for n in heavy_neighbors):
            continue

        # Classify the bridge atom
        bridge_type = _classify_single_atom_bridge(atom)
        if bridge_type is None:
            continue

        bridge_name = _SINGLE_ATOM_BRIDGES.get(bridge_type)
        if bridge_name is None:
            continue

        # Split the molecule at the bridge
        nbr_indices = [n.GetIdx() for n in heavy_neighbors]
        fragments = _split_at_bridge(mol, idx, nbr_indices)
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b, conn_atom_a, conn_atom_b = fragments

        # Check if fragments are identical
        canon_a = Chem.CanonSmiles(frag_smiles_a)
        canon_b = Chem.CanonSmiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        # Name the parent structure
        parent_name = _name_parent(canon_a)
        if parent_name is None:
            continue

        # Get locant of bridge attachment point in the parent
        locant = _get_bridge_locant(mol, idx, nbr_indices[0], ring_atoms)

        # Assemble multiplicative name (returns None for prefix-derived parents)
        result = _assemble_multiplicative_name(locant, bridge_name, parent_name)
        if result is not None:
            return result
        # Saturation-prefix parent: skip multiplicative, let caller fall through
        continue

    return None


def _try_two_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find two-atom bridges (e.g., CH2-CH2 ethylene) between identical ring systems."""
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        idx1 = a1.GetIdx()
        idx2 = a2.GetIdx()

        # Both bridge atoms must NOT be in rings
        if idx1 in ring_atoms or idx2 in ring_atoms:
            continue

        # Each bridge atom must have exactly one ring neighbor (besides its bridge partner)
        ring_nbrs_1 = [n for n in a1.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx2]
        ring_nbrs_2 = [n for n in a2.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx1]

        # Each atom in the bridge must connect to exactly one ring atom
        if len(ring_nbrs_1) != 1 or len(ring_nbrs_2) != 1:
            continue

        # Each bridge atom must have exactly 2 heavy neighbors total
        # (one ring atom + one bridge partner)
        heavy_nbrs_1 = [n for n in a1.GetNeighbors() if n.GetAtomicNum() > 1]
        heavy_nbrs_2 = [n for n in a2.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs_1) != 2 or len(heavy_nbrs_2) != 2:
            continue

        # Classify the two-atom bridge
        bridge_name = _classify_two_atom_bridge(a1, a2)
        if bridge_name is None:
            continue

        # Split at both bridge atoms
        ring_conn_1 = ring_nbrs_1[0].GetIdx()
        ring_conn_2 = ring_nbrs_2[0].GetIdx()

        fragments = _split_at_two_atom_bridge(
            mol, idx1, idx2, ring_conn_1, ring_conn_2
        )
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b = fragments

        # Check if fragments are identical
        canon_a = Chem.CanonSmiles(frag_smiles_a)
        canon_b = Chem.CanonSmiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        # Name the parent structure
        parent_name = _name_parent(canon_a)
        if parent_name is None:
            continue

        # Get locant
        locant = _get_bridge_locant(mol, idx1, ring_conn_1, ring_atoms)

        # Assemble multiplicative name (returns None for prefix-derived parents)
        result = _assemble_multiplicative_name(locant, bridge_name, parent_name)
        if result is not None:
            return result
        # Saturation-prefix parent: skip multiplicative, let caller fall through
        continue

    return None


def _classify_single_atom_bridge(atom) -> Optional[str]:
    """Classify a single bridge atom into a known type.

    Returns:
        Bridge type key (e.g., 'CH2', 'O', 'NH', 'S') or None.
    """
    sym = atom.GetSymbol()
    total_h = atom.GetTotalNumHs()

    if sym == "C" and total_h == 2:
        return "CH2"
    elif sym == "O" and total_h == 0:
        return "O"
    elif sym == "N" and total_h == 1:
        return "NH"
    elif sym == "S" and total_h == 0:
        return "S"
    return None


def _classify_two_atom_bridge(atom1, atom2) -> Optional[str]:
    """Classify a two-atom bridge into a known type."""
    sym1 = atom1.GetSymbol()
    sym2 = atom2.GetSymbol()
    h1 = atom1.GetTotalNumHs()
    h2 = atom2.GetTotalNumHs()

    for s1, s2, expected_h1, expected_h2, name in _TWO_ATOM_BRIDGES:
        if sym1 == s1 and sym2 == s2 and h1 == expected_h1 and h2 == expected_h2:
            return name
        # Check reverse order
        if sym1 == s2 and sym2 == s1 and h1 == expected_h2 and h2 == expected_h1:
            return name
    return None


def _classify_multi_bridge(atom, ring_nbr_count: int) -> Optional[str]:
    """Classify a multi-valent bridge atom by element + H count + ring neighbor count.

    Args:
        atom: RDKit atom (the bridge candidate).
        ring_nbr_count: Number of ring-atom neighbors.

    Returns:
        Bridge name (e.g., 'nitrilo', 'methylidyne', 'methanetetrayl') or None.
    """
    sym = atom.GetSymbol()
    h = atom.GetTotalNumHs()
    return _MULTI_BRIDGE_NAMES.get((sym, h, ring_nbr_count))


def _try_multi_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find star-topology bridges connecting 3+ identical ring systems.

    A multi-atom bridge is a single non-ring atom connected to 3 or more ring
    atoms, where removing the bridge produces 3+ identical fragments.

    Examples:
        N connecting 3 phenol rings -> nitrilotriphenol
        CH connecting 3 phenol rings -> methylidynetriphenol
        C connecting 4 phenol rings -> methanetetrayltetraphenol
    """
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]

        # Count ring neighbors
        ring_neighbors = [n for n in heavy_neighbors if n.GetIdx() in ring_atoms]
        ring_nbr_count = len(ring_neighbors)

        # Must have 3+ ring neighbors for multi-bridge
        if ring_nbr_count < 3:
            continue

        # All heavy neighbors must be ring atoms (no non-ring non-H substituents)
        if len(heavy_neighbors) != ring_nbr_count:
            continue

        # Classify the bridge
        bridge_name = _classify_multi_bridge(atom, ring_nbr_count)
        if bridge_name is None:
            continue

        # Split molecule by removing the bridge atom
        emol = RWMol(Chem.RWMol(mol))
        emol.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(emol)
        except Exception:
            continue

        result_mol = emol.GetMol()
        frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
        unit_count = len(frag_mols)

        if unit_count < 3:
            continue

        # Check all fragments are identical
        canon_smiles_list = [Chem.MolToSmiles(f) for f in frag_mols]
        canon_set = set(Chem.CanonSmiles(s) for s in canon_smiles_list)
        if len(canon_set) != 1:
            continue  # Non-identical fragments

        canon_parent = canon_set.pop()

        # Name the parent structure
        parent_name = _name_parent(canon_parent)
        if parent_name is None:
            continue

        # Get bridge locant using the first ring neighbor
        first_ring_nbr_idx = ring_neighbors[0].GetIdx()
        locant = _get_bridge_locant(mol, idx, first_ring_nbr_idx, ring_atoms)

        # Assemble multiplicative name with unit_count
        result = _assemble_multiplicative_name(
            locant, bridge_name, parent_name, unit_count=unit_count
        )
        if result is not None:
            return result
        continue

    return None


def _split_at_bridge(
    mol, bridge_idx: int, nbr_indices: List[int]
) -> Optional[Tuple[str, str, int, int]]:
    """Split molecule by removing a single bridge atom.

    Returns:
        (frag_smiles_a, frag_smiles_b, conn_atom_in_a, conn_atom_in_b)
        or None if split doesn't produce exactly 2 fragments.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Record neighbors before removal (atom indices will shift!)
    nbr_a, nbr_b = nbr_indices[0], nbr_indices[1]

    # Remove bridge atom
    emol.RemoveAtom(bridge_idx)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_indices = Chem.GetMolFrags(result_mol)

    if len(frag_indices) != 2:
        return None

    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    # Adjust neighbor indices for atom removal
    # When atom at bridge_idx is removed, all atoms with idx > bridge_idx shift down by 1
    adj_a = nbr_a if nbr_a < bridge_idx else nbr_a - 1
    adj_b = nbr_b if nbr_b < bridge_idx else nbr_b - 1

    return (smi_a, smi_b, adj_a, adj_b)


def _split_at_two_atom_bridge(
    mol, bridge_idx1: int, bridge_idx2: int,
    ring_conn1: int, ring_conn2: int
) -> Optional[Tuple[str, str]]:
    """Split molecule by removing two bridge atoms.

    Returns:
        (frag_smiles_a, frag_smiles_b) or None.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Remove in reverse index order to avoid index shifting issues
    idx_high = max(bridge_idx1, bridge_idx2)
    idx_low = min(bridge_idx1, bridge_idx2)

    emol.RemoveAtom(idx_high)
    emol.RemoveAtom(idx_low)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)

    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    return (smi_a, smi_b)


def _name_parent(canon_smiles: str) -> Optional[str]:
    """Name the parent structure using the fragment naming guard.

    Uses name_fragment_recursively to prevent infinite loops.

    Args:
        canon_smiles: Canonical SMILES of the parent fragment.

    Returns:
        Parent name or None if naming fails.
    """
    from ..assembly.fragment_naming import name_fragment_recursively

    # Check retained parent names first
    if canon_smiles in _RETAINED_PARENT_NAMES:
        return _RETAINED_PARENT_NAMES[canon_smiles][0]

    # Use recursive naming with depth guard
    name = name_fragment_recursively(canon_smiles)
    return name


def _get_bridge_locant(
    mol, bridge_idx: int, ring_conn_idx: int, ring_atoms: set
) -> int:
    """Determine the IUPAC locant of the bridge attachment point in the parent ring.

    For benzene-based systems, the locant is the position of the attachment
    point relative to the principal group (which is position 1).

    Args:
        mol: RDKit Mol object.
        bridge_idx: Index of the bridge atom.
        ring_conn_idx: Index of the ring atom connected to the bridge.
        ring_atoms: Set of all ring atom indices.

    Returns:
        Locant number (1-indexed).
    """
    # Find which ring the connection atom belongs to
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if ring_conn_idx in ring:
            target_ring = ring
            break

    if target_ring is None:
        return 4  # Default to 4 (most common para position)

    # Find the principal group atom in this ring
    # The principal group is typically an atom connected to a non-ring,
    # non-bridge functional group atom (N for aniline, C(=O) for acid, O for phenol)
    pg_atom_idx = _find_principal_group_atom_in_ring(
        mol, target_ring, ring_atoms, bridge_idx
    )

    if pg_atom_idx is None:
        return 4  # Default

    # Calculate the shortest path distance in the ring between pg_atom and conn_atom
    # The locant is this distance + 1 (since pg is position 1)
    ring_list = list(target_ring)

    # Find positions in ring
    pg_pos = ring_list.index(pg_atom_idx)
    conn_pos = ring_list.index(ring_conn_idx)

    ring_size = len(ring_list)
    # Shortest ring distance
    dist = abs(conn_pos - pg_pos)
    dist = min(dist, ring_size - dist)

    return dist + 1  # 1-indexed (pg is at position 1)


def _find_principal_group_atom_in_ring(
    mol, ring: tuple, ring_atoms: set, bridge_idx: int
) -> Optional[int]:
    """Find the ring atom bearing the principal characteristic group.

    The principal group atom is a ring atom that has a non-ring, non-bridge
    neighbor that is part of a functional group (N, O attached to C=O, etc.).

    Returns:
        Atom index of the ring atom bearing the principal group, or None.
    """
    ring_set = set(ring)

    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            # Skip ring atoms and the bridge atom
            if nbr_idx in ring_set or nbr_idx == bridge_idx:
                continue
            if nbr_idx in ring_atoms:
                continue  # Part of another ring

            # This is a non-ring, non-bridge substituent
            # Check if it's a functional group atom
            sym = nbr.GetSymbol()
            if sym in ("N", "O", "S"):
                return idx
            # Check for C=O type groups (carboxylic acid, aldehyde, etc.)
            if sym == "C":
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() != idx:
                        bond = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                        if (bond and bond.GetBondTypeAsDouble() == 2.0
                                and nbr2.GetSymbol() == "O"):
                            return idx

    return None


def _build_primed_locant_str(locant: int, unit_count: int) -> str:
    """Build a primed locant string for multiplicative naming.

    For each unit i in [0, unit_count), appends i primes to the locant.
    Example: locant=4, unit_count=3 -> "4,4',4''"
    Example: locant=4, unit_count=4 -> "4,4',4'',4'''"

    Args:
        locant: The IUPAC locant number.
        unit_count: Number of identical parent units.

    Returns:
        Comma-separated primed locant string.
    """
    prime = "'"
    parts = []
    for i in range(unit_count):
        parts.append(f"{locant}{prime * i}")
    return ",".join(parts)


def _assemble_multiplicative_name(
    locant: int, bridge_name: str, parent_name: str, unit_count: int = 2
) -> Optional[str]:
    """Assemble the final multiplicative name.

    Format: [locants]-[bridge][multiplier][parent]
    Examples:
        4,4'-methylenedianiline (2 units)
        4,4',4''-nitrilotriphenol (3 units)
        4,4',4'',4'''-methanetetrayltetraphenol (4 units)

    For acid names with spaces (like "benzoic acid"), the multiplier prefix
    goes before the base name: 4,4'-oxydibenzoic acid

    If the parent name starts with a saturation/modification prefix
    (e.g., "tetrahydro", "dihydro"), returns None to signal that
    multiplicative naming would produce an unparseable concatenation
    (e.g., "ditetrahydropyran") and the caller should fall through
    to substitutive naming.

    Args:
        locant: IUPAC locant of the bridge attachment point.
        bridge_name: Name of the bridge group (e.g., "methylene", "oxy").
        parent_name: IUPAC name of one parent unit (e.g., "aniline", "benzoic acid").
        unit_count: Number of identical parent units (default 2 for backward compat).

    Returns:
        Complete multiplicative name, or None if the parent name has a
        saturation prefix that would produce unparseable multiplier+prefix output.
    """
    # Check for saturation/modification prefixes in the parent name
    # that would produce unparseable "di+prefix" concatenation
    parent_lower = parent_name.lower()
    if any(parent_lower.startswith(p) for p in SATURATION_PREFIXES):
        return None

    # Get multiplier from SIMPLE_MULTIPLIERS (di, tri, tetra, etc.)
    multiplier = SIMPLE_MULTIPLIERS.get(unit_count, "")
    if not multiplier:
        return None

    # Build primed locant string: e.g., "4,4'" for 2 units, "4,4',4''" for 3
    locant_str = _build_primed_locant_str(locant, unit_count)

    # Handle names with spaces (e.g., "benzoic acid" -> "tribenzoic acid")
    if " " in parent_name:
        parts = parent_name.split(" ", 1)
        base = parts[0]  # "benzoic"
        suffix = parts[1]  # "acid"
        return f"{locant_str}-{bridge_name}{multiplier}{base} {suffix}"
    else:
        # Simple name: "aniline" -> "dianiline"
        return f"{locant_str}-{bridge_name}{multiplier}{parent_name}"
