"""
Ring assembly detection and naming per IUPAC P-28.

Ring assemblies are molecules composed of two or more identical ring systems
joined by single bonds. Examples: biphenyl, bipyridine, bithiophene.

IUPAC P-28.2.1: Ring assemblies use multiplicative prefixes (bi-, ter-, quater-)
before the parent ring name. Benzene assemblies use "phenyl" as base name.

Locant convention: First ring unprimed, second primed ('), third double-primed ('').
Connection locants are separated by commas, multi-connection pairs by colons.

Example outputs:
  - biphenyl SMILES -> "1,1'-biphenyl"
  - 2,2'-bipyridine SMILES -> "2,2'-bipyridine"
  - 4-chlorobiphenyl SMILES -> "4-chloro-1,1'-biphenyl"
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem

from ..perception.rings import get_ring_info, is_aromatic_ring


# Assembly multiplier prefixes (IUPAC P-28.2)
ASSEMBLY_MULTIPLIERS = {
    2: "bi",
    3: "ter",
    4: "quater",
}


def _system_signature(mol, system_atoms: Set[int]) -> Tuple:
    """
    Compute a ring system signature for identity comparison.

    The signature includes sorted element symbols, sorted ring sizes that
    are fully contained within the system, and aromaticity of the system.
    Two systems are identical if and only if their signatures match.

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in the ring system

    Returns:
        Tuple of (sorted_elements, sorted_ring_sizes, is_aromatic)
    """
    elements = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in system_atoms)

    ri = mol.GetRingInfo()
    ring_sizes = []
    for ring in ri.AtomRings():
        if set(ring) <= system_atoms:
            ring_sizes.append(len(ring))

    is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in system_atoms)

    return (tuple(elements), tuple(sorted(ring_sizes)), is_aromatic)


def _find_inter_system_bonds(
    mol, ring_systems: List[Set[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Find all single bonds connecting different ring systems.

    Returns list of (atom_idx_A, atom_idx_B, system_idx_A, system_idx_B).
    Only SINGLE and AROMATIC bond types are accepted (biphenyl's inter-ring
    bond may be typed as either depending on Kekulization).
    """
    connections = []
    acceptable_types = {
        rdchem.BondType.SINGLE,
        rdchem.BondType.AROMATIC,
    }

    # Build atom -> system index map
    atom_to_system = {}
    for sys_idx, system in enumerate(ring_systems):
        for atom_idx in system:
            atom_to_system[atom_idx] = sys_idx

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # Both atoms must be in ring systems, but different ones
        if a1 in atom_to_system and a2 in atom_to_system:
            s1 = atom_to_system[a1]
            s2 = atom_to_system[a2]
            if s1 != s2 and bond.GetBondType() in acceptable_types:
                connections.append((a1, a2, s1, s2))

    return connections


def _check_all_connected(
    num_systems: int, connections: List[Tuple[int, int, int, int]]
) -> bool:
    """
    Check that all ring systems are connected via inter-system bonds.

    Uses union-find to verify connectivity.
    """
    if num_systems <= 1:
        return True

    parent = list(range(num_systems))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for _, _, s1, s2 in connections:
        union(s1, s2)

    roots = set(find(i) for i in range(num_systems))
    return len(roots) == 1


def detect_ring_assembly(
    mol, ring_systems: List[Set[int]]
) -> Optional[Dict]:
    """
    Detect if a molecule is a ring assembly (identical ring systems joined
    by single bonds).

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices, one per ring system
                      (from get_ring_systems())

    Returns:
        Dict with assembly info if detected, None otherwise.
        Dict keys:
          - ring_systems: list of sets of atom indices
          - connections: list of (atom_A, atom_B, system_A, system_B) tuples
          - count: number of identical ring systems
          - ring_type: 'carbocyclic' or 'heterocyclic'
    """
    if len(ring_systems) < 2:
        return None

    # Find inter-system single bonds
    connections = _find_inter_system_bonds(mol, ring_systems)
    if not connections:
        return None

    # Check that all systems are connected
    if not _check_all_connected(len(ring_systems), connections):
        return None

    # Compare signatures -- all must be identical
    signatures = [_system_signature(mol, sys_atoms) for sys_atoms in ring_systems]
    if len(set(signatures)) != 1:
        return None

    # Additional guard: reject if there are non-ring atoms in the molecule
    # other than substituents (i.e., linker atoms between rings).
    # For true ring assemblies, the inter-ring bond is direct (no linker).
    # This is already ensured by _find_inter_system_bonds checking that
    # both atoms are IN ring systems.

    # Determine ring type
    elements = signatures[0][0]
    has_heteroatom = any(e != 'C' for e in elements)
    ring_type = 'heterocyclic' if has_heteroatom else 'carbocyclic'

    return {
        'ring_systems': ring_systems,
        'connections': connections,
        'count': len(ring_systems),
        'ring_type': ring_type,
    }


def _get_ring_parent_name(mol, system_atoms: Set[int]) -> Optional[str]:
    """
    Determine the parent name for a ring system in an assembly context.

    Per IUPAC P-28.2.1:
      - Benzene (6-membered all-C aromatic) -> "phenyl" in assemblies
      - Heterocycles -> their parent name (pyridine, thiophene, furan, etc.)
      - Cycloalkanes -> their parent name (cyclohexane, cyclopentane, etc.)

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in one ring system

    Returns:
        Parent name string, or None if cannot determine.
    """
    # Get the individual rings that belong to this system
    ri = mol.GetRingInfo()
    system_rings = [ring for ring in ri.AtomRings() if set(ring) <= system_atoms]

    if not system_rings:
        return None

    # For single-ring systems (most common in assemblies)
    if len(system_rings) == 1:
        ring = system_rings[0]
        ring_size = len(ring)

        # Check if heterocyclic
        has_heteroatom = any(
            mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring
        )

        if has_heteroatom:
            # Use the heterocycle naming infrastructure
            from .heterocycles import name_heterocycle
            return name_heterocycle(mol, ring)

        # All-carbon ring
        is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)

        if is_aromatic and ring_size == 6:
            # IUPAC P-28.2.1: benzene in assemblies uses "phenyl"
            return "phenyl"

        # Cycloalkane or cycloalkene
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(ring_size)

        # Check saturation
        has_double_bond = False
        ring_set = set(ring)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in ring_set and a2 in ring_set:
                if bond.GetBondType() == rdchem.BondType.DOUBLE:
                    has_double_bond = True
                    break

        if has_double_bond:
            return f"cyclo{prefix}ene"
        else:
            return f"cyclo{prefix}ane"

    # Multi-ring fused systems as assembly units (rare but possible)
    # e.g., binaphthalene -- two identical fused ring systems
    # For now, return None; this would need fused ring naming
    return None


def _get_connection_locant(
    mol, connecting_atom: int, system_atoms: Set[int]
) -> int:
    """
    Determine the IUPAC locant for a connecting atom within its ring system.

    For heterocyclic rings, the locant follows IUPAC numbering with the
    direction chosen to give the LOWEST locant at the connection point
    (IUPAC P-28.3.1 lowest-locant rule for assemblies).

    For carbocyclic rings, position 1 is the connecting atom itself
    (for unsubstituted benzene, all positions are equivalent).

    Args:
        mol: RDKit Mol object
        connecting_atom: Atom index of the connection point
        system_atoms: Set of atom indices in the ring system

    Returns:
        IUPAC locant number (1-indexed)
    """
    # Get the individual ring(s) containing this atom
    ri = mol.GetRingInfo()
    atom_ring = None
    for ring in ri.AtomRings():
        if connecting_atom in ring and set(ring) <= system_atoms:
            atom_ring = ring
            break

    if atom_ring is None:
        return 1  # Fallback

    ring_list = list(atom_ring)

    # Check if heterocyclic
    has_heteroatom = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_list
    )

    if has_heteroatom:
        # For heterocyclic assemblies: apply standard IUPAC numbering
        # (heteroatom at position 1), then try both numbering directions
        # and pick the one giving the LOWEST locant at the connection point.
        # orient_heterocycle may pick an arbitrary direction for single-heteroatom
        # rings, so we need to try both explicitly.
        from .heterocycles import (
            number_heterocycle_ring,
            classify_heterocycle,
        )
        from ..data.hw_heteroatoms import get_heteroatom_priority

        info = classify_heterocycle(mol, ring_list)
        heteroatoms = info['heteroatoms']

        if not heteroatoms:
            return 1

        # Find highest-priority heteroatom for position 1
        sorted_ha = sorted(
            heteroatoms, key=lambda x: get_heteroatom_priority(x[1])
        )
        start_idx = sorted_ha[0][0]
        n = len(ring_list)
        start_pos = ring_list.index(start_idx)

        # Try both directions, pick the one giving lowest connection locant
        best_locant = n  # Worst case

        for direction in (1, -1):
            oriented = []
            for i in range(n):
                oriented.append(ring_list[(start_pos + direction * i) % n])
            # Find connection locant in this orientation
            if connecting_atom in oriented:
                locant = oriented.index(connecting_atom) + 1
                best_locant = min(best_locant, locant)

        return best_locant

    # Carbocyclic: for unsubstituted benzene, connection atom = position 1
    # For substituted, we need to number from connection point to give lowest
    # locants. Start with connection = 1.
    return 1


def _get_substituent_locant(
    mol, sub_atom: int, ring_atom: int, system_atoms: Set[int],
    connecting_atom: int
) -> int:
    """
    Determine the IUPAC locant for a substituent attached to a ring in an assembly.

    For carbocyclic rings, numbering starts from the connection point (locant 1)
    and follows the direction giving the lowest locants for substituents.

    For heterocyclic rings, standard IUPAC numbering is used.

    Args:
        mol: RDKit Mol object
        sub_atom: Atom index of the substituent attachment point (in ring)
        ring_atom: Same as sub_atom (where substituent connects to ring)
        system_atoms: Set of atom indices in the ring system
        connecting_atom: Atom index of the inter-ring connection in this system

    Returns:
        IUPAC locant (1-indexed)
    """
    ri = mol.GetRingInfo()
    atom_ring = None
    for ring in ri.AtomRings():
        if sub_atom in ring and set(ring) <= system_atoms:
            atom_ring = ring
            break

    if atom_ring is None:
        return 1

    ring_list = list(atom_ring)
    ring_size = len(ring_list)

    has_heteroatom = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_list
    )

    if has_heteroatom:
        from .heterocycles import orient_heterocycle
        _, atom_to_locant = orient_heterocycle(mol, ring_list)
        return atom_to_locant.get(sub_atom, 1)

    # Carbocyclic: number from the connection atom (= locant 1)
    # Find position of connection atom and sub_atom in the ring adjacency
    # Build adjacency for the ring
    ring_set = set(ring_list)
    adj = {idx: [] for idx in ring_list}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in ring_set and a2 in ring_set:
            adj[a1].append(a2)
            adj[a2].append(a1)

    # Walk from connection_atom in both directions
    def walk(start, direction_neighbor):
        """Walk ring from start through direction_neighbor, return ordered atoms."""
        visited = [start]
        current = direction_neighbor
        while current != start:
            visited.append(current)
            nexts = [n for n in adj[current] if n not in visited or n == start]
            if not nexts:
                break
            current = nexts[0]
        return visited

    neighbors_of_conn = adj.get(connecting_atom, [])
    if len(neighbors_of_conn) < 2:
        # Shouldn't happen for a ring atom, fallback
        if connecting_atom == sub_atom:
            return 1
        return 2

    path_a = walk(connecting_atom, neighbors_of_conn[0])
    path_b = walk(connecting_atom, neighbors_of_conn[1])

    # Determine locant of sub_atom in each direction
    locant_a = path_a.index(sub_atom) + 1 if sub_atom in path_a else ring_size
    locant_b = path_b.index(sub_atom) + 1 if sub_atom in path_b else ring_size

    # Return the lower locant (IUPAC lowest locant rule)
    return min(locant_a, locant_b)


def _get_substituent_info(
    mol, ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]]
) -> List[Dict]:
    """
    Find substituents on ring assembly systems and determine their locants.

    Args:
        mol: RDKit Mol object
        ring_systems: List of ring system atom sets
        connections: Inter-system bonds

    Returns:
        List of dicts with:
          - system_idx: which ring system the substituent is on
          - ring_atom: atom in ring where substituent attaches
          - sub_atoms: list of atom indices in substituent
          - name: substituent name (e.g., "chloro", "methyl")
          - locant: IUPAC locant on the ring
    """
    all_ring_atoms = set()
    for sys_atoms in ring_systems:
        all_ring_atoms.update(sys_atoms)

    # Connection atoms (not substituents)
    connection_atoms = set()
    for a1, a2, _, _ in connections:
        connection_atoms.add(a1)
        connection_atoms.add(a2)

    # Build atom -> system index
    atom_to_system = {}
    for sys_idx, sys_atoms in enumerate(ring_systems):
        for atom_idx in sys_atoms:
            atom_to_system[atom_idx] = sys_idx

    substituents = []
    visited_sub_atoms = set()

    for atom_idx in all_ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        sys_idx = atom_to_system[atom_idx]

        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in all_ring_atoms or n_idx in visited_sub_atoms:
                continue

            # This is a substituent atom
            # Walk the substituent to find all atoms
            sub_atoms = _walk_substituent(mol, n_idx, all_ring_atoms)
            visited_sub_atoms.update(sub_atoms)

            # Name the substituent
            sub_name = _name_substituent(mol, sub_atoms, n_idx)

            # Get the connection atom for this ring system
            conn_atom = None
            for a1, a2, s1, s2 in connections:
                if s1 == sys_idx:
                    conn_atom = a1
                    break
                elif s2 == sys_idx:
                    conn_atom = a2
                    break

            if conn_atom is None:
                conn_atom = atom_idx

            locant = _get_substituent_locant(
                mol, atom_idx, atom_idx, ring_systems[sys_idx], conn_atom
            )

            substituents.append({
                'system_idx': sys_idx,
                'ring_atom': atom_idx,
                'sub_atoms': sub_atoms,
                'name': sub_name,
                'locant': locant,
            })

    return substituents


def _walk_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """Walk from start_idx through non-ring atoms to collect substituent."""
    visited = set()
    stack = [start_idx]
    result = []

    while stack:
        current = stack.pop()
        if current in visited or current in ring_atoms:
            continue
        visited.add(current)
        result.append(current)

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited and n_idx not in ring_atoms:
                stack.append(n_idx)

    return result


def _name_substituent(mol, sub_atoms: List[int], attachment_atom: int) -> str:
    """
    Name a substituent attached to a ring assembly.

    Handles common cases: halogens, simple alkyl groups.

    Args:
        mol: RDKit Mol object
        sub_atoms: List of atom indices in substituent
        attachment_atom: First atom of the substituent (bonded to ring)

    Returns:
        Substituent name (prefix form, e.g., "chloro", "methyl")
    """
    from ..assembly.naming_utils import get_alkyl_name
    from ..data.chain_names import get_chain_prefix

    if len(sub_atoms) == 1:
        atom = mol.GetAtomWithIdx(sub_atoms[0])
        symbol = atom.GetSymbol()

        # Halogens
        halogen_names = {
            'F': 'fluoro',
            'Cl': 'chloro',
            'Br': 'bromo',
            'I': 'iodo',
        }
        if symbol in halogen_names:
            return halogen_names[symbol]

        # Single non-halogen atoms
        if symbol == 'O' and atom.GetTotalNumHs() >= 1:
            return 'hydroxy'
        if symbol == 'N' and atom.GetTotalNumHs() >= 2:
            return 'amino'

    # Check for alkyl groups (all carbon + hydrogen only)
    all_c_h = all(
        mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in sub_atoms
    )
    if all_c_h and sub_atoms:
        n_carbons = sum(
            1 for i in sub_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if n_carbons > 0:
            try:
                return get_alkyl_name(n_carbons)
            except (ValueError, KeyError):
                prefix = get_chain_prefix(n_carbons)
                return f"{prefix}yl"

    # Fallback: use fragment naming
    return "substituent"


def _format_prime(ring_index: int) -> str:
    """
    Format primed notation for a ring index.

    Ring 0 -> "" (unprimed)
    Ring 1 -> "'" (single prime)
    Ring 2 -> "''" (double prime)

    Uses ASCII apostrophe (U+0027) for OPSIN compatibility.
    """
    return "'" * ring_index


def name_ring_assembly(
    mol, assembly_info: Dict, features: Any
) -> Optional[str]:
    """
    Generate IUPAC name for a ring assembly.

    IUPAC P-28.2: Ring assemblies are named with multiplicative prefixes
    (bi-, ter-, quater-) before the parent ring name, with connection
    locants using primed notation.

    Args:
        mol: RDKit Mol object
        assembly_info: Dict from detect_ring_assembly() with keys:
            ring_systems, connections, count, ring_type
        features: MolecularFeatures object (for substituent context)

    Returns:
        Complete IUPAC name string, or None if naming fails.

    Examples:
        biphenyl -> "1,1'-biphenyl"
        2,2'-bipyridine -> "2,2'-bipyridine"
        4-chlorobiphenyl -> "4-chloro-1,1'-biphenyl"
    """
    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    count = assembly_info['count']

    # Get the multiplier prefix
    multiplier = ASSEMBLY_MULTIPLIERS.get(count)
    if multiplier is None:
        return None  # Unsupported assembly size

    # Determine parent ring name from the first system
    ring_name = _get_ring_parent_name(mol, ring_systems[0])
    if ring_name is None:
        return None

    # Build connection locant string
    # Sort connections by system indices to ensure consistent ordering
    # For bi- assemblies: one connection -> "X,X'"
    # For ter- assemblies: two connections -> "X,X':X',X''"
    connection_parts = []
    for a1, a2, s1, s2 in connections:
        # Ensure s1 < s2 for consistent ordering
        if s1 > s2:
            a1, a2, s1, s2 = a2, a1, s2, s1

        loc1 = _get_connection_locant(mol, a1, ring_systems[s1])
        loc2 = _get_connection_locant(mol, a2, ring_systems[s2])

        prime1 = _format_prime(s1)
        prime2 = _format_prime(s2)

        connection_parts.append(f"{loc1}{prime1},{loc2}{prime2}")

    connection_str = ":".join(connection_parts)

    # Build assembly base name
    base_name = f"{connection_str}-{multiplier}{ring_name}"

    # Find substituents
    substituent_list = _get_substituent_info(mol, ring_systems, connections)

    if not substituent_list:
        return base_name

    # Build substituent prefix
    # Group by name for multipliers
    from collections import Counter
    from ..assembly.naming_utils import (
        get_multiplier_prefix,
        alpha_sort_key,
    )

    # Sort substituents alphabetically by name
    substituent_list.sort(key=lambda s: alpha_sort_key(s['name']))

    # Group identical substituents
    sub_groups = {}
    for sub in substituent_list:
        name = sub['name']
        if name not in sub_groups:
            sub_groups[name] = []
        prime = _format_prime(sub['system_idx'])
        sub_groups[name].append(f"{sub['locant']}{prime}")

    # Build prefix parts
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locants = sub_groups[name]
        n = len(locants)
        locant_str = ",".join(str(l) for l in locants)
        if n > 1:
            mult = get_multiplier_prefix(n)
            prefix_parts.append(f"{locant_str}-{mult}{name}")
        else:
            prefix_parts.append(f"{locant_str}-{name}")

    sub_prefix = "-".join(prefix_parts)

    return f"{sub_prefix}-{base_name}"
