"""
Benzene naming rules according to IUPAC 2013 (Blue Book).

Handles:
- Monosubstituted benzenes (chlorobenzene, nitrobenzene)
- Polysubstituted benzenes with numeric locants (1,4-dimethylbenzene)
- Benzene ring orientation for lowest locants
- Alphabetical ordering of substituents

IUPAC 2013 PIN Rules:
- Numeric locants are REQUIRED (not ortho/meta/para)
- Toluene is retained ONLY for unsubstituted methylbenzene
- Substituted methylbenzene uses "methylbenzene" (not "toluene")
- Position 1 assigned to give lowest locants via first-point-of-difference
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from rdkit import Chem

from ..assembly.naming_utils import (
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
)


# Mapping from substituent atom symbol/pattern to prefix name
# Key: (symbol, hybridization/bond_info) or simple symbol
# Value: prefix name
SUBSTITUENT_PREFIXES = {
    # Halogens
    "F": "fluoro",
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo",
    # Common groups - these are detected by the get_benzene_substituents function
    # based on the substituent structure
}

# Alkyl group names (by carbon count)
ALKYL_NAMES = {
    1: "methyl",
    2: "ethyl",
    3: "propyl",
    4: "butyl",
    5: "pentyl",
    6: "hexyl",
    7: "heptyl",
    8: "octyl",
    9: "nonyl",
    10: "decyl",
}


def is_benzene_ring(mol, ring_atoms: Tuple[int, ...]) -> bool:
    """
    Check if a ring is a benzene ring (6-membered aromatic carbocycle).

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        True if ring is benzene (6 aromatic carbons)
    """
    if len(ring_atoms) != 6:
        return False

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False
        # Must be aromatic
        if not atom.GetIsAromatic():
            return False

    return True


def get_benzene_ring(mol) -> Optional[Tuple[int, ...]]:
    """
    Find the benzene ring in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices in the benzene ring, or None if not found
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if is_benzene_ring(mol, ring):
            return ring
    return None


def get_benzene_substituents(mol, ring_atoms: Tuple[int, ...]) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to each benzene carbon.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict mapping ring atom index to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)
    """
    ring_set = set(ring_atoms)
    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip ring atoms
            if nbr_idx in ring_set:
                continue

            # Identify the substituent
            sub_info = _identify_substituent(mol, nbr_idx, ring_set)
            if sub_info:
                substituents[ring_idx].append(sub_info)

    return dict(substituents)


def _identify_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent starting from an atom attached to the ring.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens - single atom substituents
    if symbol in SUBSTITUENT_PREFIXES:
        return {
            'name': SUBSTITUENT_PREFIXES[symbol],
            'atoms': [start_idx]
        }

    # Nitrogen-based groups
    if symbol == 'N':
        return _identify_nitrogen_group(mol, start_idx, ring_atoms)

    # Oxygen-based groups
    if symbol == 'O':
        return _identify_oxygen_group(mol, start_idx, ring_atoms)

    # Carbon-based groups (alkyl or functionalized chain)
    if symbol == 'C':
        # Check for nitrile C#N pattern FIRST (BUG-2 fix)
        nitrile_result = _identify_nitrile_group(mol, start_idx, ring_atoms)
        if nitrile_result:
            return nitrile_result

        # Try simple alkyl
        alkyl_result = _identify_alkyl_group(mol, start_idx, ring_atoms)
        if alkyl_result:
            return alkyl_result

        # Try functionalized chain (chains with FG like -CCCC(=O)O)
        func_chain = _identify_functionalized_chain(mol, start_idx, ring_atoms)
        if func_chain:
            return func_chain

    return None


def _identify_nitrile_group(mol, c_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify nitrile (C#N) substituent attached to benzene.

    The nitrile carbon is directly attached to the ring. We check if this carbon
    has a triple bond to nitrogen and no other heavy atom neighbors (besides the ring).

    Args:
        mol: RDKit Mol object
        c_idx: Index of the carbon atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name': 'nitrile', 'type': 'functional', 'atoms': [...] or None
    """
    c_atom = mol.GetAtomWithIdx(c_idx)

    # Check neighbors of the nitrile carbon
    for neighbor in c_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        # Check for triple bond to nitrogen
        bond = mol.GetBondBetweenAtoms(c_idx, neighbor.GetIdx())
        if (neighbor.GetSymbol() == 'N' and
                bond and bond.GetBondType() == Chem.BondType.TRIPLE):
            # Verify the nitrogen has no other heavy atom neighbors (just the C#N)
            n_atom = neighbor
            n_heavy_neighbors = [n for n in n_atom.GetNeighbors()
                                 if n.GetSymbol() != 'H' and n.GetIdx() != c_idx]
            if not n_heavy_neighbors:
                return {
                    'name': 'nitrile',
                    'type': 'functional',
                    'atoms': [c_idx, neighbor.GetIdx()]
                }

    return None


def _detect_benzene_nitrile(mol, ring_atoms: Tuple[int, ...]) -> Dict:
    """
    Detect if benzene ring has a nitrile substituent.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict with 'is_nitrile': bool, 'nitrile_positions': list of ring atom indices
        that have nitrile attached
    """
    ring_set = set(ring_atoms)
    nitrile_positions = []

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set:
                continue

            # Check if this neighbor is a nitrile carbon
            if neighbor.GetSymbol() == 'C':
                result = _identify_nitrile_group(mol, nbr_idx, ring_set)
                if result and result.get('name') == 'nitrile':
                    nitrile_positions.append(ring_idx)
                    break  # Only count once per ring position

    return {
        'is_nitrile': len(nitrile_positions) > 0,
        'nitrile_positions': nitrile_positions
    }


def _identify_nitrogen_group(mol, n_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent (amino, nitro, N-alkylamino, etc.)."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            # Nitro group
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    h_count = n_atom.GetTotalNumHs()

    # Simple amino (-NH2)
    if h_count == 2 and len(neighbors) == 0:
        return {'name': 'amino', 'atoms': [n_idx]}

    # N-monoalkyl amino (-NHR): 1 H, 1 carbon neighbor
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, neighbors[0].GetIdx(), ring_atoms | {n_idx})
        if alkyl_atoms is not None and carbon_count > 0:
            alkyl_name = ALKYL_NAMES.get(carbon_count)
            if alkyl_name:
                return {
                    'name': f'{alkyl_name}amino',
                    'atoms': [n_idx] + alkyl_atoms
                }

    # N,N-dialkyl amino (-NR2): 0 H, 2 carbon neighbors
    if h_count == 0 and len(neighbors) == 2:
        c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
        if len(c_neighbors) == 2:
            alkyl_names_list = []
            all_sub_atoms = [n_idx]
            for cn in c_neighbors:
                alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, cn.GetIdx(), ring_atoms | {n_idx})
                if alkyl_atoms is None or carbon_count == 0:
                    break
                aname = ALKYL_NAMES.get(carbon_count)
                if not aname:
                    break
                alkyl_names_list.append(aname)
                all_sub_atoms.extend(alkyl_atoms)
            else:
                # Both identified
                alkyl_names_list.sort()
                if alkyl_names_list[0] == alkyl_names_list[1]:
                    from ..assembly.naming_utils import get_multiplier_prefix
                    mp = get_multiplier_prefix(2, alkyl_names_list[0])
                    prefix_name = f'{mp}{alkyl_names_list[0]}amino'
                else:
                    prefix_name = f'{alkyl_names_list[0]}({alkyl_names_list[1]}amino)'
                return {
                    'name': prefix_name,
                    'atoms': all_sub_atoms
                }

    # Nitroso (-NO)
    if h_count == 0 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'O':
        bond = mol.GetBondBetweenAtoms(n_idx, neighbors[0].GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            return {
                'name': 'nitroso',
                'atoms': [n_idx, neighbors[0].GetIdx()]
            }

    return None


def _collect_pure_alkyl(mol, start_idx: int, excluded: Set[int]):
    """
    BFS to collect a pure alkyl group (only C/H atoms).

    Returns:
        Tuple of (list of atom indices, carbon_count) or (None, 0) if not pure alkyl.
    """
    visited = {start_idx}
    queue = [start_idx]
    all_atoms = []
    carbon_count = 0

    while queue:
        current_idx = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        elif current_atom.GetSymbol() != 'H':
            return None, 0  # Not pure alkyl

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms, carbon_count


def _identify_oxygen_group(mol, o_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent (hydroxy, methoxy, etc.)."""
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    # Methoxy (-OCH3)
    if len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        c_atom = neighbors[0]
        # Check if it's methyl (3 hydrogens, no other heavy atoms)
        if c_atom.GetTotalNumHs() == 3:
            c_neighbors = [n for n in c_atom.GetNeighbors() if n.GetIdx() != o_idx]
            if not c_neighbors:
                return {'name': 'methoxy', 'atoms': [o_idx, c_atom.GetIdx()]}

    return None


def _identify_alkyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent.

    Uses BFS to find all connected carbons and counts total carbons
    to determine alkyl name.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = [start_idx]
    carbon_count = 0
    all_atoms = []

    while queue:
        current_idx = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        else:
            # Non-carbon in the chain - this is not a simple alkyl
            # For Phase 2, we'll handle more complex substituents
            pass

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains heteroatom - not a simple alkyl
            # Check for vinyl (ethenyl) - C=C attached to ring
            if _is_vinyl_group(mol, start_idx, ring_atoms):
                return {'name': 'ethenyl', 'atoms': all_atoms}
            return None

    # Check for branched alkyl (isopropyl, tert-butyl, etc.)
    name = _get_alkyl_name(mol, start_idx, carbon_count, ring_atoms)
    if name:
        return {'name': name, 'atoms': all_atoms}

    return None


def _is_vinyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> bool:
    """Check if the group is a vinyl (ethenyl) group."""
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check for C=C where start is attached to ring
    for neighbor in start_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        bond = mol.GetBondBetweenAtoms(start_idx, neighbor.GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            if neighbor.GetSymbol() == 'C':
                return True

    return False


def _get_alkyl_name(mol, start_idx: int, carbon_count: int, ring_atoms: Set[int]) -> Optional[str]:
    """
    Get the name for an alkyl substituent.

    For now, handles simple linear alkyls. Branched alkyls will be
    implemented in later phases.
    """
    if carbon_count in ALKYL_NAMES:
        # TODO: Check for branching (isopropyl vs propyl, etc.)
        return ALKYL_NAMES[carbon_count]

    return None


def _identify_functionalized_chain(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a chain with functional group attached to benzene ring.

    This handles cases like `-CCCC(=O)O` (butanoic acid chain) that the
    simple alkyl detection misses because they contain heteroatoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the first atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' (a proper substituent name), 'atoms', 'chain_length',
        'functional_group', or None if not a functionalized chain or cannot be named.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = [start_idx]
    all_atoms = []
    carbon_count = 0
    has_heteroatom = False

    while queue:
        current_idx = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        symbol = current_atom.GetSymbol()
        if symbol == 'C':
            carbon_count += 1
        elif symbol != 'H':
            has_heteroatom = True

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Only consider if there's a heteroatom (indicating functional group)
    if not has_heteroatom:
        return None

    # Check for common functional groups
    functional_group = _detect_chain_functional_group(mol, all_atoms)
    if not functional_group:
        return None

    # Generate a proper substituent name based on FG type and chain length
    sub_name = _name_functionalized_chain_substituent(carbon_count, functional_group)
    if not sub_name:
        # Cannot produce a valid name; return None so caller can handle gracefully
        return None

    return {
        'name': sub_name,
        'atoms': all_atoms,
        'chain_length': carbon_count,
        'functional_group': functional_group
    }


# Mapping from chain length (carbons) to substituent prefix stem
_CHAIN_SUB_STEMS = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
}

# Mapping from functional group type to substituent prefix modifier
# These convert a chain with FG into a proper IUPAC prefix substituent name
_FG_SUB_PREFIX = {
    'carboxylic_acid': {
        # -C(=O)OH chain: named as "carboxylalkyl" (e.g., 2-carboxyethyl for -CH2CH2COOH)
        # or simply use the acyl prefix approach
        1: "carboxy",            # just -COOH
        2: "carboxymethyl",      # -CH2COOH
        3: "2-carboxyethyl",     # -CH2CH2COOH
        4: "3-carboxypropyl",    # -(CH2)3COOH
        5: "4-carboxybutyl",     # -(CH2)4COOH
    },
    'aldehyde': {
        1: "formyl",             # -CHO
        2: "oxoethyl",           # -CH2CHO (2-oxoethyl)
        3: "oxopropyl",          # -(CH2)2CHO
    },
    'alcohol': {
        1: "hydroxymethyl",      # -CH2OH
        2: "hydroxyethyl",       # -CH2CH2OH (2-hydroxyethyl)
        3: "hydroxypropyl",      # -(CH2)2CH2OH
    },
}


def _name_functionalized_chain_substituent(carbon_count: int, functional_group: str) -> Optional[str]:
    """
    Generate a proper IUPAC substituent name for a functionalized chain.

    Args:
        carbon_count: Number of carbon atoms in the chain
        functional_group: Type of functional group ('carboxylic_acid', 'aldehyde', 'alcohol')

    Returns:
        Substituent prefix name string, or None if cannot be named
    """
    # Try specific FG + chain length lookup
    fg_map = _FG_SUB_PREFIX.get(functional_group, {})
    if carbon_count in fg_map:
        return fg_map[carbon_count]

    # Fallback: generic naming based on FG type
    if functional_group == 'carboxylic_acid' and carbon_count > 0:
        if carbon_count == 1:
            return "carboxy"
        # For longer chains: (N-1)-carboxyalkyl
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count - 1)
        if alkyl:
            return f"carboxy{alkyl}"

    if functional_group == 'alcohol' and carbon_count > 0:
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            return f"hydroxy{alkyl}"

    if functional_group == 'aldehyde' and carbon_count > 0:
        if carbon_count == 1:
            return "formyl"
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            return f"oxo{alkyl}"

    # Cannot name this chain
    return None


def _detect_chain_functional_group(mol, chain_atoms: List[int]) -> Optional[str]:
    """
    Detect what functional group is on a chain.

    Checks for carboxylic acid, alcohol, and aldehyde patterns.

    Args:
        mol: RDKit Mol object
        chain_atoms: List of atom indices in the chain

    Returns:
        Functional group name ('carboxylic_acid', 'alcohol', 'aldehyde') or None
    """
    chain_set = set(chain_atoms)

    # Check for carboxylic acid pattern: C(=O)O with O having H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue

        neighbors = list(atom.GetNeighbors())
        o_double = None
        o_single = None

        for nbr in neighbors:
            if nbr.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    o_double = nbr
                elif bond.GetBondType() == Chem.BondType.SINGLE:
                    if nbr.GetTotalNumHs() >= 1:  # -OH
                        o_single = nbr

        if o_double and o_single:
            return 'carboxylic_acid'

    # Check for aldehyde pattern: C(=O)H (must check before alcohol)
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() >= 1:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        return 'aldehyde'

    # Check for alcohol pattern: C-O-H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() >= 1:
            # Check it's not part of carboxylic acid (already checked above)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    c_atom = nbr
                    has_double_o = False
                    for c_nbr in c_atom.GetNeighbors():
                        if c_nbr.GetSymbol() == 'O' and c_nbr.GetIdx() != atom.GetIdx():
                            bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), c_nbr.GetIdx())
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                has_double_o = True
                    if not has_double_o:
                        return 'alcohol'

    return None


def orient_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Orient benzene ring to give lowest locants to substituents.

    IUPAC 2013 Rules:
    1. For monosubstituted: substituent position is 1
    2. For polysubstituted: use first-point-of-difference for locants
    3. For identical substituents: minimize locant set
    4. When locant sets are equal: position 1 goes to alphabetically first substituent

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring (ordered)
        substituents: Dict from get_benzene_substituents

    Returns:
        List of ring atom indices reordered so position 1 is first
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)  # Should be 6

    # Get substituted positions (indices in ring_list)
    substituted_indices = []
    for i, atom_idx in enumerate(ring_list):
        if atom_idx in substituents:
            substituted_indices.append(i)

    if not substituted_indices:
        # Unsubstituted benzene - any orientation is fine
        return ring_list

    if len(substituted_indices) == 1:
        # Monosubstituted - that position becomes 1
        start_pos = substituted_indices[0]
        return _rotate_list(ring_list, start_pos)

    # Polysubstituted - try all starting positions and both directions
    # Collect all candidates with their locant sets and alphabetic scores
    candidates = []

    for start_pos in range(n):
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            # Build oriented ring
            oriented = _build_oriented_ring(ring_list, start_pos, direction)

            # Calculate locants for this orientation
            locants = _calculate_locants(oriented, substituents)

            # Get the substituent at position 1 for alphabetical tie-breaking
            pos1_atom = oriented[0]
            pos1_sub_name = None
            if pos1_atom in substituents and substituents[pos1_atom]:
                pos1_sub_name = substituents[pos1_atom][0]['name']

            candidates.append((oriented, locants, pos1_sub_name))

    # Find the best locant set
    best_locants = None
    for _, locants, _ in candidates:
        if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
            best_locants = locants

    # Filter to only candidates with the best locant set
    best_candidates = [
        (oriented, pos1_sub) for oriented, locants, pos1_sub in candidates
        if locants == best_locants
    ]

    # If multiple candidates with same locant set, pick one where alphabetically
    # first substituent is at position 1
    if len(best_candidates) == 1:
        return best_candidates[0][0]

    # Sort by alphabetical order of position 1 substituent
    # None should sort last (no substituent at position 1)
    def sort_key(item):
        oriented, pos1_sub = item
        if pos1_sub is None:
            return 'zzzzz'  # Sort last
        return alpha_sort_key(pos1_sub)

    best_candidates.sort(key=sort_key)
    return best_candidates[0][0]


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


def _calculate_locants(
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Calculate locant set for a given orientation.

    Args:
        oriented_ring: Ring atoms in order (position 0 = locant 1)
        substituents: Dict from get_benzene_substituents

    Returns:
        Sorted list of locants (1-indexed positions)
    """
    locants = []
    for i, atom_idx in enumerate(oriented_ring):
        if atom_idx in substituents:
            locants.append(i + 1)  # Locants are 1-indexed

    return sorted(locants)


def _compare_locant_sets(a: List[int], b: List[int]) -> int:
    """
    Compare two locant sets using first-point-of-difference.

    Returns:
        < 0 if a is preferred (lower)
        > 0 if b is preferred
        0 if equal
    """
    for i in range(max(len(a), len(b))):
        val_a = a[i] if i < len(a) else float('inf')
        val_b = b[i] if i < len(b) else float('inf')

        if val_a < val_b:
            return -1
        if val_a > val_b:
            return 1

    return 0


def name_substituted_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> str:
    """
    Generate systematic name for substituted benzene.

    IUPAC 2013 PIN Rules:
    - Use numeric locants (not ortho/meta/para) for polysubstituted
    - Monosubstituted benzenes do NOT include locant (it's always 1)
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-benzene (or just substituent-benzene for mono)
    - Special case: benzonitrile (C6H5CN) uses suffix-style naming per P-66.1.1.1

    Args:
        mol: RDKit Mol object
        ring_atoms: Original ring atom tuple
        oriented_ring: Oriented ring from orient_benzene
        substituents: Dict from get_benzene_substituents

    Returns:
        IUPAC name string (e.g., "chlorobenzene" or "1,4-dimethylbenzene")
    """
    # Build locant-to-substituent mapping
    # oriented_ring[0] = position 1, oriented_ring[1] = position 2, etc.
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Group substituents by name
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for atom_idx in oriented_ring:
        if atom_idx not in substituents:
            continue

        for sub_info in substituents[atom_idx]:
            name = sub_info['name']
            locant = atom_to_locant[atom_idx]
            substituent_groups[name].append(locant)

    # Sort locants within each group
    for name in substituent_groups:
        substituent_groups[name].sort()

    # Check for nitrile - special handling for benzonitrile naming (BUG-2 fix)
    # IUPAC 2013 PIN: benzonitrile (not cyanobenzene) per P-66.1.1.1
    if 'nitrile' in substituent_groups:
        nitrile_locants = substituent_groups['nitrile']
        if len(nitrile_locants) == 1:
            # Single nitrile: use benzonitrile as parent
            # Remove nitrile from substituent groups since it becomes the parent
            del substituent_groups['nitrile']

            # Re-orient so nitrile is at position 1 for locant calculation
            nitrile_locant = nitrile_locants[0]

            # Other substituents become prefixes relative to benzonitrile
            if not substituent_groups:
                # Pure benzonitrile
                return "benzonitrile"

            # Build prefixes for other substituents
            # Need to recalculate locants relative to nitrile at position 1
            return _name_substituted_benzonitrile(
                substituent_groups, nitrile_locant, atom_to_locant, oriented_ring
            )

    # Count total number of substituents
    total_substituents = sum(len(locs) for locs in substituent_groups.values())

    # For monosubstituted benzenes, omit the locant (it's always 1)
    is_monosubstituted = total_substituents == 1

    # Build prefix strings, sorted alphabetically by substituent name
    prefixes = []
    for name in sorted(substituent_groups.keys(), key=alpha_sort_key):
        locants = substituent_groups[name]
        count = len(locants)

        if is_monosubstituted:
            # Monosubstituted: just "chloro", "methyl", etc. - no locant
            prefix_str = name
        else:
            # Polysubstituted: include locants
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_benzene_prefixes(prefixes)

    # Build final name
    return f"{prefix_part}benzene"


def _name_substituted_benzonitrile(
    substituent_groups: Dict[str, List[int]],
    nitrile_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int]
) -> str:
    """
    Name a substituted benzonitrile.

    IUPAC 2013: substituents are numbered relative to the nitrile position (position 1).
    Example: 4-chlorobenzonitrile, 4-methylbenzonitrile

    The nitrile carbon position becomes position 1 in the benzonitrile numbering.
    For a 6-membered ring with nitrile at old position N:
    - Position N becomes 1
    - Other positions are renumbered going clockwise (or counterclockwise for lowest locants)

    Args:
        substituent_groups: Dict of substituent name -> list of locants (in original numbering)
        nitrile_locant: The locant of the nitrile in the original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name string (e.g., "4-chlorobenzonitrile")
    """
    # Try both directions (clockwise and counterclockwise) and pick lowest locants
    best_groups = None
    best_locant_set = None

    for direction in [1, -1]:
        converted_groups: Dict[str, List[int]] = defaultdict(list)

        for name, locants in substituent_groups.items():
            for old_loc in locants:
                # Calculate new position relative to nitrile at position 1
                # direction = 1: clockwise numbering from nitrile
                # direction = -1: counterclockwise numbering from nitrile
                # Formula: new_pos = ((old_pos - nitrile_pos) * direction % 6) + 1
                # This ensures nitrile_pos -> 1, and other positions follow in order
                diff = (old_loc - nitrile_locant) * direction
                new_loc = (diff % 6) + 1
                if new_loc == 1:
                    # Position 1 is reserved for nitrile; this shouldn't happen
                    # for other substituents, but handle edge case
                    new_loc = 7 - new_loc  # Map to position 6 (opposite direction)
                converted_groups[name].append(new_loc)

        # Sort locants within each group
        for name in converted_groups:
            converted_groups[name].sort()

        # Calculate locant set for comparison
        all_locants = sorted([loc for locs in converted_groups.values() for loc in locs])

        if best_locant_set is None or all_locants < best_locant_set:
            best_locant_set = all_locants
            best_groups = dict(converted_groups)

    # Build prefix strings
    total_substituents = sum(len(locs) for locs in best_groups.values())
    is_monosubstituted = total_substituents == 1

    prefixes = []
    for name in sorted(best_groups.keys(), key=alpha_sort_key):
        locants = best_groups[name]
        count = len(locants)

        if is_monosubstituted:
            # Single other substituent: include locant (e.g., "4-chloro")
            prefix_str = f"{locants[0]}-{name}"
        else:
            # Multiple substituents
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    # Join prefixes
    prefix_part = _join_benzene_prefixes(prefixes)

    return f"{prefix_part}benzonitrile"


def _join_benzene_prefixes(prefixes: List[str]) -> str:
    """
    Join benzene substituent prefixes with proper hyphenation.

    When one prefix ends with a letter and the next starts with a digit,
    a hyphen is needed.

    Args:
        prefixes: List of formatted prefix strings

    Returns:
        Joined prefix string
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return prefixes[0]

    result = prefixes[0]
    for i in range(1, len(prefixes)):
        current = prefixes[i]

        # Check if we need a hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]

            if last_char.isalpha() and first_char.isdigit():
                result += "-"

        result += current

    return result


def name_benzene_derivative(mol) -> Optional[str]:
    """
    Generate name for a benzene derivative.

    This is the main entry point for benzene naming.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not a benzene derivative
    """
    # Find benzene ring
    ring_atoms = get_benzene_ring(mol)
    if ring_atoms is None:
        return None

    # Get substituents
    substituents = get_benzene_substituents(mol, ring_atoms)

    if not substituents:
        # Unsubstituted benzene - should be caught by retained names
        return "benzene"

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Generate systematic name
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)
