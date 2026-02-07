"""
Bridged polycyclic system detection and classification.

Handles classification and analysis of bridged polycyclic systems beyond simple bicyclics:
- Tricyclo (3 rings)
- Tetracyclo (4 rings)
- Pentacyclo+ (5+ rings)

IUPAC Reference: Blue Book 2013, P-23.3 (Tricyclic and polycyclic ring systems)

The von Baeyer system uses:
- Prefix: bicyclo-, tricyclo-, tetracyclo-, etc.
- Descriptor: [main branches.main bridge.secondary bridges^locants]
- Parent alkane name based on total carbons

Examples:
- Adamantane: tricyclo[3.3.1.1³⁷]decane
- Twistane: tricyclo[4.4.0.0³⁸]decane
"""

from typing import List, Optional, Set, Tuple, Dict, NamedTuple
from collections import deque
from rdkit import Chem

from ..perception.rings import get_ring_info, get_bridgehead_atoms


# ============================================================================
# Data Structures
# ============================================================================

class BridgeInfo(NamedTuple):
    """Information about a single bridge in a polycyclic system."""
    atoms: List[int]  # Atom indices in the bridge (excluding bridgeheads)
    length: int       # Number of atoms in bridge (between bridgeheads)
    start_bh: int     # Starting bridgehead atom index
    end_bh: int       # Ending bridgehead atom index


class PolycyclicInfo(NamedTuple):
    """Complete information about a polycyclic bridged system."""
    system_type: str           # 'bicyclo', 'tricyclo', 'tetracyclo', etc.
    ring_count: int            # Number of rings (cuts needed to open)
    bridgeheads: Set[int]      # All bridgehead atom indices
    main_ring: List[int]       # Atoms in main ring (for numbering)
    main_bridge: BridgeInfo    # Primary bridge
    secondary_bridges: List[BridgeInfo]  # Secondary bridges
    all_ring_atoms: Set[int]   # All atoms in the ring system


# ============================================================================
# Ring Count Calculation
# ============================================================================

def get_ring_count(mol) -> int:
    """
    Calculate the number of independent rings using the cycle rank formula.
    
    For a connected molecule: rings = bonds - atoms + 1
    This equals the number of cuts needed to convert to an acyclic structure.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Number of rings (cycle rank)
        
    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> get_ring_count(mol)
        2
        >>> mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        >>> get_ring_count(mol)
        3
    """
    # Get only ring atoms to focus on the cyclic system
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    
    if not ring_atoms:
        return 0
    
    # For the ring system, count bonds between ring atoms
    ring_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in ring_atoms and bond.GetEndAtomIdx() in ring_atoms:
            ring_bonds += 1
    
    # Cycle rank = edges - vertices + connected_components
    # For a single connected ring system, connected_components = 1
    return ring_bonds - len(ring_atoms) + 1


def count_cuts_to_open(mol) -> int:
    """
    Count how many bonds must be cut to make the ring system acyclic.
    
    This is equivalent to the ring count (cycle rank).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Number of cuts needed
    """
    return get_ring_count(mol)


# ============================================================================
# System Classification
# ============================================================================

def classify_bridged_system(mol) -> Optional[str]:
    """
    Classify a bridged polycyclic system by its ring count.
    
    Classification:
    - bicyclo: 2 rings
    - tricyclo: 3 rings
    - tetracyclo: 4 rings
    - pentacyclo: 5 rings
    - hexacyclo: 6 rings
    - heptacyclo: 7 rings
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Classification string, or None if not a bridged polycyclic
        
    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> classify_bridged_system(mol)
        'bicyclo'
        >>> mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        >>> classify_bridged_system(mol)
        'tricyclo'
    """
    ring_count = get_ring_count(mol)
    
    if ring_count < 2:
        return None
    
    # Check that this is bridged (has bridgehead atoms)
    bridgeheads = find_all_bridgeheads(mol)
    if len(bridgeheads) < 2:
        return None
    
    prefixes = {
        2: 'bicyclo',
        3: 'tricyclo',
        4: 'tetracyclo',
        5: 'pentacyclo',
        6: 'hexacyclo',
        7: 'heptacyclo',
        8: 'octacyclo',
        9: 'nonacyclo',
        10: 'decacyclo',
    }
    
    return prefixes.get(ring_count, f'{ring_count}cyclo')


def is_tricyclo_system(mol) -> bool:
    """
    Check if molecule is a tricyclo (3-ring bridged) system.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if molecule has exactly 3 rings in bridged configuration
        
    Examples:
        >>> mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        >>> is_tricyclo_system(mol)
        True
    """
    return classify_bridged_system(mol) == 'tricyclo'


def is_tetracyclo_system(mol) -> bool:
    """
    Check if molecule is a tetracyclo (4-ring bridged) system.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if molecule has exactly 4 rings in bridged configuration
    """
    return classify_bridged_system(mol) == 'tetracyclo'


def is_pentacyclo_or_higher(mol) -> bool:
    """
    Check if molecule is pentacyclo or higher (5+ rings).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        True if molecule has 5 or more rings
    """
    ring_count = get_ring_count(mol)
    return ring_count >= 5


# ============================================================================
# Bridgehead Detection
# ============================================================================

def find_all_bridgeheads(mol) -> Set[int]:
    """
    Find all bridgehead atoms in a polycyclic system.
    
    A bridgehead atom is:
    1. In 2 or more rings
    2. Has 3+ neighbors all within the ring system
    
    This extends the bicyclo bridgehead detection to handle
    systems with more than 2 bridgeheads.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Set of atom indices that are bridgeheads
        
    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> len(find_all_bridgeheads(mol))
        2
        >>> mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        >>> len(find_all_bridgeheads(mol))  
        4
    """
    ri = mol.GetRingInfo()
    
    # Count ring membership for each atom
    atom_ring_count: Dict[int, int] = {}
    for ring in ri.AtomRings():
        for idx in ring:
            atom_ring_count[idx] = atom_ring_count.get(idx, 0) + 1
    
    # Get all ring atoms
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    
    bridgeheads = set()
    
    for idx in range(mol.GetNumAtoms()):
        # Must be in multiple rings
        if atom_ring_count.get(idx, 0) < 2:
            continue
        
        # Must have 3+ neighbors
        atom = mol.GetAtomWithIdx(idx)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue
        
        # Count how many neighbors are in the ring system
        ring_neighbors = sum(1 for n in neighbors if n in ring_atoms)
        
        # Bridgeheads have 3+ ring neighbors
        if ring_neighbors >= 3:
            bridgeheads.add(idx)
    
    return bridgeheads


def get_ring_atoms(mol) -> Set[int]:
    """
    Get all atoms that are part of the ring system.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Set of atom indices in any ring
    """
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    return ring_atoms


# ============================================================================
# Bridge Path Finding
# ============================================================================

def find_all_bridge_paths(mol, bridgeheads: Set[int]) -> List[BridgeInfo]:
    """
    Find all bridges between any pair of bridgehead atoms.
    
    A bridge is a path between two bridgeheads that doesn't pass
    through any other bridgehead.
    
    Args:
        mol: RDKit Mol object
        bridgeheads: Set of bridgehead atom indices
        
    Returns:
        List of BridgeInfo objects for each bridge
    """
    ring_atoms = get_ring_atoms(mol)
    bridges: List[BridgeInfo] = []
    processed_pairs: Set[Tuple[int, int]] = set()
    
    for bh1 in bridgeheads:
        for bh2 in bridgeheads:
            if bh1 >= bh2:
                continue
            
            pair = (bh1, bh2)
            if pair in processed_pairs:
                continue
            
            # Find all paths between these two bridgeheads
            paths = _find_paths_between(mol, bh1, bh2, ring_atoms, bridgeheads)
            
            for path in paths:
                # Path includes both bridgeheads
                bridge_atoms = path[1:-1]  # Exclude bridgeheads
                bridge = BridgeInfo(
                    atoms=list(bridge_atoms),
                    length=len(bridge_atoms),
                    start_bh=bh1,
                    end_bh=bh2
                )
                bridges.append(bridge)
            
            processed_pairs.add(pair)
    
    return bridges


def _find_paths_between(
    mol,
    start: int,
    end: int,
    ring_atoms: Set[int],
    bridgeheads: Set[int]
) -> List[List[int]]:
    """
    Find all simple paths between two atoms that don't pass through other bridgeheads.
    
    Uses BFS to find all paths.
    """
    all_paths: List[List[int]] = []
    queue = deque([(start, [start])])
    
    while queue:
        current, path = queue.popleft()
        
        if current == end and len(path) > 1:
            all_paths.append(path)
            continue
        
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            
            # Skip if already in path (except target)
            if n_idx in path and n_idx != end:
                continue
            
            # Only follow ring atoms
            if n_idx not in ring_atoms:
                continue
            
            # Don't pass through other bridgeheads (except target)
            if n_idx in bridgeheads and n_idx != end and n_idx != start:
                continue
            
            queue.append((n_idx, path + [n_idx]))
    
    return all_paths


# ============================================================================
# Main Ring Identification
# ============================================================================

def find_main_ring(mol, bridgeheads: Set[int]) -> Optional[List[int]]:
    """
    Find the main ring for IUPAC numbering purposes.
    
    The main ring is defined as:
    1. The largest ring containing exactly 2 bridgeheads
    2. If tie, the ring with the most atoms
    
    Args:
        mol: RDKit Mol object
        bridgeheads: Set of bridgehead atom indices
        
    Returns:
        List of atom indices in the main ring, in order, or None
    """
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    
    candidates: List[Tuple[int, Tuple[int, ...]]] = []  # (size, ring)
    
    for ring in rings:
        ring_set = set(ring)
        bh_in_ring = ring_set & bridgeheads
        
        # Main ring should have exactly 2 bridgeheads
        if len(bh_in_ring) == 2:
            candidates.append((len(ring), ring))
    
    if not candidates:
        # Fallback: find largest ring with any bridgeheads
        for ring in rings:
            ring_set = set(ring)
            if ring_set & bridgeheads:
                candidates.append((len(ring), ring))
    
    if not candidates:
        return None
    
    # Sort by size (descending) and return largest
    candidates.sort(key=lambda x: -x[0])
    return list(candidates[0][1])


def identify_main_bridgeheads(mol, bridgeheads: Set[int]) -> Optional[Tuple[int, int]]:
    """
    Identify the two main bridgeheads for the primary bicyclic skeleton.
    
    These are the bridgeheads in the main ring that will be numbered 1 and n.
    
    Args:
        mol: RDKit Mol object
        bridgeheads: Set of all bridgehead atom indices
        
    Returns:
        Tuple of (primary_bh, secondary_bh) or None
    """
    main_ring = find_main_ring(mol, bridgeheads)
    if not main_ring:
        return None
    
    ring_set = set(main_ring)
    main_bhs = [bh for bh in bridgeheads if bh in ring_set]
    
    if len(main_bhs) < 2:
        return None
    
    # Return first two bridgeheads in the main ring
    # (Could be optimized with more sophisticated selection)
    return (main_bhs[0], main_bhs[1])


# ============================================================================
# Complete System Analysis
# ============================================================================

def analyze_polycyclic_system(mol) -> Optional[PolycyclicInfo]:
    """
    Perform complete analysis of a bridged polycyclic system.
    
    Returns all information needed for IUPAC naming:
    - System classification
    - Bridgehead positions
    - Main ring
    - All bridges with lengths
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        PolycyclicInfo namedtuple or None if not a valid polycyclic
        
    Examples:
        >>> mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        >>> info = analyze_polycyclic_system(mol)
        >>> info.system_type
        'tricyclo'
        >>> info.ring_count
        3
    """
    # Classify the system
    system_type = classify_bridged_system(mol)
    if not system_type:
        return None
    
    ring_count = get_ring_count(mol)
    bridgeheads = find_all_bridgeheads(mol)
    ring_atoms = get_ring_atoms(mol)
    
    if len(bridgeheads) < 2:
        return None
    
    # Find main ring
    main_ring = find_main_ring(mol, bridgeheads)
    if not main_ring:
        return None
    
    # Find all bridges
    all_bridges = find_all_bridge_paths(mol, bridgeheads)
    
    if not all_bridges:
        return None
    
    # Sort bridges by length (descending)
    all_bridges.sort(key=lambda b: -b.length)
    
    # First bridge is main bridge, rest are secondary
    main_bridge = all_bridges[0]
    secondary_bridges = all_bridges[1:] if len(all_bridges) > 1 else []
    
    return PolycyclicInfo(
        system_type=system_type,
        ring_count=ring_count,
        bridgeheads=bridgeheads,
        main_ring=main_ring,
        main_bridge=main_bridge,
        secondary_bridges=secondary_bridges,
        all_ring_atoms=ring_atoms
    )


# ============================================================================
# Heteroatom Detection
# ============================================================================

def get_ring_heteroatoms(mol, ring_atoms: Set[int]) -> List[Tuple[int, str]]:
    """
    Find heteroatoms (non-carbon) in the ring system.
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the ring system
        
    Returns:
        List of (atom_idx, element_symbol) for each heteroatom
    """
    heteroatoms = []
    
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))
    
    return heteroatoms


def get_heteroatom_prefix(symbol: str) -> str:
    """
    Get the IUPAC replacement prefix for a heteroatom.
    
    Args:
        symbol: Element symbol (O, N, S, etc.)
        
    Returns:
        IUPAC prefix (oxa, aza, thia, etc.)
    """
    prefixes = {
        'O': 'oxa',
        'N': 'aza',
        'S': 'thia',
        'Se': 'selena',
        'P': 'phospha',
        'Si': 'sila',
        'B': 'bora',
    }
    return prefixes.get(symbol, symbol.lower() + 'a')
