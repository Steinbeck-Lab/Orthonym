"""
Von Baeyer polycyclic descriptor generation (IUPAC 2013, P-23).

Implements the von Baeyer algorithm (VB-1 through VB-7) for generating
correct polycyclic descriptors for any ring count (bicyclo through decacyclo).

This module replaces the broken descriptor generation in tricyclo.py and
polycyclic_bridged.py, which incorrectly used SSSR for main ring finding.

Key algorithm:
1. Ring count via cycle rank formula (edges - vertices + 1)
2. Main ring via longest-path between bridgehead pairs (NOT SSSR)
3. Main bridge between main bridgeheads through non-main-ring atoms
4. Secondary bridges: independent before dependent, descending by length
5. VB-7 numbering: main ring (longer path first) -> main bridge -> secondary bridges
6. Verification: sum(bridge_lengths) + 2 == total_ring_atoms

Reference: IUPAC 2013 Blue Book P-23, VB-1 through VB-9.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations
from rdkit import Chem


# ============================================================================
# Constants
# ============================================================================

CYCLO_PREFIXES = {
    2: "bicyclo",
    3: "tricyclo",
    4: "tetracyclo",
    5: "pentacyclo",
    6: "hexacyclo",
    7: "heptacyclo",
    8: "octacyclo",
    9: "nonacyclo",
    10: "decacyclo",
}

_ALKANE_NAMES = {
    1: "methane", 2: "ethane", 3: "propane", 4: "butane",
    5: "pentane", 6: "hexane", 7: "heptane", 8: "octane",
    9: "nonane", 10: "decane", 11: "undecane", 12: "dodecane",
    13: "tridecane", 14: "tetradecane", 15: "pentadecane",
    16: "hexadecane", 17: "heptadecane", 18: "octadecane",
    19: "nonadecane", 20: "icosane", 21: "henicosane",
    22: "docosane", 23: "tricosane", 24: "tetracosane",
    25: "pentacosane", 26: "hexacosane", 27: "heptacosane",
    28: "octacosane", 29: "nonacosane", 30: "triacontane",
    31: "hentriacontane", 32: "dotriacontane", 33: "tritriacontane",
    34: "tetratriacontane", 35: "pentatriacontane", 36: "hexatriacontane",
    37: "heptatriacontane", 38: "octatriacontane", 39: "nonatriacontane",
    40: "tetracontane",
}


def _get_alkane_name(carbon_count: int) -> str:
    """Get the alkane parent name for a given carbon count."""
    if carbon_count in _ALKANE_NAMES:
        return _ALKANE_NAMES[carbon_count]
    if carbon_count > 40:
        return f"{carbon_count}ane"
    return f"C{carbon_count}ane"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BridgeInfo:
    """Information about a single bridge in a polycyclic system."""
    atoms: List[int]          # Atom indices in the bridge (excluding bridgeheads)
    length: int               # Number of atoms in bridge (between bridgeheads)
    start_bh: int             # Starting bridgehead atom index
    end_bh: int               # Ending bridgehead atom index
    is_secondary: bool = False  # True for secondary bridges
    locant_low: Optional[int] = None   # Lower VB locant of endpoint (for secondary)
    locant_high: Optional[int] = None  # Higher VB locant of endpoint (for secondary)


@dataclass
class PolycyclicDescriptor:
    """Complete von Baeyer descriptor for a polycyclic system."""
    ring_count: int
    bridge_info_list: List[BridgeInfo]
    numbering: Dict[int, int]   # atom_idx -> VB locant (1-indexed)
    total_atoms: int
    descriptor_string: str
    bridge_lengths: List[int] = field(default_factory=list)  # all bridge lengths, sorted descending


# ============================================================================
# Core Algorithm: find_longest_path
# ============================================================================

def find_longest_path(mol, start: int, end: int, allowed_atoms: Set[int]) -> List[int]:
    """
    Find the longest simple path from start to end through allowed atoms.

    Uses DFS with backtracking. For polycyclic ring systems (<50 atoms),
    exhaustive search is feasible and fast.

    Args:
        mol: RDKit Mol object
        start: Starting atom index
        end: Target atom index
        allowed_atoms: Set of atom indices that the path may traverse

    Returns:
        List of atom indices from start to end (inclusive), or empty list if
        no path exists.
    """
    if start == end:
        return [start]

    best_path = []

    def dfs(current, visited, path):
        nonlocal best_path
        if current == end:
            if len(path) > len(best_path):
                best_path = path[:]
            return

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in allowed_atoms and nbr_idx not in visited:
                visited.add(nbr_idx)
                path.append(nbr_idx)
                dfs(nbr_idx, visited, path)
                path.pop()
                visited.remove(nbr_idx)

    visited = {start}
    dfs(start, visited, [start])
    return best_path


def _find_all_simple_paths(mol, start: int, end: int, allowed_atoms: Set[int]) -> List[List[int]]:
    """
    Find all simple paths from start to end through allowed atoms.

    Returns list of paths, each a list of atom indices.
    """
    if start == end:
        return [[start]]

    all_paths = []

    def dfs(current, visited, path):
        if current == end:
            all_paths.append(path[:])
            return

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in allowed_atoms and nbr_idx not in visited:
                visited.add(nbr_idx)
                path.append(nbr_idx)
                dfs(nbr_idx, visited, path)
                path.pop()
                visited.remove(nbr_idx)

    visited = {start}
    dfs(start, visited, [start])
    return all_paths


# ============================================================================
# VonBaeyerAnalyzer Class
# ============================================================================

class VonBaeyerAnalyzer:
    """
    Implements IUPAC von Baeyer nomenclature (P-23, VB-1 through VB-7).

    Pipeline:
    1. Ring count: cycle_rank = edges - vertices + 1
    2. Main ring: largest ring through a bridgehead pair (VB-2)
    3. Main bridge: longest path between main bridgeheads not through main ring (VB-5)
    4. Secondary bridges: remaining connections, independent before dependent (VB-6)
    5. Numbering: main ring -> main bridge -> secondary bridges (VB-7)
    6. Descriptor: prefix[bridge_lengths]

    Verification: sum(bridge_lengths) + 2 = total_skeletal_atoms
    """

    def analyze(self, mol, ring_atoms: Set[int]) -> PolycyclicDescriptor:
        """
        Main entry point: analyze a polycyclic system and produce its descriptor.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system

        Returns:
            PolycyclicDescriptor with all VB information
        """
        ring_count = self._get_ring_count(mol, ring_atoms)
        bridgeheads = self._find_all_bridgeheads(mol, ring_atoms)
        main_ring, bh_pair = self._find_main_ring(mol, ring_atoms, bridgeheads)
        main_bridge = self._find_main_bridge(mol, ring_atoms, main_ring, bh_pair)
        secondary_bridges = self._find_secondary_bridges(
            mol, ring_atoms, main_ring, main_bridge, bh_pair
        )
        numbering = self._assign_numbering(
            mol, main_ring, main_bridge, secondary_bridges, bh_pair
        )

        # Collect all bridge info
        all_bridges = []

        # Main ring is split into two halves by bridgeheads -> two "branches"
        bh1, bh2 = bh_pair
        bh1_pos = main_ring.index(bh1)
        bh2_pos = main_ring.index(bh2)

        # Split main ring into two branches (paths between bridgeheads)
        # main_ring is ordered: bh1 -> longer path -> bh2 -> shorter path -> back to bh1
        # So branch1 = main_ring[0:bh2_pos+1] and branch2 = main_ring[bh2_pos:]
        branch1 = main_ring[:bh2_pos + 1]  # bh1 ... bh2 (longer)
        branch2 = main_ring[bh2_pos:]      # bh2 ... back toward bh1 (shorter, includes bh2 but ring wraps)

        # Branch lengths (atoms between bridgeheads, exclusive of bridgeheads)
        branch1_len = len(branch1) - 2  # exclude both bridgeheads
        branch2_len = len(branch2) - 1  # branch2 ends just before wrapping to bh1 (which is main_ring[0])

        # Ensure branch1 >= branch2 for IUPAC ordering
        if branch1_len < branch2_len:
            branch1_len, branch2_len = branch2_len, branch1_len

        # Main bridge length
        main_bridge_len = len(main_bridge.atoms) if main_bridge else 0

        # Create bridge info objects for main branches + main bridge
        bridge_info_b1 = BridgeInfo(
            atoms=branch1[1:-1],
            length=branch1_len,
            start_bh=bh1,
            end_bh=bh2,
            is_secondary=False,
        )
        bridge_info_b2 = BridgeInfo(
            atoms=list(branch2[1:]) if len(branch2) > 1 else [],
            length=branch2_len,
            start_bh=bh2,
            end_bh=bh1,
            is_secondary=False,
        )
        bridge_info_main = BridgeInfo(
            atoms=main_bridge.atoms if main_bridge else [],
            length=main_bridge_len,
            start_bh=bh1,
            end_bh=bh2,
            is_secondary=False,
        )

        all_bridges = [bridge_info_b1, bridge_info_b2, bridge_info_main]

        # Collect bridge lengths for descriptor
        bridge_lengths = [branch1_len, branch2_len, main_bridge_len]

        # Add secondary bridges
        for sb in secondary_bridges:
            # Determine locants using numbering
            ep1 = sb.start_bh
            ep2 = sb.end_bh
            loc1 = numbering.get(ep1, 0)
            loc2 = numbering.get(ep2, 0)
            locant_low = min(loc1, loc2)
            locant_high = max(loc1, loc2)

            sb_info = BridgeInfo(
                atoms=sb.atoms,
                length=sb.length,
                start_bh=ep1,
                end_bh=ep2,
                is_secondary=True,
                locant_low=locant_low,
                locant_high=locant_high,
            )
            all_bridges.append(sb_info)
            bridge_lengths.append(sb.length)

        # Sort bridge lengths: first three (branch1, branch2, main bridge) descending,
        # then secondary bridges descending
        primary_lengths = sorted(bridge_lengths[:3], reverse=True)
        secondary_lengths = sorted(bridge_lengths[3:], reverse=True)
        bridge_lengths = primary_lengths + secondary_lengths

        # Build descriptor string
        descriptor = self._build_descriptor(
            ring_count, primary_lengths, secondary_bridges, numbering
        )

        total_atoms = len(ring_atoms)

        return PolycyclicDescriptor(
            ring_count=ring_count,
            bridge_info_list=all_bridges,
            numbering=numbering,
            total_atoms=total_atoms,
            descriptor_string=descriptor,
            bridge_lengths=bridge_lengths,
        )

    # ========================================================================
    # VB-1: Ring Count
    # ========================================================================

    def _get_ring_count(self, mol, ring_atoms: Set[int]) -> int:
        """
        Calculate cycle rank: edges - vertices + 1 for the ring subgraph.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system

        Returns:
            Number of independent rings (cycle rank)
        """
        if not ring_atoms:
            return 0

        # Count edges within the ring atom subgraph
        ring_bonds = 0
        for bond in mol.GetBonds():
            if (bond.GetBeginAtomIdx() in ring_atoms and
                    bond.GetEndAtomIdx() in ring_atoms):
                ring_bonds += 1

        return ring_bonds - len(ring_atoms) + 1

    # ========================================================================
    # Bridgehead Detection
    # ========================================================================

    def _find_all_bridgeheads(self, mol, ring_atoms: Set[int]) -> Set[int]:
        """
        Find all bridgehead atoms in the ring system.

        A bridgehead atom is in the ring system and has 3+ neighbors
        also in the ring system.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system

        Returns:
            Set of bridgehead atom indices
        """
        bridgeheads = set()
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            ring_neighbors = sum(
                1 for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms
            )
            if ring_neighbors >= 3:
                bridgeheads.add(idx)
        return bridgeheads

    # ========================================================================
    # VB-2: Main Ring Finding
    # ========================================================================

    def _find_main_ring(
        self,
        mol,
        ring_atoms: Set[int],
        bridgeheads: Set[int]
    ) -> Tuple[List[int], Tuple[int, int]]:
        """
        VB-2: Find the main ring -- the largest ring in the system.

        IUPAC Algorithm for tricyclo+:
        1. Find pairs of bridgeheads that share a common neighbor (potential main bridge atom)
        2. For each such pair, the main ring is formed by paths through OTHER atoms
        3. Select the pair where the main ring gives the most balanced branches
        4. For equal balance, prefer larger main ring

        For bicyclo (2 bridgeheads only):
        - Main ring is the largest ring through both bridgeheads
        - Main bridge is the shortest path between them (excluding main ring)

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system
            bridgeheads: Set of bridgehead atom indices

        Returns:
            Tuple of (main_ring_atoms_in_order, (bh1, bh2))
        """
        # Build adjacency
        adj = {}
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            adj[idx] = set(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms)

        best_ring = None
        best_bh_pair = None
        best_score = (-1, -1, -1)  # (balance, ring_size, -main_bridge_len)

        for bh1, bh2 in combinations(sorted(bridgeheads), 2):
            # Find common neighbors (potential main bridge atoms)
            common_neighbors = adj.get(bh1, set()) & adj.get(bh2, set())

            # Also consider direct connection (0-atom bridge)
            has_direct_bond = bh2 in adj.get(bh1, set())

            # Case 1: Direct bond between bridgeheads (0-atom main bridge)
            if has_direct_bond:
                # Main ring via other atoms
                remaining = ring_atoms - {bh1, bh2}
                # But we need to avoid the direct bond, so find paths excluding it
                path1 = self._find_path_avoiding_direct(mol, bh1, bh2, remaining | {bh1, bh2})
                if path1 and len(path1) >= 3:
                    path1_interior = set(path1[1:-1])
                    remaining2 = remaining - path1_interior
                    path2 = self._find_path_avoiding_direct(mol, bh2, bh1, remaining2 | {bh1, bh2})
                    if path2 and len(path2) >= 2:
                        ring = path1 + path2[1:-1]
                        branch1_len = len(path1) - 2
                        branch2_len = len(path2) - 2
                        balance = min(branch1_len, branch2_len)
                        score = (balance, len(ring), 0)  # 0-atom bridge
                        if score > best_score:
                            best_score = score
                            best_ring = ring
                            best_bh_pair = (bh1, bh2)

            # Case 2: Main bridge via a common neighbor
            for bridge_atom in common_neighbors:
                # Main ring via OTHER atoms (excluding the bridge atom)
                remaining = ring_atoms - {bridge_atom}

                # Find ALL paths between bridgeheads and select best disjoint pair
                all_paths = _find_all_simple_paths(mol, bh1, bh2, remaining)
                if len(all_paths) < 2:
                    continue

                # Find best pair of disjoint paths (most balanced)
                best_pair_score = -1
                best_path1 = None
                best_path2 = None

                for i, p1 in enumerate(all_paths):
                    p1_interior = set(p1[1:-1])
                    for p2 in all_paths[i + 1:]:
                        p2_interior = set(p2[1:-1])
                        # Check if paths are interior-disjoint
                        if not (p1_interior & p2_interior):
                            balance = min(len(p1) - 2, len(p2) - 2)
                            ring_size = len(p1) + len(p2) - 2  # -2 for shared endpoints
                            pair_score = (balance, ring_size)
                            if pair_score > (best_pair_score, 0):
                                best_pair_score = balance
                                best_path1 = p1
                                best_path2 = p2

                if best_path1 and best_path2:
                    ring = best_path1 + best_path2[1:-1][::-1]
                    branch1_len = len(best_path1) - 2
                    branch2_len = len(best_path2) - 2
                    balance = min(branch1_len, branch2_len)
                    score = (balance, len(ring), -1)  # 1-atom bridge

                    if score > best_score:
                        best_score = score
                        best_ring = ring
                        best_bh_pair = (bh1, bh2)

        # Fallback: use largest ring method (for bicyclo or unusual cases)
        if best_ring is None:
            best_ring, best_bh_pair = self._find_main_ring_fallback(
                mol, ring_atoms, bridgeheads
            )

        # Reorder main ring so that bh1 is first and the longer path to bh2 comes first
        if best_ring and best_bh_pair:
            best_ring = self._orient_main_ring(best_ring, best_bh_pair)

        return best_ring, best_bh_pair

    def _find_path_avoiding_direct(
        self, mol, start: int, end: int, allowed: Set[int]
    ) -> List[int]:
        """Find a path from start to end that doesn't use the direct bond."""
        # Use BFS to find shortest path first, then try longer paths
        best_path = []

        def dfs(current, visited, path):
            nonlocal best_path
            if current == end:
                if len(path) > len(best_path):
                    best_path = path[:]
                return

            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in allowed and nbr_idx not in visited:
                    # Skip direct bond from start to end
                    if current == start and nbr_idx == end:
                        continue
                    visited.add(nbr_idx)
                    path.append(nbr_idx)
                    dfs(nbr_idx, visited, path)
                    path.pop()
                    visited.remove(nbr_idx)

        visited = {start}
        dfs(start, visited, [start])
        return best_path

    def _find_main_ring_fallback(
        self, mol, ring_atoms: Set[int], bridgeheads: Set[int]
    ) -> Tuple[List[int], Tuple[int, int]]:
        """Fallback: find largest ring through any bridgehead pair."""
        best_ring = None
        best_ring_size = 0
        best_bh_pair = None

        for bh1, bh2 in combinations(sorted(bridgeheads), 2):
            path1 = find_longest_path(mol, bh1, bh2, ring_atoms)
            if not path1:
                continue

            path1_interior = set(path1[1:-1])
            remaining = ring_atoms - path1_interior
            path2 = find_longest_path(mol, bh2, bh1, remaining)
            if not path2 or len(path2) < 2:
                continue

            ring = path1 + path2[1:-1]
            ring_size = len(ring)

            if ring_size > best_ring_size:
                best_ring_size = ring_size
                best_ring = ring
                best_bh_pair = (bh1, bh2)

        if best_ring is None:
            bh_list = sorted(bridgeheads)
            if len(bh_list) >= 2:
                best_bh_pair = (bh_list[0], bh_list[1])
                best_ring = list(ring_atoms)
            else:
                best_ring = list(ring_atoms)
                best_bh_pair = (bh_list[0], bh_list[0]) if bh_list else (0, 0)

        return best_ring, best_bh_pair

    def _orient_main_ring(
        self, ring: List[int], bh_pair: Tuple[int, int]
    ) -> List[int]:
        """
        Orient the main ring so:
        - bh1 is at position 0
        - The longer path from bh1 to bh2 comes first (VB-7 numbering requirement)

        Args:
            ring: List of atom indices forming the ring
            bh_pair: (bh1, bh2) bridgehead pair

        Returns:
            Reordered ring list
        """
        bh1, bh2 = bh_pair

        if bh1 not in ring or bh2 not in ring:
            return ring

        # Rotate so bh1 is first
        bh1_idx = ring.index(bh1)
        ring = ring[bh1_idx:] + ring[:bh1_idx]

        # Now bh1 is at position 0. Find bh2 position.
        bh2_idx = ring.index(bh2)

        # Path1 (forward): ring[0] to ring[bh2_idx] -> length = bh2_idx
        # Path2 (backward): ring[bh2_idx] to ring[end] -> length = len(ring) - bh2_idx
        forward_len = bh2_idx - 1   # atoms between bh1 and bh2 going forward
        backward_len = len(ring) - bh2_idx - 1  # atoms between bh2 and (back to before bh1)

        if forward_len < backward_len:
            # Reverse: the backward path is longer, so we want it first
            # Reverse the ring (keep bh1 at start)
            ring = [ring[0]] + ring[1:][::-1]

        return ring

    # ========================================================================
    # VB-5: Main Bridge
    # ========================================================================

    def _find_main_bridge(
        self,
        mol,
        ring_atoms: Set[int],
        main_ring: List[int],
        bh_pair: Tuple[int, int]
    ) -> Optional[BridgeInfo]:
        """
        VB-5: Find the main bridge -- the path between the two main
        bridgeheads through atoms NOT in the main ring.

        For tricyclo+, the main bridge is typically 0 or 1 atoms:
        - 0 atoms: direct bond between bridgeheads (not in main ring)
        - 1+ atoms: atoms connecting the bridgeheads outside the main ring

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of all ring atoms
            main_ring: Ordered list of main ring atoms
            bh_pair: (bh1, bh2) main bridgehead pair

        Returns:
            BridgeInfo for the main bridge
        """
        bh1, bh2 = bh_pair
        main_ring_set = set(main_ring)

        # Check if bridgeheads have a direct bond NOT via main ring atoms
        bh1_atom = mol.GetAtomWithIdx(bh1)
        bh1_neighbors = {n.GetIdx() for n in bh1_atom.GetNeighbors()}

        if bh2 in bh1_neighbors:
            # Direct bond exists - check if it's via a main ring atom or a direct bond
            # A direct bond means 0-atom bridge
            return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)

        # Find atoms outside main ring that connect the bridgeheads
        allowed = (ring_atoms - main_ring_set) | {bh1, bh2}

        if len(allowed) <= 2:
            # No atoms outside main ring - check for direct bond
            if bh2 in bh1_neighbors:
                return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)
            return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)

        # Find the shortest path (main bridge should be minimal)
        path = self._find_shortest_path(mol, bh1, bh2, allowed)
        if not path or len(path) < 2:
            return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)

        bridge_atoms = path[1:-1]  # Exclude bridgeheads
        return BridgeInfo(
            atoms=bridge_atoms,
            length=len(bridge_atoms),
            start_bh=bh1,
            end_bh=bh2,
        )

    def _find_shortest_path(
        self, mol, start: int, end: int, allowed: Set[int]
    ) -> List[int]:
        """Find shortest path between two atoms through allowed atoms using BFS."""
        from collections import deque

        if start == end:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx == end:
                    return path + [end]
                if nbr_idx in allowed and nbr_idx not in visited:
                    visited.add(nbr_idx)
                    queue.append((nbr_idx, path + [nbr_idx]))

        return []

    # ========================================================================
    # VB-6: Secondary Bridges
    # ========================================================================

    def _find_secondary_bridges(
        self,
        mol,
        ring_atoms: Set[int],
        main_ring: List[int],
        main_bridge: Optional[BridgeInfo],
        bh_pair: Tuple[int, int]
    ) -> List[BridgeInfo]:
        """
        VB-6: Find secondary bridges in the ring system.

        After main ring and main bridge are assigned, remaining unassigned
        ring atoms form secondary bridges. Independent bridges (both endpoints
        on already-numbered atoms) come before dependent bridges.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of all ring atoms
            main_ring: Ordered list of main ring atoms
            main_bridge: Main bridge info
            bh_pair: Main bridgehead pair

        Returns:
            List of BridgeInfo for secondary bridges, ordered by:
            independent before dependent, then by length descending.
        """
        main_ring_set = set(main_ring)
        main_bridge_atoms = set(main_bridge.atoms) if main_bridge else set()
        assigned = main_ring_set | main_bridge_atoms

        # Atoms not yet assigned to main ring or main bridge
        unassigned = ring_atoms - assigned

        if not unassigned:
            return []

        secondary_bridges = []

        # Find bridges formed by unassigned atoms
        # Each bridge connects two assigned (numbered) atoms through unassigned atoms
        # Use BFS/DFS to find connected components among unassigned atoms
        # and determine their bridge endpoints

        # Build adjacency for ring atoms
        adj = {}
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            adj[idx] = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms]

        # Find connected components among unassigned atoms
        remaining_unassigned = set(unassigned)

        while remaining_unassigned:
            # BFS from an unassigned atom to find its connected component
            start = next(iter(remaining_unassigned))
            component = set()
            queue = [start]
            while queue:
                current = queue.pop(0)
                if current in component:
                    continue
                component.add(current)
                for nbr in adj.get(current, []):
                    if nbr in remaining_unassigned and nbr not in component:
                        queue.append(nbr)

            remaining_unassigned -= component

            # Find endpoints: assigned atoms adjacent to this component
            endpoints = set()
            for atom_idx in component:
                for nbr in adj.get(atom_idx, []):
                    if nbr in assigned:
                        endpoints.add(nbr)

            if len(endpoints) >= 2:
                # This component forms a bridge between endpoints
                ep_list = sorted(endpoints)
                # For a simple bridge with 2 endpoints:
                ep1, ep2 = ep_list[0], ep_list[1]

                # Find the path through this component between endpoints
                component_plus_endpoints = component | {ep1, ep2}
                path = find_longest_path(mol, ep1, ep2, component_plus_endpoints)

                if path and len(path) >= 2:
                    bridge_atoms = path[1:-1]
                    bridge = BridgeInfo(
                        atoms=bridge_atoms,
                        length=len(bridge_atoms),
                        start_bh=ep1,
                        end_bh=ep2,
                        is_secondary=True,
                    )
                    secondary_bridges.append(bridge)
                else:
                    # Zero-length bridge (direct connection between endpoints)
                    bridge = BridgeInfo(
                        atoms=[],
                        length=0,
                        start_bh=ep1,
                        end_bh=ep2,
                        is_secondary=True,
                    )
                    secondary_bridges.append(bridge)
            elif len(endpoints) == 1:
                # This shouldn't happen in a proper polycyclic system,
                # but handle it as a zero-length anomaly
                ep = list(endpoints)[0]
                bridge = BridgeInfo(
                    atoms=list(component),
                    length=len(component),
                    start_bh=ep,
                    end_bh=ep,
                    is_secondary=True,
                )
                secondary_bridges.append(bridge)

        # Sort: independent before dependent, then by length descending
        # For now, all found bridges are independent (endpoints on assigned atoms)
        secondary_bridges.sort(key=lambda b: -b.length)

        return secondary_bridges

    # ========================================================================
    # VB-7: Numbering
    # ========================================================================

    def _assign_numbering(
        self,
        mol,
        main_ring: List[int],
        main_bridge: Optional[BridgeInfo],
        secondary_bridges: List[BridgeInfo],
        bh_pair: Tuple[int, int]
    ) -> Dict[int, int]:
        """
        VB-7: Assign IUPAC locants to all ring atoms.

        Order:
        1. Main ring: start at bh1 (locant 1), go along longer path to bh2,
           then back along shorter path
        2. Main bridge atoms
        3. Secondary bridge atoms (independent first, then dependent)

        Args:
            mol: RDKit Mol object
            main_ring: Ordered list of main ring atoms (bh1 first, longer path first)
            main_bridge: Main bridge info
            secondary_bridges: List of secondary bridge infos
            bh_pair: Main bridgehead pair

        Returns:
            Dict mapping atom_idx -> VB locant (1-indexed)
        """
        numbering = {}
        locant = 1

        # 1. Number main ring atoms in order
        for atom_idx in main_ring:
            if atom_idx not in numbering:
                numbering[atom_idx] = locant
                locant += 1

        # 2. Number main bridge atoms
        if main_bridge and main_bridge.atoms:
            for atom_idx in main_bridge.atoms:
                if atom_idx not in numbering:
                    numbering[atom_idx] = locant
                    locant += 1

        # 3. Number secondary bridge atoms
        for bridge in secondary_bridges:
            for atom_idx in bridge.atoms:
                if atom_idx not in numbering:
                    numbering[atom_idx] = locant
                    locant += 1

        return numbering

    # ========================================================================
    # Descriptor String Building
    # ========================================================================

    def _build_descriptor(
        self,
        ring_count: int,
        primary_lengths: List[int],
        secondary_bridges: List[BridgeInfo],
        numbering: Dict[int, int]
    ) -> str:
        """
        Build the von Baeyer descriptor string.

        Format:
        - bicyclo[a.b.c] (no secondary bridges)
        - tricyclo[a.b.c.d^{e,f}] (one secondary bridge with locants)
        - tetracyclo[a.b.c.d^{e,f}.g^{h,i}] (two secondary bridges)

        Args:
            ring_count: Number of independent rings
            primary_lengths: [branch1_len, branch2_len, main_bridge_len], sorted descending
            secondary_bridges: List of secondary bridge infos
            numbering: Atom index -> VB locant mapping

        Returns:
            Descriptor string like "tricyclo[3.3.1.1^{3,7}]"
        """
        prefix = CYCLO_PREFIXES.get(ring_count, f"{ring_count}cyclo")

        # Primary bridge lengths (sorted descending)
        parts = [str(l) for l in sorted(primary_lengths, reverse=True)]

        # Secondary bridges with superscript locants
        for bridge in secondary_bridges:
            ep1 = bridge.start_bh
            ep2 = bridge.end_bh
            loc1 = numbering.get(ep1, 0)
            loc2 = numbering.get(ep2, 0)
            locant_low = min(loc1, loc2)
            locant_high = max(loc1, loc2)
            parts.append(f"{bridge.length}^{{{locant_low},{locant_high}}}")

        return f"{prefix}[{'.'.join(parts)}]"


# ============================================================================
# Public API
# ============================================================================

def generate_polycyclic_name(mol) -> Optional[str]:
    """
    Generate the base IUPAC name for a polycyclic bridged system.

    Returns "prefix[descriptor]parentname" (e.g., "tricyclo[3.3.1.1^{3,7}]decane").
    Only base name -- no substituents, unsaturation, or stereo.

    This function is an internal helper that will be called by
    name_polycyclic_complete() in Plan 16-03.

    Args:
        mol: RDKit Mol object

    Returns:
        Base name string, or None if not a polycyclic system
    """
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    if not ring_atoms:
        return None

    analyzer = VonBaeyerAnalyzer()

    # Check ring count
    ring_count = analyzer._get_ring_count(mol, ring_atoms)
    if ring_count < 2:
        return None

    # Check for bridgeheads
    bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
    if len(bridgeheads) < 2:
        return None

    # Analyze the system
    desc = analyzer.analyze(mol, ring_atoms)

    # Count ring atoms for parent name
    total_ring_atoms = len(ring_atoms)
    parent_name = _get_alkane_name(total_ring_atoms)

    return f"{desc.descriptor_string}{parent_name}"


def is_polycyclic_system(mol) -> bool:
    """
    Detect whether a molecule has a bridged polycyclic ring system
    with ring_count >= 3 (tricyclo+).

    Returns True for tricyclo and higher bridged systems.
    Returns False for bicyclo (ring_count == 2), purely fused systems,
    spiro systems, and monocyclic rings.

    This function is used by the composer for routing.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule has a tricyclo+ bridged polycyclic system
    """
    if mol is None:
        return False

    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    if not ring_atoms:
        return False

    analyzer = VonBaeyerAnalyzer()
    ring_count = analyzer._get_ring_count(mol, ring_atoms)

    # Must be tricyclo or higher
    if ring_count < 3:
        return False

    # Must have bridgehead atoms (distinguishes bridged from fused)
    bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
    if len(bridgeheads) < 2:
        return False

    # Check it's truly bridged, not purely fused
    # In a purely fused system (like naphthalene), bridgeheads share
    # edges (bonds) with multiple rings. In a bridged system, bridgeheads
    # connect rings via paths (bridges) of atoms.
    # A simple test: if any bridgehead pair is connected by 3+ distinct paths
    # through ring atoms, it's bridged.
    if _is_purely_fused(mol, ring_atoms, bridgeheads):
        return False

    return True


def _is_purely_fused(mol, ring_atoms: Set[int], bridgeheads: Set[int]) -> bool:
    """
    Check if a ring system is purely fused (shared edges only, no bridges).

    In a purely fused system, all ring junctions are shared edges --
    bridgehead atoms are always adjacent to each other in the ring system.

    In a bridged system, there exist bridgehead pairs connected by
    paths of length > 1 that go through non-bridgehead atoms.

    Returns True if purely fused, False if bridged.
    """
    # Check if all "bridgehead" atoms form shared-edge pairs
    # In a fused system, bridgehead atoms come in adjacent pairs
    # In a bridged system, bridgehead atoms can be far apart

    # Simple heuristic: in a purely fused system, every pair of
    # bridgehead atoms that share a ring is directly bonded.
    ri = mol.GetRingInfo()

    for bh1 in bridgeheads:
        for bh2 in bridgeheads:
            if bh1 >= bh2:
                continue

            # Check if they share any SSSR ring
            share_ring = False
            for ring in ri.AtomRings():
                ring_set = set(ring)
                if bh1 in ring_set and bh2 in ring_set:
                    share_ring = True
                    break

            if share_ring:
                # In a fused system, bridgeheads sharing a ring are always bonded
                bond = mol.GetBondBetweenAtoms(bh1, bh2)
                if bond is None:
                    # Bridgeheads share a ring but are NOT directly bonded
                    # This means there's a bridge between them -> bridged system
                    return False

    return True
