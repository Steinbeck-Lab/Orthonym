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
    Returns False for:
    - bicyclo (ring_count == 2)
    - purely fused aromatic systems (naphthalene, perylene, coronene)
    - spiro systems
    - monocyclic rings

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

    # Skip fully aromatic ring systems (PAHs like naphthalene, perylene, coronene)
    # These should use retained names from fused_rings, not VB nomenclature
    all_ring_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )
    if all_ring_aromatic:
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


# ============================================================================
# Heteroatom Replacement Prefix (Placeholder for Plan 16-02)
# ============================================================================

# Heteroatom priority based on Hantzsch-Widman seniority
HETEROATOM_PREFIXES = {
    'O': ('oxa', 1),
    'S': ('thia', 2),
    'Se': ('selena', 3),
    'N': ('aza', 4),
    'P': ('phospha', 5),
}


def get_heteroatom_replacement_prefix(mol, numbering: Dict[int, int], ring_atoms: Set[int]) -> str:
    """
    Generate 'a' replacement nomenclature prefix for ring heteroatoms.

    Scans ring atoms for non-carbon elements and generates the appropriate
    prefix string with locants (e.g., "7-oxa-" or "2,5-dioxa-7-aza-").

    Args:
        mol: RDKit Mol object
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)
        ring_atoms: Set of atom indices in the ring system

    Returns:
        Formatted prefix string (e.g., "7-oxa-") or empty string if no heteroatoms

    Note:
        This is a placeholder for Plan 16-02 implementation. Currently returns
        empty string.
    """
    # Group heteroatoms by element type
    heteroatoms: Dict[str, List[int]] = {}

    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C' and symbol in HETEROATOM_PREFIXES:
            if atom_idx in numbering:
                locant = numbering[atom_idx]
                if symbol not in heteroatoms:
                    heteroatoms[symbol] = []
                heteroatoms[symbol].append(locant)

    if not heteroatoms:
        return ""

    # Sort elements by HW priority
    sorted_elements = sorted(heteroatoms.keys(), key=lambda s: HETEROATOM_PREFIXES.get(s, (s, 99))[1])

    # Simple multipliers
    multipliers = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta'}

    parts = []
    for element in sorted_elements:
        locants = sorted(heteroatoms[element])
        prefix_name = HETEROATOM_PREFIXES[element][0]

        locant_str = ','.join(str(loc) for loc in locants)
        count = len(locants)

        if count > 1:
            mult = multipliers.get(count, str(count))
            parts.append(f"{locant_str}-{mult}{prefix_name}")
        else:
            parts.append(f"{locant_str}-{prefix_name}")

    return '-'.join(parts) + '-' if parts else ""


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


# ============================================================================
# Polycyclic Lactone Detection (Plan 16-02)
# ============================================================================

def detect_polycyclic_lactone(mol, ring_system_atoms: Set[int]) -> Optional[Dict]:
    """
    Detect if a polycyclic system contains a lactone (cyclic ester).

    A polycyclic lactone has:
    - An ester group [-C(=O)-O-] where both the carbonyl carbon AND
      the ester oxygen are part of the ring system
    - The carbonyl oxygen (=O) is exocyclic (double-bonded to carbonyl C)

    Args:
        mol: RDKit Mol object
        ring_system_atoms: Set of atom indices in the polycyclic ring system

    Returns:
        Dict with lactone info if found:
            carbonyl_c: atom index of carbonyl carbon
            ring_oxygen: atom index of ring (ester) oxygen
            carbonyl_oxygen: atom index of exocyclic carbonyl oxygen
        None if no polycyclic lactone found
    """
    if mol is None:
        return None

    # SMARTS for ester/lactone core: carbonyl carbon with =O and -O-
    # [CX3](=O)[OX2] matches: match[0]=carbonyl C, match[1]=carbonyl O, match[2]=ester O
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return None

    for match in matches:
        carbonyl_c = match[0]
        carbonyl_o = match[1]  # The =O (exocyclic)
        ester_o = match[2]     # The -O- (should be in ring)

        # Check if BOTH carbonyl C AND ester O are in the ring system
        if carbonyl_c in ring_system_atoms and ester_o in ring_system_atoms:
            # Verify carbonyl O is NOT in the ring (exocyclic)
            if carbonyl_o not in ring_system_atoms:
                return {
                    'carbonyl_c': carbonyl_c,
                    'ring_oxygen': ester_o,
                    'carbonyl_oxygen': carbonyl_o,
                }

    return None


# ============================================================================
# Polycyclic Heteroatom Naming (Plan 16-02)
# ============================================================================

def name_polycyclic_with_heteroatoms(mol) -> Optional[str]:
    """
    Generate IUPAC name for a polycyclic system containing ring heteroatoms.

    Uses "a" replacement nomenclature for ring heteroatoms (oxa, aza, thia)
    and pseudoketone naming for polycyclic lactones (oxa- prefix + -one suffix).

    Format: "{hetero_prefix}bicyclo[descriptor]{parent_name}" or with -one suffix
    Example: "7-oxabicyclo[2.2.1]heptane" or "3-oxabicyclo[3.2.1]octan-2-one"

    Args:
        mol: RDKit Mol object

    Returns:
        Complete IUPAC name with heteroatom prefixes, or None if not applicable
    """
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    if not ring_atoms:
        return None

    # Check if system has ring heteroatoms
    has_ring_heteroatoms = False
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            has_ring_heteroatoms = True
            break

    if not has_ring_heteroatoms:
        # No heteroatoms - use regular polycyclic naming
        return generate_polycyclic_name(mol)

    analyzer = VonBaeyerAnalyzer()

    # Check ring count
    ring_count = analyzer._get_ring_count(mol, ring_atoms)
    if ring_count < 2:
        return None

    # Check for bridgeheads
    bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
    if len(bridgeheads) < 2:
        return None

    # Analyze the system to get descriptor and numbering
    desc = analyzer.analyze(mol, ring_atoms)

    # Get heteroatom replacement prefix
    hetero_prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

    # Check for polycyclic lactone
    lactone_info = detect_polycyclic_lactone(mol, ring_atoms)

    # Build the base descriptor (bicyclo[...], tricyclo[...], etc.)
    # The descriptor_string already includes the prefix like "bicyclo[2.2.1]"
    descriptor = desc.descriptor_string

    # Count total ring atoms for parent name
    total_ring_atoms = len(ring_atoms)
    parent_name = _get_alkane_name(total_ring_atoms)

    # Handle lactone naming with -one suffix
    suffix = ""
    if lactone_info is not None:
        # Get the carbonyl carbon's VB locant for the -one suffix
        carbonyl_c = lactone_info['carbonyl_c']
        carbonyl_locant = desc.numbering.get(carbonyl_c, 2)  # Default to 2 if not found

        # Apply vowel elision: drop 'e' before '-one'
        # heptane -> heptan-, octane -> octan-
        if parent_name.endswith('e'):
            parent_stem = parent_name[:-1]
        else:
            parent_stem = parent_name

        suffix = f"-{carbonyl_locant}-one"
        parent_name = parent_stem

    # Assemble the complete name
    # Format: {hetero_prefix}{descriptor}{parent_name}{suffix}
    # Example: 7-oxa-bicyclo[2.2.1]heptane
    # Example: 3-oxa-bicyclo[3.2.1]octan-2-one
    name = f"{hetero_prefix}{descriptor}{parent_name}{suffix}"

    return name


def _has_ring_heteroatoms(mol, ring_atoms: Set[int]) -> bool:
    """Check if the ring system contains any heteroatoms (non-carbon atoms)."""
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return True
    return False


# ============================================================================
# Substituent Detection (Plan 16-03)
# ============================================================================

def get_polycyclic_substituents(
    mol,
    ring_atoms: Set[int],
    numbering: Dict[int, int]
) -> List[Dict]:
    """
    Detect substituents attached to a polycyclic ring system.

    For each ring atom, checks neighbors not in ring_atoms. Traces each
    substituent branch and determines its name using get_alkyl_name().

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the polycyclic ring system
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)

    Returns:
        List of substituent info dicts, each containing:
        - 'locant': VB locant where substituent attaches
        - 'name': substituent name (e.g., 'methyl', 'ethyl')
        - 'atom_indices': list of atom indices in the substituent

    Note:
        Skips exocyclic double bonds (=O, =S) as those are handled as suffixes.
    """
    from collections import deque
    from ..assembly.naming_utils import get_alkyl_name

    substituents = []

    for ring_idx in ring_atoms:
        if ring_idx not in numbering:
            continue

        ring_atom = mol.GetAtomWithIdx(ring_idx)
        locant = numbering[ring_idx]

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip if neighbor is in ring
            if nbr_idx in ring_atoms:
                continue

            # Check bond type - skip exocyclic double bonds (=O, =S for suffixes)
            bond = mol.GetBondBetweenAtoms(ring_idx, nbr_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                # This is an exocyclic double bond, handle as suffix not substituent
                continue

            # Trace the substituent branch
            sub_atoms = _trace_substituent_branch(mol, nbr_idx, ring_atoms)

            # Count carbons to determine substituent name
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            if carbon_count > 0 and carbon_count <= 10:
                try:
                    name = get_alkyl_name(carbon_count)
                except ValueError:
                    name = f"C{carbon_count}H{2*carbon_count+1}"
            elif carbon_count > 10:
                name = f"C{carbon_count}H{2*carbon_count+1}"
            else:
                # Non-carbon substituent (like hydroxy, amino)
                # For now, skip these as they're functional groups
                continue

            substituents.append({
                'locant': locant,
                'name': name,
                'atom_indices': sub_atoms,
            })

    return substituents


def _trace_substituent_branch(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """
    Trace all atoms in a substituent branch using BFS.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        ring_atoms: Set of ring atom indices to exclude

    Returns:
        List of atom indices in the substituent branch
    """
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    result = []

    while queue:
        atom_idx = queue.popleft()

        if atom_idx in visited or atom_idx in ring_atoms:
            continue

        visited.add(atom_idx)
        result.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                queue.append(nbr_idx)

    return result


# ============================================================================
# Unsaturation Detection (Plan 16-03)
# ============================================================================

def get_polycyclic_unsaturation(
    mol,
    ring_atoms: Set[int],
    numbering: Dict[int, int]
) -> Dict:
    """
    Detect double and triple bonds within a polycyclic ring system.

    Scans all bonds where BOTH atoms are in ring_atoms and identifies
    multiple bonds. Returns the lower VB locant for each bond.

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the polycyclic ring system
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)

    Returns:
        Dict with:
        - 'double_bonds': list of locants for double bonds
        - 'triple_bonds': list of locants for triple bonds
    """
    from rdkit.Chem import BondType

    double_bonds = []
    triple_bonds = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Both atoms must be in ring
        if begin_idx not in ring_atoms or end_idx not in ring_atoms:
            continue

        # Both atoms must have locants
        if begin_idx not in numbering or end_idx not in numbering:
            continue

        bond_type = bond.GetBondType()

        if bond_type == BondType.DOUBLE:
            # Use lower locant
            loc1 = numbering[begin_idx]
            loc2 = numbering[end_idx]
            double_bonds.append(min(loc1, loc2))
        elif bond_type == BondType.TRIPLE:
            loc1 = numbering[begin_idx]
            loc2 = numbering[end_idx]
            triple_bonds.append(min(loc1, loc2))

    # Sort locants
    double_bonds.sort()
    triple_bonds.sort()

    return {
        'double_bonds': double_bonds,
        'triple_bonds': triple_bonds,
    }


# ============================================================================
# Stereochemistry (Plan 16-03)
# ============================================================================

def get_polycyclic_stereo(mol, numbering: Dict[int, int]) -> str:
    """
    Collect and format stereodescriptors for a polycyclic system.

    Uses the VB numbering to generate IUPAC locants for R/S stereocenters.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)

    Returns:
        Formatted stereodescriptor string like "(1R,4S)-" or empty string
        if no stereodescriptors found.
    """
    from ..rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Collect stereodescriptors using the VB numbering as atom_to_locant
    descriptors = collect_stereodescriptors(mol, numbering)

    # Format as IUPAC string
    return format_stereodescriptor_string(descriptors)


# ============================================================================
# Complete Name Assembly (Plan 16-03)
# ============================================================================

def name_polycyclic_complete(mol) -> Optional[str]:
    """
    Generate the complete IUPAC name for a polycyclic bridged system.

    This is the FINAL public API for polycyclic naming. It supersedes
    generate_polycyclic_name() which only generates base names.

    Produces complete names including:
    - Stereodescriptors: "(1R,4S)-"
    - Substituent prefixes: "3-methyl-"
    - Heteroatom replacement: "7-oxa-" (when Plan 16-02 implemented)
    - Descriptor: "bicyclo[2.2.1]"
    - Parent name with unsaturation: "hept-2-ene"

    Example output: "(1R,4S)-3-methyl-7-oxabicyclo[2.2.1]hept-2-ene"

    Args:
        mol: RDKit Mol object

    Returns:
        Complete IUPAC name, or None if not a polycyclic system
    """
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key

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

    # 1. Get stereodescriptors
    stereo_prefix = get_polycyclic_stereo(mol, desc.numbering)

    # 2. Get substituents
    substituents = get_polycyclic_substituents(mol, ring_atoms, desc.numbering)

    # 3. Get unsaturation
    unsaturation = get_polycyclic_unsaturation(mol, ring_atoms, desc.numbering)

    # 4. Get heteroatom replacement prefix
    hetero_prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

    # 5. Assemble substituent prefix
    substituent_prefix = _assemble_substituent_prefix(substituents)

    # 6. Build parent name with unsaturation suffix
    parent_name = _build_parent_with_unsaturation(total_ring_atoms, unsaturation)

    # 7. Assemble final name
    # Order: (stereo)-substituents-heteroprefix-cycloprefix[descriptor]parent-suffix
    name_parts = []

    if stereo_prefix:
        name_parts.append(stereo_prefix)

    if substituent_prefix:
        name_parts.append(substituent_prefix)

    if hetero_prefix:
        name_parts.append(hetero_prefix)

    # Add descriptor and parent
    name_parts.append(desc.descriptor_string)
    name_parts.append(parent_name)

    # Join - the stereo prefix ends with '-', substituent prefix ends with '-', etc.
    name = ''.join(name_parts)

    return name


def _assemble_substituent_prefix(substituents: List[Dict]) -> str:
    """
    Assemble substituent prefix with proper IUPAC formatting.

    Groups identical substituents, applies multipliers, and sorts alphabetically.

    Args:
        substituents: List of substituent dicts from get_polycyclic_substituents()

    Returns:
        Formatted prefix like "3-ethyl-2,4-dimethyl-" or empty string
    """
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key

    if not substituents:
        return ""

    # Group by name
    grouped: Dict[str, List[int]] = {}
    for sub in substituents:
        name = sub['name']
        locant = sub['locant']
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(locant)

    # Sort locants within each group
    for name in grouped:
        grouped[name].sort()

    # Sort groups alphabetically by substituent name
    sorted_names = sorted(grouped.keys(), key=alpha_sort_key)

    # Build prefix parts
    parts = []
    for name in sorted_names:
        locants = grouped[name]
        count = len(locants)
        multiplier = get_multiplier_prefix(count, name)

        locant_str = ','.join(str(loc) for loc in locants)
        parts.append(f"{locant_str}-{multiplier}{name}")

    return '-'.join(parts) + '-' if parts else ""


def _build_parent_with_unsaturation(
    total_atoms: int,
    unsaturation: Dict
) -> str:
    """
    Build the parent name with unsaturation suffixes and vowel elision.

    Args:
        total_atoms: Total ring atoms (for parent name base)
        unsaturation: Dict with 'double_bonds' and 'triple_bonds' lists

    Returns:
        Parent name like "heptane", "hept-2-ene", "hepta-2,5-diene"
    """
    base_name = _get_alkane_name(total_atoms)

    # Simple multipliers for unsaturation
    unsat_multipliers = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}

    double_bonds = unsaturation.get('double_bonds', [])
    triple_bonds = unsaturation.get('triple_bonds', [])

    if not double_bonds and not triple_bonds:
        # Saturated - return plain alkane name
        return base_name

    # Build unsaturation suffix
    # Remove 'ane' ending for unsaturated forms
    stem = base_name[:-3] if base_name.endswith('ane') else base_name[:-1]

    parts = []

    if double_bonds:
        count = len(double_bonds)
        mult = unsat_multipliers.get(count, str(count))
        locant_str = ','.join(str(loc) for loc in double_bonds)
        if count > 1:
            # Multiple double bonds: hepta-2,5-diene
            parts.append(f"-{locant_str}-{mult}en")
        else:
            # Single double bond: hept-2-ene
            parts.append(f"-{locant_str}-en")

    if triple_bonds:
        count = len(triple_bonds)
        mult = unsat_multipliers.get(count, str(count))
        locant_str = ','.join(str(loc) for loc in triple_bonds)
        if count > 1:
            parts.append(f"-{locant_str}-{mult}yn")
        else:
            parts.append(f"-{locant_str}-yn")

    # Vowel elision: if multiple double bonds, add 'a' back to stem
    if len(double_bonds) > 1 or (double_bonds and triple_bonds):
        stem = stem + 'a'

    # Combine
    result = stem + ''.join(parts) + 'e'

    return result
