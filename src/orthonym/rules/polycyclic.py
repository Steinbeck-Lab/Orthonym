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

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations
from rdkit import Chem

logger = logging.getLogger(__name__)


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
    11: "undecacyclo",
    12: "dodecacyclo",
    13: "tridecacyclo",
    14: "tetradecacyclo",
    15: "pentadecacyclo",
    16: "hexadecacyclo",
    17: "heptadecacyclo",
    18: "octadecacyclo",
    19: "nonadecacyclo",
    20: "icosacyclo",
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


def cyclo_ring_count_word(ring_count: int) -> Optional[str]:
    """The von Baeyer ring-count term for ``ring_count`` rings, or ``None``.

    P-23.1.9 (``BlueBookV2.md:9558``): "The number of rings is indicated by the
    nondetachable prefix 'bicyclo' (not dicyclo), 'tricyclo', 'tetracyclo',
    etc." -- restated at P-23.2.6.1.1 (``:9645``). Both sentences end in "etc.":
    the series is OPEN-ENDED and the Blue Book prints **no table** of these
    words (the highest one attested anywhere in the text is ``hexacyclo``,
    ``:9731``). The term is therefore COMPUTED -- simple multiplying prefix
    (Table 1.4, P-14.2.1) + ``cyclo`` -- with the single irregularity that 2 is
    ``bi``, not ``di``. No vowel elision applies: ``cyclo`` starts with a
    consonant.

    ``CYCLO_PREFIXES`` is consulted first purely as a fast, human-auditable
    path; it agrees with the composition at every entry it holds (asserted by
    ``test_table_agrees_with_composition``), so it is a cache of the rule and
    not a second, competing definition of it.

    Returns ``None`` when no word can be formed, so callers FAIL CLOSED rather
    than emit the non-word the previous f-string fallback produced (``"21cyclo"``
    for 21 rings). That fallback was NOT latent: the PIN path
    (``name_polycyclic_complete``) caps nothing above ``ring_count < 2``, and a
    21-ring cage really did emit ``21cyclo[...]tetratetracontane``. The
    ``MAX_CAGE_RINGS = 8`` ceiling guards only the opt-in general-engine path in
    ``vonbaeyer_universal``, which is a different caller.
    """
    if ring_count == 2:
        return "bicyclo"  # P-23.1.9: 'bicyclo', explicitly NOT 'dicyclo'
    word = CYCLO_PREFIXES.get(ring_count)
    if word is not None:
        return word
    from ..assembly.naming_utils import simple_multiplier_word
    multiplier = simple_multiplier_word(ring_count)
    if multiplier is None:
        return None
    return multiplier + "cyclo"


def _get_alkane_name(carbon_count: int) -> str:
    """Get the alkane parent name for a given carbon count."""
    if carbon_count in _ALKANE_NAMES:
        return _ALKANE_NAMES[carbon_count]
    # Delegate to centralized chain names for any size
    from ..data.chain_names import get_chain_name
    return get_chain_name(carbon_count)


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
    is_dependent: bool = False  # True for dependent secondary bridges (VB-7)
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

# WR-02 (code review 2026-06-02): hard cap on DFS node-expansions. Simple-path
# enumeration over a dense ring graph is combinatorial in the cycle count, NOT
# linear in atom count, so the "<50 atoms => fast" assumption fails on a
# highly-bridged cage (fullerene fragment, dense cage input) — exactly the kind
# of system the relaxed SUB-02 bridgehead predicate now admits into the von
# Baeyer analyzer. Without a bound an adversarial single SMILES could hang the
# namer (a denial of service, not merely "slow"). A legitimate polycyclic
# finishes in orders of magnitude fewer expansions than this cap, so real inputs
# are byte-identical; on a pathological input the search aborts and returns the
# best/partial result found so far (a valid, if not provably optimal, path —
# the caller's _find_main_ring_fallback / shortest-path handling degrades
# gracefully from there).
_MAX_DFS_EXPANSIONS = 200_000


def find_longest_path(mol, start: int, end: int, allowed_atoms: Set[int]) -> List[int]:
    """
    Find the longest simple path from start to end through allowed atoms.

    Uses DFS with backtracking. For polycyclic ring systems (<50 atoms),
    exhaustive search is feasible and fast. The search is bounded by
    ``_MAX_DFS_EXPANSIONS`` (WR-02) so a pathological dense cage cannot hang;
    on abort the best path found so far is returned.

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
    expansions = 0

    def dfs(current, visited, path):
        nonlocal best_path, expansions
        expansions += 1
        if expansions > _MAX_DFS_EXPANSIONS:
            return  # WR-02 cap: abort exploration, keep best-so-far
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

    Returns list of paths, each a list of atom indices. The enumeration is
    bounded by ``_MAX_DFS_EXPANSIONS`` (WR-02) — on a pathological dense graph
    it returns the paths found before the cap rather than hanging.
    """
    if start == end:
        return [[start]]

    all_paths = []
    expansions = 0

    def dfs(current, visited, path):
        nonlocal expansions
        expansions += 1
        if expansions > _MAX_DFS_EXPANSIONS:
            return  # WR-02 cap: abort enumeration, keep paths-so-far
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

    @staticmethod
    def _canonical_atom_order(mol) -> Optional[List[int]]:
        """A fully spelling-invariant atom order: the order in which RDKit
        emits atoms in the canonical SMILES.

        ``CanonicalRankAtoms(breakTies=True)`` is NOT invariant across input
        orderings for symmetric molecules (its tie-break depends on the start
        atom), which left symmetric cages (homocubane, nortricyclene)
        non-deterministic. The canonical-SMILES output order is invariant
        (verified across random spellings), so renumbering by it gives a
        molecule whose atom indices are identical for every spelling.

        Returns ``order`` such that ``order[new_idx] = old_idx``, or None if the
        order cannot be obtained.
        """
        try:
            # MolToSmiles records the canonical output order on ``mol`` as the
            # private string property "_smilesAtomOutputOrder" (e.g.
            # "[2,3,4,5,1,0,6]"); GetPropsAsDict does not surface it, so read it
            # explicitly via GetProp.
            Chem.MolToSmiles(mol)
            if not mol.HasProp("_smilesAtomOutputOrder"):
                return None
            raw = mol.GetProp("_smilesAtomOutputOrder").strip()
            order = [int(x) for x in raw.strip("[]").split(",") if x != ""]
            n = mol.GetNumAtoms()
            if len(order) == n and sorted(order) == list(range(n)):
                return order
        except Exception:
            return None
        return None

    @staticmethod
    def _descriptor_is_valid(result: "PolycyclicDescriptor",
                             ring_atoms: Set[int]) -> bool:
        """Well-formed iff bridge lengths are all non-negative, the
        P-23.2.6.1.1 invariant holds (sum + 2 == ring atoms), and every ring
        atom received a locant. Malformed descriptors are OPSIN-unparseable.
        """
        if result is None:
            return False
        if any(l < 0 for l in result.bridge_lengths):
            return False
        if sum(result.bridge_lengths) + 2 != result.total_atoms:
            return False
        return set(result.numbering.keys()) >= set(ring_atoms)

    @staticmethod
    def _is_unsubstituted_ring_system(mol, ring_atoms: Set[int]) -> bool:
        """True iff no ring atom carries an exocyclic heavy-atom substituent.

        Only such 'pure cages' (prismane, cubane, nortricyclene, ...) are
        renumbered for determinism: their name is just the descriptor + parent,
        so re-numbering the core can only permute the secondary-bridge
        superscript locants -- which OPSIN round-trips either way -- and can
        never misplace a substituent. Substituted/heteroatom-bearing cages skip
        the renumber so their downstream locants are untouched (renumbering
        them was shown to break round-trips).
        """
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != "C":
                return False
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1:
                    return False
        return True

    def analyze(self, mol, ring_atoms: Set[int]) -> PolycyclicDescriptor:
        """
        Main entry point: analyze a polycyclic system and produce its descriptor.

        Determinism (WS-D): for an UNSUBSTITUTED cage the cascade is run on a
        copy renumbered into a total RDKit canonical-rank order (identical for
        every SMILES spelling), making the descriptor spelling-independent; the
        numbering is then mapped back to the caller's atom indices. If the
        canonical run yields a malformed descriptor it falls back to the
        original order, so the result is never worse than the un-renumbered
        path. SUBSTITUTED / heteroatom cages are NOT renumbered (that shifts
        substituent locants and breaks round-trips); they still tie-break on
        canonical ranks inside ``_find_main_ring``.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system

        Returns:
            PolycyclicDescriptor with all VB information
        """
        if self._is_unsubstituted_ring_system(mol, ring_atoms):
            order = self._canonical_atom_order(mol)
            if order is not None:
                old_to_new = {old: new for new, old in enumerate(order)}
                new_to_old = {new: old for new, old in enumerate(order)}
                try:
                    canon_mol = Chem.RenumberAtoms(mol, order)
                    canon_ring = {old_to_new[i] for i in ring_atoms}
                    # Pure cage: no substituents to misplace, so we may also
                    # search the main-bridgehead/direction choices for the PIN
                    # (lowest secondary-bridge superscript locant set,
                    # P-23.2.6.2.4/.5).
                    canon_result = self._analyze_impl(
                        canon_mol, canon_ring, optimize_orientation=True
                    )
                except Exception:
                    canon_result = None
                if self._descriptor_is_valid(canon_result, canon_ring):
                    canon_result.numbering = {
                        new_to_old[k]: v
                        for k, v in canon_result.numbering.items()
                        if k in new_to_old
                    }
                    for bridge in canon_result.bridge_info_list:
                        bridge.atoms = [new_to_old.get(a, a) for a in bridge.atoms]
                        bridge.start_bh = new_to_old.get(bridge.start_bh, bridge.start_bh)
                        bridge.end_bh = new_to_old.get(bridge.end_bh, bridge.end_bh)
                    return canon_result

        return self._analyze_impl(mol, ring_atoms)

    def _analyze_impl(self, mol, ring_atoms: Set[int],
                      optimize_orientation: bool = False) -> PolycyclicDescriptor:
        """Von Baeyer cascade (VB-1..VB-7) on the given atom ordering.

        When ``optimize_orientation`` is set (pure-cage path only) the main
        bridgehead chosen as locant 1 and the traversal direction are selected
        to give the lowest secondary-bridge superscript locant set
        (P-23.2.6.2.4/.5). This is RT-safe only without substituents, hence the
        caller gate; it is what turns nortricyclene 0^3,5 into the PIN 0^2,6.
        """
        ring_count = self._get_ring_count(mol, ring_atoms)
        bridgeheads = self._find_all_bridgeheads(mol, ring_atoms)
        main_ring, bh_pair = self._find_main_ring(mol, ring_atoms, bridgeheads)
        main_bridge = self._find_main_bridge(mol, ring_atoms, main_ring, bh_pair)

        if optimize_orientation:
            main_ring, bh_pair = self._select_pin_orientation(
                mol, ring_atoms, main_ring, main_bridge, bh_pair
            )
        secondary_bridges = self._find_secondary_bridges(
            mol, ring_atoms, main_ring, main_bridge, bh_pair
        )
        # VB-7: Classify, order, orient, and number secondary bridges
        # Uses two-pass algorithm: numbering order then citation order
        secondary_bridges, numbering = self._order_and_number_secondary_bridges(
            mol, secondary_bridges, main_ring, main_bridge, bh_pair
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

        # Add secondary bridges, filtering out invalid ones.
        # Invalid bridges: self-loops, unmapped endpoints, degenerate locants.
        valid_secondary = []
        for sb in secondary_bridges:
            ep1 = sb.start_bh
            ep2 = sb.end_bh

            # Filter self-loop bridges (same atom as both endpoints)
            if ep1 == ep2:
                logger.debug(
                    "Filtering self-loop bridge in analyze: ep=%s, len=%d",
                    ep1, sb.length
                )
                continue

            loc1 = numbering.get(ep1, 0)
            loc2 = numbering.get(ep2, 0)

            # Filter bridges with unmapped endpoints
            if loc1 is None or loc2 is None or loc1 == 0 or loc2 == 0:
                logger.debug(
                    "Filtering bridge with unmapped endpoint in analyze: "
                    "ep1=%s(loc=%s), ep2=%s(loc=%s)", ep1, loc1, ep2, loc2
                )
                continue

            locant_low = min(loc1, loc2)
            locant_high = max(loc1, loc2)

            # Filter degenerate bridges with identical locants
            if locant_low == locant_high:
                logger.debug(
                    "Filtering degenerate bridge in analyze: loc=%s",
                    locant_low
                )
                continue

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
            valid_secondary.append(sb)

        # Sort bridge lengths: first three (branch1, branch2, main bridge) descending,
        # then secondary bridges descending
        primary_lengths = sorted(bridge_lengths[:3], reverse=True)
        secondary_lengths = sorted(bridge_lengths[3:], reverse=True)
        bridge_lengths = primary_lengths + secondary_lengths

        # IUPAC VB-6 citation order re-sort for descriptor string
        # VB-6.2: decreasing length; VB-6.4: lowest locants as ascending set;
        # VB-6.5: lowest locants in citation order.
        # Independent bridges cited before dependent bridges.
        def _citation_sort_key(bridge):
            loc1 = numbering.get(bridge.start_bh, 0)
            loc2 = numbering.get(bridge.end_bh, 0)
            locant_low = min(loc1, loc2)
            locant_high = max(loc1, loc2)
            dep = 1 if bridge.is_dependent else 0
            return (dep, -bridge.length, locant_low, locant_high)

        citation_ordered = sorted(valid_secondary, key=_citation_sort_key)

        # Build descriptor string (using citation-ordered bridges)
        descriptor = self._build_descriptor(
            ring_count, primary_lengths, citation_ordered, numbering
        )

        total_atoms = len(ring_atoms)

        # VB invariant: sum(bridge_lengths) + 2 == total_ring_atoms
        total_bridge_len = sum(bridge_lengths)
        if total_bridge_len + 2 != total_atoms:
            logger.error(
                "VB invariant violated: sum(%s) + 2 = %d, expected %d total ring atoms",
                bridge_lengths, total_bridge_len + 2, total_atoms
            )

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
        # SUB-02/D-08: delegate to the SINGLE consolidated predicate (identical
        # ring_neighbours>=3 rule, operating on the passed ring component).
        from ..perception.rings import find_ring_bridgeheads
        return find_ring_bridgeheads(mol, ring_atoms)

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

        # Spelling-invariant tie-break key: RDKit canonical atom ranks are
        # identical for every SMILES spelling of the same molecule. breakTies=
        # True forces a TOTAL order so that even symmetry-equivalent atoms in a
        # cage (homocubane etc.) get a stable, reproducible ordering -- without
        # it the path-enumeration (GetNeighbors) order would break rank ties
        # non-deterministically. This makes the main-ring selection fully
        # order-independent (WS-D determinism).
        try:
            _ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        except Exception:
            _ranks = list(range(mol.GetNumAtoms()))

        def _rank_key(atom_indices):
            return tuple(sorted(_ranks[i] for i in atom_indices))

        best_ring = None
        best_bh_pair = None
        # Score tuple, compared with ">" (higher wins). Faithful to the
        # P-23.2.1 -> P-23.2.4 -> P-23.2.6.2.1 cascade:
        #   (ring_size,            # P-23.2.1: main ring includes max skeletal atoms
        #    main_bridge_len,      # P-23.2.4: main bridge as large as possible
        #    balance,              # P-23.2.6.2.1: main ring divided as symmetrically as possible
        #    tie_key)              # deterministic canon-rank tie-break (lowest ranks win)
        best_score = None

        def _consider(ring, bh1, bh2, branch1_len, branch2_len, main_bridge_len):
            nonlocal best_score, best_ring, best_bh_pair
            balance = min(branch1_len, branch2_len)
            tie_key = tuple(-r for r in _rank_key(ring))
            score = (len(ring), main_bridge_len, balance, tie_key)
            if best_score is None or score > best_score:
                best_score = score
                best_ring = ring
                best_bh_pair = (bh1, bh2)

        def _best_disjoint_pair(allowed, exclude_direct):
            """Largest, then most symmetric, pair of interior-disjoint bh1->bh2
            paths through ``allowed``. ``exclude_direct`` drops the trivial
            direct edge so a 0-atom main bridge keeps the rest as the main ring.
            Deterministic: ties broken by the lowest canon-rank tuple.
            """
            all_paths = _find_all_simple_paths(mol, bh1, bh2, allowed)
            if exclude_direct:
                all_paths = [p for p in all_paths if len(p) >= 3]
            if len(all_paths) < 2:
                return None
            best_local = None
            best_pair = None
            for i, p1 in enumerate(all_paths):
                p1_interior = set(p1[1:-1])
                for p2 in all_paths[i + 1:]:
                    p2_interior = set(p2[1:-1])
                    if p1_interior & p2_interior:
                        continue
                    ring_size = len(p1) + len(p2) - 2  # shared endpoints
                    balance = min(len(p1) - 2, len(p2) - 2)
                    key = (ring_size, balance,
                           tuple(-r for r in _rank_key(set(p1) | set(p2))))
                    if best_local is None or key > best_local:
                        best_local = key
                        best_pair = (p1, p2)
            return best_pair

        for bh1, bh2 in combinations(sorted(bridgeheads), 2):
            # Find common neighbors (potential 1-atom main bridge atoms)
            common_neighbors = adj.get(bh1, set()) & adj.get(bh2, set())

            # Also consider direct connection (0-atom main bridge)
            has_direct_bond = bh2 in adj.get(bh1, set())

            # Case 1: Direct bond between bridgeheads (0-atom main bridge).
            # The main ring is the largest pair of interior-disjoint paths that
            # do NOT use the direct edge; the direct bond is the main bridge.
            # (Previously this greedily grabbed a single longest path, which on
            # a dense cage consumed every atom and left no second branch -- so a
            # 0-atom-bridge main ring was never even generated, e.g. prismane.)
            if has_direct_bond:
                pair = _best_disjoint_pair(ring_atoms, exclude_direct=True)
                if pair:
                    p1, p2 = pair
                    ring = p1 + p2[1:-1][::-1]
                    _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, 0)

            # Case 2: 1-atom main bridge via a common neighbor.
            for bridge_atom in sorted(common_neighbors):
                pair = _best_disjoint_pair(ring_atoms - {bridge_atom},
                                           exclude_direct=False)
                if pair:
                    p1, p2 = pair
                    ring = p1 + p2[1:-1][::-1]
                    _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, 1)

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
        """Find a path from start to end that doesn't use the direct bond.

        Bounded by ``_MAX_DFS_EXPANSIONS`` (WR-02) so a dense cage cannot hang;
        on abort the best path found so far is returned.
        """
        # Use BFS to find shortest path first, then try longer paths
        best_path = []
        expansions = 0

        def dfs(current, visited, path):
            nonlocal best_path, expansions
            expansions += 1
            if expansions > _MAX_DFS_EXPANSIONS:
                return  # WR-02 cap: abort exploration, keep best-so-far
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
        elif forward_len == backward_len:
            # Symmetric main ring (equal branches): the traversal direction is a
            # genuine tie. Break it deterministically by choosing the rotation
            # whose post-bridgehead atom sequence is lexicographically smaller.
            # On the canonically-renumbered molecule the atom index IS the
            # canonical rank, so this yields a spelling-independent (and
            # lowest-locant) orientation -- fixing the symmetric-cage flip
            # (e.g. homocubane) without touching the decomposition.
            reflected = [ring[0]] + ring[1:][::-1]
            if reflected[1:] < ring[1:]:
                ring = reflected

        return ring

    def _select_pin_orientation(
        self, mol, ring_atoms: Set[int], main_ring: List[int],
        main_bridge: Optional[BridgeInfo], bh_pair: Tuple[int, int]
    ) -> List[int]:
        """Re-orient the main ring to the PIN numbering (RT-safe, pure cages).

        Holds the decomposition (which atoms are main-ring / main-bridge /
        secondary) FIXED and only varies which main bridgehead becomes locant 1
        and the traversal direction, then keeps the orientation whose secondary
        bridges get the lowest superscript locants as an ascending set
        (P-23.2.6.2.4), then the lowest citation sequence (P-23.2.6.2.5), with a
        deterministic atom-index backstop. Both main bridgeheads stay the main
        bridgeheads, so the main bridge is unchanged and the cage is unchanged
        -- only the numbering moves.
        """
        bh1, bh2 = bh_pair
        ring_set = set(main_ring)
        if bh1 not in ring_set or bh2 not in ring_set or bh1 == bh2:
            return main_ring
        n = len(main_ring)
        BIG = 10 ** 6

        candidates = []
        seen = set()
        for start, partner in ((bh1, bh2), (bh2, bh1)):
            s_idx = main_ring.index(start)
            rot = main_ring[s_idx:] + main_ring[:s_idx]
            p_idx = rot.index(partner)
            forward_len = p_idx - 1
            backward_len = n - p_idx - 1
            reflected = [rot[0]] + rot[1:][::-1]
            # Keep the longer branch first (VB-7); admit both directions on a tie.
            options = []
            if forward_len > backward_len:
                options = [rot]
            elif backward_len > forward_len:
                options = [reflected]
            else:
                options = [rot, reflected]
            for opt in options:
                key = tuple(opt)
                if key not in seen:
                    seen.add(key)
                    candidates.append((opt, (start, partner)))

        main_bridge_len = len(main_bridge.atoms) if main_bridge else 0

        best_key = None
        best_ring = main_ring
        best_bhp = bh_pair
        for cand_ring, cand_bhp in candidates:
            cb1, cb2 = cand_bhp
            # Primary branch lengths for this orientation (mirror _analyze_impl).
            # cb1 is at index 0 in cand_ring by construction.
            try:
                b2_pos = cand_ring.index(cb2)
            except ValueError:
                continue
            branch1_len = b2_pos - 1
            branch2_len = len(cand_ring) - b2_pos - 1
            # Reject malformed primaries (degenerate / adjacent bridgeheads).
            if branch1_len < 0 or branch2_len < 0:
                continue
            sec = self._find_secondary_bridges(
                mol, ring_atoms, cand_ring, main_bridge, cand_bhp
            )
            sec, numbering = self._order_and_number_secondary_bridges(
                mol, sec, cand_ring, main_bridge, cand_bhp
            )
            if set(numbering.keys()) < set(ring_atoms):
                continue  # incomplete numbering -> not a usable orientation
            pairs = []
            for b in sec:
                l1 = numbering.get(b.start_bh)
                l2 = numbering.get(b.end_bh)
                if not l1 or not l2:
                    pairs.append((BIG, BIG))
                else:
                    pairs.append((min(l1, l2), max(l1, l2)))
            flat_set = tuple(sorted(x for pr in pairs for x in pr))       # P-23.2.6.2.4
            citation = tuple(x for pr in pairs for x in pr)               # P-23.2.6.2.5
            idx_backstop = tuple(cand_ring)
            key = (flat_set, citation, idx_backstop)
            if best_key is None or key < best_key:
                best_key = key
                best_ring = cand_ring
                best_bhp = cand_bhp
        return best_ring, best_bhp

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

        secondary_bridges = []

        # --- Phase 1: Find bridges through unassigned atoms ---
        if unassigned:
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
                # BFS from an unassigned atom to find its connected component.
                # Seed at the LOWEST atom index (not set-iteration order) so the
                # component discovery is order-independent (WS-D determinism).
                start = min(remaining_unassigned)
                component = set()
                queue = deque([start])
                while queue:
                    current = queue.popleft()
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

        # --- Phase 2: Detect zero-length secondary bridges ---
        # When all ring atoms are already assigned (e.g., cubane), unassigned is
        # empty but additional zero-length bridges may exist as direct bonds
        # between already-numbered atoms that are NOT edges of the main ring
        # or main bridge. These represent additional ring closures.

        # Compute edges already accounted for (main ring + main bridge)
        main_ring_edges = set()
        for i in range(len(main_ring)):
            a = main_ring[i]
            b = main_ring[(i + 1) % len(main_ring)]
            main_ring_edges.add((min(a, b), max(a, b)))

        main_bridge_edges = set()
        if main_bridge and main_bridge.atoms:
            full_bridge = [bh_pair[0]] + main_bridge.atoms + [bh_pair[1]]
            for i in range(len(full_bridge) - 1):
                a = full_bridge[i]
                b = full_bridge[i + 1]
                main_bridge_edges.add((min(a, b), max(a, b)))
        elif main_bridge:
            # Zero-atom main bridge = direct bond between bridgeheads
            a, b = bh_pair
            main_bridge_edges.add((min(a, b), max(a, b)))

        accounted_edges = main_ring_edges | main_bridge_edges

        # Also add edges from bridges found via unassigned atoms (Phase 1)
        for bridge in secondary_bridges:
            full_sb = [bridge.start_bh] + bridge.atoms + [bridge.end_bh]
            for i in range(len(full_sb) - 1):
                a = full_sb[i]
                b = full_sb[i + 1]
                accounted_edges.add((min(a, b), max(a, b)))

        # Compute how many more secondary bridges are needed
        # IUPAC VB: ring_count rings require (ring_count + 1) total bridge lengths
        # in the descriptor. The first 3 are primary (branch1, branch2, main_bridge).
        # Secondary bridges needed = ring_count - 2.
        ring_count = self._get_ring_count(mol, ring_atoms)
        needed_more = (ring_count - 2) - len(secondary_bridges)

        if needed_more > 0:
            # Find bonds between ring atoms that are not accounted for
            zero_length_candidates = []
            for bond in mol.GetBonds():
                a_idx = bond.GetBeginAtomIdx()
                b_idx = bond.GetEndAtomIdx()
                if a_idx in ring_atoms and b_idx in ring_atoms:
                    edge = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if edge not in accounted_edges:
                        zero_length_candidates.append(edge)
                        accounted_edges.add(edge)  # Don't double count

            # Determinism (WS-D): RDKit GetBonds() iteration order is NOT
            # canonical even after RenumberAtoms, and on over-determined cages
            # (cubane, prismane, homocubane) there are more unaccounted ring
            # bonds than zero-length bridges needed, so the [:needed_more] slice
            # would otherwise pick a different subset per spelling. Sort by
            # (low, high) atom index so the chosen subset is order-independent.
            zero_length_candidates.sort(key=lambda e: (e[0], e[1]))

            # Add zero-length bridges for unaccounted ring bonds
            for a_idx, b_idx in zero_length_candidates[:needed_more]:
                bridge = BridgeInfo(
                    atoms=[],
                    length=0,
                    start_bh=a_idx,
                    end_bh=b_idx,
                    is_secondary=True,
                )
                secondary_bridges.append(bridge)

        # Bridge ordering is handled by _order_and_number_secondary_bridges()
        # in analyze(), which implements proper VB-7 independent/dependent
        # classification and multi-criteria sort.

        return secondary_bridges

    # ========================================================================
    # VB-7: Classification, Ordering, and Numbering
    # ========================================================================

    def _order_and_number_secondary_bridges(
        self,
        mol,
        secondary_bridges: List[BridgeInfo],
        main_ring: List[int],
        main_bridge: Optional[BridgeInfo],
        bh_pair: Tuple[int, int]
    ) -> Tuple[List[BridgeInfo], Dict[int, int]]:
        """
        VB-7: Classify, order, orient, and number all ring atoms.

        Implements a two-pass approach to resolve the circular dependency
        between locant assignment and bridge ordering:

        1. Number main ring and main bridge atoms (locants are fixed)
        2. Classify secondary bridges as independent/dependent (VB-5)
        3. Sort independent bridges by VB-7 criteria and number them
        4. Iteratively resolve and number dependent bridges
        5. Orient each bridge so numbering starts from higher-numbered
           bridgehead (VB-7: "numbered starting from atom next to
           higher-numbered bridgehead")

        Sort criteria match OPSIN's VonBaeyerSecondaryBridgeSort:
        (-locant_high, -locant_low, -bridge_length), which is consistent
        with IUPAC VB-7 ("beginning with the one attached to the
        highest-numbered bridgehead") and VB-7.2 ("longer bridges
        numbered before shorter bridges").

        Args:
            mol: RDKit Mol object
            secondary_bridges: Unordered list of secondary bridges
            main_ring: Ordered list of main ring atoms
            main_bridge: Main bridge info
            bh_pair: Main bridgehead pair

        Returns:
            Tuple of (secondary_bridges in numbering order, complete numbering dict)
        """
        numbering = {}
        locant = 1

        # Step 1: Number main ring atoms
        for atom_idx in main_ring:
            if atom_idx not in numbering:
                numbering[atom_idx] = locant
                locant += 1

        # Step 2: Number main bridge atoms
        if main_bridge and main_bridge.atoms:
            for atom_idx in main_bridge.atoms:
                if atom_idx not in numbering:
                    numbering[atom_idx] = locant
                    locant += 1

        if not secondary_bridges:
            return secondary_bridges, numbering

        # Step 3: Classify secondary bridges as independent or dependent
        # VB-5: Independent = both endpoints on main ring or main bridge
        #        Dependent = at least one endpoint on a secondary bridge
        main_ring_set = set(main_ring)
        main_bridge_atom_set = set(main_bridge.atoms) if main_bridge else set()
        assigned = main_ring_set | main_bridge_atom_set

        independent = []
        dependent = []
        for bridge in secondary_bridges:
            if bridge.start_bh in assigned and bridge.end_bh in assigned:
                bridge.is_dependent = False
                independent.append(bridge)
            else:
                bridge.is_dependent = True
                dependent.append(bridge)

        # Step 4: Sort independent bridges by VB-7 numbering criteria
        # VB-7: "beginning with the one attached to the highest-numbered bridgehead"
        # VB-7.2: "longer bridges numbered before shorter bridges"
        # Matches OPSIN: (-locant_high, -locant_low, -bridge_length)
        def _numbering_sort_key(bridge, numb):
            loc1 = numb.get(bridge.start_bh, 0)
            loc2 = numb.get(bridge.end_bh, 0)
            locant_high = max(loc1, loc2)
            locant_low = min(loc1, loc2)
            return (-locant_high, -locant_low, -bridge.length)

        independent.sort(key=lambda b: _numbering_sort_key(b, numbering))

        # Step 5: Orient and number independent bridge atoms
        ordered = []
        for bridge in independent:
            # VB-7: "numbered starting from atom next to higher-numbered bridgehead"
            if bridge.atoms:
                loc_start = numbering.get(bridge.start_bh, 0)
                loc_end = numbering.get(bridge.end_bh, 0)
                if loc_start < loc_end:
                    # end_bh has higher locant; reverse so atoms[0] is next to end_bh
                    bridge.atoms = list(reversed(bridge.atoms))

            for atom_idx in bridge.atoms:
                if atom_idx not in numbering:
                    numbering[atom_idx] = locant
                    locant += 1

            assigned.update(bridge.atoms)
            ordered.append(bridge)

        # Step 6: Iteratively resolve dependent bridges
        # After numbering independent bridges, some dependent bridges may
        # now have both endpoints assigned. Process them in rounds.
        remaining = list(dependent)
        while remaining:
            newly_resolvable = []
            still_remaining = []
            for b in remaining:
                if b.start_bh in assigned and b.end_bh in assigned:
                    newly_resolvable.append(b)
                else:
                    still_remaining.append(b)

            if not newly_resolvable:
                # Force-add remaining (shouldn't happen in valid polycyclics)
                logger.warning(
                    "Unresolvable dependent bridges remain: %d bridges",
                    len(still_remaining)
                )
                for bridge in still_remaining:
                    if bridge.atoms:
                        loc_start = numbering.get(bridge.start_bh, 0)
                        loc_end = numbering.get(bridge.end_bh, 0)
                        if loc_start < loc_end:
                            bridge.atoms = list(reversed(bridge.atoms))
                    for atom_idx in bridge.atoms:
                        if atom_idx not in numbering:
                            numbering[atom_idx] = locant
                            locant += 1
                    assigned.update(bridge.atoms)
                    ordered.append(bridge)
                break

            # Sort by VB-7 criteria with current numbering
            newly_resolvable.sort(key=lambda b: _numbering_sort_key(b, numbering))

            for bridge in newly_resolvable:
                if bridge.atoms:
                    loc_start = numbering.get(bridge.start_bh, 0)
                    loc_end = numbering.get(bridge.end_bh, 0)
                    if loc_start < loc_end:
                        bridge.atoms = list(reversed(bridge.atoms))

                for atom_idx in bridge.atoms:
                    if atom_idx not in numbering:
                        numbering[atom_idx] = locant
                        locant += 1

                assigned.update(bridge.atoms)
                ordered.append(bridge)

            remaining = still_remaining

        return ordered, numbering

    # ========================================================================
    # VB-7: Numbering (legacy — used by _order_and_number_secondary_bridges)
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

        Format (OPSIN-compatible):
        - bicyclo[a.b.c] (no secondary bridges)
        - tricyclo[a.b.c.d(e,f)] (one secondary bridge with locants in parentheses)
        - tetracyclo[a.b.c.d(e,f).g(h,i)] (two secondary bridges)

        Secondary bridge locants are written in parentheses after the bridge
        length for unambiguous parsing. OPSIN accepts this format.
        e.g. "1(3,7)" means bridge of length 1 with locants 3 and 7.

        Args:
            ring_count: Number of independent rings
            primary_lengths: [branch1_len, branch2_len, main_bridge_len], sorted descending
            secondary_bridges: List of secondary bridge infos
            numbering: Atom index -> VB locant mapping

        Returns:
            Descriptor string like "tricyclo[3.3.1.1(3,7)]"
        """
        prefix = cyclo_ring_count_word(ring_count)
        if prefix is None:
            # No ring-count word exists for this count -> refuse the whole
            # molecule rather than ship a non-word. Raising (not returning a
            # sentinel) is this module's established fail-closed idiom: the
            # limit is caught once at ``Orthonym.name``, so the molecule
            # reports 'unknown organic compound' instead of cascading into a
            # fragment namer that would name a single sub-ring.
            from ..errors import unsupported_ring_system
            logger.info(
                "no von Baeyer ring-count word for ring_count=%s; refuse",
                ring_count)
            raise unsupported_ring_system()

        # Primary bridge lengths (sorted descending)
        parts = [str(l) for l in sorted(primary_lengths, reverse=True)]

        # Secondary bridges with parenthesized locants (OPSIN-compatible)
        # Format: bridge_length(locant_low,locant_high)
        # e.g., "0(3,7)", "1(3,7)" -- unambiguous for multi-digit locants
        for bridge in secondary_bridges:
            ep1 = bridge.start_bh
            ep2 = bridge.end_bh
            loc1 = numbering.get(ep1)
            loc2 = numbering.get(ep2)
            # Skip bridges with unmapped atoms (invalid locants)
            if loc1 is None or loc2 is None or loc1 == 0 or loc2 == 0:
                logger.warning(
                    "Secondary bridge endpoint not in numbering dict: "
                    "ep1=%s(loc=%s), ep2=%s(loc=%s)", ep1, loc1, ep2, loc2
                )
                continue
            locant_low = min(loc1, loc2)
            locant_high = max(loc1, loc2)
            parts.append(self._format_secondary_locants(
                bridge.length, locant_low, locant_high))

        return f"{prefix}[{'.'.join(parts)}]"

    @staticmethod
    def _format_secondary_locants(length: int, locant_low: int, locant_high: int) -> str:
        """Format one secondary-bridge term in PIN superscript typography.

        ``length^low,high`` e.g. ``0^2,6`` (P-23.2.5.1 / P-23.2.6.1.2). OPSIN
        parses both this caret form and the older ``length(low,high)``
        parenthesis form, so the change is typography only -- not round-trip.
        """
        return f"{length}^{locant_low},{locant_high}"


# ============================================================================
# Ring Connectivity Helpers
# ============================================================================

def _get_largest_connected_ring_component(mol, ring_atoms: Set[int]) -> Set[int]:
    """
    Find the largest connected component of ring atoms connected by ring bonds.

    In molecules with multiple disconnected ring systems (e.g., a piperazine
    connected by a chain to a fused purine), this returns only the largest
    ring subsystem. This prevents the VB analyzer from treating disconnected
    ring systems as a single bridged polycyclic.

    Two ring atoms are "connected" if the bond between them is itself part
    of a ring (IsInRing() == True). Bonds between ring atoms that are NOT
    ring bonds (e.g., the biaryl bond in biphenyl) do not connect components.

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of all ring atom indices

    Returns:
        The largest connected component (set of atom indices).
        Returns the original set if all atoms are in one component.
    """
    if not ring_atoms:
        return ring_atoms

    # Build adjacency list using only ring bonds
    from collections import deque
    adj = {idx: [] for idx in ring_atoms}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            nbr_idx = bond.GetOtherAtomIdx(idx)
            if nbr_idx in ring_atoms and bond.IsInRing():
                adj[idx].append(nbr_idx)

    # BFS to find connected components
    visited = set()
    components = []
    for start in ring_atoms:
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    queue.append(nbr)
        components.append(component)

    if len(components) <= 1:
        return ring_atoms

    # Return the largest component
    return max(components, key=len)


def vonbaeyer_cage_has_aromaticity(mol, cage_atoms) -> bool:
    """G0 fail-closed safety (DD7 S1 — "fail closed, never hallucinate").

    Von Baeyer (P-23) and bicyclo nomenclature describe SATURATED bridged ring
    skeletons; unsaturation is expressible only as ``-ene``/``-yne`` with
    locants, and aromaticity CANNOT be represented at all. Naming a cage that
    contains aromatic ring atoms therefore silently DROPS the aromaticity and
    emits a structurally WRONG (de-aromatised) cage — e.g. benzonorbornadiene
    ``C1C2C=CC1c1ccccc12`` -> ``tricyclo[4.4.0.1(2,5)]undec-3-ene`` (the benzo
    ring desaturated). The correct PIN is a bridged-fused name (P-25.4, e.g.
    ``1,4-dihydro-1,4-methanonaphthalene``), a Phase-G1 build; until then the
    caller must fail closed (raise the limit -> ``unknown organic compound`` /
    ``OrthonymLimitError``) rather than emit the wrong saturated cage.

    ``cage_atoms`` MUST be the EXACT atom set the namer numbers — the von-Baeyer
    descriptor's ``numbering`` keys for ``name_polycyclic_complete``, or
    ``get_complete_bicyclo_data()['ring_atoms']`` for the bicyclo path — NOT the
    molecule's largest connected ring component. Keying off the actual cage
    means a PENDANT aromatic ring joined by a single (non-ring) bond — e.g. a
    naphthyl on norbornane, even when the naphthyl is LARGER than the cage — is
    never part of the cage and so can never trip the guard (it is correctly
    named as a substituent). (WR-01.)
    """
    return any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in cage_atoms)


# ============================================================================
# Public API
# ============================================================================

def generate_polycyclic_name(mol) -> Optional[str]:
    """
    Generate the base IUPAC name for a polycyclic bridged system.

    Returns "prefix[descriptor]parentname" (e.g., "tricyclo[3.3.1.1(3,7)]decane").
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

    # Filter to largest connected ring component
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)

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

    # Use total_atoms from VB analysis (sum(bridge_lengths) + 2) rather than
    # len(ring_atoms), which may miss non-ring atoms in the VB framework.
    total_ring_atoms = desc.total_atoms
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

    # Filter to largest connected ring component (via ring bonds only).
    # This prevents disconnected ring systems (e.g., piperazine + purine
    # connected by a chain) from being treated as one VB system.
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)

    # Skip fully aromatic ring systems (PAHs like naphthalene, perylene, coronene)
    # These should use retained names from fused_rings, not VB nomenclature.
    # EXEMPT (Wave-2 completion): a divalent O/S that is a genuine BRIDGE across
    # non-adjacent positions of another ring (1,4-epoxynaphthalene) — RDKit's
    # extended aromaticity marks it aromatic, but the system is bridged-fused
    # (P-25.4), not a plain fused aromatic; the VB path delegates it to
    # name_bridged_fused_pin. Fusion chalcogens (dibenzofuran) stay skipped.
    all_ring_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )
    if all_ring_aromatic:
        from .bridged_fused import (has_aromatic_chalcogen_bridge,
                                    has_aromatic_mancude_bridge)
        # Also exempt an all-carbon UNSATURATED bridge (P-25.4.2.1.1 etheno):
        # RDKit aromatizes the -CH=CH- bridge of 1,4-ethenonaphthalene, so the
        # system reads fully-aromatic even though it is bridged-fused, not a
        # plain PAH. name_bridged_fused_pin names it; a real PAH returns False.
        if (not has_aromatic_chalcogen_bridge(mol)
                and not has_aromatic_mancude_bridge(mol)):
            return False

    # Purely CATA-fused skip (Wave-2 completion, honours the documented
    # "Returns False for purely fused systems" contract): when every SSSR
    # ring-pair shares <=2 atoms AND no atom belongs to >=3 SSSR rings, the
    # system is ortho-fused (fluorene, 9,10-dihydroanthracene) — fusion
    # nomenclature territory, NOT a von Baeyer cage. Without this, sp3
    # positions defeated the all-aromatic skip and fluorene-9-carboxylic
    # acid RAISED UNSUPPORTED_RING_SYSTEM before the working PAH-substituent
    # path could fire. Bridged/peri cages keep True: adamantane (pair share
    # 3), cubane / acenaphthylene (ring membership 3).
    _sssr = [set(r) for r in ri.AtomRings()
             if set(r) <= ring_atoms]
    if len(_sssr) >= 2:
        _max_share = max(
            (len(a & b) for i, a in enumerate(_sssr)
             for b in _sssr[i + 1:]), default=0)
        _membership = {}
        for r in _sssr:
            for a in r:
                _membership[a] = _membership.get(a, 0) + 1
        if _max_share <= 2 and max(_membership.values(), default=0) <= 2:
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

# v29 Phase 2 T2a: the replacement-prefix table and its λ helper moved to
# ``rules/ring_replacement.py`` (the single source of truth, so extending the
# element table is a data-only change in one place). Re-exported here because the
# name ``polycyclic.HETEROATOM_PREFIXES`` is part of this module's surface.
from .ring_replacement import (  # noqa: E402,F401  (re-export)
    HETEROATOM_PREFIXES,
    build_replacement_prefix as _build_ring_replacement_prefix,
    vb_lambda_for_atom as _vb_lambda_for_atom,
)


def get_heteroatom_replacement_prefix(
    mol, numbering: Dict[int, int], ring_atoms: Set[int],
) -> Optional[str]:
    """
    Generate the 'a' replacement-nomenclature prefix for ring heteroatoms.

    Thin wrapper over ``rules/ring_replacement.build_replacement_prefix``, which
    owns the construction (element table, Table-2.8 citation order, λ tokens,
    multiplying prefixes, the no-trailing-hyphen rule of P-23.3.1). The string
    returned here is byte-identical to what this function built inline before
    v29 Phase 2 — the three PIN callers (``name_polycyclic_complete``,
    ``bicyclo.py``) see no change.

    Args:
        mol: RDKit Mol object
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)
        ring_atoms: Set of atom indices in the ring system

    Returns:
        Formatted prefix string (e.g. "7-oxa"); ``""`` when there is no
        heteroatom to express; or ``None`` when some skeletal atom CANNOT be
        expressed, in which case **the caller must refuse**.

    The ``None`` case is the fix for what this docstring used to concede: the old
    ``-> str`` signature could not report an atom it failed to express, so an
    off-table skeletal element was dropped from the name while the ring stem kept
    counting it. Its three PIN von Baeyer callers
    (``name_polycyclic_with_heteroatoms``, ``name_polycyclic_complete``,
    ``bicyclo.get_complete_bicyclo_data``) rested on an unverified assumption that
    such an element never reaches them; they now get the signal instead of the
    proof obligation. ``""`` and ``None`` are deliberately distinct -- ``if not
    prefix`` would conflate "carbocycle" with "refuse", so callers test
    ``is None``.
    """
    replacement = _build_ring_replacement_prefix(mol, numbering, ring_atoms)
    if replacement.unexpressed:
        return None
    return replacement.prefix


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
    if hetero_prefix is None:
        return None  # a skeletal atom is unexpressible -> never name the cage

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
    # Example: 7-oxabicyclo[2.2.1]heptane
    # Example: 3-oxabicyclo[3.2.1]octan-2-one
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
    numbering: Dict[int, int],
    exclude_atoms: Optional[Set[int]] = None
) -> List[Dict]:
    """
    Detect substituents attached to a polycyclic ring system.

    For each ring atom, checks neighbors not in ring_atoms. Traces each
    substituent branch and determines its name using get_alkyl_name().

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the polycyclic ring system
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)
        exclude_atoms: Optional set of non-ring atom indices to skip
                       (e.g., atoms that are part of functional groups)

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

            # Skip if neighbor is a known FG atom (e.g., COOH carbon, =O, -OH)
            if exclude_atoms and nbr_idx in exclude_atoms:
                continue

            # ---- exocyclic double bond ----
            # This used to be a blanket skip for ALL of them, justified by
            # "=O, =S are handled as suffixes". That premise holds only for the
            # chalcogens -- and the recognised ones have in any case already
            # been removed by the exclude_atoms check above, which carries
            # _detect_ring_functional_groups' fg_atoms. It is FALSE for carbon:
            # no suffix path exists for an exocyclic =CH2, so the atom simply
            # vanished from the name. A name that omits an atom is a wrong
            # STRUCTURE, and only the downstream OPSIN round-trip caught it,
            # which means with no JVM present the wrong name shipped.
            #
            # Carbon is therefore named here as the P-29.2 free-valence prefix
            # it is (methylidene / ethylidene / propan-2-ylidene / ...), and
            # the whole ring system fails closed when it cannot be. Non-carbon
            # keeps the existing skip: promoting an unrecognised ring C=O to an
            # 'oxo' prefix would ship a non-PIN name where the code correctly
            # abstains today, which is a different phase's decision to make.
            #
            # The verdict itself comes from the shared P-29.2 primitive rather
            # than being re-derived here: this file was the second place to
            # read an attachment bond order, and every further copy is another
            # detector that can drift.
            bond = mol.GetBondBetweenAtoms(ring_idx, nbr_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                if neighbor.GetAtomicNum() != 6:
                    continue
                from ..assembly.substituent_enumerator import (
                    carbon_free_valence_prefix)
                ylidene_atoms = _trace_substituent_branch(
                    mol, nbr_idx, ring_atoms)
                verdict = carbon_free_valence_prefix(
                    mol, ylidene_atoms, nbr_idx)
                if verdict.prefix is None:
                    from ..errors import unsupported_ring_system
                    logger.debug(
                        "exocyclic =C at locant %s: %s — failing closed",
                        locant, verdict.basis)
                    raise unsupported_ring_system()
                substituents.append({
                    'locant': locant,
                    'name': verdict.prefix,
                    'atom_indices': ylidene_atoms,
                })
                continue

            # Trace the substituent branch
            sub_atoms = _trace_substituent_branch(mol, nbr_idx, ring_atoms)

            # Count carbons and check for heteroatoms in the substituent
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Check for alkoxy substituent: starts with O, followed by alkyl
            # e.g., -O-CH3 (methoxy), -O-C2H5 (ethoxy)
            first_atom = mol.GetAtomWithIdx(nbr_idx)
            if (first_atom.GetSymbol() == 'O'
                    and carbon_count > 0
                    and len(sub_atoms) >= 2):
                # Check if the O is bonded to only C and the ring atom
                # (single bond to ring, single bond to alkyl)
                o_c_neighbors = [
                    n for n in first_atom.GetNeighbors()
                    if n.GetIdx() in set(sub_atoms) and n.GetSymbol() == 'C'
                ]
                if o_c_neighbors:
                    _ALKOXY_NAMES = {
                        1: 'methoxy', 2: 'ethoxy', 3: 'propoxy',
                        4: 'butoxy', 5: 'pentyloxy', 6: 'hexyloxy',
                    }
                    alkoxy_name = _ALKOXY_NAMES.get(carbon_count)
                    if alkoxy_name:
                        substituents.append({
                            'locant': locant,
                            'name': alkoxy_name,
                            'atom_indices': sub_atoms,
                        })
                        continue

            if carbon_count > 0:
                name = None
                # Try recursive naming for branched pure-alkyl subs
                all_c_h = all(
                    mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                    for i in sub_atoms
                )
                if all_c_h and len(sub_atoms) > 1:
                    from ..assembly.substituent_naming import name_substituent_fragment
                    name = name_substituent_fragment(
                        mol, sub_atoms, nbr_idx, list(ring_atoms)
                    )
                if name is None:
                    try:
                        name = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
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
# Functional Group Detection on Ring System (Plan 16-09)
# ============================================================================

# FG seniority for determining principal group on polycyclic rings.
# Higher index = higher seniority. Order follows IUPAC P-41 (the same relative
# order as seniority.SENIORITY_ORDER): carboxylic_acid > nitrile > aldehyde >
# ketone > alcohol > amine. WSD-01 (Phase 175) added 'nitrile' (between aldehyde
# and carboxylic_acid) and 'amine' (below alcohol); the pre-existing
# alcohol<ketone<aldehyde<carboxylic_acid relative order is preserved.
_FG_SENIORITY = {
    'amine': 1,
    'alcohol': 2,
    'ketone': 3,
    'aldehyde': 4,
    'nitrile': 5,
    'carboxylic_acid': 6,
}


def _detect_ring_functional_groups(
    mol,
    ring_atoms: Set[int],
    numbering: Dict[int, int]
) -> Dict:
    """
    Detect functional groups on a polycyclic ring system.

    Identifies:
    - Exocyclic C=O on ring carbons (ketone -> suffix -one)
    - -OH attached to ring carbons (alcohol -> suffix -ol or prefix hydroxy-)
    - -COOH attached to ring carbons (carboxylic acid -> suffix -carboxylic acid)
    - -CHO attached to ring carbons (aldehyde -> suffix -carbaldehyde or prefix formyl-)

    Applies seniority rules: highest-seniority FG becomes suffix, rest become prefixes.

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the polycyclic ring system
        numbering: Dict mapping atom_idx -> VB locant (1-indexed)

    Returns:
        Dict with:
        - 'suffix': dict with 'suffix', 'locants', 'type' or None
        - 'prefixes': list of dicts with 'name' and 'locant'
        - 'fg_atoms': set of non-ring atom indices that are part of FGs
                      (used to exclude them from substituent detection)
    """
    # Collect all detected FGs with their info
    detected_fgs = []  # list of dicts
    fg_atoms = set()  # non-ring atoms that are part of functional groups

    # --- 1. Detect exocyclic C=O on ring carbons (ketone) ---
    ketone_locants = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_atoms:
                continue
            if neighbor.GetSymbol() != 'O':
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                # Check this O is NOT an ester oxygen (i.e., O has no other heavy atom neighbors)
                o_heavy_neighbors = [
                    n for n in neighbor.GetNeighbors()
                    if n.GetIdx() != atom_idx
                ]
                if not o_heavy_neighbors:
                    locant = numbering.get(atom_idx, 0)
                    if locant > 0:
                        ketone_locants.append(locant)
                        fg_atoms.add(nbr_idx)  # Track the =O atom

    if ketone_locants:
        ketone_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['ketone'],
            'type': 'ketone',
            'locants': ketone_locants,
            'suffix': 'one',
            'suffix_type': 'inline',
            'prefix_name': 'oxo',
        })

    # --- 2. Detect -OH attached to ring carbons (alcohol) ---
    alcohol_locants = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_atoms:
                continue
            if neighbor.GetSymbol() != 'O':
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if bond and bond.GetBondTypeAsDouble() == 1.0:
                # Single bond O not in ring - check if it's a hydroxyl (has H)
                o_atom = neighbor
                # Check: O with exactly 1 H (hydroxyl), not ester O-C
                o_heavy_neighbors = [
                    n for n in o_atom.GetNeighbors()
                    if n.GetIdx() != atom_idx
                ]
                # If O has no other heavy neighbors, it's -OH (implicit H)
                if not o_heavy_neighbors:
                    locant = numbering.get(atom_idx, 0)
                    if locant > 0:
                        alcohol_locants.append(locant)
                        fg_atoms.add(nbr_idx)  # Track the -OH oxygen

    if alcohol_locants:
        alcohol_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['alcohol'],
            'type': 'alcohol',
            'locants': alcohol_locants,
            'suffix': 'ol',
            'suffix_type': 'inline',
            'prefix_name': 'hydroxy',
        })

    # --- 3. Detect -COOH attached to ring carbons (carboxylic acid) ---
    # Pattern: ring-C bonded to C(=O)(OH) where C is NOT in ring
    carbox_locants = []
    carbox_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if carbox_pattern is not None:
        matches = mol.GetSubstructMatches(carbox_pattern)
        for match in matches:
            c_idx = match[0]
            if c_idx in ring_atoms:
                continue  # Carbonyl C should NOT be in ring for -COOH suffix
            # Check if this C is bonded to a ring atom
            c_atom = mol.GetAtomWithIdx(c_idx)
            for neighbor in c_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in ring_atoms and nbr_idx in numbering:
                    locant = numbering[nbr_idx]
                    if locant not in carbox_locants:
                        carbox_locants.append(locant)
                        # Track all COOH atoms (C, =O, -OH)
                        fg_atoms.add(c_idx)
                        fg_atoms.add(match[1])  # =O
                        fg_atoms.add(match[2])  # -OH

    if carbox_locants:
        carbox_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['carboxylic_acid'],
            'type': 'carboxylic_acid',
            'locants': carbox_locants,
            'suffix': 'carboxylic acid',
            'suffix_type': 'appended',
            'prefix_name': 'carboxy',
        })

    # --- 4. Detect -CHO attached to ring carbons (aldehyde) ---
    # Pattern: ring-C bonded to C(=O)H where C is NOT in ring
    aldehyde_locants = []
    aldehyde_pattern = Chem.MolFromSmarts('[CX3H1](=O)')
    if aldehyde_pattern is not None:
        matches = mol.GetSubstructMatches(aldehyde_pattern)
        for match in matches:
            c_idx = match[0]
            if c_idx in ring_atoms:
                continue  # Aldehyde C should NOT be in ring
            c_atom = mol.GetAtomWithIdx(c_idx)
            for neighbor in c_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in ring_atoms and nbr_idx in numbering:
                    locant = numbering[nbr_idx]
                    if locant not in aldehyde_locants:
                        aldehyde_locants.append(locant)
                        # Track all CHO atoms (C, =O)
                        fg_atoms.add(c_idx)
                        fg_atoms.add(match[1])  # =O

    if aldehyde_locants:
        aldehyde_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['aldehyde'],
            'type': 'aldehyde',
            'locants': aldehyde_locants,
            'suffix': 'carbaldehyde',
            'suffix_type': 'appended',
            'prefix_name': 'formyl',
        })

    # --- 5. Detect -NH2 / -NHR attached to ring carbons (amine) [WSD-01/RING-09] ---
    # Mirrors the alcohol block: a single-bonded exocyclic N that is a genuine
    # primary/secondary amine (sp3, >=1 H, not an amide/imine/nitrile N).
    amine_locants = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_atoms:
                continue
            if neighbor.GetSymbol() != 'N':
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if not (bond and bond.GetBondTypeAsDouble() == 1.0):
                continue
            # primary/secondary amine N: has >=1 H and only single bonds
            if neighbor.GetTotalNumHs() < 1:
                continue
            if any(b.GetBondTypeAsDouble() != 1.0 for b in neighbor.GetBonds()):
                continue
            # exclude amide/imide N (a neighbor C bearing a double bond to O/S/N)
            is_amide = False
            for nn in neighbor.GetNeighbors():
                if nn.GetIdx() == atom_idx or nn.GetSymbol() != 'C':
                    continue
                if any(b.GetBondTypeAsDouble() == 2.0
                       and b.GetOtherAtom(nn).GetSymbol() in ('O', 'S', 'N')
                       for b in nn.GetBonds()):
                    is_amide = True
                    break
            if is_amide:
                continue
            locant = numbering.get(atom_idx, 0)
            if locant > 0:
                amine_locants.append(locant)
                fg_atoms.add(nbr_idx)  # track the amine N

    if amine_locants:
        amine_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['amine'],
            'type': 'amine',
            'locants': amine_locants,
            'suffix': 'amine',
            'suffix_type': 'inline',
            'prefix_name': 'amino',
        })

    # --- 6. Detect exocyclic -C#N on ring carbons (nitrile) [WSD-01] ---
    # Mirrors the aldehyde block: the nitrile carbon is exocyclic; the suffix
    # attaches to the ring carbon it is bonded to.
    nitrile_locants = []
    nitrile_pattern = Chem.MolFromSmarts('[CX2]#[NX1]')
    if nitrile_pattern is not None:
        for match in mol.GetSubstructMatches(nitrile_pattern):
            c_idx, n_idx = match[0], match[1]
            if c_idx in ring_atoms:
                continue  # nitrile C must NOT be in the ring
            c_atom = mol.GetAtomWithIdx(c_idx)
            for neighbor in c_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in ring_atoms and nbr_idx in numbering:
                    locant = numbering[nbr_idx]
                    if locant not in nitrile_locants:
                        nitrile_locants.append(locant)
                        fg_atoms.add(c_idx)  # nitrile C
                        fg_atoms.add(n_idx)  # nitrile N

    if nitrile_locants:
        nitrile_locants.sort()
        detected_fgs.append({
            'seniority': _FG_SENIORITY['nitrile'],
            'type': 'nitrile',
            'locants': nitrile_locants,
            'suffix': 'carbonitrile',
            'suffix_type': 'appended',
            'prefix_name': 'cyano',
        })

    # --- No FGs detected ---
    if not detected_fgs:
        return {'suffix': None, 'prefixes': [], 'fg_atoms': fg_atoms}

    # --- Determine principal group (highest seniority) ---
    detected_fgs.sort(key=lambda x: x['seniority'], reverse=True)
    principal = detected_fgs[0]

    # Build suffix info
    suffix_info = {
        'suffix': principal['suffix'],
        'locants': principal['locants'],
        'type': principal['suffix_type'],
    }

    # Build prefix list from non-principal FGs
    prefixes = []
    for fg in detected_fgs[1:]:
        for loc in fg['locants']:
            prefixes.append({
                'name': fg['prefix_name'],
                'locant': loc,
            })

    return {'suffix': suffix_info, 'prefixes': prefixes, 'fg_atoms': fg_atoms}


# ============================================================================
# Complete Name Assembly (Plan 16-03)
# ============================================================================

def name_polycyclic_complete(mol, features=None):
    """
    Generate the complete IUPAC name for a polycyclic bridged system.

    This is the FINAL public API for polycyclic naming. It supersedes
    generate_polycyclic_name() which only generates base names.

    Produces complete names including:
    - Stereodescriptors: "(1R,4S)-"
    - Substituent prefixes: "3-methyl-"
    - Functional group prefixes: "5-hydroxy-" (non-principal groups)
    - Heteroatom replacement: "7-oxa-" (when Plan 16-02 implemented)
    - Descriptor: "bicyclo[2.2.1]"
    - Parent name with unsaturation: "hept-2-ene"
    - Functional group suffix: "-2-one", "-1-ol", "-carboxylic acid"

    Example output: "(1R,4S)-5-hydroxy-3-methyl-7-oxabicyclo[2.2.1]heptan-2-one"

    Args:
        mol: RDKit Mol object
        features: Optional MolecularFeatures object for functional group detection.
                  If None, FG detection is performed directly via SMARTS.

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is True (polycyclic handler discovers
        substituents via get_polycyclic_substituents + _detect_ring_functional_groups),
        or None if not a polycyclic system.
    """
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key

    if mol is None:
        return None

    # Check retained names for polycyclic/tricyclic compounds FIRST.
    # Only applies to bare (unsubstituted, all-carbon) ring systems:
    # if every heavy atom is in a ring, we can use retained names directly.
    canonical = Chem.MolToSmiles(mol, canonical=True)
    ri = mol.GetRingInfo()
    ring_atoms_all = set()
    for ring in ri.AtomRings():
        ring_atoms_all.update(ring)
    all_heavy = {a.GetIdx() for a in mol.GetAtoms()}
    if all_heavy == ring_atoms_all:
        # Pure ring system -- check retained name lookups
        from ..data.bicyclo_systems import get_retained_bicyclo_name
        retained = get_retained_bicyclo_name(canonical)
        if not retained:
            from .tricyclo import get_retained_tricyclo_name
            retained = get_retained_tricyclo_name(canonical)
        if retained:
            atom_to_locant = {idx: idx + 1 for idx in sorted(ring_atoms_all)}
            return (retained, ring_atoms_all, atom_to_locant, True)

    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    if not ring_atoms:
        return None

    # Filter to largest connected ring component (via ring bonds only).
    # Prevents disconnected ring systems from inflating atom counts.
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)

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

    # G0 fail-closed safety (DD7 S1): von Baeyer cannot represent aromaticity.
    # Check the EXACT cage atoms the descriptor numbers (WR-01: NOT the largest
    # ring component — a pendant aromatic ring never enters desc.numbering). If
    # the cage carries an aromatic atom, naming it here would silently
    # de-aromatise it into a WRONG saturated cage; refuse instead (raise the
    # named limit, caught at Orthonym.name). Raising (not returning None) is
    # required so the molecule fails closed rather than cascading to a fragment
    # namer that would name a single sub-ring ('cyclopentene' for benzonorbornadiene).
    # v22 Phase G1 (DD7 COV-01): a fused-aromatic core + bridge
    # (benzonorbornadiene-type) has a CORRECT bridged-fused PIN
    # (1,4-dihydro-1,4-methanonaphthalene); von Baeyer would de-aromatise it.
    # Try the P-25.4 constructor whenever the cage carries RDKit aromaticity OR
    # is a MANCUDE bridged cage whose aromaticity RDKit fails to perceive
    # (Wave-2 P5: 1,4:5,8-dimethanonaphthalene — the two bridges warp the
    # naphthalene out of plane, so RDKit marks nothing aromatic and von Baeyer
    # would emit a WRONG -polyene cage). name_bridged_fused_pin returns a name
    # only for the class it can name correctly (kekulizing + validating its own
    # aromatic residual), and None otherwise -> we then fail closed (G0).
    from .bridged_fused import name_bridged_fused_pin
    bridged = name_bridged_fused_pin(mol)
    if bridged is not None:
        return bridged
    if vonbaeyer_cage_has_aromaticity(mol, desc.numbering):
        # v23 IH-01 (Phase 2): a partially-saturated PAH (e.g. 9,10-dihydro-
        # anthracene) reaches this von-Baeyer path because its cage still
        # carries the intact aromatic ring(s); von Baeyer would de-aromatise
        # it into a WRONG saturated cage. The carbocyclic partial-saturation
        # constructor names it correctly (hydro prefix on the mancude parent,
        # automorphism-min numbering) and fails closed otherwise — try it
        # before refusing, exactly as the bridged-fused delegation above.
        from .partial_saturation import detect_carbocyclic_partial_saturation
        _sat_ring_atoms = set()
        for _r in ri.AtomRings():
            _sat_ring_atoms.update(_r)
        _sat = detect_carbocyclic_partial_saturation(mol, _sat_ring_atoms)
        if _sat is not None:
            from .polycyclics import _assemble_partially_saturated_carbocycle_name
            _sat_name = _assemble_partially_saturated_carbocycle_name(mol, _sat)
            if _sat_name:
                # substituents_included=False -> the complex-ring caller enriches
                # any ring substituents via the stored atom_to_locant.
                return (_sat_name, _sat_ring_atoms, _sat['atom_to_locant'], False)
        # IN-01: no smiles arg — Orthonym.name back-fills the original input SMILES.
        from ..errors import unsupported_ring_system
        raise unsupported_ring_system()

    # Use total_atoms from VB analysis (sum(bridge_lengths) + 2) rather than
    # len(ring_atoms), which may miss non-ring atoms in the VB framework.
    total_ring_atoms = desc.total_atoms

    # 1. Get stereodescriptors
    stereo_prefix = get_polycyclic_stereo(mol, desc.numbering)

    # 2. Detect functional groups FIRST (needed to exclude FG atoms from substituents)
    fg_info = _detect_ring_functional_groups(mol, ring_atoms, desc.numbering)
    fg_atoms = fg_info.get('fg_atoms', set())

    # 3. Get substituents (excluding FG atoms like COOH carbons, =O oxygens)
    substituents = get_polycyclic_substituents(
        mol, ring_atoms, desc.numbering, exclude_atoms=fg_atoms
    )

    # 4. Get unsaturation
    unsaturation = get_polycyclic_unsaturation(mol, ring_atoms, desc.numbering)

    # 5. Get heteroatom replacement prefix
    hetero_prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)
    if hetero_prefix is None:
        return None  # a skeletal atom is unexpressible -> never name the cage

    # 6. Assemble substituent prefix (alkyl + FG prefixes combined)
    fg_prefixes = fg_info.get('prefixes', [])
    substituent_prefix = _assemble_substituent_prefix(substituents, fg_prefixes)

    # 7. Build parent name with unsaturation and FG suffix
    fg_suffix_info = fg_info.get('suffix', None)
    parent_name = _build_parent_with_unsaturation(
        total_ring_atoms, unsaturation, fg_suffix=fg_suffix_info
    )

    # 8. Assemble final name
    # Order: (stereo)-substituents-heteroprefix-cycloprefix[descriptor]parent-suffix
    name_parts = []

    if stereo_prefix:
        name_parts.append(stereo_prefix)

    if substituent_prefix:
        # S1 (v24) — same stray-hyphen class: a substituent prefix attaches DIRECTLY
        # to the parent hydride (P-31/P-23). Keep its trailing '-' ONLY when a
        # locant-initial heteroatom replacement prefix follows (3,3-dimethyl-2-oxabicyclo…);
        # drop it when the letter-initial descriptor follows directly, else we emit
        # `1-methyl-tricyclo…` instead of the PIN `1-methyltricyclo…`.
        following = hetero_prefix or desc.descriptor_string
        if following[:1].isdigit():
            name_parts.append(substituent_prefix)
        else:
            name_parts.append(substituent_prefix.rstrip('-'))

    if hetero_prefix:
        name_parts.append(hetero_prefix)

    # Add descriptor and parent
    name_parts.append(desc.descriptor_string)
    name_parts.append(parent_name)

    # Join - the stereo prefix ends with '-'; the substituent prefix's trailing '-'
    # is handled above (kept before a locant, dropped before the descriptor).
    name = ''.join(name_parts)

    return (name, ring_atoms, desc.numbering, True)


def _assemble_substituent_prefix(
    substituents: List[Dict],
    fg_prefixes: Optional[List[Dict]] = None
) -> str:
    """
    Assemble substituent prefix with proper IUPAC formatting.

    Groups identical substituents, applies multipliers, and sorts alphabetically.
    Includes functional group prefixes (hydroxy-, oxo-, etc.) when present.

    Args:
        substituents: List of substituent dicts from get_polycyclic_substituents()
        fg_prefixes: Optional list of FG prefix dicts with 'name' and 'locant' keys

    Returns:
        Formatted prefix like "5-hydroxy-3-ethyl-2,4-dimethyl-" or empty string
    """
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key

    if not substituents and not fg_prefixes:
        return ""

    # Group by name
    grouped: Dict[str, List[int]] = {}

    # Add alkyl substituents
    for sub in substituents:
        name = sub['name']
        locant = sub['locant']
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(locant)

    # Add functional group prefixes
    if fg_prefixes:
        for fg in fg_prefixes:
            name = fg['name']
            locant = fg['locant']
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
    unsaturation: Dict,
    fg_suffix: Optional[Dict] = None
) -> str:
    """
    Build the parent name with unsaturation suffixes, vowel elision,
    and optional functional group suffix.

    Args:
        total_atoms: Total ring atoms (for parent name base)
        unsaturation: Dict with 'double_bonds' and 'triple_bonds' lists
        fg_suffix: Optional dict with FG suffix info:
            'suffix': suffix string (e.g., 'one', 'ol', 'carboxylic acid')
            'locants': list of VB locants
            'type': 'inline' (modifies parent ending) or 'appended' (added after parent)

    Returns:
        Parent name like "heptane", "hept-2-ene", "decan-2-one",
        "decane-1-carboxylic acid"
    """
    base_name = _get_alkane_name(total_atoms)

    # Multipliers for unsaturation and functional group counts
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    _mult = {1: ''}
    _mult.update(SIMPLE_MULTIPLIERS)
    unsat_multipliers = _mult
    fg_multipliers = _mult

    double_bonds = unsaturation.get('double_bonds', [])
    triple_bonds = unsaturation.get('triple_bonds', [])
    has_unsaturation = bool(double_bonds) or bool(triple_bonds)

    # --- No FG suffix: existing behavior ---
    if fg_suffix is None:
        if not has_unsaturation:
            return base_name

        stem = base_name[:-3] if base_name.endswith('ane') else base_name[:-1]
        parts = []

        if double_bonds:
            count = len(double_bonds)
            mult = unsat_multipliers.get(count, str(count))
            locant_str = ','.join(str(loc) for loc in double_bonds)
            if count > 1:
                parts.append(f"-{locant_str}-{mult}en")
            else:
                parts.append(f"-{locant_str}-en")

        if triple_bonds:
            count = len(triple_bonds)
            mult = unsat_multipliers.get(count, str(count))
            locant_str = ','.join(str(loc) for loc in triple_bonds)
            if count > 1:
                parts.append(f"-{locant_str}-{mult}yn")
            else:
                parts.append(f"-{locant_str}-yn")

        if len(double_bonds) > 1 or (double_bonds and triple_bonds):
            stem = stem + 'a'

        result = stem + ''.join(parts) + 'e'
        return result

    # --- FG suffix present ---
    suffix_text = fg_suffix['suffix']
    suffix_locants = fg_suffix.get('locants', [])
    suffix_type = fg_suffix.get('type', 'inline')

    # For 'appended' type (carboxylic acid, carbaldehyde), the suffix is appended
    # to the full parent name with a hyphen
    if suffix_type == 'appended':
        # Build base name with unsaturation first
        if has_unsaturation:
            stem = base_name[:-3] if base_name.endswith('ane') else base_name[:-1]
            parts = []
            if double_bonds:
                count = len(double_bonds)
                mult = unsat_multipliers.get(count, str(count))
                locant_str = ','.join(str(loc) for loc in double_bonds)
                if count > 1:
                    parts.append(f"-{locant_str}-{mult}en")
                else:
                    parts.append(f"-{locant_str}-en")
            if triple_bonds:
                count = len(triple_bonds)
                mult = unsat_multipliers.get(count, str(count))
                locant_str = ','.join(str(loc) for loc in triple_bonds)
                if count > 1:
                    parts.append(f"-{locant_str}-{mult}yn")
                else:
                    parts.append(f"-{locant_str}-yn")
            if len(double_bonds) > 1 or (double_bonds and triple_bonds):
                stem = stem + 'a'
            parent_base = stem + ''.join(parts) + 'e'
        else:
            parent_base = base_name

        # Append suffix: "decane-1-carboxylic acid"
        # P-63.1.2: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol);
        # _join_multiplied_suffix is a no-op for 'carb...'/one/amine suffixes.
        from ..assembly.naming_utils import _join_multiplied_suffix
        count = len(suffix_locants)
        mult = fg_multipliers.get(count, str(count))
        full_suffix = _join_multiplied_suffix(mult, suffix_text)
        if suffix_locants:
            locant_str = ','.join(str(loc) for loc in suffix_locants)
            return f"{parent_base}-{locant_str}-{full_suffix}"
        else:
            return f"{parent_base}-{full_suffix}"

    # For 'inline' type (one, ol, amine), replace the terminal 'e' or 'ane'
    # Examples: decane -> decan-2-one, decane -> decan-1-ol
    # With unsaturation: dec-5-en-2-one

    # Get the stem (remove 'ane' or 'e')
    stem = base_name[:-3] if base_name.endswith('ane') else base_name[:-1]

    # Build unsaturation part
    unsat_parts = []
    if double_bonds:
        count = len(double_bonds)
        mult = unsat_multipliers.get(count, str(count))
        locant_str = ','.join(str(loc) for loc in double_bonds)
        if count > 1:
            unsat_parts.append(f"-{locant_str}-{mult}en")
        else:
            unsat_parts.append(f"-{locant_str}-en")

    if triple_bonds:
        count = len(triple_bonds)
        mult = unsat_multipliers.get(count, str(count))
        locant_str = ','.join(str(loc) for loc in triple_bonds)
        if count > 1:
            unsat_parts.append(f"-{locant_str}-{mult}yn")
        else:
            unsat_parts.append(f"-{locant_str}-yn")

    if len(double_bonds) > 1 or (double_bonds and triple_bonds):
        stem = stem + 'a'

    # Build suffix part with locants. P-63.1.2: elide multiplier-final 'a' before
    # '-ol' (tetra+ol -> tetrol) via the shared helper (no-op for one/amine).
    from ..assembly.naming_utils import _join_multiplied_suffix
    fg_count = len(suffix_locants)
    fg_mult = fg_multipliers.get(fg_count, str(fg_count))
    fg_full = _join_multiplied_suffix(fg_mult, suffix_text)

    if suffix_locants:
        locant_str = ','.join(str(loc) for loc in suffix_locants)
        suffix_part = f"-{locant_str}-{fg_full}"
    else:
        suffix_part = f"-{fg_full}"

    # Vowel elision: check if suffix starts with a vowel
    # If so, drop trailing 'e' from unsaturation or 'an' stem
    # e.g., decan + -2-one -> decan-2-one (not decane-2-one)
    # but decan + -2-ol -> decan-2-ol (not decane-2-ol)
    # The 'an' ending already has no 'e', so just add the suffix

    if has_unsaturation:
        # With unsaturation: stem + unsat_parts + suffix_part
        # e.g., dec-5-en-2-one
        result = stem + ''.join(unsat_parts) + suffix_part
    else:
        # Saturated: stem + 'an' + suffix_part
        # Vowel elision: before vowel suffix, keep 'an' (decan-2-one)
        # Before consonant suffix, add 'e' (decane-...)
        # Common suffixes starting with vowel: one, ol, amine
        # Common suffixes starting with consonant: thiol, thione
        first_char_of_suffix = suffix_text[0] if suffix_text else ''
        if first_char_of_suffix in ('a', 'e', 'i', 'o', 'u'):
            result = stem + 'an' + suffix_part
        else:
            result = stem + 'ane' + suffix_part

    return result
