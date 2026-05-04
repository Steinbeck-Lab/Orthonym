"""Cyclic phane parent hydride nomenclature (IUPAC P-26.4).

Detects molecules whose topology fits the IUPAC P-26.4 "cyclic phane"
class (two or more disjoint small rings linked by acyclic chain
segments of length >= 2 atoms whose linkage closes a macrocyclic
ring; mutually exclusive with ring-assembly, spiro/fused/bridged-
fused, and multiplicative cases).

Public API:

* ``is_cyclophane(mol) -> bool``                           -- topology gate
* ``name_cyclophane(mol) -> Optional[str]``                -- top-level handler
* ``_classify_phane_topology(mol) -> PhaneTopology``       -- sub-class enum
* ``_build_composite_locant(ring_idx, ring_locant, style)``-- composite-locant emitter
* ``_enumerate_inter_ring_chains(mol, small_rings)``       -- BFS chain walker
* ``PhaneTopology``                                         -- enum

Source:

* 155-CONTEXT.md D-03 (mutual-exclusion topology gate; corrected SSSR
  criterion per 155-AUDIT-A.md Critical Finding 0).
* 155-CONTEXT.md D-04 (sub-class enum {PARACYCLOPHANE, METACYCLOPHANE,
  ORTHOCYCLOPHANE, GENERIC_CYCLOPHANE}).
* 155-CONTEXT.md D-05 (composite-locant dual rendering: ASCII default,
  Unicode superscript option).
* 155-CONTEXT.md D-16 (mutual-exclusion contract enforced at the topology
  gate; Phase 154 D-11 pattern -- canonical helper reuse via
  ``multiplicative._is_pure_single_bond_assembly``).
* 155-CONTEXT.md D-20 (root-cause-only; no postprocessor band-aids).
* 155-AUDIT-A.md Critical Finding 0 (D-03 wording correction: SSSR-based
  criterion replaces the merged-ring-system phrasing because cyclophane
  macrocycles share atoms with both small rings, merging into a single
  ring system via ``perception.rings.get_ring_systems``).
* IUPAC 2013 Blue Book P-26.4 "Cyclic Phane Parent Hydrides".
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem


# ---------------------------------------------------------------------------
# Sub-class enum (D-04)
# ---------------------------------------------------------------------------


class PhaneTopology(Enum):
    """Cyclophane sub-class per IUPAC P-26.4.

    Source: 155-CONTEXT.md D-04; 155-AUDIT-A.md §4.
    """

    PARACYCLOPHANE = "paracyclophane"
    METACYCLOPHANE = "metacyclophane"
    ORTHOCYCLOPHANE = "orthocyclophane"
    GENERIC_CYCLOPHANE = "generic"


# Maps PhaneTopology -> base name fragment used in name composition.
# Source: 155-AUDIT-A.md §4.
_PHANE_BASE_NAMES: Dict[PhaneTopology, str] = {
    PhaneTopology.PARACYCLOPHANE: "paracyclophane",
    PhaneTopology.METACYCLOPHANE: "metacyclophane",
    PhaneTopology.ORTHOCYCLOPHANE: "orthocyclophane",
    PhaneTopology.GENERIC_CYCLOPHANE: "cyclophane",
}


# Small-ring vs macrocycle threshold for the SSSR-based topology gate.
# A "small" ring (linker / aromatic component) has size <= 8; the macrocyclic
# bridging ring has size > 8. Empirical default; works for [2.2]/[2.2.2]/etc
# paracyclophane through [4]paracyclophane (which is a fused cyclobutyl-benzene,
# correctly rejected by the disjointness check). Source: 155-AUDIT-A.md §1.
_SMALL_RING_MAX_SIZE = 8


# ---------------------------------------------------------------------------
# Public API: is_cyclophane (D-03 corrected SSSR criterion)
# ---------------------------------------------------------------------------


def is_cyclophane(mol: Optional[Chem.Mol]) -> bool:
    """Return True iff ``mol`` matches the IUPAC P-26.4 cyclophane topology.

    Topology gate (corrected per 155-AUDIT-A.md Critical Finding 0):

    1. >= 2 disjoint SSSR rings of size <= 8 (small / linker rings).
    2. >= 1 macrocyclic SSSR ring of size > 8.
    3. Shortest atom-disjoint chain between two small rings has
       >= 2 intermediate atoms (P-26.4 minimum bridge length).
    4. The chain shares atoms with at least one macrocyclic SSSR ring
       (i.e., the linkage closes a cycle, not an acyclic substituent).
    5. Mutual exclusion: NOT pure single-bond ring assembly (Phase 151
       territory; canonical helper reuse via
       ``multiplicative._is_pure_single_bond_assembly``).

    Returns False for None input.

    Source: 155-CONTEXT.md D-03 + D-16; 155-AUDIT-A.md Critical Finding 0.
    """
    if mol is None:
        return False

    Chem.GetSSSR(mol)
    sssr_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if len(sssr_rings) < 2:
        return False

    # Total ring nodes (gate (b) in spirit -- D-03 originally said "ring atoms";
    # use the union of SSSR rings as the corrected analogue).
    all_ring_atoms: Set[int] = set()
    for r in sssr_rings:
        all_ring_atoms.update(r)
    if len(all_ring_atoms) < 6:
        return False

    # Mutual-exclusion (d.1): pure single-bond ring assembly = Phase 151
    # territory. Canonical helper reuse (D-16 -- avoid duplication).
    from .multiplicative import _is_pure_single_bond_assembly

    if _is_pure_single_bond_assembly(mol):
        return False

    # Mutual-exclusion (d.2): spiro = handled by the existing spiro pipeline.
    # A spiro junction is a single atom shared by 2 rings only; cyclophanes
    # never have spiro atoms by construction (the macrocycle shares >= 2 atoms
    # with each linker ring), so we use perception.rings.get_spiro_atoms as a
    # conservative reject when a spiro junction sits inside any linker ring.
    from ..perception.rings import get_spiro_atoms

    spiro_atoms = get_spiro_atoms(mol)
    if spiro_atoms & all_ring_atoms:
        return False

    # Partition SSSR into small rings (size <= 8, the linker rings) and
    # macrocyclic rings (size > 8, the bridging cycles).
    small_rings: List[Set[int]] = [r for r in sssr_rings if len(r) <= _SMALL_RING_MAX_SIZE]
    macro_rings: List[Set[int]] = [r for r in sssr_rings if len(r) > _SMALL_RING_MAX_SIZE]

    # (a) corrected: need >= 2 small rings and >= 1 macro ring.
    if len(small_rings) < 2 or not macro_rings:
        return False

    # Find at least one disjoint pair of small rings whose linker chain has
    # >= 2 intermediate atoms AND the chain belongs to a macro ring.
    for i in range(len(small_rings)):
        for j in range(i + 1, len(small_rings)):
            ring_i = small_rings[i]
            ring_j = small_rings[j]
            if ring_i & ring_j:
                # Disjointness check (rejects fused / spiro / bridged systems
                # that already merge SSSR rings).
                continue
            chain_atoms = _shortest_chain_atoms(mol, ring_i, ring_j)
            if chain_atoms is None or len(chain_atoms) < 2:
                # No path or path < 2 intermediate atoms (1-atom bridge =
                # multiplicative territory; 0-atom = ring_assembly).
                continue
            # (4) chain belongs to a macro ring => closes a macrocycle.
            for macro in macro_rings:
                if chain_atoms <= macro:
                    return True
                if chain_atoms & macro:
                    # Partial overlap is enough -- the chain is part of a
                    # macrocyclic ring (the macro ring may include linker-ring
                    # atoms at its boundary).
                    return True
    return False


# ---------------------------------------------------------------------------
# _enumerate_inter_ring_chains (helper)
# ---------------------------------------------------------------------------


def _enumerate_inter_ring_chains(
    mol: Chem.Mol,
    small_rings: List[Set[int]],
) -> List[List[int]]:
    """Return chain atom-index lists for every disjoint pair of small rings.

    A chain = ordered intermediate atoms (NOT in either endpoint ring) on the
    shortest path between two disjoint small rings. Empty list if no chains.

    Used by ``_classify_phane_topology`` to determine ring-attachment positions.

    Source: 155-CONTEXT.md D-03 (graph-topology classification);
    155-RESEARCH.md Code Examples §1.
    """
    chains: List[List[int]] = []
    for i in range(len(small_rings)):
        for j in range(i + 1, len(small_rings)):
            ring_i = small_rings[i]
            ring_j = small_rings[j]
            if ring_i & ring_j:
                continue
            chain = _shortest_chain_path(mol, ring_i, ring_j)
            if chain is not None:
                chains.append(chain)
    return chains


def _shortest_chain_atoms(
    mol: Chem.Mol,
    ring_a: Set[int],
    ring_b: Set[int],
) -> Optional[Set[int]]:
    """Return atoms on the shortest path from ring_a to ring_b, excluding
    the endpoint rings themselves. None if no path exists.
    """
    chain_path = _shortest_chain_path(mol, ring_a, ring_b)
    if chain_path is None:
        return None
    return set(chain_path)


def _shortest_chain_path(
    mol: Chem.Mol,
    ring_a: Set[int],
    ring_b: Set[int],
) -> Optional[List[int]]:
    """BFS-shortest path of intermediate atoms between ring_a and ring_b.

    Returns the path (excluding ring_a and ring_b atoms) as a list ordered
    from the ring_a entry-point neighbour to the atom adjacent to ring_b.
    None if no path exists.
    """
    parent: Dict[int, Optional[int]] = {}
    visited: Set[int] = set()
    queue: deque = deque()
    # Seed: every neighbor of ring_a that's NOT in ring_a.
    for a in ring_a:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_a:
                continue
            if ni in ring_b:
                # Direct ring_a -> ring_b bond = ring assembly territory; no
                # intermediate atoms on the chain. Return None (not []) so
                # callers that distinguish "no chain found" from "empty chain
                # found" treat direct adjacency as a non-cyclophane signal
                # (matches _shortest_chain_path_via_anchor's contract;
                # 155-REVIEW.md WR-02).
                return None
            if ni not in visited:
                visited.add(ni)
                parent[ni] = None  # sentinel: came from ring_a
                queue.append(ni)
    while queue:
        cur = queue.popleft()
        atom = mol.GetAtomWithIdx(cur)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_a:
                continue
            if ni in ring_b:
                # Reconstruct path
                path: List[int] = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            if ni not in visited:
                visited.add(ni)
                parent[ni] = cur
                queue.append(ni)
    return None


# ---------------------------------------------------------------------------
# _classify_phane_topology (D-04)
# ---------------------------------------------------------------------------


def _classify_phane_topology(mol: Chem.Mol) -> PhaneTopology:
    """Classify cyclophane sub-class per the inter-ring chain attachment positions.

    For each inter-ring chain, identify the two ring atoms it attaches to in
    each linker ring; compute their relative ring positions (1,2 / 1,3 / 1,4)
    on six-membered aromatic linker rings.

    * All attachments at 1,4 positions on aromatic linker rings -> PARACYCLOPHANE
    * All at 1,3 -> METACYCLOPHANE
    * All at 1,2 -> ORTHOCYCLOPHANE
    * Otherwise (mixed, non-aromatic, or non-six-membered linkers) -> GENERIC_CYCLOPHANE

    Source: 155-CONTEXT.md D-04; 155-AUDIT-A.md §4.
    """
    if mol is None:
        return PhaneTopology.GENERIC_CYCLOPHANE

    Chem.GetSSSR(mol)
    sssr_rings = [list(r) for r in mol.GetRingInfo().AtomRings()]
    small_rings: List[Set[int]] = [
        set(r) for r in sssr_rings if len(r) <= _SMALL_RING_MAX_SIZE
    ]

    # All linker rings must be carbocyclic-aromatic six-membered for a strict
    # para/meta/ortho classification. Else GENERIC.
    if not _all_linkers_are_carbocyclic_benzene(mol, small_rings):
        return PhaneTopology.GENERIC_CYCLOPHANE

    # Build a ring-position map: atom_idx -> (ring_index, position_in_ring 0-based).
    ring_position: Dict[Tuple[int, int], int] = {}
    for ring_idx, ring in enumerate(sssr_rings):
        if len(ring) <= _SMALL_RING_MAX_SIZE:
            ordered = _ordered_ring_atoms(mol, ring)
            for pos, atom_idx in enumerate(ordered):
                ring_position[(ring_idx, atom_idx)] = pos

    # For each disjoint small-ring pair, find every inter-ring chain and the
    # two endpoints' positions.
    relative_positions: List[int] = []
    small_ring_indices = [
        i for i, r in enumerate(sssr_rings) if len(r) <= _SMALL_RING_MAX_SIZE
    ]
    for ai_idx, ai in enumerate(small_ring_indices):
        for bi in small_ring_indices[ai_idx + 1 :]:
            ring_a = set(sssr_rings[ai])
            ring_b = set(sssr_rings[bi])
            if ring_a & ring_b:
                continue
            anchors = _all_chain_anchor_pairs(mol, ring_a, ring_b)
            for anchor_a, anchor_b in anchors:
                rel_a = _ring_distance(
                    sssr_rings[ai], anchor_a, _find_other_anchor_on_ring(
                        mol, ring_a, anchor_a, ring_b
                    )
                )
                rel_b = _ring_distance(
                    sssr_rings[bi], anchor_b, _find_other_anchor_on_ring(
                        mol, ring_b, anchor_b, ring_a
                    )
                )
                if rel_a is not None:
                    relative_positions.append(rel_a)
                if rel_b is not None:
                    relative_positions.append(rel_b)

    if not relative_positions:
        return PhaneTopology.GENERIC_CYCLOPHANE
    unique = set(relative_positions)
    if unique == {3}:  # 1,4 distance = 3 bonds apart on a 6-ring
        return PhaneTopology.PARACYCLOPHANE
    if unique == {2}:  # 1,3 distance
        return PhaneTopology.METACYCLOPHANE
    if unique == {1}:  # 1,2 distance
        return PhaneTopology.ORTHOCYCLOPHANE
    return PhaneTopology.GENERIC_CYCLOPHANE


def _all_linkers_are_carbocyclic_benzene(
    mol: Chem.Mol,
    small_rings: List[Set[int]],
) -> bool:
    """Return True iff every small ring is a 6-membered aromatic carbocycle."""
    for ring in small_rings:
        if len(ring) != 6:
            return False
        for a in ring:
            atom = mol.GetAtomWithIdx(a)
            if atom.GetSymbol() != "C":
                return False
            if not atom.GetIsAromatic():
                return False
    return True


def _ordered_ring_atoms(mol: Chem.Mol, ring: List[int]) -> List[int]:
    """Return ring atoms ordered along the cyclic walk (RDKit-canonical for SSSR)."""
    # AtomRings() ordering is already cyclic per RDKit convention; preserve it.
    return list(ring)


def _all_chain_anchor_pairs(
    mol: Chem.Mol,
    ring_a: Set[int],
    ring_b: Set[int],
) -> List[Tuple[int, int]]:
    """Return all (anchor_a_idx, anchor_b_idx) pairs where an inter-ring chain
    actually crosses between ring_a and ring_b. Multiple chains => multiple pairs.

    An anchor is a ring atom with a non-ring (i.e., chain) neighbour that BFS-
    reaches the other ring without re-entering the source ring.
    """
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for a in ring_a:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_a:
                continue
            # BFS from ni avoiding ring_a atoms; if reaches ring_b, record anchor pair.
            if ni in ring_b:
                continue  # direct bond = ring assembly; not a chain
            terminal_b = _bfs_to_ring(mol, ni, ring_a, ring_b)
            if terminal_b is not None:
                pair = (a, terminal_b)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return pairs


def _bfs_to_ring(
    mol: Chem.Mol,
    start: int,
    forbidden: Set[int],
    target: Set[int],
) -> Optional[int]:
    """BFS from ``start`` (NOT a member of ``forbidden``) seeking any atom in
    ``target``; cannot re-enter ``forbidden``. Returns the target atom reached
    or None.
    """
    visited: Set[int] = {start}
    queue: deque = deque([start])
    while queue:
        cur = queue.popleft()
        atom = mol.GetAtomWithIdx(cur)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in forbidden:
                continue
            if ni in target:
                return ni
            if ni in visited:
                continue
            visited.add(ni)
            queue.append(ni)
    return None


def _find_other_anchor_on_ring(
    mol: Chem.Mol,
    ring: Set[int],
    anchor: int,
    other_ring: Set[int],
) -> Optional[int]:
    """Find a second anchor atom on ``ring`` whose chain reaches ``other_ring``
    via a different chain than the one anchored at ``anchor``.
    """
    for a in ring:
        if a == anchor:
            continue
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring:
                continue
            if ni in other_ring:
                continue
            terminal = _bfs_to_ring(mol, ni, ring, other_ring)
            if terminal is not None:
                return a
    return None


def _ring_distance(
    ring: List[int],
    a_idx: int,
    b_idx: Optional[int],
) -> Optional[int]:
    """Return the bond-distance between two atoms on a cyclic ring (min over
    both directions). None if either atom not on the ring.
    """
    if b_idx is None:
        return None
    if a_idx not in ring or b_idx not in ring:
        return None
    n = len(ring)
    pa = ring.index(a_idx)
    pb = ring.index(b_idx)
    d = abs(pa - pb)
    return min(d, n - d)


# ---------------------------------------------------------------------------
# _build_composite_locant (D-05)
# ---------------------------------------------------------------------------


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _build_composite_locant(
    ring_idx: int,
    ring_locant: int,
    style: str = "ascii",
) -> str:
    """Emit a composite locant. ASCII (production / OPSIN-friendly) returns
    ``f"{ring_idx}({ring_locant})"``; Unicode superscript (documentation form)
    returns ``f"{ring_idx}<sup>{ring_locant}</sup>"`` rendered with Unicode
    digit superscripts.

    Args:
      ring_idx:     ring index in the phane parent enumeration (1-based).
      ring_locant:  atom locant within the ring (1-based).
      style:        ``"ascii"`` (default; production) or ``"superscript"``.

    Raises:
      ValueError on unknown style.

    Source: 155-CONTEXT.md D-05; 155-AUDIT-A.md §5.
    """
    if style == "ascii":
        return f"{ring_idx}({ring_locant})"
    if style == "superscript":
        return f"{ring_idx}{str(ring_locant).translate(_SUPERSCRIPT_DIGITS)}"
    raise ValueError(f"Unknown style: {style!r} (must be 'ascii' or 'superscript')")


# ---------------------------------------------------------------------------
# name_cyclophane (top-level handler)
# ---------------------------------------------------------------------------


def name_cyclophane(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Emit the IUPAC PIN cyclophane name for ``mol``, or None.

    Production output: bracket-prefix semi-systematic form
    ``[m.n.<...>]paracyclophane`` (or metacyclophane / orthocyclophane /
    cyclophane per the topology classification). Bridge lengths are listed in
    non-ascending order per Blue Book P-26.4 convention.

    Returns None for None input, non-cyclophane topology, or for cyclophanes
    with linker rings other than carbocyclic benzene (these are tagged
    ``GENERIC_CYCLOPHANE`` and quarantined per 155-AUDIT-A.md §8 R3 -- the
    ``is_cyclophane`` gate accepts them but the name composition is deferred
    to v19 / Phase 156 grammar pre-validation).

    Source: 155-CONTEXT.md D-04 + D-05; 155-AUDIT-A.md §3 + §10.
    """
    if mol is None:
        return None
    if not is_cyclophane(mol):
        return None  # mutual-exclusion gate (D-16)

    topology = _classify_phane_topology(mol)
    if topology is PhaneTopology.GENERIC_CYCLOPHANE:
        # R3 quarantine: heterocyclic linker / bridge cases ship in a later
        # sub-phase. Sub-phase 155.A emits the bracket-prefix form for
        # carbocyclic benzene linkers only.
        return None

    Chem.GetSSSR(mol)
    sssr_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    small_rings = [r for r in sssr_rings if len(r) <= _SMALL_RING_MAX_SIZE]

    # Determine bridge lengths from the inter-ring chains; non-ascending order.
    bridge_lengths: List[int] = []
    for i in range(len(small_rings)):
        for j in range(i + 1, len(small_rings)):
            ring_i = small_rings[i]
            ring_j = small_rings[j]
            if ring_i & ring_j:
                continue
            anchors = _all_chain_anchor_pairs(mol, ring_i, ring_j)
            for anchor_a, _ in anchors:
                # length = number of intermediate atoms on the shortest path
                # from anchor_a (out of ring_i) to ring_j.
                start_atom = mol.GetAtomWithIdx(anchor_a)
                start_neighbours = [
                    nbr.GetIdx()
                    for nbr in start_atom.GetNeighbors()
                    if nbr.GetIdx() not in ring_i and nbr.GetIdx() not in ring_j
                ]
                shortest_len: Optional[int] = None
                for entry in start_neighbours:
                    # Path length from entry to (anchor on ring_j).
                    path_len = _shortest_path_length_to_target(
                        mol, entry, ring_i, ring_j
                    )
                    if path_len is None:
                        continue
                    candidate = path_len + 1  # +1 for the entry atom itself
                    if shortest_len is None or candidate < shortest_len:
                        shortest_len = candidate
                if shortest_len is not None:
                    bridge_lengths.append(shortest_len)
    if not bridge_lengths:
        return None
    # De-duplicate symmetric chains: each chain is counted once per anchor; for
    # paracyclophane the two anchors yield the same chain; collapse.
    bridge_lengths = sorted(set(bridge_lengths), reverse=True) if len(bridge_lengths) <= 1 else _collapse_bridge_count(bridge_lengths, mol, small_rings)
    bridge_lengths.sort(reverse=True)

    bracket_prefix = "[" + ".".join(str(b) for b in bridge_lengths) + "]"
    base_name = _PHANE_BASE_NAMES[topology]
    return f"{bracket_prefix}{base_name}"


def _shortest_path_length_to_target(
    mol: Chem.Mol,
    start: int,
    forbidden: Set[int],
    target: Set[int],
) -> Optional[int]:
    """BFS from ``start`` to nearest atom in ``target`` while avoiding ``forbidden``.

    Returns the number of intermediate atoms (start counts as 0) before
    reaching target, or None if no path. The returned value excludes the
    target atom but includes the start atom (i.e., len(intermediate atoms)
    starting at ``start`` and stopping when adjacent to target).
    """
    if start in forbidden:
        return None
    if start in target:
        return 0
    visited: Set[int] = {start}
    queue: deque = deque([(start, 0)])
    while queue:
        cur, dist = queue.popleft()
        atom = mol.GetAtomWithIdx(cur)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in forbidden:
                continue
            if ni in target:
                return dist  # # intermediate atoms BEFORE reaching target
            if ni in visited:
                continue
            visited.add(ni)
            queue.append((ni, dist + 1))
    return None


def _collapse_bridge_count(
    bridge_lengths: List[int],
    mol: Chem.Mol,
    small_rings: List[Set[int]],
) -> List[int]:
    """Collapse symmetric anchor-pair duplicates: every chain is enumerated
    twice (once per endpoint anchor). For a typical 2-bridge cyclophane each
    bridge appears as 2 entries with the same length; we want exactly the
    distinct bridge-length count.

    Heuristic: the actual bridge count equals the number of *distinct chains*
    between disjoint small-ring pairs. For the standard case (2 disjoint
    benzenes joined by 2 chains -> [2.2]paracyclophane) the bridge_lengths
    list has 4 entries (2 bridges x 2 anchors); we return [len, len].

    Source: 155-AUDIT-A.md §8 + canonical chain enumeration analysis.
    """
    # Count distinct chains by enumerating chain atom paths and de-duplicating.
    chain_paths: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()
    for i in range(len(small_rings)):
        for j in range(i + 1, len(small_rings)):
            ring_i = small_rings[i]
            ring_j = small_rings[j]
            if ring_i & ring_j:
                continue
            for anchor_a, anchor_b in _all_chain_anchor_pairs(mol, ring_i, ring_j):
                path = _shortest_chain_path_via_anchor(
                    mol, anchor_a, anchor_b, ring_i, ring_j
                )
                if path is None:
                    continue
                key = tuple(sorted(path))
                if key in seen:
                    continue
                seen.add(key)
                chain_paths.append(tuple(path))
    return [len(p) for p in chain_paths]


def _shortest_chain_path_via_anchor(
    mol: Chem.Mol,
    anchor_a: int,
    anchor_b: int,
    ring_a: Set[int],
    ring_b: Set[int],
) -> Optional[List[int]]:
    """Shortest chain atoms (intermediate, excluding anchors and rings) on the
    path from ``anchor_a`` to ``anchor_b``, walking only through non-ring atoms.
    """
    parent: Dict[int, Optional[int]] = {}
    visited: Set[int] = {anchor_a}
    queue: deque = deque()
    atom_a = mol.GetAtomWithIdx(anchor_a)
    for nbr in atom_a.GetNeighbors():
        ni = nbr.GetIdx()
        if ni in ring_a or ni in ring_b:
            continue
        if ni == anchor_b:
            # Direct anchor-to-anchor adjacency = no intermediate chain atoms,
            # which is ring-assembly territory (P-28), not cyclophane.
            # Return None (not []) so _collapse_bridge_count's
            # `if path is None: continue` guard correctly drops the degenerate
            # entry instead of recording it as a 0-length bridge that would
            # later emit `[0.X]paracyclophane` (155-REVIEW.md WR-02).
            return None
        if ni not in visited:
            visited.add(ni)
            parent[ni] = None  # came from anchor_a directly
            queue.append(ni)
    while queue:
        cur = queue.popleft()
        atom = mol.GetAtomWithIdx(cur)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_a:
                continue
            if ni == anchor_b:
                # reconstruct intermediate path (between anchor_a and anchor_b)
                path: List[int] = []
                node: Optional[int] = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            if ni in ring_b:
                continue  # only allow reaching anchor_b, not other ring_b atoms
            if ni in visited:
                continue
            visited.add(ni)
            parent[ni] = cur
            queue.append(ni)
    return None
