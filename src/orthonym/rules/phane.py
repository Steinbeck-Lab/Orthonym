"""Cyclic phane parent hydride nomenclature (IUPAC P-26.4).

Detects molecules whose topology fits the IUPAC P-26.4 "cyclic phane"
class (two or more disjoint small rings linked by acyclic chain
segments of length >= 2 atoms whose linkage closes a macrocyclic
ring; mutually exclusive with ring-assembly, spiro/fused/bridged-
fused, and multiplicative cases).

Public API:

* ``is_cyclophane(mol) -> bool`` -- topology gate
* ``name_cyclophane(mol) -> Optional[str]`` -- top-level handler
* ``_classify_phane_topology(mol) -> PhaneTopology`` -- sub-class enum
* ``_build_composite_locant(ring_idx, ring_locant, style)``-- composite-locant emitter
* ``_enumerate_inter_ring_chains(mol, small_rings)`` -- BFS chain walker
* ``PhaneTopology`` -- enum

Source:

* 155-CONTEXT.md (mutual-exclusion topology gate; corrected SSSR
  criterion per 155-AUDIT-A.md Critical Finding 0).
* 155-CONTEXT.md (sub-class enum {PARACYCLOPHANE, METACYCLOPHANE,
  ORTHOCYCLOPHANE, GENERIC_CYCLOPHANE}).
* 155-CONTEXT.md (composite-locant dual rendering: ASCII default,
  Unicode superscript option).
* 155-CONTEXT.md (mutual-exclusion contract enforced at the topology
  gate; pattern -- canonical helper reuse via
  ``multiplicative._is_pure_single_bond_assembly``).
* 155-CONTEXT.md (root-cause-only; no postprocessor band-aids).
* 155-AUDIT-A.md Critical Finding 0 (wording correction: SSSR-based
  criterion replaces the merged-ring-system phrasing because cyclophane
  macrocycles share atoms with both small rings, merging into a single
  ring system via ``perception.rings.get_ring_systems``).
* IUPAC 2013 Blue Book P-26.4 "Cyclic Phane Parent Hydrides".
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem


# ---------------------------------------------------------------------------
# Sub-class enum
# ---------------------------------------------------------------------------


class PhaneTopology(Enum):
    """Cyclophane sub-class per IUPAC P-26.4.

    Source: 155-CONTEXT.md; 155-AUDIT-A.md §4.
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
# Public API: is_cyclophane (corrected SSSR criterion)
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
    5. Mutual exclusion: NOT pure single-bond ring assembly (
       territory; canonical helper reuse via
       ``multiplicative._is_pure_single_bond_assembly``).

    Returns False for None input.

    Source: 155-CONTEXT.md +; 155-AUDIT-A.md Critical Finding 0.
    """
    if mol is None:
        return False

    Chem.GetSSSR(mol)
    sssr_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if len(sssr_rings) < 2:
        return False

    # Total ring nodes (gate (b) in spirit -- originally said "ring atoms";
    # use the union of SSSR rings as the corrected analogue).
    all_ring_atoms: Set[int] = set()
    for r in sssr_rings:
        all_ring_atoms.update(r)
    if len(all_ring_atoms) < 6:
        return False

    # Mutual-exclusion (d.1): pure single-bond ring assembly =
    # territory. Canonical helper reuse (-- avoid duplication).
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
            # Require the full chain to be a subset of some macro SSSR ring.
            # `chain_atoms` already excludes the linker-ring endpoints (per
            # `_shortest_chain_path` which never crosses ring_a / ring_b
            # atoms), so a genuine cyclophane chain lies entirely inside the
            # macrocyclic SSSR ring. The previous `chain_atoms & macro`
            # partial-overlap fallback admitted false positives where one
            # chain atom happened to belong to some unrelated macro ring at
            # random; without a Blue Book P-26.4 fixture demonstrating the
            # partial-overlap necessity, the looser test risks accepting
            # non-cyclophane topologies (155-REVIEW.md WR-04).
            for macro in macro_rings:
                if chain_atoms <= macro:
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

    Source: 155-CONTEXT.md (graph-topology classification);
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
# _classify_phane_topology
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

    Source: 155-CONTEXT.md; 155-AUDIT-A.md §4.
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
# _build_composite_locant
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
      ring_idx: ring index in the phane parent enumeration (1-based).
      ring_locant: atom locant within the ring (1-based).
      style: ``"ascii"`` (default; production) or ``"superscript"``.

    Raises:
      ValueError on unknown style.

    Source: 155-CONTEXT.md; 155-AUDIT-A.md §5.
    """
    if style == "ascii":
        return f"{ring_idx}({ring_locant})"
    if style == "superscript":
        return f"{ring_idx}{str(ring_locant).translate(_SUPERSCRIPT_DIGITS)}"
    raise ValueError(f"Unknown style: {style!r} (must be 'ascii' or 'superscript')")


# ---------------------------------------------------------------------------
# P-26 PIN subsystem (Wave-8 P8): simplification -> simplified-skeleton PIN
# ---------------------------------------------------------------------------
#
# Replaces the semi-systematic bracket-prefix composer below (`[m.n]para-
# cyclophane`) with the IUPAC P-26.2/.3/.4 "simplified skeletal name" PIN
# (`1,4(1,4)-dibenzenacyclohexaphane`). Scope THIS PHASE (.7): monocyclic
# skeleton, all amplificants IDENTICAL benzene rings, no substituents, no
# skeletal ('a') heteroatom replacement, no indicated hydrogen. Every other
# class (von Baeyer / spiro skeletons, mixed/different amplificants,
# substituted phanes, heteroatom-replaced phanes) is a DESIGN CONTRACT this
# phase -- the comparators/helpers exist and are unit-tested against the Blue
# Book directly, but `build_phane_pin` FAILS CLOSED (returns None) for any
# molecule outside the verified scope. OPSIN 2.9 cannot parse ANY phane name
# (verified 2026-07-16) so there is no RT oracle; verification is BB-name-
# exact fixtures + the `_phane_formula_veto` source-level atom-conservation
# guard (.12) -- see


class SkeletonClass(Enum):
    """P-26.2.1 simplified-skeleton structural class. Deliberately separate
    from ``PhaneTopology`` (the legacy para/meta/ortho sub-class enum used by
    the semi-systematic composer / `is_cyclophane` gate) to avoid churn in
    `test_phane.py`."""

    ACYCLIC = "acyclic"
    MONOCYCLIC = "monocyclic"
    VON_BAEYER = "von_baeyer"
    SPIRO = "spiro"


# ---------------------------------------------------------------------------
# .2 -- amplification-prefix transform (P-26.2.2.1)
# ---------------------------------------------------------------------------


def _amplification_prefix(parent_name: str) -> str:
    """P-26.2.2.1 (BlueBookV2.md:14894): final 'e' -> 'a', else append 'a'.
    Bracketed locant prefixes (e.g. '[1,2]oxazole') keep their brackets
    (P-26.4.2.2 note,:15048). ``parent_name`` is the OPSIN-parseable PIN
    parent-hydride name of the amplificant ring/ring system."""
    if parent_name.endswith("e"):
        return parent_name[:-1] + "a"
    return parent_name + "a"


# ---------------------------------------------------------------------------
# .3 -- simplified skeletal-name builder (P-26.2.1)
# ---------------------------------------------------------------------------


def _phane_node_multiplier(n: int) -> str:
    """Multiplying affix for the TOTAL skeleton node count (superatoms +
    bridge atoms), per P-26.2.1. Reuses the shared simple-multiplier table
    (di..icosa) with the compositional `chain_names` fallback above 20 (BB
    `cyclotetratriacontaphane`, P-26.5.2 example,:15165) -- no duplication."""
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    if n in SIMPLE_MULTIPLIERS:
        return SIMPLE_MULTIPLIERS[n]
    from ..data.chain_names import get_chain_prefix

    prefix = get_chain_prefix(n)
    if not prefix.endswith("a"):
        prefix += "a"
    return prefix


def _simplified_skeletal_name(n: int, skeleton_class: "SkeletonClass") -> Optional[str]:
    """P-26.2.1 (BlueBookV2.md:14857): simplified-skeleton name = structure
    prefix ('cyclo' for monocyclic; none for acyclic) + multiplying affix
    (node count, INCLUDING superatoms) + 'phane'.

    Returns None for VON_BAEYER / SPIRO (.10 -- no verified
    canonicalizable BB fixture this phase; fail closed rather than guess the
    `bicyclo[..]`/`spiro[..]` skeleton-descriptor grammar)."""
    if skeleton_class is SkeletonClass.MONOCYCLIC:
        return f"cyclo{_phane_node_multiplier(n)}phane"
    if skeleton_class is SkeletonClass.ACYCLIC:
        return f"{_phane_node_multiplier(n)}phane"
    return None


# ---------------------------------------------------------------------------
# .4 -- multiplicative amplificant term di/bis (P-26.2.3)
# ---------------------------------------------------------------------------

# A prefix "begins with a multiplying prefix" (P-26.2.3.2) if it starts with
# a bracketed locant set (heterocycle, e.g. '[1,3]dioxola') or a von-Baeyer /
# ring-assembly / spiro name component ('bicyclo', 'tricyclo', 'spiro',...).
# Fuzzy per the Blue Book's own "or a name component that could take one"
# clause (.4 open question) -- every amplificant buildable THIS phase
# is a bare benzene/pyridine-family prefix, so this never fires in practice;
# it exists so the leaf unit test (bicyclo[2.2.1]heptana / [1,3]dioxola) is
# real, not fabricated.
_COMPLEX_AMPLIFICANT_PREFIX_RE = re.compile(r"^(?:\[|spiro|(?:bi|tri|tetra)cyclo)")


def _multiplied_amplificant(prefix: str, count: int) -> Optional[str]:
    """P-26.2.3: 'di'/'tri'/'tetra' before a SIMPLE amplification prefix
    (P-26.2.3.1,:14930 -- 'dibenzena', 'tripyridina'); 'bis'/'tris' +
    parenthesization when the prefix begins with a multiplying prefix or
    bracketed locants (P-26.2.3.2,:14932 -- 'bis(bicyclo[2.2.1]heptana)',
    'bis([1,3]dioxola)'). Returns None if ``count`` has no multiplier in the
    reused tables (not reachable this phase)."""
    if count <= 1:
        return prefix
    if _COMPLEX_AMPLIFICANT_PREFIX_RE.match(prefix):
        from ..assembly.naming_utils import COMPLEX_MULTIPLIERS

        mult = COMPLEX_MULTIPLIERS.get(count)
        if mult is None:
            return None
        return f"{mult}({prefix})"
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    mult = SIMPLE_MULTIPLIERS.get(count)
    if mult is None:
        return None
    return f"{mult}{prefix}"


# ---------------------------------------------------------------------------
# .1 -- simplification engine (amplificant perception + skeleton graph)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Amplificant:
    """One amplificant (ring/ring-system replacing a superatom) in a
    simplified phane skeleton (P-26.1.4).

    Attributes:
      atoms: frozenset of atom indices making up the amplificant ring.
      ring_atoms_ordered: the RDKit SSSR cyclic-order atom-index tuple for
        this ring (needed for `_ring_distance` to compute the amplificant's
        OWN attachment-locant set, P-26.3.2).
      parent_name: the OPSIN-parseable PIN parent-hydride name of the
        isolated ring (e.g. "benzene")..1 scope this phase: benzene
        only -- any other ring makes `_simplify` return None (fail closed;
        the P-26.2.2.2.2 disallowed-parent gate is future-phase work once
        more parent kinds are supported).
      attachment_atoms: the (exactly 2, this phase) ring atom indices bonded
        to a skeleton (bridge) neighbour.
    """

    atoms: FrozenSet[int]
    ring_atoms_ordered: Tuple[int, ...]
    parent_name: str
    attachment_atoms: Tuple[int, ...]


@dataclass(frozen=True)
class PhaneStructure:
    """Simplified-skeleton structure produced by `_simplify` (.1)."""

    mol: Chem.Mol
    amplificants: Tuple[Amplificant, ...]
    bridge_atoms: FrozenSet[int]
    skeleton_class: "SkeletonClass"
    # Ordered cyclic node walk (MONOCYCLIC only): each entry is ('amp', index
    # into `amplificants`) or ('bridge', atom_idx). None for skeleton classes
    # without a computed node walk (VON_BAEYER / SPIRO --.10).
    node_cycle: Optional[Tuple[Tuple[str, int], ...]]


def _node_key_builder(bridge_atoms: Set[int], small_rings: List[Set[int]]):
    """Return a fast atom_idx -> ('bridge', idx) | ('amp', ring_idx) | None
    lookup closure."""
    atom_to_ring: Dict[int, int] = {}
    for ridx, r in enumerate(small_rings):
        for a in r:
            atom_to_ring[a] = ridx

    def _key(atom_idx: int) -> Optional[Tuple[str, int]]:
        if atom_idx in bridge_atoms:
            return ("bridge", atom_idx)
        if atom_idx in atom_to_ring:
            return ("amp", atom_to_ring[atom_idx])
        return None

    return _key


def _walk_cycle(
    adjacency: Dict[Tuple[str, int], Set[Tuple[str, int]]],
    start: Tuple[str, int],
) -> Optional[Tuple[Tuple[str, int], ...]]:
    """Walk a 2-regular graph as a single cycle starting at ``start``.

    Returns the ordered node tuple, or None if ``start`` isn't 2-regular-
    connected (defensive; `_simplify` already checks every node's degree).
    """
    neighbours = list(adjacency.get(start, ()))
    if len(neighbours) != 2:
        return None
    order: List[Tuple[str, int]] = [start]
    prev, cur = start, neighbours[0]
    seen = {start}
    while cur != start:
        order.append(cur)
        seen.add(cur)
        candidates = [n for n in adjacency[cur] if n != prev]
        if not candidates:
            return None
        prev, cur = cur, candidates[0]
        if len(order) > len(adjacency) + 1:
            return None  # safety valve against a malformed graph
    return tuple(order)


def _simplify(mol: Optional[Chem.Mol]) -> Optional["PhaneStructure"]:
    """.1: perceive amplificants + build the simplified-skeleton graph.

    BB rule: P-26.1.1-.1.6 (14809-14837); allowed amplificant parents
    P-26.2.2.2.1 (14904) -- THIS PHASE restricted to benzene (mancude
    carbocyclic 6-ring, all-carbon, aromatic); anything else fails closed
    (heteroaryl/fused amplificants + the full P-26.2.2.2.2 disallowed-parent
    gate are future-phase work once a general isolated-ring PIN namer is
    wired in here).

    Representative case: ``C1Cc2ccc(cc2)CCc2ccc1cc2`` -> 2 benzene
    amplificants, 4 bridge CH2 atoms, 6-node monocyclic skeleton.

    Returns None (fail closed) for: non-cyclophane topology; < 2 small
    rings; any non-benzene small ring; fused/shared-atom small rings (P-26.1.4
    fused-amplificant collapse -- future phase); a direct ring-to-ring bond
    (0-atom bridge, ring-assembly territory); any skeleton node whose degree
    != 2 (branch/substituent atoms, heteroatom bridges with extra valence,
    von-Baeyer/spiro skeletons -- all future-phase /.9/8.10); any
    amplificant with attachment-atom count != 2.
    """
    if mol is None or not is_cyclophane(mol):
        return None

    Chem.GetSSSR(mol)
    sssr_rings_raw = [list(r) for r in mol.GetRingInfo().AtomRings()]
    small_rings_raw = [r for r in sssr_rings_raw if len(r) <= _SMALL_RING_MAX_SIZE]
    small_rings = [set(r) for r in small_rings_raw]

    if len(small_rings) < 2:
        return None
    if not _all_linkers_are_carbocyclic_benzene(mol, small_rings):
        return None  # .1 scope: benzene amplificants only this phase
    for i in range(len(small_rings)):
        for j in range(i + 1, len(small_rings)):
            if small_rings[i] & small_rings[j]:
                return None  # fused amplificants -- future phase (P-26.1.4)

    all_small_atoms: Set[int] = set()
    for r in small_rings:
        all_small_atoms |= r
    bridge_atoms: Set[int] = {
        a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in all_small_atoms
    }
    # .9 safety net: skeletal heteroatom bridges ('a'-replacement) are a
    # design contract this phase (unwired to emission) -- a non-carbon bridge
    # atom must never be silently named as a plain hydrocarbon phane.
    if any(mol.GetAtomWithIdx(a).GetSymbol() != "C" for a in bridge_atoms):
        return None

    node_key = _node_key_builder(bridge_atoms, small_rings)
    amp_attachment: List[Set[int]] = [set() for _ in small_rings]
    skeleton_adj: Dict[Tuple[str, int], Set[Tuple[str, int]]] = {}

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ku, kv = node_key(u), node_key(v)
        if ku is None or kv is None:
            continue
        if ku[0] == "amp" and kv[0] == "amp":
            if ku[1] != kv[1]:
                return None  # direct ring-to-ring bond -- ring assembly, not phane
            continue  # internal ring bond
        if ku == kv:
            continue
        skeleton_adj.setdefault(ku, set()).add(kv)
        skeleton_adj.setdefault(kv, set()).add(ku)
        if ku[0] == "amp":
            amp_attachment[ku[1]].add(u)
        if kv[0] == "amp":
            amp_attachment[kv[1]].add(v)

    total_nodes = len(small_rings) + len(bridge_atoms)
    if len(skeleton_adj) != total_nodes:
        return None
    for neighbours in skeleton_adj.values():
        if len(neighbours) != 2:
            return None  # not a simple cycle -- branch/substituent or von-Baeyer/spiro node
    for ridx in range(len(small_rings)):
        if len(amp_attachment[ridx]) != 2:
            return None  # an amplificant must have exactly 2 attachment atoms this phase

    amplificants: List[Amplificant] = []
    for ridx, ordered in enumerate(small_rings_raw):
        amplificants.append(
            Amplificant(
                atoms=frozenset(small_rings[ridx]),
                ring_atoms_ordered=tuple(ordered),
                parent_name="benzene",
                attachment_atoms=tuple(sorted(amp_attachment[ridx])),
            )
        )

    node_cycle = _walk_cycle(skeleton_adj, ("amp", 0))
    if node_cycle is None or len(node_cycle) != total_nodes:
        return None

    return PhaneStructure(
        mol=mol,
        amplificants=tuple(amplificants),
        bridge_atoms=frozenset(bridge_atoms),
        skeleton_class=SkeletonClass.MONOCYCLIC,
        node_cycle=node_cycle,
    )


# ---------------------------------------------------------------------------
# .5 -- skeleton numbering + superatom-locant assignment
# ---------------------------------------------------------------------------


def _number_skeleton(struct: "PhaneStructure") -> Optional[Dict[Tuple[str, int], int]]:
    """P-26.3.1 (14935) + P-26.4.1.1 (14969): number the simplified
    skeleton so superatoms get the LOWEST locant set.

    MONOCYCLIC scope (.5): enumerate every (start-amplificant,
    direction) pair -- optimal numbering always assigns locant 1 to SOME
    amplificant (any set containing 1 beats any set that doesn't at the
    first point of difference) -- and keep the candidate whose sorted
    superatom-locant tuple is lexicographically lowest.

    Representative case: 6-node monocycle, 2 superatoms 2 bridge atoms apart
    -> numbers to {1,4} (not {1,3}/{2,5}).

    Returns None for non-MONOCYCLIC structures (.10 fail-closed) or a
    structure with no node_cycle.
    """
    if struct.skeleton_class is not SkeletonClass.MONOCYCLIC or struct.node_cycle is None:
        return None
    n = len(struct.node_cycle)
    amp_positions = [i for i, node in enumerate(struct.node_cycle) if node[0] == "amp"]
    if not amp_positions:
        return None

    best_locants: Optional[Dict[Tuple[str, int], int]] = None
    best_key: Optional[Tuple[int, ...]] = None
    for start_pos in amp_positions:
        for direction in (1, -1):
            locants: Dict[Tuple[str, int], int] = {}
            for k in range(n):
                pos = (start_pos + direction * k) % n
                locants[struct.node_cycle[pos]] = k + 1
            key = tuple(sorted(locants[struct.node_cycle[p]] for p in amp_positions))
            if best_key is None or key < best_key:
                best_key = key
                best_locants = locants
    return best_locants


# ---------------------------------------------------------------------------
# .6 -- attachment-locant perception + ordering
# ---------------------------------------------------------------------------


def _attachment_locants(
    struct: "PhaneStructure",
    skeleton_locants: Dict[Tuple[str, int], int],
) -> Optional[List[Tuple[int, Tuple[int, int]]]]:
    """P-26.3.2 (14939) + P-26.3.2.2 (14957): each amplificant's own
    attachment-locant set, FIRST-cited locant adjacent to the LOWER skeleton
    locant.

    Benzene-only scope (.6): the isolated ring is fully symmetric, so
    ANY attachment atom may be assigned ring-locant 1 -- we choose the one
    adjacent (via its bridge) to the lower-numbered skeleton neighbour, then
    the other attachment atom gets `1 + ring_distance` (para -> (1,4), meta
    -> (1,3), ortho -> (1,2)).

    Returns a list parallel to ``struct.amplificants``:
    ``(superatom_locant, (attach_locant_1, attach_locant_2))``. None
    (fail-closed) if the adjacency can't be resolved (defensive).
    """
    if struct.node_cycle is None:
        return None
    n = len(struct.node_cycle)
    results: List[Tuple[int, Tuple[int, int]]] = []
    for amp_idx, amp in enumerate(struct.amplificants):
        node = ("amp", amp_idx)
        if node not in skeleton_locants:
            return None
        superatom_locant = skeleton_locants[node]
        pos = struct.node_cycle.index(node)
        prev_node = struct.node_cycle[(pos - 1) % n]
        next_node = struct.node_cycle[(pos + 1) % n]
        prev_locant = skeleton_locants.get(prev_node)
        next_locant = skeleton_locants.get(next_node)
        if prev_locant is None or next_locant is None:
            return None
        lower_neighbor_node = prev_node if prev_locant < next_locant else next_node

        atom_toward: Dict[int, Tuple[str, int]] = {}
        for a in amp.attachment_atoms:
            for nbr in struct.mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in struct.bridge_atoms:
                    atom_toward[a] = ("bridge", ni)
        atom_toward_lower = None
        atom_toward_other = None
        for a, nk in atom_toward.items():
            if nk == lower_neighbor_node:
                atom_toward_lower = a
            else:
                atom_toward_other = a
        if atom_toward_lower is None or atom_toward_other is None:
            return None

        d = _ring_distance(list(amp.ring_atoms_ordered), atom_toward_lower, atom_toward_other)
        if d is None:
            return None
        results.append((superatom_locant, (1, 1 + d)))
    return results


# ---------------------------------------------------------------------------
# .12 -- source-level formula-conservation veto
# ---------------------------------------------------------------------------


def _phane_formula_veto(struct: Optional["PhaneStructure"], mol: Optional[Chem.Mol]) -> bool:
    """The RT-gate FAILS OPEN with no Java (OPSIN cannot parse any phane
    name), so this is the ONLY guard against a phane emission that silently
    drops or mutates an atom (.12). Returns True iff every heavy atom
    in ``mol`` is accounted for EXACTLY ONCE by ``struct`` (amplificant ring
    atoms + bridge atoms), with matching element symbols."""
    if struct is None or mol is None:
        return False
    accounted: Dict[int, str] = {}
    for amp in struct.amplificants:
        for a in amp.atoms:
            if a in accounted:
                return False  # double-counted atom
            accounted[a] = mol.GetAtomWithIdx(a).GetSymbol()
    for a in struct.bridge_atoms:
        if a in accounted:
            return False
        accounted[a] = mol.GetAtomWithIdx(a).GetSymbol()
    if len(accounted) != mol.GetNumAtoms():
        return False
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in accounted or accounted[idx] != atom.GetSymbol():
            return False
    return True


# ---------------------------------------------------------------------------
# .7 -- assemble the monocyclic all-benzene homophane PIN (integration)
# ---------------------------------------------------------------------------


def build_phane_pin(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Assemble the P-26 simplified-skeletal PIN for ``mol``, or None.

    .7 scope: MONOCYCLIC skeleton, all amplificants IDENTICAL benzene
    rings sharing the SAME attachment-locant set (P-26.3.2.1 contraction).
    Every other class (mixed amplificants, von Baeyer/spiro skeletons,
    substituents, 'a'-replacement) is fail-closed this phase -- see the
    module docstring above.1 and Tasks 8.9-8.11.

    Source: BB P-26.2.1/.2.2.1/.2.3/.3.1/.3.2/.3.2.1/.4.1.1; anchor
    `1,4(1,4)-dibenzenacyclohexaphane` (P-26.3.2.1,:14947); meta homolog
    `1,4(1,3)-dibenzenacyclohexaphane` (P-26.4.1.4,:15024).
    """
    struct = _simplify(mol)
    if struct is None:
        return None
    if struct.skeleton_class is not SkeletonClass.MONOCYCLIC:
        return None
    if any(a.parent_name != "benzene" for a in struct.amplificants):
        return None  # .7 scope: all-benzene only this phase

    skeleton_locants = _number_skeleton(struct)
    if skeleton_locants is None:
        return None
    per_amp = _attachment_locants(struct, skeleton_locants)
    if per_amp is None or len(per_amp) != len(struct.amplificants):
        return None

    attach_sets = {a[1] for a in per_amp}
    if len(attach_sets) != 1:
        # Different attachment patterns across amplificants -- the
        # per-amplificant (non-contracted) citation form is P-26.3.2.2/
        # .11 territory with no verified in-corpus fixture this phase.
        return None
    shared_attach = next(iter(attach_sets))

    if not _phane_formula_veto(struct, mol):
        return None

    parent_name = struct.amplificants[0].parent_name
    prefix = _amplification_prefix(parent_name)
    n_amplificants = len(struct.amplificants)
    multiplied_prefix = _multiplied_amplificant(prefix, n_amplificants)
    if multiplied_prefix is None:
        return None

    total_nodes = len(struct.node_cycle)
    skeletal_name = _simplified_skeletal_name(total_nodes, struct.skeleton_class)
    if skeletal_name is None:
        return None

    superatom_locants = sorted(sup for sup, _ in per_amp)
    superatom_str = ",".join(str(x) for x in superatom_locants)
    attach_str = ",".join(str(x) for x in shared_attach)
    return f"{superatom_str}({attach_str})-{multiplied_prefix}{skeletal_name}"


# ---------------------------------------------------------------------------
# .8 -- composite-locant citation ordering for substituted phanes
# (DESIGN CONTRACT -- unwired to emission; no substituted-phane molecule is
# buildable this phase. Pure comparator, unit-tested against the BB
# heptachloro tuple directly per the plan's.8 verification bullet.)
# ---------------------------------------------------------------------------


def _composite_locant_sort_key(locant: str) -> Tuple[int, int]:
    """P-26.4.3.2/.3.3 (15093/:15101): composite-locant citation order --
    primary (skeleton) locant first (ascending); a PLAIN locant (no
    amplificant superscript) sorts before any composite locant sharing the
    same primary; composite locants then sort by ascending superscript.

    Accepts ASCII composite notation ``"N"`` (plain) or ``"N(k)"``
    (composite, matching `_build_composite_locant(style='ascii')`).
    """
    m = re.match(r"^(\d+)(?:\((\d+)\))?$", locant)
    if not m:
        raise ValueError(f"Malformed composite locant: {locant!r}")
    primary = int(m.group(1))
    superscript = int(m.group(2)) if m.group(2) else 0
    return (primary, superscript)


# ---------------------------------------------------------------------------
# .9 -- 'a'-replacement heterophanes (DESIGN CONTRACT -- unwired to
# emission this phase; `_simplify` already fails closed on any non-carbon
# bridge atom,.1's safety net above, so a heteroatom-bridge phane
# never reaches a hydrocarbon-phane misname. Pure naming helper, unit-tested
# against the BB trithia example directly.)
# ---------------------------------------------------------------------------


def _apply_skeletal_replacement(
    heteroatom_locants: Dict[int, str],
    base_name: str,
) -> str:
    """P-26.5.1 (15155 `(PIN)`): nondetachable 'a'-replacement prefixes for
    skeleton heteroatoms, cited by ascending locant then BB element order
    (O>S>Se>Te>N>P>As>Sb>Bi>Si>Ge>Sn>Pb>B>Al>Ga>In>Tl, P-26.5.4.2:15228),
    prepended to the hydrocarbon phane ``base_name``.

    ``heteroatom_locants`` maps skeleton locant -> element symbol (e.g.
    ``{2: "S", 4: "S", 6: "S"}``). Elements are grouped by their nondetachable
    stem (thia/oxa/aza/...) via the shared `skeletal_replacement.REPLACEMENT_TERMS`
    table (P-15.4 -- reused, not duplicated); a single element's locant list is
    contracted with a multiplying prefix + hyphen, e.g. ``"2,4,6-trithia-"``.
    Different-element groups cite in BB P-26.5.4.2 seniority order (reuses
    `skeletal_replacement._A_CITATION_ORDER`).
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    from .skeletal_replacement import REPLACEMENT_TERMS, _A_CITATION_ORDER

    by_element: Dict[str, List[int]] = {}
    for locant, element in heteroatom_locants.items():
        by_element.setdefault(element, []).append(locant)

    parts: List[str] = []
    for element in _A_CITATION_ORDER:
        if element not in by_element or element not in REPLACEMENT_TERMS:
            continue
        locants = sorted(by_element[element])
        count = len(locants)
        stem = REPLACEMENT_TERMS[element]
        mult = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        loc_str = ",".join(str(x) for x in locants)
        parts.append(f"{loc_str}-{mult}{stem}")
    return "-".join(parts) + "-" + base_name


# ---------------------------------------------------------------------------
# .11 -- multi-different-amplificant seniority numbering (DESIGN
# CONTRACT -- unwired; `build_phane_pin` fail-closes on any molecule with
# differing amplificants,.7's `attach_sets` uniqueness check above /
# the parent-name-uniqueness check. Pure comparator reusing the existing
# P-44.2 ring_system_score, unit-tested against the BB worked-example order.)
# ---------------------------------------------------------------------------


def _phane_amplificant_seniority_key(
    ring_score: tuple,
    attachment_locants: Tuple[int, ...],
) -> Tuple[tuple, Tuple[int, ...]]:
    """P-26.4.1.3 (14990) + P-26.4.2.2 (15044): when >=2 DIFFERENT
    amplificants compete for the lower superatom locant, the MOST SENIOR
    ring (P-44.2, lower ``ring_system_score`` = more senior -- see
    ``rules.ring_selection.ring_system_score``) gets the lower locant;
    residual ties broken by the lower attachment-locant set.

    Sorting a list of ``(ring_score, attachment_locants)`` ascending and
    assigning superatom locants in that order reproduces the BB worked
    example `1(8,5)-quinolina-4(1,4)-phenanthrena-7(1,4)-naphthalena-
    cyclononaphane` (14998): quinoline is N-heterocyclic (senior to any
    carbocycle on P-44.2.1(a)/(b)) -> locant 1; phenanthrene (3 rings) is
    senior to naphthalene (2 rings) on P-44.2.1(d) -> locant 4 vs 7.
    """
    return (ring_score, attachment_locants)


# ---------------------------------------------------------------------------
# name_cyclophane (top-level handler)
# ---------------------------------------------------------------------------


def name_cyclophane(mol: Optional[Chem.Mol]) -> Optional[str]:
    """Emit the IUPAC P-26 simplified-skeletal PIN for ``mol``, or None.

    Wave-8 P8: delegates to `build_phane_pin` (.7), which builds the
    P-26.2/.3/.4 simplified-skeleton PIN (`1,4(1,4)-dibenzenacyclohexaphane`)
    for the monocyclic all-benzene-homophane class. Falls back to the legacy
    semi-systematic bracket-prefix composer (`[m.n]paracyclophane`) ONLY for
    topologies `build_phane_pin` doesn't (yet) cover, so `is_cyclophane`-
    positive molecules outside the P-26 PIN scope still get *some* name from
    the pre-existing (production-withheld, unit-tested) composer rather than
    silently returning None here -- the dispatch-level fail-closed decision
    is made in `dispatch_table._handle_cyclophane` (.12), not here.

    Returns None for None input or non-cyclophane topology.

    Source: 155-CONTEXT.md +; 155-AUDIT-A.md §3 + §10;
     .7.
    """
    if mol is None:
        return None
    if not is_cyclophane(mol):
        return None  # mutual-exclusion gate

    pin = build_phane_pin(mol)
    if pin is not None:
        return pin

    topology = _classify_phane_topology(mol)
    if topology is PhaneTopology.GENERIC_CYCLOPHANE:
        # R3 quarantine: heterocyclic linker / bridge cases ship in a later
        # sub-phase. Sub-.A emits the bracket-prefix form for
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
    # paracyclophane the two anchors yield the same chain; collapse via the
    # distinct-paths walk. Run unconditionally — the prior `<= 1` fast-path
    # silently passed through a single anchor-pair entry as `[X]<base>`, a
    # structurally invalid one-bridge cyclophane (a cyclophane requires >= 2
    # bridges; one bridge between two disjoint rings cannot close a macrocycle).
    # A < 2-bridge result indicates the topology gate let through a non-
    # cyclophane and we fail closed by returning None (155-REVIEW.md WR-03).
    bridge_lengths = _collapse_bridge_count(bridge_lengths, mol, small_rings)
    if len(bridge_lengths) < 2:
        return None
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


# ---------------------------------------------------------------------------
# P-52.2.7.3 boundary: phane preferred over ring-assembly at >= 7 rings
# ---------------------------------------------------------------------------
# BB P-52.2.7.3 (BlueBookV2.md:24088): "Phane names are preferred IUPAC names
# rather than ring assembly names when seven or more rings or ring systems are
# present." The full linear-phane amplification/simplification engine
# (P-52.2.5, >= 7 nodes) is NOT built in Wave-2 P2. This module owns only the
# P-26.4 CYCLOphane class above. The [n]cyclophane / large-ring-assembly path
# is p1_chains_b territory (P-28.5 extended ASSEMBLY_MULTIPLIERS; INTERNAL
# oracle since OPSIN cannot parse e.g. [5]paracyclophane). The guard below is
# the fail-closed seam: it detects the >= 7-ring-assembly trigger but ALWAYS
# declines, so Orthonym never emits a WRONG phane name for this class.


def is_seven_plus_ring_assembly(mol) -> bool:
    """True iff ``mol`` is >= 7 ring systems joined only by single bonds
    (the P-52.2.7.3 trigger). Detection only — used by the scope guard's
    contract tests; does NOT drive emission."""
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if len(atom_rings) < 7:
        return False
    # Each ring must be linked to the assembly by a single (non-fused,
    # non-shared-atom) bond: assemblies have no atoms shared between rings.
    all_ring_atoms = set()
    for r in atom_rings:
        s = set(r)
        if s & all_ring_atoms:
            return False  # shared atom -> fused/spiro, not a single-bond assembly
        all_ring_atoms |= s
    return True


def linear_phane_scope_guard(mol) -> "Optional[str]":
    """Fail-closed boundary for P-52.2.7.3 linear-phane PIN generation.

    ALWAYS returns None: the linear-phane amplification engine (P-52.2.5,
    >= 7 nodes) is not implemented in Wave-2 P2. Present so (a) the boundary is
    contract-tested and (b) a future engine has a named seam. Never emits a
    (possibly wrong) phane name. BB P-52.2.7.3 (BlueBookV2.md:24088)."""
    return None
