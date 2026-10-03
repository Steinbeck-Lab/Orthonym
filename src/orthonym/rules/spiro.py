"""
Spiro compound naming module.

Handles naming of spiro systems where rings share exactly one atom each
(the spiro center). Generates IUPAC spiro[a.b] descriptors for monospiro
and dispiro[a.b.c.d] for polyspiro compounds.

IUPAC / rules for spiro naming:
- Monospiro descriptor: spiro[a.b] where a <= b
- a = smaller_ring_size - 1, b = larger_ring_size - 1
- The -1 accounts for the shared spiro center
- Numbering starts at atom adjacent to spiro center in smaller ring
- Goes around smaller ring, through spiro center, then around larger ring

IUPAC: dispiro/trispiro naming for multiple spiro centers
- dispiro[a.b.c.d] where a,b,c,d are segment sizes between spiro atoms
- Numbering starts in terminal ring, proceeds through spiro atoms

IUPAC: heterocyclic spiro compounds use skeletal replacement
'a' prefixes (oxa, aza, thia, etc.) with locants from spiro numbering

Examples:
    spiro[4.5]decane - cyclopentane fused to cyclohexane (5-1=4, 6-1=5)
    spiro[5.5]undecane - two cyclohexanes sharing one carbon (6-1=5, 6-1=5)
    dispiro[2.1.2.1]octane - three rings sharing two spiro centers
"""

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem

if TYPE_CHECKING:  # ``_build_hetero_prefix``'s return type. Import-time-free:
    # the runtime import stays function-local, matching this module's habit.
    from .ring_replacement import ReplacementPrefix

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix

# a phase-04: shared IUPAC heteroatom priority (halogen-aware).
# ``sort_heteroatoms_by_priority`` supplies the skeletal-replacement CITATION
# order. The 'a'-prefix SPELLING comes from ``get_heteroatom_prefix`` above, i.e.
# from Table 1.5 -- the correct table for the spiro/von Baeyer contexts in this
# module. ``get_hw_prefix`` (Table 2.4, Hantzsch-Widman monocycles) used to be
# consulted first here and is deliberately no longer imported: the two tables
# disagree by design for Al and In (``aluma``/``indiga`` vs ``alumina``/``inda``),
# so reaching into the HW table from a non-HW context is a wrong-prefix bug. The
# note this comment replaced justified the HW-first order by saying
# ``get_heteroatom_prefix`` "has typos like Te->'tea'" -- that was not a typo but
# its ``symbol.lower + 'a'`` fabrication fallback, which is now removed.
from ..data.hw_heteroatoms import (
    get_heteroatom_priority,
    sort_heteroatoms_by_priority,
)
from ..perception.molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from ..perception.rings import get_spiro_atoms
from ..rules.lambda_convention import (
    LAMBDA as _LAMBDA,
)

# a phase : the / Table-2.8 lambda-convention logic was promoted
# to the shared rules/lambda_convention.py so spiro, acyclic skeletal-replacement
# and the mononuclear-hydride namers share one fail-closed implementation. The
# private aliases preserve the spiro public surface (test_spiro_g4.py imports
# spiro._nonstandard_bonding_number) with byte-identical behaviour.
from ..rules.lambda_convention import (  # noqa: F401 — re-export for test compat
    STANDARD_BONDING_NUMBER as _STANDARD_BONDING_NUMBER,
)
from ..rules.lambda_convention import (
    nonstandard_bonding_number as _nonstandard_bonding_number,
)

# a phase-02 /: locant comparator reuse — no parallel comparator
# permitted in this module. Imported at the top so the source-grep lock in
# tests/unit/rules/test_mixed_spiro_fused.py and test_spiro_numbering.py
# can verify the invariant without inspecting individual function bodies.
from ..rules.locants import compare_locant_sets  # noqa: F401 — re-export lock
from ..rules.polycyclic_bridged import get_heteroatom_prefix

# a phase-02 /: locant tuple type alias for cascade-step-6
# suppliers. Plain int for typical ring atoms; (int, str) tuple for
# fusion atoms with a letter suffix (e.g., (4, 'a') -> "4a").
_Locant = Union[int, Tuple[int, str]]

# Polyspiro prefix names indexed by number of spiro centers
_POLYSPIRO_PREFIXES = [
    '',           # 0 (unused)
    'spiro',      # 1 = monospiro
    'dispiro',    # 2
    'trispiro',   # 3
    'tetraspiro', # 4
    'pentaspiro', # 5
]

def is_spiro_system(mol) -> bool:
    """
    Check if molecule is a pure spiro system.

    A pure spiro system has N spiro atoms connecting N+1 rings, with
    no additional fused or bridged ring junctions. Molecules that have
    spiro atoms but also have additional polycyclic complexity (fused,
    bridged, etc.) are NOT classified as spiro systems.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a pure spiro system
    """
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return False

    # A pure spiro system with N spiro atoms should have exactly N+1 SSSR rings.
    # If there are more rings, the molecule has additional fused/bridged
    # ring junctions and should not be classified as a spiro system.
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_spiro = len(spiro_atoms)

    return n_rings == n_spiro + 1


def get_spiro_ring_sizes(mol, spiro_center: int) -> Tuple[int, int]:
    """
    Get the sizes of the two rings sharing a spiro center.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Tuple of (smaller_ring_size, larger_ring_size), sorted

    Raises:
        ValueError: If spiro_center is not in exactly 2 rings
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        raise ValueError(
            f"Spiro center at atom {spiro_center} must be in exactly 2 rings, "
            f"found {len(rings_containing)}"
        )

    size1 = len(rings_containing[0])
    size2 = len(rings_containing[1])
    return (min(size1, size2), max(size1, size2))


def generate_spiro_descriptor(mol) -> Optional[str]:
    """
    Generate the spiro descriptor for a spiro compound.

    For monospiro (1 spiro center): spiro[a.b] where a <= b.
    For polyspiro (2+ spiro centers): dispiro[a.b.c.d], trispiro[...], etc.

    Args:
        mol: RDKit Mol object

    Returns:
        Spiro descriptor string like "spiro[4.5]" or "dispiro[2.1.2.1]",
        or None if not spiro
    """
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return None

    if len(spiro_atoms) == 1:
        spiro_center = list(spiro_atoms)[0]
        try:
            smaller_ring, larger_ring = get_spiro_ring_sizes(mol, spiro_center)
        except ValueError:
            return None
        a = smaller_ring - 1
        b = larger_ring - 1
        return f"spiro[{a}.{b}]"

    # Polyspiro (dispiro, trispiro, etc.)
    return _generate_polyspiro_descriptor(mol, spiro_atoms)


def _generate_polyspiro_descriptor(mol, spiro_atoms: Set[int]) -> Optional[str]:
    """
    Generate the polyspiro descriptor (dispiro, trispiro, etc.).

    For N spiro centers sharing N+1 rings, the descriptor has 2*N segments.
    """
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    # Wave2 /: tri+ polyspiro descriptors need
    # superscript revisit locants and the branched von-Baeyer spiro walk;
    # _reorder_segments_iupac is only correct for dispiro (3 rings). A 4+-ring
    # descriptor in ring-sequential order names a DIFFERENT constitution
    # (OPSIN reparses trispiro[4.2.2.2.2.5]icosane to another molecule).
    # Wave2 P3- /.3.1/.3.2): build the branched superscript-revisit
    # descriptor. Fail closed (None -> UNSUPPORTED_RING_SYSTEM refusal via the
    # tier_a_ring pure-polyspiro guard) if the branched walk cannot be resolved.
    if len(spiro_atoms) >= 3:
        built = _build_branched_polyspiro(mol, spiro_atoms)
        if built is None:
            return None
        return built[0]

    ring_chain, spiro_chain = _build_ring_chain(mol, all_rings, spiro_atoms)
    if ring_chain is None or spiro_chain is None:
        return None
    if len(spiro_chain) != len(spiro_atoms):
        return None

    segments = _compute_spiro_segments(mol, ring_chain, spiro_chain)
    if not segments:
        return None

    n_spiro = len(spiro_atoms)
    if n_spiro < len(_POLYSPIRO_PREFIXES):
        prefix = _POLYSPIRO_PREFIXES[n_spiro]
    else:
        prefix = f'{n_spiro}spiro'

    # (the Blue Book): "Each time a spiro atom is reached for the
    # second time its locant, which has already been assigned, is cited as a
    # superscript number to the number of the preceding linking atoms";
    # 'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981), and:9991 recommends the
    # superscripts for every polyspiro PIN. The walk [t1. b. t2. d] numbers
    # the first spiro atom t1 + 1 and the second t1 + b + 2 (the order of
    # ``_dispiro_numbering_candidates``), and the walk returns to each of them
    # after the terminal ring t2 and the return arc d. Written '^n', the
    # superscript convention of the branched walk (``_branched_spiro_walk``).
    if n_spiro == 2 and len(segments) == 4:
        t1, b, t2, d = segments
        seg_str = f"{t1}.{b}.{t2}^{t1 + b + 2}.{d}^{t1 + 1}"
    else:
        seg_str = '.'.join(str(s) for s in segments)
    return f"{prefix}[{seg_str}]"


def _spiro_ring_adjacency(
    mol, all_rings: List[List[int]], spiro_atoms: Set[int]
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """Ring adjacency graph for a pure spiro tree: ring i ~ ring j when they
    share exactly one spiro atom (and NO fused edge). Returns None if any two
    rings share >1 atom (fused/bridged — not a pure spiro system)."""
    n = len(all_rings)
    adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(all_rings[i]) & set(all_rings[j])
            if not shared:
                continue
            if len(shared) != 1:
                return None  # fused/bridged edge — not pure spiro
            sa = next(iter(shared))
            if sa not in spiro_atoms:
                return None
            adj[i].append((j, sa))
            adj[j].append((i, sa))
    return adj


def _branched_spiro_walk(
    mol,
    all_rings: List[List[int]],
    adj: Dict[int, List[Tuple[int, int]]],
    spiro_atoms: Set[int],
    start_ring: int,
    start_neighbor: int,
) -> Optional[Tuple[List[str], Dict[int, int]]]:
    """Perform one branched von-Baeyer spiro walk.

    Starting in ``start_ring`` at the non-spiro neighbour ``start_neighbor`` of
    that ring's single spiro atom, walk the ring tree via a depth-first
    traversal: number non-spiro atoms as they are first encountered, number a
    spiro atom on first visit, and each time a spiro atom is revisited emit its
    (first-visit) locant as a superscript on the preceding segment. Segment
    lengths count the linking (non-spiro) atoms traversed since the previous
    spiro atom. Returns (descriptor_tokens, atom_to_locant) or None if the walk
    cannot be resolved (e.g. a ring is not a simple cycle)."""
    numbering: Dict[int, int] = {}
    counter = [0]
    tokens: List[str] = []
    seg = [0]  # linking atoms accumulated since last spiro atom

    def _number(atom: int):
        counter[0] += 1
        numbering[atom] = counter[0]

    def _flush(superscript: Optional[int]):
        if superscript is None:
            tokens.append(str(seg[0]))
        else:
            tokens.append(f"{seg[0]}^{superscript}")
        seg[0] = 0

    def _ring_cycle_order(ring: List[int], entry: int) -> Optional[List[int]]:
        """Return the ring atoms as a cyclic order starting at ``entry``."""
        ring_set = set(ring)
        adj_map: Dict[int, List[int]] = {a: [] for a in ring}
        for a in ring:
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in ring_set:
                    adj_map[a].append(ni)
        if any(len(v) != 2 for v in adj_map.values()):
            return None  # not a simple cycle
        order = [entry]
        prev = None
        cur = entry
        while len(order) < len(ring):
            nxts = [x for x in adj_map[cur] if x != prev]
            if not nxts:
                return None
            nxt = nxts[0]
            order.append(nxt)
            prev, cur = cur, nxt
        return order

    def _traverse(ring_idx: int, entry_spiro: Optional[int], parent_ring: Optional[int]):
        """Traverse ``ring_idx`` in cyclic order.

        entry_spiro is the spiro atom we entered on (already numbered) for a
        non-root ring, or None for the root (start) terminal ring. For a
        non-root ring the cyclic walk closes back at entry_spiro and emits the
        closing ``seg^{entry_spiro.locant}`` (the revisit). Along the way, each
        first-visit child spiro atom closes its own leading segment, is numbered,
        and its child ring is recursed into (which returns having emitted its own
        superscript close)."""
        ring = all_rings[ring_idx]
        if entry_spiro is None:
            order = _ring_cycle_order(ring, start_neighbor)
            if order is None:
                return False
        else:
            full = _ring_cycle_order(ring, entry_spiro)
            if full is None:
                return False
            order = full[1:]  # skip entry_spiro (already numbered)
        # child spiro atoms on this ring keyed by atom idx -> child ring idx
        child_of: Dict[int, int] = {}
        for (nbr_ring, sa) in adj[ring_idx]:
            if nbr_ring != parent_ring:
                child_of[sa] = nbr_ring
        for atom in order:
            if atom in spiro_atoms:
                # first-visit child spiro atom: close the leading segment,
                # number it, descend the child ring (which closes back here).
                _number(atom)
                _flush(None)
                if atom in child_of:
                    if not _traverse(child_of[atom], atom, ring_idx):
                        return False
            else:
                _number(atom)
                seg[0] += 1
        # Non-root ring: the cycle returns to entry_spiro -> closing revisit.
        if entry_spiro is not None:
            _flush(numbering[entry_spiro])
        return True

    if not _traverse(start_ring, None, None):
        return None
    return tokens, numbering


def _build_branched_polyspiro(
    mol, spiro_atoms: Set[int]
) -> Optional[Tuple[str, Dict[int, int]]]:
    """ branched polyspiro: superscript-revisit descriptor + numbering.

    Enumerates candidate starts (each non-spiro neighbour of the single spiro
    atom in each smallest terminal ring) and directions, runs the von-Baeyer
    spiro walk for each, then chooses per (lowest spiro-atom locant
    set) and (lowest descriptor numbers at first point of
    difference). Fail closed (None) if the topology is not a clean spiro tree
    or the descriptor-atom count does not equal total_atoms - n_spiro."""
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]
    adj = _spiro_ring_adjacency(mol, all_rings, spiro_atoms)
    if adj is None:
        return None
    n_spiro = len(spiro_atoms)
    if n_spiro + 1 != len(all_rings):
        return None  # not exactly N spiro atoms sharing N+1 rings
    # each spiro atom must be in exactly 2 rings (standard spiro)
    for sa in spiro_atoms:
        if sum(1 for r in all_rings if sa in r) != 2:
            return None

    ring_atom_union: Set[int] = set()
    for r in all_rings:
        ring_atom_union |= set(r)
    total_atoms = len(ring_atom_union)

    terminal = [i for i, r in enumerate(all_rings)
                if len(set(r) & spiro_atoms) == 1]
    if not terminal:
        return None
    min_term_size = min(len(all_rings[i]) for i in terminal)
    start_rings = [i for i in terminal if len(all_rings[i]) == min_term_size]

    candidates: List[Tuple[Tuple[int, ...], List[int], str, Dict[int, int]]] = []
    for sr in start_rings:
        ring = all_rings[sr]
        sa = next(iter(set(ring) & spiro_atoms))
        ring_set = set(ring)
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(sa).GetNeighbors()
                     if n.GetIdx() in ring_set]
        for start_nbr in neighbors:
            walked = _branched_spiro_walk(
                mol, all_rings, adj, spiro_atoms, sr, start_nbr)
            if walked is None:
                continue
            tokens, numbering = walked
            if len(numbering) != total_atoms:
                continue
            # integrity: descriptor linking-atom count == total - n_spiro
            seg_sum = 0
            for t in tokens:
                seg_sum += int(t.split('^')[0])
            if seg_sum != total_atoms - n_spiro:
                continue
            spiro_locs = tuple(sorted(numbering[s] for s in spiro_atoms))
            desc_nums = [int(t.split('^')[0]) for t in tokens]
            candidates.append((spiro_locs, desc_nums, '.'.join(tokens), numbering))

    if not candidates:
        return None
    # lowest spiro-atom locant set, then lowest descriptor
    candidates.sort(key=lambda c: (list(c[0]), c[1], c[2]))
    _, _, desc_body, numbering = candidates[0]
    prefix = (_POLYSPIRO_PREFIXES[n_spiro]
              if n_spiro < len(_POLYSPIRO_PREFIXES) else f'{n_spiro}spiro')
    return f"{prefix}[{desc_body}]", numbering


def _build_ring_chain(
    mol,
    all_rings: List[List[int]],
    spiro_atoms: Set[int],
) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    """
    Build an ordered chain of rings connected by spiro atoms.

    For dispiro (2 spiro atoms, 3 rings):
    result = [terminal_ring_1, middle_ring, terminal_ring_2]
    spiro_order = [spiro_atom_1, spiro_atom_2]
    """
    ring_spiro_map = []
    for ring in all_rings:
        ring_set = set(ring)
        shared = ring_set & spiro_atoms
        ring_spiro_map.append(shared)

    terminal_indices = [i for i, s in enumerate(ring_spiro_map) if len(s) == 1]
    if len(terminal_indices) < 2:
        return None, None

    n_rings = len(all_rings)
    ring_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n_rings)}
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            common_spiro = ring_spiro_map[i] & ring_spiro_map[j]
            if common_spiro:
                for sa in common_spiro:
                    ring_adj[i].append((j, sa))
                    ring_adj[j].append((i, sa))

    best_result = None

    for start_idx in terminal_indices:
        visited_rings = {start_idx}
        chain = [all_rings[start_idx]]
        spiro_order: List[int] = []
        current = start_idx

        while True:
            found_next = False
            for next_ring, via_spiro in ring_adj[current]:
                if next_ring not in visited_rings:
                    visited_rings.add(next_ring)
                    chain.append(all_rings[next_ring])
                    spiro_order.append(via_spiro)
                    current = next_ring
                    found_next = True
                    break
            if not found_next:
                break

        if len(spiro_order) == len(spiro_atoms):
            segments = _compute_spiro_segments(mol, chain, spiro_order)
            if segments:
                if best_result is None or segments < best_result[2]:
                    best_result = (chain, spiro_order, segments)

    if best_result is None:
        return None, None
    return best_result[0], best_result[1]


def _compute_spiro_segments(
    mol,
    ring_chain: List[List[int]],
    spiro_chain: List[int],
) -> List[int]:
    """
    Compute the segment sizes for a polyspiro descriptor.

    For dispiro with ring_chain = [ring_A, ring_B, ring_C] and
    spiro_chain = [s1, s2]:
    - Segment 1: non-spiro atoms in ring_A
    - Segment 2: one path through ring_B between s1 and s2
    - Segment 3: non-spiro atoms in ring_C
    - Segment 4: other path through ring_B between s1 and s2
    """
    segments: List[int] = []
    n_rings = len(ring_chain)

    for ring_idx in range(n_rings):
        ring = ring_chain[ring_idx]
        ring_set = set(ring)
        spiro_in_ring = [s for s in spiro_chain if s in ring_set]

        if len(spiro_in_ring) == 1:
            # Terminal ring: one segment = non-spiro atoms
            seg = len(ring) - 1
            segments.append(seg)

        elif len(spiro_in_ring) == 2:
            # Middle ring: two segments (two paths between the two spiro atoms)
            s1, s2 = spiro_in_ring[0], spiro_in_ring[1]
            path1, path2 = _find_two_paths(mol, ring, s1, s2)
            seg_a = len(path1) - 2
            seg_b = len(path2) - 2
            segments.append(min(seg_a, seg_b))
            segments.append(max(seg_a, seg_b))

    # Reorder for IUPAC format: interleave terminal and middle ring segments
    if n_rings >= 3:
        segments = _reorder_segments_iupac(segments, n_rings)

    return segments


def _reorder_segments_iupac(
    raw_segments: List[int],
    n_rings: int,
) -> List[int]:
    """
    Reorder segments from ring-sequential order to IUPAC descriptor order.

    Ring-sequential: [term1, mid_short, mid_long, term2]
    IUPAC: [term1, mid_short, term2, mid_long]
    """
    if n_rings == 3 and len(raw_segments) == 4:
        return [raw_segments[0], raw_segments[1], raw_segments[3], raw_segments[2]]
    return raw_segments


def _find_two_paths(
    mol,
    ring: List[int],
    start: int,
    end: int,
) -> Tuple[List[int], List[int]]:
    """
    Find the two paths around a ring between start and end atoms.
    """
    ring_set = set(ring)
    adj: Dict[int, List[int]] = {a: [] for a in ring}
    for a in ring:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring_set:
                adj[a].append(nbr.GetIdx())

    neighbors_of_start = [n for n in adj[start] if n in ring_set]
    if len(neighbors_of_start) < 2:
        return ([start, end], [start, end])

    path1 = _ring_walk_to(adj, start, end, neighbors_of_start[0])
    path2 = _ring_walk_to(adj, start, end, neighbors_of_start[1])
    return path1, path2


def _ring_walk_to(
    adj: Dict[int, List[int]],
    start: int,
    end: int,
    first_step: int,
) -> List[int]:
    """Walk around a ring from start via first_step until we reach end."""
    path = [start, first_step]
    visited = {start, first_step}
    current = first_step

    max_steps = len(adj) + 1
    for _ in range(max_steps):
        if current == end:
            break
        for nbr in adj[current]:
            if nbr not in visited:
                path.append(nbr)
                visited.add(nbr)
                current = nbr
                break
    return path


def _walk_ring_from_spiro(
    mol,
    ring: List[int],
    spiro_center: int,
    already_visited: Set[int],
) -> List[int]:
    """
    Walk around a ring starting from the atom adjacent to spiro_center,
    going around the ring and ending at spiro_center.
    Returns atoms in walk order: [non-spiro atoms..., spiro_center]
    """
    ring_set = set(ring)
    spiro_atom = mol.GetAtomWithIdx(spiro_center)

    ring_neighbors = []
    for nbr in spiro_atom.GetNeighbors():
        if nbr.GetIdx() in ring_set and nbr.GetIdx() != spiro_center:
            ring_neighbors.append(nbr.GetIdx())

    if not ring_neighbors:
        return list(ring)

    start = None
    for nbr in ring_neighbors:
        if nbr not in already_visited:
            start = nbr
            break
    if start is None:
        start = ring_neighbors[0]

    ordered = [start]
    walk_visited = {start, spiro_center}

    current = start
    while len(ordered) < len(ring) - 1:
        atom = mol.GetAtomWithIdx(current)
        found = False
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in walk_visited:
                ordered.append(nbr_idx)
                walk_visited.add(nbr_idx)
                current = nbr_idx
                found = True
                break
        if not found:
            break

    ordered.append(spiro_center)
    return ordered


def _walk_ring_between_spiros(
    mol,
    ring: List[int],
    entry_spiro: int,
    exit_spiro: int,
    already_visited: Set[int],
) -> List[int]:
    """Walk through a middle ring from entry_spiro to exit_spiro.

     "Linear polyspiro alicyclic ring systems"
    (``the Blue Book Blue Book``): *"...proceeding consecutively, always
    by the SHORTER path, to the other terminal ring through each spiro atom
    and then back to the first spiro atom..."* -- the first middle-ring arc
    (numbered right after the first spiro atom) must be the one with FEWER
    linking atoms; the longer arc is only numbered later, on the way back
    (the descriptor's 4th/last segment). ``_compute_spiro_segments`` (the
    descriptor-string builder) already applies ``min(seg_a, seg_b)`` first --
    this picker used to choose the OPPOSITE (more unvisited atoms = the
    LONGER arc) first, so the numbering silently disagreed with its own
    descriptor string whenever the two arcs differ in length.
    Invisible for symmetric middle rings (both arcs equal, e.g. every
    existing ``dispiro[a.b.c.b]``-shaped test/fixture) and for all-carbon
    skeletons (swapping the two arcs is a graph automorphism, so the wrong
    choice still names the same molecule) -- it only produces a WRONG name
    once a heteroatom breaks that symmetry (confirmed: OPSIN round-trip
    ``inchi_mismatch`` on ``C1C2(CCC2)C11CCO1`` and ``C1CC11CCC11CO1``,
    fixed by this change; see ``tests/unit/rules/test_vonbaeyer_spiro.py``).
    """
    path1, path2 = _find_two_paths(mol, ring, entry_spiro, exit_spiro)
    unvisited1 = sum(1 for a in path1 if a not in already_visited)
    unvisited2 = sum(1 for a in path2 if a not in already_visited)
    return path1 if unvisited1 <= unvisited2 else path2


def _walk_unvisited_ring_atoms(
    mol,
    ring: List[int],
    entry_spiro: int,
    already_visited: Set[int],
) -> List[int]:
    """Walk the unvisited portion of a middle ring."""
    ring_set = set(ring)
    unvisited = [a for a in ring if a not in already_visited]
    if not unvisited:
        return []

    start = None
    for a in unvisited:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in already_visited and nbr.GetIdx() in ring_set:
                start = a
                break
        if start:
            break

    if start is None:
        return unvisited

    ordered = [start]
    walk_visited = {start}
    current = start
    while len(ordered) < len(unvisited):
        atom = mol.GetAtomWithIdx(current)
        found = False
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if (nbr_idx in ring_set
                    and nbr_idx not in walk_visited
                    and nbr_idx not in already_visited):
                ordered.append(nbr_idx)
                walk_visited.add(nbr_idx)
                current = nbr_idx
                found = True
                break
        if not found:
            break
    return ordered


def _build_polyspiro_numbering_sequence(
    mol,
    ring_chain: List[List[int]],
    spiro_chain: List[int],
) -> List[int]:
    """
    Build the IUPAC numbering sequence for a polyspiro system.
    """
    n_rings = len(ring_chain)
    sequence: List[int] = []
    visited: Set[int] = set()

    for ring_idx in range(n_rings):
        ring = ring_chain[ring_idx]

        if ring_idx == 0:
            entry_spiro = spiro_chain[0]
            ordered = _walk_ring_from_spiro(mol, ring, entry_spiro, visited)
            for a in ordered:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)
        elif ring_idx == n_rings - 1:
            entry_spiro = spiro_chain[-1]
            ordered = _walk_ring_from_spiro(mol, ring, entry_spiro, visited)
            for a in ordered:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)
        else:
            entry_spiro = spiro_chain[ring_idx - 1]
            exit_spiro = spiro_chain[ring_idx]
            path = _walk_ring_between_spiros(
                mol, ring, entry_spiro, exit_spiro, visited
            )
            for a in path:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)

    # Return path through middle rings (remaining unvisited atoms)
    for ring_idx in range(n_rings - 2, 0, -1):
        ring = ring_chain[ring_idx]
        unvisited_in_ring = [a for a in ring if a not in visited]
        if unvisited_in_ring:
            entry_spiro = spiro_chain[ring_idx]
            path = _walk_unvisited_ring_atoms(mol, ring, entry_spiro, visited)
            for a in path:
                if a not in visited:
                    sequence.append(a)
                    visited.add(a)

    return sequence


def _dispiro_numbering_candidates(
    mol, ring_chain: List[List[int]], spiro_chain: List[int],
) -> List[Dict[int, int]]:
    """Enumerate every -legal numbering of a 3-ring (dispiro) chain.

     fixes the numbering ONLY where the two options actually differ
    (the smaller terminal ring first; the shorter middle-ring arc first). It
    leaves three genuine free choices unresolved, each a real degree of
    freedom the Blue Book does not break itself:
      (1) which physical terminal ring is numbered first, when the two
          terminal rings TIE in size;
      (2) the traversal DIRECTION within each terminal ring (which of the
          spiro atom's two ring-neighbours starts the count);
      (3) which middle-ring arc is numbered first, when the two arcs TIE
          in length.
    Task F  found that leaving these to raw RDKit ring/neighbour
    iteration order (rather than enumerating them) made the heteroatom locant
    depend on the input SMILES atom order -- a determinism-gate violation,
    even though every resulting name still round-trips to the same molecule.
    This enumerates every combination so the caller can apply 's
    "low locants to heteroatoms" rule DETERMINISTICALLY (mirrors
    ``get_spiro_numbering``, the monospiro sibling, which resolves the same
    two kinds of freedom the same way)."""
    term1, mid, term2 = ring_chain
    s1, s2 = spiro_chain

    orientations = [(term1, term2, s1, s2)]
    if len(term1) == len(term2):
        orientations.append((term2, term1, s2, s1))

    candidates: List[Dict[int, int]] = []
    for t1, t2, a1, a2 in orientations:
        path_x, path_y = _find_two_paths(mol, mid, a1, a2)
        int_x, int_y = path_x[1:-1], path_y[1:-1]
        if len(int_x) < len(int_y):
            mid_orders = [(int_x, int_y)]
        elif len(int_y) < len(int_x):
            mid_orders = [(int_y, int_x)]
        else:
            mid_orders = [(int_x, int_y), (int_y, int_x)]

        t1_traversals = _spiro_ring_traversals(mol, t1, a1)
        t2_traversals = _spiro_ring_traversals(mol, t2, a2)
        for first_mid, second_mid in mid_orders:
            # (the Blue Book Blue Book): the descriptor is cited
            # "...through each spiro atom and then BACK TO THE FIRST spiro atom".
            # The first middle arc (segment ``b``) is numbered forward, a1->a2,
            # right after the first spiro atom; the second/return arc (the last
            # descriptor segment, ``d``) is numbered on the way back, a2->a1, so
            # the atom adjacent to the SECOND spiro atom takes the lower locant.
            # ``_find_two_paths`` yields both interiors in a1->a2 order, so the
            # return arc must be reversed. Invisible for symmetric middle rings
            # (both arcs equal) and all-carbon skeletons (arc swap is a graph
            # automorphism), it only fixes a name once a heteroatom or exocyclic
            # group breaks that symmetry -- e.g. C1CC11COC11CCC1 was numbered
            # 9-oxadispiro[2.0.3.2]nonane (a DIFFERENT molecule on OPSIN reparse)
            # instead of the descriptor-consistent 8-oxadispiro[2.0.3.2]nonane.
            second_mid = list(second_mid)[::-1]
            for t1_seq in t1_traversals:
                for t2_seq in t2_traversals:
                    sequence = (list(t1_seq) + [a1] + list(first_mid) + [a2]
                                + list(t2_seq) + list(second_mid))
                    if len(sequence) == len(set(sequence)):
                        candidates.append(
                            {atom_idx: i + 1
                             for i, atom_idx in enumerate(sequence)})
    return candidates


def _get_polyspiro_numbering(
    mol, spiro_atoms: Set[int], suffix_ring_atoms: Optional[Set[int]] = None,
    prefix_ring_atoms: Optional[List[int]] = None,
    rank_full_molecule: bool = False,
) -> Optional[Dict[int, int]]:
    """Generate IUPAC numbering for a polyspiro system.

    ``suffix_ring_atoms`` (a phase SUBST-01 parity with ``get_spiro_numbering``,
    the monospiro sibling): the free valence of a spiro SUBSTITUENT,
    -- ranked after heteroatoms in the lowest-locant tiebreak below, so a
    polyspiro substituent's attachment point gets the lowest locant available
    once the heteroatom placement (if any) is settled.

    ``prefix_ring_atoms`` and ``rank_full_molecule``: as in
    ``get_spiro_numbering`` (f), then (g) and (j) for a numbering of
    the whole molecule)."""
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    # branched polyspiro (>=3 spiro atoms) uses the superscript-revisit
    # walk, whose numbering is defined BY the descriptor citation order. Reuse
    # the same walk here so heteroatom prefixes see a consistent numbering.
    if len(spiro_atoms) >= 3:
        built = _build_branched_polyspiro(mol, spiro_atoms)
        if built is None:
            return None
        return built[1]

    ring_chain, spiro_chain = _build_ring_chain(mol, all_rings, spiro_atoms)
    if ring_chain is None:
        return None

    # a performance pass: choose DETERMINISTICALLY among every -legal
    # numbering (see _dispiro_numbering_candidates) by lowest
    # heteroatom locants, then a canonical-rank tiebreak -- same selection
    # shape as ``get_spiro_numbering``'s monospiro ``_key``. Falls back to the
    # single-candidate sequential walk only if the candidate enumeration finds
    # nothing (defensive; every dispiro chain _build_ring_chain accepts should
    # enumerate at least the one candidate the old walk produced).
    candidates = _dispiro_numbering_candidates(mol, ring_chain, spiro_chain)
    if not candidates:
        sequence = _build_polyspiro_numbering_sequence(
            mol, ring_chain, spiro_chain)
        if not sequence:
            return None
        return {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}

    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    suffix_set = set(suffix_ring_atoms or ())
    prefix_list = list(prefix_ring_atoms or ())

    def _key(mapping: Dict[int, int]):
        heteros = [(a, loc) for a, loc in mapping.items()
                   if mol.GetAtomWithIdx(a).GetSymbol() != 'C']
        het_locs = sorted(loc for _a, loc in heteros)
        het_by_seniority = sorted(
            (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc)
            for a, loc in heteros)
        # a phase SUBST-01 parity: lowest locants to the free-valence/suffix
        # atoms, ranked after heteroatoms (same order as get_spiro_numbering).
        suffix_locs = sorted(loc for a, loc in mapping.items()
                              if a in suffix_set)
        # (f) (the Blue Book), as in get_spiro_numbering: empty
        # when the caller passes no prefix atoms, so the ordering is unchanged.
        prefix_locs = sorted(mapping[a] for a in prefix_list if a in mapping)
        return (het_locs, het_by_seniority, suffix_locs, prefix_locs)

    def _canon_key(mapping: Dict[int, int]):
        seq = [a for a, _loc in sorted(mapping.items(), key=lambda kv: kv[1])]
        return tuple(canon[a] for a in seq)

    ring_set = set().union(*(set(r) for r in ring_chain))
    return _select_spiro_numbering(
        mol, candidates, _key, _canon_key, ring_set, prefix_list, suffix_set,
        rank_full_molecule)


def _spiro_exocyclic_prefix_atoms(mol, ring_atoms: Set[int]) -> List[int]:
    """The (f) input read from the molecule itself: each ring atom of the
    spiro system once for every exocyclic heavy neighbour (a gem-dimethyl carbon
    twice), as ``general_engine._prefix_ring_atoms`` gives the general engine.
    For the substituent-prefix names of ``name_spiro_system``'s callers, which
    cite every exocyclic group as a prefix."""
    out: List[int] = []
    for a in sorted(ring_atoms):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in ring_atoms and nb.GetAtomicNum() > 1:
                out.append(a)
    return out


def _spiro_cip_locant_key(mol, mapping: Dict[int, int]) -> tuple:
    """ (j) key (``naming_utils.cip_locant_rank_key``) of one spiro
    numbering: each numbered ring atom's R/S/r/s and each ring double bond's
    Z/E, at its locant (the lower locant of the bond)."""
    if not (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE
                   for b in mol.GetBonds())):
        return ()
    try:
        from ..assembly.naming_utils import cip_locant_rank_key
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)
        items = []
        for a, loc in mapping.items():
            atom = mol.GetAtomWithIdx(a)
            if atom.HasProp('_CIPCode'):
                items.append((loc, atom.GetProp('_CIPCode')))
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (i in mapping and j in mapping
                    and b.GetBondType() == Chem.BondType.DOUBLE
                    and b.HasProp('_CIPCode')):
                items.append((min(mapping[i], mapping[j]), b.GetProp('_CIPCode')))
        return cip_locant_rank_key(items)
    except Exception:
        return ()


def _select_spiro_numbering(
    mol, candidates, key, canon_key, ring_set: Set[int], prefix_atoms,
    suffix_set: Set[int], rank_full_molecule: bool,
) -> Dict[int, int]:
    """The numbering with the lowest ``key``; a tie goes to (g), then (j),
    when the numbering is one of the whole molecule, then to the canonical ranks.

     NUMBERING (the Blue Book): "(g) lowest locants for the
    substituent cited first as a prefix in the name" (:3307), then "(j) When
    there is a choice for lower locants related to the presence of stereogenic
    centers or stereoisomers, the lower locant is assigned to CIP
    stereodescriptors Z, R, M, and r... that are preferred to E, S, P, and s"
    (:3346); '(5R,7S)-1,8-dioxadispiro[4.1.4^7.2^5]tridecane (PIN)' and
    '(1S,5R,7S)-1,7-dimethylspiro[4.5]decane (PIN)',:49166,
    :49172). The prefix names come from ``heterocycles._ring_prefix_names`` (the
    same (g) source as the heterocycle numbering); when one cannot be named, (g)
    and (j) are both skipped, so a tie (g) would break never goes to (j). (j) is
    skipped for an isotopically modified molecule, because (i) (:3336) comes
    first and this key does not model it. Only for ``rank_full_molecule``: the
    callers that number a ring-only submolecule would read prefix names and CIP
    labels of a different molecule. Without it the result equals the old
    ``min(candidates, key=(key, canon_key))``."""
    best = min(key(m) for m in candidates)
    tied = [m for m in candidates if key(m) == best]
    if len(tied) > 1 and rank_full_molecule:
        try:
            from .heterocycles import _ring_prefix_names
            names = _ring_prefix_names(
                mol, set(ring_set), set(prefix_atoms), set(suffix_set))
        except Exception:
            names = None
        if names is not None:
            isotopic = any(a.GetIsotope() for a in mol.GetAtoms())

            def _g_j(mapping: Dict[int, int]):
                g_key = tuple(sorted(
                    (name, mapping[idx])
                    for idx, lst in names.items() if idx in mapping
                    for name in lst))
                j_key = () if isotopic else _spiro_cip_locant_key(mol, mapping)
                return (g_key, j_key)

            return min(tied, key=lambda m: (_g_j(m), canon_key(m)))
    return min(tied, key=canon_key)


def _spiro_ring_traversals(
    mol, ring: List[int], spiro_center: int
) -> List[List[int]]:
    """Enumerate each directional ordering of a ring's NON-spiro atoms.

    Each ordering starts at a spiro-centre neighbour and walks away from the
    centre around the ring. A simple ring yields two orderings (one per spiro
    neighbour, i.e. the two numbering directions); each has ``len(ring) - 1``
    atoms (the spiro centre itself is excluded). Used to enumerate the
    candidate IUPAC spiro numberings so the lowest-locant rule can choose
    among them deterministically.
    """
    ring_set = set(ring)
    spiro_neighbors = [
        n.GetIdx() for n in mol.GetAtomWithIdx(spiro_center).GetNeighbors()
        if n.GetIdx() in ring_set
    ]
    traversals: List[List[int]] = []
    for start in spiro_neighbors:
        ordered = [start]
        visited = {spiro_center, start}
        current = start
        while len(ordered) < len(ring) - 1:
            nxt = None
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in ring_set and ni not in visited:
                    nxt = ni
                    break
            if nxt is None:
                break
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(ordered) == len(ring) - 1:
            traversals.append(ordered)
    return traversals


def get_spiro_numbering(
    mol, spiro_center: int, suffix_ring_atoms: Optional[Set[int]] = None,
    prefix_ring_atoms: Optional[List[int]] = None,
    rank_full_molecule: bool = False,
) -> Dict[int, int]:
    """
    Generate IUPAC numbering for a monospiro system.

    IUPAC /: numbering starts at an atom adjacent to the
    spiro centre in the SMALLER ring, proceeds around that ring, through the
    spiro centre, then around the larger ring.

    Among the directional choices (which spiro-neighbour starts each ring, and
    — when the two rings are the same size — which ring is numbered first), the
    chosen numbering gives the LOWEST locants to the heteroatoms considered
    together, then to the most senior heteroatom /,
    then — a phase SUBST-01, mirroring ``get_bicyclo_numbering`` — to the
    ``suffix_ring_atoms`` (the free valence of a spiro SUBSTITUENT,.
    A spelling-independent canonical-rank tiebreak makes the result fully
    deterministic for symmetric systems (e.g. spiro[5.5] acetals). This both
    fixes the latent SMILES-order dependence in heteroatom locants (a tetra-
    valent ``1-oxaspiro[4.5]decane`` was flipping to ``4-oxaspiro[4.5]decane``)
    and yields the correct lowest-locant PIN. Without the suffix tier, a
    symmetric carbocyclic spiro substituent flipped between equivalent locants
    (spiro[5.5]undecan-3-yl vs -9-yl) by SMILES order.

    Returns a dict mapping atom index to IUPAC locant (1-indexed), or {} when
    the spiro centre is not in exactly two rings.
    """
    ri = mol.GetRingInfo()
    rings_containing = [list(r) for r in ri.AtomRings() if spiro_center in r]
    if len(rings_containing) != 2:
        return {}
    r1, r2 = rings_containing

    # Smaller ring numbered first; equal-sized rings -> either may be first, so
    # enumerate both and let the lowest-locant rule decide.
    if len(r1) < len(r2):
        size_pairs = [(r1, r2)]
    elif len(r2) < len(r1):
        size_pairs = [(r2, r1)]
    else:
        size_pairs = [(r1, r2), (r2, r1)]

    candidates: List[Dict[int, int]] = []
    for smaller, larger in size_pairs:
        for small_seq in _spiro_ring_traversals(mol, smaller, spiro_center):
            for large_seq in _spiro_ring_traversals(mol, larger, spiro_center):
                sequence = small_seq + [spiro_center] + large_seq
                candidates.append(
                    {atom_idx: i + 1 for i, atom_idx in enumerate(sequence)}
                )
    if not candidates:
        return {}

    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    suffix_set = set(suffix_ring_atoms or ())
    prefix_list = list(prefix_ring_atoms or ())

    # Wave2: ring multiple bonds of the spiro system, so the
    # numbering choice can give them low locants. Computed once — every
    # candidate maps the same atom set.
    ring_atom_set = set(r1) | set(r2)
    mult_bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
         b.GetBondType() == Chem.BondType.DOUBLE)
        for b in bonds_of(mol)
        if b.GetBeginAtomIdx() in ring_atom_set
        and b.GetEndAtomIdx() in ring_atom_set
        and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
    ]

    def _key(mapping: Dict[int, int]):
        heteros = [
            (a, loc) for a, loc in mapping.items()
            if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
        ]
        # (1) lowest locants for ALL heteroatoms together
        het_locs = sorted(loc for _a, loc in heteros)
        # (2) then lowest locants to the most senior heteroatom (O > S >...)
        het_by_seniority = sorted(
            (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc)
            for a, loc in heteros
        )
        # (2b) (the Blue Book, '## Spiro ring systems
        # containing atoms with nonstandard bonding numbers'): "Heteroatoms having
        # nonstandard bonding numbers receive lowest locants in accordance with the
        # numbering of the corresponding spiro ring system", and
        # (:10857): "If there is a choice, lower locants are assigned to
        # heteroatoms with the higher bonding number" -- '2λ6,4λ4-dithiaspiro[5.5]
        # undecane (PIN)' (:10863). Part of the parent hydride numbering, so it
        # follows the heteroatom tiers, as does for von Baeyer rings
        # (bicyclo.get_bicyclo_numbering). Sorted (locant, -bonding number): the
        # λ atoms' locant set first, then λ6 before λ4 on a positional tie. Empty
        # without λ atoms, so their ordering is unchanged. Same λ read as the one
        # the spiro name cites (_nonstandard_bonding_number).
        lam_locs = sorted(
            (loc, -lam) for a, loc in heteros
            for lam in (_nonstandard_bonding_number(mol, a),) if lam is not None
        )
        # (3) a phase SUBST-01: lowest locants to the free-valence / suffix atoms
        # (after heteroatoms) so a symmetric spiro SUBSTITUENT is minimal AND
        # deterministic (spiro[5.5]undecan-3-yl, never -9-yl).
        suffix_locs = sorted(loc for a, loc in mapping.items() if a in suffix_set)
        # (3b) Wave2 /.2): lowest locants to ring multiple
        # bonds as a set, then to double bonds. Ranked after heteroatoms
        # and the free-valence tier: free valence
        # outranks unsaturated sites); empty for saturated systems, so their
        # ordering is unchanged.
        unsat_all = sorted(
            min(mapping[a], mapping[b]) for a, b, _is_dbl in mult_bonds
        )
        unsat_dbl = sorted(
            min(mapping[a], mapping[b]) for a, b, is_dbl in mult_bonds if is_dbl
        )
        # (3c) (the Blue Book) "(f) detachable alphabetized prefixes,
        # all considered together in a series of increasing numerical order"
        # (:3301), after the suffixes and the 'ene'/'yne' endings. ``prefix_ring_atoms``
        # lists each prefix-bearing ring atom once per prefix (empty when the
        # caller has none, so the ordering is unchanged there).
        prefix_locs = sorted(mapping[a] for a in prefix_list if a in mapping)
        return (het_locs, het_by_seniority, lam_locs, suffix_locs, unsat_all,
                unsat_dbl, prefix_locs)

    def _canon_key(mapping: Dict[int, int]):
        # (4) deterministic, spelling-independent tiebreak for symmetric rings
        seq = [a for a, _loc in sorted(mapping.items(), key=lambda kv: kv[1])]
        return tuple(canon[a] for a in seq)

    # (5) (g) and (j) for a numbering of the whole molecule.
    return _select_spiro_numbering(
        mol, candidates, _key, _canon_key, ring_atom_set, prefix_list,
        suffix_set, rank_full_molecule)


def _get_ring_adjacent_to_spiro(
    mol, ring_atoms: List[int], spiro_center: int
) -> List[int]:
    """Order ring atoms starting from atom adjacent to spiro center."""
    ring_set = set(ring_atoms)
    spiro_atom = mol.GetAtomWithIdx(spiro_center)

    ring_neighbors = []
    for neighbor in spiro_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_set:
            ring_neighbors.append(neighbor.GetIdx())

    if len(ring_neighbors) != 2:
        return ring_atoms

    start_atom = ring_neighbors[0]
    ordered = [start_atom]
    visited = {start_atom}
    current = start_atom

    while len(ordered) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                ordered.append(nbr_idx)
                visited.add(nbr_idx)
                current = nbr_idx
                break

    return ordered


def get_spiro_substituents(
    mol,
    spiro_center: int,
    atom_to_locant: Dict[int, int]
) -> Dict[int, str]:
    """
    Find substituents on a spiro system.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Dictionary mapping locant to substituent name
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    spiro_atoms = set()
    for ring in rings_containing:
        spiro_atoms.update(ring)

    substituents = {}
    for atom_idx in spiro_atoms:
        if atom_idx not in atom_to_locant:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        locant = atom_to_locant[atom_idx]
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in spiro_atoms:
                carbon_count = _count_substituent_carbons(mol, nbr_idx, spiro_atoms)
                if carbon_count > 0:
                    from ..data.chain_names import get_alkyl_name as _gal
                    substituents[locant] = _gal(carbon_count)

    return substituents


def _count_substituent_carbons(mol, start_idx: int, exclude: Set[int]) -> int:
    """Count carbon atoms in a substituent group via BFS."""
    from collections import deque

    visited: Set[int] = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def _get_alkane_name(atom_count: int) -> str:
    """Get the alkane name for a given atom count."""
    return f"{_get_chain_prefix(atom_count)}ane"


def _unsaturated_spiro_parent(
    total_atoms: int,
    double_locants: List[int],
    triple_locants: List[int],
) -> Optional[str]:
    """Unsaturated spiro parent stem per ('dec-6-ene',
    'undeca-1,8-diene'). Reuses the shared hydrocarbon-name grammar
    (composition_primitives) so the ene/yne morphology has one source of
    truth. Returns None when no chain prefix exists for ``total_atoms``."""
    from ..assembly.composition_primitives import _build_hydrocarbon_name
    stem = _get_chain_prefix(total_atoms)
    if not stem:
        return None
    return _build_hydrocarbon_name(stem, double_locants, triple_locants)


def _spiro_system_numbering(
    mol, spiro_atoms_set: Set[int], ring_atoms: Set[int],
) -> Optional[Dict[int, int]]:
    """The numbering of a pure spiro system inside the whole molecule ``mol``:
    the numbering with (f) over the exocyclic groups, then (g) and
    (j) (``_select_spiro_numbering``)."""
    prefix_atoms = _spiro_exocyclic_prefix_atoms(mol, set(ring_atoms))
    if len(spiro_atoms_set) == 1:
        return get_spiro_numbering(
            mol, next(iter(spiro_atoms_set)), prefix_ring_atoms=prefix_atoms,
            rank_full_molecule=True)
    return _get_polyspiro_numbering(
        mol, spiro_atoms_set, prefix_ring_atoms=prefix_atoms,
        rank_full_molecule=True)


def name_spiro_system(mol):
    """
    Generate the complete IUPAC name for a spiro compound.

    Handles monospiro, dispiro, trispiro hydrocarbons and heterospiro
    compounds with skeletal replacement 'a' prefixes.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is False (spiro handler does not
        discover substituents via universal pipeline), or None if not
        a valid spiro system.

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')
        >>> result = name_spiro_system(mol)
        >>> result[0]
        'spiro[4.5]decane'
    """
    descriptor = generate_spiro_descriptor(mol)
    if descriptor is None:
        return None

    spiro_atoms_set = get_spiro_atoms(mol)
    n_spiro = len(spiro_atoms_set)

    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    if n_spiro == 1:
        spiro_center = list(spiro_atoms_set)[0]
        rings_containing = [r for r in all_rings if spiro_center in set(r)]
        if len(rings_containing) != 2:
            return None
        total_atoms = len(rings_containing[0]) + len(rings_containing[1]) - 1
    else:
        spiro_ring_atoms: Set[int] = set()
        for ring in all_rings:
            ring_set = set(ring)
            if ring_set & spiro_atoms_set:
                spiro_ring_atoms |= ring_set
        total_atoms = len(spiro_ring_atoms)

    # Collect all ring atoms in the spiro system
    ring_atoms_to_check: Set[int] = set()
    for ring in all_rings:
        ring_set = set(ring)
        if ring_set & spiro_atoms_set:
            ring_atoms_to_check |= ring_set

    # Compute IUPAC numbering for the spiro system. The callers cite every
    # exocyclic group as a substituent prefix on this numbering, so (f)
    # "detachable alphabetized prefixes, all considered together in a series of
    # increasing numerical order" (the Blue Book) ranks them, then (g) and
    # (j): '(1S,5R,7S)-1,7-dimethylspiro[4.5]decane (PIN)' (:49172), not
    # '4,9-dimethyl'. The direction round each ring is free.
    atom_to_locant = _spiro_system_numbering(
        mol, spiro_atoms_set, ring_atoms_to_check) or {}

    # --- Wave2 T6a: ring unsaturation splice / ---
    # Map each ring multiple bond onto the fixed spiro numbering and emit the
    # unsaturated parent stem (spiro[4.5]dec-6-ene). The numbering itself is
    # ene-aware (get_spiro_numbering tier 3b). Anything not expressible with
    # plain consecutive locants fails closed — never a silently-saturated
    # '-ane' for an unsaturated system.
    ring_mult_bonds = []
    for bond in bonds_of(mol):
        a_idx, b_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a_idx not in ring_atoms_to_check or b_idx not in ring_atoms_to_check:
            continue
        btype = bond.GetBondType()
        if btype == Chem.BondType.SINGLE:
            continue
        if btype not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            # Aromatic / dative ring bond — not a plain ene/yne system.
            return None
        ring_mult_bonds.append((a_idx, b_idx, btype))

    if not ring_mult_bonds:
        parent_name = _get_alkane_name(total_atoms)
    else:
        if n_spiro != 1 or not atom_to_locant:
            # Polyspiro unsaturation on a dispiro+ numbering) is
            # not implemented — fail closed rather than drop the bond.
            return None
        double_locs: List[int] = []
        triple_locs: List[int] = []
        for a_idx, b_idx, btype in ring_mult_bonds:
            loc_a = atom_to_locant.get(a_idx)
            loc_b = atom_to_locant.get(b_idx)
            if loc_a is None or loc_b is None or abs(loc_a - loc_b) != 1:
                # Needs a compound locant — unsupported.
                return None
            if btype == Chem.BondType.DOUBLE:
                double_locs.append(min(loc_a, loc_b))
            else:
                triple_locs.append(min(loc_a, loc_b))
        parent_name = _unsaturated_spiro_parent(
            total_atoms, sorted(double_locs), sorted(triple_locs)
        )
        if parent_name is None:
            return None

    has_heteroatoms = any(
        mol.GetAtomWithIdx(idx).GetSymbol() != 'C'
        for idx in ring_atoms_to_check
    )

    if has_heteroatoms:
        replacement = _build_hetero_prefix(
            mol, spiro_atoms_set, ring_atoms_to_check)
        if replacement is None:
            # FAIL CLOSED. Falling through to the bare ``descriptor + parent``
            # below would emit a HYDROCARBON name for a heteroatom-containing
            # spiro system -- ``spiro[5.5]undecane`` for an Sc/V/Tl/Po ring --
            # because ``parent_name`` counts every skeletal atom while no
            # morpheme spells the heteroatom. That is the same wrong-structure
            # shape as the invented ``ala`` prefix, just silent instead of loud.
            return None
        name = f"{replacement.prefix}{descriptor}{parent_name}"
        return (name, ring_atoms_to_check, atom_to_locant, False)

    name = f"{descriptor}{parent_name}"
    return (name, ring_atoms_to_check, atom_to_locant, False)


def name_charged_spiro_system(mol, cation_idx: int) -> str:
    """ method (1): name a spiro system carrying a single cationic
    ring HETEROATOM (a quaternary onium at a spiro junction, e.g. a quaternary
    ring N+) as the neutral skeletal-replacement ('a') spiro PARENT plus the
    parent-hydride '-ium' suffix cited at the cation's spiro locant.

    A spiro-junction onium -- a ring N+ whose four bonds are ALL ring bonds --
    cannot be named by the neutralize -> re-enter route the rest of the charged
    router uses: removing the charge leaves an over-valent neutral heteroatom
    (a 4-bonded neutral N) that RDKit ``SanitizeMol`` rejects, so no neutral
    parent name is ever produced and the molecule abstains. ``name_spiro_system``
    however already names the aza-/thia-/phospha-... spiro PARENT directly off
    the CHARGED mol -- the cation reads as an ordinary skeletal heteroatom for
    the replacement prefix and the spiro numbering -- so the cation name is that
    parent with ``-<locant>-ium`` appended at the same locant.

    Method (1) (neutral 'a' parent + '-ium'/'-ylium' suffix) gives the PREFERRED
    IUPAC name; it is preferred to the 'azonia' cationic skeletal-replacement
    alternative, "Method (1) gives preferred IUPAC names", the Blue Book;
    ``1-methyl-1-azabicyclo[2.2.1]heptan-1-ium`` (PIN) vs the ``azonia`` form,
    the Blue Book). So this builder emits ``...azaspiro...-ium``, never ``...azonia-
    spiro...``.

    The cation reference-set spellings ``7-azoniadispiro[5.0.5.3]pentadecane``
    etc. are the non-PIN method-(2) alternative; the PIN this returns is
    ``6-azadispiro[5.0.5.3]pentadecan-6-ium`` (the heteroatom takes the lowest
    spiro locant,. Both spellings OPSIN-round-trip to the same
    structure, so the caller's RT gate accepts the emission and 0-wrong holds.

    Returns the candidate '-ium' name, or '' when the shape is out of scope
    (not a spiro system; the cation is not a ring heteroatom of charge +1; or
    the spiro parent cannot be named). This builder does NO OPSIN validation --
    the CALLER MUST OPSIN-round-trip the returned name against ``mol`` and
    abstain on any mismatch (0-wrong absolute).

    Examples (each OPSIN-round-trips to the input on the caller's RT gate):
        >>> from rdkit import Chem
        >>> m = Chem.MolFromSmiles('C1CCCC[N+]12CCCCC2')
        >>> cat = next(a.GetIdx for a in m.GetAtoms
        ... if a.GetFormalCharge == 1)
        >>> name_charged_spiro_system(m, cat)
        '6-azaspiro[5.5]undecan-6-ium'
    """
    try:
        atom = mol.GetAtomWithIdx(cation_idx)
    except (RuntimeError, IndexError, OverflowError):
        return ''
    # Scope: a single cationic (+1) ring HETEROATOM. A spiro ring CARBON is
    # already 4-coordinate, so a spiro-carbon cation is an over-valent (invalid)
    # centre -- and only a skeletal heteroatom takes the replacement 'a' prefix
    # this construction depends on. Both non-cases fall out here rather than
    # producing a hydrocarbon name that mis-spells the charged atom.
    if (atom.GetFormalCharge() != 1
            or not atom.IsInRing()
            or atom.GetSymbol() == 'C'):
        return ''
    if not is_spiro_system(mol):
        return ''
    # Scope to a charged SPIRO ATOM (the junction). A spiro-junction onium has
    # all four bonds in rings, so it cannot be neutralized in place (the neutral
    # atom is over-valent) and has no exocyclic substituent to sever -- exactly
    # the case the in-place / DEMOTE ring emitters decline and this builder owns.
    # A charged ring atom that is NOT a spiro junction has an in-scope degree and
    # is already named by the in-place ring-cation path, so restricting here
    # cannot override a working emission (0 regression by construction).
    if cation_idx not in get_spiro_atoms(mol):
        return ''

    result = name_spiro_system(mol)
    if result is None:
        return ''
    parent_name, ring_atoms, atom_to_locant, _subs = result
    # The cation must be a skeletal atom of the numbered spiro ring set; a cation
    # off it has no spiro locant to cite the '-ium' at.
    if cation_idx not in ring_atoms:
        return ''
    locant = atom_to_locant.get(cation_idx)
    if locant is None:
        return ''

    #: the parent-hydride '-ium' cation suffix cited at the cation's
    # skeletal locant, with elision of the parent's terminal 'e'
    # (pentadecane -> pentadecan-6-ium). An unsaturated parent already carries a
    # trailing locant on its ene/yne ending and the same elide-then-append rule
    # holds; the caller RT-gates, so any shape this simple rule spells wrongly is
    # rejected there rather than shipped.
    stem = parent_name[:-1] if parent_name.endswith('e') else parent_name
    return f"{stem}-{locant}-ium"


def _build_hetero_prefix(
    mol,
    spiro_atoms_set: Set[int],
    ring_atoms: Set[int],
) -> Optional['ReplacementPrefix']:
    """
    Build the skeletal replacement 'a' prefix for heterospiro compounds.

    Generates prefix strings like "2,8-dioxa", "2-oxa-6-thia" using
    spiro numbering locants and IUPAC priority ordering.

    Returns ``None`` -- fail closed -- whenever any skeletal non-carbon ring atom
    cannot be expressed, so the caller emits no name at all.

    Returns the shared ``ring_replacement.ReplacementPrefix``: ``.prefix`` is the
    string (byte-identical to what this builder always returned) and ``.per_atom``
    is ``(atom_idx, morpheme)`` for every heteroatom that string spells, built in
    the SAME loop that spells it. ``.unexpressed`` is always ```` -- this builder
    REFUSES rather than reporting an inexpressible atom, and the totality gate
    below is what guarantees that.

    Why the decomposition is returned and not re-derived by callers
    --------------------------------------------------------------
    A consumer that needs to know which atom each morpheme spells (the binding
    spine binds one token per replacement morpheme) previously had no source for
    it on this path, and the spiro analyzer simply reported none -- the
    ``UNBOUND_MORPHEME`` half of the `` sibling drift. The available
    shortcut -- read ``build_replacement_prefix(...).per_atom`` instead -- would
    take the STRING from this builder and the DECOMPOSITION from a different one:
    the two agree on all 14 Table-1.5 stems today, but "two tables that agree
    today" is exactly the latent wrong-prefix bug documented below for Al/In. A
    decomposition must come from the speller, so the speller returns it.

    Why the totality gate is HERE and not only in the callers
    --------------------------------------------------------
     spiro replacement is a Table-1.5 context, and that table is a CLOSED
    list. This builder has two callers: ``name_spiro_system`` (the PIN spiro
    dispatch, tried BEFORE the general engine) and
    ``vonbaeyer_universal.analyze_spiro_universal``. Only the second one checked
    ``build_replacement_prefix(...).unexpressed``, so the PIN path reached
    ``polycyclic_bridged.get_heteroatom_prefix``'s ``symbol.lower + 'a'``
    fallback and shipped ``2-alaspiro[5.5]undecane`` -- ``ala`` is not a Blue Book
    term. Gating inside the builder makes both callers safe, and any future third
    caller safe by construction, which is what the sibling-drift that caused this
    defect actually calls for.
    """
    from .ring_replacement import ReplacementPrefix, build_replacement_prefix

    n_spiro = len(spiro_atoms_set)

    if n_spiro == 1:
        spiro_center = list(spiro_atoms_set)[0]
        numbering = get_spiro_numbering(mol, spiro_center)
    else:
        numbering = _get_polyspiro_numbering(mol, spiro_atoms_set)

    if not numbering:
        return None

    # TOTALITY GATE /. The shared primitive owns the
    # Table-1.5 element set and reports every skeletal atom it cannot spell:
    # an off-table element, or an in-table one the numbering does not reach (no
    # locant to cite). Refuse before spelling anything -- a ring stem must never
    # count an atom that no morpheme in the name spells.
    totality = build_replacement_prefix(mol, numbering, set(ring_atoms))
    if totality.unexpressed:
        return None

    from ..assembly.naming_utils import get_multiplier_prefix

    heteroatom_info = []
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            locant = numbering.get(atom_idx)
            if locant is None:
                # Unreachable: the gate above already reported this atom as
                # unexpressed. Kept as an assertion of the invariant rather than
                # the silent ``continue`` that used to drop the atom here.
                return None
            heteroatom_info.append((locant, symbol, atom_idx))

    if not heteroatom_info:
        return None

    # Group each element's (locant, lambda, atom_idx) records. The lambda bonding
    # number is None for standard-valence atoms. ``atom_idx`` is
    # carried through so the per-atom decomposition below is produced by the loop
    # that spells the morpheme, not reconstructed from the locants afterwards.
    by_element: Dict[str, List[Tuple[int, Optional[int], int]]] = {}
    for locant, symbol, atom_idx in heteroatom_info:
        lam = _nonstandard_bonding_number(mol, atom_idx)
        by_element.setdefault(symbol, []).append((locant, lam, atom_idx))

    # Cite elements in skeletal-replacement seniority order
    # / hw_heteroatoms: O > S > Se > Te > N > P >... > Si > B),
    # replacing the old hard-coded list (which omitted Te/Ge/As/Sb).
    prefix_parts = []
    per_atom: List[Tuple[int, str]] = []
    for element in sort_heteroatoms_by_priority(list(by_element.keys())):
        # Sort on the LOCANT only. A numbering is a bijection, so no two atoms of
        # one element share a locant and nothing below the first key is ever
        # compared -- but the old bare ``sorted`` would have compared a
        # ``lambda`` of ``None`` against an ``int`` and raised if one ever did.
        entries = sorted(by_element[element], key=lambda e: e[0])
        # Spiro replacement is a **Table 1.5** context, so the prefix
        # comes from the Table-1.5 source only. This line used to read
        # ``get_hw_prefix(element) or get_heteroatom_prefix(element)``, consulting
        # the Hantzsch-Widman table (Table 2.4) FIRST from a non-HW context. The
        # two tables agree on all 14 elements this path can reach, so dropping the
        # HW consultation is byte-identical today -- but they disagree by design
        # for Al (``aluma`` vs ``alumina``) and In (``indiga`` vs ``inda``), which
        # made the HW-first order a latent wrong-prefix bug the moment either
        # table gained a metal. The gate above already refused anything off-table.
        prefix_name = get_heteroatom_prefix(element)
        if prefix_name is None:
            return None  # unreachable past the gate; never fabricate
        count = len(entries)
        # Shared multiplying-prefix generator (di/tri/.../hexa/hepta/octa...)
        # — replaces the old penta-capped dict that emitted the malformed
        # "6-oxa" instead of "hexaoxa" (DD7 spiro multiplier bug).
        mult = get_multiplier_prefix(count, prefix_name)
        locant_tokens = [
            (f"{loc}{_LAMBDA}{lam}" if lam is not None else str(loc))
            for loc, lam, _idx in entries
        ]
        locant_str = ','.join(locant_tokens)
        prefix_parts.append(f"{locant_str}-{mult}{prefix_name}")
        # One entry per HETEROATOM (not per distinct morpheme): a multiplied
        # morpheme -- ``dioxa`` -- is written once but spells ``oxa`` once per
        # multiplicand, so a consumer binding one token per entry matches what
        # the name says. ``prefix_name`` is the exact stem just spelled above.
        per_atom.extend((idx, prefix_name) for _loc, _lam, idx in entries)

    if not prefix_parts:
        return None

    # Join parts with hyphens; no trailing hyphen -- the last 'a' prefix
    # connects directly to the spiro descriptor (e.g., "2-oxa-6-thiaspiro")
    return ReplacementPrefix(
        prefix='-'.join(prefix_parts),
        per_atom=tuple(sorted(per_atom)),
        # This builder refuses (returns None) instead of reporting an
        # inexpressible atom -- the totality gate above is what makes that true,
        # so every atom it reaches is in ``per_atom``.
        unexpressed=(),
    )


def get_rings_from_spiro_center(
    mol, spiro_center: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get the two rings sharing a spiro center.

    Args:
        mol: RDKit Mol object
        spiro_center: Atom index of the spiro center

    Returns:
        Tuple of two ring tuples: (smaller_ring, larger_ring)

    Raises:
        ValueError: If spiro_center is not in exactly 2 rings
    """
    ri = mol.GetRingInfo()
    rings_containing = [ring for ring in ri.AtomRings() if spiro_center in ring]

    if len(rings_containing) != 2:
        raise ValueError(
            f"Spiro center at atom {spiro_center} must be in exactly 2 rings"
        )

    ring1 = tuple(rings_containing[0])
    ring2 = tuple(rings_containing[1])

    if len(ring1) <= len(ring2):
        return (ring1, ring2)
    else:
        return (ring2, ring1)


# ============================================================================
# a phase-02 — Mixed spiro/fused detector + name builder + cascade suppliers
# ============================================================================
#
# Source: 151-02-PLAN.md tasks 2-3; internal notes-B.md verdict
# (PURE_SPIRO_PARTIAL · MIXED_SPIRO_FUSED_MISSING · Q-05 NESTED_FORM_PARSEABLE);
# 151-internal notes ///; AUTONOM-1990-insights.md
#
# Design notes (codified from audit):
# * lock — `is_spiro_system` body remains byte-identical. Mixed
# cases are a SEPARATE detector (`is_mixed_spiro_fused`) and a
# SEPARATE name builder (`name_mixed_spiro_fused`).
# * + AUTONOM — name_mixed_spiro_fused implements separable
# parts: identify fused component → name via existing fused-ring
# pipeline → identify spiro side ring → name algorithmically →
# recombine with primed locant on the side-ring's spiro-attachment
# locant only (Q-05 OPSIN preview confirmed parseable).
# * — no postprocessor band-aids. When the algorithm cannot
# name a fixture, return None and let the caller log to
# AUTONOM-followups.md.
# * scope: monospiro mixed cases (1 spiro centre joining a fused
# component to a single side ring). Multi-spiro mixed cases return
# None and are logged as follow-ups.
# ============================================================================


def is_mixed_spiro_fused(mol, allow_vonbaeyer: bool = False,
                         restrict_atoms: Optional[Set[int]] = None) -> bool:
    """
    Detect a mixed spiro / fused ring system AMENABLE TO AUTONOM NAMING.

    A mixed spiro/fused system has:
      (a) exactly ONE spiro atom (scope; multi-spiro mixed →)
      (b) at least one fused-ring junction (rings share an edge)
      (c) the spiro centre cleanly separates the ring graph into
          a FUSED component (≥2 rings sharing edges) on one side AND
          a SINGLE algorithmic side ring on the other — i.e.,
          ``_classify_rings_around_spiro_center`` succeeds.
      (d) is NOT a recognized natural-product backbone (RESEARCH Pitfall 3
          false-positive guard — steroids and alkaloids may carry RDKit
          ring perception artifacts that look spiro-like).

    Mutually exclusive with `is_spiro_system` per a phase-02:
    `is_spiro_system` returns True only when n_rings == n_spiro + 1.

    Topology constraint (c) is critical for canary stability: hexacyclic
    natural-product variants (e.g., aconitane derivatives with one spiro
    centre between two multi-ring fused components) must NOT route to
    Branch 5b — they belong to the polycyclic-bridged Von Baeyer branch.
    Logged as follow-up #7 for both-sides-fused topology.

    a phase-02 / / (b).

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True iff (a) AND (b) AND (c) AND (d) hold.
    """
    if mol is None:
        return False
    spiro_atoms = get_spiro_atoms(mol)
    if restrict_atoms is not None:
        # Task C: name ONE ring system of a molecule that has several disjoint
        # spiro cores (two bridged spiro-hydantoins, a spiro core plus a
        # spiro-substituent...). Consider only the spiro atoms and rings inside
        # this system, so the sibling spiro core is not miscounted as a second
        # spiro atom (which would void this monospiro-only namer).
        spiro_atoms = {a for a in spiro_atoms if a in restrict_atoms}
    if not spiro_atoms:
        return False
    if len(spiro_atoms) != 1:
        return False  # scope: monospiro mixed only.
    ri = mol.GetRingInfo()
    n_rings_here = (sum(1 for r in ri.AtomRings() if set(r) <= restrict_atoms)
                    if restrict_atoms is not None else ri.NumRings())
    if n_rings_here <= len(spiro_atoms) + 1:
        return False  # pure spiro — defer to is_spiro_system
    # FALSE-POSITIVE GUARD per RESEARCH Pitfall 3: steroid + alkaloid
    # backbones short-circuit to False.
    from ..perception.natural_products import detect_natural_product
    if detect_natural_product(mol) is not None:
        return False
    # CANARY-STABILITY GUARD per a phase-02 (b): only claim
    # mixed-spiro-fused when the AUTONOM separable topology applies
    # AND we can actually name the fused part. This restricts the new
    # branch to canonical Q-05 OPSIN-validated forms (indoline, isoquinoline,
    # chromane, indane, tetrahydroquinoline, etc.) and lets exotic
    # large polycyclic natural products (palytoxin-class, aconitane-class)
    # continue to flow through the polycyclic-bridged Von Baeyer branch
    # they were on before Plan 151-02. Follow-up #7 + #8 lift these
    # restrictions once both-sides-fused / multi-spiro-mixed naming is
    # implemented.
    spiro_center = list(spiro_atoms)[0]
    all_rings = [list(r) for r in ri.AtomRings()
                 if restrict_atoms is None or set(r) <= restrict_atoms]
    classification = _classify_rings_around_spiro_center(
        mol, spiro_center, all_rings,
    )
    if classification is None:
        return False
    fused_rings, side_rings = classification
    if len(side_rings) != 1:
        return False
    # Cap the fused-component size at 2 rings. AUTONOM separable form
    # was conceived for benzo-5-saturated, benzo-6-saturated, and similar
    # 2-ring fused components — the catalog (FUSED_HETEROCYCLE_DATA) covers
    # exactly that surface. Larger fused parts (3+) require either
    # systematic ortho-fused naming (which the catalog miss case in
    # _name_fused_component falls through to a generic synthesis that
    # rarely produces a roundtrippable name) OR Von Baeyer treatment
    # (better preserved by the existing polycyclic-bridged dispatch).
    # Task C: the 2-ring cap is a CATALOG limit (FUSED_HETEROCYCLE_DATA covers
    # benzo-fused 2-ring surfaces). The floor's von-Baeyer route
    # (allow_vonbaeyer) names an N-ring saturated bicyclo/tricyclo component
    # through analyze_cage_universal, so the cap is lifted there; the PIN path
    # (default) keeps the conservative 2-ring bound.
    if len(fused_rings) > 2 and not allow_vonbaeyer:
        return False
    # Verify the fused component has a CATALOG name BEFORE claiming the
    # input as mixed-spiro-fused. Without this guard the dispatch hijacks
    # molecules whose fused part is uncategorized (e.g., 12-ring fused
    # natural-product backbones) and produces partial names.
    fused_named = _name_fused_component(mol, fused_rings, allow_vonbaeyer=allow_vonbaeyer)
    if fused_named is None:
        return False
    return True


def get_spiro_iupac_locants(mol) -> Optional[Dict[int, _Locant]]:
    """
    Cascade-step-6 supplier for pure spiro systems (a phase-02).

    Wraps the existing `_get_polyspiro_numbering` (multi-spiro) and
    `get_spiro_numbering` (monospiro) helpers, returning the same
    atom -> locant map shape that a phase's `_build_ring_pos`
    consumes via the `_has_iupac_locants` cascade-step-6 gate.

    Coverage invariant per Pitfall 7: returns None on partial coverage
    so the cascade-step-6 gate falls through to the sorted-int proxy.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom_idx -> int locant, covering ALL ring atoms,
        OR None if mol is not a pure spiro system OR coverage is partial.
    """
    if mol is None:
        return None
    if not is_spiro_system(mol):
        return None
    spiro_atoms = get_spiro_atoms(mol)
    if not spiro_atoms:
        return None
    ri = mol.GetRingInfo()
    ring_atoms: Set[int] = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)
    # The same numbering as name_spiro_system (f), (g), (j) over the
    # molecule's exocyclic groups), so every consumer of these locants agrees
    # with the spiro parent name.
    numbering = _spiro_system_numbering(mol, set(spiro_atoms), ring_atoms)
    if not numbering:
        return None
    if not (set(numbering.keys()) >= ring_atoms):
        return None  # Pitfall 7: partial coverage -> None
    # Filter to ring atoms only (drop exocyclic side-chain locants if any)
    return {k: v for k, v in numbering.items() if k in ring_atoms}


# ============================================================================
# AUTONOM separable-parts helpers (private)
# ============================================================================


def _classify_rings_around_spiro_center(
    mol, spiro_center: int, all_rings: List[List[int]],
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Partition rings into (FUSED_GROUP, SIDE_GROUP) at a spiro centre.

    The FUSED_GROUP is the connected component of rings sharing edges
    (fused-ring junctions, ≥2 shared atoms) reachable from one of the
    two rings at the spiro centre. The SIDE_GROUP is the connected
    component on the other side of the spiro centre.

    Returns (fused_rings, side_rings) or None if the topology does
    not separate cleanly (e.g., both sides are fused, or the spiro
    centre is not in exactly 2 rings).
    """
    rings_at_center = [r for r in all_rings if spiro_center in r]
    if len(rings_at_center) != 2:
        return None
    ring_a, ring_b = rings_at_center

    # Build fused-only adjacency (rings sharing ≥2 atoms = fused edge,
    # excluding the spiro-only adjacency at the centre itself).
    n = len(all_rings)
    fused_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            shared = set(all_rings[i]) & set(all_rings[j])
            if len(shared) >= 2:
                fused_adj[i].append(j)
                fused_adj[j].append(i)

    def _component(start_ring_idx: int) -> Set[int]:
        seen = {start_ring_idx}
        stack = [start_ring_idx]
        while stack:
            cur = stack.pop()
            for nbr in fused_adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen

    idx_a = all_rings.index(ring_a)
    idx_b = all_rings.index(ring_b)
    comp_a = _component(idx_a)
    comp_b = _component(idx_b)

    # If both rings are in the SAME fused component, the topology is
    # not a clean separable spiro/fused split — defer to.
    if comp_a == comp_b:
        return None

    # Component sizes determine which side is "fused part" (>=2 rings)
    # vs "side ring" (1 ring). Both sides must include >=1 ring;
    # at least one side must have >=2 rings (fused).
    if len(comp_a) >= 2 and len(comp_b) == 1:
        fused_idx, side_idx = comp_a, comp_b
    elif len(comp_b) >= 2 and len(comp_a) == 1:
        fused_idx, side_idx = comp_b, comp_a
    else:
        # Both sides are 1-ring (would be pure spiro, not mixed) or
        # both >=2 (two fused components on either side — exotic).
        return None

    fused_rings = [all_rings[i] for i in fused_idx]
    side_rings = [all_rings[i] for i in side_idx]
    return fused_rings, side_rings


def _extract_subfragment(
    mol, atom_indices: Set[int], skeletal_revert: bool = False,
) -> Optional[Tuple[Chem.Mol, Dict[int, int]]]:
    """Build an RDKit Mol containing only the specified atoms (and the
    bonds between them). Returns (frag_mol, orig_to_frag_idx_map)
    or None on failure.

    For each kept atom, the count of OUT-OF-FRAGMENT neighbors in the
    original molecule is added as explicit hydrogens on the fragment
    atom. This preserves valence so the fragment sanitizes cleanly and
    downstream naming helpers (which expect bare-skeleton input) see
    a chemically valid molecule.

    ``skeletal_revert`` (Task C, side-ring naming): an out-of-fragment neighbor
    reached by a DOUBLE/TRIPLE bond (an exocyclic =O / =N / =S substituent -- a
    sulfone/sulfoxide oxide, a carbonyl, an imine) is a SUBSTITUENT, not part of
    the ring skeleton. Counting it as explicit H (the default) mis-valences the
    ring atom -- a sulfone S(=O)(=O) becomes ``[SH2]`` (λ⁴) instead of the plain
    divalent ring S (λ², ``1,3-thiazolidine``), and the whole molecule then
    abstains. Under ``skeletal_revert`` such a multiple-bond substituent adds NO
    explicit H, so the ring atom reverts to its STANDARD skeletal valence (filled
    by implicit H) and the =O/=N/=S is named separately as an ``oxo``/``dioxo``
    substituent by the whole-molecule composer. Single-bond out-of-fragment
    neighbors still add one H each (unchanged). Used only by ``_name_side_ring``;
    the fused-component path keeps the degree-based default.

    The fragment is sanitized so downstream naming helpers see a
    well-formed molecule (aromaticity perception, ring info, valence).
    """
    if not atom_indices:
        return None
    rwmol = Chem.RWMol()
    orig_to_frag: Dict[int, int] = {}
    for orig_idx in sorted(atom_indices):
        atom = mol.GetAtomWithIdx(orig_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        # Hydrogen budget: existing explicit Hs + (degree - in_fragment_neighbors)
        # to compensate for the bonds we are dropping.
        original_in_frag_neighbors = sum(
            1 for n in atom.GetNeighbors()
            if n.GetIdx() in atom_indices
        )
        if skeletal_revert:
            # count only SINGLE-bond out-of-fragment neighbors; a multiple-bond
            # exocyclic substituent reverts the ring atom to skeletal valence.
            out_of_frag_neighbors = sum(
                1 for b in atom.GetBonds()
                if b.GetOtherAtom(atom).GetIdx() not in atom_indices
                and b.GetBondType() == Chem.BondType.SINGLE
            )
        else:
            out_of_frag_neighbors = atom.GetDegree() - original_in_frag_neighbors
        new_atom.SetNumExplicitHs(
            atom.GetTotalNumHs() + out_of_frag_neighbors
        )
        new_atom.SetNoImplicit(False)
        # Preserve aromaticity flag — we re-perceive after sanitize.
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        new_idx = rwmol.AddAtom(new_atom)
        orig_to_frag[orig_idx] = new_idx
    for bond in bonds_of(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in orig_to_frag and b in orig_to_frag:
            rwmol.AddBond(orig_to_frag[a], orig_to_frag[b], bond.GetBondType())
    frag = rwmol.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    return frag, orig_to_frag


def _name_vonbaeyer_fused_component(
    mol, fused_atoms: Set[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a SATURATED von-Baeyer fused/bridged component as
    ``<hetero>bicyclo[...]<parent>`` (Task C, floor-only route for the
    (c)-aliphatic spiro-of-fused bucket).

    A monospiro atom can join a von-Baeyer bicyclic/tricyclic component
    (``bicyclo[2.2.1]heptane``, ``3-azabicyclo[3.3.0]octane``,
    ``7-oxabicyclo[4.1.0]heptane``...) to a second ring. Such a component has
    no ortho-fusion / retained-catalog name, so ``_name_fused_component``'s two
    systematic branches decline and ``name_mixed_spiro_fused`` used to abstain.
     names the whole thing in the SEPARABLE form
    ``spiro[bicyclo[...]-x,y'-<comp2>]``; this builds the von-Baeyer half via
    the shared ``analyze_cage_universal`` engine (which numbers the component in
    ORIGINAL-mol indices, so the spiro-junction atom gets its von-Baeyer locant
    directly). Returns ``(name, orig_idx -> int locant)`` or None. The offer RT
    gate is the backstop (a wrong descriptor voids -> abstain, never a wrong
    ship)."""
    from .polycyclic import _build_parent_with_unsaturation
    from .vonbaeyer_universal import analyze_cage_universal
    cage = analyze_cage_universal(mol, cage_atoms=set(fused_atoms),
                                  allow_mancude=True)
    if cage is None:
        return None
    parent_block = _build_parent_with_unsaturation(
        cage.total_atoms, cage.unsaturation, fg_suffix=None)
    name = cage.hetero_prefix + cage.descriptor + parent_block
    atom_to_locant = {
        a: loc for a, loc in cage.atom_to_locant.items()
        if isinstance(loc, int)
    }
    if not (set(atom_to_locant.keys()) >= set(fused_atoms)):
        return None  # partial numbering -> fail closed
    return name, atom_to_locant


def _name_fused_component(
    mol, fused_rings: List[List[int]], allow_vonbaeyer: bool = False,
    force_vonbaeyer: bool = False,
) -> Optional[Tuple[str, Dict[int, Union[int, str]]]]:
    """Name the fused component fragment.

    ``force_vonbaeyer`` (Task C floor retry): bypass the catalog / systematic
    fusion namer entirely and name the component as its faithful kekulized
    von-Baeyer polyene. Used ONLY on the offer-RT-gated retry path for a
    molecule whose systematic fused-spiro name failed round-trip (a wrong or
    unparseable fusion descriptor) -- the von-Baeyer form is always
    constitution-faithful, so this recovers breadth at 0-wrong (uglier name).

    Returns (name, orig_atom_idx -> locant_in_name) on success, or None. A
    locant is an integer peripheral position, or a letter-suffixed fusion
    locant ('4a'/'8a') as a STRING for a ring-fusion atom.

    Strategy (mirrors the composer cascade):
      1. Build the fragment as an isolated mol.
      2. Use ``match_fused_heterocycle_core`` to fetch the catalog-supplied
         IUPAC locant mapping (which `name_fused_heterocycle` discards
         on the retained-name early-exit path).
      3. Fall back to `name_ortho_fused_bicyclic` for systematic naming
         when no catalog match.
      4. Otherwise return None (follow-up).
    """
    fused_atoms: Set[int] = set()
    for r in fused_rings:
        fused_atoms.update(r)
    extracted = _extract_subfragment(mol, fused_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}

    if force_vonbaeyer:
        return _name_vonbaeyer_fused_component(mol, fused_atoms)

    # Prefer the heterocycle catalog path first — direct call to
    # match_fused_heterocycle_core preserves the locant mapping that
    # name_fused_heterocycle's retained-name shortcut discards.
    try:
        from ..data.fused_heterocycles import match_fused_heterocycle_core
        from .fused_rings import name_fused_heterocycle, name_ortho_fused_bicyclic
    except ImportError:
        return None

    core_match = match_fused_heterocycle_core(frag)
    if core_match is not None:
        core_name, atom_mapping_in_frag, _core_smiles = core_match
        # atom_mapping_in_frag: Dict[int, int|str] — locants for each frag atom.
        # CP2b: carry a letter-suffixed fusion locant ('9a'/'4a') back as a
        # STRING (not coerced to base int), exactly as the systematic branch does
        # post-`:1704`, so a SUBSTITUENT on a ring-fusion atom is cited with its
        # real letter locant ('9a-methyl…') instead of the wrong peripheral
        # integer ('9-methyl…', a different constitution OPSIN rejects). The
        # spiro DESCRIPTOR locant stays an integer peripheral position: a spiro
        # junction that can only be numbered as a fusion atom is VOIDed
        # (fail-closed) by name_mixed_spiro_fused's guard (`:2268`) — OPSIN
        # rejects a lettered locant in the spiro slot. (The old int coercion
        # here silently mis-lettered fusion-atom decorations; the spirobi hydro
        # path takes uncoerced locants from `_component_raw_catalog_locants`
        # directly, so it is unaffected.)
        atom_to_locant_in_orig: Dict[int, Union[int, str, Tuple[int, str]]] = {}
        for frag_idx, locant in atom_mapping_in_frag.items():
            if frag_idx not in frag_to_orig:
                continue
            if isinstance(locant, int):
                atom_to_locant_in_orig[frag_to_orig[frag_idx]] = locant
            elif isinstance(locant, str):
                if locant:
                    atom_to_locant_in_orig[frag_to_orig[frag_idx]] = locant
            elif isinstance(locant, tuple) and locant:
                base = locant[0] if isinstance(locant[0], int) else 0
                if base:
                    atom_to_locant_in_orig[frag_to_orig[frag_idx]] = base
        # Orient the catalog numbering by LOWEST locants to the cited positions
        # / — mirroring the systematic branch's cited-atom
        # orientation below. A symmetric catalog core (quinolizidine's ring-swap
        # 3<->7, 2<->8,... automorphism) has several equally-established
        # numberings, and ``match_fused_heterocycle_core`` returns an arbitrary
        # one, so a spiro/substituent citation would flip locants (e.g. spiro
        # ``3'`` vs ``7'``) with SMILES atom order. Enumerate the component's
        # automorphism-equivalent numberings and pick the one giving the lowest
        # locants to the atoms that carry a decoration (the spiro junction + every
        # substituent-bearing ring atom, i.e. any component atom with a neighbour
        # OUTSIDE the component), tie-broken DETERMINISTICALLY on
        # ``Chem.CanonicalRankAtoms(breakTies=True)`` — never on raw atom index.
        cited_orig = {
            a for a in fused_atoms
            if any(nb.GetIdx() not in fused_atoms
                   for nb in mol.GetAtomWithIdx(a).GetNeighbors())
        }
        atom_to_locant_in_orig = _orient_catalog_numbering(
            frag, orig_to_frag, atom_to_locant_in_orig, cited_orig,
        )
        return core_name, atom_to_locant_in_orig

    # No catalog hit — try systematic ortho-fused naming.
    result = name_ortho_fused_bicyclic(frag)
    if result is None:
        # Try the heterocycle path one more time in case it produces a
        # name (e.g., algorithmic _try_algorithmic_fusion_name).
        result = name_fused_heterocycle(frag)
        if result is None:
            # Task C (floor-only): a SATURATED von-Baeyer fused/bridged
            # component (bicyclo/tricyclo) has no ortho-fusion name — degrade
            # to its von-Baeyer name so the separable spiro form is
            # still reachable. Gated to the best-effort floor
            # (allow_vonbaeyer); the PIN path (default False) is untouched.
            if allow_vonbaeyer:
                vb = _name_vonbaeyer_fused_component(mol, fused_atoms)
                if vb is not None:
                    return vb
            return None

    name, _ring_atoms, atom_to_locant_in_frag, _subs_included = result

    if not atom_to_locant_in_frag:
        # Systematic fallback: ``name_fused_heterocycle`` /
        # ``name_ortho_fused_bicyclic`` returned a NAME (e.g.
        # ``6,7-dihydro-5H-cyclopenta[d]pyrimidine``) but an EMPTY locant map.
        # Task C: the old fallback, ``_synthesize_fused_locants``, walked the
        # periphery WITHOUT respecting the stem's heteroatom positions, so it
        # placed the spiro carbon at locant 1 — but the NAME
        # ``cyclopenta[d]pyrimidine`` fixes a pyrimidine NITROGEN at 1, so
        # ``spiro[...-1,3'-...]`` puts the spiro junction on an N and OPSIN
        # rejects it. ``compute_fused_numbering`` is the SAME authority
        # ``name_fused_heterocycle`` used to spell the name (fused_rings.py:293/
        # 426), so its numbering is CONSISTENT with the stem — the pyrimidine
        # N's sit at 1,3 and the spiro carbon necessarily lands on a valid
        # carbon peripheral locant. ``_orient_catalog_numbering`` then picks the
        # automorphism giving the cited atoms (spiro junction + every
        # substituent-bearing ring atom) the lowest locants /
        #. Falls back to the legacy walk only if the authority
        # declines (all-carbon systems where both agree anyway).
        cited_orig = {
            a for a in fused_atoms
            if any(nb.GetIdx() not in fused_atoms
                   for nb in mol.GetAtomWithIdx(a).GetNeighbors())
        }
        # The right numbering AUTHORITY depends on whether the fused component
        # carries heteroatoms, because the two candidates disagree and each is
        # right for exactly one case:
        # * HETEROATOM-containing (cyclopenta[d]pyrimidine, benzo[d]pyrimidine,
        # furo[3,4-b]pyridine...): the heteroatoms have FIXED canonical
        # locants (N at 1,3 in a pyrimidine). ``_synthesize_fused_locants``'s
        # peripheral walk IGNORED them and put the spiro CARBON at locant 1 —
        # a nitrogen position — so OPSIN rejected the descriptor.
        # ``compute_fused_numbering`` is the authority that spelled the name
        # (fused_rings.py:293/426) and respects the heteroatom positions, so
        # the spiro carbon lands on a valid carbon locant consistent with the
        # stem.
        # * ALL-CARBON (hexahydronaphthalene, decahydronaphthalene...): there
        # is no heteroatom constraint; the name-builder oriented the hydro
        # prefix to the lowest carbon locants and the peripheral walk matches
        # it, whereas ``compute_fused_numbering`` can pick a DIFFERENT
        # automorphism (spiro 1 -> 8) that desyncs from the baked-in
        # ``1,2,3,4,5,6-hexahydro`` prefix. Keep the legacy walk here.
        # (Any residual mismatch is caught by the offer RT gate — abstain, never
        # a wrong constitution.)
        frag_has_hetero = any(
            frag.GetAtomWithIdx(i).GetSymbol() != 'C'
            for i in range(frag.GetNumAtoms())
        )
        atom_to_locant_in_frag = None
        if frag_has_hetero:
            from .fused_rings import _fused_locant_to_output
            from .fusion_numbering import compute_fused_numbering
            cfn = compute_fused_numbering(frag, set(range(frag.GetNumAtoms())))
            if cfn:
                atom_to_locant_in_frag = {
                    a: _fused_locant_to_output(loc) for a, loc in cfn.items()
                }
        if not atom_to_locant_in_frag:
            cited_frag = {orig_to_frag[a] for a in cited_orig if a in orig_to_frag}
            atom_to_locant_in_frag = _synthesize_fused_locants(frag, cited_frag)
        if not atom_to_locant_in_frag:
            return None

    # Carry the fused numbering back to original atom indices. A letter-suffixed
    # fusion locant ('4a'/'8a') is kept as a STRING so a substituent on a fusion
    # atom is cited '4a-methyl'; the spiro DESCRIPTOR locant is an integer
    # peripheral position (a spiro junction never sits on a fusion atom in this
    # ortho-fused scope, and name_mixed_spiro_fused voids the candidate if one
    # somehow did — OPSIN rejects a lettered locant in the spiro slot).
    atom_to_locant_in_orig: Dict[int, Union[int, str, Tuple[int, str]]] = {}
    for frag_idx, locant in atom_to_locant_in_frag.items():
        if frag_idx not in frag_to_orig:
            continue
        if isinstance(locant, int):
            atom_to_locant_in_orig[frag_to_orig[frag_idx]] = locant
        elif isinstance(locant, str):
            if locant:
                atom_to_locant_in_orig[frag_to_orig[frag_idx]] = locant
        elif isinstance(locant, tuple) and locant:
            base = locant[0] if isinstance(locant[0], int) else 0
            if base:
                atom_to_locant_in_orig[frag_to_orig[frag_idx]] = base

    return name, atom_to_locant_in_orig


def _orient_catalog_numbering(
    frag,
    orig_to_frag: Dict[int, int],
    atom_to_locant_in_orig: Dict[int, Union[int, str, Tuple[int, str]]],
    cited_orig: Set[int],
) -> Dict[int, Union[int, str, Tuple[int, str]]]:
    """Choose the LOWEST-locant automorphism numbering of a catalog-matched fused
    component for the CITED positions, deterministically /.

    ``match_fused_heterocycle_core`` returns ONE of a symmetric ring system's
    equally-established numberings — quinolizidine's ring-swap automorphism maps
    ``3<->7``, ``2<->8``, ``1<->9``, ``4<->6`` while fixing ``5``(N) and ``9a`` —
    picked by RDKit substructure-match order and therefore SMILES-atom-order
    dependent. A spiro / substituent citation would then flip (spiro ``3'`` vs
    ``7'``) with atom order. Enumerating the automorphism-equivalent numberings
    (``_component_numberings``) and selecting the one giving the lowest locants to
    the cited atoms (the spiro junction + every substituent-bearing ring atom) —
    tie-broken on ``Chem.CanonicalRankAtoms(breakTies=True)``, never on raw atom
    index — makes the emitted locants atom-order-invariant AND lowest (PIN). This
    mirrors the systematic branch's ``_synthesize_fused_locants`` orientation.

    Letter-suffixed fusion locants are preserved as strings. Falls back to the
    input map when there is a single numbering or no cited atom to steer it."""
    cited = {a for a in cited_orig if a in atom_to_locant_in_orig}
    if not cited:
        return atom_to_locant_in_orig
    numberings = _component_numberings(frag, orig_to_frag, atom_to_locant_in_orig)
    if len(numberings) <= 1:
        return atom_to_locant_in_orig
    try:
        ranks = list(Chem.CanonicalRankAtoms(frag, breakTies=True))
    except Exception:
        ranks = list(range(frag.GetNumAtoms()))

    def _select_key(locs: Dict[int, Union[int, str, Tuple[int, str]]]):
        # (1) lowest locants to the cited positions; (2) a fully deterministic,
        # atom-order-invariant binding of every locant to the canonical rank of
        # the atom that received it.
        cited_key = tuple(sorted(
            _locant_sort_key(locs[a]) for a in cited if a in locs))
        bind_key = tuple(sorted(
            (_locant_sort_key(loc), ranks[orig_to_frag[a]])
            for a, loc in locs.items() if a in orig_to_frag))
        return (cited_key, bind_key)

    return min(numberings, key=_select_key)


def _synthesize_fused_locants(
    frag, cited_atoms: Optional[Set[int]] = None,
) -> Dict[int, Union[int, str]]:
    """Synthesize the IUPAC fused-ring locant map for a 2-ring ortho-fused
    saturated bicyclic fragment (decalin / hydrindane / octahydropentalene
    shape) that ``name_ortho_fused_bicyclic`` named (``decahydronaphthalene``,
    ``octahydro-1H-indene``, …) but for which it supplied no locant map.

    Numbers per ** / ** (the Blue Book — a fusion
    position takes a LETTER following the preceding peripheral locant):

      * peripheral atoms get integers ``1..k`` (``k = total ring atoms - 2``);
      * each of the two ring-fusion atoms gets a **letter locant**
        ``"{preceding-peripheral}a"`` — decalin (6+6) → ``4a``/``8a``,
        octahydroindene (6+5) → ``3a``/``7a``, octahydropentalene (5+5) →
        ``3a``/``6a``. The SMALLER ring is numbered first (the retained-name
        numbering of naphthalene/indene/pentalene); equal rings are symmetric,
        so both orders are enumerated and the tie-break resolves them.

    Orientation (which peripheral is ``1``, which fusion is ``{k1}a`` vs
    ``{k1+k2}a``) is chosen by **lowest locants to the cited positions**
    (``cited_atoms`` = the spiro junction + substituent-bearing ring atoms),
    letter locants sorting ``4 < 4a < 5`` (``_locant_sort_key``). Ties are
    broken **deterministically** on ``Chem.CanonicalRankAtoms(breakTies=True)``
    — never on raw atom index / iteration order — so the emitted name is
    identical across atom orders.

    Returns ``{atom_idx -> int | '4a'-style str}`` covering all ring atoms, or
    ``{}`` when the fragment is not a clean 2-ring ORTHO-fused system (the
    caller then falls through to its existing degradation — never a fabricated
    or out-of-range integer).
    """
    ri = frag.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings()]
    if len(rings) != 2:
        return {}
    r1, r2 = rings
    shared = set(r1) & set(r2)
    if len(shared) != 2:
        return {}
    fa, fb = sorted(shared)
    # Ortho fusion only: the two shared atoms lie on a common edge. Two
    # NON-adjacent shared atoms are a bridged (von Baeyer) system whose
    # numbering is rule-fixed elsewhere — decline rather than mis-letter it.
    if frag.GetBondBetweenAtoms(fa, fb) is None:
        return {}

    ring_atoms = set(r1) | set(r2)
    adj: Dict[int, List[int]] = {a: [] for a in ring_atoms}
    for a in ring_atoms:
        for nbr in frag.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() in ring_atoms:
                adj[a].append(nbr.GetIdx())

    try:
        ranks = list(Chem.CanonicalRankAtoms(frag, breakTies=True))
    except Exception:
        ranks = list(range(frag.GetNumAtoms()))

    def _peripheral_path(ring_set, start_fusion, end_fusion):
        """Ordered non-fusion atoms of ``ring_set`` as a simple path, starting
        at the non-fusion ring atom adjacent to ``start_fusion`` and ending at
        the one adjacent to ``end_fusion``; None if it is not a clean path."""
        nonfusion = [a for a in ring_set if a not in shared]
        if not nonfusion:
            return []
        starts = [a for a in nonfusion if start_fusion in adj[a]]
        if not starts:
            return None
        cur = min(starts, key=lambda a: ranks[a])
        path = [cur]
        seen = {cur}
        while len(path) < len(nonfusion):
            nxts = [n for n in adj[cur]
                    if n in ring_set and n not in shared and n not in seen]
            if not nxts:
                break
            cur = min(nxts, key=lambda a: ranks[a])
            path.append(cur)
            seen.add(cur)
        if len(path) != len(nonfusion):
            return None
        if end_fusion not in adj[path[-1]]:
            return None
        return path

    # Smaller ring numbered first; equal size -> enumerate both orders.
    if len(r1) < len(r2):
        ring_orders = [(set(r1), set(r2))]
    elif len(r2) < len(r1):
        ring_orders = [(set(r2), set(r1))]
    else:
        ring_orders = [(set(r1), set(r2)), (set(r2), set(r1))]

    numberings: List[Dict[int, Union[int, str]]] = []
    for ring_a, ring_b in ring_orders:
        k_a = len(ring_a) - 2
        k_b = len(ring_b) - 2
        for f_mid in (fa, fb):
            f_start = fb if f_mid == fa else fa
            path_a = _peripheral_path(ring_a, f_start, f_mid)
            if path_a is None or len(path_a) != k_a:
                continue
            path_b = _peripheral_path(ring_b, f_mid, f_start)
            if path_b is None or len(path_b) != k_b:
                continue
            locs: Dict[int, Union[int, str]] = {}
            n = 1
            for a in path_a:
                locs[a] = n
                n += 1
            locs[f_mid] = f"{k_a}a"
            for a in path_b:
                locs[a] = n
                n += 1
            locs[f_start] = f"{k_a + k_b}a"
            if set(locs.keys()) >= ring_atoms:
                numberings.append(locs)

    if not numberings:
        return {}

    cited = {a for a in (cited_atoms or set()) if a in ring_atoms}

    def _select_key(locs: Dict[int, Union[int, str]]):
        # (1) lowest locants to the cited positions (spiro atom + substituents);
        # (2) a fully deterministic, atom-order-invariant binding of every
        # locant to the canonical rank of the atom that received it.
        cited_key = tuple(sorted(_locant_sort_key(locs[a]) for a in cited))
        bind_key = tuple(sorted(
            (_locant_sort_key(loc), ranks[a]) for a, loc in locs.items()))
        return (cited_key, bind_key)

    return min(numberings, key=_select_key)


def _name_side_ring(
    mol, side_ring: List[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a single algorithmic side ring.

    Returns (name, atom_idx_in_orig -> locant) or None.
    """
    side_atoms = set(side_ring)
    # skeletal_revert: a ring atom's exocyclic =O/=N/=S (sulfone/sulfoxide oxide,
    # carbonyl, imine) is a SUBSTITUENT, not a skeletal H — so the ring S of a
    # sulfone stays a plain divalent thioether (λ², ``1,3-thiazolidine``) and the
    # composer adds the ``1,1-dioxo`` separately, instead of the mis-valenced
    # ``[SH2]`` (λ⁴) the degree-based default produced (Task C).
    extracted = _extract_subfragment(mol, side_atoms, skeletal_revert=True)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted

    # Detect heteroatoms in the side ring
    has_hetero = any(
        a.GetSymbol() != "C" and a.GetIdx() in [orig_to_frag[i]
                                                for i in side_atoms]
        for a in frag.GetAtoms()
    )

    if has_hetero:
        try:
            from .heterocycles import name_heterocycle
        except ImportError:
            return None
        ring_atoms_in_frag = list(orig_to_frag[i] for i in side_atoms)
        try:
            name = name_heterocycle(frag, ring_atoms_in_frag)
        except Exception:
            return None
        if not name:
            return None
        # CT.2: for a PARTIALLY-SATURATED mancude heteromonocycle
        # (thiazoline, dihydropyran,...) ``name_heterocycle`` numbers the ring
        # through ``_mancude_hydro_select`` (the authority that owns the hydro /
        # indicated-H locants). The independent ``_walk_side_ring_locants`` walk
        # can DISAGREE with that numbering — e.g. it put the thiazoline spiro
        # atom at ``2`` (a heteroatom-adjacent carbon that cannot be the sp3
        # spiro junction) while the stem ``4,5-dihydro-1,3-thiazole`` numbers it
        # ``5``. The spiro locant is then cited inconsistently with the stem and
        # the assembled name denotes a different constitution (C=N silently
        # saturated), which rejects -> abstain. Take the map from the
        # SAME numbering authority so stem and spiro locant agree. Falls back to
        # the walk for aromatic / fully-saturated rings (unchanged).
        from .heterocycles import _mancude_hydro_numbering
        frag_locants = _mancude_hydro_numbering(frag, set(ring_atoms_in_frag))
        if frag_locants is not None:
            frag_to_orig = {v: k for k, v in orig_to_frag.items()}
            atom_to_locant = {
                frag_to_orig[fi]: loc
                for fi, loc in frag_locants.items()
                if fi in frag_to_orig
            }
        else:
            # Task C: a FULLY-SATURATED heteromonocyclic side ring (imidazolidine,
            # piperidine, pyrrolidine, 1,3-diazinane, oxane...) is NOT mancude, so
            # ``_mancude_hydro_numbering`` declines. The old fallback,
            # ``_walk_side_ring_locants``, numbered the ring by an INDEPENDENT walk
            # from one heteroatom that DISAGREED with the name: for imidazolidine
            # it placed the two ring N's at locants 1 and 4 (walking around) while
            # the NAME ``imidazolidine`` fixes N at 1,3, and it handed the spiro
            # CARBON a locant that landed on a nitrogen position (``3'``). OPSIN
            # rejects ``spiro[chromane-4,3'-imidazolidine]`` (a spiro junction on
            # an N) but accepts the name-consistent ``...-4,5'-...`` (the spiro C
            # at a carbon). ``_number_hetero_side_ring`` gives the heteroatoms
            # their canonical lowest locants, so the stem and the
            # numbering agree) and the spiro junction the lowest locant among
            # those, DETERMINISTICALLY (independent of RDKit atom order,
            # unlike ``orient_heterocycle``). (Reached only via
            # ``name_mixed_spiro_fused``; every emission is still offer-RT-gated.)
            atom_to_locant = _number_hetero_side_ring(mol, list(side_ring))
            if not atom_to_locant:
                atom_to_locant = _walk_side_ring_locants(
                    mol, side_ring, hetero_first=True,
                )
    else:
        # Carbocyclic side ring: cyclo<N>ane / cyclo<N>ene if unsaturated
        ring_size = len(side_ring)
        # Detect at least one double bond inside the ring
        has_double = False
        ring_set = set(side_ring)
        for bond in bonds_of(mol):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_set and b in ring_set:
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_double = True
                    break
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    has_double = True
                    break
        prefix = _get_chain_prefix(ring_size)
        suffix = "ene" if has_double else "ane"
        # cycloprop -> cycloprop + ane = cyclopropane
        # Strip trailing 'a' before -ane already handled by chain_names.
        name = f"cyclo{prefix}{suffix}"
        atom_to_locant = _walk_side_ring_locants(
            mol, side_ring, hetero_first=False,
        )

    return name, atom_to_locant


def _number_hetero_side_ring(mol, side_ring: List[int]) -> Optional[Dict[int, int]]:
    """Number a saturated heteromonocyclic spiro side ring DETERMINISTICALLY and
    consistently with its stem name (Task C).

    /.4 give the heteroatoms the lowest locants as a set, then the
    most-senior heteroatom the lowest locant; then gives the spiro
    junction the lowest locant among the numberings that still satisfy the
    heteroatom rule. This is the ONE numbering that is (a) consistent with the
    stem produced by ``name_heterocycle`` (so ``imidazolidine``'s two N's sit at
    1,3 and the spiro CARBON never lands on an N — the OPSIN-fatal
    ``spiro[...-3'-imidazolidine]`` defect) and (b) independent of RDKit atom
    order (``orient_heterocycle`` alone gave the spiro atom 2/3/4/5 depending on
    input order — a determinism hazard).

    Enumerates all 2N cyclic numberings (N rotations x 2 directions), keeps the
    ones minimising the heteroatom key, then picks the lowest spiro locant, then
    the lowest substituent-bearing (decorated) locants, tie-broken on a
    canonical atom-rank signature. Returns ``orig_atom_idx -> locant`` or None.
    """
    ring_set = set(side_ring)
    n = len(side_ring)
    if n < 3:
        return None
    adj: Dict[int, List[int]] = {a: [] for a in side_ring}
    for a in side_ring:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in ring_set:
                adj[a].append(nb.GetIdx())
    if any(len(v) != 2 for v in adj.values()):
        return None  # not a simple monocycle

    # single cyclic order (walk from the lowest-index atom)
    start = min(side_ring)
    order = [start, adj[start][0]]
    while len(order) < n:
        cur = order[-1]
        nxt = [x for x in adj[cur] if x != order[-2]]
        if not nxt:
            return None
        order.append(nxt[0])

    spiro_set = get_spiro_atoms(mol) & ring_set
    decorated = {
        a for a in side_ring
        if any(nb.GetIdx() not in ring_set
               for nb in mol.GetAtomWithIdx(a).GetNeighbors())
    }
    try:
        canon = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        canon = [0] * mol.GetNumAtoms()

    def numbering(rot: int, direction: int) -> Dict[int, int]:
        seq = order[rot:] + order[:rot]
        if direction == -1:
            seq = [seq[0]] + list(reversed(seq[1:]))
        return {a: i + 1 for i, a in enumerate(seq)}

    best = None
    for rot in range(n):
        for direction in (1, -1):
            m = numbering(rot, direction)
            hetero_locs = sorted(
                m[a] for a in side_ring
                if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
            )
            # senior heteroatom lowest: (seniority, locant) ascending
            senior = tuple(sorted(
                (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), m[a])
                for a in side_ring
                if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
            ))
            spiro_loc = min((m[a] for a in spiro_set), default=n + 1)
            deco = tuple(sorted(m[a] for a in decorated))
            canon_sig = tuple(m[a] for a in sorted(side_ring, key=lambda x: canon[x]))
            key = (tuple(hetero_locs), senior, spiro_loc, deco, canon_sig)
            if best is None or key < best[0]:
                best = (key, m)
    return best[1] if best else None


def _walk_side_ring_locants(
    mol, ring: List[int], *, hetero_first: bool,
) -> Dict[int, int]:
    """Number a side ring 1..N walking around the ring.

    For carbocyclic side rings, locant 1 is chosen at the spiro-attached
    atom (i.e., the atom at the spiro centre). For heterocyclic side
    rings, locant 1 is the highest-priority heteroatom (O > S > Se > N >
    P > Si > B per IUPAC. The walk direction is the one giving
    the lowest locant set for the spiro attachment atom (reuse).

    Returns Dict[atom_idx -> locant_in_side_ring].
    """
    ring_set = set(ring)

    # Build adjacency limited to ring atoms.
    adj: Dict[int, List[int]] = {a: [] for a in ring}
    for a in ring:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() in ring_set:
                adj[a].append(nbr.GetIdx())

    # Choose start atom.
    spiro_set = get_spiro_atoms(mol)
    spiro_in_ring = [a for a in ring if a in spiro_set]
    start: int
    if hetero_first:
        # a phase-04: use canonical IUPAC priority from
        # data.hw_heteroatoms (which includes halogens F < Cl < Br < I <
        # O < S <... per. The previous local dict was
        # missing halogen entries, causing F/Cl/Br/I ring atoms to fall
        # through to default priority 99 (least senior) when IUPAC
        # requires them to be MOST senior.
        hetero_candidates = sorted(
            (a for a in ring if mol.GetAtomWithIdx(a).GetSymbol() != "C"),
            key=lambda a: get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()),
        )
        if hetero_candidates:
            start = hetero_candidates[0]
        elif spiro_in_ring:
            start = spiro_in_ring[0]
        else:
            start = ring[0]
    else:
        # Carbocyclic: locant 1 at the spiro centre per spiro-descriptor
        # conventions for the side ring.
        start = spiro_in_ring[0] if spiro_in_ring else ring[0]

    # Walk the ring and assign 1, 2,..., N.
    # Two directions possible — pick the one giving the LOWEST set of
    # spiro-locants (reuse).
    candidates = []
    for first_step in adj[start]:
        path = [start, first_step]
        visited = {start, first_step}
        current = first_step
        while len(path) < len(ring):
            extended = False
            for nbr in adj[current]:
                if nbr not in visited:
                    path.append(nbr)
                    visited.add(nbr)
                    current = nbr
                    extended = True
                    break
            if not extended:
                break
        if len(path) != len(ring):
            continue
        atom_to_locant = {a: i + 1 for i, a in enumerate(path)}
        # Spiro-locant set for this ordering
        spiro_locants = sorted(atom_to_locant[a] for a in spiro_in_ring)
        candidates.append((spiro_locants, atom_to_locant))

    if not candidates:
        return {}
    # Pick lowest spiro-locant set per first-point-of-difference
    candidates.sort(key=lambda x: x[0])  # tuple comparison = first-pt-of-diff
    # Verify with compare_locant_sets to honor the / lock —
    # tuple-sort and compare_locant_sets agree on plain int lists.
    best = candidates[0]
    for cand in candidates[1:]:
        if compare_locant_sets(cand[0], best[0]) < 0:
            best = cand
    return best[1]


def _component_alpha_key(name: str) -> str:
    """Alphanumerical sort key for a spiro ring-component name /
    . Compare by the ring-component name itself, ignoring a leading
    indicated-hydrogen descriptor — e.g. ``1H-indene`` -> ``indene`` so that
    ``cyclopentane`` (c) sorts before ``indene`` (i), per the Note
    ("the first ring to be cited is determined by alphabetical order and not
    by seniority of the rings or ring systems")."""
    import re
    s = name.strip().lower()
    s = re.sub(r'^\d+h-', '', s)  # drop a leading '1h-'/'3h-' indicated H
    return s


def _bracket_component_locants(name: str) -> str:
    """Enclose the locants that belong to a spiro ring-component name in square
    brackets: ``1-benzothiophene`` -> ``[1]benzothiophene``, ``1,3-oxazole`` ->
    ``[1,3]oxazole``.

     Note (the Blue Book): "The double set of brackets in this name
    occurs because the spiro name requires them and brackets are used to enclose
    locants belonging to component names (see ", in
    '3H,3'H-2,2'-spirobi[[1]benzothiophene] (PIN)' (:10160). (:10295):
    "All locants present in bicyclic fused benzo ring component or
    Hantzsch-Widman named component are placed in brackets (without primes for
    the second component...)".

    Only the heteroatom locants that open a one-word mancude or Hantzsch-Widman
    component name are moved; a name without leading locants (``quinoline``,
    ``thieno[2,3-b]furan``, ``oxolane``) and a name carrying hydro prefixes or a
    lambda descriptor are returned unchanged.
    """
    import re
    m = re.match(r'^(\d+(?:,\d+)*)-([a-z][a-z\[\],\d-]*)$', name)
    if m is None:
        return name
    rest = m.group(2)
    if '-' in rest.split('[', 1)[0] or 'hydro' in rest:
        return name
    return f"[{m.group(1)}]{rest}"


def _strip_consumed_indicated_h(component_name: str, spiro_locant) -> str:
    """Drop a leading indicated-hydrogen descriptor (``1H-``) from a spiro
    ring-component name when the spiro atom sits at that locant.

    A mancude component such as ``1H-indene`` carries indicated hydrogen to
    place its single sp3 centre. When the quaternary spiro atom occupies that
    centre, the indicated hydrogen is no longer needed in the complete
    structure /, e.g. spiro at indene C1 -> ``indene``,
    giving ``spiro[cyclopentane-1,1'-indene]`` (NOT ``...1H-indene]``).

    Conservative: only strips when the indicated-H locant equals the spiro
    attachment locant; otherwise the name is returned unchanged (fail-safe)."""
    import re
    m = re.match(r'^(\d+)H-(.+)$', component_name)
    if m and str(spiro_locant) == m.group(1):
        return m.group(2)
    return component_name


def _prime_token(base: int, prime: str) -> str:
    """Format a locant with its prime suffix: (5, "'") -> "5'"."""
    return f"{base}{prime}"


def _extract_leading_indicated_h(name: str) -> Tuple[List[int], str]:
    """Split a leading indicated-hydrogen descriptor off a component name.
    ``"1H,3H-benzo[...]"`` -> ``([1, 3], "benzo[...]")``; ``"benzo[...]"`` ->
    ``(, "benzo[...]")``. Only strips a full ``<loc>H(,<loc>H)*-`` run."""
    import re
    m = re.match(r'^((?:\d+H,)*\d+H)-(.+)$', name)
    if not m:
        return [], name
    locs = [int(tok[:-1]) for tok in m.group(1).split(',')]
    return locs, m.group(2)


def _build_lambda_ih_front_prefix(
    mol,
    components: List[Dict],
) -> Tuple[str, Dict[int, str]]:
    """: build the combined front-of-name prefix for a λ / indicated-H
    spiro system and return ``(prefix, {component_id: stripped_name})``.

    ``components`` is an ordered list of dicts, each with keys:
      ``id`` (any hashable), ``name`` (component name), ``prime`` (prime suffix
      string: '', "'", "''", "'''"), ``spiro_atoms`` (list of the component's
      spiro atom original indices), ``map`` (orig_idx -> int locant for THIS
      component's own numbering).

    The prefix is ``<indicated-H set>-<lambda set>-`` /.5): the
    indicated-H descriptors of every component are front-cited, each locant
    carrying that component's prime, in ascending (prime, locant) order; then
    each λ (nonstandard-valence) spiro atom is cited as ``<loc><prime>lambda<n>``
    in ascending order. Empty string when neither is present."""
    ih_tokens: List[Tuple[Tuple[int, int], str]] = []
    lambda_tokens: List[Tuple[Tuple[int, int], str]] = []
    stripped: Dict = {}
    for comp in components:
        locs, bare = _extract_leading_indicated_h(comp['name'])
        stripped[comp['id']] = bare
        prime = comp['prime']
        prime_rank = len(prime)
        #: indicated-H whose locant COINCIDES with one of this component's
        # own spiro-junction atoms is CONSUMED at the junction (removed with the
        # component's name), NOT front-cited. Only indicated-H at NON-spiro
        # positions moves to the front. (e.g. 9H-fluorene / 1H-indene spiro'd at
        # 9 and 1 emit a bare 'fluorene'/'indene' with no front '9H,1''H-'; the
        # 1'H,3'H of a benzodithiophene spiro'd at 2'/6' DO front-cite.)
        comp_spiro_locs = {
            comp['map'].get(sa)
            for sa in comp.get('spiro_atoms', [])
        }
        comp_spiro_locs.discard(None)
        for loc in locs:
            if loc in comp_spiro_locs:
                continue
            ih_tokens.append(((prime_rank, loc), f"{loc}{prime}H"))
    # λ tokens: each spiro atom with a nonstandard bonding number is cited ONCE.
    # A junction λ atom is shared by two components (it has a locant/prime in
    # each); cite it at its LOWEST (locant, prime_rank) — the lowest-LOCANT rule
    # selects a terminal's low bare locant (1) over the central's
    # higher primed one (2'/6'), even when the terminal is more-primed.
    best_by_atom: Dict[int, Tuple[Tuple[int, int], str]] = {}
    for comp in components:
        prime = comp['prime']
        prime_rank = len(prime)
        for sa in comp.get('spiro_atoms', []):
            lam = _nonstandard_bonding_number(mol, sa)
            if lam is None:
                continue
            loc = comp['map'].get(sa)
            if loc is None:
                continue
            key = (loc, prime_rank)
            tok = f"{loc}{prime}{_LAMBDA}{lam}"
            if sa not in best_by_atom or key < best_by_atom[sa][0]:
                best_by_atom[sa] = (key, tok)
    lambda_tokens = list(best_by_atom.values())
    parts = []
    if ih_tokens:
        ih_tokens.sort(key=lambda t: t[0])
        parts.append(','.join(tok for _, tok in ih_tokens))
    if lambda_tokens:
        lambda_tokens.sort(key=lambda t: t[0])
        parts.append(','.join(tok for _, tok in lambda_tokens))
    prefix = '-'.join(parts) + '-' if parts else ''
    return prefix, stripped


def name_mixed_spiro_fused(
    mol, allow_vonbaeyer_component: bool = False,
    force_vonbaeyer_component: bool = False,
    restrict_atoms: Optional[Set[int]] = None,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """
    Build the AUTONOM separable-parts name for a mixed spiro/fused system.

    ``allow_vonbaeyer_component`` (Task C, best-effort FLOOR only): additionally
    name a SATURATED von-Baeyer fused/bridged component (bicyclo/tricyclo) via
    ``analyze_cage_universal``, so a spiro-of-bicyclic degrades to the
    separable ``spiro[bicyclo[...]-x,y'-<comp2>]`` covering name instead of
    abstaining. Default False keeps the PIN path byte-identical.

    Algorithm (a phase-02 + AUTONOM):
      1. Identify the spiro centre (must be exactly 1 in scope).
      2. Partition rings at the centre into FUSED component and SIDE ring.
      3. Name the fused component via existing fused-ring pipeline.
      4. Name the side ring algorithmically (cycloalkane / heterocycle).
      5. Recombine: spiro[<fused-name>-<f_loc>,<s_loc>'-<side-name>].
         The prime sits on the side-ring's spiro-attachment locant only,
         per Q-05 OPSIN preview (NESTED_FORM_PARSEABLE).
      6. (follow-up) Re-calculate unsaturation when one part becomes
         fully saturated by the spiro junction.

    Return shape MUST match name_spiro_system per composer.py:3098-3105:
        (name, ring_atoms, atom_to_locant, substituents_included=False)

    Out-of-scope for (return None, log to AUTONOM-followups):
      - Multi-spiro mixed cases (n_spiro > 1 + fused junctions).
      - Cases where _classify_rings_around_spiro_center cannot cleanly
        separate the topology (both sides fused, exotic 4-way junctions).
      - Cases where the fused component name builder declines (no
        retained name AND _name_saturated_fused_carbocyclic returns None).

    Source: 151-internal notes; AUTONOM-1990-insights.md;
            IUPAC; Q-05 OPSIN preview (internal notes-B.md).
    """
    allow_vb = allow_vonbaeyer_component or force_vonbaeyer_component
    if not is_mixed_spiro_fused(mol, allow_vonbaeyer=allow_vb,
                                restrict_atoms=restrict_atoms):
        return None

    spiro_atoms = get_spiro_atoms(mol)
    if restrict_atoms is not None:
        # Task C: name ONLY this ring system's spiro core when the molecule has
        # several disjoint spiro cores (each is named independently by the
        # per-ring-system recursion; the sibling core is a substituent branch).
        spiro_atoms = {a for a in spiro_atoms if a in restrict_atoms}
    if len(spiro_atoms) != 1:
        # scope: monospiro mixed only. Multi-spiro mixed is logged
        # to AUTONOM-followups for in Task 2 commit message.
        return None
    spiro_center = list(spiro_atoms)[0]

    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()
                 if restrict_atoms is None or set(r) <= restrict_atoms]

    classification = _classify_rings_around_spiro_center(
        mol, spiro_center, all_rings,
    )
    if classification is None:
        return None
    fused_rings, side_rings = classification

    if len(side_rings) != 1:
        return None  #: multi-side-ring mixed (rare)

    # Step 2: name the fused component
    fused_named = _name_fused_component(
        mol, fused_rings, allow_vonbaeyer=allow_vb,
        force_vonbaeyer=force_vonbaeyer_component)
    if fused_named is None:
        return None
    fused_name, fused_atom_to_locant = fused_named

    # Step 3: name the side ring
    side_named = _name_side_ring(mol, side_rings[0])
    if side_named is None:
        return None
    side_name, side_atom_to_locant = side_named

    core_ring_atoms_pin: Set[int] = set()
    for r in fused_rings:
        core_ring_atoms_pin.update(r)
    side_atoms_pin = set(side_rings[0])
    core_ring_atoms_pin |= side_atoms_pin
    if not allow_vb and restrict_atoms is None:
        # (the Blue Book): each component by its own preferred
        # (mancude) name, in alphanumerical order, with the indicated hydrogen
        # and hydro prefixes of the complete structure cited in front
        # ('2'H-spiro[cyclopentane-1,1'-isoquinoline] (PIN)',:19607).
        fused_atoms_pin = core_ring_atoms_pin - (side_atoms_pin - {spiro_center})
        st_fused = _monospiro_component_state(
            mol, fused_atoms_pin, spiro_center, fused_name,
            fused_atom_to_locant, is_cage=False)
        st_side = _monospiro_component_state(
            mol, side_atoms_pin, spiro_center, side_name,
            side_atom_to_locant, is_cage=False)
        asm = (_assemble_hoisted_monospiro(spiro_center, st_fused, st_side)
               if st_fused is not None and st_side is not None else None)
        if asm is not None:
            front, core, combined_pin, _first, _second = asm
            if set(combined_pin) >= core_ring_atoms_pin:
                return (_attach_hoisted_front(front, core), core_ring_atoms_pin,
                        combined_pin, False)
        _label_old_form = True
    else:
        # The von Baeyer floor and the per-ring-system path build only this
        # spliced form; it is decided below from the component names.
        _label_old_form = None

    # Step 4: locate the spiro centre in each part's locant map.
    f_loc = fused_atom_to_locant.get(spiro_center)
    s_loc = side_atom_to_locant.get(spiro_center)
    if f_loc is None or s_loc is None:
        return None
    # VOID + OPSIN): the spiro-descriptor locants MUST be integer
    # peripheral positions. A lettered fusion locant ('4a') is rejected by OPSIN
    # in the spiro slot, so a spiro junction that can only be numbered as a
    # ring-fusion atom fails closed (abstain) rather than emit a fabricated or
    # invalid locant — 0-wrong is absolute.
    if not isinstance(f_loc, int) or not isinstance(s_loc, int):
        return None

    # Step 5: assemble the nested form.
    # spiro[<comp1>-<l1>,<l2>'-<comp2>]
    # (+ its Note): the two ring components are cited in
    # ALPHANUMERICAL order of the component name — NOT fused-component-first
    # and NOT by ring seniority. The first-cited component is unprimed, the
    # second primed. Indicated hydrogen at the spiro locant is dropped
    # (consumed by the quaternary spiro junction;.
    fused_key = _component_alpha_key(fused_name)
    side_key = _component_alpha_key(side_name)
    fused_primed = fused_key > side_key  # fused cited SECOND iff it sorts later
    if not fused_primed:
        first_name, first_loc, second_name, second_loc = (
            fused_name, f_loc, side_name, s_loc,
        )
    else:
        first_name, first_loc, second_name, second_loc = (
            side_name, s_loc, fused_name, f_loc,
        )
    first_name = _strip_consumed_indicated_h(first_name, first_loc)
    second_name = _strip_consumed_indicated_h(second_name, second_loc)
    name = f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"

    # Build the combined atom_to_locant map for the cascade-step-6 supplier,
    # primed-consistent with the cited order: the SECOND-cited (primed)
    # component's atoms get the tuple (locant, "'") marker; the first-cited
    # (unprimed) component keeps plain integer locants. The spiro centre lives
    # in both partitions — keep its unprimed (first-cited) locant.
    unprimed_map = side_atom_to_locant if fused_primed else fused_atom_to_locant
    primed_map = fused_atom_to_locant if fused_primed else side_atom_to_locant
    combined_locants: Dict[int, _Locant] = {}
    for atom_idx, locant in unprimed_map.items():
        combined_locants[atom_idx] = locant
    for atom_idx, locant in primed_map.items():
        if atom_idx == spiro_center:
            continue
        # _build_ring_pos coerces ints to (n, '') tuples when ANY tuple is
        # present in the dict, so this primed marking is internally consistent.
        combined_locants[atom_idx] = (locant, "'")

    # Coverage invariant: the combined locant map must cover every atom of the
    # spiro CORE — the fused component plus the single side ring. Ring systems
    # attached to the core through a single bond (a PENDANT ring substituent —
    # e.g. an appended (2-hydroxyphenyl)methylidene or an arylmethyl on a spiro
    # hydantoin/oxindole) are NOT part of the spiro parent; they are named by
    # the cascade step-6 substituent supplier. Requiring the map to cover ALL
    # rings in the molecule (the old invariant) rejected every such
    # spiro-core-plus-pendant-ring molecule outright (name_mixed_spiro_fused ->
    # None -> abstain), when the correct behaviour is to OFFER the core parent
    # and let the RT gate (/ OPSIN) decide once the substituent supplier
    # has attached the pendant rings (a project rule: producers OFFER, they do not
    # RETURN a terminal None). 0-wrong is preserved: if a pendant ring (or any
    # off-core atom) cannot be named, the assembled name fails round-trip and
    # the molecule abstains — it never ships a wrong or atom-dropping
    # constitution. For a molecule with NO pendant ring, core == all rings, so
    # this is behaviour-identical to the old invariant (no regression on the
    # cases that already emitted; measured +3 RT-true / 0 wrong / 0 lost over a
    # 99-molecule spiro-of-fused abstainer sample).
    core_ring_atoms: Set[int] = set()
    for r in fused_rings:
        core_ring_atoms.update(r)
    for r in side_rings:
        core_ring_atoms.update(r)
    if not (set(combined_locants.keys()) >= core_ring_atoms):
        return None  # Pitfall 7: partial coverage of the spiro CORE -> None

    if _label_old_form is None:
        # (the Blue Book) / (:10260): the indicated
        # hydrogen and hydro prefixes of the complete structure are cited in
        # front of the spiro name; a component that keeps its own inside the
        # brackets ('spiro[2,3-dihydro-1H-indene-1,1'-cyclohexane]') is not the
        # PIN. A component without them ('bicyclo[2.2.1]heptane') leaves the
        # spliced form as it is.
        _label_old_form = any(_component_cites_added_hydrogen(n)
                              for n in (first_name, second_name))
    if _label_old_form:
        # The preferred form above could not be built: this name keeps a
        # component's own hydro prefixes / indicated hydrogen inside the
        # brackets, so it is not the PIN, the Blue Book).
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return (name, core_ring_atoms, combined_locants, False)


_ADDED_HYDROGEN_RE = re.compile(
    r"(?:^|[-\[(,])\d+[a-z]?'*H-"
    r"|\d+[a-z]?'*(?:,\d+[a-z]?'*)*-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?hydro")


def _component_cites_added_hydrogen(component_name: str) -> bool:
    """Does a spiro component name carry indicated hydrogen ('1H-indene') or hydro
    prefixes ('2,3-dihydro-1H-indene') of its own?"""
    return bool(component_name) and _ADDED_HYDROGEN_RE.search(component_name) is not None


def _partition_rings_at_spiro(
    mol, spiro_center: int, all_rings: List[List[int]],
) -> Optional[Tuple[Set[int], Set[int]]]:
    """Partition ring INDICES into the two fused components meeting at a single
    spiro atom /. Each component is the set of rings reachable
    from one of the spiro atom's two rings via fused edges (>=2 shared atoms),
    never crossing the spiro atom. Returns (comp_a_ring_idxs, comp_b_ring_idxs)
    or None when the topology is not a clean two-sided split (e.g. the two rings
    at the spiro atom belong to one fused component, or a ring is left over)."""
    rings_at = [r for r in all_rings if spiro_center in r]
    if len(rings_at) != 2:
        return None
    n = len(all_rings)
    fused_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if len(set(all_rings[i]) & set(all_rings[j])) >= 2:
                fused_adj[i].append(j)
                fused_adj[j].append(i)

    def _component(start: int) -> Set[int]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nbr in fused_adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen

    idx_a = all_rings.index(rings_at[0])
    idx_b = all_rings.index(rings_at[1])
    comp_a = _component(idx_a)
    comp_b = _component(idx_b)
    if comp_a & comp_b:
        # The two rings at the spiro atom are themselves fused into one
        # component — not a spiro-separable system (it is fused/bridged).
        return None
    if comp_a | comp_b != set(range(n)):
        return None  # a ring is unaccounted for — not a clean monospiro split
    return comp_a, comp_b


def _name_component_either(mol, ring_list, allow_vonbaeyer, force_vonbaeyer=False):
    """Name ONE spiro-side component (a list of ring atom lists): a single ring
    via ``_name_side_ring``, a multi-ring fused/bridged system via
    ``_name_fused_component``. Returns ``(name, atom_to_locant)`` or None."""
    if len(ring_list) == 1:
        return _name_side_ring(mol, ring_list[0])
    return _name_fused_component(mol, ring_list, allow_vonbaeyer=allow_vonbaeyer,
                                 force_vonbaeyer=force_vonbaeyer)


def _name_general_monospiro_fused(
    mol, allow_vonbaeyer: bool = True, force_vonbaeyer: bool = False,
    restrict_atoms: Optional[Set[int]] = None,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Task C (floor-only): the separable name for a monospiro system
    whose BOTH sides may be fused/bridged ring systems — the ``both-sides-fused``
    (c)-bucket that ``name_mixed_spiro_fused`` declines (it requires one side to
    be a single ring).

    Partitions the spiro system at the single spiro atom into two independent
    ring components (``_partition_rings_at_spiro`` over the spiro subsystem,
    pendant rings excluded), names EACH with the full ring-parent / von-Baeyer
    engine (recursively), then assembles
    ``spiro[<comp1>-x,y'-<comp2>]`` in alphanumerical component order. Every
    emission is offer-RT-gated (0-wrong). Returns the ``name_spiro_system`` shape
    or None. ``restrict_atoms`` scopes to ONE ring system of a molecule with
    several disjoint spiro cores."""
    spiro_atoms = get_spiro_atoms(mol)
    if restrict_atoms is not None:
        spiro_atoms = {a for a in spiro_atoms if a in restrict_atoms}
    if len(spiro_atoms) != 1:
        return None
    spiro_center = next(iter(spiro_atoms))
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()
                 if restrict_atoms is None or set(r) <= restrict_atoms]
    rings_at = [i for i, r in enumerate(all_rings) if spiro_center in r]
    if len(rings_at) != 2:
        return None
    # fused adjacency over all rings; the two sides are the fused-reachable
    # closures of the spiro atom's two rings (pendant ring systems, not fused to
    # either, are naturally excluded and handled by the substituent supplier).
    n = len(all_rings)
    fused_adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if len(set(all_rings[i]) & set(all_rings[j])) >= 2:
                fused_adj[i].append(j); fused_adj[j].append(i)

    def closure(start):
        seen = {start}; stack = [start]
        while stack:
            cur = stack.pop()
            for nb in fused_adj[cur]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        return seen

    comp_a = closure(rings_at[0])
    comp_b = closure(rings_at[1])
    if comp_a & comp_b:
        return None  # the two rings are themselves fused -> not spiro-separable
    a_rings = [all_rings[i] for i in comp_a]
    b_rings = [all_rings[i] for i in comp_b]
    return _assemble_monospiro_from_sides(
        mol, spiro_center, a_rings, b_rings, allow_vonbaeyer, force_vonbaeyer)


def _assemble_monospiro_from_sides(
    mol, spiro_center: int,
    a_rings: List[List[int]], b_rings: List[List[int]],
    allow_vonbaeyer: bool, force_vonbaeyer: bool = False,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Shared separable-name assembly for a monospiro system ALREADY
    partitioned at its single spiro atom into two ring sides (``a_rings`` /
    ``b_rings`` are lists of SSSR ring atom lists; the spiro atom appears in both
    sides). Names EACH side as a full parent via ``_name_component_either``
    (a single ring -> monocycle namer; a multi-ring fused/bridged system ->
    ``_name_fused_component``, which degrades a saturated von-Baeyer cage through
    ``analyze_cage_universal`` under ``allow_vonbaeyer``), orders the two
    alphanumerically Note, first-cited unprimed), primes the second and
    returns the ``name_spiro_system`` shape ``(name, core_atoms,
    combined_locants, False)`` or None.

    Factored out of ``_name_general_monospiro_fused`` so the masked-spiro namer
    (``_name_masked_spiro``, whose spiro atom is invisible to ``get_spiro_atoms``)
    reuses the identical assembly. Fail-closed on a lettered spiro locant, a
    naming decline or partial core coverage; every caller is offer-RT-gated."""
    # Scope: a pure spiro of two MONOCYCLES (spiro[4.5]decane) is a von-Baeyer
    # spiro that ``analyze_spiro_universal`` names better (single descriptor), so
    # defer to it -- otherwise this greedy separable form would intercept and
    # slightly regress the ZINC pure-spiro conversion. At least one side must be
    # a genuine fused/bridged component. (A masked spiro atom is in >=3 SSSR
    # rings, so one side always carries >=2 rings and this never fires there.)
    if len(a_rings) == 1 and len(b_rings) == 1:
        return None
    a_atoms = set().union(*a_rings)
    b_atoms = set().union(*b_rings)
    if a_atoms & b_atoms != {spiro_center}:
        return None  # the two sides must meet ONLY at the spiro atom

    a_named = _name_component_either(mol, a_rings, allow_vonbaeyer, force_vonbaeyer)
    b_named = _name_component_either(mol, b_rings, allow_vonbaeyer, force_vonbaeyer)
    if a_named is None or b_named is None:
        return None
    a_name, a_map = a_named
    b_name, b_map = b_named
    a_loc = a_map.get(spiro_center)
    b_loc = b_map.get(spiro_center)
    # spiro-descriptor locants must be integer peripheral positions;
    # a lettered fusion locant is OPSIN-invalid in the spiro slot -> fail closed.
    if not isinstance(a_loc, int) or not isinstance(b_loc, int):
        return None

    # Alphanumerical component order Note): first-cited unprimed.
    a_primed = _component_alpha_key(a_name) > _component_alpha_key(b_name)
    if not a_primed:
        first_name, first_loc, second_name, second_loc = a_name, a_loc, b_name, b_loc
        unprimed_map, primed_map = a_map, b_map
    else:
        first_name, first_loc, second_name, second_loc = b_name, b_loc, a_name, a_loc
        unprimed_map, primed_map = b_map, a_map
    first_name = _strip_consumed_indicated_h(first_name, first_loc)
    second_name = _strip_consumed_indicated_h(second_name, second_loc)
    name = f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"

    combined_locants: Dict[int, _Locant] = {}
    for atom_idx, locant in unprimed_map.items():
        combined_locants[atom_idx] = locant
    for atom_idx, locant in primed_map.items():
        if atom_idx == spiro_center:
            continue
        combined_locants[atom_idx] = (locant, "'")
    core_ring_atoms = a_atoms | b_atoms
    if not (set(combined_locants.keys()) >= core_ring_atoms):
        return None
    return (name, core_ring_atoms, combined_locants, False)


def find_masked_spiro_atoms(
    mol, restrict_atoms: Optional[Set[int]] = None,
) -> Set[int]:
    """Detect MASKED spiro atoms — a spiro atom that is ALSO a von-Baeyer
    bridgehead, so it lies in >=3 SSSR rings and ``get_spiro_atoms`` (which needs
    EXACTLY 2 SSSR-ring membership) misses it entirely.

    Root-cause detection rule (general, M4 L1a): within a connected ring system,
    the TRUE masked spiro atom is a ring atom in >=3 SSSR rings whose removal
    splits its OWN ring-atom-induced subgraph into EXACTLY 2 connected
    ring-components — a genuine spiro cut-vertex. An atom in >=3 SSSR rings whose
    removal leaves its ring system in 1 component is a bicyclo/von-Baeyer
    BRIDGEHEAD (the other bridges keep it connected), NOT spiro, and is excluded.

    The split is evaluated per-ring-system (the atom's own connected component of
    ring atoms), so a molecule with several disjoint ring systems or a pendant
    ring never miscounts. ``restrict_atoms`` scopes the whole search to one ring
    system. Returns the set of masked spiro atom indices (usually 0 or 1).
    """
    ri = mol.GetRingInfo()
    scope = set(restrict_atoms) if restrict_atoms is not None else None
    sssr = [set(r) for r in ri.AtomRings()
            if scope is None or set(r) <= scope]
    ring_atoms: Set[int] = set()
    for r in sssr:
        ring_atoms.update(r)
    # membership count over SSSR rings
    cnt: Dict[int, int] = {}
    for r in sssr:
        for a in r:
            cnt[a] = cnt.get(a, 0) + 1
    # ring-atom-induced adjacency (ring bonds only, within scope)
    adj: Dict[int, Set[int]] = {a: set() for a in ring_atoms}
    for b in bonds_of(mol):
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in ring_atoms and v in ring_atoms and b.IsInRing():
            adj[u].add(v)
            adj[v].add(u)

    def _own_system(seed: int) -> Set[int]:
        seen = {seed}
        stack = [seed]
        while stack:
            y = stack.pop()
            for z in adj[y]:
                if z not in seen:
                    seen.add(z)
                    stack.append(z)
        return seen

    def _splits_into_two(a: int) -> bool:
        system = _own_system(a)
        nodes = system - {a}
        if not nodes:
            return False
        seen: Set[int] = set()
        ncomp = 0
        for s in nodes:
            if s in seen:
                continue
            ncomp += 1
            if ncomp > 2:
                return False
            stack = [s]
            while stack:
                y = stack.pop()
                if y in seen:
                    continue
                seen.add(y)
                stack.extend((adj[y] & nodes) - seen)
        return ncomp == 2

    return {a for a in ring_atoms if cnt.get(a, 0) >= 3 and _splits_into_two(a)}


def _name_masked_spiro(
    mol, allow_vonbaeyer: bool = True, force_vonbaeyer: bool = False,
    restrict_atoms: Optional[Set[int]] = None,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """M4 L1a (best-effort FLOOR only): name a MASKED-SPIRO system — a monospiro
    whose spiro atom is ALSO a von-Baeyer bridgehead (>=3 SSSR rings), invisible
    to ``get_spiro_atoms`` so every other spiro namer bails and the whole system
    VOIDS. These are spiro-with-von-Baeyer-component systems.

    Algorithm: detect the true spiro cut-vertex (``find_masked_spiro_atoms``),
    split its ring-atom subgraph at it into two sides (each + the spiro atom),
    then hand both sides to the shared ``_assemble_monospiro_from_sides`` — which
    names each as a full parent (a von-Baeyer bicyclic via
    ``analyze_cage_universal``; a single ring via the monocycle namer) and builds
    the separable ``spiro[<sideA>-x,y'-<sideB>]`` name with the combined
    (primed-second) locant map.

    ``restrict_atoms`` scopes to ONE ring system (a molecule may hold several
    disjoint spiro cores). Returns the ``name_spiro_system`` shape or None
    (fail-closed on !=1 masked atom, a non-two-way split, a naming decline, or a
    lettered spiro locant). Every emission is offer-RT-gated (0-wrong); the PIN /
    default path never reaches this floor, so it stays byte-identical."""
    masked = find_masked_spiro_atoms(mol, restrict_atoms)
    if len(masked) != 1:
        return None
    spiro_center = next(iter(masked))

    ri = mol.GetRingInfo()
    scope = set(restrict_atoms) if restrict_atoms is not None else None
    sssr = [set(r) for r in ri.AtomRings()
            if scope is None or set(r) <= scope]
    ring_atoms: Set[int] = set()
    for r in sssr:
        ring_atoms.update(r)
    adj: Dict[int, Set[int]] = {a: set() for a in ring_atoms}
    for b in bonds_of(mol):
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in ring_atoms and v in ring_atoms and b.IsInRing():
            adj[u].add(v)
            adj[v].add(u)

    # the spiro atom's own ring system, then split at the spiro atom
    system = {spiro_center}
    stack = [spiro_center]
    while stack:
        y = stack.pop()
        for z in adj[y]:
            if z not in system:
                system.add(z)
                stack.append(z)
    nodes = system - {spiro_center}
    seen: Set[int] = set()
    sides: List[Set[int]] = []
    for s in nodes:
        if s in seen:
            continue
        comp: Set[int] = set()
        st = [s]
        while st:
            y = st.pop()
            if y in seen:
                continue
            seen.add(y)
            comp.add(y)
            st.extend((adj[y] & nodes) - seen)
        sides.append(comp)
    if len(sides) != 2:
        return None
    side_a_atoms = sides[0] | {spiro_center}
    side_b_atoms = sides[1] | {spiro_center}
    # assign each SSSR ring (of this system) to the side that contains it — a
    # ring through the spiro atom belongs to exactly one side (its non-spiro
    # atoms lie in one component); rings of other disjoint systems are excluded.
    a_rings = [list(r) for r in sssr if r <= side_a_atoms]
    b_rings = [list(r) for r in sssr if r <= side_b_atoms]
    if not a_rings or not b_rings:
        return None
    return _assemble_monospiro_from_sides(
        mol, spiro_center, a_rings, b_rings, allow_vonbaeyer, force_vonbaeyer)


_POLYSPIRO_MULT = {2: "dispiro", 3: "trispiro", 4: "tetraspiro", 5: "pentaspiro"}


def _fused_components_atoms(mol) -> List[Set[int]]:
    """Connected components of the ring-FUSED adjacency graph (rings sharing an
    edge), returned as atom sets. Spiro-joined rings (single shared atom) are in
    DIFFERENT components; a pendant ring system is its own component."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = len(rings)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if len(rings[i] & rings[j]) >= 2:
                adj[i].append(j); adj[j].append(i)
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]; atoms: Set[int] = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); atoms |= rings[x]
            stack.extend(adj[x])
        comps.append(atoms)
    return comps


def _name_linear_polyspiro_fused(
    mol, allow_vonbaeyer: bool = True, force_vonbaeyer: bool = False,
    restrict_atoms: Optional[Set[int]] = None,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Task C (floor-only): the separable name for a LINEAR polyspiro
    system — ``dispiro``/``trispiro``… of fused/ring components joined in a chain
    at ≥2 spiro atoms (the polyspiro (c)-bucket ``name_mixed_spiro_fused`` and the
    monospiro namer both decline).

    Topology: the FUSED components form a PATH graph whose edges are the spiro
    atoms (a spiro atom joins two adjacent components at a single shared atom).
    Names each component with the full ring-parent/von-Baeyer engine (the
    monospiro recursive pattern iterated along the chain), then assembles
    ``<mult>spiro[C0-l0,l1a'-C1-l1b',l2a''-C2-…]`` with prime rank = position in
    the chain and each junction citing its two components' spiro locants.

    Deterministic terminal ordering; every emission is offer-RT-gated (0-wrong).
    Returns the ``name_spiro_system`` shape or None (fail-closed on any non-path
    topology, lettered spiro locant, or naming decline)."""
    spiro_atoms = get_spiro_atoms(mol)
    if restrict_atoms is not None:
        spiro_atoms = {a for a in spiro_atoms if a in restrict_atoms}
    if len(spiro_atoms) < 2:
        return None
    comps = _fused_components_atoms(mol)
    if restrict_atoms is not None:
        comps = [c for c in comps if c <= restrict_atoms]
    ncomp = len(comps)
    # spiro-component graph: each spiro atom joins the 2 components it lies in.
    edges: List[Tuple[int, int, int]] = []  # (comp_i, comp_j, spiro_atom)
    for s in spiro_atoms:
        in_comps = [k for k, atoms in enumerate(comps) if s in atoms]
        if len(in_comps) != 2:
            return None  # a spiro atom must join exactly two fused components
        edges.append((in_comps[0], in_comps[1], s))
    # nodes of the spiro core = components touched by a spiro atom
    core_nodes = set()
    for i, j, _s in edges:
        core_nodes.add(i); core_nodes.add(j)
    # the core must be a simple PATH: |edges| == |nodes|-1, all degree <= 2,
    # exactly two endpoints of degree 1, and connected.
    if len(edges) != len(core_nodes) - 1:
        return None
    deg: Dict[int, int] = {k: 0 for k in core_nodes}
    nbr: Dict[int, List[Tuple[int, int]]] = {k: [] for k in core_nodes}
    for i, j, s in edges:
        deg[i] += 1; deg[j] += 1
        nbr[i].append((j, s)); nbr[j].append((i, s))
    endpoints = [k for k in core_nodes if deg[k] == 1]
    if len(endpoints) != 2 or any(deg[k] > 2 for k in core_nodes):
        return None
    # order the chain from a deterministic endpoint: name both terminals, start
    # from the alphanumerically-smaller (tie -> lower min atom index).
    def _term_name(k):
        r = _name_component_either_atoms(mol, comps[k], allow_vonbaeyer, force_vonbaeyer)
        return r
    end_named = {k: _term_name(k) for k in endpoints}
    if any(v is None for v in end_named.values()):
        return None
    e0, e1 = endpoints
    key0 = (_component_alpha_key(end_named[e0][0]), min(comps[e0]))
    key1 = (_component_alpha_key(end_named[e1][0]), min(comps[e1]))
    start = e0 if key0 <= key1 else e1
    # walk the path
    order: List[int] = [start]
    junction: List[int] = []  # spiro atom between order[i] and order[i+1]
    prev = None
    cur = start
    while True:
        nxts = [(k, s) for (k, s) in nbr[cur] if k != prev]
        if not nxts:
            break
        k, s = nxts[0]
        junction.append(s); order.append(k); prev, cur = cur, k
    if len(order) != len(core_nodes):
        return None
    # Scope (the two-path rule): the SEPARABLE named form is for a
    # chain with at least one POLYCYCLIC (fused/bridged, must-be-named)
    # component. An ALL-MONOCYCLIC dispiro/polyspiro takes the numeric
    # ``dispiro[a.b.c.d]`` von-Baeyer path instead, which
    # ``analyze_spiro_universal`` already produces and which is its PIN -- so
    # defer to it rather than ship (or block it with) a separable form OPSIN may
    # reject. Count rings per component.
    ri2 = mol.GetRingInfo()
    def _ring_count(atoms):
        return sum(1 for r in ri2.AtomRings() if set(r) <= atoms)
    if all(_ring_count(comps[k]) == 1 for k in order):
        return None
    # name every component; collect its spiro-atom -> locant map
    named = []
    for k in order:
        r = _name_component_either_atoms(mol, comps[k], allow_vonbaeyer, force_vonbaeyer)
        if r is None:
            return None
        named.append(r)  # (name, atom_to_locant)
    # spiro locants must be integer peripheral positions in BOTH components
    for pos, s in enumerate(junction):
        li = named[pos][1].get(s)
        lj = named[pos + 1][1].get(s)
        if not isinstance(li, int) or not isinstance(lj, int):
            return None
    # assemble '<mult>spiro[C0-l0,l1a'-C1-l1b',l2a''-C2-...]'
    mult = _POLYSPIRO_MULT.get(len(junction))
    if mult is None:
        return None

    def _pr(n):  # prime string of rank n
        return "'" * n

    c0_name = _strip_consumed_indicated_h(named[0][0], named[0][1][junction[0]])
    tokens = [c0_name]
    for pos, s in enumerate(junction):
        left = f"{named[pos][1][s]}{_pr(pos)}"
        right = f"{named[pos + 1][1][s]}{_pr(pos + 1)}"
        comp_name = named[pos + 1][0]
        # strip indicated-H consumed by THIS component's spiro junction(s)
        comp_name = _strip_consumed_indicated_h(comp_name, named[pos + 1][1][s])
        tokens.append(f"-{left},{right}-{comp_name}")
    name = f"{mult}[{''.join(tokens)}]"

    # combined locants: component at chain position p gets prime rank p; a shared
    # spiro atom is kept at its LOWER-prime component's locant.
    combined_locants: Dict[int, _Locant] = {}
    core_ring_atoms: Set[int] = set()
    for p, k in enumerate(order):
        core_ring_atoms |= comps[k]
        amap = named[p][1]
        for atom_idx, locant in amap.items():
            if atom_idx in combined_locants:
                continue  # already assigned by a lower-prime component (spiro atom)
            # Keep EVERY ring atom, including a fused component's ring-FUSION
            # atoms whose locant is a lettered string ('4a', '8a'). Dropping
            # them here (the old ``isinstance(locant, int)`` skip) made the
            # coverage invariant below fail for any spiro chain that includes a
            # fused component (e.g. a chromane / benzopyran side), so the whole
            # core abstained. The spiro-DESCRIPTOR junction locants are proven
            # integer separately above; this map only feeds
            # substituent placement, and ``_locant_display`` renders a primed
            # lettered locant (('8a', "'") -> "8a'") correctly.
            combined_locants[atom_idx] = locant if p == 0 else (locant, _pr(p))
    if not (set(combined_locants.keys()) >= core_ring_atoms):
        return None
    return (name, core_ring_atoms, combined_locants, False)


def _name_component_either_atoms(mol, comp_atoms: Set[int], allow_vonbaeyer, force_vonbaeyer=False):
    """Name ONE component given its ATOM set: a single ring via
    ``_name_side_ring``, a multi-ring fused/bridged system via
    ``_name_fused_component``. Returns ``(name, atom_to_locant)`` or None."""
    ri = mol.GetRingInfo()
    ring_list = [list(r) for r in ri.AtomRings() if set(r) <= comp_atoms]
    if not ring_list:
        return None
    return _name_component_either(mol, ring_list, allow_vonbaeyer, force_vonbaeyer)


def _canonical_spiro_locant(extracted, loc_map: Dict[int, int], spiro_center: int):
    """Lowest locant the spiro atom may take given the component's symmetry
    . ``extracted`` is the (frag_mol, orig_to_frag) tuple from
    ``_extract_subfragment``; atoms sharing the spiro atom's CanonicalRankAtoms
    class are symmetry-equivalent (a valid alternative numbering), so the spiro
    atom's deterministic locant is the minimum locant over that orbit."""
    frag, orig_to_frag = extracted
    if spiro_center not in orig_to_frag:
        return loc_map.get(spiro_center)
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    try:
        ranks = list(Chem.CanonicalRankAtoms(frag, breakTies=False))
    except Exception:
        return loc_map.get(spiro_center)
    target_rank = ranks[orig_to_frag[spiro_center]]
    candidate_locants = []
    for frag_idx, rank in enumerate(ranks):
        if rank != target_rank:
            continue
        orig = frag_to_orig.get(frag_idx)
        if orig is not None and orig in loc_map:
            candidate_locants.append(loc_map[orig])
    if not candidate_locants:
        return loc_map.get(spiro_center)
    # A component map may now carry letter-suffixed fusion locants ('4a'); sort
    # via _locant_sort_key so min never compares int with str. The spiro atom
    # is a peripheral (integer) position in scope, so the result is an int.
    return min(candidate_locants, key=_locant_sort_key)


def _locant_sort_key(loc) -> Tuple[int, str]:
    """Order a locant that may be a plain int or a letter-suffixed fusion locant
    (``'4a'``): primary numeric part, then the letter tail (``4`` < ``4a``)."""
    if isinstance(loc, int):
        return (loc, "")
    s = str(loc)
    digits = "".join(c for c in s if c.isdigit())
    tail = "".join(c for c in s if not c.isdigit())
    return (int(digits) if digits else 0, tail)


def _reanchor_locmap_to_canonical_spiro(
    mol, extracted, loc_map: Dict[int, int], spiro_center: int, canonical_loc,
):
    """Canonicalise a spiro component's ``{orig_idx: locant}`` map by choosing,
    among the component fragment's graph automorphisms, the numbering that (1)
    places the spiro atom at ``canonical_loc`` (the descriptor locant from
    ``_canonical_spiro_locant``) and (2) gives the LOWEST locants to the
    substituent-bearing ring atoms lowest-locants rule), broken
    deterministically so the emitted name is atom-order independent.

    WHY (spiro-hoist). Two coupled defects on the component-spiro path, both
    invisible until a decorated spiro-of-fused actually emits:

    * DESCRIPTOR/MAP INCONSISTENCY. The spiro descriptor locant is the lowest over
      the spiro atom's symmetry orbit, but a fused-ring catalog numbers
      the spiro atom at a FIXED, possibly-different locant. When they disagree the
      substituent citations hoisted from this map (via
      ``composer._integrate_universal_prefixes`` consuming ``combined_locants``)
      contradict the descriptor -- fluorescein's benzofuranone carbonyl was cited
      ``1-oxo`` while the descriptor said ``spiro[...-1,...]``, an impossible
      collision OPSIN rejects, so the dye abstained. Pinning the spiro atom to
      ``canonical_loc`` and permuting every other atom along the same automorphism
      repairs it (carbonyl 1->3, ring carboxy 6->5).

    * SYMMETRIC-COMPONENT NON-DETERMINISM. A symmetric fused component (xanthene's
      two equivalent benzo rings) admits several automorphisms that all satisfy
      the spiro constraint; which original substituent lands on which locant then
      depended on RDKit atom order (``3'-methoxy-6'-phosphonooxy`` vs the swap).
      The substituent-locant tie-break picks one deterministically.

    Determinism: the automorphism set and the tie-break (lowest substituent-locant
    tuple, then lowest full-locant tuple) are both atom-order invariant.

    Fail-safe / strict-improvement, scoped by the caller to non-von-Baeyer
    components (a cage's numbering is rule-fixed, not automorphism-free). Returns
    the input map when it is already canonical (a clean canary stays byte-
    identical) or when no automorphism satisfies the spiro constraint (the RT gate
    then abstains, exactly as before). Every candidate is still RT-gated
    downstream, so no emission is ever made wrong.
    """
    frag, orig_to_frag = extracted
    if spiro_center not in orig_to_frag:
        return loc_map
    target = canonical_loc if canonical_loc is not None \
        else loc_map.get(spiro_center)
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    # Whole-molecule canonical ranks (atom-order invariant): used to give each
    # substituent-bearing ring atom an order-invariant SIGNATURE, so a symmetric
    # component (xanthene's two equal benzo rings) assigns a given substituent to
    # a given locant deterministically -- the lowest locant going to the
    # lowest-canonical-rank substituent (a stable proxy for the first-cited
    # rule; PIN-preference is not guaranteed, but the RT gate backstops and the
    # result is reproducible across atom orders, which is the hard requirement).
    try:
        _ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    except Exception:
        _ranks = [0] * mol.GetNumAtoms()
    # Ring atoms carrying an exocyclic (substituent / suffix) neighbour -- the
    # atoms whose locants the lowest-locants rule minimises. ``sub_sig`` is the
    # minimum canonical rank over each atom's exocyclic neighbours.
    sub_atoms = set()
    sub_sig: Dict[int, int] = {}
    for orig in loc_map:
        if orig == spiro_center:
            continue
        exo_ranks = [
            _ranks[nb.GetIdx()]
            for nb in mol.GetAtomWithIdx(orig).GetNeighbors()
            if nb.GetIdx() not in loc_map and nb.GetIdx() != spiro_center
        ]
        if exo_ranks:
            sub_atoms.add(orig)
            sub_sig[orig] = min(exo_ranks)
    try:
        autos = frag.GetSubstructMatches(
            frag, uniquify=False, maxMatches=5000, useChirality=False)
    except Exception:
        return loc_map
    best = None
    best_key = None
    for auto in autos:
        induced: Dict[int, int] = {}
        ok = True
        for orig, loc in loc_map.items():
            fi = orig_to_frag.get(orig)
            if fi is None or fi >= len(auto):
                ok = False
                break
            img = frag_to_orig.get(auto[fi])
            if img is None or img not in loc_map:
                ok = False
                break
            induced[orig] = loc_map[img]
        if not ok:
            continue
        if target is not None and induced.get(spiro_center) != target:
            continue
        # (locant, substituent-signature) pairs: minimise the locants first
        # (lowest-locants rule), then bind each locant to a specific substituent
        # deterministically (breaks the symmetric-component atom-order tie).
        sub_key = tuple(sorted(
            (_locant_sort_key(induced[o]), sub_sig[o]) for o in sub_atoms))
        all_key = tuple(sorted(_locant_sort_key(v) for v in induced.values()))
        key = (sub_key, all_key)
        if best_key is None or key < best_key:
            best_key = key
            best = induced
    return best if best is not None else loc_map


# --------------------------------------------------------------------------
# -- hydro / indicated-hydrogen hoisting for 'spirobi' components
#
# (the Blue Book): "Where appropriate the maximum number of noncumulative
# double bonds is added (i.e., the system is made mancude) AFTER CONSTRUCTION OF
# THE COMPLETE SKELETON. Indicated hydrogen of individual components is
# not cited. No indicated hydrogen is cited when none is present in the spiro
# system. If indicated hydrogen is needed, it is cited in front of the spiro atom
# locants."
#
# Two consequences drive everything below:
#
# 1. The bracket holds the MANCUDE component ring system -- never a hydro form
# and never the component's own indicated hydrogen. Saturation is expressed
# by 'hydro' prefixes on the ASSEMBLED name, outside the bracket.
# Template, the Blue Book: 1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole]
# -> [hydro][indicated-H][spiro locants]-spirobi[mancude component].
# All ~20 'spirobi' examples in the Blue Book cite indicated hydrogen
# OUTSIDE the bracket; NONE cites it inside.
#
# 2. Because the skeleton is made mancude as a WHOLE, the hydro locants are
# NOT the component's own hydro locants. The spiro atom has four single
# ring bonds, so it is sp3 by construction and takes no double bond; the
# maximum matching over the REMAINING component atoms is the mancude form.
# Atoms left unmatched are indicated hydrogen; atoms saturated in the real
# molecule but matched in the mancude form are the 'hydro' positions.
#
# Validated against five Blue Book spirobi PINs before use (see the module test):
# the Blue Book 1,1'-spirobi[indene] (spiro at the sp3, no IH)
# the Blue Book 1H,1'H-2,2'-spirobi[naphthalene] (9 non-spiro atoms -> 1 IH)
# the Blue Book 3H,3'H-2,2'-spirobi[[1]benzothiophene] (divalent S ineligible)
# the Blue Book 1'H,2H-1,2'-spirobi[azulene] (IH locant + prime order)
# the Blue Book 2'H,3H-2,3'-spirobi[[1]benzothiophene] (isolated C2' forced IH)
# --------------------------------------------------------------------------

# The mancude matching recursion is exponential in the worst case; spirobi
# components are small fused ring systems, so cap and fail closed above it.
_MANCUDE_MATCH_ATOM_CAP = 32


def _ring_skeleton_smiles(frag, carbon_skeleton: bool = False) -> str:
    """Canonical SMILES of a ring component's SKELETON: every bond single, no
    aromaticity, no hydrogen, no charge. With ``carbon_skeleton`` every atom is
    also carbon -- the 'saturated bi- or polycyclic alicyclic hydrocarbon' of
     (the Blue Book) that skeletal replacement ('a') nomenclature
    starts from."""
    rw = Chem.RWMol()
    for atom in frag.GetAtoms():
        new = Chem.Atom(6 if carbon_skeleton else atom.GetAtomicNum())
        new.SetNoImplicit(True)
        rw.AddAtom(new)
    for bond in frag.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw.GetMol(), canonical=True)


def _spirobi_locant_key(loc, primes: int = 0) -> Tuple[int, str, int]:
    """ sort key for a locant of an assembled spiro system.

    ** "Lowest set of locants"** (``the Blue Book Blue Book``):
    *"Primed locants are placed immediately after the corresponding unprimed
    locants in a set arranged in ascending order; locants consisting of a number
    and a lower-case letter with or without primes as 4a and 4'a (not 4a') are
    placed immediately after the corresponding numeric locant"*.

    So the order is ``4 < 4' < 4a < 4'a < 5`` -- keyed ``(number, letter,
    primes)``. It is NOT "every unprimed locant before every primed locant":
    the spirobi worked example ``the Blue Book``
    ``2-phospha-3,3'-spirobi[bicyclo[3.3.1]nonane]-6',7-diene (PIN)`` cites
    ``6'`` BEFORE ``7``, and ``the Blue Book`` ``1'H,2H-1,2'-spirobi[azulene] (PIN)``
    cites ``1'H`` before ``2H``.
    """
    if isinstance(loc, int):
        return (loc, '', primes)
    text = str(loc)
    digits = 0
    while digits < len(text) and text[digits].isdigit():
        digits += 1
    if digits == 0:
        return (10 ** 6, text, primes)
    return (int(text[:digits]), text[digits:], primes)


def _spirobi_locant_display(loc, primes: int = 0) -> str:
    """Render a locant with ``primes`` prime marks: ``(3, 1)`` -> ``3'``.

    The prime goes after the NUMBER, not after the whole locant:
    (``the Blue Book``) spells the primed fusion locant ``4'a`` and says explicitly
    ``(not 4a')``.
    """
    text = str(loc)
    digits = 0
    while digits < len(text) and text[digits].isdigit():
        digits += 1
    return f"{text[:digits]}{chr(39) * primes}{text[digits:]}"


def _mancude_max_matching(adj, nodes, key_of):
    """Maximum-cardinality matching over ``nodes``; among the maximum matchings,
    minimise the sorted tuple of UNMATCHED locant keys lowest set).

    Returns ``(pairs, unmatched)``. Each pair becomes one noncumulative double
    bond of the mancude system; each unmatched atom is an indicated-hydrogen
    position.

    Same recursion as ``partial_saturation._max_oxo_matching``, extended to
    carry the matched PAIRS -- needed here because the mancude component has to
    be BUILT (bond orders set), not merely counted. Neighbour lists are sorted
    so the result is order-independent.
    """
    from functools import lru_cache

    node_set = frozenset(nodes)

    @lru_cache(maxsize=None)
    def rec(avail):
        if not avail:
            return (0, (), ())
        first = min(avail)
        rest = avail - {first}
        size, unmatched_key, pairs = rec(rest)
        best = (size, tuple(sorted(unmatched_key + (key_of[first],))), pairs)
        for other in adj[first]:
            if other not in rest:
                continue
            size2, unmatched_key2, pairs2 = rec(rest - {other})
            cand = (size2 + 1, unmatched_key2, pairs2 + ((first, other),))
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand
        return best

    _size, _key, pairs = rec(node_set)
    matched = {atom for pair in pairs for atom in pair}
    return list(pairs), set(node_set) - matched


def _component_ring_degree(mol, idx: int, comp_atoms: Set[int]) -> int:
    """Number of the atom's neighbours that lie inside this component."""
    return sum(1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
               if nbr.GetIdx() in comp_atoms)


def _can_bear_ring_double_bond(mol, idx: int, comp_atoms: Set[int]) -> bool:
    """True when this skeletal atom has spare standard valence for one ring
    double bond -- the eligibility test for the mancude assignment.

    A divalent ring O or S (furan, thiophene, and the O atoms of
    [1,3,2]benzodioxathiole) has none, which is why such atoms are never doubly
    bonded in a mancude system and never carry indicated hydrogen. A ring N with
    two ring bonds does have spare valence (pyridine-type when matched,
    pyrrole-type N-H when unmatched, hence ``1H-indole``).

    Charged skeletons are declined: their valence bookkeeping is not derived
    here, and this predicate must never over-report eligibility.
    """
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0:
        return False
    try:
        default = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    except Exception:
        return False
    if default < 0:
        return False
    return default - _component_ring_degree(mol, idx, comp_atoms) >= 1


def _is_saturated_in_component(mol, idx: int, comp_atoms: Set[int]) -> bool:
    """True when the atom sits in no multiple or aromatic bond of its component
    (i.e. it is sp3 in the real molecule)."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetIsAromatic():
        return False
    for bond in atom.GetBonds():
        if bond.GetOtherAtomIdx(idx) not in comp_atoms:
            continue
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
                                  Chem.BondType.AROMATIC):
            return False
    return True


def _component_eligible_adjacency(mol, atoms: Set[int], comp_atoms: Set[int]):
    """``(eligible_atoms, adjacency)`` restricted to atoms of ``atoms`` that can
    bear a ring double bond. Adjacency lists are sorted (determinism)."""
    eligible = sorted(a for a in atoms
                      if _can_bear_ring_double_bond(mol, a, comp_atoms))
    elig = set(eligible)
    adj = {a: sorted(nbr.GetIdx() for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
                     if nbr.GetIdx() in elig)
           for a in eligible}
    return eligible, adj


def _component_raw_catalog_locants(mol, comp_atoms: Set[int]):
    """Raw (UNCOERCED) catalog locants for a spirobi component.

    ``_name_fused_component`` coerces lettered fusion locants to their base
    integer (``'7a'`` -> ``7``), which is harmless for a spiro descriptor but
    would silently mis-spell a hydro prefix sitting on a fusion atom. The hydro
    prefix needs the real locant, so the catalog is consulted directly here.

    Returns ``(frag, orig_to_frag, {orig_atom: raw_locant})`` or None.
    """
    extracted = _extract_subfragment(mol, comp_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    try:
        from ..data.fused_heterocycles import match_fused_heterocycle_core
    except ImportError:
        return None
    match = match_fused_heterocycle_core(frag)
    if match is None:
        return None
    _core_name, frag_locants, _core_smiles = match
    raw: Dict[int, object] = {}
    for frag_idx, locant in frag_locants.items():
        orig = frag_to_orig.get(frag_idx)
        if orig is None:
            continue
        raw[orig] = locant
    if set(raw) != set(comp_atoms):
        return None
    return frag, orig_to_frag, raw


def _component_numberings(frag, orig_to_frag, raw_locants):
    """Every valid numbering of the component ring system, as
    ``{orig_atom: raw_locant}`` maps.

    : *"The established numbering system of the polycyclic ring system
    component is retained"*. Where the ring system is symmetric several
    numberings are equally established (indane positions 1 and 3 are mirror
    images), and the spiro rules then choose between them. The alternatives are
    exactly ``raw_locants o sigma`` for the automorphisms sigma of the component,
    so they are enumerated from the fragment's self-matches rather than guessed.
    """
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    try:
        automorphisms = frag.GetSubstructMatches(
            frag, uniquify=False, useChirality=False, maxMatches=64)
    except Exception:
        automorphisms = ()
    if not automorphisms:
        return [dict(raw_locants)]
    out = []
    seen = set()
    for match in automorphisms:
        candidate = {}
        ok = True
        for orig, frag_idx in orig_to_frag.items():
            if frag_idx >= len(match):
                ok = False
                break
            image = frag_to_orig.get(match[frag_idx])
            if image is None or image not in raw_locants:
                ok = False
                break
            candidate[orig] = raw_locants[image]
        if not ok or len(set(map(str, candidate.values()))) != len(candidate):
            continue
        fingerprint = tuple(sorted((k, str(v)) for k, v in candidate.items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(candidate)
    return out or [dict(raw_locants)]


def _mancude_component_name(mol, comp_atoms: Set[int], raw_locants) -> Optional[str]:
    """Name the MANCUDE ring system of a spirobi component, with the
    component's own indicated hydrogen removed: *"Indicated hydrogen
    of individual components is not cited"*).

    The mancude form is BUILT from the component skeleton -- every ring bond
    reset to single, then the maximum noncumulative double-bond assignment
    applied -- and named through the ordinary fused-ring cascade, so the answer
    comes from the ring-system catalog rather than from string surgery on the
    saturated component's name. ``indane`` -> ``1H-indene`` -> ``indene``.
    """
    extracted = _extract_subfragment(mol, comp_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_atoms = set(orig_to_frag.values())
    if len(frag_atoms) > _MANCUDE_MATCH_ATOM_CAP:
        return None
    # Eligibility / adjacency in FRAGMENT index space.
    eligible = sorted(
        f for orig, f in orig_to_frag.items()
        if _can_bear_ring_double_bond(mol, orig, comp_atoms))
    elig = set(eligible)
    adj = {f: sorted(nbr.GetIdx() for nbr in frag.GetAtomWithIdx(f).GetNeighbors()
                     if nbr.GetIdx() in elig)
           for f in eligible}
    if not eligible:
        return None
    key_of = {f: _spirobi_locant_key(raw_locants[orig])
              for orig, f in orig_to_frag.items() if f in elig}
    pairs, _unmatched = _mancude_max_matching(adj, eligible, key_of)
    rw = Chem.RWMol(frag)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for first, second in pairs:
        bond = rw.GetBondBetweenAtoms(first, second)
        if bond is None:
            return None
        bond.SetBondType(Chem.BondType.DOUBLE)
    mancude = rw.GetMol()
    try:
        Chem.SanitizeMol(mancude)
    except Exception:
        return None
    rings = [list(r) for r in mancude.GetRingInfo().AtomRings()]
    if not rings:
        return None
    named = _name_fused_component(mancude, rings)
    if named is None:
        return None
    _ih_locants, bare = _extract_leading_indicated_h(named[0])
    return bare


def _spirobi_component_saturation(mol, comp_atoms: Set[int], spiro_center: int,
                                  numbering):
    """ saturation report for ONE spirobi component under ``numbering``.

    Returns ``(hydro_locants, indicated_h_locants)`` -- both as raw locants in
    the component's own numbering -- or None (fail closed).

    The spiro atom carries four single ring bonds, so it is excluded from the
    mancude assignment; the maximum matching over the remaining atoms is the
    mancude form of the ASSEMBLED skeleton. Unmatched atoms are indicated
    hydrogen. Atoms that are sp3 in the real molecule but matched in the mancude
    form are the 'hydro' positions.
    """
    others = set(comp_atoms) - {spiro_center}
    if any(a not in numbering for a in others):
        return None
    if len(others) > _MANCUDE_MATCH_ATOM_CAP:
        return None
    eligible, adj = _component_eligible_adjacency(mol, others, comp_atoms)
    elig = set(eligible)
    key_of = {a: _spirobi_locant_key(numbering[a]) for a in eligible}
    _pairs, unmatched = _mancude_max_matching(adj, eligible, key_of)
    real_sp3 = {a for a in others
                if _is_saturated_in_component(mol, a, comp_atoms)}
    # An indicated-hydrogen position must really carry hydrogen in the molecule.
    # When the lowest-locant mancude choice does not, the correct name needs a
    # joint indicated-H/hydro locant optimisation that is NOT derived here --
    # decline rather than guess (the failure mode would be a wrong STRUCTURE).
    #
    # ⚠ MUTATION-TESTED, and the result is recorded because it is not what it
    # looks like: deleting THIS guard alone breaks no test (it "survives"). It is
    # not dead code -- it is REDUNDANT WITH the parity guard below on every
    # witness reachable here. Whenever the lowest-locant mancude choice misses a
    # real sp3 atom, the leftover set also comes out odd, so the parity check
    # refuses the same molecule. Deleting BOTH is killed. Attempts to build a
    # witness that isolates this guard failed for a structural reason: forcing
    # the mancude sp3 onto a non-lowest locant also forces an extra sp3 fusion
    # atom, which flips the parity. Kept as the guard that states the intent,
    # since the two express different rules and the parity coincidence is not
    # something a future edit should be allowed to rely on.
    if not unmatched <= real_sp3:
        return None
    # Atoms with no spare valence (divalent ring O/S) are sp3 in the mancude form
    # too: they gained no hydrogen and must not attract a hydro prefix.
    hydro = (real_sp3 & elig) - unmatched
    if len(hydro) % 2 != 0:
        return None  # hydro atoms pair into reduced double bonds
    return ({numbering[a] for a in hydro},
            {numbering[a] for a in unmatched})


def _spirobi_component_report(mol, comp_atoms: Set[int], spiro_center: int,
                             spiro_locant):
    """Resolve ONE spirobi component: pick the numbering that places the spiro
    atom at ``spiro_locant`` gives the spiro atom the low locant
    first) and, among those, the one with the lowest hydro/indicated-H locant
    set. Returns ``(mancude_name, hydro, indicated_h)`` or None.
    """
    resolved = _component_raw_catalog_locants(mol, comp_atoms)
    if resolved is None:
        return None
    frag, orig_to_frag, raw_locants = resolved
    target = _spirobi_locant_key(spiro_locant)
    best = None
    for numbering in _component_numberings(frag, orig_to_frag, raw_locants):
        if spiro_center not in numbering:
            continue
        if _spirobi_locant_key(numbering[spiro_center]) != target:
            continue
        report = _spirobi_component_saturation(
            mol, comp_atoms, spiro_center, numbering)
        if report is None:
            continue
        hydro, indicated = report
        rank = (tuple(sorted(_spirobi_locant_key(x) for x in hydro)),
                tuple(sorted(_spirobi_locant_key(x) for x in indicated)))
        if best is None or rank < best[0]:
            best = (rank, numbering, hydro, indicated)
    if best is None:
        return None
    _rank, numbering, hydro, indicated = best
    mancude = _mancude_component_name(mol, comp_atoms, numbering)
    if mancude is None:
        return None
    return mancude, hydro, indicated


def _spirobi_saturation_prefix(mol, spiro_center: int, components):
    """ front-of-name prefix for a partially saturated spirobi system.

    ``components`` is ``[(atoms, spiro_locant, primes),...]`` in citation order
    (unprimed first). Returns ``(front_prefix, mancude_component_name)`` or None
    (fail closed).

    Assembly order is the Blue Book template ``the Blue Book``
    ``1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole]``: hydro prefix,
    then indicated hydrogen, then the spiro locants.
    """
    from .partial_saturation import get_saturation_prefix

    hydro_tokens = []
    indicated_tokens = []
    mancude_names = set()
    for atoms, spiro_locant, primes in components:
        report = _spirobi_component_report(mol, atoms, spiro_center, spiro_locant)
        if report is None:
            return None
        mancude, hydro, indicated = report
        mancude_names.add(mancude)
        for locant in hydro:
            hydro_tokens.append((_spirobi_locant_key(locant, primes),
                                 _spirobi_locant_display(locant, primes)))
        for locant in indicated:
            indicated_tokens.append((_spirobi_locant_key(locant, primes),
                                     _spirobi_locant_display(locant, primes)))
    # Both components are graph-isomorphic, so a disagreement here means one of
    # them was mis-resolved -- decline rather than pick a side.
    if len(mancude_names) != 1:
        return None
    component_name = mancude_names.pop()
    if not component_name:
        return None

    front = ""
    if hydro_tokens:
        multiplier = get_saturation_prefix(len(hydro_tokens))
        if not multiplier:
            return None
        locants = ",".join(text for _key, text in sorted(hydro_tokens))
        front += f"{locants}-{multiplier}-"
    if indicated_tokens:
        locants = ",".join(f"{text}H" for _key, text in sorted(indicated_tokens))
        front += f"{locants}-"
    return front, component_name


def _kekule_ring_unsaturated(mol, comp_atoms: Set[int]) -> Optional[Set[int]]:
    """Atoms of a spiro component that carry a ring DOUBLE (or triple) bond of
    the component in a Kekule form of the molecule. A pyrrole-type ring N-H, an
    sp3 ring atom and a ring atom bearing only an exocyclic =O are not in the
    set: none of them takes a double bond of the mancude ring. Returns None when
    the molecule cannot be kekulized or a component atom is charged (the valence
    bookkeeping of a charged skeleton is not derived here)."""
    if any(mol.GetAtomWithIdx(a).GetFormalCharge() for a in comp_atoms):
        return None
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    out: Set[int] = set()
    for bond in kek.GetBonds():
        x, y = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if x in comp_atoms and y in comp_atoms and bond.GetBondType() in (
                Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            out.update((x, y))
    return out


def _constrained_mancude_matching(adj, nodes, key_of, may_be_unmatched):
    """Maximum matching over ``nodes`` in which every UNMATCHED node lies in
    ``may_be_unmatched``; among those, the one with the lowest sorted tuple of
    unmatched locant keys. Returns ``(pairs, unmatched)`` or None when no
    matching leaves only allowed nodes unmatched.

    The unmatched nodes are the indicated-hydrogen positions of the mancude
    parent. They must be atoms that really carry that hydrogen, so a node that
    is unsaturated in the molecule must be matched: the lowest-locant choice is
    made among the descriptions of THIS structure (b), the Blue Book
    :3246), e.g. 5'H, not 3'H, for the oxazole of
    '(1R)-5'H-spiro[indene-1,2'-[1,3]oxazole] (PIN)' (:49162)."""
    from functools import lru_cache

    node_set = frozenset(nodes)
    allowed = frozenset(may_be_unmatched)

    @lru_cache(maxsize=None)
    def rec(avail):
        if not avail:
            return (0, (), ())
        first = min(avail)
        rest = avail - {first}
        best = None
        if first in allowed:
            sub = rec(rest)
            if sub is not None:
                best = (sub[0], tuple(sorted(sub[1] + (key_of[first],))), sub[2])
        for other in adj[first]:
            if other not in rest:
                continue
            sub = rec(rest - {other})
            if sub is None:
                continue
            cand = (sub[0] + 1, sub[1], sub[2] + ((first, other),))
            if best is None or cand[0] > best[0] or (
                    cand[0] == best[0] and cand[1] < best[1]):
                best = cand
        return best

    res = rec(node_set)
    if res is None:
        return None
    pairs = list(res[2])
    matched = {atom for pair in pairs for atom in pair}
    return pairs, set(node_set) - matched


def _hoisted_saturation(mol, comp_atoms: Set[int], spiro_center: int,
                        numbering, unsaturated: Set[int]):
    """ / saturation of ONE component under ``numbering``,
    described on its mancude parent: ``(hydro_locants, indicated_h_locants)``
    as raw locants, or None.

    The spiro atom (four single ring bonds) is left out of the mancude
    assignment. The indicated-hydrogen positions are the unmatched atoms of a
    MAXIMUM matching that leaves only atoms saturated in the molecule unmatched
    (``_constrained_mancude_matching``); the other saturated atoms that could
    take a ring double bond are the 'hydro' positions."""
    others = set(comp_atoms) - {spiro_center}
    if any(a not in numbering for a in others):
        return None
    if len(others) > _MANCUDE_MATCH_ATOM_CAP:
        return None
    eligible, adj = _component_eligible_adjacency(mol, others, comp_atoms)
    if any(a in unsaturated for a in others if a not in set(eligible)):
        return None  # a double bond on an atom with no spare valence
    key_of = {a: _spirobi_locant_key(numbering[a]) for a in eligible}
    full_pairs, _u = _mancude_max_matching(adj, eligible, key_of)
    saturated = {a for a in eligible if a not in unsaturated}
    res = _constrained_mancude_matching(adj, eligible, key_of, saturated)
    if res is None:
        return None
    pairs, unmatched = res
    if len(pairs) != len(full_pairs):
        return None  # the structure is not its mancude parent plus hydro
    hydro = saturated - unmatched
    if len(hydro) % 2:
        return None
    return ({numbering[a] for a in hydro}, {numbering[a] for a in unmatched})


def _hoisted_state_rank(mol, numbering, spiro_center, hydro, indicated,
                        prefix_atoms):
    """Choice among the numberings of one component, the Blue Book
    :3219): the spiro atom first,:10168), then (b) indicated
    hydrogen, (e) hydro prefixes, (f) the detachable prefixes."""
    return (_spirobi_locant_key(numbering[spiro_center]),
            tuple(sorted(_spirobi_locant_key(x) for x in indicated)),
            tuple(sorted(_spirobi_locant_key(x) for x in hydro)),
            tuple(sorted(_spirobi_locant_key(numbering[a])
                         for a in prefix_atoms if a in numbering)))


def _skeleton_copy(frag):
    """The fragment with every bond single and no aromaticity, atom indices
    unchanged: its automorphisms are the symmetries of the RING SYSTEM, which
    fix its equally established numberings, whatever the
    saturation of the real fragment."""
    rw = Chem.RWMol(frag)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    skel = rw.GetMol()
    skel.UpdatePropertyCache(strict=False)
    return skel


def _hoisted_fused_state(mol, comp_atoms: Set[int], spiro_center: int,
                         unsaturated: Set[int], prefix_atoms,
                         base_numbering=None):
    """A polycyclic spiro component named by its mancude fusion or retained
    name, with its saturation hoisted (``_hoisted_saturation``). The numbering
    is the catalog's, else ``base_numbering`` (the numbering under which the
    component's own name was built); every symmetry-equivalent numbering of the
    ring system is considered. Returns ``{'name', 'map', 'loc', 'hydro', 'ih'}``
    or None."""
    resolved = _component_raw_catalog_locants(mol, comp_atoms)
    if resolved is not None:
        frag, orig_to_frag, raw_locants = resolved
    else:
        extracted = _extract_subfragment(mol, comp_atoms)
        if extracted is None:
            return None
        frag, orig_to_frag = extracted
        raw_locants = {a: base_numbering[a] for a in comp_atoms
                       if base_numbering and a in base_numbering}
        if set(raw_locants) != set(comp_atoms) or len(
                set(map(str, raw_locants.values()))) != len(raw_locants):
            # the numbering of the mancude parent itself, from its namer
            raw_locants = _mancude_parent_numbering(mol, comp_atoms, frag,
                                                    orig_to_frag)
            if raw_locants is None:
                return None
    try:
        numberings = _component_numberings(_skeleton_copy(frag), orig_to_frag,
                                           raw_locants)
    except Exception:
        return None
    best = None
    for numbering in numberings:
        if not isinstance(numbering.get(spiro_center), int):
            continue
        sat = _hoisted_saturation(mol, comp_atoms, spiro_center, numbering,
                                  unsaturated)
        if sat is None:
            continue
        hydro, indicated = sat
        rank = _hoisted_state_rank(mol, numbering, spiro_center, hydro,
                                   indicated, prefix_atoms)
        if best is None or rank < best[0]:
            best = (rank, numbering, hydro, indicated)
    if best is None:
        return None
    _rank, numbering, hydro, indicated = best
    mancude = _checked_mancude_component_name(mol, comp_atoms, numbering)
    if not mancude:
        return None
    return {'name': _bracket_component_locants(mancude), 'map': dict(numbering),
            'loc': numbering[spiro_center], 'hydro': hydro, 'ih': indicated}


def _checked_mancude_component_name(mol, comp_atoms: Set[int], numbering):
    """The name of the MANCUDE ring system of a spiro component (its own
    indicated hydrogen removed), provided the numbering of that named ring
    system is ``numbering`` or one of its symmetry-equivalent forms -- so the
    hoisted hydro and indicated-hydrogen locants are locants of the parent the
    name cites. The mancude form is built from the skeleton (every ring bond
    single, then a maximum set of noncumulative double bonds) and named by the
    fused-ring catalog, or for a carbocycle by the polycyclic hydrocarbon
    namer. Returns None when the name or its numbering cannot be confirmed."""
    extracted = _extract_subfragment(mol, comp_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    if frag.GetNumAtoms() > _MANCUDE_MATCH_ATOM_CAP:
        return None
    eligible = sorted(
        f for orig, f in orig_to_frag.items()
        if _can_bear_ring_double_bond(mol, orig, comp_atoms))
    elig = set(eligible)
    if not eligible:
        return None
    adj = {f: sorted(n.GetIdx() for n in frag.GetAtomWithIdx(f).GetNeighbors()
                     if n.GetIdx() in elig) for f in eligible}
    key_of = {f: _spirobi_locant_key(numbering[orig])
              for orig, f in orig_to_frag.items() if f in elig}
    full_pairs, _unmatched = _mancude_max_matching(adj, eligible, key_of)
    # The ring system is the same whichever atom keeps the indicated hydrogen of
    # the free parent ('1H-' or '4H-quinolizine'); the namer is asked for each
    # placement, lowest locant first, until it names the system.
    if len(eligible) == 2 * len(full_pairs):
        placements = [full_pairs]
    else:
        placements = []
        for f in sorted(eligible, key=lambda x: key_of[x]):
            res = _constrained_mancude_matching(adj, eligible, key_of, {f})
            if res is not None and len(res[0]) == len(full_pairs):
                placements.append(res[0])
    frag_to_orig = {f: o for o, f in orig_to_frag.items()}
    wanted = {a: str(numbering[a]) for a in comp_atoms}
    skeleton = _skeleton_copy(frag)
    for pairs in placements:
        bare = _name_mancude_placement(frag, pairs)
        if bare is None:
            continue
        name, authority = bare
        auth_orig = {frag_to_orig[f]: loc for f, loc in authority.items()
                     if f in frag_to_orig}
        if set(auth_orig) != set(comp_atoms):
            continue
        variants = _component_numberings(skeleton, orig_to_frag, auth_orig)
        if not any({a: str(v[a]) for a in comp_atoms} == wanted
                   for v in variants if set(v) == set(comp_atoms)):
            continue
        _ih, bare_name = _extract_leading_indicated_h(name)
        return bare_name
    return None


def _mancude_parent_numbering(mol, comp_atoms: Set[int], frag, orig_to_frag):
    """``{orig_atom: locant}`` of the mancude ring system with this skeleton, as
    its namer numbers it (the fused-ring catalog, or the polycyclic hydrocarbon
    namer for a carbocycle), or None."""
    eligible = sorted(
        f for orig, f in orig_to_frag.items()
        if _can_bear_ring_double_bond(mol, orig, comp_atoms))
    if not eligible or len(eligible) > _MANCUDE_MATCH_ATOM_CAP:
        return None
    elig = set(eligible)
    adj = {f: sorted(n.GetIdx() for n in frag.GetAtomWithIdx(f).GetNeighbors()
                     if n.GetIdx() in elig) for f in eligible}
    key_of = {f: (f, '', 0) for f in eligible}
    full_pairs, _u = _mancude_max_matching(adj, eligible, key_of)
    if len(eligible) == 2 * len(full_pairs):
        placements = [full_pairs]
    else:
        placements = []
        for f in eligible:
            res = _constrained_mancude_matching(adj, eligible, key_of, {f})
            if res is not None and len(res[0]) == len(full_pairs):
                placements.append(res[0])
    frag_to_orig = {f: o for o, f in orig_to_frag.items()}
    for pairs in placements:
        named = _name_mancude_placement(frag, pairs)
        if named is None:
            continue
        out = {frag_to_orig[f]: loc for f, loc in named[1].items()
               if f in frag_to_orig}
        if set(out) == set(comp_atoms) and len(
                set(map(str, out.values()))) == len(out):
            return out
    return None


def _name_mancude_placement(frag, pairs):
    """Name the fragment with exactly the double bonds ``pairs`` (every other
    ring bond single): ``(name, {frag_idx: locant})`` from the fused-ring
    catalog, or for a carbocycle from the polycyclic hydrocarbon namer; None
    when neither names it."""
    rw = Chem.RWMol(frag)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for first, second in pairs:
        rw.GetBondBetweenAtoms(first, second).SetBondType(Chem.BondType.DOUBLE)
    mancude = rw.GetMol()
    try:
        Chem.SanitizeMol(mancude)
    except Exception:
        return None
    rings = [list(r) for r in mancude.GetRingInfo().AtomRings()]
    named = _name_fused_component(mancude, rings)
    if named is not None and named[0] and named[1]:
        return named
    if not all(a.GetAtomicNum() == 6 for a in mancude.GetAtoms()):
        return None
    try:
        from .polycyclics import get_polycyclic_iupac_locants, identify_polycyclic
        name = identify_polycyclic(mancude)
        authority = get_polycyclic_iupac_locants(mancude, name) if name else None
    except Exception:
        return None
    if not name or not authority:
        return None
    return name, {k: (f"{v[0]}{v[1]}" if isinstance(v, tuple) else v)
                  for k, v in authority.items()}


def _mancude_monocycle_name(mol, comp_atoms: Set[int]) -> Optional[str]:
    """The name of the MANCUDE heteromonocycle with this skeleton, its own
    indicated hydrogen removed: the retained name where one is the PIN
    ('furan', 'pyridine', 'thiophene', 'pyran'), else the Hantzsch-Widman name
    ('1,3-oxazole'), from the ring namer ``heterocycles.name_heterocycle``. The
    mancude form is built from the skeleton (every bond single, then a maximum
    set of noncumulative double bonds), not edited from another name."""
    extracted = _extract_subfragment(mol, comp_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    eligible = sorted(
        f for orig, f in orig_to_frag.items()
        if _can_bear_ring_double_bond(mol, orig, comp_atoms))
    elig = set(eligible)
    adj = {f: sorted(n.GetIdx() for n in frag.GetAtomWithIdx(f).GetNeighbors()
                     if n.GetIdx() in elig) for f in eligible}
    pairs, _unmatched = _mancude_max_matching(
        adj, eligible, {f: (f, '', 0) for f in eligible})
    rw = Chem.RWMol(frag)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for first, second in pairs:
        rw.GetBondBetweenAtoms(first, second).SetBondType(Chem.BondType.DOUBLE)
    mancude = rw.GetMol()
    try:
        Chem.SanitizeMol(mancude)
        from .heterocycles import name_heterocycle
        named = name_heterocycle(mancude, list(range(mancude.GetNumAtoms())))
    except Exception:
        return None
    if not named or ' ' in named:
        return None
    _ih, bare = _extract_leading_indicated_h(named)
    return bare


def _hoisted_hw_monocycle_state(mol, comp_atoms: Set[int], spiro_center: int,
                                unsaturated: Set[int], prefix_atoms):
    """An unsaturated heteromonocyclic spiro component named by its mancude
    Hantzsch-Widman name ('[1,3]oxazole'), numbered by the heteroatoms first
     and then as in ``_hoisted_state_rank``, with its saturation
    hoisted. Returns the state dict or None."""
    name = _mancude_monocycle_name(mol, comp_atoms)
    if not name:
        return None
    ring = [a for a in comp_atoms]
    adj = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
               if n.GetIdx() in comp_atoms] for a in ring}
    if any(len(v) != 2 for v in adj.values()):
        return None
    from ..data.hw_heteroatoms import HETEROATOM_PRIORITY as _HP
    hetero = [a for a in ring if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    walks = []
    for start in sorted(ring):
        for first in sorted(adj[start]):
            path = [start, first]
            while len(path) < len(ring):
                nxt = [n for n in adj[path[-1]] if n != path[-2]]
                if len(nxt) != 1:
                    break
                path.append(nxt[0])
            if len(path) == len(ring):
                walks.append({a: i + 1 for i, a in enumerate(path)})
    if not walks:
        return None

    def het_key(num):
        return (sorted(num[a] for a in hetero),
                [p for _l, p in sorted(
                    (num[a], _HP.get(mol.GetAtomWithIdx(a).GetSymbol(), 999))
                    for a in hetero)])
    top = min(het_key(w) for w in walks)
    best = None
    for numbering in walks:
        if het_key(numbering) != top:
            continue
        sat = _hoisted_saturation(mol, comp_atoms, spiro_center, numbering,
                                  unsaturated)
        if sat is None:
            continue
        hydro, indicated = sat
        rank = _hoisted_state_rank(mol, numbering, spiro_center, hydro,
                                   indicated, prefix_atoms)
        if best is None or rank < best[0]:
            best = (rank, numbering, hydro, indicated)
    if best is None:
        return None
    _rank, numbering, hydro, indicated = best
    return {'name': _bracket_component_locants(name), 'map': dict(numbering),
            'loc': numbering[spiro_center], 'hydro': hydro, 'ih': indicated}


def _component_needs_hoist(mol, comp_atoms: Set[int], spiro_center: int,
                           unsaturated: Set[int]) -> bool:
    """True when an atom of the component other than the spiro atom could take
    a ring double bond but carries none: its saturation must be cited in front
    of the spiro name."""
    return any(
        a != spiro_center and a not in unsaturated
        and _can_bear_ring_double_bond(mol, a, comp_atoms)
        for a in comp_atoms)


def _pin_component_alpha_key(name: str) -> Tuple[str, str]:
    """ / alphanumerical order of two ring-component names
    (the Blue Book): the Roman letters first ('cyclopentane' before
    'isoquinoline', 'indene' before '[1,3]oxazole'); when they are equal, the
    fusion locants and letters, heteroatom locants and von Baeyer numbers
    ('thieno[2,3-b]furan' before 'thieno[3,2-b]furan',:10310)."""
    import re
    s = name.strip().lower()
    letters = re.sub(r'[^a-z]', '', re.sub(r'\[[^\]]*\]', '', s))
    return (letters, s)


def _assemble_hoisted_monospiro(spiro_center: int, state_a, state_b):
    """ (the Blue Book) two-component spiro name from two
    component states: the component names in alphanumerical order, the second
    primed, and the hydro prefixes and indicated hydrogen of the complete
    structure in front ('4'a,5',6',7',8',8'a-hexahydro-1'H-spiro[imidazolidine-
    4,2'-quinoxaline] (PIN)',:17050). Returns ``(front, core, combined_map,
    unprimed_state, primed_state)`` or None."""
    from .partial_saturation import get_saturation_prefix
    if _pin_component_alpha_key(state_a['name']) <= _pin_component_alpha_key(
            state_b['name']):
        first, second = state_a, state_b
    else:
        first, second = state_b, state_a
    hydro_tokens = []
    indicated_tokens = []
    for state, primes in ((first, 0), (second, 1)):
        for loc in state['hydro']:
            hydro_tokens.append((_spirobi_locant_key(loc, primes),
                                 _spirobi_locant_display(loc, primes)))
        for loc in state['ih']:
            indicated_tokens.append((_spirobi_locant_key(loc, primes),
                                     _spirobi_locant_display(loc, primes)))
    hydro_part = ""
    if hydro_tokens:
        multiplier = get_saturation_prefix(len(hydro_tokens))
        if not multiplier:
            return None
        hydro_part = (f"{','.join(t for _k, t in sorted(hydro_tokens))}-"
                      f"{multiplier}")
    ih_part = ",".join(f"{t}H" for _k, t in sorted(indicated_tokens))
    front = (hydro_part, ih_part)
    core = (f"spiro[{first['name']}-{first['loc']},"
            f"{second['loc']}'-{second['name']}]")
    combined: Dict[int, _Locant] = {}
    for atom_idx, locant in first['map'].items():
        combined[atom_idx] = locant
    for atom_idx, locant in second['map'].items():
        if atom_idx == spiro_center:
            continue
        combined[atom_idx] = (locant, "'")
    return front, core, combined, first, second


def _attach_hoisted_front(front, rest: str) -> str:
    """Cite the hoisted hydro prefix and indicated hydrogen in front of the spiro
    name: hydro prefix, then indicated hydrogen, then the name
    ('4'a,5',6',7',8',8'a-hexahydro-1'H-spiro[...]', the Blue Book;
    '1,2-dihydrospiro[...]'). A hyphen separates a term from a following
    locant (a), the Blue Book, "to separate locants from words
    or word fragments"); 'hydro' joins a following letter directly."""
    hydro_part, ih_part = front
    out = rest
    if ih_part:
        out = f"{ih_part}-{out}"
    if hydro_part:
        out = f"{hydro_part}-{out}" if out[:1].isdigit() else f"{hydro_part}{out}"
    return out


def _plain_component_state(name: str, numbering, spiro_center: int):
    """State of a component whose name needs no hoisting: a saturated
    monocycle, a von Baeyer cage without double bonds, or a fused system whose
    only sp3 ring atom is the spiro atom. The component's own indicated
    hydrogen at the spiro atom is dropped."""
    loc = numbering.get(spiro_center)
    if not isinstance(loc, int):
        return None
    shown = _bracket_component_locants(_strip_consumed_indicated_h(name, loc))
    if _extract_leading_indicated_h(shown)[0]:
        return None  # indicated hydrogen left inside the bracket
    return {'name': shown, 'map': dict(numbering), 'loc': loc,
            'hydro': set(), 'ih': set()}


def _ortho_fused_five_plus(mol, comp_atoms: Set[int]) -> bool:
    """True when the component is an ortho-fused ring system (rings sharing one
    bond, no atom in three rings) of rings of five or more members: such a
    system with a double bond is named by fusion nomenclature, as hydro
    derivatives of its mancude parent, not by von Baeyer names
    : fusion nomenclature for systems with at least two rings of five or
    more members)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()
             if set(r) <= set(comp_atoms)]
    if len(rings) < 2 or any(len(r) < 5 for r in rings):
        return False
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) > 2:
                return False
    for a in comp_atoms:
        if sum(1 for r in rings if a in r) > 2:
            return False
    return True


def _monospiro_component_state(mol, comp_atoms: Set[int], spiro_center: int,
                               name: str, numbering, is_cage: bool):
    """The state of one component of a two-component spiro system, or
    None when its preferred form cannot be built here (the caller then keeps
    its older name and labels it below the PIN)."""
    unsaturated = _kekule_ring_unsaturated(mol, comp_atoms)
    if unsaturated is None:
        return None
    others = set(comp_atoms) - {spiro_center}
    prefix_atoms = _spiro_exocyclic_prefix_atoms(mol, set(comp_atoms))
    ring_count = len([r for r in mol.GetRingInfo().AtomRings()
                      if set(r) <= set(comp_atoms)])
    has_unsat = bool(unsaturated & others)
    if (is_cage or ('cyclo[' in name and ring_count > 1)) and not (
            has_unsat and _ortho_fused_five_plus(mol, comp_atoms)):
        # (the Blue Book): a double bond of a von Baeyer
        # component is an 'ene' ending after the last bracket of the spiro
        # name, a form this builder does not write.
        return None if has_unsat else _plain_component_state(
            name, numbering, spiro_center)
    if ring_count == 1:
        if not has_unsat:
            return _plain_component_state(name, numbering, spiro_center)
        if all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in comp_atoms):
            # an unsaturated carbocycle: 'ene' after the bracket,
            # '...spiro[[1]benzofuran-2,1'-cyclohexan]-2'-ene (PIN)',:17052).
            return None
        return _hoisted_hw_monocycle_state(mol, comp_atoms, spiro_center,
                                           unsaturated, prefix_atoms)
    # a von Baeyer numbering is not the numbering of the fused parent
    base = None if (is_cage or 'cyclo[' in name) else numbering
    hoisted = _hoisted_fused_state(mol, comp_atoms, spiro_center, unsaturated,
                                   prefix_atoms, base_numbering=base)
    if hoisted is not None:
        return hoisted
    if not _component_needs_hoist(mol, comp_atoms, spiro_center, unsaturated):
        # a fused system outside the catalog lookup (a carbocyclic template
        # such as fluorene) whose only sp3 ring atom is the spiro atom
        return _plain_component_state(name, numbering, spiro_center)
    return None


def is_spirobi(mol) -> bool:
    """: monospiro ring system with two IDENTICAL (polycyclic)
    components joined at one spiro atom (e.g. 1,1'-spirobi[indene]).

    These have ``n_rings > n_spiro + 1`` (each component is itself polycyclic),
    so ``is_spiro_system`` rejects them and they are NOT mixed-spiro-fused
    (which requires one side to be a single ring). Fail-closed: exactly one
    spiro atom, both fused components polycyclic and graph-isomorphic, and the
    whole system unsubstituted (substituted spirobi numbering is a follow-on).
    """
    if mol is None:
        return False
    return _name_spirobi_core(mol) is not None


def _name_spirobi_core(mol):
    """Shared core for is_spirobi / name_spirobi. Returns
    (name, all_ring_atoms, combined_locants) or None (fail-closed)."""
    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) != 1:
        return None
    spiro_center = next(iter(spiro_atoms))
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]
    # Pure monospiro (2 rings) is name_spiro_system's job, not spirobi.
    if len(all_rings) <= 2:
        return None
    part = _partition_rings_at_spiro(mol, spiro_center, all_rings)
    if part is None:
        return None
    comp_a_idx, comp_b_idx = part
    # spirobi requires BOTH components polycyclic (>=2 rings each); a 1-ring
    # side is mixed-spiro-fused or pure spiro.
    if len(comp_a_idx) < 2 or len(comp_b_idx) < 2:
        return None

    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms.update(r)
    # Fail-closed on substituted spirobi: every heavy atom must be a ring atom
    # (substituted-spirobi prime/locant selection is a documented follow-on).
    for atom in atoms_of(mol):
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None

    rings_a = [all_rings[i] for i in comp_a_idx]
    rings_b = [all_rings[i] for i in comp_b_idx]
    atoms_a: Set[int] = set()
    for r in rings_a:
        atoms_a.update(r)
    atoms_b: Set[int] = set()
    for r in rings_b:
        atoms_b.update(r)

    # Components must be IDENTICAL RING COMPONENTS: the same atoms
    # joined the same way. (the Blue Book): "the maximum number of
    # noncumulative double bonds is added (i.e., the system is made mancude) after
    # construction of the complete skeleton", so the comparison is made on the
    # skeleton (bond orders, aromaticity and hydrogen ignored): the two halves of
    # '2,4'-spirobi[[1]benzopyran] (PIN)' (:10182) carry their double bonds in
    # different places. A different skeleton is a DIFFERENT-component spiro
    #, not spirobi -- decline here.
    ext_a = _extract_subfragment(mol, atoms_a)
    ext_b = _extract_subfragment(mol, atoms_b)
    if ext_a is None or ext_b is None:
        return None
    if _ring_skeleton_smiles(ext_a[0]) != _ring_skeleton_smiles(ext_b[0]):
        return None

    named_a = _name_fused_component(mol, rings_a)
    named_b = _name_fused_component(mol, rings_b)
    if named_a is None or named_b is None:
        return None
    name_a, loc_map_a = named_a
    name_b, loc_map_b = named_b
    # lowest locant at the spiro atom, computed DETERMINISTICALLY: a
    # symmetric component (e.g. indane positions 1 and 3 are mirror-equivalent)
    # lets the catalog substructure-match place the spiro atom at either of two
    # equivalent locants depending on SMILES atom order — so pick the minimum
    # over the spiro atom's symmetry orbit (atoms with equal CanonicalRankAtoms
    # in the component fragment), not whichever match RDKit returned first.
    loc_a = _canonical_spiro_locant(ext_a, loc_map_a, spiro_center)
    loc_b = _canonical_spiro_locant(ext_b, loc_map_b, spiro_center)
    if loc_a is None or loc_b is None:
        return None

    #: the lower number at the spiro atom is unprimed.
    if loc_a <= loc_b:
        lo, hi = loc_a, loc_b
        unprimed_map, primed_map = loc_map_a, loc_map_b
        unprimed_atoms, primed_atoms = atoms_a, atoms_b
    else:
        lo, hi = loc_b, loc_a
        unprimed_map, primed_map = loc_map_b, loc_map_a
        unprimed_atoms, primed_atoms = atoms_b, atoms_a

    #: a NONSTANDARD (λ) spiro atom carries its λ token on the UNPRIMED
    # locant (e.g. 2lambda4,2'-spirobi[[1,3,2]benzodioxathiole]).
    lam = _nonstandard_bonding_number(mol, spiro_center)
    lo_tok = f"{lo}{_LAMBDA}{lam}" if lam is not None else str(lo)

    #: a component atom that COULD carry a ring double bond but does not
    # is saturation that has to be expressed OUTSIDE the bracket -- as a hydro
    # prefix or as indicated hydrogen. Gating on that predicate (rather than on
    # "is anything sp3") keeps every fully-mancude component -- indene,
    # [1,3,2]benzodioxathiole, whose divalent ring O atoms are sp3 but can never
    # be doubly bonded -- on the long-standing code path below, byte-identical.
    needs_hoist = any(
        _is_saturated_in_component(mol, atom, comp)
        and _can_bear_ring_double_bond(mol, atom, comp)
        for comp in (unprimed_atoms, primed_atoms)
        for atom in comp if atom != spiro_center
    )
    if needs_hoist:
        hoisted = _spirobi_saturation_prefix(mol, spiro_center, [
            (unprimed_atoms, lo, 0),
            (primed_atoms, hi, 1),
        ])
        if hoisted is None:
            return None  # fail closed: saturation we cannot spell is not named
        front, component_name = hoisted
        component_name = _bracket_component_locants(component_name)
        name = f"{front}{lo_tok},{hi}'-spirobi[{component_name}]"
    else:
        # /: indicated hydrogen of the individual component is
        # not cited when the spiro atom occupies that locant. Each half is named
        # with its own indicated hydrogen ('2H-1-benzopyran', '4H-1-benzopyran'
        # for the halves of '2,4'-spirobi[[1]benzopyran]'); both must reduce to
        # one component name, else the namers disagree -> decline.
        unprimed_name = name_a if unprimed_map is loc_map_a else name_b
        primed_name = name_b if unprimed_map is loc_map_a else name_a
        component_name = _strip_consumed_indicated_h(unprimed_name, lo)
        if _strip_consumed_indicated_h(primed_name, hi) != component_name:
            return None
        component_name = _bracket_component_locants(component_name)
        name = f"{lo_tok},{hi}'-spirobi[{component_name}]"

    combined_locants: Dict[int, _Locant] = {}
    for atom_idx, locant in unprimed_map.items():
        combined_locants[atom_idx] = locant
    for atom_idx, locant in primed_map.items():
        if atom_idx == spiro_center:
            continue
        combined_locants[atom_idx] = (locant, "'")

    if not (set(combined_locants.keys()) >= all_ring_atoms):
        return None  # coverage invariant (Pitfall 7)
    return name, all_ring_atoms, combined_locants


def name_spirobi(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the ``spirobi`` name for two identical polycyclic
    components at one spiro atom (``1,1'-spirobi[indene]``). Return shape
    matches ``name_mixed_spiro_fused`` (name, ring_atoms, atom_to_locant,
    substituents_included=False). Fail-closed (see ``_name_spirobi_core``)."""
    core = _name_spirobi_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def _name_spiroter_core(mol):
    """: three IDENTICAL polycyclic components sharing ONE nonstandard
    (λ) spiro atom that lies in THREE rings (a λ6 spiroter, e.g.
    ``2lambda6,2',2''-spiroter[[1,3,2]benzodioxathiole]``). Returns
    ``(name, all_ring_atoms, combined_locants)`` or None (fail-closed).

    Fail closed unless: exactly one atom in >=3 rings with a nonstandard bonding
    number, that atom is a cut vertex splitting the ring graph into exactly 3
    components, all three graph-isomorphic and identically named, and the system
    is unsubstituted."""
    ri = mol.GetRingInfo()
    all_rings = [set(r) for r in ri.AtomRings()]
    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms |= r
    for atom in atoms_of(mol):  # unsubstituted only
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None

    # Locate the unique λ spiro atom in >=3 rings.
    spiro_center = None
    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if sum(1 for r in all_rings if idx in r) < 3:
            continue
        if _nonstandard_bonding_number(mol, idx) is None:
            continue
        if spiro_center is not None:
            return None  # >1 candidate -> not a clean λ spiroter
        spiro_center = idx
    if spiro_center is None:
        return None
    lam = _nonstandard_bonding_number(mol, spiro_center)

    _, adj = _ring_atom_graph(mol)
    comps = _ring_components_excluding(adj, spiro_center)
    if len(comps) != 3:
        return None

    named = []
    canon = None
    for comp in comps:
        atoms = comp | {spiro_center}
        nm = _name_spiro_component(mol, atoms, spiro_center)
        if nm is None:
            return None
        ext = _extract_subfragment(mol, atoms)
        if ext is None:
            return None
        loc = _canonical_spiro_locant(ext, nm[1], spiro_center)
        if loc is None:
            return None
        c_smi = Chem.MolToSmiles(ext[0])
        if canon is None:
            canon = (nm[0], c_smi)
        elif nm[0] != canon[0] or c_smi != canon[1]:
            return None  # not all identical
        named.append((nm[0], nm[1], loc, comp))

    # All identical: unprimed / primed / double-primed by ascending spiro locant
    # (they are equal here). Emit the λ token on the unprimed spiro locant.
    named.sort(key=lambda t: t[2])
    lo = named[0][2]
    mid = named[1][2]
    hi = named[2][2]
    component_name = _strip_consumed_indicated_h(named[0][0], lo)
    lo_tok = f"{lo}{_LAMBDA}{lam}" if lam is not None else str(lo)
    name = f"{lo_tok},{mid}',{hi}''-spiroter[{component_name}]"

    combined: Dict[int, _Locant] = {}
    for atom_idx, locant in named[0][1].items():
        combined[atom_idx] = locant
    for prime, entry in (("'", named[1]), ("''", named[2])):
        for atom_idx, locant in entry[1].items():
            if atom_idx == spiro_center or atom_idx in combined:
                continue
            combined[atom_idx] = (locant, prime)
    if not (set(combined.keys()) >= all_ring_atoms):
        return None
    return name, all_ring_atoms, combined


def is_spiroter(mol) -> bool:
    """P-24.8.3: three identical polycyclic components + one λ spiro atom."""
    if mol is None:
        return False
    return _name_spiroter_core(mol) is not None


def name_spiroter(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the ``spiroter`` name. Fail-closed."""
    core = _name_spiroter_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def _spiro_component_alpha_key(name: str) -> str:
    """Alphanumerical sort key / for a spiro
    ring-component name. Unlike the shared ``_component_alpha_key`` this ALSO
    removes every bracketed locant/fusion group so the *ring-name* letters drive
    the order: ``[1,3,2]benzodioxathiole`` -> ``benzodioxathiole`` and
    ``dibenzo[b,d]thiophene`` -> ``dibenzothiophene``, giving the BB citation
    order benzodioxathiole < benzoxadithiole < dibenzothiophene (the Blue Book).
    Kept local so the two-component ``_component_alpha_key`` behaviour (which
    must NOT strip internal brackets) is untouched."""
    import re
    s = name.strip().lower()
    s = re.sub(r'^\d+h-', '', s)      # drop a leading indicated-H descriptor
    s = re.sub(r'\[[^\]]*\]', '', s)  # drop every bracketed locant/fusion group
    return s


def _name_spiro_named_components_core(mol):
    """: a monospiro ring system built from THREE ring components (all
    distinct — two different individual ring systems in the BB example) sharing
    ONE nonstandard (λ) spiro atom that lies in THREE rings.

    BB (the Blue Book) / example the Blue Book:
    ``2lambda6-spiro[[1,3,2]benzodioxathiole-2,2'-([1,2,3]benzoxadithiole)-2,5''-dibenzo[b,d]thiophene]``.
    The components are cited in alphanumerical order; the
    SECOND-cited name is enclosed in parentheses to flag this unusual situation;
    the λⁿ symbol, preceded by the lowest locant denoting the spiro atom, is
    placed at the front; spiro-locant pairs are cited between consecutive
    component names (``2,2'`` then ``2,5''`` — the second pair re-cites the
    lowest unprimed locant). Prime multiplicity: comp1 unprimed, comp2 ``'``,
    comp3 ``''``.

    Returns ``(name, all_ring_atoms, combined_locants)`` or None (fail-closed).

    Fail closed unless: exactly one atom in >=3 rings with a nonstandard bonding
    number; that atom is a cut vertex splitting the ring graph into exactly 3
    components; all three DISTINCT (distinct name AND distinct capped fragment —
    an all-identical triple is spiroter, a 2-identical triple is
     'bis' which OPSIN 2.9 cannot round-trip); each nameable; the
    system unsubstituted; the lowest spiro locant is the UNPRIMED (comp1) one
    (so the derived front token + inter-component pairs match the BB example),
    and no component needs a residual (non-consumed) indicated-H descriptor."""
    ri = mol.GetRingInfo()
    all_rings = [set(r) for r in ri.AtomRings()]
    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms |= r
    for atom in atoms_of(mol):  # unsubstituted only
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None

    # Locate the unique λ spiro atom in >=3 rings (identical to _name_spiroter).
    spiro_center = None
    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if sum(1 for r in all_rings if idx in r) < 3:
            continue
        if _nonstandard_bonding_number(mol, idx) is None:
            continue
        if spiro_center is not None:
            return None
        spiro_center = idx
    if spiro_center is None:
        return None
    lam = _nonstandard_bonding_number(mol, spiro_center)

    _, adj = _ring_atom_graph(mol)
    comps = _ring_components_excluding(adj, spiro_center)
    if len(comps) != 3:
        return None

    named = []  # (name, loc_map, spiro_locant, comp_atoms, frag_canon)
    for comp in comps:
        atoms = comp | {spiro_center}
        nm = _name_spiro_component(mol, atoms, spiro_center)
        if nm is None:
            return None
        ext = _extract_subfragment(mol, atoms)
        if ext is None:
            return None
        loc = _canonical_spiro_locant(ext, nm[1], spiro_center)
        if loc is None:
            return None
        norm = _normalize_hypervalent_ring_heteroatoms(ext[0])
        frag_canon = (Chem.MolToSmiles(norm) if norm is not None
                      else Chem.MolToSmiles(ext[0]))
        named.append((nm[0], nm[1], loc, comp, frag_canon))

    # All three DISTINCT: 3-identical -> spiroter; 2-identical ->
    # 'bis', which OPSIN 2.9 cannot round-trip. Decline both.
    if len({t[0] for t in named}) != 3 or len({t[4] for t in named}) != 3:
        return None

    # Residual indicated-H (after the spiro atom consumes its own, is
    # not handled by this targeted build -> fail closed rather than misplace it.
    for nm0, _lmap, loc0, _c, _fc in named:
        stripped = _strip_consumed_indicated_h(nm0, loc0)
        residual, _bare = _extract_leading_indicated_h(stripped)
        if residual:
            return None

    # Alphanumerical citation order /; deterministic tie-break
    # by capped-fragment canonical SMILES (the three are distinct).
    named.sort(key=lambda t: (_spiro_component_alpha_key(t[0]), t[4]))
    c0, c1, c2 = named

    # "Lowest locant denoting the spiro atom" (the Blue Book): min over (numeral,
    # prime-rank). Require it to be the UNPRIMED comp1 locant so the derived
    # front token and inter-component pairs are the BB-canonical construction;
    # otherwise fail closed (exotic primed-front variants are out of scope).
    lowest = min((c0[2], 0), (c1[2], 1), (c2[2], 2))
    if lowest != (c0[2], 0):
        return None
    lo = c0[2]
    lo_tok = f"{lo}{_LAMBDA}{lam}" if lam is not None else str(lo)

    n0 = _strip_consumed_indicated_h(c0[0], c0[2])
    n1 = _strip_consumed_indicated_h(c1[0], c1[2])
    n2 = _strip_consumed_indicated_h(c2[0], c2[2])
    # Inter-component spiro-locant pairs: (comp1 unprimed locant, next comp's
    # primed locant). The second pair re-cites the lowest unprimed locant.
    name = (
        f"{lo_tok}-spiro["
        f"{n0}-{c0[2]},{c1[2]}'-"
        f"({n1})-"
        f"{lo},{c2[2]}''-"
        f"{n2}]"
    )

    combined: Dict[int, _Locant] = {}
    for atom_idx, locant in c0[1].items():
        combined[atom_idx] = locant
    for prime, entry in (("'", c1), ("''", c2)):
        for atom_idx, locant in entry[1].items():
            if atom_idx == spiro_center or atom_idx in combined:
                continue
            combined[atom_idx] = (locant, prime)
    if not (set(combined.keys()) >= all_ring_atoms):
        return None
    return name, all_ring_atoms, combined


def is_spiro_named_components(mol) -> bool:
    """P-24.8.4.2: three distinct ring components + one λ spiro atom in 3 rings."""
    if mol is None:
        return False
    return _name_spiro_named_components_core(mol) is not None


def name_spiro_named_components(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the three-component monospiro name. Fail-closed."""
    core = _name_spiro_named_components_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def _name_dispiroter_core(mol):
    """: three IDENTICAL polycyclic components sharing exactly TWO spiro
    atoms -> ``a,a':b',b''-dispiroter[component]`` (e.g.
    ``3,3':6',6''-dispiroter[bicyclo[3.1.0]hexane]``). Returns
    ``(name, all_ring_atoms, combined_locants)`` or None (fail-closed).

    Fail closed unless: exactly 2 spiro atoms, exactly 3 components (a linear
    chain terminal-middle-terminal where the middle component holds BOTH spiro
    atoms), all three graph-isomorphic and identically named, and the system is
    unsubstituted.: the lowest spiro-atom locant set is chosen (compare
    the two terminals as unprimed vs double-primed). The 'a'-replacement
    heterocyclic dispiroter form (b)) is a documented fail-closed
    follow-on."""
    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) != 2:
        return None
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]

    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms.update(r)
    # Unsubstituted only.
    for atom in atoms_of(mol):
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None

    # Split the ring-atom graph by removing BOTH spiro atoms -> 3 components.
    _, adj = _ring_atom_graph(mol)
    sa_list = sorted(spiro_atoms)
    seen: Set[int] = set(sa_list)
    comps: List[Set[int]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        comp = {start}
        seen.add(start)
        while stack:
            cur = stack.pop()
            for nbr in adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    comp.add(nbr)
                    stack.append(nbr)
        comps.append(comp)
    if len(comps) != 3:
        return None

    # Attach each spiro atom to the components it touches. A terminal component
    # touches exactly ONE spiro atom; the middle component touches BOTH.
    def _touching_spiros(comp: Set[int]) -> List[int]:
        touch = []
        for sa in sa_list:
            if any(nbr in comp for nbr in adj[sa]):
                touch.append(sa)
        return touch

    terminals = []
    middle = None
    for comp in comps:
        touch = _touching_spiros(comp)
        if len(touch) == 1:
            terminals.append((comp, touch[0]))
        elif len(touch) == 2:
            if middle is not None:
                return None
            middle = (comp, touch)
        else:
            return None
    if middle is None or len(terminals) != 2:
        return None

    # Name each component (component atoms + its spiro atom(s)); require identity.
    def _name_comp(comp_atoms: Set[int], spiros: List[int]):
        atoms = set(comp_atoms) | set(spiros)
        # name via the first spiro atom (numbering is component-intrinsic)
        named = _name_spiro_component(mol, atoms, spiros[0])
        if named is None:
            return None
        ext = _extract_subfragment(mol, atoms)
        if ext is None:
            return None
        return named[0], named[1], ext

    mid_named = _name_comp(middle[0], middle[1])
    if mid_named is None:
        return None
    mid_name, mid_loc, mid_ext = mid_named

    term_named = []
    for comp, sa in terminals:
        tn = _name_comp(comp, [sa])
        if tn is None:
            return None
        term_named.append((tn, sa))

    # All three components must be identical (isomorphic skeleton + same name).
    canon_mid = Chem.MolToSmiles(mid_ext[0])
    for (tn, _sa) in term_named:
        if tn[0] != mid_name:
            return None
        if Chem.MolToSmiles(tn[2][0]) != canon_mid:
            return None

    # Spiro locants (canonical, symmetry-lowest) on each component.
    mid_loc_1 = _canonical_spiro_locant(mid_ext, mid_loc, middle[1][0])
    mid_loc_2 = _canonical_spiro_locant(mid_ext, mid_loc, middle[1][1])
    if mid_loc_1 is None or mid_loc_2 is None:
        return None
    term_locs = []
    for (tn, sa) in term_named:
        tl = _canonical_spiro_locant(tn[2], tn[1], sa)
        if tl is None:
            return None
        term_locs.append((tl, sa, tn))

    # Middle spiro-atom locants: the one shared with the unprimed terminal is
    # primed as the lower, the other primed as the higher; per pick the
    # lowest spiro-atom locant set overall. Both terminals identical here, so
    # order the middle's two primed locants ascending and pair each spiro atom.
    # Map: spiro atom -> its middle-component locant.
    mid_locant_of = {middle[1][0]: mid_loc_1, middle[1][1]: mid_loc_2}
    # Choose which terminal is unprimed vs double-primed to minimise the full
    # spiro-atom locant set (unprimed term loc, its middle primed loc,
    # other middle primed loc, double-primed term loc).
    best = None
    for order in ([0, 1], [1, 0]):
        (u_loc, u_sa, u_tn) = term_locs[order[0]]
        (d_loc, d_sa, d_tn) = term_locs[order[1]]
        u_mid = mid_locant_of[u_sa]
        d_mid = mid_locant_of[d_sa]
        locset = (u_loc, u_mid, d_mid, d_loc)
        key = (min(u_loc, d_loc), u_loc, u_mid, d_mid, d_loc)
        if best is None or key < best[0]:
            best = (key, u_loc, u_mid, d_mid, d_loc, order)
    _, u_loc, u_mid, d_mid, d_loc, _order = best

    component_name = _strip_consumed_indicated_h(mid_name, u_loc)
    # a,a':b',b''-dispiroter[component]
    name = f"{u_loc},{u_mid}':{d_mid}',{d_loc}''-dispiroter[{component_name}]"

    # Combined locants: unprimed terminal, primed middle, double-primed terminal.
    (u_loc0, u_sa0, u_tn0) = term_locs[_order[0]]
    (d_loc0, d_sa0, d_tn0) = term_locs[_order[1]]
    combined: Dict[int, _Locant] = {}
    for atom_idx, locant in u_tn0[1].items():
        combined[atom_idx] = locant
    for atom_idx, locant in mid_loc.items():
        if atom_idx in combined:
            continue
        combined[atom_idx] = (locant, "'")
    for atom_idx, locant in d_tn0[1].items():
        if atom_idx in combined:
            continue
        combined[atom_idx] = (locant, "''")
    if not (set(combined.keys()) >= all_ring_atoms):
        return None  # coverage invariant (Pitfall 7)
    return name, all_ring_atoms, combined


def is_dispiroter(mol) -> bool:
    """: three identical polycyclic components sharing two spiro atoms."""
    if mol is None:
        return False
    return _name_dispiroter_core(mol) is not None


def name_dispiroter(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the ``dispiroter`` name. Fail-closed (see
    ``_name_dispiroter_core``)."""
    core = _name_dispiroter_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def _fused_ring_components(mol) -> List[Tuple[Set[int], Set[int]]]:
    """Partition the ring system into fused-ring COMPONENTS (maximal ring sets
    connected by fused edges, i.e. sharing >=2 atoms). Returns a list of
    (component_atoms, spiro_atoms_touched) tuples. Spiro atoms (the single-atom
    junctions between components) belong to every component they touch."""
    spiro = set(get_spiro_atoms(mol))
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    n = len(rings)
    fadj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if len(rings[i] & rings[j]) >= 2:
                fadj[i].append(j)
                fadj[j].append(i)
    seen: Set[int] = set()
    out: List[Tuple[Set[int], Set[int]]] = []
    for s in range(n):
        if s in seen:
            continue
        stack = [s]
        comp_rings = {s}
        seen.add(s)
        while stack:
            c = stack.pop()
            for nb in fadj[c]:
                if nb not in seen:
                    seen.add(nb)
                    comp_rings.add(nb)
                    stack.append(nb)
        atoms: Set[int] = set()
        for ridx in comp_rings:
            atoms |= rings[ridx]
        out.append((atoms, atoms & spiro))
    return out


def _name_unbranched_polyspiro_different_core(mol):
    """: unbranched polyspiro with DIFFERENT ring components, >=1
    polycyclic, along a LINEAR chain (terminal-middle-terminal). Returns
    ``(name, all_ring_atoms, combined_locants)`` or None (fail-closed).

    Currently builds the 3-component / 2-spiro-atom linear chain
    (``dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]``): a central component
    holding BOTH spiro atoms and two terminals each holding one. Terminals are
    cited in alphanumerical order; the first is unprimed, the middle
    primed, the last double-primed. Fail closed unless the components are NOT
    all identical (else dispiroter, Task 3), the chain is unbranched (no
    component holds >2 spiro atoms), and every component names."""
    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) != 2:
        return None
    ri = mol.GetRingInfo()
    all_rings = [list(r) for r in ri.AtomRings()]
    all_ring_atoms: Set[int] = set()
    for r in all_rings:
        all_ring_atoms.update(r)
    for atom in atoms_of(mol):  # unsubstituted only
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None

    comps = _fused_ring_components(mol)
    if len(comps) != 3:
        return None
    # scope: >=1 component must be polycyclic (a component spanning >=2
    # fused rings). A pure spiro of monocyclic rings uses the von-Baeyer
    # descriptor form (dispiro[a.b.c.d]) via name_spiro_system, not this
    # component-name form.
    ri_full = mol.GetRingInfo()
    all_ring_list = [set(r) for r in ri_full.AtomRings()]
    def _ring_count_in(atoms: Set[int]) -> int:
        return sum(1 for r in all_ring_list if r <= atoms)
    if not any(_ring_count_in(atoms) >= 2 for atoms, _s in comps):
        return None
    middle = None
    terminals = []
    for atoms, spiros in comps:
        if len(spiros) == 2:
            if middle is not None:
                return None  # >1 two-spiro component -> branched (Task 6)
            middle = (atoms, sorted(spiros))
        elif len(spiros) == 1:
            terminals.append((atoms, next(iter(spiros))))
        else:
            return None
    if middle is None or len(terminals) != 2:
        return None

    # Name each terminal + spiro locant.
    term_info = []
    for atoms, sa in terminals:
        named = _name_spiro_component(mol, atoms, sa)
        if named is None:
            return None
        ext = _extract_subfragment(mol, atoms)
        if ext is None:
            return None
        loc = _canonical_spiro_locant(ext, named[1], sa)
        if loc is None:
            return None
        term_info.append({'atoms': atoms, 'spiro': sa, 'name': named[0],
                          'loc': loc, 'map': named[1]})

    # alphanumerical: first-cited (unprimed) terminal is the lower key.
    if _component_alpha_key(term_info[0]['name']) <= _component_alpha_key(term_info[1]['name']):
        first, last = term_info[0], term_info[1]
    else:
        first, last = term_info[1], term_info[0]

    # NOT all-identical (else dispiroter owns it).
    if first['name'] == last['name']:
        mid_named = _name_spiro_component(mol, middle[0], middle[1][0])
        if mid_named is not None and mid_named[0] == first['name']:
            return None

    # Middle component numbered so the FIRST (unprimed) terminal's junction spiro
    # atom is locant 1; the other spiro atom takes its walk position.
    mid_named = _name_spiro_component(mol, middle[0], first['spiro'])
    if mid_named is None:
        return None
    mid_name, mid_map = mid_named
    mid_loc_first = mid_map.get(first['spiro'])
    mid_loc_last = mid_map.get(last['spiro'])
    if mid_loc_first is None or mid_loc_last is None:
        return None

    #: build the combined indicated-H + λ front-of-name prefix (empty
    # for a plain, standard-valence system). Indicated-H that is NOT consumed by
    # the spiro atom is front-cited & primed; a λ (hypervalent) spiro atom is
    # cited as '<loc><prime>lambda<n>'. Then the descriptor cites the components
    # with their leading indicated-H stripped (it moved to the front).
    front_prefix, _stripped = _build_lambda_ih_front_prefix(mol, [
        {'id': 'first', 'name': first['name'], 'prime': '',
         'spiro_atoms': [first['spiro']], 'map': first['map']},
        {'id': 'mid', 'name': mid_name, 'prime': "'",
         'spiro_atoms': [first['spiro'], last['spiro']], 'map': mid_map},
        {'id': 'last', 'name': last['name'], 'prime': "''",
         'spiro_atoms': [last['spiro']], 'map': last['map']},
    ])
    if front_prefix:
        # Front prefix present -> the descriptor drops leading indicated-H that
        # was moved to the front; consumed-at-spiro stripping still
        # applies for any component with no front-cited indicated-H.
        _, first_bare = _extract_leading_indicated_h(first['name'])
        _, mid_bare = _extract_leading_indicated_h(mid_name)
        _, last_bare = _extract_leading_indicated_h(last['name'])
        first_name, mid_name_s, last_name = first_bare, mid_bare, last_bare
    else:
        first_name = _strip_consumed_indicated_h(first['name'], first['loc'])
        mid_name_s = _strip_consumed_indicated_h(mid_name, mid_loc_first)
        last_name = _strip_consumed_indicated_h(last['name'], last['loc'])
    name = (f"{front_prefix}dispiro[{first_name}-{first['loc']},{mid_loc_first}'-"
            f"{mid_name_s}-{mid_loc_last}',{last['loc']}''-{last_name}]")

    combined: Dict[int, _Locant] = {}
    for a, loc in first['map'].items():
        combined[a] = loc
    for a, loc in mid_map.items():
        if a in combined:
            continue
        combined[a] = (loc, "'")
    for a, loc in last['map'].items():
        if a in combined:
            continue
        combined[a] = (loc, "''")
    if not (set(combined.keys()) >= all_ring_atoms):
        return None
    return name, all_ring_atoms, combined


def is_branched_polyspiro(mol) -> bool:
    """: BRANCHED polyspiro — a central fused-ring component that carries
    THREE OR MORE spiro junctions (a branching node). Detection only: the full
    branched component-name build (heteromonocycle central components,
    3-junction walk) is a documented follow-on, so ``name_branched_polyspiro``
    fails closed. Recognising the class here routes it to a fail-closed tag
    instead of leaking a structure-dropping partial name from the ortho-fused
    path (accuracy-first: never a wrong name)."""
    if mol is None:
        return False
    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) < 3:
        return False
    # A pure spiro tree (n_rings == n_spiro + 1) of MONOCYCLES is a von-Baeyer
    # branched descriptor (Task 2) — not this component-name class.
    ri = mol.GetRingInfo()
    if ri.NumRings() == len(spiro_atoms) + 1:
        return False
    for _atoms, spiros in _fused_ring_components(mol):
        if len(spiros) >= 3:
            return True
    return False


def _number_central_monocycle_branch(
    mol, component_atoms: Set[int], spiro_atoms: List[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Number a monocyclic HETEROCYCLIC central branch component (>=2 spiro
    junctions) so heteroatoms take the lowest locant SET / element
    seniority), then the spiro-junction atoms take the lowest locant SET
    . Returns ``(hw_name, {orig_idx: locant})`` or None (fail-closed)."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    rings = frag.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = list(rings[0])
    ring_size = len(ring)
    ring_set = set(ring)
    hetero_frag = {fi for fi in ring
                   if frag.GetAtomWithIdx(fi).GetAtomicNum() not in (1, 6)}
    if not hetero_frag:
        return None
    # saturated only (mancude central branch is a follow-on)
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in frag.GetBonds()
           if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set):
        return None
    fadj: Dict[int, List[int]] = {a: [] for a in ring}
    for bond in frag.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            fadj[i].append(j)
            fadj[j].append(i)
    spiro_frag = {orig_to_frag[s] for s in spiro_atoms if s in orig_to_frag}
    if len(spiro_frag) != len(spiro_atoms):
        return None
    from ..data.hw_heteroatoms import HETEROATOM_PRIORITY as _HP

    def _sym(fi):
        return frag.GetAtomWithIdx(fi).GetSymbol()

    best = None
    for start in ring:
        for first in fadj[start]:
            path = [start, first]
            visited = {start, first}
            cur = first
            ok = True
            while len(path) < ring_size:
                nxts = [nb for nb in fadj[cur] if nb not in visited]
                if len(nxts) != 1:
                    ok = False
                    break
                cur = nxts[0]
                path.append(cur)
                visited.add(cur)
            if not ok or len(path) != ring_size:
                continue
            pos = {fi: idx + 1 for idx, fi in enumerate(path)}
            het_locs = sorted(pos[fi] for fi in hetero_frag)
            het_prio = sorted((pos[fi], _HP.get(_sym(fi), 999)) for fi in hetero_frag)
            spiro_locs = sorted(pos[fi] for fi in spiro_frag)
            key = (het_locs, [p for _, p in het_prio], spiro_locs)
            if best is None or key < best[0]:
                best = (key, pos)
    if best is None:
        return None
    pos = best[1]
    numbered_het = sorted(((pos[fi], _sym(fi)) for fi in hetero_frag))
    name = _hw_bracket_name(numbered_het, ring_size)
    if not name:
        return None
    a2l = {frag_to_orig[fi]: pos[fi] for fi in ring if fi in frag_to_orig}
    return name, a2l


def _name_branched_polyspiro_different_core(mol):
    """: branched polyspiro with a monocyclic-heterocycle CENTRAL branch
    node carrying THREE spiro junctions and three DIFFERENT terminal components,
    e.g. ``trispiro[cyclohexane-1,2'-[1,5]dithiocane-6',1''-cyclopentane-4',
    2'''-indene]``. Returns ``(name, all_ring_atoms, combined_locants)`` or None
    (fail-closed). Scope: exactly one central monocyclic-HW branch with 3 spiro
    junctions + three mono-junction terminals, all unsubstituted, every
    component nameable."""
    spiro_atoms = get_spiro_atoms(mol)
    if len(spiro_atoms) != 3:
        return None
    ri = mol.GetRingInfo()
    all_ring_atoms: Set[int] = set()
    for r in ri.AtomRings():
        all_ring_atoms.update(r)
    for atom in mol.GetAtoms():  # unsubstituted ring system only
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in all_ring_atoms:
            return None
    comps = _fused_ring_components(mol)
    central = None
    terminals = []
    for atoms, spiros in comps:
        if len(spiros) == 3:
            if central is not None:
                return None
            central = (atoms, sorted(spiros))
        elif len(spiros) == 1:
            terminals.append((atoms, next(iter(spiros))))
        else:
            return None
    if central is None or len(terminals) != 3:
        return None

    # Central branch: number it (heteroatoms lowest, then junctions lowest).
    cen = _number_central_monocycle_branch(mol, central[0], central[1])
    if cen is None:
        return None
    cen_name, cen_map = cen

    # Name each terminal + its own spiro locant.
    term_info = []
    for atoms, sa in terminals:
        named = _name_spiro_component(mol, atoms, sa)
        if named is None:
            return None
        ext = _extract_subfragment(mol, atoms)
        if ext is None:
            return None
        loc = _canonical_spiro_locant(ext, named[1], sa)
        if loc is None:
            return None
        term_info.append({
            'atoms': atoms, 'spiro': sa, 'name': named[0], 'loc': loc,
            'map': named[1], 'cen_loc': cen_map.get(sa),
        })
    if any(t['cen_loc'] is None for t in term_info):
        return None

    # alphanumerical citation: terminals cited in ascending name key.
    term_info.sort(key=lambda t: (_component_alpha_key(t['name']), t['cen_loc']))
    t1, t2, t3 = term_info

    # Branched descriptor: term1 - central - term2 - (back to
    # central) - term3, with sequential primes in citation order
    # (term1=unprimed, central=', term2='', term3=''').
    n1 = _strip_consumed_indicated_h(t1['name'], t1['loc'])
    ncen = _strip_consumed_indicated_h(cen_name, t1['cen_loc'])
    n2 = _strip_consumed_indicated_h(t2['name'], t2['loc'])
    n3 = _strip_consumed_indicated_h(t3['name'], t3['loc'])
    name = (
        f"trispiro[{n1}-{t1['loc']},{t1['cen_loc']}'-{ncen}-"
        f"{t2['cen_loc']}',{t2['loc']}''-{n2}-"
        f"{t3['cen_loc']}',{t3['loc']}'''-{n3}]"
    )

    combined: Dict[int, _Locant] = {}
    for a, loc in t1['map'].items():
        combined[a] = loc
    for a, loc in cen_map.items():
        if a not in combined:
            combined[a] = (loc, "'")
    for a, loc in t2['map'].items():
        if a not in combined:
            combined[a] = (loc, "''")
    for a, loc in t3['map'].items():
        if a not in combined:
            combined[a] = (loc, "'''")
    if not (set(combined.keys()) >= all_ring_atoms):
        return None
    return name, all_ring_atoms, combined


def name_branched_polyspiro(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """ branched polyspiro with different terminal components around a
    monocyclic-heterocycle central branch node. Fail-closed (return None) when
    the topology / components are outside the built class."""
    core = _name_branched_polyspiro_different_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def is_lambda_multiring_spiro(mol) -> bool:
    """P-24.8.1.3 detection: a single spiro atom shared by THREE OR MORE rings
    that carries a NONSTANDARD (λ) bonding number (e.g. a λ6 S in three
    monocyclic rings). ``get_spiro_atoms`` misses it (it requires exactly 2
    rings), so without this the system falls through to a partial monocyclic
    name. The component-name / von-Baeyer-λ build for this rare form is a
    documented follow-on (Task 8 deferred), so this is detection-only and routes
    to a FAIL-CLOSED tag — never a wrong (structure-dropping) name."""
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        in_rings = sum(1 for r in rings if idx in r)
        if in_rings < 3:
            continue
        if _nonstandard_bonding_number(mol, idx) is None:
            continue
        # cut vertex separating >=3 ring components -> spiro (not fused/bridged)
        _, adj = _ring_atom_graph(mol)
        comps = _ring_components_excluding(adj, idx)
        if len(comps) >= 3:
            return True
    return False


def name_lambda_multiring_spiro(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """ λ spiro atom in >=3 monocyclic rings — FAIL CLOSED (return
    None). The λ von-Baeyer-descriptor build is a documented follow-on; refusing
    here prevents a structure-dropping partial monocyclic name."""
    return None


def is_unbranched_polyspiro_different(mol) -> bool:
    """: unbranched polyspiro, different components, >=1 polycyclic."""
    if mol is None:
        return False
    return _name_unbranched_polyspiro_different_core(mol) is not None


def name_unbranched_polyspiro_different(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the unbranched-polyspiro-different name. Fail-closed."""
    core = _name_unbranched_polyspiro_different_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


# ============================================================================
# — spiro systems with at least one von Baeyer (bridged) ring component
# (a phase(c) — spiro-of-von-Baeyer). The existing spirobi / mixed-spiro-fused
# paths rely on ``get_spiro_atoms`` (atom in EXACTLY 2 SSSR rings) + the
# catalog/ortho-fused ``_name_fused_component``; a von Baeyer cage shares its
# spiro atom across >2 SSSR rings (cage-bridge spiro) and is named by its
# ``bicyclo[...]`` descriptor, so neither path fires and the whole system
# mis-routes to the pure-VB polycyclic-bridged branch (VonBaeyerAnalyzer
# invariant-fails -> unknown). This adds a dedicated, fail-closed path.
# ============================================================================


def _ring_atom_graph(mol) -> Tuple[Set[int], Dict[int, Set[int]]]:
    """Ring-atom set + adjacency limited to ring-ring bonds."""
    ring_atoms: Set[int] = set()
    for r in mol.GetRingInfo().AtomRings():
        ring_atoms.update(r)
    adj: Dict[int, Set[int]] = {a: set() for a in ring_atoms}
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_atoms and j in ring_atoms:
            adj[i].add(j)
            adj[j].add(i)
    return ring_atoms, adj


def _ring_components_excluding(
    adj: Dict[int, Set[int]], exclude: int,
) -> List[Set[int]]:
    """Connected components of the ring-atom graph with ``exclude`` removed."""
    seen: Set[int] = {exclude}
    comps: List[Set[int]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        comp: Set[int] = {start}
        seen.add(start)
        while stack:
            cur = stack.pop()
            for nbr in adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    comp.add(nbr)
                    stack.append(nbr)
        comps.append(comp)
    return comps


def find_monospiro_separation_atom(
    mol,
) -> Optional[Tuple[int, List[Set[int]]]]:
    """The unique ring atom whose removal splits the ring-atom graph into exactly
    two components (a monospiro junction), ROBUST to a spiro atom that sits in
    >2 SSSR rings (a von-Baeyer cage-bridge spiro, where ``get_spiro_atoms``
    returns nothing). A spiro atom has exactly four ring bonds (two into each
    component) and is a cut vertex of the ring-atom graph; a VB bridgehead is
    NOT a cut vertex (the other bridges keep the cage connected) and an
    ortho-fusion junction atom has three ring bonds, not four. Returns
    ``(spiro_atom, [component_a_atoms, component_b_atoms])`` or None when there
    is not exactly one such clean two-way split (polyspiro / fused / bridged)."""
    ring_atoms, adj = _ring_atom_graph(mol)
    candidates: List[Tuple[int, List[Set[int]]]] = []
    for atom in ring_atoms:
        if len(adj[atom]) != 4:
            continue
        comps = _ring_components_excluding(adj, atom)
        if len(comps) == 2 and all(adj[atom] & c for c in comps):
            candidates.append((atom, comps))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _name_carbocyclic_monocycle_component(
    mol, component_atoms: Set[int], spiro_center: int,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a single CARBOCYCLIC ring component of a spiro system, numbering it
    with locant 1 at the spiro atom (the side-ring convention). Returns
    ``(cyclo<N>ane / cyclo<stem>a-<enes>-diene / -ene, {orig_idx: locant})`` or
    None for a heteroatom ring (fail-closed).

     tail #7: a ring bearing C=C (the spiro-cyclohexadienone side of a
    spiro-quinone alkaloid) is now rendered as the mancude ``cyclohexa-2,5-diene``
    component. The double-bond locants are cited UNPRIMED here; the caller's spiro
    assembler is responsible for the primed/unprimed placement of the
    OTHER (non-first-cited) component and RT-verifies the whole name . The
    numbering walks from the spiro atom (locant 1) in the direction giving the
    C=C set the lowest locants; a saturated ring is byte-identical
    to the old output (no ene infix)."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    rings = frag.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = list(rings[0])
    for fi in ring:  # carbocyclic only
        if frag.GetAtomWithIdx(fi).GetAtomicNum() != 6:
            return None
    ring_set = set(ring)
    # Collect ring C=C bonds (frozenset pairs); any ring triple/aromatic bond is
    # out of this v1 scope (fail-closed -> a richer namer or abstain).
    double_pairs: Set[frozenset] = set()
    for bond in frag.GetBonds():
        if (bond.GetBeginAtomIdx() in ring_set
                and bond.GetEndAtomIdx() in ring_set):
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                double_pairs.add(frozenset(
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            elif bt != Chem.BondType.SINGLE:
                return None  # aromatic / triple ring bond -> out of scope
    fadj: Dict[int, List[int]] = {a: [] for a in ring}
    for bond in frag.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            fadj[i].append(j)
            fadj[j].append(i)
    start = orig_to_frag[spiro_center]
    n = len(ring)
    stem = _get_chain_prefix(n)
    best = None  # (ene_locant_tuple, path)
    for first in sorted(fadj[start]):
        path = [start, first]
        visited = {start, first}
        cur = first
        while len(path) < n:
            nxt = next((nb for nb in fadj[cur] if nb not in visited), None)
            if nxt is None:
                break
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(path) != n:
            continue
        pos = {a: i + 1 for i, a in enumerate(path)}
        # each ring C=C -> its lower endpoint locant; a bond closing n->1 is the
        # ring-closure double bond and would be cited at locant n.
        ene_locs = []
        for pr in double_pairs:
            a, b = tuple(pr)
            la, lb = pos[a], pos[b]
            # consecutive-in-ring endpoints: cite the lower, unless it is the
            # 1<->n closure bond (locant n).
            ene_locs.append(min(la, lb) if abs(la - lb) == 1 else n)
        key = tuple(sorted(ene_locs))
        if best is None or key < best[0]:
            best = (key, path)
    if best is None:
        return None
    ene_key, path = best
    a2l = {frag_to_orig[a]: i + 1 for i, a in enumerate(path)}
    if not double_pairs:
        return f"cyclo{stem}ane", a2l
    # mancude component: 'cyclohexa-2,5-diene' etc. Multiplied ene infix uses the
    # euphonic 'a' (di/tri-ene); a single C=C is 'cyclohex-2-ene'.
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    loc_str = ",".join(str(x) for x in ene_key)
    if len(ene_key) == 1:
        return f"cyclo{stem}-{loc_str}-ene", a2l
    mult = SIMPLE_MULTIPLIERS.get(len(ene_key))
    if mult is None:
        return None
    return f"cyclo{stem}a-{loc_str}-{mult}ene", a2l


def _tricyclo_plus_spiro_component(
    mol, component_atoms: Set[int], spiro_atom: Optional[int] = None,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """ spiro component that is a TRIcyclic+ von-Baeyer cage (possibly
    hetero / unsaturated). Names it as the full parent hydride via the audited
    ``analyze_cage_universal`` + ``_spell_ring_analysis``, returning
    ``(name, {orig_idx: locant})`` or None (fail-closed).

    ``spiro_atom`` (default None) is the spiro-junction atom index; when passed
    it makes the von-Baeyer numbering give that atom the lowest locant per
     (:10272;:10289 "the spiro atom... is given preference for low
    locant"). The spiro-atom locant is read by the caller off the returned map;
     arbitrates. Callers that use this only as a polycyclic-ness predicate
    pass no spiro atom, so the numbering is byte-identical for them."""
    from .terminal_ring import _spell_ring_analysis
    from .vonbaeyer_universal import (
        analyze_cage_universal,
        audit_von_baeyer_descriptor,
    )
    try:
        res = analyze_cage_universal(
            mol, cage_atoms=set(component_atoms), allow_mancude=True,
            spiro_atom=spiro_atom)
    except Exception:
        return None
    if res is None:
        return None
    # Re-prove the reconstruction audit at the emission point (kekulized), the
    # same guard terminal_ring applies before spelling a cage.
    kek = Chem.RWMol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    if not audit_von_baeyer_descriptor(
            kek.GetMol(), set(res.cage_atoms), res.atom_to_locant, res.descriptor):
        return None
    name = _spell_ring_analysis(res, None)  # full parent hydride (keeps the 'e')
    if not name or ' ' in name:
        return None
    return name, dict(res.atom_to_locant)


def _name_vonbaeyer_spiro_component(
    mol, component_atoms: Set[int], spiro_center: Optional[int] = None,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a von-Baeyer (bicyclic cage) component by its SYSTEMATIC
    ``bicyclo[...]alkane`` descriptor — NOT a retained name: the component-name
    spiro PIN cites the von Baeyer name, e.g. ``bicyclo[2.2.1]heptane``
    not ``norbornane`` (and the alphanumerical citation order depends on it).
    Returns ``(name, {orig_idx: locant})`` or None (non-bicyclo cage, tricyclo+,
    or unnamed heteroatom cage -> follow-on, fail-closed).

     /: a von Baeyer cage may carry skeletal heteroatoms
    named by 'a'-replacement. The descriptor is that of the corresponding
    all-CARBON cage. Among the admissible von-Baeyer numberings the one giving
    the LOWEST locant to the spiro atom is chosen '2,9'' locant set),
    then the lowest heteroatom locants — coordinated so the
    result is spelling-independent."""
    from .bicyclo import (
        _enumerate_bicyclo_numberings,
        _get_alkane_name,
        _legacy_bicyclo_numbering,
        find_true_bridgeheads,
        generate_bicyclo_descriptor,
        get_bicyclo_numbering,
        is_bicyclo_system,
    )
    from .locants import compare_locant_sets

    # Share the ONE heteroatom-seniority table the tricyclo+ spiro branch already
    # uses (VonBaeyerAnalyzer._locant_criteria_key), so both spiro von-Baeyer
    # branches rank heteroatoms by the same / order (do not
    # redefine or hard-code it here). Lazy import -- avoids a module-load cycle.
    from .polycyclic import VonBaeyerAnalyzer
    _HETERO_SENIORITY = VonBaeyerAnalyzer._HETERO_SENIORITY
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    ring_atom_idxs = [a.GetIdx() for a in frag.GetAtoms() if a.IsInRing()]
    hetero_frag = {
        i for i in ring_atom_idxs if frag.GetAtomWithIdx(i).GetAtomicNum() != 6
    }
    if hetero_frag:
        rw = Chem.RWMol(frag)
        for a in rw.GetAtoms():
            if a.GetAtomicNum() != 6:
                a.SetAtomicNum(6)
                a.SetNoImplicit(False)
                a.SetFormalCharge(0)
        skel = rw.GetMol()
        try:
            Chem.SanitizeMol(skel)
        except Exception:
            return None
    else:
        skel = frag
    if not is_bicyclo_system(skel):
        return None
    descriptor = generate_bicyclo_descriptor(skel)
    if not descriptor:
        return None

    # Ring multiple bonds of the component (mapped to the frag).:
    # low locants to spiro junction, then heteroatoms, then double bonds.
    frag_multibonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in frag.GetBonds()
        if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
        and b.GetBeginAtomIdx() in set(ring_atom_idxs)
        and b.GetEndAtomIdx() in set(ring_atom_idxs)
    ]
    spiro_frag = orig_to_frag.get(spiro_center) if spiro_center is not None else None
    if hetero_frag or spiro_frag is not None or frag_multibonds:
        # Enumerate the admissible von-Baeyer numberings (topology-defined on the
        # carbon skeleton) and pick per (lowest spiro-atom locant), then
        # (lowest heteroatom locants), then (lowest
        # double-bond locants). Deterministic tie-break.
        bridgeheads = list(find_true_bridgeheads(skel))
        if len(bridgeheads) != 2:
            return None
        cands: List[Dict[int, int]] = []
        legacy = _legacy_bicyclo_numbering(skel)
        if legacy:
            cands.append(legacy)
        cands.extend(_enumerate_bicyclo_numberings(skel, bridgeheads))
        if not cands:
            return None

        def _key(a2l):
            spiro_loc = (a2l.get(spiro_frag, 10 ** 6)
                         if spiro_frag is not None else 0)
            het = sorted(a2l[i] for i in hetero_frag if i in a2l)
            # [BBv2:9789]: on a heteroatom locant-SET tie the SENIOR
            # element (O > S > Se > Te > N > P >...) takes the LOWER locant.
            # Heteroatom locants ORDERED by decreasing element seniority (rank
            # ascending) -> compared as an ordered tuple, never via
            # compare_locant_sets (a set-compare is exactly the SP1.5 bug: it ties
            # {2,6} with itself and falls through to candidate/atom-index order).
            sen = tuple(loc for _rank, loc in sorted(
                (_HETERO_SENIORITY.get(frag.GetAtomWithIdx(i).GetSymbol(), 99),
                 a2l[i])
                for i in hetero_frag if i in a2l
            ))
            ene = sorted(min(a2l[a], a2l[b]) for a, b in frag_multibonds
                         if a in a2l and b in a2l)
            return (spiro_loc, het, sen, ene)

        best = None
        for c in cands:
            k = _key(c)
            if best is None:
                best = (k, c)
                continue
            #: lowest spiro-atom locant.
            if k[0] != best[0][0]:
                if k[0] < best[0][0]:
                    best = (k, c)
                continue
            #: lowest heteroatom locant SET.
            ch = compare_locant_sets(k[1], best[0][1])
            if ch != 0:
                if ch < 0:
                    best = (k, c)
                continue
            #: element-seniority tiebreak (ORDERED vector) on a SET tie.
            if k[2] != best[0][2]:
                if k[2] < best[0][2]:
                    best = (k, c)
                continue
            #: lowest double-bond locants.
            if compare_locant_sets(k[3], best[0][3]) < 0:
                best = (k, c)
        numbering = best[1]
    else:
        numbering = get_bicyclo_numbering(skel)
    if not numbering:
        return None
    name = descriptor + _get_alkane_name(len(ring_atom_idxs))
    a2l = {
        frag_to_orig[fi]: loc
        for fi, loc in numbering.items()
        if fi in frag_to_orig
    }
    return name, a2l


# --- Carbocyclic-PAH numbered templates (a phase(b)) ----------------------
# A fused CARBOCYCLIC ring system (fluorene,...) is not in the heterocycle
# catalog and is not an ortho-fused BICYCLIC, so ``_name_fused_component``
# declines it. Its IUPAC peripheral numbering is a fixed table; store it as a
# numbered template (locants from OPSIN ``-o extendedsmi`` $_AV:) and map a
# component's atoms onto it by substructure match. Fail-closed beyond the
# registered templates (a general carbo-PAH numbering engine is out of scope).


def _carbopah_loc_base(loc) -> int:
    """Base integer of a (possibly letter-suffixed) locant: '9a' -> 9."""
    return int("".join(c for c in str(loc) if c.isdigit()))


# (display_name, template SMILES [atom order == OPSIN locant order], locants).
_CARBO_PAH_TEMPLATE_SPECS = [
    # fluorene |$_AV:1;2;3;4;4a;4b;5;6;7;8;8a;9;9a$| (C9 = the sp3 spiro centre)
    ("9H-fluorene", "C1=CC=CC=2C3=CC=CC=C3CC12",
     [1, 2, 3, 4, "4a", "4b", 5, 6, 7, 8, "8a", 9, "9a"]),
    # 2H-indene |$_AV:1;2;3;3a;4;5;6;7;7a$| (C2 = the sp3 spiro centre)
    ("2H-indene", "C=1CC=C2C=CC=CC12",
     [1, 2, 3, "3a", 4, 5, 6, 7, "7a"]),
]


def _build_carbo_pah_templates():
    out = []
    for name, smi, locants in _CARBO_PAH_TEMPLATE_SPECS:
        tmpl = Chem.MolFromSmiles(smi)
        if tmpl is not None:
            out.append((name, tmpl, locants, Chem.CanonSmiles(smi)))
    return out


_CARBO_PAH_TEMPLATES = _build_carbo_pah_templates()


def _name_carbopah_spiro_component(
    mol, component_atoms: Set[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a fused CARBOCYCLIC PAH component (e.g. fluorene) by matching it to
    a numbered template. Returns ``(name, {orig_idx: base_int_locant})`` or None
    (not a registered carbo-PAH skeleton -> fail-closed)."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    frag_canon = Chem.MolToSmiles(frag)
    for name, tmpl, locants, tmpl_canon in _CARBO_PAH_TEMPLATES:
        if frag_canon != tmpl_canon:
            continue
        match = frag.GetSubstructMatch(tmpl)  # match[i] = frag atom for template i
        if not match or len(match) != tmpl.GetNumAtoms():
            continue
        a2l: Dict[int, int] = {}
        for ti, frag_atom in enumerate(match):
            orig = frag_to_orig.get(frag_atom)
            if orig is not None:
                a2l[orig] = _carbopah_loc_base(locants[ti])
        return name, a2l
    return None


# --- Fused-HETEROCYCLE numbered templates for spiro components ---------------
# A fused heterocyclic ring system whose ring heteroatom is the (hypervalent, λ)
# spiro atom cannot be matched against the global fused-heterocycle catalog: the
# extracted fragment carries a hypervalent heteroatom (e.g. λ4 S, λ5 P) so its
# skeleton SMILES does not equal any catalog key. This local, normalized-skeleton
# template table names those fused spiro components: it strips the excess Hs on
# the hypervalent ring heteroatoms to standard valence, then matches by canonical
# SMILES and reads the fixed IUPAC peripheral numbering. Fail-closed beyond the
# registered templates (a general fused-hetero spiro-component engine is out of
# scope). Locants are (display_name, skeleton SMILES, per-atom IUPAC locants in
# that SMILES' atom order).
_FUSED_HET_SPIRO_TEMPLATE_SPECS = [
    # [1,3,2]benzoxazaphosphole OPSIN 'O1PNC2=C1C=CC=C2' |$_AV:1;2;3;3a;7a;7;6;5;4$|
    # (P = the λ spiro centre at locant 2). Cited with front indicated-H (3H).
    ("3H-[1,3,2]benzoxazaphosphole", "O1PNC2=C1C=CC=C2",
     [1, 2, 3, "3a", "7a", 7, 6, 5, 4]),
    # benzo[1,2-c:4,5-c']dithiophene OPSIN 'C1C=2C(CS1)=CC=1C(=CSC1)C2'
    # |$_AV:1;8a;3a;3;2;4;4a;7a;7;6;5;8$| (both S = λ spiro centres at 2 and 6).
    ("1H,3H-benzo[1,2-c:4,5-c']dithiophene", "C1C=2C(CS1)=CC=1C(=CSC1)C2",
     [1, "8a", "3a", 3, 2, 4, "4a", "7a", 7, 6, 5, 8]),
    # [1,2,3]benzoxadithiole OPSIN 'O1SSC2=C1C=CC=C2' |$_AV:1;2;3;3a;7a;7;6;5;4$|
    # (the middle S = the λ spiro centre at locant 2). Component of the
    # target (the Blue Book).
    ("[1,2,3]benzoxadithiole", "O1SSC2=C1C=CC=C2",
     [1, 2, 3, "3a", "7a", 7, 6, 5, 4]),
    # dibenzo[b,d]thiophene OPSIN 'C1=CC=CC=2SC3=C(C21)C=CC=C3'
    # |$_AV:1;2;3;4;4a;5;5a;9a;9b;9;8;7;6$| (S = the λ spiro centre at locant 5).
    # PIN form for the spiro name (the Blue Book); the standalone catalog
    # still uses the short 'dibenzothiophene' — untouched here.
    ("dibenzo[b,d]thiophene", "C1=CC=CC=2SC3=C(C21)C=CC=C3",
     [1, 2, 3, 4, "4a", 5, "5a", "9a", "9b", 9, 8, 7, 6]),
]


def _build_fused_het_spiro_templates():
    out = []
    for name, smi, locants in _FUSED_HET_SPIRO_TEMPLATE_SPECS:
        tmpl = Chem.MolFromSmiles(smi)
        if tmpl is not None:
            out.append((name, tmpl, locants, Chem.CanonSmiles(smi)))
    return out


_FUSED_HET_SPIRO_TEMPLATES = _build_fused_het_spiro_templates()


def _normalize_hypervalent_ring_heteroatoms(frag) -> Optional[Chem.Mol]:
    """Return a copy of ``frag`` with excess explicit Hs stripped from its
    hypervalent (λ) ring heteroatoms so the resulting skeleton matches a
    standard-valence template. None if the normalized molecule cannot sanitize."""
    rw = Chem.RWMol(frag)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() not in (1, 6):
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
    norm = rw.GetMol()
    try:
        Chem.SanitizeMol(norm)
    except Exception:
        return None
    return norm


def _name_fused_het_spiro_component(
    mol, component_atoms: Set[int],
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Name a fused HETEROCYCLIC spiro component whose ring heteroatom is the
    (hypervalent, λ) spiro atom, by matching the normalized skeleton to a
    numbered template. Returns ``(name, {orig_idx: base_int_locant})`` or None
    (not a registered fused-hetero spiro skeleton -> fail-closed)."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    norm = _normalize_hypervalent_ring_heteroatoms(frag)
    if norm is None:
        return None
    norm_canon = Chem.MolToSmiles(norm)
    for name, tmpl, locants, tmpl_canon in _FUSED_HET_SPIRO_TEMPLATES:
        if norm_canon != tmpl_canon:
            continue
        match = norm.GetSubstructMatch(tmpl)  # match[i] = norm atom for template i
        if not match or len(match) != tmpl.GetNumAtoms():
            continue
        a2l: Dict[int, int] = {}
        for ti, norm_atom in enumerate(match):
            # norm shares atom indices with frag (same RWMol atom order).
            orig = frag_to_orig.get(norm_atom)
            if orig is not None:
                a2l[orig] = _carbopah_loc_base(locants[ti])
        return name, a2l
    return None


def _component_rings(mol, component_atoms: Set[int]) -> List[List[int]]:
    """Rings of ``mol`` fully contained in ``component_atoms`` (the list-of-rings
    interface that ``_name_fused_component`` expects)."""
    return [
        list(r) for r in mol.GetRingInfo().AtomRings()
        if set(r) <= component_atoms
    ]


def _hw_bracket_name(heteroatoms: List[Tuple[int, str]], ring_size: int) -> str:
    """Hantzsch-Widman name of a monocyclic heterocycle spiro component with the
     enclosing-marks convention: a multiplied-locant HW name cites its
    heteroatom locants inside square brackets, grouped by element in HW citation
    order (O > S > Se > Te > N > P...), each group ascending, e.g.
    ``[1,5]dithiocane`` and ``[1,3,5,2]triazaphosphinine``.

    ``heteroatoms`` is the numbered ``(locant, element)`` set of the chosen ring
    numbering (saturated ring assumed; mancude rings are handled by the caller
    via ``build_hw_name``'s aromatic path). A single-heteroatom ring carries no
    locant/bracket at all (``thiane``, ``thiolane``)."""
    from .heterocycles import build_hw_name
    # Saturation: a saturated monocyclic HW component (thiane, dithiocane,...);
    # the mancude/aromatic case is decided by the caller.
    if len(heteroatoms) <= 1:
        # No bracket, no locant single-heteroatom stems).
        return build_hw_name(heteroatoms, ring_size, True, False)
    # Multi-heteroatom: build the plain HW name, then splice the ascending
    # locant prefix into square brackets with element-citation order.
    from ..data.hw_heteroatoms import HETEROATOM_PRIORITY as _HP
    plain = build_hw_name(heteroatoms, ring_size, True, False)
    if not plain:
        return None  # HW has no prefix for one of these elements -> refuse
    # Strip the leading 'x,y-' locant prefix build_hw_name emits.
    import re as _re
    m = _re.match(r'^[\d,]+-(.+)$', plain)
    stem = m.group(1) if m else plain
    # Element-citation-order locant list.
    by_elem: Dict[str, List[int]] = {}
    for loc, el in heteroatoms:
        by_elem.setdefault(el, []).append(loc)
    order = sorted(by_elem.keys(), key=lambda e: _HP.get(e, 999))
    bracket_locs = []
    for el in order:
        bracket_locs.extend(sorted(by_elem[el]))
    return f"[{','.join(str(x) for x in bracket_locs)}]{stem}"


def _name_hw_monocycle_component(
    mol, component_atoms: Set[int], spiro_center: int,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """/.7/.8: name a monocyclic HETEROCYCLIC ring spiro component by its
    Hantzsch-Widman name with the enclosing-marks convention (``thiane``,
    ``thiolane``, ``[1,5]dithiocane``, ``[1,3,5,2]triazaphosphinine``). Returns
    ``(name, {orig_idx: locant})`` or None (fail-closed) for a carbocyclic ring
    (handled elsewhere) or any ring whose numbering cannot be resolved
    deterministically.

    Numbering /: heteroatoms take the lowest locant SET
    first (element seniority tie-break), then among those numberings the spiro
    atom takes the lowest locant. A λ (nonstandard-valence) spiro atom is NOT
    λ-cited inside the component name — the λ descriptor is a front-of-name
    prefix, so the ring is numbered/named as if the spiro atom were of
    standard valence."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    rings = frag.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = list(rings[0])
    ring_size = len(ring)
    # Must contain at least one skeletal heteroatom (else carbocyclic path).
    hetero_frag = {fi for fi in ring
                   if frag.GetAtomWithIdx(fi).GetAtomicNum() not in (1, 6)}
    if not hetero_frag:
        return None
    # Saturated ring only (mancude monocyclic HW spiro components are a
    # follow-on): every ring bond single. (An aromatic/mancude ring like the
    # item-2 triazaphosphinine is delegated to the mancude branch below.)
    ring_set = set(ring)
    has_ring_multibond = any(
        b.GetBondType() != Chem.BondType.SINGLE
        for b in frag.GetBonds()
        if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set
    )
    # Element symbols for the frag heteroatoms.
    def _sym(fi):
        return frag.GetAtomWithIdx(fi).GetSymbol()
    # Adjacency around the ring.
    fadj: Dict[int, List[int]] = {a: [] for a in ring}
    for bond in frag.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            fadj[i].append(j)
            fadj[j].append(i)
    spiro_frag = orig_to_frag.get(spiro_center)
    if spiro_frag is None:
        return None
    # Enumerate all ring numberings (each start atom, each direction).
    from ..data.hw_heteroatoms import HETEROATOM_PRIORITY as _HP
    best = None  # (het_locset, het_priority_key, spiro_loc, walk)
    for start in ring:
        for first in fadj[start]:
            path = [start, first]
            visited = {start, first}
            cur = first
            ok = True
            while len(path) < ring_size:
                nxts = [nb for nb in fadj[cur] if nb not in visited]
                if len(nxts) != 1:
                    ok = False
                    break
                cur = nxts[0]
                path.append(cur)
                visited.add(cur)
            if not ok or len(path) != ring_size:
                continue
            pos = {fi: idx + 1 for idx, fi in enumerate(path)}
            het_locs = sorted(pos[fi] for fi in hetero_frag)
            # Element-seniority tie-break: lowest locants to the most senior
            # element. Represent as (locant, priority) sorted.
            het_prio = sorted((pos[fi], _HP.get(_sym(fi), 999)) for fi in hetero_frag)
            spiro_loc = pos[spiro_frag]
            key = (het_locs, [p for _, p in het_prio], spiro_loc)
            if best is None or key < best[0]:
                best = (key, pos)
    if best is None:
        return None
    pos = best[1]
    numbered_het = sorted(((pos[fi], _sym(fi)) for fi in hetero_frag))
    is_mancude = has_ring_multibond or any(
        frag.GetAtomWithIdx(fi).GetIsAromatic() for fi in ring
    )
    if is_mancude:
        # Mancude/aromatic monocyclic HW component: build_hw_name aromatic path,
        # then apply the bracket convention.
        from .heterocycles import build_hw_name
        plain = build_hw_name(numbered_het, ring_size, False, True)
        if not plain:
            return None  # HW has no prefix for one of these elements -> refuse
        if len(numbered_het) > 1:
            import re as _re
            m = _re.match(r'^[\d,]+-(.+)$', plain)
            stem = m.group(1) if m else plain
            by_elem: Dict[str, List[int]] = {}
            for loc, el in numbered_het:
                by_elem.setdefault(el, []).append(loc)
            order = sorted(by_elem.keys(), key=lambda e: _HP.get(e, 999))
            bl = []
            for el in order:
                bl.extend(sorted(by_elem[el]))
            name = f"[{','.join(str(x) for x in bl)}]{stem}"
        else:
            name = plain
    else:
        # retained heterocycle PIN (piperidine, pyrrolidine, morpholine,
        # piperazine,...) is preferred over the Hantzsch-Widman systematic stem
        # (azinane, azolidine, 1,4-oxazinane, 1,4-diazinane) for a spiro COMPONENT
        # too: names each component by its own preferred ring name, and
        # makes the retained saturated-heterocycle name the PIN. The
        # retained name and the HW stem share the SAME heteroatom-lowest-
        # locant numbering (N=1 for piperidine; O=1,N=4 for morpholine), so the `pos`
        # map computed above is unchanged -- only the ring word swaps. `frag` is the
        # extracted single saturated ring, so get_retained_name (keyed by exact
        # canonical SMILES over the PIN-gated ALL_RETAINED_NAMES) returns a name ONLY
        # for that ring: oxane/thiane/oxolane/thiolane/azetidine map to their own HW
        # word (no-op) and non-retained rings (azepane,...) return None (stay HW).
        # Every spiro emission is offer-RT-gated downstream, so a numbering the swap
        # would misstate abstains rather than ships. (a phase, 11C2 #31.)
        name = None
        try:
            from ..data import get_retained_name
            retained = get_retained_name(Chem.MolToSmiles(frag))
        except Exception:
            retained = None
        if retained:
            name = retained
        else:
            name = _hw_bracket_name(numbered_het, ring_size)
    if not name:
        return None
    a2l = {frag_to_orig[fi]: pos[fi] for fi in ring if fi in frag_to_orig}
    return name, a2l


def _name_skeletal_replacement_monocycle_component(
    mol, component_atoms: Set[int], spiro_center: int,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """: a SATURATED monocyclic ring too large for a Hantzsch-Widman stem
    (ring size > 10) that carries skeletal heteroatoms is a skeletal-replacement
    ('a') component. It is named here as its all-carbon parent (``cyclododecane``,
    ``cycloundecane``...) — the form that goes INSIDE the spiro bracket — and its
    ring heteroatoms are returned in the locant map so ``_name_spiro_vonbaeyer_core``
    can hoist them to the front 'a'-prefix (``2',12'-dioxa``) per. This is
    the ``... before skeletal replacement`` two-step: name the hydrocarbon
    ring system first, apply 'a' prefixes second.

    Ring is numbered spiro-atom = 1, then the direction giving the lowest
    heteroatom locant SET (the front-prefix locants). Returns
    ``(carbon_parent_name, {orig_idx: locant})`` covering EVERY ring atom, or None
    (fail-closed): carbocyclic ring, ring size <= 10 (HW applies), any ring
    unsaturation (a mancude large heteromonocycle needs indicated-H — a follow-on),
    or a ring the walk cannot resolve deterministically."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    frag, orig_to_frag = extracted
    rings = frag.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = list(rings[0])
    ring_size = len(ring)
    if ring_size <= 10:
        return None  # Hantzsch-Widman stem exists -> not this path
    ring_set = set(ring)
    hetero = {fi for fi in ring
              if frag.GetAtomWithIdx(fi).GetAtomicNum() not in (1, 6)}
    if not hetero:
        return None  # carbocyclic large ring -> _name_carbocyclic_monocycle_component
    # Saturated ring only: every ring bond single (a mancude large heteromonocycle
    # would need indicated hydrogen inside the carbon-parent stem, unsupported here).
    if any(
        b.GetBondType() != Chem.BondType.SINGLE
        for b in frag.GetBonds()
        if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set
    ):
        return None
    spiro_frag = orig_to_frag.get(spiro_center)
    if spiro_frag is None or spiro_frag not in ring_set:
        return None
    fadj: Dict[int, List[int]] = {a: [] for a in ring}
    for bond in frag.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            fadj[i].append(j)
            fadj[j].append(i)
    # Number from the spiro atom (locant 1); pick the walk direction giving the
    # lowest heteroatom locant set low-locant preference, applied to the
    # 'a'-prefix locants since the spiro atom is fixed at 1).
    best = None  # (het_locs, pos)
    for first in fadj[spiro_frag]:
        path = [spiro_frag, first]
        visited = {spiro_frag, first}
        cur = first
        ok = True
        while len(path) < ring_size:
            nxts = [nb for nb in fadj[cur] if nb not in visited]
            if len(nxts) != 1:
                ok = False
                break
            cur = nxts[0]
            path.append(cur)
            visited.add(cur)
        if not ok or len(path) != ring_size:
            continue
        pos = {fi: idx + 1 for idx, fi in enumerate(path)}
        het_locs = sorted(pos[fi] for fi in hetero)
        if best is None or het_locs < best[0]:
            best = (het_locs, pos)
    if best is None:
        return None
    pos = best[1]
    parent = f"cyclo{_get_alkane_name(ring_size)}"
    frag_to_orig = {v: k for k, v in orig_to_frag.items()}
    a2l = {frag_to_orig[fi]: pos[fi] for fi in ring if fi in frag_to_orig}
    if len(a2l) != ring_size:
        return None
    return parent, a2l


def _name_spiro_component(
    mol, component_atoms: Set[int], spiro_center: int,
) -> Optional[Tuple[str, Dict[int, int]]]:
    """Dispatch one spiro component to the right namer, in order: a single
    saturated carbocyclic ring; a monocyclic heterocycle (Hantzsch-Widman); a
    von Baeyer cage (systematic bicyclo); a fused heterocycle (catalog, e.g.
    xanthene); a fused carbocyclic PAH (template, e.g. fluorene). Returns
    ``(name, {orig_idx: locant})`` or None (fail-closed when none of these name
    the component)."""
    extracted = _extract_subfragment(mol, component_atoms)
    if extracted is None:
        return None
    if extracted[0].GetRingInfo().NumRings() == 1:
        carbo = _name_carbocyclic_monocycle_component(
            mol, component_atoms, spiro_center
        )
        if carbo is not None:
            return carbo
        # Hantzsch-Widman stems exist only for ring sizes 3-10; a larger saturated
        # heteromonocycle is a skeletal-replacement ('a') component:
        # named as its carbon parent here, heteroatoms hoisted to the front by
        # _name_spiro_vonbaeyer_core.
        ring = extracted[0].GetRingInfo().AtomRings()[0]
        if len(ring) > 10:
            return _name_skeletal_replacement_monocycle_component(
                mol, component_atoms, spiro_center
            )
        return _name_hw_monocycle_component(mol, component_atoms, spiro_center)
    vb = _name_vonbaeyer_spiro_component(mol, component_atoms, spiro_center)
    if vb is not None:
        return vb
    fused = _name_fused_component(mol, _component_rings(mol, component_atoms))
    if fused is not None:
        return fused
    carbopah = _name_carbopah_spiro_component(mol, component_atoms)
    if carbopah is not None:
        return carbopah
    fused_het = _name_fused_het_spiro_component(mol, component_atoms)
    if fused_het is not None:
        return fused_het
    # LAST resort: a TRIcyclic+ von-Baeyer cage component (the
    # fused-tricyclic half of a spiro terpenoid / alkaloid) that no retained /
    # catalog fused name above covers. Systematic von Baeyer is the last resort
    # for a fused system retained names win), so this MUST sit after every
    # catalog namer -- otherwise it intercepts e.g. benzo[1,2-c:4,5-c']dithiophene.
    # Thread the spiro junction so gives it the lowest locant.
    return _tricyclo_plus_spiro_component(
        mol, set(component_atoms), spiro_atom=spiro_center)


def _cage_side_ene(mol, cage_a: Set[int], cage_b: Set[int],
                   locmap_a: Dict[int, int], locmap_b: Dict[int, int]):
    """Per-side ring DOUBLE-bond locants for the von-Baeyer CAGE components of a
    monospiro system, each in that component's OWN von-Baeyer numbering.

    Returns ``(side_a_locs, side_b_locs)`` (sorted int lists), or None
    (fail-closed) if any cage ring bond is aromatic / a triple bond / needs a
    compound (non-consecutive) locant. Only bonds with BOTH endpoints in a cage
    side are considered -- a mancude / aromatic / HW / catalog component carries
    its unsaturation INSIDE its own retained name and is deliberately ignored
    here (so its aromatic bonds never force a decline)."""
    cage_all = cage_a | cage_b
    la: List[int] = []
    lb: List[int] = []
    for bond in bonds_of(mol):
        x, y = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if x not in cage_all or y not in cage_all:
            continue
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            continue
        if bt != Chem.BondType.DOUBLE:
            return None  # aromatic / triple inside a cage -> unsupported
        if x in cage_a and y in cage_a:
            lx, ly, tgt = locmap_a.get(x), locmap_a.get(y), la
        elif x in cage_b and y in cage_b:
            lx, ly, tgt = locmap_b.get(x), locmap_b.get(y), lb
        else:
            return None  # double bond straddling two components (shouldn't occur)
        if lx is None or ly is None or abs(lx - ly) != 1:
            return None  # compound / cross-component locant -> unsupported
        tgt.append(min(lx, ly))
    la.sort()
    lb.sort()
    return la, lb


def _splice_cage_ene(component_name: str, int_locs: List[int],
                     prime: str) -> Optional[str]:
    """Splice ring unsaturation INSIDE a von-Baeyer CAGE spiro component name,
    re-anchored to that component's numbering, per (NOT appended after
    the spiro bracket -- that is OPSIN-grammar-invalid):

        'bicyclo[3.2.1]octane' + [3], '' -> 'bicyclo[3.2.1]oct-3-ene'
        'bicyclo[2.2.2]octane' + [2,5],'' -> 'bicyclo[2.2.2]octa-2,5-diene'
        <cage in the primed component> -> 'bicyclo[..]...-3'-ene' (prime kept)

    ``int_locs`` are the component's OWN double-bond locants; ``prime`` is '' for
    the unprimed component or "'" for the primed one. Returns the unchanged name
    when there is no cage unsaturation, or None (fail-closed) if the trailing
    hydride word is not an '...ane' this transform recognises."""
    if not int_locs:
        return component_name
    head, sep, word = component_name.rpartition(']')
    if not sep or not word.endswith('ane'):
        return None
    stem = word[:-3]  # strip the 'ane' hydride suffix -> 'oct', 'nonacos',...
    toks = ','.join(f"{n}{prime}" for n in int_locs)
    if len(int_locs) == 1:
        ene = f"{stem}-{toks}-ene"
    else:
        from ..assembly.naming_utils import get_suffix_multiplier_prefix
        mult = get_suffix_multiplier_prefix(len(int_locs), 'ene')
        ene = f"{stem}a-{toks}-{mult}ene"  # euphonic 'a' before di/tri...
    return f"{head}{sep}{ene}"


def _name_spiro_vonbaeyer_core(mol):
    """Shared core for is_spiro_vonbaeyer / name_spiro_vonbaeyer. Returns
    ``(name, all_ring_atoms, combined_locants)`` or None (fail-closed)."""
    found = find_monospiro_separation_atom(mol)
    if found is None:
        return None
    spiro_center, (comp_a, comp_b) = found
    all_ring_atoms: Set[int] = comp_a | comp_b | {spiro_center}
    # Substituents are ALLOWED: the parent name + combined_locants (incl. primed
    # component locants) are numbered here, and the downstream cascade-step-6
    # supplier places the detachable prefixes on that numbering exactly as it does
    # for a simple spiro ('8-chlorospiro[4.5]decane'). The numbering is the spiro-
    # nomenclature-fixed one (not re-optimised for lowest substituent locants), so
    # a decorated spiro-VB is RT-valid but not guaranteed lowest-locant PIN; the
    # round-trip gate backstops any misplacement.
    atoms_a = comp_a | {spiro_center}
    atoms_b = comp_b | {spiro_center}
    ext_a = _extract_subfragment(mol, atoms_a)
    ext_b = _extract_subfragment(mol, atoms_b)
    if ext_a is None or ext_b is None:
        return None
    from .bicyclo import is_bicyclo_system
    # At least one component must be a von Baeyer cage OR a registered carbo-PAH
    # (fluorene) OR the spiro atom must be a λ (nonstandard-valence) heteroatom
    # joining two DIFFERENT ring components, e.g. the λ5-P spiro of
    # [1,3,2]benzoxazaphosphole + [1,3,5,2]triazaphosphinine); otherwise this is
    # plain spiro / spirobi / mixed-spiro-fused — the existing branches own those
    # and are checked FIRST in the composer dispatch, so this path only sees what
    # they declined. Keeping a positive gate (rather than relying solely on
    # dispatch order) bounds the blast radius.
    a_vb = is_bicyclo_system(ext_a[0])
    b_vb = is_bicyclo_system(ext_b[0])
    lambda_spiro = _nonstandard_bonding_number(mol, spiro_center) is not None and \
        mol.GetAtomWithIdx(spiro_center).GetAtomicNum() not in (1, 6)
    # extension: a TRIcyclic+ von-Baeyer cage side also admits this path
    # (is_bicyclo is False for it), else the gate rejects a genuine monospiro.
    a_poly = (not a_vb) and _tricyclo_plus_spiro_component(mol, atoms_a) is not None
    b_poly = (not b_vb) and _tricyclo_plus_spiro_component(mol, atoms_b) is not None
    if not (
        a_vb or b_vb or a_poly or b_poly
        or _name_carbopah_spiro_component(mol, atoms_a) is not None
        or _name_carbopah_spiro_component(mol, atoms_b) is not None
        or lambda_spiro
    ):
        return None
    named_a = _name_spiro_component(mol, atoms_a, spiro_center)
    named_b = _name_spiro_component(mol, atoms_b, spiro_center)
    if named_a is None or named_b is None:
        return None
    name_a, locmap_a = named_a
    name_b, locmap_b = named_b
    # lowest spiro locant per component, deterministic over the spiro
    # atom's symmetry orbit (CanonicalRankAtoms) — the spirobi tie-break reused.
    loc_a = _canonical_spiro_locant(ext_a, locmap_a, spiro_center)
    loc_b = _canonical_spiro_locant(ext_b, locmap_b, spiro_center)
    if loc_a is None or loc_b is None:
        return None

    # spiro-hoist: re-anchor each component's substituent-hoisting locmap so
    # ``locmap[spiro] == descriptor locant``. When the fused-ring catalog numbered
    # the spiro atom at a locant OTHER than the canonical (lowest-orbit) one used
    # in the ``spiro[...]`` descriptor, the hoisted substituent citations were
    # inconsistent with it (fluorescein's carbonyl ``1-oxo`` colliding with
    # ``spiro[...-1,...]``), making the whole name OPSIN-unparseable.
    #
    # SCOPE: only components whose numbering has genuine symmetry freedom -- the
    # fused-ring CATALOG / retained / fused-PAH components (a symmetric parent like
    # ``1,3-dihydro-2-benzofuran`` may legitimately be numbered 1<->3). A von
    # Baeyer cage (``bicyclo[..]``/``tricyclo[..]``) has a RULE-FIXED numbering
    # that is NOT free to permute by graph automorphism, so re-anchoring one would
    # emit a different (wrong-numbered) cage name -- these are skipped. It is a
    # no-op whenever descriptor and map already agree (every clean canary), and
    # every emission remains RT-gated (0-wrong) regardless.
    if 'cyclo[' not in name_a:
        locmap_a = _reanchor_locmap_to_canonical_spiro(
            mol, ext_a, locmap_a, spiro_center, loc_a)
    if 'cyclo[' not in name_b:
        locmap_b = _reanchor_locmap_to_canonical_spiro(
            mol, ext_b, locmap_b, spiro_center, loc_b)

    #: ring unsaturation of a von-Baeyer CAGE component is cited INSIDE
    # that component's bracketed name (re-anchored to its own numbering), NOT
    # appended after the spiro brackets. Compute the per-side double-bond locants
    # here (each in its side's numbering) and splice them in as the component
    # names are assembled below. Fail closed if a cage bond is aromatic / triple /
    # needs a compound locant (never a silently-saturated name).
    _cage_a = atoms_a if a_vb else set()
    _cage_b = atoms_b if b_vb else set()
    _ene_sides = _cage_side_ene(mol, _cage_a, _cage_b, locmap_a, locmap_b)
    if _ene_sides is None:
        return None
    ene_a, ene_b = _ene_sides
    _spirobi_trailing = ''
    _pin_front = None
    _non_pin_form = False

    # (the Blue Book): "When ring components of 'spirobi'
    # compounds are named by von Baeyer nomenclature, heteroatoms are indicated by
    # skeletal replacement ('a') nomenclature. The spirobi ring system is named as
    # the saturated bi- or polycyclic alicyclic hydrocarbon". Two von Baeyer cages
    # are therefore identical components when their CARBON skeletons are, whatever
    # their heteroatoms ('6-sila-2,2'-spirobi[bicyclo[2.2.1]heptane] (PIN)',
    #:10201) and double bonds ('2,2'-spirobi[bicyclo[2.2.1]heptan]-5-ene (PIN)',
    #:16771); any other component must match atom for atom, as before.
    _both_cages = a_vb and b_vb
    identical = name_a == name_b and (
        _ring_skeleton_smiles(ext_a[0], carbon_skeleton=True)
        == _ring_skeleton_smiles(ext_b[0], carbon_skeleton=True)
        if _both_cages
        else Chem.MolToSmiles(ext_a[0]) == Chem.MolToSmiles(ext_b[0]))
    if identical:
        # spirobi multiplicative form for two identical components.
        # (:10168): "the lower number at the spiro atom is unprimed".
        # With equal spiro locants, (:10186): "low locants are given to
        # the spiro atom, then to the heteroatoms", and (:16761):
        # "low locants are assigned, in order, to spiro junction(s), heteroatoms
        # and double bonds".
        if loc_a == loc_b and _both_cages:
            a_first = _spirobi_cage_order_key(
                mol, atoms_a, atoms_b, spiro_center, locmap_a, locmap_b,
                ene_a, ene_b) <= _spirobi_cage_order_key(
                mol, atoms_b, atoms_a, spiro_center, locmap_b, locmap_a,
                ene_b, ene_a)
        else:
            a_first = loc_a <= loc_b
        if a_first:
            lo, hi = loc_a, loc_b
            unprimed_map, primed_map = locmap_a, locmap_b
            ene_unp, ene_pri = ene_a, ene_b
        else:
            lo, hi = loc_b, loc_a
            unprimed_map, primed_map = locmap_b, locmap_a
            ene_unp, ene_pri = ene_b, ene_a
        component = _bracket_component_locants(
            _strip_consumed_indicated_h(name_a, lo))
        # spirobi: unsaturation of the two IDENTICAL cages is cited as a
        # multiplicative suffix AFTER the closing bracket (`...nonane]-6,6'-diene`,
        # a Blue-Book PIN form) -- NOT spliced inside (that is the component-name
        # rule, handled in the else branch below). ``_spirobi_trailing``
        # is appended at the end (after any 'a'-prefix). (:16761):
        # "The final letter 'e' of the saturated hydrocarbon name is elided if
        # followed by a vowel" ('...heptan]-5-ene', '...nonane]-6,6'-diene'), and
        # the locants follow (:3193), primed after the same unprimed
        # number ('...nonane]-6',7-diene',:16779).
        if ene_unp or ene_pri:
            _tok = sorted(
                [(_spirobi_locant_key(n, 0), str(n)) for n in ene_unp]
                + [(_spirobi_locant_key(n, 1), f"{n}'") for n in ene_pri])
            from ..assembly.naming_utils import get_suffix_multiplier_prefix
            _mult = get_suffix_multiplier_prefix(len(_tok), 'ene')
            _spirobi_trailing = (
                f"-{','.join(t for _k, t in _tok)}-{_mult}ene")
            if not _mult and component.endswith('e'):
                component = component[:-1]
        name = f"{lo},{hi}'-spirobi[{component}]"
    else:
        # component-name spiro: cite components in ALPHANUMERICAL order
        # of the component name (NOT ring seniority); first cited is unprimed.
        if _component_alpha_key(name_a) <= _component_alpha_key(name_b):
            first_name, first_loc = name_a, loc_a
            second_name, second_loc = name_b, loc_b
            unprimed_map, primed_map = locmap_a, locmap_b
            first_ene, second_ene = ene_a, ene_b
        else:
            first_name, first_loc = name_b, loc_b
            second_name, second_loc = name_a, loc_a
            unprimed_map, primed_map = locmap_b, locmap_a
            first_ene, second_ene = ene_b, ene_a
        if lambda_spiro:
            #: the λ spiro atom + any indicated-H are cited at the
            # FRONT of the name (indicated-H set, then λ set); the descriptor
            # cites the components with their leading indicated-H stripped.
            front_prefix, _stripped = _build_lambda_ih_front_prefix(mol, [
                {'id': 'first', 'name': first_name, 'prime': '',
                 'spiro_atoms': [spiro_center], 'map': unprimed_map},
                {'id': 'second', 'name': second_name, 'prime': "'",
                 'spiro_atoms': [spiro_center], 'map': primed_map},
            ])
            _, first_bare = _extract_leading_indicated_h(first_name)
            _, second_bare = _extract_leading_indicated_h(second_name)
            first_bare = _splice_cage_ene(first_bare, first_ene, '')
            second_bare = _splice_cage_ene(second_bare, second_ene, "'")
            if first_bare is None or second_bare is None:
                return None
            name = (f"{front_prefix}spiro[{first_bare}-{first_loc},"
                    f"{second_loc}'-{second_bare}]")
        else:
            # (the Blue Book): each component by its own preferred
            # (mancude) name, in alphanumerical order, with the indicated
            # hydrogen and hydro prefixes of the complete structure cited in
            # front ('2'H,5H-spiro[thieno[2,3-b]furan-4,3'-thieno[3,2-b]furan]
            # (PIN)',:10310).
            st_a = _monospiro_component_state(
                mol, atoms_a, spiro_center, name_a, locmap_a, is_cage=a_vb)
            st_b = _monospiro_component_state(
                mol, atoms_b, spiro_center, name_b, locmap_b, is_cage=b_vb)
            asm = (_assemble_hoisted_monospiro(spiro_center, st_a, st_b)
                   if st_a is not None and st_b is not None else None)
            if asm is not None:
                _pin_front, name, _comb, _first, _second = asm
                unprimed_map, primed_map = _first['map'], _second['map']
            else:
                _non_pin_form = True
                first_name = _strip_consumed_indicated_h(first_name, first_loc)
                second_name = _strip_consumed_indicated_h(second_name, second_loc)
                first_name = _splice_cage_ene(first_name, first_ene, '')
                second_name = _splice_cage_ene(second_name, second_ene, "'")
                if first_name is None or second_name is None:
                    return None
                name = f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"

    combined_locants: Dict[int, _Locant] = {}
    for atom_idx, locant in unprimed_map.items():
        combined_locants[atom_idx] = locant
    for atom_idx, locant in primed_map.items():
        if atom_idx == spiro_center:
            continue
        combined_locants[atom_idx] = (locant, "'")
    if not (set(combined_locants.keys()) >= all_ring_atoms):
        return None  # coverage invariant (Pitfall 7)

    #: skeletal heteroatoms of a von-Baeyer CAGE component are cited as
    # an 'a'-replacement prefix BEFORE 'spiro'. Heteroatoms belonging to a fused
    # / catalog component (e.g. the O of xanthene) are already implicit in that
    # component's retained name and must NOT be double-cited. Only the atoms of
    # bicyclo-cage-named components are 'a'-expressible here.
    a_expressible: Set[int] = set()
    if name_a.startswith('bicyclo'):
        a_expressible |= atoms_a
    if name_b.startswith('bicyclo'):
        a_expressible |= atoms_b

    #: a large saturated heteromonocycle component named as its carbon
    # parent (`cyclododecane`, from _name_skeletal_replacement_monocycle_component)
    # has its skeletal heteroatoms cited by the SAME front 'a'-prefix as a cage's.
    # Detect it as a `cyclo...`-named component whose ring carries heteroatoms
    # (a genuine carbocyclic `cyclohexane` component carries none, so it is never
    # hoisted). HW-named heteromonocycles start with a bracket or a heteroatom stem
    # (`[1,3]dithiane`, `thiane`, `oxolane`) — never `cyclo` — so they are excluded.
    def _is_skelrepl_monocycle(cname: str, catoms: Set[int]) -> bool:
        return cname.startswith('cyclo') and any(
            mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6) for a in catoms
        )
    if _is_skelrepl_monocycle(name_a, atoms_a):
        a_expressible |= atoms_a
    if _is_skelrepl_monocycle(name_b, atoms_b):
        a_expressible |= atoms_b
    hetero_prefix = _spiro_vb_a_prefix(
        mol, a_expressible, spiro_center, unprimed_map, primed_map
    )
    if hetero_prefix is None:
        # A heteroatom in a VB cage that we could not deterministically cite ->
        # fail closed (never a silently-carbo name).
        if any(
            mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
            for a in a_expressible
        ):
            return None
    elif hetero_prefix:
        # /: the replacement 'a'-prefix joins the descriptor
        # DIRECTLY when it starts with a letter ('3-thia' + 'spiro[4.5]...' ->
        # '3-thiaspiro[4.5]...'), but a hyphen separates it from a LOCANT-initial
        # descriptor -- a spirobi cites its spiro locants first ('3,3'-spirobi'),
        # so '6,6'-dioxa' + '3,3'-spirobi[...]' -> '6,6'-dioxa-3,3'-spirobi[...]',
        # NOT '...dioxa3,3'-spirobi...'.
        from ..assembly.composition_primitives import _join_prefix_to_name
        name = _join_prefix_to_name(hetero_prefix, name)

    # Cage unsaturation of a COMPONENT-NAME spiro was spliced INSIDE
    # each component's bracketed name above (re-anchored to that component's
    # numbering) as the name was assembled -- see ``_cage_side_ene`` /
    # ``_splice_cage_ene`` -- because the old trailing '-3-ene' form is
    # OPSIN-grammar-invalid there (-C1C2C6 Pattern A1). A SPIROBI (identical
    # cages, instead cites its multiplicative '-6,6'-diene' suffix AFTER
    # the bracket (a Blue-Book PIN form); that suffix is appended here.
    if _spirobi_trailing:
        name = name + _spirobi_trailing
    if _pin_front:
        name = _attach_hoisted_front(_pin_front, name)
    if _non_pin_form:
        # The preferred form could not be built: this name keeps a component's
        # own hydro prefixes / indicated hydrogen inside the brackets or a
        # component double bond spliced into it, so it is not the PIN,
        # the Blue Book;,:16761).
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return name, all_ring_atoms, combined_locants


def _spirobi_cage_order_key(mol, first_atoms: Set[int], second_atoms: Set[int],
                            spiro_center: int, first_map, second_map,
                            first_ene, second_ene):
    """Locant key of a 'spirobi' von Baeyer system when ``first_atoms`` is the
    unprimed cage: the heteroatom locants as one set, then per heteroatom in the
    order of (O > S > Se > Te > N >...), then the double-bond
    locants, each compared in the order of (5 < 5' < 6 < 6').
    '5,6'-dioxa-2,2'-spirobi[bicyclo[2.2.2]octane]-7,7'-diene (PIN)'
    (the Blue Book), '6-oxa-6'-thia-2,2'-spirobi[bicyclo[2.2.1]heptane]
    (PIN)' (:10203)."""
    hetero: Dict[str, List[Tuple]] = {}
    all_keys: List[Tuple] = []
    for atoms, amap, primes in ((first_atoms, first_map, 0),
                                (second_atoms, second_map, 1)):
        for a in atoms:
            if a == spiro_center:
                continue
            sym = mol.GetAtomWithIdx(a).GetSymbol()
            if sym in ('C', 'H') or a not in amap:
                continue
            key = _spirobi_locant_key(amap[a], primes)
            all_keys.append(key)
            hetero.setdefault(sym, []).append(key)
    per_element = tuple(
        tuple(sorted(hetero[el]))
        for el in sort_heteroatoms_by_priority(list(hetero)))
    enes = tuple(sorted([_spirobi_locant_key(n, 0) for n in first_ene]
                        + [_spirobi_locant_key(n, 1) for n in second_ene]))
    return (tuple(sorted(all_keys)), per_element, enes)


def _spiro_vb_a_prefix(
    mol,
    a_expressible: Set[int],
    spiro_center: int,
    unprimed_map: Dict[int, int],
    primed_map: Dict[int, int],
) -> Optional[str]:
    """ 'a'-replacement prefix (e.g. ``3-thia``) for skeletal
    heteroatoms of a von-Baeyer CAGE component, cited before 'spiro'. Returns the
    prefix string ('' when there are no heteroatoms), or None (fail-closed) if a
    heteroatom is the spiro atom, has a nonstandard bonding number (λ family),
    is off the Table-1.5 replacement list, or cannot be located deterministically.

    This is the THIRD spiro-context speller in the tree (with
    ``_build_hetero_prefix`` and the shared ``build_replacement_prefix``) and it
    carried the same ``get_hw_prefix(...) or get_heteroatom_prefix(...)`` line,
    hence the same ``symbol.lower + 'a'`` fabrication. It is gated by element
    membership rather than by the shared primitive because its locants come from
    a primed/unprimed component pair, not from one flat ``numbering`` map, so the
    primitive's signature does not apply here -- the closed-table membership test
    is the part that must not diverge.
    """
    from .ring_replacement import HETEROATOM_PREFIXES

    hetero = [
        a for a in a_expressible
        if mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
    ]
    if not hetero:
        return ''
    from ..assembly.naming_utils import get_multiplier_prefix
    # element -> list of (locant_token, sort_key). Prime the token if the atom
    # is in the primed component.
    by_element: Dict[str, List[Tuple[str, Tuple]]] = {}
    for a in hetero:
        if a == spiro_center:
            return None  # λ-spiro family, not
        if _nonstandard_bonding_number(mol, a) is not None:
            return None  # λ heteroatom -> Tasks 8-11
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym not in HETEROATOM_PREFIXES:
            return None  # off Table 1.5: no morpheme exists -> refuse
        # (the Blue Book): "Primed locants are placed immediately
        # after the corresponding unprimed locants in a set arranged in
        # ascending order" ('5,6'-dioxa':16775, '2'',7-dioxa':16787).
        if a in unprimed_map:
            loc = unprimed_map[a]
            token = str(loc)
            key = _spirobi_locant_key(loc, 0)
        elif a in primed_map:
            loc = primed_map[a]
            token = f"{loc}'"
            key = _spirobi_locant_key(loc, 1)
        else:
            return None
        by_element.setdefault(sym, []).append((token, key))
    parts = []
    for element in sort_heteroatoms_by_priority(list(by_element.keys())):
        entries = sorted(by_element[element], key=lambda t: t[1])
        # Table 1.5 context, as in ``_build_hetero_prefix``: do not
        # consult the Hantzsch-Widman table from here.
        prefix_name = get_heteroatom_prefix(element)
        if prefix_name is None:
            return None  # unreachable past the membership gate; never fabricate
        mult = get_multiplier_prefix(len(entries), prefix_name)
        locs = ','.join(t[0] for t in entries)
        parts.append(f"{locs}-{mult}{prefix_name}")
    # Join the per-element 'a'-prefix terms with a hyphen -- the PIN separator
    # between replacement-prefix locant-terms /, e.g.
    # ``2-oxa-6-aza``. Sibling spellers ``_build_hetero_prefix`` and
    # ``build_replacement_prefix`` join the same way; this one had diverged to
    # ``''.join`` and emitted ``2-oxa6-aza``. The last term connects directly to
    # the following ``spiro`` descriptor (no trailing hyphen), so a plain
    # ``'-'.join`` is exactly right.
    return '-'.join(parts)


def is_spiro_vonbaeyer(mol) -> bool:
    """: monospiro system with >=1 von Baeyer (bridged) ring component
    (e.g. ``spiro[bicyclo[2.2.1]heptane-2,1'-cyclohexane]``,
    ``2,2'-spirobi[bicyclo[2.2.1]heptane]``). Fail-closed (see
    ``_name_spiro_vonbaeyer_core``)."""
    if mol is None:
        return False
    return _name_spiro_vonbaeyer_core(mol) is not None


def name_spiro_vonbaeyer(
    mol,
) -> Optional[Tuple[str, Set[int], Dict[int, _Locant], bool]]:
    """Build the component-name spiro PIN for a monospiro system with at
    least one von Baeyer cage component. Return shape matches ``name_spirobi``
    (name, ring_atoms, atom_to_locant, substituents_included=False)."""
    core = _name_spiro_vonbaeyer_core(mol)
    if core is None:
        return None
    name, all_ring_atoms, combined_locants = core
    return (name, all_ring_atoms, combined_locants, False)


def get_mixed_spiro_fused_iupac_locants(
    mol,
) -> Optional[Dict[int, _Locant]]:
    """
    Cascade-step-6 supplier for mixed spiro/fused systems (a phase-02).

    Wraps `name_mixed_spiro_fused` and returns the combined atom-to-locant
    map (covering ALL ring atoms) or None if naming declined.

    Coverage invariant per Pitfall 7: returns None on partial coverage
    so the cascade-step-6 gate falls through to the sorted-int proxy.
    """
    if mol is None:
        return None
    result = name_mixed_spiro_fused(mol)
    if result is None:
        return None
    _name, ring_atoms, atom_to_locant, _subs_included = result
    if not (set(atom_to_locant.keys()) >= ring_atoms):
        return None
    return {k: v for k, v in atom_to_locant.items() if k in ring_atoms}
