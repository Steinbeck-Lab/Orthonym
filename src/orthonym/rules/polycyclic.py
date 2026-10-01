"""
Von Baeyer polycyclic descriptor generation (IUPAC 2013,.

Implements the von Baeyer algorithm (through) for generating
correct polycyclic descriptors for any ring count (bicyclo through decacyclo).

This module replaces the broken descriptor generation in tricyclo.py and
polycyclic_bridged.py, which incorrectly used SSSR for main ring finding.

Key algorithm:
1. Ring count via cycle rank formula (edges - vertices + 1)
2. Main ring via longest-path between bridgehead pairs (NOT SSSR)
3. Main bridge between main bridgeheads through non-main-ring atoms
4. Secondary bridges: independent before dependent, descending by length
5. numbering: main ring (longer path first) -> main bridge -> secondary bridges
6. Verification: sum(bridge_lengths) + 2 == total_ring_atoms

Reference: IUPAC 2013 Blue Book, through.
"""

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..assembly.fragment_naming import _fragment_guard  # Lever N: budget replay on a memo hit
from ..assembly.memo import cache_or_compute  # R2(a): structure-keyed main-ring memo
from ..perception.molcache import atoms_of, bonds_of, cached_by_key
from .locants import compound_aware_multibond_set  # (1)/(2), shared w/ bicyclo.py
from ..assembly.fragment_naming import (  # M2.5 macrocycle-hang budgets
    spend_analysis_call,
    spend_perf_work,
)

logger = logging.getLogger(__name__)

# R2(a): integer extractor for RDKit's ``_smilesAtomOutputOrder`` prop (e.g. "[0,1,2,...]"),
# used to build the atom-order component of the von Baeyer main-ring structure key.
_ORDER_INT_RE = re.compile(r"\d+")


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


def fusion_nomenclature_applies(mol, ring_atoms) -> bool:
    """True when the ring system ``ring_atoms`` holds an ortho-fused cluster with at
    least two rings of five or more members, i.e. when a fusion (or bridged fusion)
    name exists and a von Baeyer name of the system is therefore NOT its PIN.

     "Five-membered ring requirement" (the Blue Book,:23710): "Fusion
    nomenclature gives preferred IUPAC names only to compounds having at least two
    rings of at least five or more members. [...] When fusion names are not allowed,
    unsaturated von Baeyer ring system names are preferred IUPAC names."
    (:19532) ranks (c) fused and (d) bridged fused ring systems above (e) nonfused
    bridged (von Baeyer) systems, and:23883 "the bridged fused ring name is
    preferred to the von Baeyer name". BB rows: 'decahydronaphthalene (PIN)' over
    'bicyclo[4.4.0]decane' (:24233); 'hexadecahydro-1H-8,12-methanobenzo[13]annulene
    (PIN)' over 'tricyclo[12.3.1.0^5,10]octadecane' (:23875-23879); 'hexahydro-1H-
    4,7-methanoindene (PIN)' (:49311, the tricyclo[5.2.1.0^2,6] skeleton);
    'tetrahydro-4,8-ethanopyrano[4,3-c]pyran-...-tetrone (PIN) {not 4,9-dioxatricyclo
    [4.4.2.0^2,7]dodecane-...}' (:32535). Boundary rows where the von Baeyer name IS
    the PIN (this returns False for each): 'bicyclo[4.1.0]hepta-1,3,5-triene',
    'bicyclo[4.2.0]octa-1,3,5,7-tetraene' (:23718,:23725; one ring of five or more),
    'tetracyclo[3.2.0.0^2,7.0^4,6]heptane' (:10824: its two five-membered rings meet
    a four-membered ring each by one bond but share three atoms with each other, so
    they are bridged, not fused), 'cubane (PIN) pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]
    octane' (:9889), 'tetracyclo[2.2.0.0^2,6.0^3,5]hexane (PIN)' (:9897),
    '3,6,8-trioxatricyclo[3.2.1.0^2,4]octane (PIN)' (:48763).

    A cluster is grown along rings that share exactly one bond (ortho-fusion); every
    ring in it must meet every other ring in at most one atom or exactly one bond, as
    in a fused ring system. Smallest rings come from RDKit's symmetrized SSSR. A miss
    only leaves a label as it was; it never changes a name.

    A cluster that holds every ring atom while the system has more rings than the
    cluster closes those rings by bonds only. (:14025): "An atom or group of
    atoms is named as a bridge", so such a cluster gives no bridged fused name (a decalin
    with a cyclobutane closed across it, tricyclo[4.4.0.0^5,10]decane); the search goes on
    for a cluster that leaves a bridge atom or is the whole system."""
    ring_atoms = frozenset(ring_atoms)

    def _fresh():
        rings = []
        for r in Chem.GetSymmSSSR(mol):
            fr = frozenset(r)
            if fr <= ring_atoms and fr not in rings:
                rings.append(fr)
        n = len(rings)
        # the number of rings of the system (bonds - atoms + components)
        n_bonds = sum(1 for b in mol.GetBonds()
                      if b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms)
        n_comp, seen = 0, set()
        for start in ring_atoms:
            if start in seen:
                continue
            n_comp += 1
            todo = [start]
            seen.add(start)
            while todo:
                cur = todo.pop()
                for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                    k = nb.GetIdx()
                    if k in ring_atoms and k not in seen:
                        seen.add(k)
                        todo.append(k)
        cycle_rank = n_bonds - len(ring_atoms) + n_comp

        def _leaves_a_bridge_atom_or_is_whole(cluster):
            covered = frozenset().union(*(rings[k] for k in cluster))
            return covered != ring_atoms or len(cluster) >= cycle_rank

        def _one_bond(i, j):
            s = rings[i] & rings[j]
            if len(s) != 2:
                return False
            a, b = tuple(s)
            return mol.GetBondBetweenAtoms(a, b) is not None

        def _compatible(i, j):
            s = rings[i] & rings[j]
            return len(s) <= 1 or _one_bond(i, j)

        for start in range(n):
            if len(rings[start]) < 5:
                continue
            stack = [(start, (start,))]
            while stack:
                cur, path = stack.pop()
                for j in range(n):
                    if j in path or not _one_bond(cur, j):
                        continue
                    if not all(_compatible(j, k) for k in path):
                        continue
                    if len(rings[j]) >= 5 and _leaves_a_bridge_atom_or_is_whole(path + (j,)):
                        return True
                    if len(path) < 8:
                        stack.append((j, path + (j,)))
        return False

    return cached_by_key(mol, "vb_fusion_applies", ring_atoms, _fresh)


#: "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES" (the Blue Book,:9881):
#: "The retained names adamantane and cubane are used in general nomenclature and as
#: preferred IUPAC names." Table 2.6: 'adamantane (PIN) tricyclo[3.3.1.1^3,7]decane'
#: (:9885), 'cubane (PIN) pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane' (:9889). Keys: the
#: RDKit canonical SMILES of the all-carbon, all-single-bond skeleton.
_RETAINED_VON_BAEYER_SKELETONS = {
    "C1C2CC3CC1CC(C2)C3": "adamantane",
    "C12C3C4C1C1C2C3C41": "cubane",
}


def retained_von_baeyer_parent(mol, ring_atoms) -> Optional[str]:
    """'adamantane' / 'cubane' when the all-carbon ring system ``ring_atoms`` is that
    retained parent hydride, else None. A ring heteroatom returns None."""
    ring_atoms = frozenset(ring_atoms)

    def _fresh():
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring_atoms):
            return None
        if len(ring_atoms) not in (8, 10):
            return None
        rw = Chem.RWMol()
        index = {i: rw.AddAtom(Chem.Atom(6)) for i in sorted(ring_atoms)}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in index and b in index:
                rw.AddBond(index[a], index[b], Chem.BondType.SINGLE)
        skeleton = rw.GetMol()
        Chem.SanitizeMol(skeleton)
        return _RETAINED_VON_BAEYER_SKELETONS.get(Chem.MolToSmiles(skeleton))

    return cached_by_key(mol, "vb_retained_parent", ring_atoms, _fresh)


def _record_if_fusion_nameable(mol, ring_atoms, result) -> None:
    """Mark a legal von Baeyer descriptor as a non-PIN fragment when the ring system has
    a preferred name of another kind: a fusion name (``fusion_nomenclature_applies``) or
    a retained name (``retained_von_baeyer_parent``: adamantane, cubane). Name-scoped:
    only a shipped name that CONTAINS this descriptor is demoted from pin_verified, and a
    von Baeyer descriptor fixes its skeleton, so every name carrying it names such a
    system. The name itself is unchanged."""
    if result is None or result.legality is not True:
        return
    descriptor = getattr(result, "descriptor_string", None)
    if not descriptor:
        return
    try:
        if (fusion_nomenclature_applies(mol, ring_atoms)
                or retained_von_baeyer_parent(mol, ring_atoms)):
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(descriptor)
    except Exception:  # a label helper must never break naming
        logger.debug("fusion_nomenclature_applies failed", exc_info=True)


def von_baeyer_ring_count(mol, cage_atoms) -> Optional[int]:
    """The number of rings counts over ``cage_atoms``, or ``None``.

     (``the Blue Book``) DEFINES the quantity: "A 'polycyclic
    system' contains a number of rings equal to the **minimum number of
    scissions required to convert the system into an acyclic skeleton**." Restated
    at (``:9645``): "The number of rings is equal to the number of
    bond cuts necessary to transform the polycyclic system into an acyclic
    skeleton." That is the graph's circuit rank, ``E - V + C`` over the induced
    cage subgraph (``C`` = connected components), and it is exactly the number
    ``cyclo_ring_count_word`` spells -- the two are the count and its word, so
    they live together here.

    Why this is NOT ``GetRingInfo.NumRings`` (a phase T3b)
    -------------------------------------------------------------
    RDKit's ring info is the **symmetrized** SSSR, which deliberately keeps
    extra symmetry-equivalent smallest rings, so its cardinality OVER-COUNTS the
     number on exactly the symmetric cages von Baeyer nomenclature is
    for:

    * adamantane -- Blue Book ``tricyclo[3.3.1.1^3,7]decane`` (PIN, ``:9840``),
      3 rings -- symmetrized ring count **4**;
    * cubane -- ``pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane`` (PIN, ``:9889``),
      5 rings -- symmetrized ring count **6**;
    * a 12-atom bridged cage whose reference name is ``heptacyclo[...]dodecane``,
      7 rings -- symmetrized ring count **11**.

    ``MAX_CAGE_RINGS`` was compared against the symmetrized count, so a cap
    meant to bound "8 rings" refused *heptacyclo* cages.

    Two other places in the tree already compute this quantity correctly but
    privately -- ``VonBaeyerAnalyzer._get_ring_count`` (as ``E - V + 1``, so it
    under-counts a DISCONNECTED atom set) and ``rules/bicyclo.py:163``, whose
    comment had already diagnosed the hazard in prose ("cycle_rank... is always
    reliable regardless of SSSR issues"). Neither was the bug, so neither is
    rerouted here: ``_get_ring_count`` has 7 PIN-path call sites and swapping its
    disconnected-set behaviour is a separate, separately-gated change. New
    callers should use THIS function.

    Returns ``None`` when ``cage_atoms`` is empty, so callers fail closed rather
    than treat "no cage" as a ring count.
    """
    cage = set(cage_atoms)
    if not cage:
        return None
    edges = 0
    adj = {i: set() for i in cage}
    for bond in bonds_of(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in cage and b in cage:
            edges += 1
            adj[a].add(b)
            adj[b].add(a)
    # connected components of the induced subgraph (C in E - V + C)
    seen: set = set()
    components = 0
    for start in sorted(cage):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            for nbr in adj[cur]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
    return edges - len(cage) + components


def cyclo_ring_count_word(ring_count: int) -> Optional[str]:
    """The von Baeyer ring-count term for ``ring_count`` rings, or ``None``.

     (``the Blue Book``): "The number of rings is indicated by the
    nondetachable prefix 'bicyclo' (not dicyclo), 'tricyclo', 'tetracyclo',
    etc." -- restated at (``:9645``). Both sentences end in "etc.":
    the series is OPEN-ENDED and the Blue Book prints **no table** of these
    words (the highest one attested anywhere in the text is ``hexacyclo``,
    ``:9731``). The term is therefore COMPUTED -- simple multiplying prefix
    (Table 1.4, + ``cyclo`` -- with the single irregularity that 2 is
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
        return "bicyclo"  #: 'bicyclo', explicitly NOT 'dicyclo'
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
    is_dependent: bool = False  # True for dependent secondary bridges
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
    #: Verdict of ``VonBaeyerAnalyzer._legality_verified`` -- True only when
    #: ``descriptor_string``, read by the Blue Book's own numbering rules,
    #: rebuilds exactly this cage under ``numbering``. ``None`` means NOT
    #: ADJUDICATED (a raw ``_analyze_impl`` candidate); ``analyze`` sets it on
    #: every return path. **Anything that spells a name from this object must
    #: require ``legality is True``** -- a descriptor that fails is not a worse
    #: name, it is a name for a different molecule.
    legality: Optional[bool] = None


# ============================================================================
# Core Algorithm: find_longest_path
# ============================================================================

# (code review 2026-06-02): hard cap on DFS node-expansions. Simple-path
# enumeration over a dense ring graph is combinatorial in the cycle count, NOT
# linear in atom count, so the "<50 atoms => fast" assumption fails on a
# highly-bridged cage (fullerene fragment, dense cage input) — exactly the kind
# of system the relaxed bridgehead predicate now admits into the von
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
    ``_MAX_DFS_EXPANSIONS``  so a pathological dense cage cannot hang;
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
        spend_perf_work()  # M2.5: charge the per-molecule op budget (raises to abstain)
        if expansions > _MAX_DFS_EXPANSIONS:
            return  # cap: abort exploration, keep best-so-far
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


def _find_all_simple_paths(mol, start: int, end: int, allowed_atoms: Set[int],
                           adj: Optional[Dict[int, List[int]]] = None) -> List[List[int]]:
    """Memoising front of:func:`_find_all_simple_paths_impl` (a performance pass, a lever).

    Within one naming scope the same ``(mol object, start, end, allowed)`` enumeration is
    computed once; 90 % of the 24,450 enumerations on 300 a holdout split molecules were repeats.
    The entry records the perf-budget units the DFS charged and a hit charges the SAME
    units again, so the budget trajectory (and any ``PerfBudgetExceeded`` abstention) is
    unchanged. Callers receive fresh list copies (they filter and slice the result).

    ``cached_by_key`` returns ``_fresh`` unchanged outside a scope or for an ``RWMol``,
    so behaviour without a scope is identical; a ``PerfBudgetExceeded`` raised inside the
    DFS propagates before anything is stored. Mirrors the ``_find_main_ring`` memo front,
    reusing the module-level ``cached_by_key`` / ``_fragment_guard`` / ``spend_perf_work``.
    """
    state = {"hit": True}

    def _fresh():
        state["hit"] = False
        before = getattr(_fragment_guard, "perf_budget", None)
        paths = _find_all_simple_paths_impl(mol, start, end, allowed_atoms, adj)
        after = getattr(_fragment_guard, "perf_budget", None)
        units = (before - after) if (before is not None and after is not None and before > after) else 0
        return (tuple(tuple(p) for p in paths), units)

    val = cached_by_key(mol, "polycyclic.simple_paths", (start, end, frozenset(allowed_atoms)), _fresh)
    if state["hit"] and val[1]:
        spend_perf_work(val[1])
    return [list(p) for p in val[0]]


def _find_all_simple_paths_impl(mol, start: int, end: int, allowed_atoms: Set[int],
                                adj: Optional[Dict[int, List[int]]] = None) -> List[List[int]]:
    """
    Find all simple paths from start to end through allowed atoms.

    Returns list of paths, each a list of atom indices. The enumeration is
    bounded by ``_MAX_DFS_EXPANSIONS``  — on a pathological dense graph
    it returns the paths found before the cap rather than hanging.

    ``adj`` (a lever): an optional precomputed ``Dict[int, List[int]]`` of
    neighbour indices in ``GetNeighbors`` order, keyed by every atom this
    DFS may visit. When given, the DFS walks it instead of re-calling
    ``GetNeighbors`` live on every step -- order-identical, just cheaper.
    """
    if start == end:
        return [[start]]

    all_paths = []
    expansions = 0

    def dfs(current, visited, path):
        nonlocal expansions
        expansions += 1
        spend_perf_work()  # M2.5: charge the per-molecule op budget (raises to abstain)
        if expansions > _MAX_DFS_EXPANSIONS:
            return  # cap: abort enumeration, keep paths-so-far
        if current == end:
            all_paths.append(path[:])
            return

        if adj is not None:
            nbrs = adj[current]                      # GetNeighbors order, precomputed (a lever)
        else:
            nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()]
        for nbr_idx in nbrs:
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
    Implements IUPAC von Baeyer nomenclature, through).

    Pipeline:
    1. Ring count: cycle_rank = edges - vertices + 1
    2. Main ring: largest ring through a bridgehead pair
    3. Main bridge: longest path between main bridgeheads not through main ring
    4. Secondary bridges: remaining connections, independent before dependent
    5. Numbering: main ring -> main bridge -> secondary bridges
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
        """Well-formed iff bridge lengths are all non-negative, the atom-count
        invariant holds (sum + 2 == ring atoms), and every ring atom received a
        locant. Malformed descriptors are OPSIN-unparseable.

        **** "Naming polycyclic alicyclic hydrocarbons"
        (``the Blue Book Blue Book``), rule **** (``:9651``):
        *"The name is terminated by the name of the alkane representing the
        total number of ring atoms; this number corresponds to the sum of the
        arabic numbers in the numerical descriptor enclosed by brackets plus two
        (for the two main bridgehead atoms)."* This code cited for
        it until; that rule (``:9645``) is a different clause -- it
        fixes the ring-count WORD ('tricyclo', 'tetracyclo') -- and is enforced
        separately inside ``audit_von_baeyer_descriptor``.

        This is an ATOM COUNT and nothing more. A descriptor can satisfy it and
        still cite bridges the molecule does not have, so it is NOT sufficient to
        spell a name from: use ``_legality_verified``.
        """
        if result is None:
            return False
        if any(l < 0 for l in result.bridge_lengths):
            return False
        if sum(result.bridge_lengths) + 2 != result.total_atoms:
            return False
        return set(result.numbering.keys()) >= set(ring_atoms)

    @staticmethod
    def _legality_verified(mol, ring_atoms: Set[int],
                           result: "PolycyclicDescriptor") -> bool:
        """True iff the descriptor STRING rebuilds exactly this cage under this
        numbering -- the proof ``_descriptor_is_valid`` is not.

        ``audit_von_baeyer_descriptor``  reconstructs the skeleton
        the emitted string denotes from / / and
        requires SET EQUALITY with the molecule's own cage bonds in locant
        space. It proves LEGALITY (this name denotes this molecule under this
        numbering), never PREFERENCE.

        Composed with the arithmetic check rather than replacing it: the
        arithmetic clause is the cheap pre-filter and names the specific
         violation in the log. Composing adds no false rejection --
        measured 0 cages with ``audit=True, arith=False`` over 8,201 enumerated
        cages and over 1,547 accepted corpus cages .

        Deliberately Java-free: both downstream OPSIN gates are documented
        FAIL-OPEN when no jar is present, so with Java absent this is the only
        thing between a bridge-dropping descriptor and an emitted name.
        """
        if result is None or not result.numbering:
            return False
        cage = set(ring_atoms)
        if not VonBaeyerAnalyzer._descriptor_is_valid(result, cage):
            return False
        from .vonbaeyer_universal import audit_von_baeyer_descriptor
        try:
            return bool(audit_von_baeyer_descriptor(
                mol, cage, result.numbering, result.descriptor_string))
        except Exception:
            # A checker that cannot reach a verdict has not proved legality.
            logger.debug("legality audit raised for %s",
                         result.descriptor_string, exc_info=True)
            return False

    @staticmethod
    def _is_unsubstituted_ring_system(mol, ring_atoms: Set[int]) -> bool:
        """True iff no ring atom carries an exocyclic heavy-atom substituent.

        Only such 'pure cages' (prismane, cubane, nortricyclene,...) are
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

    def analyze(self, mol, ring_atoms: Set[int],
                spiro_atom: Optional[int] = None) -> PolycyclicDescriptor:
        """
        Main entry point: analyze a polycyclic system and produce its descriptor.

        ``spiro_atom`` (default None) is the atom index of the spiro junction
        when this cage is a COMPONENT of a spiro ring system; passing it makes
        the numbering selection give that atom the lowest locant,
        ABOVE the heteroatom criteria. Every whole-molecule caller leaves it None
        and the analysis is byte-identical. Only the substituted/heteroatom
        branch honours it -- a spiro component is always "substituted" (its spiro
        atom bonds into the other component), so it never takes the pure-cage
        renumber branch; if it somehow did, the criterion is a safe no-op there.

        Determinism : for an UNSUBSTITUTED cage the cascade is run on a
        copy renumbered into a total RDKit canonical-rank order (identical for
        every SMILES spelling), making the descriptor spelling-independent; the
        numbering is then mapped back to the caller's atom indices. If the
        canonical run yields a malformed descriptor the cascade falls back to
        the original atom order. SUBSTITUTED / heteroatom cages are NOT
        renumbered (that shifts substituent locants and breaks round-trips);
        they still tie-break on canonical ranks inside ``_find_main_ring``.

        **The fallback carries no quality guarantee.** This docstring used to
        claim the fallback "is never worse than the un-renumbered path" -- it IS
        the un-renumbered path, so the sentence was true only vacuously, and it
        read as a validation that did not exist: the fallback result was
        returned with no check at all, which is how descriptors whose brackets
        do not account for every skeletal atom reached the namer .

        What IS guaranteed: every return path is adjudicated by
        ``_legality_verified`` and the verdict published as
        ``PolycyclicDescriptor.legality``. ``analyze`` still returns its best
        analysis when that verdict is False -- the numbering remains useful to
        callers that only inspect the cage -- so **every caller that SPELLS A
        NAME must require ``legality is True``** and fail closed otherwise.

        Args:
            mol: RDKit Mol object
            ring_atoms: Set of atom indices in the ring system

        Returns:
            PolycyclicDescriptor with all VB information, ``legality`` set, or
            None when the cascade could not produce an analysis at all.
        """
        # M2.5: charge one ANALYSIS-CALL unit. The substituent recursion of a
        # symmetric cob(III)yrinate re-invokes von-Baeyer analyze THOUSANDS of
        # times (measured 553 in 16 s vs the nameable chlorophyll's 19); the
        # call budget (armed at the outermost name, shared with the fused
        # matcher) exhausts on that explosion and abstains the whole molecule.
        spend_analysis_call()
        result = self._analyze_once(mol, ring_atoms, spiro_atom)
        # / main-bridge PREFERENCE (Case 3 in _find_main_ring
        # + the largest-main-bridge selection in _find_main_bridge) is applied
        # first. It is preference-only: on a cage whose secondary bridges cannot
        # be numbered consistently under the preferred main bicycle the result
        # fails the legality audit. Rather than let a preference improvement cost
        # a legal (if non-preferred) name -- a regression -- degrade cleanly by
        # re-running the cascade with the preference disabled and returning that
        # legacy result whenever it is legal. Molecules whose preferred main
        # bicycle IS numbered consistently are unaffected (single pass).
        if (result is not None and result.legality is False
                and getattr(self, "_allow_multiatom_bridge", True)):
            self._allow_multiatom_bridge = False
            try:
                legacy = self._analyze_once(mol, ring_atoms, spiro_atom)
            finally:
                self._allow_multiatom_bridge = True
            if legacy is not None and legacy.legality:
                _record_if_fusion_nameable(mol, ring_atoms, legacy)
                return legacy
        _record_if_fusion_nameable(mol, ring_atoms, result)
        return result

    def _analyze_once(self, mol, ring_atoms: Set[int],
                      spiro_atom: Optional[int] = None) -> "PolycyclicDescriptor":
        """One pass of the von Baeyer cascade (both the pure-cage and the
        substituted/heteroatom branches). Factored out of:meth:`analyze` so the
         main-bridge preference can be retried-then-degraded without
        re-charging the analysis-call budget. See:meth:`analyze`."""
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
                    # /.5).
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
                    # Adjudicated against the ORIGINAL mol/ring_atoms, i.e. the
                    # indices the caller will use, not the canonical copy.
                    canon_result.legality = self._legality_verified(
                        mol, ring_atoms, canon_result)
                    return canon_result

        # Substituted / heteroatom cages: the canonical renumber above is unsafe
        # here (it shifts substituent locants), but the numbering freedom that
        # leaves open is still governed -- (:9777) and
        # (:3219). Rank the legal alternatives on locants instead of leaving the
        # arbitrary canonical-rank backstop to decide.
        incumbent = self._analyze_impl(mol, ring_atoms)
        if incumbent is None:
            return None
        if not incumbent.numbering:
            # No locants at all: nothing can be spelled from this, and the
            # audit cannot even be evaluated. Not adjudicable == not legal.
            incumbent.legality = False
            return incumbent
        bridgeheads = self._find_all_bridgeheads(mol, ring_atoms)
        chosen = self._choose_lowest_locant_numbering(
            mol, ring_atoms, bridgeheads, incumbent, spiro_atom
        )
        # This branch previously returned UNCHECKED -- the whole of site 2.
        chosen.legality = self._legality_verified(mol, ring_atoms, chosen)
        return chosen

    def _analyze_impl(self, mol, ring_atoms: Set[int],
                      optimize_orientation: bool = False,
                      forced: Optional[Tuple[List[int], Tuple[int, int]]] = None,
                      ) -> PolycyclicDescriptor:
        """Von Baeyer cascade (..) on the given atom ordering.

        When ``optimize_orientation`` is set (pure-cage path only) the main
        bridgehead chosen as locant 1 and the traversal direction are selected
        to give the lowest secondary-bridge superscript locant set
        /.5). This is RT-safe only without substituents, hence the
        caller gate; it is what turns nortricyclene 0^3,5 into the PIN 0^2,6.
        """
        ring_count = self._get_ring_count(mol, ring_atoms)
        bridgeheads = self._find_all_bridgeheads(mol, ring_atoms)
        if forced is not None:
            main_ring, bh_pair = list(forced[0]), forced[1]
        else:
            main_ring, bh_pair = self._find_main_ring(mol, ring_atoms, bridgeheads)
        main_bridge = self._find_main_bridge(mol, ring_atoms, main_ring, bh_pair)

        if optimize_orientation:
            main_ring, bh_pair = self._select_pin_orientation(
                mol, ring_atoms, main_ring, main_bridge, bh_pair
            )
            # M4#2 Fix B: _select_pin_orientation may swap which bridgehead is
            # locant 1, but _find_main_bridge stored main_bridge.atoms[0] adjacent to
            # the ORIGINAL bh_pair[0]. Re-derive the main bridge for the chosen bh_pair
            # so atoms[0] is adjacent to the live locant-1 bridgehead, matching
            # reconstruct_von_baeyer_skeleton's convention (main bridge "beginning with
            # the atom next to the first bridgehead",:9589). Same atom set
            # (a swap only reverses the traversal), just correctly oriented -- this is
            # exactly what the forced= path already does per candidate. Without it the
            # engine emits the byte-correct descriptor string but a numbering whose
            # main-bridge edges disagree with it, so the audit rejects the PIN.
            main_bridge = self._find_main_bridge(
                mol, ring_atoms, main_ring, bh_pair
            )
        secondary_bridges = self._find_secondary_bridges(
            mol, ring_atoms, main_ring, main_bridge, bh_pair
        )
        #: Classify, order, orient, and number secondary bridges
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
        branch1 = main_ring[:bh2_pos + 1]  # bh1... bh2 (longer)
        branch2 = main_ring[bh2_pos:]      # bh2... back toward bh1 (shorter, includes bh2 but ring wraps)

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

        # IUPAC citation order re-sort for descriptor string
        #.2: decreasing length;.4: lowest locants as ascending set;
        #.5: lowest locants in citation order.
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

        # (:9651, under "Naming polycyclic alicyclic
        # hydrocarbons"): the alkane stem equals the bracket sum + 2. A
        # violation means the descriptor drops a skeletal atom the stem still
        # counts -- `tetracyclo[5.1.1.2^3,6]dodecane` brackets 11 atoms and says
        # 12.
        #
        # This is a DIAGNOSTIC, not the decision. `_analyze_impl` is a candidate
        # generator -- `_choose_lowest_locant_numbering` calls it in a loop and
        # discards most results -- so refusing here would only mean "this
        # candidate lost", not "this molecule cannot be named". The decision is
        # made once, on the result `analyze` actually returns, by
        # `_legality_verified`, and published as `PolycyclicDescriptor.legality`
        # for the name-producers to enforce. Until there was no such
        # decision anywhere and this log line was the whole of the response.
        total_bridge_len = sum(bridge_lengths)
        if total_bridge_len + 2 != total_atoms:
            logger.debug(
                "P-23.2.6.1.4 violated: sum(%s) + 2 = %d, expected %d total "
                "ring atoms (candidate is adjudicated in analyze)",
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
    #: Ring Count
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
        for bond in bonds_of(mol):
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
        # /: delegate to the SINGLE consolidated predicate (identical
        # ring_neighbours>=3 rule, operating on the passed ring component).
        from ..perception.rings import find_ring_bridgeheads
        return find_ring_bridgeheads(mol, ring_atoms)

    # ========================================================================
    #: Main Ring Finding
    # ========================================================================

    def _find_main_ring(
        self,
        mol,
        ring_atoms: Set[int],
        bridgeheads: Set[int],
        return_candidates: bool = False,
    ) -> Tuple[List[int], Tuple[int, int]]:
        """Memoising front of:meth:`_find_main_ring_impl` (Lever N, 2026-09-12).

        The main-ring search is called twice per ring system (``_analyze_impl`` and
        ``_choose_lowest_locant_numbering``) and is the costliest step for bridged cages
        (47 s of a 350 s tail profile). Within one naming scope the result for
        (mol object, ring atoms, bridgeheads) is computed once; the memo entry also
        records how many perf-budget units the computation charged, and a hit charges
        the SAME units again, so the budget trajectory (and any PerfBudgetExceeded
        abstention) is unchanged. RWMol and calls outside a scope run the impl directly.
        """
        key = (frozenset(ring_atoms), frozenset(bridgeheads))
        state = {"hit": True}

        def _fresh():
            state["hit"] = False
            before = getattr(_fragment_guard, "perf_budget", None)
            ring, bhp, oriented = self._find_main_ring_impl(mol, ring_atoms, bridgeheads, return_candidates=True)
            after = getattr(_fragment_guard, "perf_budget", None)
            units = (before - after) if (before is not None and after is not None and before > after) else 0
            return (tuple(ring) if ring else ring, bhp, tuple((tuple(r), b) for r, b in oriented), units)

        # R2(a): an OUTER memo keyed by molecular STRUCTURE, not by mol object. Two
        # different mol objects parsed from the same SMILES with the SAME atom output
        # order (and same ring atoms / bridgeheads) share the result within one scope;
        # a different atom order -> different key -> recompute (indices differ, so the
        # answer does). The per-object ``cached_by_key`` stays as the INNER layer, so a
        # repeat on the SAME object skips the ``_find_main_ring_impl`` work — but note
        # ``_struct_key`` still computes the canonical SMILES on EVERY non-RWMol call
        # (including same-object repeats); the inner memo saves the search, not the
        # SMILES. RWMol (and a mol whose output order is unavailable) falls back to the
        # per-object memo only.
        def _struct_key():
            smi = Chem.MolToSmiles(mol, canonical=True)
            if not mol.HasProp('_smilesAtomOutputOrder'):
                return None
            order = tuple(int(x) for x in _ORDER_INT_RE.findall(mol.GetProp('_smilesAtomOutputOrder')))
            bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in bonds_of(mol))
            return (smi, order, bonds, frozenset(ring_atoms), frozenset(bridgeheads))

        def _by_object():
            return cached_by_key(mol, "polycyclic.main_ring", key, _fresh)

        skey = None if isinstance(mol, Chem.RWMol) else _struct_key()
        if skey is None:
            val = _by_object()
        else:
            val = cache_or_compute("polycyclic.main_ring_struct", skey, _by_object)
        if state["hit"] and val[3]:
            spend_perf_work(val[3])
        ring = list(val[0]) if val[0] else val[0]
        if return_candidates:
            return ring, val[1], [(list(r), b) for r, b in val[2]]
        return ring, val[1]

    def _find_main_ring_impl(
        self,
        mol,
        ring_atoms: Set[int],
        bridgeheads: Set[int],
        return_candidates: bool = False,
    ) -> Tuple[List[int], Tuple[int, int]]:
        """
        : Find the main ring -- the largest ring in the system.

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

        adj_list = {idx: [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
                    for idx in ring_atoms}          # R2(c): GetNeighbors order, built once

        # Spelling-invariant tie-break key: RDKit canonical atom ranks are
        # identical for every SMILES spelling of the same molecule. breakTies=
        # True forces a TOTAL order so that even symmetry-equivalent atoms in a
        # cage (homocubane etc.) get a stable, reproducible ordering -- without
        # it the path-enumeration (GetNeighbors) order would break rank ties
        # non-deterministically. This makes the main-ring selection fully
        # order-independent (determinism).
        try:
            _ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        except Exception:
            _ranks = list(range(mol.GetNumAtoms()))

        def _rank_key(atom_indices):
            return tuple(sorted(_ranks[i] for i in atom_indices))

        best_ring = None
        best_bh_pair = None
        # Score tuple, compared with ">" (higher wins). Faithful to the
        # -> -> cascade:
        # (ring_size, #: main ring includes max skeletal atoms
        # main_bridge_len, #: main bridge as large as possible
        # balance, #: main ring divided as symmetrically as possible
        # tie_key) # deterministic canon-rank tie-break (lowest ranks win)
        best_score = None
        # Every decomposition tied at the top of the STRUCTURAL part of the score
        # (ring_size, main_bridge_len, balance). These all yield the same von
        # Baeyer descriptor, so does not distinguish them -- the choice
        # between them is exactly the "choice for numbering" that and
        # govern. Retained so ``_choose_lowest_locant_numbering`` can rank
        # them on locants instead of on the arbitrary canonical-rank backstop.
        tied_candidates: List[Tuple[List[int], Tuple[int, int]]] = []

        def _consider(ring, bh1, bh2, branch1_len, branch2_len, main_bridge_len):
            nonlocal best_score, best_ring, best_bh_pair
            balance = min(branch1_len, branch2_len)
            tie_key = tuple(-r for r in _rank_key(ring))
            structural = (len(ring), main_bridge_len, balance)
            score = (structural, tie_key)
            if best_score is None or structural > best_score[0]:
                tied_candidates.clear()
            if best_score is None or structural >= best_score[0]:
                tied_candidates.append((ring, (bh1, bh2)))
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
            all_paths = _find_all_simple_paths(mol, bh1, bh2, allowed, adj=adj_list)
            if exclude_direct:
                all_paths = [p for p in all_paths if len(p) >= 3]
            if len(all_paths) < 2:
                return None
            best_local = None
            best_pair = None
            for i, p1 in enumerate(all_paths):
                p1_interior = set(p1[1:-1])
                for p2 in all_paths[i + 1:]:
                    spend_perf_work()  # M2.5: charge the O(paths^2) pairing loop
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

            # Case 3 /: a 2+-atom main bridge. Cases 1
            # and 2 only ever offer a 0- or 1-atom main bridge, so a bridgehead
            # pair joined to the rest of the cage by a longer third path never had
            # its true, larger main bridge scored -- the ``main_bridge_len`` slot
            # of the score tuple could hold only 0 or 1, and Blue Book PINs such
            # as tricyclo[9.3.3.1^1,11] (the Blue Book) came out with a
            # non-preferred main bicycle. The three primary segments of the main
            # bicycle are three mutually interior-disjoint bh1->bh2 paths: the two
            # longest form the main ring, largest ring), and the longest
            # of the remaining disjoint paths is the main bridge, largest
            # bridge). The (ring_size, main_bridge_len, balance) score then ranks
            # this faithfully against Cases 1/2, and among equal main bridges
            # prefers the more symmetric main-ring division. Cases
            # 1/2 are kept: this is a strict ADD, skipped whenever the largest
            # available main bridge is only 0 or 1 atom.
            #
            # Case 3 is preference-only and can, on a cage whose secondary
            # bridges cannot be numbered consistently under the new main bicycle,
            # yield a descriptor that fails the legality audit. ``analyze`` runs
            # the cascade with this enabled first and, only if the result is
            # illegal, re-runs it disabled -- so the preferred main bridge can
            # never COST a legal (if non-preferred) name (accurate-or-degrade,
            # never a regression).
            if not getattr(self, "_allow_multiatom_bridge", True):
                continue
            all_paths = _find_all_simple_paths(mol, bh1, bh2, ring_atoms, adj=adj_list)
            interiors = [set(p[1:-1]) for p in all_paths]
            npaths = len(all_paths)
            for i in range(npaths):
                p1 = all_paths[i]
                int_i = interiors[i]
                for j in range(i + 1, npaths):
                    spend_perf_work()  # M2.5: charge the O(paths^2) pairing loop
                    int_j = interiors[j]
                    if int_i & int_j:
                        continue
                    used = int_i | int_j
                    # Largest remaining interior-disjoint path is the main bridge.
                    best_mb_len = -1
                    for k in range(npaths):
                        if k == i or k == j or interiors[k] & used:
                            continue
                        if len(interiors[k]) > best_mb_len:
                            best_mb_len = len(interiors[k])
                    if best_mb_len < 2:
                        # 0/1-atom main bridges are already covered by Cases 1/2.
                        continue
                    p2 = all_paths[j]
                    ring = p1 + p2[1:-1][::-1]
                    _consider(ring, bh1, bh2, len(p1) - 2, len(p2) - 2, best_mb_len)

        # Fallback: use largest ring method (for bicyclo or unusual cases)
        if best_ring is None:
            best_ring, best_bh_pair = self._find_main_ring_fallback(
                mol, ring_atoms, bridgeheads
            )

        # Reorder main ring so that bh1 is first and the longer path to bh2 comes first
        if best_ring and best_bh_pair:
            best_ring = self._orient_main_ring(best_ring, best_bh_pair)

        if return_candidates:
            oriented = []
            seen_c = set()
            for ring, bhp in tied_candidates:
                ring = self._orient_main_ring(ring, bhp)
                k = (tuple(ring), bhp)
                if k not in seen_c:
                    seen_c.add(k)
                    oriented.append((ring, bhp))
            return best_ring, best_bh_pair, oriented

        return best_ring, best_bh_pair

    def _find_path_avoiding_direct(
        self, mol, start: int, end: int, allowed: Set[int]
    ) -> List[int]:
        """Find a path from start to end that doesn't use the direct bond.

        Bounded by ``_MAX_DFS_EXPANSIONS``  so a dense cage cannot hang;
        on abort the best path found so far is returned.
        """
        # Use BFS to find shortest path first, then try longer paths
        best_path = []
        expansions = 0

        def dfs(current, visited, path):
            nonlocal best_path, expansions
            expansions += 1
            if expansions > _MAX_DFS_EXPANSIONS:
                return  # cap: abort exploration, keep best-so-far
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
        - The longer path from bh1 to bh2 comes first (numbering requirement)

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
        , then the lowest citation sequence, then
        the /.3 double-(and triple-)bond locant criteria
        (``_unsaturation_locant_key``, SP2), with a deterministic atom-index
        backstop. Both main bridgeheads stay the main bridgeheads, so the main
        bridge is unchanged and the cage is unchanged -- only the numbering
        moves. The tiers fix the descriptor's OWN numbering first;
         then breaks any REMAINING choice using the unsaturation
        locants -- exactly the residual-freedom case its own text names ("if
        there is a choice of names and numbering").
        """
        bh1, bh2 = bh_pair
        ring_set = set(main_ring)
        if bh1 not in ring_set or bh2 not in ring_set or bh1 == bh2:
            return main_ring
        n = len(main_ring)
        BIG = 10 ** 6

        # (:16635) / (:16675): ring double/triple bonds,
        # derived once from mol/ring_atoms (candidate-independent -- only the
        # NUMBERING varies per candidate below).
        ring_double_bonds = []
        ring_triple_bonds = []
        for b in bonds_of(mol):
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a1 in ring_atoms and a2 in ring_atoms:
                order = b.GetBondTypeAsDouble()
                if order == 2.0:
                    ring_double_bonds.append((a1, a2))
                elif order == 3.0:
                    ring_triple_bonds.append((a1, a2))

        candidates = []
        seen = set()
        for start, partner in ((bh1, bh2), (bh2, bh1)):
            s_idx = main_ring.index(start)
            rot = main_ring[s_idx:] + main_ring[:s_idx]
            p_idx = rot.index(partner)
            forward_len = p_idx - 1
            backward_len = n - p_idx - 1
            reflected = [rot[0]] + rot[1:][::-1]
            # Keep the longer branch first ; admit both directions on a tie.
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
            flat_set = tuple(sorted(x for pr in pairs for x in pr))       #
            citation = tuple(x for pr in pairs for x in pr)               #
            unsat_key = self._unsaturation_locant_key(
                numbering, ring_double_bonds, ring_triple_bonds)          # /.3
            idx_backstop = tuple(cand_ring)
            key = (flat_set, citation, unsat_key, idx_backstop)
            if best_key is None or key < best_key:
                best_key = key
                best_ring = cand_ring
                best_bhp = cand_bhp
        return best_ring, best_bhp

    @staticmethod
    def _unsaturation_locant_key(numbering, ring_double_bonds, ring_triple_bonds):
        """ (:16633) / (:16675) numbering criteria for
        double-(and triple-)bond locants, as a sort key (lower wins).

         "If there is a choice of names and numbering...":
          (1) (:16635) a minimum number of COMPOUND locants -- a double bond
              needs a compound locant when its endpoints' locants do not
              differ by exactly one;
          (2) (:16657) comparing double-bond locants with any parenthetical
              (compound) partner IGNORED;
          (3) (:16674) "if there is still a choice, low locants are selected
              considering ALL locants (including those in parentheses) as a
              set" -- verbatim example ``tetracyclo[7.7.1.1^3,7.1^11,15]
              nonadeca-3,11(18)-diene [not...-3(19),11-diene; the locant
              set '3,11,18' is lower than '3,11,19']``.
         (:16675, both double AND triple bonds present):
          (1) (:16679) lower locants to the combined multiple-bond set;
          (2) (:16683) lower locants to the double bonds (ignoring the
              triple-bond locants);
          (3) (:16689) compound locants kept to a minimum.

        Criteria (1)/(2) here are compound-AWARE (``compound_aware_
        multibond_set``, shared with ``bicyclo.py``'s own
        branch), NOT the naive cited-locant compare (2) uses --
        BB's own worked example for this criterion (bicyclo[8.3.1]tetradeca-
        4,6,10-trien-2-yne, PIN,:16693) is a counter-example to naive
        cited-locant comparison: the rejected numbering has a numerically
        LOWER naive cited set despite needing a compound locant. See that
        helper's docstring (locants.py) for the full derivation.
        """
        def cited(bonds):
            # The single "cited" (non-parenthetical) locant per bond -- the
            # lower of its two endpoints, matching the compound-locant
            # convention "the higher locant is cited in parentheses" (:16637).
            # Used ONLY by the no-triple branch below --
            # (2) explicitly says "any number in parentheses is
            # ignored" (:16657), a genuinely naive compare (verified against
            # the hexadeca-triene worked example there), unlike (1)/(2) of
            # just above.
            return tuple(sorted(
                min(numbering[a], numbering[b])
                for a, b in bonds if a in numbering and b in numbering
            ))

        def compound_count():
            return sum(
                1 for a, b in ring_double_bonds
                if a in numbering and b in numbering
                and abs(numbering[a] - numbering[b]) != 1
            )

        if ring_triple_bonds:
            return (
                compound_aware_multibond_set(
                    numbering, ring_double_bonds + ring_triple_bonds),  # (1)
                compound_aware_multibond_set(
                    numbering, ring_double_bonds),                     # (2)
                compound_count(),                                       # (3)
            )

        # (3): non-compound bonds contribute only their single
        # (lower) locant; compound bonds contribute BOTH endpoints.
        full = []
        for a, b in ring_double_bonds:
            if a not in numbering or b not in numbering:
                continue
            lo, hi = min(numbering[a], numbering[b]), max(numbering[a], numbering[b])
            full.append(lo)
            if hi - lo != 1:
                full.append(hi)
        return (
            compound_count(),           # (1)
            cited(ring_double_bonds),   # (2)
            tuple(sorted(full)),        # (3)
        )

    #: Cap on the number of alternative numberings ranked by
    #: ``_choose_lowest_locant_numbering``. Symmetric cages can present many
    #: tied decompositions; the cascade is O(candidates x cage size) and this
    #: keeps a pathological cage from dominating a naming run. On exceeding the
    #: cap the incumbent numbering is kept (never a partial ranking, which would
    #: be order-dependent).
    _MAX_LOCANT_CANDIDATES = 64

    #: (:9765) heteroatom citation order, used by as the
    #: "decreasing seniority order of heteroatoms" tie-break.
    _HETERO_SENIORITY = {
        sym: rank for rank, sym in enumerate(
            ("F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As", "Sb",
             "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl")
        )
    }

    def _principal_group_ring_atoms(self, mol, ring_atoms) -> Set[int]:
        """Cage atoms that bear the principal characteristic group (c)).

        Sourced from the existing seniority primitives -- ``detect_functional_groups``
        -> ``get_principal_group`` seniority) -> ``_pg_attachment_atoms``
        (the per-FG locant-bearing SMARTS index) -- rather than from a local
        heteroatom guess, which cannot tell an ``-ol`` suffix from an ``amino``
        prefix.

        ``_pg_attachment_atoms`` falls back to SMARTS atom 0 for any FG absent
        from ``PG_ATTACHMENT_INDICES``, and several O/N SMARTS lead with the
        heteroatom rather than the carbon that carries the locant. That fallback
        atom is therefore often OFF the cage. Handled explicitly: an attachment
        atom outside the ring contributes the ring atom(s) it is bonded to, so a
        table gap degrades to "locate via the bond" instead of silently dropping
        the principal group from the (c) term.
        """
        try:
            from ..perception.functional_groups import detect_functional_groups
            from .parent_selection import _pg_attachment_atoms
            from .seniority import get_principal_group
        except Exception:
            return set()
        try:
            fgs = detect_functional_groups(mol)
            pg_name, pg_matches = get_principal_group(mol, fgs)
        except Exception:
            return set()
        if not pg_name or not pg_matches:
            return set()

        out: Set[int] = set()
        for match in pg_matches:
            attachments = _pg_attachment_atoms(pg_name, tuple(match))
            for a in attachments:
                if a in ring_atoms:
                    out.add(a)
                else:
                    # Table gap / heteroatom-leading SMARTS: walk the one bond
                    # back onto the cage.
                    try:
                        nbrs = mol.GetAtomWithIdx(a).GetNeighbors()
                    except Exception:
                        continue
                    for nbr in nbrs:
                        if nbr.GetIdx() in ring_atoms:
                            out.add(nbr.GetIdx())
        return out

    def _locant_criteria_key(self, mol, ring_atoms, desc, spiro_atom=None):
        """The -> numbering cascade, as a sort key (lower wins).

        Applied ONLY among numberings that produce an identical von Baeyer
        descriptor, so this can never change the ring analysis -- it only
        chooses between numberings that left open.

        Order (each verified against the Blue Book, heading + sentence):
          0. (:10272; PIN example:10289 "the spiro atom... is given
             preference for low locant";:10186 "low locants are given to the
             spiro atom, THEN to the heteroatoms"): when this cage is a component
             of a spiro ring system, the spiro-junction atom takes the lowest
             locant, ABOVE the heteroatom criteria. This criterion is present
             ONLY when ``spiro_atom`` is passed (the tricyclo+ spiro-component
             path threads it); for every whole-molecule cage caller
             ``spiro_atom`` is None and the returned key is byte-identical to the
             pre-spiro 5-tuple, so no whole-molecule numbering can change.
          1. (:9777) "Low locants are assigned to the heteroatoms
             considered together as a set compared in increasing numerical
             order." Skeletal heteroatoms are part of the parent hydride and so
             precede suffixes -- preamble (1) (:3226).
          2. (:9789) heteroatoms in decreasing seniority order.
          3. (c) (:3256) "principal characteristic groups and free
             valences (suffixes)".
          4. (e) (:3286) saturation/unsaturation.
          5. (f) (:3296) detachable alphabetized prefixes.

        This is the tricyclo+ mirror of the spiro-priority already enforced on
        the spiro BICYCLIC branch (``spiro._name_vonbaeyer_spiro_component._key``,
        which sorts on ``(spiro_loc, het, sen, ene)``).
        """
        numbering = desc.numbering
        pg_ring_atoms = self._principal_group_ring_atoms(mol, ring_atoms)

        hetero, hetero_rank, pg, unsat, prefixes = [], [], [], [], []
        for idx in ring_atoms:
            loc = numbering.get(idx)
            if not loc:
                continue
            atom = mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            if sym != "C":
                hetero.append(loc)
                hetero_rank.append((loc, self._HETERO_SENIORITY.get(sym, 99)))
            if idx in pg_ring_atoms:
                # (c): the PRINCIPAL characteristic group only. Every other
                # substituent is a detachable prefix and is ranked at (f) -- the
                # two must not be pooled, or an -amino prefix ties with an -ol
                # suffix and the arbitrary backstop decides between them.
                pg.append(loc)
                continue
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_atoms:
                    continue
                if nbr.GetAtomicNum() <= 1:
                    continue
                prefixes.append(loc)
                break
        for bond in bonds_of(mol):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_atoms and b in ring_atoms and \
                    bond.GetBondTypeAsDouble() > 1.0:
                la, lb = numbering.get(a), numbering.get(b)
                if la and lb:
                    unsat.append(min(la, lb))

        rest = (
            tuple(sorted(hetero)),
            tuple(r for _, r in sorted(hetero_rank)),
            tuple(sorted(pg)),
            tuple(sorted(unsat)),
            tuple(sorted(prefixes)),
        )
        if spiro_atom is None:
            # Byte-identical to the pre-spiro key for every whole-molecule caller.
            return rest
        #: spiro-junction locant is the MOST senior criterion. A spiro
        # atom missing from this numbering sorts last (never chosen over one that
        # places it).
        spiro_loc = numbering.get(spiro_atom, 10 ** 6)
        return (spiro_loc,) + rest

    def _choose_lowest_locant_numbering(self, mol, ring_atoms, bridgeheads,
                                        incumbent, spiro_atom=None):
        """Pick the lowest-locant numbering among the legal alternatives.

         fixes the descriptor but frequently leaves several numberings
        open (which main bridgehead is locant 1, the traversal direction, and --
        on a symmetric cage -- which of several equivalent decompositions is
        used). (:9777) *"When there is a choice for numbering, the
        following criteria are applied in order until a decision can be made"*
        and (:3219) then govern that choice; before this pass it was made
        by an arbitrary canonical-rank backstop.

        Only candidates whose ``descriptor_string`` is IDENTICAL to the
        incumbent's are considered, so the ring analysis is provably untouched
        and the change is confined to locants.
        """
        try:
            _, _, candidates = self._find_main_ring(
                mol, ring_atoms, bridgeheads, return_candidates=True
            )
        except Exception:
            return incumbent
        if not candidates or len(candidates) > self._MAX_LOCANT_CANDIDATES:
            return incumbent

        # Expand each tied decomposition over its legal orientations: either
        # main bridgehead may be locant 1 ":9591" -- "starting with
        # one of the bridgeheads"), and on a symmetric main ring either
        # traversal direction is legal.
        expanded = []
        seen = set()
        for ring, bhp in candidates:
            for cand_ring, cand_bhp in self._orientation_variants(ring, bhp):
                k = (tuple(cand_ring), cand_bhp)
                if k not in seen:
                    seen.add(k)
                    expanded.append((cand_ring, cand_bhp))
        if len(expanded) > self._MAX_LOCANT_CANDIDATES:
            return incumbent

        best = incumbent
        best_key = self._locant_criteria_key(mol, ring_atoms, incumbent, spiro_atom)
        for cand in expanded:
            try:
                desc = self._analyze_impl(mol, ring_atoms, forced=cand)
            except Exception:
                continue
            if desc is None or not desc.numbering:
                continue
            # The descriptor is settled by -- a candidate that changes it
            # is a different ring analysis, not a renumbering. Reject it.
            if desc.descriptor_string != incumbent.descriptor_string:
                continue
            # Every skeletal atom must receive exactly one locant: a candidate
            # that drops or doubles an atom is malformed, not merely worse.
            locs = [desc.numbering.get(i) for i in ring_atoms]
            if any(l is None for l in locs) or len(set(locs)) != len(ring_atoms):
                continue
            key = self._locant_criteria_key(mol, ring_atoms, desc, spiro_atom)
            if key < best_key:
                best_key = key
                best = desc
        return best

    def _orientation_variants(self, main_ring, bh_pair):
        """Legal (ring, bh_pair) orientations of one fixed decomposition.

         "Numbering bicyclic alicyclic hydrocarbons" (:9589), sentence
        :9591: *"The bicyclic ring system is numbered starting with one of the
        bridgeheads and proceeding first along the longer segment of the main
        ring to the second bridgehead, then back to the first bridgehead along
        the unnumbered segment of the main ring."* Either bridgehead may start;
        the longer segment must come first, so the direction is free only when
        the two segments are equal.
        """
        bh1, bh2 = bh_pair
        out, seen = [], set()
        if bh1 not in main_ring or bh2 not in main_ring or bh1 == bh2:
            return [(list(main_ring), bh_pair)]
        n = len(main_ring)
        for start, partner in ((bh1, bh2), (bh2, bh1)):
            s = main_ring.index(start)
            rot = main_ring[s:] + main_ring[:s]
            p = rot.index(partner)
            forward_len, backward_len = p - 1, n - p - 1
            reflected = [rot[0]] + rot[1:][::-1]
            if forward_len > backward_len:
                options = [rot]
            elif backward_len > forward_len:
                options = [reflected]
            else:
                options = [rot, reflected]
            for opt in options:
                k = tuple(opt)
                if k not in seen:
                    seen.add(k)
                    out.append((opt, (start, partner)))
        return out

    # ========================================================================
    #: Main Bridge
    # ========================================================================

    def _find_main_bridge(
        self,
        mol,
        ring_atoms: Set[int],
        main_ring: List[int],
        bh_pair: Tuple[int, int]
    ) -> Optional[BridgeInfo]:
        """
        : Find the main bridge -- the path between the two main
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

        # Find atoms outside the main ring that connect the two main bridgeheads.
        allowed = (ring_atoms - main_ring_set) | {bh1, bh2}

        if len(allowed) <= 2:
            # No atoms outside the main ring: the main bridge is a 0-atom bridge
            # (a direct bond between the bridgeheads) or absent -- both length 0.
            return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)

        # "Selection of the main bridge" (the Blue Book): the main
        # bridge "is the bridge that includes as many of the atoms as possible
        # that are not included in the main ring" -- i.e. the LONGEST bridge
        # between the two main bridgeheads through non-main-ring atoms, not the
        # shortest. (in the class docstring always said "longest"; the
        # implementation took the shortest path, so whenever a bridgehead pair
        # was joined by more than one non-main-ring path the main bridge was
        # understated and a longer path was demoted to a secondary bridge -- e.g.
        # tricyclo[9.3.3.1^1,11] came out as the non-preferred [9.3.1.3^1,11].)
        # A 0-atom direct bond, if present, is enumerated as the length-2 path
        # [bh1, bh2] and so competes as a 0-length candidate. Deterministic
        # tie-break among equal-length longest bridges: the lowest canonical-rank
        # interior (spelling-invariant, matching _find_main_ring's tie key).
        paths = _find_all_simple_paths(mol, bh1, bh2, allowed)
        paths = [p for p in paths if len(p) >= 2]
        if not paths:
            return BridgeInfo(atoms=[], length=0, start_bh=bh1, end_bh=bh2)
        try:
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        except Exception:
            ranks = list(range(mol.GetNumAtoms()))
        best = max(paths, key=lambda p: (len(p), tuple(-ranks[a] for a in p[1:-1])))

        bridge_atoms = best[1:-1]  # Exclude bridgeheads
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
    #: Secondary Bridges
    # ========================================================================

    def _walk_to_covered(self, ep, comp, covered, adj):
        """From assigned endpoint ``ep``, BFS through UNCOVERED component atoms to
        the nearest already-covered atom (the junction where a dependent bridge lands
        on an existing bridge,. Returns ``(junction, chain)`` where ``chain``
        is the uncovered component atoms on the path (excluding ``ep`` and the
        junction), or ``(None, )`` if no covered atom is reachable.

        Deterministic: sorted neighbours, shortest path. M4#2 Fix A.
        """
        # Direct length-0 dependent bond: ep is adjacent to a covered atom already.
        direct = [nb for nb in sorted(adj.get(ep, [])) if nb in covered]
        if direct and not any(nb in comp and nb not in covered
                              for nb in adj.get(ep, [])):
            return direct[0], []
        q = deque()
        visited = {ep}
        for nb in sorted(adj.get(ep, [])):
            if nb in comp and nb not in covered:
                q.append((nb, [nb]))
                visited.add(nb)
        while q:
            cur, path = q.popleft()
            covered_nbrs = [nb for nb in sorted(adj.get(cur, [])) if nb in covered]
            if covered_nbrs:
                return covered_nbrs[0], path
            for nb in sorted(adj.get(cur, [])):
                if nb in comp and nb not in covered and nb not in visited:
                    visited.add(nb)
                    q.append((nb, path + [nb]))
        return None, []

    def _endpoint_locants(self, main_ring, main_bridge):
        """The locant each main-ring / main-bridge atom will receive, computed the
        way ``_order_and_number_secondary_bridges`` Steps 1-2 number them (main ring
        1..n in order, then the main bridge). Lets the branched-component tie-break
        rank endpoints by their LOCANT rather than by raw atom index.
         M4#2 Fix A (a review point 1)."""
        loc = {}
        for i, a in enumerate(main_ring):
            loc.setdefault(a, i + 1)
        n = len(loc)
        if main_bridge and main_bridge.atoms:
            for a in main_bridge.atoms:
                if a not in loc:
                    n += 1
                    loc[a] = n
        return loc

    def _decompose_branched_component(self, mol, component, endpoints, assigned, adj,
                                      main_ring=None, main_bridge=None, bh_pair=None):
        """ M4#2 Fix A: decompose a BRANCHED secondary-bridge component (one that
        attaches to ≥3 assigned endpoints, or whose 2-endpoint longest path leaves
        component atoms uncovered) into an independent bridge plus dependent
        bridge(s).

        The independent (trunk) bridge is the longest path between two assigned
        endpoints that covers the most component atoms. Each remaining assigned
        endpoint then anchors a dependent bridge whose far end is an interior atom of
        an already-placed bridge "a secondary bridge that links at least one
        bridgehead that is part of a secondary bridge"); the existing Step-6 resolver
        in ``_order_and_number_secondary_bridges`` numbers these once the trunk's
        atoms are numbered.

        Returns a ``list[BridgeInfo]`` covering EVERY component atom (fail-closed:
        returns ``None`` if it cannot, so the caller degrades rather than silently
        dropping an atom -- the old bug this replaces).

        Trunk tie-break (a review point 1): among equal-length trunks the winner is the
        one whose two endpoints have the LOWEST locants, computed from
        the main-ring/main-bridge numbering; atom index is only the final determinism
        backstop. Because the locant of an endpoint depends on the main-ring
        orientation, ``_select_pin_orientation``'s lowest-locant search now selects
        the PIN decomposition -- which an orientation-invariant atom-index tie-break
        could not.

        Source: https://iupac.qmul.ac.uk/BlueBook/P2.html / /.
        """
        comp = set(component)
        eps = sorted(endpoints)
        ep_loc = (self._endpoint_locants(main_ring, main_bridge)
                  if main_ring is not None else {})
        _BIG = 10 ** 9

        def _loc(a):
            return ep_loc.get(a, _BIG)

        # Trunk (independent bridge): longest endpoint-to-endpoint path covering the
        # most component atoms; tie-break by lowest endpoint LOCANTS,
        # then raw atom index + path as the deterministic backstop.
        best = None
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                a, b = eps[i], eps[j]
                path = find_longest_path(mol, a, b, comp | {a, b})
                if not path or len(path) < 2:
                    continue
                interior = path[1:-1]
                if not set(interior) <= comp:
                    continue
                lo_loc, hi_loc = sorted((_loc(a), _loc(b)))
                key = (-len(interior), lo_loc, hi_loc, a, b, tuple(path))
                if best is None or key < best[0]:
                    best = (key, a, b, interior)
        if best is None:
            return None
        _, ta, tb, trunk_interior = best
        bridges = [BridgeInfo(atoms=list(trunk_interior),
                              length=len(trunk_interior),
                              start_bh=ta, end_bh=tb, is_secondary=True)]
        covered = set(trunk_interior)
        used_eps = {ta, tb}
        # Dependent bridges: each remaining endpoint walks inward to a covered atom.
        for ep in eps:
            if ep in used_eps:
                continue
            junction, chain = self._walk_to_covered(ep, comp, covered, adj)
            if junction is None:
                return None
            bridges.append(BridgeInfo(atoms=list(chain), length=len(chain),
                                      start_bh=ep, end_bh=junction,
                                      is_secondary=True))
            covered |= set(chain)
        # Completeness invariant (fail-closed): every component atom must be covered.
        if not comp <= covered:
            return None
        return bridges

    def _find_secondary_bridges(
        self,
        mol,
        ring_atoms: Set[int],
        main_ring: List[int],
        main_bridge: Optional[BridgeInfo],
        bh_pair: Tuple[int, int]
    ) -> List[BridgeInfo]:
        """
        : Find secondary bridges in the ring system.

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

        # --- a phase: Find bridges through unassigned atoms ---
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
                # component discovery is order-independent (determinism).
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
                    ep1, ep2 = ep_list[0], ep_list[1]
                    component_plus_endpoints = component | {ep1, ep2}
                    path = find_longest_path(mol, ep1, ep2, component_plus_endpoints)

                    # M4#2 Fix A: a SIMPLE component has exactly 2 endpoints and a
                    # path that covers every component atom -- keep that byte-identical.
                    # A BRANCHED component (≥3 endpoints, or a 2-endpoint path that
                    # leaves atoms uncovered) previously collapsed to this one path and
                    # silently dropped the rest; decompose it into an independent bridge
                    # + dependent bridge(s) instead.
                    simple = (
                        len(endpoints) == 2
                        and ((path and set(path[1:-1]) == component)
                             or (not path and not component)))
                    if simple:
                        if path and len(path) >= 2:
                            bridge_atoms = path[1:-1]
                            secondary_bridges.append(BridgeInfo(
                                atoms=bridge_atoms, length=len(bridge_atoms),
                                start_bh=ep1, end_bh=ep2, is_secondary=True))
                        else:
                            # Zero-length bridge (direct connection between endpoints)
                            secondary_bridges.append(BridgeInfo(
                                atoms=[], length=0, start_bh=ep1, end_bh=ep2,
                                is_secondary=True))
                    else:
                        decomposed = self._decompose_branched_component(
                            mol, component, endpoints, assigned, adj,
                            main_ring, main_bridge, bh_pair)
                        if decomposed is not None:
                            secondary_bridges.extend(decomposed)
                        elif path and len(path) >= 2:
                            # Fail-closed fallback: the old single-path bridge (may
                            # fail the audit and degrade, never silently drops here
                            # because the audit re-checks coverage).
                            secondary_bridges.append(BridgeInfo(
                                atoms=path[1:-1], length=len(path[1:-1]),
                                start_bh=ep1, end_bh=ep2, is_secondary=True))
                        else:
                            secondary_bridges.append(BridgeInfo(
                                atoms=[], length=0, start_bh=ep1, end_bh=ep2,
                                is_secondary=True))
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

        # --- a phase: Detect zero-length secondary bridges ---
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

        # Also add edges from bridges found via unassigned atoms (a phase)
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
            for bond in bonds_of(mol):
                a_idx = bond.GetBeginAtomIdx()
                b_idx = bond.GetEndAtomIdx()
                if a_idx in ring_atoms and b_idx in ring_atoms:
                    edge = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if edge not in accounted_edges:
                        zero_length_candidates.append(edge)
                        accounted_edges.add(edge)  # Don't double count

            # Determinism : RDKit GetBonds iteration order is NOT
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

        # Bridge ordering is handled by _order_and_number_secondary_bridges
        # in analyze, which implements proper independent/dependent
        # classification and multi-criteria sort.

        return secondary_bridges

    # ========================================================================
    #: Classification, Ordering, and Numbering
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
        : Classify, order, orient, and number all ring atoms.

        Implements a two-pass approach to resolve the circular dependency
        between locant assignment and bridge ordering:

        1. Number main ring and main bridge atoms (locants are fixed)
        2. Classify secondary bridges as independent/dependent
        3. Sort independent bridges by criteria and number them
        4. Iteratively resolve and number dependent bridges
        5. Orient each bridge so numbering starts from higher-numbered
           bridgehead (: "numbered starting from atom next to
           higher-numbered bridgehead")

        Sort criteria match OPSIN's VonBaeyerSecondaryBridgeSort:
        (-locant_high, -locant_low, -bridge_length), which is consistent
        with IUPAC ("beginning with the one attached to the
        highest-numbered bridgehead") and.2 ("longer bridges
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
        #: Independent = both endpoints on main ring or main bridge
        # Dependent = at least one endpoint on a secondary bridge
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

        # Step 4: Sort independent bridges by numbering criteria
        #: "beginning with the one attached to the highest-numbered bridgehead"
        #.2: "longer bridges numbered before shorter bridges"
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
            #: "numbered starting from atom next to higher-numbered bridgehead"
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

            # Sort by criteria with current numbering
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
    #: Numbering (legacy — used by _order_and_number_secondary_bridges)
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
        : Assign IUPAC locants to all ring atoms.

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

        ``length^low,high`` e.g. ``0^2,6`` /. OPSIN
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
    of a ring (IsInRing == True). Bonds between ring atoms that are NOT
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

    Von Baeyer and bicyclo nomenclature describe SATURATED bridged ring
    skeletons; unsaturation is expressible only as ``-ene``/``-yne`` with
    locants, and aromaticity CANNOT be represented at all. Naming a cage that
    contains aromatic ring atoms therefore silently DROPS the aromaticity and
    emits a structurally WRONG (de-aromatised) cage — e.g. benzonorbornadiene
    ``C1C2C=CC1c1ccccc12`` -> ``tricyclo[4.4.0.1(2,5)]undec-3-ene`` (the benzo
    ring desaturated). The correct PIN is a bridged-fused name, e.g.
    ``1,4-dihydro-1,4-methanonaphthalene``), a Phase-G1 build; until then the
    caller must fail closed (raise the limit -> ``unknown organic compound`` /
    ``OrthonymLimitError``) rather than emit the wrong saturated cage.

    ``cage_atoms`` MUST be the EXACT atom set the namer numbers — the von-Baeyer
    descriptor's ``numbering`` keys for ``name_polycyclic_complete``, or
    ``get_complete_bicyclo_data['ring_atoms']`` for the bicyclo path — NOT the
    molecule's largest connected ring component. Keying off the actual cage
    means a PENDANT aromatic ring joined by a single (non-ring) bond — e.g. a
    naphthyl on norbornane, even when the naphthyl is LARGER than the cage — is
    never part of the cage and so can never trip the guard (it is correctly
    named as a substituent). (.)
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
    name_polycyclic_complete in Plan 16-03.

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

    # Legality gate  -- the descriptor and the alkane stem below
    # are both read straight off `desc`, so an unverified descriptor here spells
    # a cage the molecule does not have. Measured at HEAD, this function emitted
    # `tetracyclo[3.1.1.2^1,4]decane` for C1CC23CC(C2)C12CC3C2: the brackets
    # account for 9 atoms, `decane` says 10.
    if desc is None or desc.legality is not True:
        return None

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
    #, not a plain fused aromatic; the VB path delegates it to
    # name_bridged_fused_pin. Fusion chalcogens (dibenzofuran) stay skipped.
    all_ring_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )
    if all_ring_aromatic:
        from .bridged_fused import has_aromatic_chalcogen_bridge, has_aromatic_mancude_bridge
        # Also exempt an all-carbon UNSATURATED bridge etheno):
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

# a phase T2a: the replacement-prefix table and its λ helper moved to
# ``rules/ring_replacement.py`` (the single source of truth, so extending the
# element table is a data-only change in one place). Re-exported here because the
# name ``polycyclic.HETEROATOM_PREFIXES`` is part of this module's surface.
from ..perception.molcache import bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from .ring_replacement import (  # noqa: E402,F401 (re-export)
    HETEROATOM_PREFIXES,
)
from .ring_replacement import (
    build_replacement_prefix as _build_ring_replacement_prefix,
)
from ..perception.smarts_cache import compiled as _compiled_smarts


def get_heteroatom_replacement_prefix(
    mol, numbering: Dict[int, int], ring_atoms: Set[int],
) -> Optional[str]:
    """
    Generate the 'a' replacement-nomenclature prefix for ring heteroatoms.

    Thin wrapper over ``rules/ring_replacement.build_replacement_prefix``, which
    owns the construction (element table, Table-2.8 citation order, λ tokens,
    multiplying prefixes, the no-trailing-hyphen rule of. The string
    returned here is byte-identical to what this function built inline before
     a phase — the three PIN callers (``name_polycyclic_complete``,
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
    pattern = _compiled_smarts("[CX3](=O)[OX2]")
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

    # Legality gate . Doubly load-bearing here: an unverified
    # descriptor is spelled below AND its numbering is what places every
    # 'oxa'/'aza'/'thia' locant, so a numbering the descriptor disagrees with
    # puts the heteroatoms on the wrong skeletal positions.
    if desc is None or desc.legality is not True:
        return None

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
    substituent branch and determines its name using get_alkyl_name.

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
            # Carbon is therefore named here as the free-valence prefix
            # it is (methylidene / ethylidene / propan-2-ylidene /...), and
            # the whole ring system fails closed when it cannot be. Non-carbon
            # keeps the existing skip: promoting an unrecognised ring C=O to an
            # 'oxo' prefix would ship a non-PIN name where the code correctly
            # abstains today, which is a different phase's decision to make.
            #
            # The verdict itself comes from the shared primitive rather
            # than being re-derived here: this file was the second place to
            # read an attachment bond order, and every further copy is another
            # detector that can drift.
            bond = mol.GetBondBetweenAtoms(ring_idx, nbr_idx)
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                if neighbor.GetAtomicNum() != 6:
                    continue
                from ..assembly.substituent_enumerator import carbon_free_valence_prefix
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

            # macrocycle F3: an OXYGEN-attached ACYLOXY (-O-C(=O)-, an ester) or
            # PEROXY (-O-O-, a hydroperoxy) branch is named by the GENERAL CASCADE, which
            # produces the correct token (acetyloxy / hydroperoxy). The naive _ALKOXY_NAMES
            # path below is a drifted detector that mis-named these by carbon count alone
            # (acetate -O-C(=O)-CH3 -> 'ethoxy', dropping the carbonyl O; -OOH -> dropped) --
            # a WRONG CONSTITUTION that then abstained on.
            #
            # SCOPE (a review 5.1): fires ONLY at the BEST-EFFORT tier and ONLY on O-attached
            # acyloxy/peroxy branches, so it can NEVER change a default/PIN-tier name.
            # - best-effort gate: at the PIN tier these cages correctly ABSTAIN (a
            # substituent-prefix form of an ester/hydroperoxy is not the PIN); F3 must
            # not emit a non-PIN prefix at the PIN tier ("non-PIN forms are best-effort
            # only"). A carbon-attached ester (-C(=O)OMe) is NOT diverted -- it keeps its
            # functional-class ester PIN ('methyl...carboxylate'), which F3 must not
            # displace with '(methoxycarbonyl)'.
            # - O-attached acyloxy/peroxy only: simple alkoxy (-O-alkyl) and every
            # carbon-attached branch keep the existing paths (byte-identical).
            # Fail-open: a cascade None/refusal falls through to the legacy paths.
            from ..metrics.provenance import best_effort_ctx as _f3_be_ctx
            _first = mol.GetAtomWithIdx(nbr_idx)
            _f3_route = False
            if bool(_f3_be_ctx.get()) and _first.GetSymbol() == 'O':
                _sub_set = set(sub_atoms)
                for _n in _first.GetNeighbors():
                    if _n.GetIdx() not in _sub_set:
                        continue
                    if _n.GetSymbol() == 'O':           # peroxy -O-O-
                        _f3_route = True
                    elif _n.GetSymbol() == 'C' and any(
                            b.GetBondTypeAsDouble() == 2.0
                            and b.GetOtherAtom(_n).GetSymbol() in ('O', 'S', 'N')
                            for b in _n.GetBonds()):     # acyloxy -O-C(=O)-
                        _f3_route = True
            if _f3_route:
                from ..assembly.substituent_enumerator import name_substituent
                _tok = None
                try:
                    _tok = name_substituent(mol, list(sub_atoms), nbr_idx)
                except Exception:  # noqa: BLE001 -- cascade decline is not a crash
                    _tok = None
                if _tok and _tok != 'substituent' and not _tok.startswith('C'):
                    substituents.append({
                        'locant': locant,
                        'name': _tok,
                        'atom_indices': sub_atoms,
                    })
                    continue
                # else fall through to the legacy paths (fail-open)

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
# Higher index = higher seniority. Order follows IUPAC (the same relative
# order as seniority.SENIORITY_ORDER): carboxylic_acid > nitrile > aldehyde >
# ketone > alcohol > amine. -01 (a phase) added 'nitrile' (between aldehyde
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
    carbox_pattern = _compiled_smarts('[CX3](=O)[OX2H1]')
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
    aldehyde_pattern = _compiled_smarts('[CX3H1](=O)')
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

    # --- 5. Detect -NH2 / -NHR attached to ring carbons (amine) [-01/] ---
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

    # --- 6. Detect exocyclic -C#N on ring carbons (nitrile) [-01] ---
    # Mirrors the aldehyde block: the nitrile carbon is exocyclic; the suffix
    # attaches to the ring carbon it is bonded to.
    nitrile_locants = []
    nitrile_pattern = _compiled_smarts('[CX2]#[NX1]')
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
    generate_polycyclic_name which only generates base names.

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
    all_heavy = {a.GetIdx() for a in atoms_of(mol)}
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
    # Check the EXACT cage atoms the descriptor numbers (: NOT the largest
    # ring component — a pendant aromatic ring never enters desc.numbering). If
    # the cage carries an aromatic atom, naming it here would silently
    # de-aromatise it into a WRONG saturated cage; refuse instead (raise the
    # named limit, caught at Orthonym.name). Raising (not returning None) is
    # required so the molecule fails closed rather than cascading to a fragment
    # namer that would name a single sub-ring ('cyclopentene' for benzonorbornadiene).
    # Phase G1 (DD7): a fused-aromatic core + bridge
    # (benzonorbornadiene-type) has a CORRECT bridged-fused PIN
    # (1,4-dihydro-1,4-methanonaphthalene); von Baeyer would de-aromatise it.
    # Try the constructor whenever the cage carries RDKit aromaticity OR
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
        # (a phase): a partially-saturated PAH (e.g. 9,10-dihydro-
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
                # substituents_included=True: the partial-saturation assembler
                # is now the SINGLE speller of every ring substituent prefix
                # multiplicity cannot be split across two formatters)
                # and it fails closed on any it cannot name, so there is nothing
                # left for the caller to enrich and enriching would double-cite.
                return (_sat_name, _sat_ring_atoms, _sat['atom_to_locant'], True)
        #: no smiles arg — Orthonym.name back-fills the original input SMILES.
        from ..errors import unsupported_ring_system
        raise unsupported_ring_system()

    # LEGALITY GATE . Everything below SPELLS `desc`: its
    # descriptor string, its numbering (stereo, substituent and heteroatom
    # locants) and its atom count all feed the name. A descriptor that does not
    # rebuild this cage names a DIFFERENT molecule, so refuse rather than emit.
    #
    # Placed after the bridged-fused and aromatic delegations above on purpose:
    # neither of those spells `desc`, and 15 corpus cages with an unverifiable
    # descriptor are named correctly by the partial-saturation constructor.
    #
    # Raising rather than returning None, for the same reason as the G0 refusal
    # directly above: None cascades to a fragment namer that would name one
    # sub-ring of the cage.
    if desc is None or desc.legality is not True:
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

    # 4. Get unsaturation.
    # macrocycle F2 (1)): a ring double bond whose two locants are NOT
    # consecutive (it spans a bridge, e.g. (15,36)) must be cited with the compound
    # locant ``15(36)``, not the bare ``min`` — the bare form makes OPSIN read the wrong
    # (15=16) bond and, when 16 is a carbonyl, raises ``C valency: 5`` (a cumulated
    # ketene). ``render_ring_unsaturation`` already produces the compound form (and is
    # the same primitive ``vonbaeyer_universal.analyze_cage_universal`` uses); route this
    # path through it instead of ``get_polycyclic_unsaturation``'s bare ``min(loc1,loc2)``.
    # Byte-identical for CONSECUTIVE ring enes (``_locant`` returns the bare int when
    # ``hi == lo + 1``), which is every ring ene in the PIN gold corpus.
    from .ring_unsaturation import render_ring_unsaturation
    # Restrict to ring_atoms exactly as get_polycyclic_unsaturation did (both endpoints
    # in the ring system), NOT render's numbering-only domain (a review RISK: numbering may
    # carry non-ring VB-framework atoms). Do NOT kekulize a fresh mol -- both renderers
    # skip AROMATIC bonds, so pass the same mol; kekulizing would fabricate ring enes.
    _ring_numbering = {a: loc for a, loc in desc.numbering.items() if a in ring_atoms}
    _ring_unsat = render_ring_unsaturation(mol, _ring_numbering)
    if _ring_unsat is None:
        # Non-citable (a non-consecutively-numbered yne) -> keep the legacy bare emission
        # (/gate-caught downstream; fail-closed, never a wrong-bond citation).
        unsaturation = get_polycyclic_unsaturation(mol, ring_atoms, desc.numbering)
    else:
        # STRING locants, already (1)-formatted and sorted by (lo,hi); pass
        # them through unchanged (``_build_parent_with_unsaturation`` str-formats and
        # must not re-sort them lexically).
        unsaturation = {
            'double_bonds': list(_ring_unsat.double_locants),
            'triple_bonds': list(_ring_unsat.triple_locants),
        }

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
    # The stereo prefix is applied LAST (see step 9) so it can be RT-gated at the
    # top level; here we build the constitution (stereo-free) name only.
    name_parts = []

    if substituent_prefix:
        # S1  — same stray-hyphen class: a substituent prefix attaches DIRECTLY
        # to the parent hydride /. Keep its trailing '-' ONLY when a
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

    # Join the constitution (stereo-free) name. The substituent prefix's trailing
    # '-' is handled above (kept before a locant, dropped before the descriptor).
    constitution_name = ''.join(name_parts)

    # 9. Apply stereo. Wave E — 0-wrong hardening: at the TOP LEVEL, route the
    # stereo through inject_stereo_reanchored_rt_gated, which RT-gates the numbering
    # and OMITS a stereo layer OPSIN cannot verify (rather than shipping a
    # pseudoasymmetric von-Baeyer descriptor like `(1r,5s)-` that does not
    # round-trip — a residual 0-wrong leak via the namer's OPSIN-validity stereo carve-out).
    # This is byte-identical for every currently-round-tripping stereo
    # name: candidate A reuses the same collect_stereodescriptors(mol, numbering)
    # this path always used (include_near_parent_ez is a no-op), so a name that
    # round-trips keeps its exact descriptor block. When naming a SUB-fragment
    # (not top level) the descriptor is applied directly as before — no per-fragment
    # OPSIN calls, and the whole-molecule gate downstream still applies.
    if stereo_prefix:
        from ..assembly.fragment_naming import is_top_level_naming
        if is_top_level_naming():
            from ..rules.stereochemistry import inject_stereo_reanchored_rt_gated
            name = inject_stereo_reanchored_rt_gated(
                constitution_name, mol, desc.numbering,
            )
        else:
            name = stereo_prefix + constitution_name
    else:
        name = constitution_name

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
        substituents: List of substituent dicts from get_polycyclic_substituents
        fg_prefixes: Optional list of FG prefix dicts with 'name' and 'locant' keys

    Returns:
        Formatted prefix like "5-hydroxy-3-ethyl-2,4-dimethyl-" or empty string
    """
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix

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
    # macrocycle F1 /: a COMPOUND substituent prefix (one that
    # carries an internal locant or is multi-token, e.g. the ylidene
    # ``3-methoxy-3-oxopropan-2-ylidene``) is cited in enclosing marks —
    # ``20-(3-methoxy-3-oxopropan-2-ylidene)`` — not bare; the bare form leaves OPSIN
    # unable to assign the substituent's internal locants. ``enclose_if_compound`` is the
    # shared primitive: it wraps iff compound/complex and leaves a SIMPLE prefix
    # (``methyl``, ``oxo``) bare (byte-identical), and the multiplier stays OUTSIDE the
    # marks (``bis(...)``) because it is emitted before the enclosed name. The alpha-sort
    # + multiplier are computed on the RAW name above, unchanged.
    from ..assembly.naming_utils import enclose_if_compound
    parts = []
    for name in sorted_names:
        locants = grouped[name]
        count = len(locants)
        multiplier = get_multiplier_prefix(count, name)

        locant_str = ','.join(str(loc) for loc in locants)
        from ..assembly.naming_utils import multiplied_component as _mc
        parts.append(f"{locant_str}-{_mc(count, name, enclose_if_compound(name))}")

    return '-'.join(parts) + '-' if parts else ""


def _needs_connective_a(double_bonds: List, triple_bonds: List) -> bool:
    """ /: does the ring/VB stem take a connective 'a'
    before the unsaturation endings?

    The connective 'a' is inserted before the FIRST cited unsaturation ending
    only when that ending is MULTIPLIED (i.e. begins with a consonant:
    ``dien-``, ``diyn-``, ``trien-``...). A single, vowel-initial ``en-``/
    ``yn-`` ending elides it. Endings are cited ene-before-yne, so the first
    ending is the ene block when any double bond is present, else the yne block.

        oct-3-en-7-yne (single en + single yn -> first ending 'en', elide)
        octa-3,7-diyne (diyne -> first ending 'diyn', retain)
        octa-1,3-dien-5-yne (dien + yn -> first ending 'dien', retain)
        oct-1-en-3,5-diyne (en + diyn -> first ending 'en', elide)

    The old test ``len(double_bonds) > 1 or (double_bonds and triple_bonds)``
    was wrong in two symmetric ways: it added 'a' for a single en + yn (case
    ``oct-1-en-3-yne``) and dropped it for a bare polyyne (case
    ``octa-1,3-diyne``)."""
    first_count = len(double_bonds) if double_bonds else len(triple_bonds)
    return first_count > 1


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

        if _needs_connective_a(double_bonds, triple_bonds):
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
            if _needs_connective_a(double_bonds, triple_bonds):
                stem = stem + 'a'
            parent_base = stem + ''.join(parts) + 'e'
        else:
            parent_base = base_name

        # Append suffix: "decane-1-carboxylic acid"
        #: elide multiplier-final 'a' before '-ol' (tetra+ol -> tetrol);
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

    if _needs_connective_a(double_bonds, triple_bonds):
        stem = stem + 'a'

    # "ELISION OF VOWELS", (a) (the Blue Book): "the
    # terminal letter 'e' in names of parent hydrides or endings 'ene' and
    # 'yne' when followed by a suffix or 'en' ending beginning with 'a', 'e',
    # 'i', 'o', 'u', or 'y'". Restated at:25013 -- "If, and only if, the
    # COMPLETE SUFFIX (that is, the suffix plus its multiplying prefixes, if
    # any) begins with a vowel, a terminal letter 'e' (if any) of the parent
    # hydride name is elided."
    #
    # This used to be decided here twice and wrongly. The unsaturated branch
    # did not decide at all -- `unsat_parts` hardcodes '-en'/'-yn', so the
    # terminal 'e' of the 'ene'/'yne' ending was dropped unconditionally and
    # 'oct-2-ene-4,8-dione' came out 'oct-2-en-4,8-dione'. The saturated
    # branch tested the BARE suffix ('one') rather than the complete one
    # ('dione'), so 'pentane-2,4-dione' came out 'pentan-2,4-dione'. Both are
    # consonant-initial complete suffixes, which RETAIN the 'e'.
    #
    # `format_suffix_with_locants` already owns exactly this decision for the
    # general chain engine -- it routes the multiplier through
    # `_join_multiplied_suffix` (tetra+ol -> tetrol, and then tests
    # the complete suffix against the shared `_ELISION_VOWELS`. Delegating to
    # it removes the second implementation rather than repairing it. The
    # unsaturation block is passed as part of the stem because it carries its
    # own locants ('-2,5-dien') and is not the plain 'an'/'en' infix the helper
    # names that parameter for.
    from ..assembly.naming_utils import format_suffix_with_locants

    fg_count = len(suffix_locants)
    fg_mult = fg_multipliers.get(fg_count, str(fg_count))

    if has_unsaturation:
        parent_stem = stem + ''.join(unsat_parts)
        unsat_infix = ''
    else:
        parent_stem = stem
        unsat_infix = 'an'

    return format_suffix_with_locants(
        parent_stem, unsat_infix, suffix_text, list(suffix_locants), fg_mult,
    )
