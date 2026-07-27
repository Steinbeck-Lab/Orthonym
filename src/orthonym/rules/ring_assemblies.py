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

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem import rdchem

from ..perception.rings import get_ring_info, is_aromatic_ring


# Phase 155.B D-09: detect indicated-H prefix in a ring stem name like
# "1H-indole" so name_ring_assembly can replicate the descriptor across
# all primed rings ("1H,1'H-2,2'-biindole") instead of leaving it embedded
# as a substring of a single ring name ("2,2'-bi1H-indole").
# Source: 155-CONTEXT.md D-09; HERITAGE-followups.md Follow-up 12 placement
#         subset; IUPAC P-31.1.4.
_INDICATED_H_RE = re.compile(r"^(\d+)H-(.*)$")

# Phase 155 WR-05: detect embedded P-31.1.4.4 indicated-H descriptors inside
# a ring stem name like "phenanthridin-6(5H)-one" or "acridin-9(10H)-one".
# The placement-replication logic in `_INDICATED_H_RE` only covers the
# leading "<n>H-" prefix form; embedded "(<n>H)" descriptors require a
# different per-ring threading strategy (deferred to Phase 151 Follow-up
# 12's deeper assembly-builder rewrite). Until that rewrite lands, ring
# assemblies whose component stem carries an embedded descriptor fail
# closed (return None) rather than emit the buggy mid-string `bi...`
# token (e.g., the pre-fix `2,2'-biphenanthridin-6(5H)-one` form).
# Source: 155-REVIEW.md WR-05; IUPAC P-31.1.4.4.
_INDICATED_H_EMBEDDED_RE = re.compile(r"\(\d+H\)")

# Phase 151-03 D-21 type alias: cascade-step-6 supplier returns int|tuple
# locants. Tuples are reserved for fusion-atom locants like (8, 'a');
# ring assemblies use plain ints because primes are name-format-layer only
# (per 151-PATTERNS.md Pattern S-3 and CONTEXT D-19/D-20).
_Locant = Union[int, Tuple[int, str]]


# Assembly multiplier prefixes (IUPAC P-28.2)
# Source: IUPAC Blue Book Table P-28.2, verified against OPSIN multipliers.xml
ASSEMBLY_MULTIPLIERS = {
    2: "bi",
    3: "ter",
    4: "quater",
    5: "quinque",
    6: "sexi",
    7: "septi",
    8: "octi",
    9: "novi",
    10: "deci",
    # P-28.5 / IUPAC multiplying affixes: ring assemblies of more than six (and
    # beyond the old deci cap) identical cyclic systems. OPSIN-verified that the
    # explicit-locant assembly with 'undeci' (11) round-trips. Counts above 12
    # keep the .get(count)->None decline (no OPSIN-verifiable affix -> fail closed).
    11: "undeci",
    12: "dodeci",
}


# IUPAC P-28.2.1: the multiplied parent-hydride name of a ring assembly is
# enclosed in parentheses "to avoid confusion with von Baeyer names". This is
# required for cycloalkane / cycloalkene monocycles (cyclopropane, cyclohexene,
# cyclopentylidene) and von Baeyer polycyclics (bicyclo[2.2.1]heptane,
# spiro[...]). Mancude rings (phenyl, pyridine, furan, naphthalene, indole) are
# NOT enclosed: 1,1'-biphenyl, 2,2'-bipyridine, 2,3'-bifuran, 1,2'-binaphthalene.
# Worked examples P-28.2.1; all PINs OPSIN-RT-verified (v22 G3 / COV-03).
_VON_BAEYER_NAME_RE = re.compile(
    r"^cyclo[a-z]+(?:ane|ene|yne|adiene|atriene|ylidene)$"
)


def _needs_von_baeyer_parens(name: str) -> bool:
    """True if a ring-assembly component name must be parenthesized (P-28.2.1).

    Triggers for cycloalkane / cycloalkene / ylidene monocycles and von Baeyer
    polycyclics (bicyclo[..]/tricyclo[..]/spiro[..]); mancude ring names (which
    never start with a 'cyclo' hydrocarbon stem) are left bare.
    """
    if not name:
        return False
    # von Baeyer descriptor: "...cyclo[<digit>..." or "spiro[<digit>...".
    if re.search(r"cyclo\[\d", name) or name.startswith("spiro["):
        return True
    # Monocyclic cycloalkane / cycloalkene / ylidene hydrocarbon stem.
    return bool(_VON_BAEYER_NAME_RE.match(name))


def _enclose_component(name: str) -> str:
    """Enclose a ring-assembly component name in parentheses when P-28.2.1
    requires it (von Baeyer disambiguation); otherwise return it unchanged."""
    return f"({name})" if _needs_von_baeyer_parens(name) else name


def _to_ylidene(name: str) -> Optional[str]:
    """Convert a saturated carbocycle parent-hydride name to its ylidene
    substituent-group name for a P-28.2.2 double-bond junction
    (cyclopentane -> cyclopentylidene). Returns None if ``name`` is not a clean
    'cyclo...ane' hydrocarbon ring, so the caller fails closed (no wrong names)."""
    if name and name.startswith("cyclo") and name.endswith("ane"):
        return name[:-3] + "ylidene"
    return None


def _is_saturated_carbocycle(mol, system_atoms: Set[int]) -> bool:
    """True if a ring system is a SINGLE all-carbon, non-aromatic ring with no
    internal ring double bond -- the monocyclic cycloalkane class nameable as a
    P-28.2.2 ylidene assembly (matching ``_to_ylidene``'s 'cyclo...ane'
    capability). Multi-ring saturated carbocycles (norbornane, decalin) are
    EXCLUDED so detect_ring_assembly does not claim a double-bond junction it
    cannot name -- otherwise namer.py early-returns on the assembly and the
    namer's None falls through to a fail-closed 'unknown'. A10: claim only what
    we can name; everything else keeps its prior (von Baeyer / substituent) path."""
    ri = mol.GetRingInfo()
    rings_in_system = [r for r in ri.AtomRings() if set(r) <= system_atoms]
    if len(rings_in_system) != 1:
        return False
    for i in system_atoms:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != "C" or a.GetIsAromatic():
            return False
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            a1 in system_atoms
            and a2 in system_atoms
            and bond.GetBondType() == rdchem.BondType.DOUBLE
        ):
            return False
    return True


def _system_signature(mol, system_atoms: Set[int]) -> Tuple:
    """
    Compute a ring system signature for identity comparison.

    The signature includes sorted element symbols, sorted ring sizes that
    are fully contained within the system, aromaticity of the system, and
    the intra-system double/triple bond counts. Two systems are identical
    if and only if their signatures match.

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in the ring system

    Returns:
        Tuple of (sorted_elements, sorted_ring_sizes, is_aromatic,
        double_bond_count, triple_bond_count)

    Note (W8-P11 leak fix): element/size/aromaticity alone cannot
    distinguish e.g. cyclooctane from cyclooctene (both all-carbon,
    ring_sizes=(8,), non-aromatic). Without a saturation check, a
    cyclooctene+cyclooctane pair joined by a single bond falsely compared
    EQUAL and was accepted as an identical-ring "bi-" assembly; the namer
    then applied the FIRST ring's (unsaturated) name to BOTH components,
    emitting e.g. "1,1'-bi(cyclooctene)" for a molecule where only one ring
    actually has the double bond -- a wrong-structure name that only
    OPSIN's SELF-01 self-consistency gate caught (which fails OPEN with no
    Java available). Counting intra-system double/triple bonds makes
    genuinely-identical rings compare equal (same connectivity -> same
    bond-order multiset) while rejecting rings that merely share element
    composition, size and aromaticity but differ in saturation.
    """
    elements = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in system_atoms)

    ri = mol.GetRingInfo()
    ring_sizes = []
    for ring in ri.AtomRings():
        if set(ring) <= system_atoms:
            ring_sizes.append(len(ring))

    is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in system_atoms)

    double_bonds = 0
    triple_bonds = 0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in system_atoms and a2 in system_atoms:
            bt = bond.GetBondType()
            if bt == rdchem.BondType.DOUBLE:
                double_bonds += 1
            elif bt == rdchem.BondType.TRIPLE:
                triple_bonds += 1

    return (tuple(elements), tuple(sorted(ring_sizes)), is_aromatic,
            double_bonds, triple_bonds)


# P-28.4.2 skeletal-replacement ring-assembly heteroatoms (element-seniority
# citation order O > S > Se > Te; reuse the 'a'-terms from skeletal_replacement).
_RA_REPL_TERMS = {'O': 'oxa', 'S': 'thia', 'Se': 'selena', 'Te': 'tellura'}
_RA_REPL_ORDER = ['O', 'S', 'Se', 'Te']  # element seniority (P-28.4.2 set order)


def _is_replacement_assembly_candidate(
    mol, ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]]
) -> bool:
    """P-28.4.2 gate: exactly two SAME-SIZE saturated monocyclic components,
    each all-carbon except AT MOST one neutral divalent ring heteroatom from
    {O,S,Se,Te}, joined by exactly one ring-to-ring SINGLE bond. At least one
    component must carry a heteroatom (else it is a plain carbocyclic assembly
    handled elsewhere). Fail-closed for anything richer."""
    if len(ring_systems) != 2 or len(connections) != 1:
        return False
    a1, a2, _, _ = connections[0]
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is None or bond.GetBondType() != rdchem.BondType.SINGLE:
        return False
    ri = mol.GetRingInfo()
    sizes = []
    total_hetero = 0
    for sys_atoms in ring_systems:
        rings = [r for r in ri.AtomRings() if set(r) <= sys_atoms]
        if len(rings) != 1:
            return False  # multi-ring component -> not this class
        ring = rings[0]
        if set(ring) != set(sys_atoms):
            return False  # exocyclic atoms in the system -> not a clean monocycle
        sizes.append(len(ring))
        het = 0
        for i in ring:
            a = mol.GetAtomWithIdx(i)
            sym = a.GetSymbol()
            if sym == 'C':
                if a.GetIsAromatic():
                    return False
                continue
            if (sym not in _RA_REPL_TERMS or a.GetFormalCharge() != 0
                    or a.GetDegree() != 2 or a.GetIsAromatic()):
                return False
            het += 1
        if het > 1:
            return False  # >1 heteroatom per ring -> outside this narrow class
        total_hetero += het
        # No ring double bonds (saturated replacement ring).
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in sys_atoms and j in sys_atoms and \
                    b.GetBondType() == rdchem.BondType.DOUBLE:
                return False
    if sizes[0] != sizes[1] or total_hetero == 0:
        return False
    # No substituents beyond the inter-ring bond (each ring atom degree <=2
    # except the two attachment atoms which are degree 3 via the junction).
    attach = {a1, a2}
    for sys_atoms in ring_systems:
        for i in sys_atoms:
            deg = mol.GetAtomWithIdx(i).GetDegree()
            if i in attach:
                if deg != 3:
                    return False
            elif deg != 2:
                return False
    return True


def _name_replacement_ring_assembly(mol, assembly_info: Dict) -> Optional[str]:
    """P-28.4.2: name a two-component same-skeleton monocyclic ring assembly by
    skeletal-replacement ('a') nomenclature. Each ring is numbered from its
    attachment atom (locant 1); the heteroatom takes the lower of the two ring
    directions. Across the two rings, the unprimed/primed assignment is the one
    giving the LOWEST COMBINED heteroatom locant set; 'a'-prefixes are cited in
    element-seniority order (O>S>Se>Te). Emits e.g.
    '3'-oxa-2-thia-1,1'-bi(cyclotetradecane)'. Fail-closed on any ambiguity."""
    from ..data.chain_names import get_chain_prefix
    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    a1, a2, s1, s2 = connections[0]
    ri = mol.GetRingInfo()

    def _component(sys_atoms, attach):
        """Return (ring_size, hetero_locant_or_None, hetero_symbol_or_None):
        number the monocycle from ``attach`` (=1), choosing the direction that
        gives the single heteroatom (if any) the lowest locant."""
        ring = next(r for r in ri.AtomRings() if set(r) <= sys_atoms)
        n = len(ring)
        # Build the two cyclic orderings starting at `attach`.
        # Neighbours of attach within the ring:
        nbrs = [x.GetIdx() for x in mol.GetAtomWithIdx(attach).GetNeighbors()
                if x.GetIdx() in sys_atoms]
        if len(nbrs) != 2:
            return None
        best = None  # (hetero_locant or n+1, ordering)
        for start_nbr in nbrs:
            order = [attach, start_nbr]
            prev, cur = attach, start_nbr
            while len(order) < n:
                nxt = next(
                    (x.GetIdx() for x in mol.GetAtomWithIdx(cur).GetNeighbors()
                     if x.GetIdx() in sys_atoms and x.GetIdx() != prev),
                    None)
                if nxt is None:
                    break
                order.append(nxt)
                prev, cur = cur, nxt
            if len(order) != n:
                continue
            het_loc = None
            het_sym = None
            for pos, idx in enumerate(order, start=1):
                if mol.GetAtomWithIdx(idx).GetSymbol() != 'C':
                    het_loc = pos
                    het_sym = mol.GetAtomWithIdx(idx).GetSymbol()
                    break
            key = het_loc if het_loc is not None else n + 1
            if best is None or key < best[0]:
                best = (key, het_loc, het_sym)
        if best is None:
            return None
        return (n, best[1], best[2])

    c1 = _component(ring_systems[s1], a1)
    c2 = _component(ring_systems[s2], a2)
    if c1 is None or c2 is None:
        return None
    n1, loc1, sym1 = c1
    n2, loc2, sym2 = c2
    if n1 != n2:
        return None

    # Assign which ring is unprimed to minimise the combined heteroatom locant
    # set (P-28.4.2: low locants to heteroatoms as a set). Each ring contributes
    # at most one heteroatom; compare the two orientations.
    def _combined(order):
        # order = list of (loc, sym, is_primed) sorted by (loc, prime)
        out = []
        for loc, sym, primed in order:
            if loc is None:
                continue
            out.append((loc, primed))
        return sorted(out)

    # Orientation A: ring1 unprimed, ring2 primed.
    optA = [(loc1, sym1, False), (loc2, sym2, True)]
    # Orientation B: ring2 unprimed, ring1 primed.
    optB = [(loc2, sym2, False), (loc1, sym1, True)]
    setA = _combined(optA)
    setB = _combined(optB)
    chosen = optA if setA <= setB else optB

    # Cite 'a'-prefixes in element-seniority order (O>S>Se>Te), each with its
    # (possibly primed) locant.
    cited = []
    for loc, sym, primed in chosen:
        if loc is None or sym is None:
            continue
        term = _RA_REPL_TERMS.get(sym)
        if term is None:
            return None
        prime = "'" if primed else ""
        cited.append((_RA_REPL_ORDER.index(sym), f"{loc}{prime}-{term}"))
    if not cited:
        return None
    cited.sort(key=lambda x: x[0])
    prefix = "-".join(part for _, part in cited)

    stem = get_chain_prefix(n1)  # e.g. 'tetradec' for 14
    # Both attachment atoms are locant 1 in their rings -> '1,1'-bi(cyclo...ane)'.
    return f"{prefix}-1,1'-bi(cyclo{stem}ane)"


def _find_inter_system_bonds(
    mol, ring_systems: List[Set[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Find all bonds connecting different ring systems.

    Returns list of (atom_idx_A, atom_idx_B, system_idx_A, system_idx_B).
    SINGLE and AROMATIC bonds are the single-bond junction (P-28.2.1; biphenyl's
    inter-ring bond may be typed as either depending on Kekulization). DOUBLE
    bonds are accepted for the P-28.2.2 double-bond junction (bi(...ylidene));
    detect_ring_assembly gates which double-bond junctions are actually claimed.
    """
    connections = []
    acceptable_types = {
        rdchem.BondType.SINGLE,
        rdchem.BondType.AROMATIC,
        rdchem.BondType.DOUBLE,
    }

    # Build atom -> set of system indices (an atom can be in multiple systems
    # for spiro compounds where the spiro center is shared)
    atom_to_systems: Dict[int, Set[int]] = {}
    for sys_idx, system in enumerate(ring_systems):
        for atom_idx in system:
            if atom_idx not in atom_to_systems:
                atom_to_systems[atom_idx] = set()
            atom_to_systems[atom_idx].add(sys_idx)

    # Atoms in multiple systems (spiro centers) -- skip bonds involving these
    shared_atoms = {idx for idx, systems in atom_to_systems.items() if len(systems) > 1}

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # Skip bonds involving spiro/shared atoms
        if a1 in shared_atoms or a2 in shared_atoms:
            continue

        # Both atoms must be in ring systems, but different ones
        if a1 in atom_to_systems and a2 in atom_to_systems:
            systems1 = atom_to_systems[a1]
            systems2 = atom_to_systems[a2]
            # Each should be in exactly one system for a true assembly bond
            if len(systems1) == 1 and len(systems2) == 1:
                s1 = next(iter(systems1))
                s2 = next(iter(systems2))
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


def _check_path_topology(
    num_systems: int, connections: List[Tuple[int, int, int, int]]
) -> bool:
    """Phase 151-03 D-15: linear-path requirement for ring assemblies.

    Every system must have degree <= 2 in the inter-system bond graph.
    Branched arrangements (e.g., 1,3,5-triphenylbenzene where the central
    benzene has degree 3) are NOT ring assemblies per IUPAC P-28.2; they
    fall through to substituent-based naming.

    Args:
        num_systems: number of ring systems.
        connections: list of (atom_a, atom_b, system_a, system_b) tuples
            from ``_find_inter_system_bonds``.

    Returns:
        True iff every system index appears at most twice across
        ``connections`` (a linear path / single bond ring assembly);
        False otherwise.

    Source: 151-CONTEXT.md D-15.
    Source: 151-RESEARCH.md §"Code Examples" Example 4.
    Source: HERITAGE-1990 §5; IUPAC Blue Book P-28.2.
    """
    if num_systems <= 1:
        return True
    deg: Counter = Counter()
    for _, _, s1, s2 in connections:
        deg[s1] += 1
        deg[s2] += 1
    if not all(d <= 2 for d in deg.values()):
        return False
    # Phase 151-04 WR-02: reject cyclic ring-system arrangements.
    # A linear path of N systems has exactly N-1 inter-system bonds.
    # A cycle has N (every node degree 2 — passes the degree check).
    # IUPAC P-28.2 requires a linear path; a cyclic arrangement is a
    # different topology (would be a fused/bridged macrocyclic system).
    if len(connections) != num_systems - 1:
        return False
    return True


def _order_systems_along_path(
    num_systems: int,
    connections: List[Tuple[int, int, int, int]],
) -> Optional[List[int]]:
    """Phase 151-04 BLK-02 Scenario A root-cause fix.

    Given a linear-path inter-system bond graph (verified by
    ``_check_path_topology``), return the system indices in path order
    starting from one terminal end.

    For a path A—B—C, returns ``[A, B, C]`` where A and C are the
    terminal systems (degree 1 in the inter-system graph) and B is the
    middle system (degree 2). This is the canonical IUPAC P-28.2.1
    ordering: terminal rings get unprimed and double-primed namespaces,
    middle rings get the single-prime namespace.

    Without this reordering, ``get_ring_systems`` returns ring systems
    in atom-traversal order — the middle ring of a substituted terphenyl
    can land at index 0, causing the connection-string emission to drop
    the prime on the middle ring's back-attachment locant
    (e.g., emitting ``1,1':4,1''-terphenyl`` instead of the canonical
    ``1,1':4',1''-terphenyl``). This was the BLK-02 defect captured at
    151-04-DIAGNOSTIC.md.

    Args:
        num_systems: number of ring systems.
        connections: list of (atom_a, atom_b, system_a, system_b) tuples
            from ``_find_inter_system_bonds``.

    Returns:
        List of system indices in path order, or None if a unique linear
        ordering cannot be determined (caller should fall through).

    Source: IUPAC Blue Book P-28.2.1 (single-prime namespace = middle ring).
    Source: 151-04-PLAN.md Task 3 Scenario A.
    """
    if num_systems <= 1:
        return list(range(num_systems))

    # Build adjacency map (system -> set of neighbor systems)
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_systems)}
    for _, _, s1, s2 in connections:
        adj[s1].add(s2)
        adj[s2].add(s1)

    # Tree-shape gate: linear path => exactly two terminals (degree 1).
    # If 0 terminals, graph is a cycle (rejected by _check_path_topology
    # tree-shape gate). If >2 terminals, branched (also rejected).
    terminals = [i for i, neighbors in adj.items() if len(neighbors) == 1]
    if len(terminals) != 2:
        return None

    # Pick the lower-indexed terminal as the start to make the ordering
    # deterministic. Walking from the OTHER terminal would give the
    # reversed path, but per IUPAC P-28.2.1 lowest-locant the symmetric
    # case is handled downstream by compare_locant_sets in
    # _compute_per_system_ring_locants.
    start = min(terminals)

    order: List[int] = [start]
    seen: Set[int] = {start}
    current = start
    while len(order) < num_systems:
        next_neighbors = adj[current] - seen
        if not next_neighbors:
            # Disconnected — should not happen given _check_all_connected
            return None
        # Linear-path invariant: at most one unseen neighbor.
        if len(next_neighbors) > 1:
            return None
        nxt = next(iter(next_neighbors))
        order.append(nxt)
        seen.add(nxt)
        current = nxt

    return order


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

    Phase 154.B D-11 cross-handler contract:
        Ring-assembly detection (this function) and multiplicative naming
        (rules.multiplicative.name_multiplicative) are MUTUALLY EXCLUSIVE
        by topology.  Ring assemblies = identical rings joined directly by
        a single bond (no bridge atom).  Multiplicative = identical parent
        units joined by 1+ bridge atoms (oxy / methylene / nitrilo / ...).
        The split is enforced symmetrically:
          - this function rejects atom-bridged cases via
            _find_inter_system_bonds (which only matches ring-to-ring
            single bonds; bridge atoms are NOT in rings, so atom-bridged
            cases never produce inter-system bonds)
          - name_multiplicative rejects single-bond-only cases via
            _is_pure_single_bond_assembly at the entry point (D-11 guard)
        Cross-handler regression test:
        tests/integration/test_assembly_vs_multiplicative_dispatch.py.

    Source: 154-CONTEXT.md D-11; 151-CONTEXT.md D-15 (path-topology contract).
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

    # Phase 151-03 D-15: linear-path requirement (every system degree <= 2).
    # Rejects branched arrangements like 1,3,5-triphenylbenzene where the
    # central system has degree 3. HERITAGE-1990 §5 + IUPAC P-28.2.
    if not _check_path_topology(len(ring_systems), connections):
        return None

    # Compare signatures -- all must be identical
    signatures = [_system_signature(mol, sys_atoms) for sys_atoms in ring_systems]
    if len(set(signatures)) != 1:
        # P-28.4.2: ring-assembly components need NOT be identical when they
        # share the same ring SKELETON and differ only by skeletal-replacement
        # heteroatoms; then 'a'-nomenclature names the assembly
        # (3'-oxa-2-thia-1,1'-bi(cyclotetradecane)). Admit ONLY that narrow
        # class (2 same-size single-heteroatom saturated monocycles, one
        # ring-to-ring single-bond junction) tagged for the replacement namer;
        # everything else keeps failing closed.
        if _is_replacement_assembly_candidate(mol, ring_systems, connections):
            return {
                'ring_systems': ring_systems,
                'connections': connections,
                'count': len(ring_systems),
                'ring_type': 'heterocyclic',
                'double_bond_junction': False,
                'replacement': True,
            }
        return None

    # IUPAC P-28.2.2: a junction may be a DOUBLE bond (bi(...ylidene)).
    # _find_inter_system_bonds now also returns double-bond junctions; restrict
    # the ones this detector CLAIMS to the saturated-carbocycle class that the
    # ylidene namer can build. Other double-bond junctions return None here so
    # the molecule keeps its prior (substituent-based) naming instead of
    # early-returning in namer.py to a fail-closed dead end (A10, no wrong names).
    double_bond_junction = any(
        (bond := mol.GetBondBetweenAtoms(a1, a2)) is not None
        and bond.GetBondType() == rdchem.BondType.DOUBLE
        for a1, a2, _, _ in connections
    )
    if double_bond_junction and not all(
        _is_saturated_carbocycle(mol, sys_atoms) for sys_atoms in ring_systems
    ):
        return None

    # Additional guard: reject if there are non-ring atoms in the molecule
    # other than substituents (i.e., linker atoms between rings).
    # For true ring assemblies, the inter-ring bond is direct (no linker).
    # This is already ensured by _find_inter_system_bonds checking that
    # both atoms are IN ring systems.

    # Phase 151-04 BLK-02 Scenario A root-cause fix: reorder ring_systems
    # along the inter-system path so the middle ring of a ter-/quater-/
    # quinque- assembly lands at the correct primed-namespace index.
    # Without this reordering, ``get_ring_systems`` returns systems in
    # atom-traversal order — for substituted ring assemblies the middle
    # ring can land at index 0, causing the connection-string to drop
    # the prime on the middle ring's back-attachment locant
    # (emitting ``1,1':4,1''-terphenyl`` instead of canonical
    # ``1,1':4',1''-terphenyl``). See 151-04-DIAGNOSTIC.md.
    path_order = _order_systems_along_path(len(ring_systems), connections)
    if path_order is not None and path_order != list(range(len(ring_systems))):
        # Build remapping: old_idx -> new_idx
        new_index_of = {old: new for new, old in enumerate(path_order)}
        ring_systems = [ring_systems[old] for old in path_order]
        connections = [
            (a1, a2, new_index_of[s1], new_index_of[s2])
            for a1, a2, s1, s2 in connections
        ]

    # Determine ring type
    elements = signatures[0][0]
    has_heteroatom = any(e != 'C' for e in elements)
    ring_type = 'heterocyclic' if has_heteroatom else 'carbocyclic'

    return {
        'ring_systems': ring_systems,
        'connections': connections,
        'count': len(ring_systems),
        'ring_type': ring_type,
        'double_bond_junction': double_bond_junction,
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
            # Use the heterocycle naming infrastructure.
            from .heterocycles import name_heterocycle
            # P-28.2.1 / P-31.1.7.3: the component name is the free PARENT
            # HYDRIDE name of the ring, so its indicated hydrogen must be
            # computed on the ISOLATED ring — not on the ring embedded in the
            # assembly, where the inter-ring bond at the connection atom
            # suppresses the indicated-H (e.g. the assembly's ring gave
            # '1,4-oxaphosphinine' with no '4H', but the parent hydride is
            # '4H-1,4-oxaphosphinine'). Extract the ring as a standalone mol,
            # name it there, then let the assembly composer add the connection
            # locant + enclosing parentheses. Fail-safe: if the isolated
            # fragment cannot be built or named, fall back to the embedded name.
            try:
                frag_smi = Chem.MolFragmentToSmiles(
                    mol, atomsToUse=sorted(ring), canonical=True)
                frag = Chem.MolFromSmiles(frag_smi) if frag_smi else None
            except Exception:
                frag = None
            if frag is not None:
                frag_rings = [r for r in frag.GetRingInfo().AtomRings()
                              if len(r) == ring_size]
                if len(frag_rings) == 1:
                    iso = name_heterocycle(frag, frag_rings[0])
                    if iso:
                        return iso
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

    # Multi-ring fused systems as assembly units (e.g., binaphthalene, biquinoline)
    # Per IUPAC P-28.1: fused ring systems can serve as identical ring assembly units.
    # Use existing infrastructure: dictionary fast path (150+ entries), then algorithmic fallback.

    # Step 1: Extract subsystem SMILES for lookup
    system_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(system_atoms), canonical=True)
    if not system_smi:
        return None

    # Step 2: Create standalone mol for canonical SMILES matching
    sub_mol = Chem.MolFromSmiles(system_smi)
    if sub_mol is None:
        return None
    can_smi = Chem.MolToSmiles(sub_mol, canonical=True)

    # Step 3: Try fused heterocycle dictionary (fast path, 150+ entries from Phase 109)
    from ..data.fused_heterocycles import get_fused_heterocycle_name
    fused_result = get_fused_heterocycle_name(sub_mol)
    if fused_result:
        name, _tautomer_locant = fused_result
        # Include indicated H prefix if present (e.g., "1H-indole")
        return name

    # Step 4: Try retained names for carbocyclic fused systems (naphthalene, anthracene, etc.)
    from ..data.retained_names import get_retained_name
    retained = get_retained_name(can_smi)
    if retained:
        return retained

    # Step 5: Try algorithmic fused ring generator (Phase 112 fallback)
    from .fused_rings import _try_algorithmic_fusion_name
    algo_name = _try_algorithmic_fusion_name(sub_mol)
    if algo_name:
        return algo_name

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


def _number_carbocyclic_from_anchor(
    mol, system_atoms: Set[int], anchor_atom: int,
    other_inter_system_atoms: Set[int],
) -> Optional[Dict[int, int]]:
    """Phase 151-03 D-18 / D-19: per-ring numbering for a carbocyclic
    assembly component anchored at ``anchor_atom`` (which becomes locant 1).

    Numbers the ring atoms in the direction that gives the lowest locant
    set for the OTHER inter-system bond atoms (D-19 first-point-of-difference
    via ``compare_locant_sets``). Used by ``_compute_per_system_ring_locants``
    to produce the per-system IUPAC locant maps that drive name emission
    AND the cascade-step-6 supplier.

    Args:
        mol: RDKit Mol object.
        system_atoms: atoms forming this ring system.
        anchor_atom: atom that becomes locant 1 (the inter-system bond
            attaching this ring to the previous chain ring; for terminal
            rings, the only inter-system bond atom).
        other_inter_system_atoms: other inter-system bond atoms in this
            same ring system (e.g., for the middle ring of terphenyl,
            this is the singleton {atom-bonded-to-ring-2}).

    Returns:
        Dict mapping atom_idx -> int locant covering all ring atoms in
        the (single-)ring system. Returns None for multi-ring systems
        (those use absolute numbering via _get_ring_parent_name). For
        single-ring systems, the lowest-locant direction wins per D-19.

    Source: 151-CONTEXT.md D-18, D-19, D-20.
    Source: HERITAGE-1990 §3 (criterion order — lowest-locant tiebreak).
    Source: IUPAC Blue Book P-28.2.1.
    """
    from .locants import compare_locant_sets

    ri = mol.GetRingInfo()
    rings = [ring for ring in ri.AtomRings() if set(ring) <= system_atoms]
    if len(rings) != 1:
        return None  # multi-ring fused systems use absolute numbering
    ring_list = list(rings[0])
    n = len(ring_list)
    if anchor_atom not in ring_list:
        return None
    start_pos = ring_list.index(anchor_atom)

    best_locants: Optional[Dict[int, int]] = None
    best_other_set: Optional[List[int]] = None

    for direction in (1, -1):
        # Build oriented sequence: anchor_atom -> locant 1, walking direction
        oriented = [
            ring_list[(start_pos + direction * i) % n] for i in range(n)
        ]
        atom_to_locant = {a: i + 1 for i, a in enumerate(oriented)}
        if not other_inter_system_atoms:
            # Terminal ring: any direction equally valid; pick deterministically
            # (direction +1 first; later substituent tiebreak handled elsewhere)
            return atom_to_locant
        other_set = sorted(
            atom_to_locant[a]
            for a in other_inter_system_atoms
            if a in atom_to_locant
        )
        if not other_set:
            continue
        if best_other_set is None or compare_locant_sets(
            other_set, best_other_set
        ) < 0:
            best_other_set = other_set
            best_locants = atom_to_locant

    return best_locants


def _compute_per_system_ring_locants(
    mol, assembly_info: Dict,
) -> Optional[List[Dict[int, int]]]:
    """Phase 151-03 D-18 / D-21: per-system IUPAC locant maps for a ring
    assembly. Each entry is a Dict[int, int] mapping atom_idx -> locant
    (within that system's own numbering).

    For carbocyclic single-ring components: locant 1 = the inter-system
    bond atom going TOWARDS the lower-indexed neighbour ring; the second
    bond atom (if present) gets the lowest-locant ring-walk distance per
    D-19 ``compare_locant_sets`` tiebreak.

    For heterocyclic components: heteroatom-priority numbering via
    ``_get_connection_locant`` (existing logic, D-18 own-numbering).

    For multi-ring fused components (e.g., biindole): absolute numbering
    by per-component canonical SMILES lookup is delegated to
    ``_get_connection_locant`` (which falls through to fused-heterocycle
    catalog via ``name_heterocycle``).

    Args:
        mol: RDKit Mol object.
        assembly_info: dict from detect_ring_assembly.

    Returns:
        List of per-system Dict[int, int] locant maps, indexed by system
        position (matching assembly_info['ring_systems'] order). Returns
        None if any system cannot be numbered fully.

    Source: 151-CONTEXT.md D-18, D-19, D-21.
    Source: 151-PATTERNS.md Pattern S-3 (cascade-step-6 supplier).
    """
    ring_systems = assembly_info["ring_systems"]
    connections = assembly_info["connections"]

    # Build per-system bond inventory: for each system idx, the list of
    # (own_atom, other_system_idx) pairs. The "anchor" for numbering is
    # the bond going TOWARDS the lower-indexed neighbour system; for the
    # leftmost terminal (sys 0), this is just its single bond.
    bonds_per_sys: Dict[int, List[Tuple[int, int]]] = {
        i: [] for i in range(len(ring_systems))
    }
    for a1, a2, s1, s2 in connections:
        bonds_per_sys[s1].append((a1, s2))
        bonds_per_sys[s2].append((a2, s1))

    result: List[Dict[int, int]] = []
    for sys_idx, sys_atoms in enumerate(ring_systems):
        my_bonds = bonds_per_sys[sys_idx]
        # Pick anchor = bond to the lower-indexed neighbour system.
        # If ``sys_idx == 0`` there is no lower neighbour — pick its
        # single bond's own-atom as the anchor.
        anchor_atom: Optional[int] = None
        other_atoms: Set[int] = set()
        for own_atom, other_sys in my_bonds:
            if other_sys < sys_idx:
                anchor_atom = own_atom
            else:
                other_atoms.add(own_atom)
        if anchor_atom is None and my_bonds:
            # Sys 0 (no lower neighbour): use its only own-atom as anchor.
            anchor_atom = my_bonds[0][0]
            other_atoms = {a for a, _ in my_bonds[1:]}

        ri = mol.GetRingInfo()
        sys_rings = [r for r in ri.AtomRings() if set(r) <= sys_atoms]

        # Determine if single-ring carbocyclic vs everything else.
        if len(sys_rings) == 1 and anchor_atom is not None:
            ring = sys_rings[0]
            has_hetero = any(
                mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring
            )
            if not has_hetero:
                # Carbocyclic single ring: number from anchor with D-19
                # lowest-locant tiebreak on other_atoms.
                m = _number_carbocyclic_from_anchor(
                    mol, sys_atoms, anchor_atom, other_atoms
                )
                if m is not None and set(m.keys()) == set(ring):
                    result.append(m)
                    continue

        # Heterocyclic single-ring: number the ring with heteroatom-priority
        # IUPAC P-25 absolute numbering (heteroatom = locant 1). Walking
        # direction is chosen to give the LOWEST locant set to the
        # inter-system bond atoms (P-28.2.1 connection-locant lowest-locant
        # rule + D-19 first-point-of-difference via compare_locant_sets).
        sys_map: Dict[int, int] = {}
        if len(sys_rings) == 1:
            ring = sys_rings[0]
            n = len(ring)
            ring_list = list(ring)
            from .heterocycles import classify_heterocycle
            from ..data.hw_heteroatoms import get_heteroatom_priority
            from .locants import compare_locant_sets

            try:
                info_het = classify_heterocycle(mol, ring_list)
                heteroatoms = info_het.get("heteroatoms", [])
            except Exception:
                heteroatoms = []

            if heteroatoms:
                sorted_ha = sorted(
                    heteroatoms,
                    key=lambda x: get_heteroatom_priority(x[1]),
                )
                start_idx = sorted_ha[0][0]
                start_pos = ring_list.index(start_idx)
                conn_atoms = (
                    ([anchor_atom] if anchor_atom is not None else [])
                    + sorted(other_atoms)
                )
                best_map: Optional[Dict[int, int]] = None
                best_locant_set: Optional[List[int]] = None
                for direction in (1, -1):
                    oriented = [
                        ring_list[(start_pos + direction * i) % n]
                        for i in range(n)
                    ]
                    cand_map = {a: i + 1 for i, a in enumerate(oriented)}
                    cand_set = sorted(cand_map[a] for a in conn_atoms if a in cand_map)
                    if not cand_set:
                        continue
                    if best_locant_set is None or compare_locant_sets(
                        cand_set, best_locant_set
                    ) < 0:
                        best_locant_set = cand_set
                        best_map = cand_map
                if best_map is not None:
                    sys_map = best_map
            if not sys_map:
                # No heteroatoms classified or numbering failed — fall
                # back to anchor-based carbocyclic numbering.
                if anchor_atom is not None and anchor_atom in ring:
                    fallback = _number_carbocyclic_from_anchor(
                        mol, sys_atoms, anchor_atom, other_atoms
                    )
                    if fallback is not None:
                        sys_map = fallback
            result.append(sys_map)
        else:
            # Multi-ring fused: defer to absolute numbering via
            # _get_connection_locant per atom. Coverage may be partial
            # for some catalogue entries; the supplier checks coverage
            # downstream and returns None if incomplete.
            for atom_idx in sys_atoms:
                loc = _get_connection_locant(mol, atom_idx, sys_atoms)
                sys_map[atom_idx] = loc
            result.append(sys_map)

    return result


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
        """Walk ring from start through direction_neighbor, return ordered atoms.

        Only step to UNVISITED neighbours: the old ``or n == start`` clause let
        the walk pick the start atom back as ``nexts[0]`` on the second step and
        terminate prematurely, so the short path to a META substituent was never
        traversed (it returned the long-way locant 5 instead of the correct 3
        for '[1,1'-biphenyl]-3-…'). The loop now enumerates every ring atom once
        in ring order and stops when no unvisited neighbour remains."""
        visited = [start]
        current = direction_neighbor
        while current != start:
            visited.append(current)
            nexts = [n for n in adj[current] if n not in visited]
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


def _reassign_carbocyclic_locants(mol, ring_systems, connections, substituents):
    """P-14.4: number each CARBOCYCLIC ring system ONCE (connection atom = 1),
    choosing the single direction giving the lowest locant SET to ALL its
    substituents together.

    :func:`_get_substituent_locant` picks each substituent's own min-locant
    direction independently; when two substituents sit on the SAME ring that
    collides (3,5-disubstituted benzene ring -> each min()s to 3 -> the invalid
    '3,3' set, which OPSIN rejects). Choosing one shared direction per ring makes
    the set '3,5'. Heterocyclic rings keep their fixed orient_heterocycle
    numbering (handled in _get_substituent_locant) and are skipped here."""
    ri = mol.GetRingInfo()
    by_sys: Dict[int, List[Dict]] = {}
    for s in substituents:
        by_sys.setdefault(s['system_idx'], []).append(s)
    for sys_idx, subs in by_sys.items():
        if len(subs) < 2:
            continue  # single substituent: per-atom min is already correct
        sys_atoms = ring_systems[sys_idx]
        conn = None
        for a1, a2, s1, s2 in connections:
            if s1 == sys_idx:
                conn = a1
                break
            if s2 == sys_idx:
                conn = a2
                break
        if conn is None:
            continue
        ring = None
        for r in ri.AtomRings():
            if conn in r and set(r) <= sys_atoms:
                ring = r
                break
        if ring is None:
            continue
        # carbocyclic only (heteroatom rings keep orient_heterocycle numbering)
        if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
            continue
        rset = set(ring)
        adj: Dict[int, List[int]] = {i: [] for i in ring}
        for b in mol.GetBonds():
            x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if x in rset and y in rset:
                adj[x].append(y)
                adj[y].append(x)

        def _positions(first):
            visited = [conn]
            cur = first
            while cur != conn:
                visited.append(cur)
                nxt = [n for n in adj[cur] if n not in visited]
                if not nxt:
                    break
                cur = nxt[0]
            return {a: i + 1 for i, a in enumerate(visited)}

        nbrs = adj.get(conn, [])
        if len(nbrs) < 2:
            continue
        pa = _positions(nbrs[0])
        pb = _positions(nbrs[1])
        big = len(ring) + 1
        set_a = sorted(pa.get(s['ring_atom'], big) for s in subs)
        set_b = sorted(pb.get(s['ring_atom'], big) for s in subs)
        chosen = pa if set_a <= set_b else pb
        for s in subs:
            if s['ring_atom'] in chosen:
                s['locant'] = chosen[s['ring_atom']]


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

    # P-14.4 set-lowest numbering for rings bearing >=2 substituents (fixes the
    # per-substituent min() collision, e.g. 3,5-disubstituted -> '3,3').
    _reassign_carbocyclic_locants(mol, ring_systems, connections, substituents)
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

    # P-29.2 free-valence gate. Every route below names the fragment from its
    # ATOM COUNT, which cannot distinguish -CH3 from =CH2:
    # 'C=C1CCC(C2CCCCC2)CC1' came out as 4-methyl-1,1'-bi(cyclohexane), a
    # different molecule. Shared primitive, same three-way contract as the two
    # general chokepoints; a single bond defers and changes nothing.
    from ..assembly.substituent_enumerator import carbon_free_valence_prefix

    _fv = carbon_free_valence_prefix(mol, sub_atoms, attachment_atom)
    if _fv.prefix is not None:
        return _fv.prefix
    if _fv.must_fail_closed:
        return None

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
            # Try recursive naming (handles retained names + branched subs)
            if len(sub_atoms) > 1:
                from ..assembly.substituent_naming import name_substituent_fragment
                rec_name = name_substituent_fragment(
                    mol, sub_atoms, sub_atoms[0], []
                )
                if rec_name:
                    return rec_name
            try:
                return get_alkyl_name(n_carbons)
            except (ValueError, KeyError):
                prefix = get_chain_prefix(n_carbons)
                return f"{prefix}yl"

    # Check for alkoxy groups (O + alkyl chain)
    if sub_atoms:
        first_atom = mol.GetAtomWithIdx(attachment_atom)
        if first_atom.GetSymbol() == 'O' and attachment_atom in sub_atoms:
            other_atoms = [i for i in sub_atoms if i != attachment_atom]
            carbon_count = sum(
                1 for i in other_atoms
                if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
            )
            all_simple = all(
                mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                for i in other_atoms
            )
            if all_simple and carbon_count > 0:
                _ALKOXY = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy',
                           4: 'butoxy', 5: 'pentyloxy'}
                if carbon_count in _ALKOXY:
                    return _ALKOXY[carbon_count]
                prefix = get_chain_prefix(carbon_count)
                return f"{prefix}oxy"

    # Explicit -C(=O)OH recognizer (Wave-2 completion): the recursive
    # fallback below mis-prefixed a carboxyl as 'formyl' (the -OH oxygen was
    # dropped in the fragment round-trip — a wrong name the RT gate had to
    # suppress). P-65.1.1.2: the prefix for -COOH is 'carboxy'.
    if _is_carboxyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carboxy'

    # Fallback: use recursive naming via name_fragment_recursively()
    # This handles compound substituents (C + heteroatoms) on ring assemblies,
    # such as COOH, CONH2, CHO, etc., that the simple patterns above miss.
    if sub_atoms and len(sub_atoms) <= 25:
        try:
            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms)
            if frag_smiles:
                from ..assembly.fragment_naming import name_fragment_recursively
                from ..assembly.substituent_naming import parent_to_prefix
                frag_name = name_fragment_recursively(frag_smiles)
                if frag_name:
                    carbon_count = sum(
                        1 for i in sub_atoms
                        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                    )
                    prefix = parent_to_prefix(frag_name, chain_length=carbon_count)
                    if prefix:
                        return prefix
        except Exception:
            pass

    # Last resort: return generic placeholder (should be rare after recursive fallback)
    return "substituent"


def _is_carboxyl_substituent(mol, sub_atoms: List[int],
                             attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -C(=O)OH group attached via
    its carbon: {C, =O (degree 1), -OH (degree 1)}."""
    if len(sub_atoms) != 3:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if c.GetSymbol() != 'C' or c.GetFormalCharge() != 0:
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 2:
        return False
    oxo = oh = None
    for i in others:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'O' or a.GetDegree() != 1 or a.GetFormalCharge():
            return False
        bond = mol.GetBondBetweenAtoms(attachment_atom, i)
        if bond is None:
            return False
        if bond.GetBondType() == Chem.BondType.DOUBLE and a.GetTotalNumHs() == 0:
            oxo = i
        elif bond.GetBondType() == Chem.BondType.SINGLE and a.GetTotalNumHs() == 1:
            oh = i
    return oxo is not None and oh is not None


def _is_formyl_substituent(mol, sub_atoms: List[int],
                           attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -CHO (formyl) group attached
    via its carbon: {C(H)(=O)}. Named as the added-carbon suffix -carbaldehyde
    (P-66.6.1) on a ring assembly."""
    if len(sub_atoms) != 2:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if (c.GetSymbol() != 'C' or c.GetFormalCharge() != 0
            or c.GetTotalNumHs() != 1):
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 1:
        return False
    o = mol.GetAtomWithIdx(others[0])
    if (o.GetSymbol() != 'O' or o.GetDegree() != 1 or o.GetFormalCharge()
            or o.GetTotalNumHs() != 0):
        return False
    bond = mol.GetBondBetweenAtoms(attachment_atom, others[0])
    return bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE


def _is_cyano_substituent(mol, sub_atoms: List[int],
                          attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -C#N (cyano) group attached via
    its carbon. Named as the added-carbon suffix -carbonitrile (P-66.5.1) on a
    ring assembly."""
    if len(sub_atoms) != 2:
        return False
    c = mol.GetAtomWithIdx(attachment_atom)
    if (c.GetSymbol() != 'C' or c.GetFormalCharge() != 0
            or c.GetTotalNumHs() != 0):
        return False
    others = [i for i in sub_atoms if i != attachment_atom]
    if len(others) != 1:
        return False
    n = mol.GetAtomWithIdx(others[0])
    if (n.GetSymbol() != 'N' or n.GetDegree() != 1 or n.GetFormalCharge()
            or n.GetTotalNumHs() != 0):
        return False
    bond = mol.GetBondBetweenAtoms(attachment_atom, others[0])
    return bond is not None and bond.GetBondType() == Chem.BondType.TRIPLE


def _is_primary_amine_substituent(mol, sub_atoms: List[int],
                                  attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral primary -NH2 attached directly
    to the ring. Named as the suffix -amine (P-62.2.1)."""
    if len(sub_atoms) != 1 or sub_atoms[0] != attachment_atom:
        return False
    n = mol.GetAtomWithIdx(attachment_atom)
    return (n.GetSymbol() == 'N' and n.GetFormalCharge() == 0
            and n.GetDegree() == 1 and n.GetTotalNumHs() == 2)


def _is_hydroxy_substituent(mol, sub_atoms: List[int],
                            attachment_atom: int) -> bool:
    """True iff *sub_atoms* is exactly a neutral -OH attached directly to the
    ring. Named as the suffix -ol (P-63.1)."""
    if len(sub_atoms) != 1 or sub_atoms[0] != attachment_atom:
        return False
    o = mol.GetAtomWithIdx(attachment_atom)
    return (o.GetSymbol() == 'O' and o.GetFormalCharge() == 0
            and o.GetDegree() == 1 and o.GetTotalNumHs() == 1)


def _ring_assembly_pcg_suffix(mol, sub_atoms: List[int],
                              attachment_atom: int) -> Optional[str]:
    """Classify a ring-assembly substituent as a PRINCIPAL characteristic group
    expressible as a SUFFIX on the enclosed assembly parent (P-28.2.1 + the P-66
    suffix table). Returns the suffix stem, or None if the group is not a
    single-kind suffix-expressible PCG handled here.

      -C(=O)OH -> 'carboxylic acid'   (added exocyclic C)
      -CHO     -> 'carbaldehyde'       (added exocyclic C)
      -C#N     -> 'carbonitrile'       (added exocyclic C)
      -NH2     -> 'amine'              (direct ring attachment)
      -OH      -> 'ol'                 (direct ring attachment)
    """
    if _is_carboxyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carboxylic acid'
    if _is_formyl_substituent(mol, sub_atoms, attachment_atom):
        return 'carbaldehyde'
    if _is_cyano_substituent(mol, sub_atoms, attachment_atom):
        return 'carbonitrile'
    if _is_primary_amine_substituent(mol, sub_atoms, attachment_atom):
        return 'amine'
    if _is_hydroxy_substituent(mol, sub_atoms, attachment_atom):
        return 'ol'
    return None


# Suffix-expressible ring-assembly PCG kind -> its detachable PREFIX form, used
# when a group is DEMOTED (a more-senior PCG on the same assembly takes the
# suffix). All RT-verified against OPSIN 2.9 (P-66.6.1 carboxy, aldehyde->formyl,
# nitrile->cyano, P-63.1 hydroxy, P-62 amino).
_RING_ASSEMBLY_PCG_PREFIX = {
    'carboxylic acid': 'carboxy',
    'carbaldehyde': 'formyl',
    'carbonitrile': 'cyano',
    'amine': 'amino',
    'ol': 'hydroxy',
}

# Suffix-expressible ring-assembly PCG kind -> its rules.seniority key (so the
# senior group is picked with the SHARED seniority order, never a hand-coded one).
_RING_ASSEMBLY_PCG_SENIORITY_KEY = {
    'carboxylic acid': 'carboxylic_acid',
    'carbonitrile': 'nitrile',
    'carbaldehyde': 'aldehyde',
    'ol': 'alcohol',
    'amine': 'primary_amine',
}


def _is_nitro_substituent(mol, sub_atoms: List[int],
                          attachment_atom: int) -> bool:
    """True iff *sub_atoms* is a nitro group -NO2 attached via its nitrogen
    (matches both the [N+](=O)[O-] and neutral N(=O)=O drawings). Prefix
    'nitro' (P-61.5.1)."""
    if len(sub_atoms) != 3:
        return False
    n = mol.GetAtomWithIdx(attachment_atom)
    if n.GetSymbol() != 'N':
        return False
    oxygens = [i for i in sub_atoms if i != attachment_atom]
    if len(oxygens) != 2:
        return False
    return all(
        mol.GetAtomWithIdx(i).GetSymbol() == 'O'
        and mol.GetAtomWithIdx(i).GetDegree() == 1
        for i in oxygens
    )


def _mixed_ring_assembly_prefix_name(
    mol, sub_atoms: List[int], attachment_atom: int, pcg_kind: Optional[str]
) -> Optional[str]:
    """Prefix name for a NON-principal substituent in the mixed prefix+suffix
    ring-assembly builder. Fails CLOSED (returns None) for anything outside the
    supported carbocyclic-biaryl class, so the caller abstains rather than ship
    a garbage/dropped prefix (the local ``_name_substituent`` returns the string
    ``"substituent"`` / an ``"unknown organic compound"`` mangle for groups it
    cannot name -- a no-Java leak).

    Supported: demoted suffix-expressible PCG kinds (carboxy/formyl/cyano/
    amino/hydroxy), nitro, single halogen, all-C/H alkyl, and O-attached alkoxy.
    Every suffix-expressible group is one of the recognised PCG kinds, so a group
    MORE senior than the chosen suffix can only reach here via an unrecognised
    heteroatom pattern -> None -> abstain (never a wrong suffix)."""
    if pcg_kind is not None:
        return _RING_ASSEMBLY_PCG_PREFIX.get(pcg_kind)  # None -> abstain
    if _is_nitro_substituent(mol, sub_atoms, attachment_atom):
        return 'nitro'
    # Single halogen.
    if len(sub_atoms) == 1:
        sym = mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol()
        halo = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
        if sym in halo:
            return halo[sym]
    # All-carbon/hydrogen alkyl.
    if all(mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in sub_atoms):
        nm = _name_substituent(mol, sub_atoms, attachment_atom)
        if nm and nm != 'substituent' and 'unknown' not in nm:
            return nm
        return None
    # O-attached alkoxy (O then only C/H).
    a = mol.GetAtomWithIdx(attachment_atom)
    if a.GetSymbol() == 'O' and attachment_atom in sub_atoms:
        rest = [i for i in sub_atoms if i != attachment_atom]
        if rest and all(
            mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H') for i in rest
        ):
            nm = _name_substituent(mol, sub_atoms, attachment_atom)
            if nm and nm != 'substituent' and 'unknown' not in nm:
                return nm
    return None


def _build_mixed_pcg_ring_assembly(
    mol, substituent_list: List[Dict], pcg_kinds: List[Optional[str]],
    ring_systems: List[Set[int]], connections: List[Tuple[int, int, int, int]],
    ring_name: str, multiplier: str, connection_str: str, count: int,
) -> Optional[str]:
    """Name a 2-ring CARBOCYCLIC assembly (biphenyl class) that bears a
    suffix-expressible PCG mixed with other substituents (P-28.2.1 + P-66).

    The most SENIOR PCG (via rules.seniority) is expressed as the assembly
    SUFFIX on the enclosed parent; every other substituent becomes a detachable
    PREFIX. The PCG-bearing ring is numbered UNPRIMED so the suffix takes the
    lowest locant (P-31.1.4.3.4); the numbering direction of each ring is chosen
    by first-point-of-difference on (suffix locants, then all substituent
    locants, then the alphabetically-first prefix). PIN form places NO hyphen
    between the prefix block and the opening bracket (`**P-16.2.4.2**`, BB 6968:
    no hyphen after a numerical prefix before an enclosing mark). The witness
    6,6'-dinitro[1,1'-biphenyl]-2,2'-dicarboxylic acid is real but lives at BB
    49803 under P-93 (axial chirality) -- it is NOT a P-16.3.3 example, and
    P-16.3.3 contains no biphenyl.

    Returns the PIN, or None to fail CLOSED (out-of-class / un-nameable prefix),
    letting ``name_ring_assembly`` fall through to its veto / prefix-only path.
    """
    from .seniority import compare_seniority
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key

    # Scope: exactly a 2-component, single-bond, all-carbon single-ring assembly.
    if count != 2 or len(connections) != 1 or len(ring_systems) != 2:
        return None
    present = [k for k in pcg_kinds if k is not None]
    if not present:
        return None  # no suffix-expressible PCG -> not this builder's job

    ri = mol.GetRingInfo()
    a1, a2, s1, s2 = connections[0]
    junction = {s1: a1, s2: a2}
    ring_of: Dict[int, Tuple[int, ...]] = {}
    for sidx, satoms in enumerate(ring_systems):
        rings = [r for r in ri.AtomRings() if set(r) <= set(satoms)]
        if len(rings) != 1:
            return None  # fused component -> out of class
        ring = rings[0]
        if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
            return None  # heterocyclic assembly numbers junction != 1 -> abstain
        if sidx not in junction or junction[sidx] not in ring:
            return None
        ring_of[sidx] = ring

    # Senior PCG kind (shared seniority order; lower index = more senior).
    senior = present[0]
    for k in present[1:]:
        if compare_seniority(
            _RING_ASSEMBLY_PCG_SENIORITY_KEY[k],
            _RING_ASSEMBLY_PCG_SENIORITY_KEY[senior],
        ) < 0:
            senior = k

    # Partition; compute prefix names now (abstain on any un-nameable group).
    suffix_subs, prefix_subs = [], []
    for s, k in zip(substituent_list, pcg_kinds):
        if k == senior:
            suffix_subs.append(s)
        else:
            nm = _mixed_ring_assembly_prefix_name(
                mol, s['sub_atoms'], s['sub_atoms'][0], k)
            if nm is None:
                return None  # un-nameable substituent -> fail closed
            prefix_subs.append((s, nm))
    if not suffix_subs:
        return None

    # Per-ring position maps (junction = locant 1; both walk directions).
    def _position_maps(ring: Tuple[int, ...], conn: int) -> List[Dict[int, int]]:
        rset = set(ring)
        adj: Dict[int, List[int]] = {i: [] for i in ring}
        for b in mol.GetBonds():
            x, y = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if x in rset and y in rset:
                adj[x].append(y)
                adj[y].append(x)
        maps: List[Dict[int, int]] = []
        for first in adj[conn]:
            visited = [conn]
            cur = first
            while cur != conn:
                visited.append(cur)
                nxt = [nn for nn in adj[cur] if nn not in visited]
                if not nxt:
                    break
                cur = nxt[0]
            if len(visited) == len(ring):
                maps.append({a: i + 1 for i, a in enumerate(visited)})
        return maps or [{a: i + 1 for i, a in enumerate(ring)}]

    pos_maps = {
        sidx: _position_maps(ring_of[sidx], junction[sidx])
        for sidx in ring_of
    }

    # Search priming (which system is unprimed) x direction per ring; pick the
    # assignment with lowest (suffix keys, all-substituent keys, alpha-first).
    # A locant key is (number, prime_count): unprimed (0) < primed (1) at equal
    # number, so plain tuple sort implements the P-31.1.4.3.4 lowest-locant rule.
    best = None
    best_key = None
    systems = list(ring_of)  # [0, 1]
    for unprimed in systems:
        prime_of = {unprimed: 0, [x for x in systems if x != unprimed][0]: 1}
        for m0 in pos_maps[systems[0]]:
            for m1 in pos_maps[systems[1]]:
                pmap = {systems[0]: m0, systems[1]: m1}

                def _key(s):
                    sy = s['system_idx']
                    return (pmap[sy][s['ring_atom']], prime_of[sy])

                suffix_keys = sorted(_key(s) for s in suffix_subs)
                all_keys = sorted(
                    [_key(s) for s in suffix_subs]
                    + [_key(s) for s, _ in prefix_subs]
                )
                alpha_first = None
                if prefix_subs:
                    first_sub = min(prefix_subs, key=lambda sn: alpha_sort_key(sn[1]))
                    alpha_first = _key(first_sub[0])
                cand = (suffix_keys, all_keys, alpha_first or (0, 0))
                if best_key is None or cand < best_key:
                    best_key = cand
                    best = (prime_of, pmap)

    prime_of, pmap = best

    def _loc(s):
        sy = s['system_idx']
        return pmap[sy][s['ring_atom']], prime_of[sy]

    # Suffix block.
    suffix_keys = sorted(_loc(s) for s in suffix_subs)
    suffix_loc_str = ",".join(
        f"{num}{_format_prime(pr)}" for num, pr in suffix_keys)
    smult = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}.get(len(suffix_keys))
    if smult is None:
        return None
    core = (f"[{connection_str}-{multiplier}{ring_name}]"
            f"-{suffix_loc_str}-{smult}{senior}")

    # Prefix block (alphanumerical, grouped multipliers). Directly abuts '[' with
    # NO hyphen (P-16.5.1.1: no hyphen before an opening enclosing mark).
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for s, nm in prefix_subs:
        by_name.setdefault(nm, []).append(_loc(s))
    prefix_parts = []
    for nm in sorted(by_name, key=alpha_sort_key):
        keys = sorted(by_name[nm])
        loc_str = ",".join(f"{num}{_format_prime(pr)}" for num, pr in keys)
        if len(keys) > 1:
            prefix_parts.append(f"{loc_str}-{get_multiplier_prefix(len(keys), nm)}{nm}")
        else:
            prefix_parts.append(f"{loc_str}-{nm}")
    prefix_block = "-".join(prefix_parts)

    return f"{prefix_block}{core}" if prefix_block else core


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
    # P-28.4.2: skeletal-replacement ring assembly (mixed/single-heteroatom
    # same-skeleton monocycles) -> dedicated 'a'-nomenclature namer.
    if assembly_info.get('replacement'):
        return _name_replacement_ring_assembly(mol, assembly_info)

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

    # Phase 155 WR-05: ring stems carrying an embedded P-31.1.4.4 indicated-H
    # descriptor (e.g. "phenanthridin-6(5H)-one", "acridin-9(10H)-one") need a
    # per-ring threading strategy the placement-subset fix doesn't implement;
    # fail closed (return None) so a higher-level fallback can attempt naming
    # rather than emit the buggy mid-string `2,2'-biphenanthridin-6(5H)-one`
    # form. Deeper rewrite is Phase 151 Follow-up 12 main-thread territory.
    if _INDICATED_H_EMBEDDED_RE.search(ring_name):
        return None

    # Build connection locant string per IUPAC P-28.2.1.
    # Phase 151-03 D-18 / D-19: use per-system numbering so ter-/quater-
    # assemblies emit the correct middle-ring back-attachment locant
    # (4 for para-terphenyl, 6 for terpyridine, 5 for terthiophene).
    # The pre-151-03 path called _get_connection_locant per pair, which
    # always returned 1 for carbocyclic atoms — yielding the buggy
    # "1,1':1',1''-terphenyl" output captured in 151-AUDIT-C.md.
    per_system_locants = _compute_per_system_ring_locants(mol, assembly_info)

    def _lookup_locant(per_system, sys_idx, atom_idx, sys_atoms):
        """Locant lookup with safe fallback to legacy single-bond helper."""
        if (
            per_system is not None
            and 0 <= sys_idx < len(per_system)
            and atom_idx in per_system[sys_idx]
        ):
            return per_system[sys_idx][atom_idx]
        return _get_connection_locant(mol, atom_idx, sys_atoms)

    # Substituents on the assembly, computed once and reused below (the ylidene
    # branch, the P-28.3.1 citation-order tiebreak, and the prefix builder).
    substituent_list = _get_substituent_info(mol, ring_systems, connections)

    # P-28.2.1 (connection/"free valence" locant) + P-31.1.4.3.4 (lowest
    # locants to substituents): for a 2-component assembly, decide ONCE,
    # deterministically, which physical ring system is UNPRIMED by
    # first-point-of-difference on the COMBINED citation set the assembly
    # actually uses -- connection locants first (tier 1, P-28.2.1), then
    # substituent-prefix locants (tier 2, P-31.1.4.3.4). ``ring_systems``/
    # ``connections`` order coming out of ``detect_ring_assembly`` is RDKit
    # atom-index (SMILES-spelling) dependent, so without this the choice of
    # which ring is "system 0" (unprimed) was non-deterministic AND, whenever
    # a substituent broke the tie, sometimes wrong (a substituent on the ring
    # that happened to land at index 1 was cited with a prime it should never
    # carry -- "4'-chloro-1,1'-biphenyl" instead of the PIN
    # "4-chloro-1,1'-biphenyl"). Mirrors the analogous heteroatom
    # combined-locant-set tiebreak in ``_name_replacement_ring_assembly``
    # (P-28.4.2), generalised from heteroatoms to substituents. Relabeling
    # here -- BEFORE the connection string and substituent locants are built
    # below -- keeps every downstream consumer (the connection-locant loop,
    # the single-kind-suffix branch, the mixed prefix+suffix builder, and the
    # plain substituent-prefix branch) automatically consistent: they all key
    # off the same (now canonical) ``system_idx``.
    if count == 2 and len(ring_systems) == 2:
        def _combined_key(swap: bool):
            conn_key = []
            for a1, a2, s1, s2 in connections:
                for atom, sys in ((a1, s1), (a2, s2)):
                    eff_sys = (1 - sys) if swap else sys
                    loc = _lookup_locant(
                        per_system_locants, sys, atom, ring_systems[sys])
                    conn_key.append((loc, eff_sys))
            sub_key = []
            for s in substituent_list:
                eff_sys = (1 - s['system_idx']) if swap else s['system_idx']
                sub_key.append((s['locant'], eff_sys))
            return (sorted(conn_key), sorted(sub_key))

        if _combined_key(True) < _combined_key(False):
            ring_systems = [ring_systems[1], ring_systems[0]]
            connections = [(a1, a2, 1 - s1, 1 - s2)
                           for (a1, a2, s1, s2) in connections]
            if per_system_locants is not None and len(per_system_locants) == 2:
                per_system_locants = [per_system_locants[1], per_system_locants[0]]
            for s in substituent_list:
                s['system_idx'] = 1 - s['system_idx']

    # Sort connections by system indices to ensure consistent ordering.
    # For bi- assemblies: one connection -> "X,X'"
    # For ter- assemblies: two connections -> "X,X':X',X''"
    connection_parts = []
    sorted_connections = []
    for a1, a2, s1, s2 in connections:
        if s1 > s2:
            a1, a2, s1, s2 = a2, a1, s2, s1
        sorted_connections.append((a1, a2, s1, s2))
    sorted_connections.sort(key=lambda c: (c[2], c[3]))
    for a1, a2, s1, s2 in sorted_connections:
        loc1 = _lookup_locant(per_system_locants, s1, a1, ring_systems[s1])
        loc2 = _lookup_locant(per_system_locants, s2, a2, ring_systems[s2])
        prime1 = _format_prime(s1)
        prime2 = _format_prime(s2)
        # IUPAC P-28.2.1 + P-28.3.1 (8 Oct 2025 erratum): "lowest locants ...
        # then order of citation" — the unprimed (first-cited) ring takes the
        # lower attachment locant. For a 2-component assembly of identical rings
        # with no distinguishing substituent, get_ring_systems order is
        # atom-index (SMILES-spelling) dependent, so without this tiebreak the
        # asymmetric case is order-dependent (2,3'-bifuran vs 3,2'-bifuran).
        # Relabeling the unprimed ring is valid only when no substituent
        # differentiates the two identical rings.
        if count == 2 and not substituent_list and loc2 < loc1:
            loc1, loc2 = loc2, loc1
        connection_parts.append(f"{loc1}{prime1},{loc2}{prime2}")

    connection_str = ":".join(connection_parts)

    # IUPAC P-28.2.2: a double-bond junction is named with the ylidene
    # substituent-group form (method 2), enclosed in parentheses to avoid
    # confusion with von Baeyer names: 1,1'-bi(cyclopentylidene). The detector
    # restricts entry to saturated carbocycles; convert cyclopentane ->
    # cyclopentylidene here. Fail closed (return None -> prior naming) if the
    # component is not a clean cyclo...ane ring or carries substituents, since
    # substituted ylidene assemblies are outside the built class (no wrong names).
    if assembly_info.get("double_bond_junction"):
        ylidene = _to_ylidene(ring_name)
        if ylidene is None:
            return None
        if substituent_list:
            return None
        return f"{connection_str}-{multiplier}{_enclose_component(ylidene)}"

    # Phase 155.B D-09: indicated-H placement subset for ring assemblies.
    # If ring_name carries an indicated-H prefix like "1H-indole", emit the
    # descriptor once per primed ring ("1H,1'H-2,2'-biindole") instead of
    # leaving it embedded inside the multiplied stem ("2,2'-bi1H-indole",
    # the buggy pre-fix output).  The deeper assembly-builder rewrite
    # (per-ring indicated-H locants generically threaded into the assembly
    # base-name for biindole-class assemblies) stays in Phase 151's deferred-
    # warnings backlog (Follow-up 12 main thread).
    # Source: 155-CONTEXT.md D-09; HERITAGE-followups.md Follow-up 12 placement
    #         subset; IUPAC P-31.1.4.  Reuses _format_prime above.
    indicated_h_match = _INDICATED_H_RE.match(ring_name)
    if indicated_h_match:
        locant_int, ring_stem = (
            indicated_h_match.group(1),
            indicated_h_match.group(2),
        )
        # P-28.2.1 / P-31.1.7.3: a COMPOUND component name (a skeletal-'a'
        # mancude heterocycle carrying its own heteroatom-locant set, e.g.
        # '4H-1,4-oxaphosphinine') is ENCLOSED in parentheses with its
        # indicated-H kept INSIDE, cited once — '4,4'-bi(4H-1,4-oxaphosphinine)'.
        # This differs from the biindole class (bare stem 'indole', no internal
        # locants), which front-replicates the indicated-H per primed ring
        # ("1H,1'H-2,2'-biindole", P-31.1.4). Distinguish by whether the stem
        # itself carries a locant set (a comma-locant '<d>,<d>-' prefix).
        if re.match(r"^\d[\d,]*-", ring_stem):
            base_name = (
                f"{connection_str}-{multiplier}({ring_name})"
            )
        else:
            indicated_h_replicated = ",".join(
                f"{locant_int}{_format_prime(i)}H" for i in range(count)
            ) + "-"
            base_name = (
                f"{indicated_h_replicated}{connection_str}-{multiplier}{ring_stem}"
            )
    else:
        # IUPAC P-28.2.1: enclose the component in parentheses when needed to
        # avoid confusion with von Baeyer names (cycloalkanes / spiro / bicyclo);
        # mancude rings (phenyl/pyridine/furan/...) stay bare.
        base_name = f"{connection_str}-{multiplier}{_enclose_component(ring_name)}"

    if not substituent_list:
        return base_name

    # P-28.2.1 / P-16.5.2.1 / P-66 suffix table: a PRINCIPAL characteristic
    # group on a ring assembly must be expressed as a SUFFIX on the enclosed
    # assembly parent, never as a prefix. Added-carbon groups —
    # '[1,1'-biphenyl]-4,4'-dicarboxylic acid', '[1,1'-biphenyl]-4-carbaldehyde',
    # '[1,1'-biphenyl]-4-carbonitrile' — and direct groups —
    # '[1,1'-biphenyl]-4-amine', '[1,1'-biphenyl]-4,4'-diol' — all follow the
    # same enclosed-parent + locant + multiplied-suffix shape. Prefix forms for
    # these mis-name or drop atoms (CHO -> 'formaldehydyl', CN -> 'hydrogen
    # cyanidyl' — leaks the OPSIN self-consistency gate suppresses to 'unknown').
    # Scope (fail-closed): EVERY substituent is the SAME suffix-expressible PCG;
    # mixed / other decorations keep the established prefix-only path below.
    _pcg_kinds = [
        _ring_assembly_pcg_suffix(mol, s['sub_atoms'], s['sub_atoms'][0])
        for s in substituent_list
    ]
    if _pcg_kinds and all(k is not None for k in _pcg_kinds) and \
            len(set(_pcg_kinds)) == 1:
        _suffix_stem = _pcg_kinds[0]
        suffix_pairs = [(s['locant'], s['system_idx'])
                        for s in substituent_list]
        # P-14.4(c) determinism: the suffix takes the LOWEST locants -- the
        # ring carrying it must be the UNPRIMED one ([1,1'-biphenyl]-4-
        # carboxylic acid, never -4'-). get_ring_systems order is SMILES-
        # spelling dependent, so for a symmetric 2-ring connection (equal
        # attachment locants) relabel the priming when that lowers the
        # suffix locant set; asymmetric connections keep their citation-
        # order labels (relabelling would alter the connection locants).
        if count == 2 and sorted_connections:
            _a1, _a2, _s1, _s2 = sorted_connections[0]
            _l1 = _lookup_locant(per_system_locants, _s1, _a1,
                                 ring_systems[_s1])
            _l2 = _lookup_locant(per_system_locants, _s2, _a2,
                                 ring_systems[_s2])
            if _l1 == _l2:
                _swapped = [(loc, 1 - sys_idx) for loc, sys_idx in
                            suffix_pairs]
                if sorted(_swapped) < sorted(suffix_pairs):
                    suffix_pairs = _swapped
        suffix_locants = sorted(suffix_pairs)
        locant_str = ",".join(
            f"{loc}{_format_prime(sys_idx)}" for loc, sys_idx in suffix_locants
        )
        n = len(suffix_locants)
        mult = {1: '', 2: 'di', 3: 'tri', 4: 'tetra'}.get(n)
        if mult is None:
            return None
        if indicated_h_match:
            return None  # indicated-H + suffix threading not built
        return (f"[{connection_str}-{multiplier}{ring_name}]"
                f"-{locant_str}-{mult}{_suffix_stem}")

    # P-28.2.1 + P-66: MIXED prefix+suffix assembly (a suffix-expressible PCG
    # coexists with other substituents). The senior PCG becomes the suffix, the
    # rest become prefixes. Only fires for the 2-ring carbocyclic (biphenyl)
    # class; fails closed (returns None) otherwise, falling through to the veto /
    # prefix-only path below. Purely additive: upgrades wrong-PIN prefix names
    # ('4'-chloro-4-hydroxy-1,1'-biphenyl') and fail-closed 'unknown' cases to
    # the PIN, and closes the nitro gate-off leak; never regresses a working name.
    if any(k is not None for k in _pcg_kinds):
        _mixed = _build_mixed_pcg_ring_assembly(
            mol, substituent_list, _pcg_kinds, ring_systems, connections,
            ring_name, multiplier, connection_str, count)
        if _mixed is not None:
            return _mixed

    # Fail-closed veto: a -CHO / -C#N on a ring assembly has NO valid prefix
    # form — the generic substituent namer emits the bogus 'formaldehydyl' /
    # 'hydrogen cyanidyl' (OPSIN-unparseable; suppressed WITH Java, shipped WRONG
    # without it). The single-kind suffix path above already consumed the pure
    # cases; reaching here means the group is MIXED with others, which needs the
    # prefix+suffix ring-assembly builder (not yet built). Refuse rather than
    # ship a wrong name (accuracy #1).
    for _s in substituent_list:
        _a = _s['sub_atoms'][0]
        if (_is_formyl_substituent(mol, _s['sub_atoms'], _a)
                or _is_cyano_substituent(mol, _s['sub_atoms'], _a)):
            return None
        # P-29.2: _name_substituent returns None when the attachment bond is
        # double/triple and no prefix spells that free valence. Refuse for the
        # same reason as the two cases above -- a name built around a missing
        # or single-valence token describes a different molecule.
        if not _s.get('name'):
            return None

    # Build substituent prefix
    # Group by name for multipliers
    from collections import Counter
    from ..assembly.naming_utils import (
        get_multiplier_prefix,
        alpha_sort_key,
    )

    # Sort substituents alphabetically by name
    substituent_list.sort(key=lambda s: alpha_sort_key(s['name']))

    # Group identical substituents. Locant/prime pairs are collected as
    # (locant, system_idx) keys -- NOT pre-formatted strings -- and sorted
    # numerically (ascending locant, unprimed before primed at equal locant)
    # before formatting. ``substituent_list`` iteration order upstream traces
    # back to a Python-set walk over ring atoms in ``_get_substituent_info``,
    # which is atom-index (SMILES-spelling) dependent; without this explicit
    # sort, two chemically-identical substituents (e.g. both ring chloro
    # atoms in 4,4'-dichlorobiphenyl) could be cited in either order,
    # producing non-deterministic "4,4'-dichloro-..." vs "4',4-dichloro-..."
    # output for the same molecule.
    sub_groups: Dict[str, List[Tuple[int, int]]] = {}
    for sub in substituent_list:
        name = sub['name']
        sub_groups.setdefault(name, []).append((sub['locant'], sub['system_idx']))

    # Build prefix parts
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        keys = sorted(sub_groups[name])
        n = len(keys)
        locant_str = ",".join(
            f"{loc}{_format_prime(sys_idx)}" for loc, sys_idx in keys)
        if n > 1:
            mult = get_multiplier_prefix(n, name)
            prefix_parts.append(f"{locant_str}-{mult}{name}")
        else:
            prefix_parts.append(f"{locant_str}-{name}")

    sub_prefix = "-".join(prefix_parts)

    return f"{sub_prefix}-{base_name}"


def name_ring_assembly_prefix(
    mol, assembly_info: Dict, attachment_atom_idx: int
) -> Optional[str]:
    """Generate ring assembly substituent prefix per IUPAC P-28.3.

    When a ring assembly (identical rings joined by single bonds) appears as
    a substituent on a parent chain, the prefix uses square-bracket notation
    with primed connection locants and an attachment locant with -yl suffix.

    Format: ``[connection_locants-multiplierRing_name]-attach_locant-yl``

    Args:
        mol: RDKit Mol object
        assembly_info: Dict from detect_ring_assembly() with keys:
            ring_systems, connections, count, ring_type
        attachment_atom_idx: Atom index where assembly connects to parent chain

    Returns:
        Prefix string like ``[1,1'-biphenyl]-4-yl`` or None on failure.

    Examples:
        biphenyl attached at para position -> "[1,1'-biphenyl]-4-yl"
        bipyridine attached at position 5 -> "[2,2'-bipyridin]-5-yl"
    """
    ring_systems = assembly_info['ring_systems']
    connections = assembly_info['connections']
    count = assembly_info['count']

    multiplier = ASSEMBLY_MULTIPLIERS.get(count)
    if multiplier is None:
        return None

    ring_name = _get_ring_parent_name(mol, ring_systems[0])
    if ring_name is None:
        return None

    # Phase 155 WR-05: same fail-closed as in `name_ring_assembly` for embedded
    # P-31.1.4.4 descriptors. Both call sites must agree on the contract.
    if _INDICATED_H_EMBEDDED_RE.search(ring_name):
        return None

    # Build connection locant string (reuse existing logic from name_ring_assembly)
    connection_parts = []
    for a1, a2, s1, s2 in connections:
        if s1 > s2:
            a1, a2, s1, s2 = a2, a1, s2, s1
        loc1 = _get_connection_locant(mol, a1, ring_systems[s1])
        loc2 = _get_connection_locant(mol, a2, ring_systems[s2])
        prime1 = _format_prime(s1)
        prime2 = _format_prime(s2)
        connection_parts.append(f"{loc1}{prime1},{loc2}{prime2}")
    connection_str = ":".join(connection_parts)

    # Find which ring system the attachment atom belongs to
    attach_system_idx = None
    for i, sys_atoms in enumerate(ring_systems):
        if attachment_atom_idx in sys_atoms:
            attach_system_idx = i
            break
    if attach_system_idx is None:
        return None

    # Find the inter-ring connection atom in the attachment ring system.
    # The assembly numbering starts from the inter-ring bond (locant 1),
    # so we need to compute the attachment locant relative to that point.
    inter_ring_conn_atom = None
    for a1, a2, s1, s2 in connections:
        if s1 == attach_system_idx:
            inter_ring_conn_atom = a1
            break
        elif s2 == attach_system_idx:
            inter_ring_conn_atom = a2
            break

    if inter_ring_conn_atom is not None:
        # Use _get_substituent_locant which numbers from the inter-ring
        # connection point (locant 1) and gives the correct position for
        # the chain attachment point
        attach_locant = _get_substituent_locant(
            mol, attachment_atom_idx, attachment_atom_idx,
            ring_systems[attach_system_idx], inter_ring_conn_atom
        )
    else:
        # Fallback: use _get_connection_locant (for single-ring edge cases)
        attach_locant = _get_connection_locant(
            mol, attachment_atom_idx, ring_systems[attach_system_idx]
        )
    attach_prime = _format_prime(attach_system_idx)

    # Phase 155.B D-09: indicated-H placement subset for ring-assembly
    # SUBSTITUENT prefix path. Mirrors the parent-path replication in
    # `name_ring_assembly` (line 1201-1212) so a biindole-bearing
    # substituent emits `[1H,1'H-2,2'-biindol]-5-yl` instead of the pre-fix
    # buggy form `[2,2'-bi1H-indol]-5-yl`. Both call sites now agree on
    # the replication contract -- a biindole-bearing molecule names
    # consistently whether the assembly is parent or substituent.
    # Source: 155-REVIEW.md WR-01; HERITAGE-followups.md Follow-up 12
    #         placement subset; IUPAC P-31.1.4.  Reuses _format_prime
    #         and _INDICATED_H_RE.
    indicated_h_match = _INDICATED_H_RE.match(ring_name)
    if indicated_h_match:
        locant_int, ring_stem = (
            indicated_h_match.group(1),
            indicated_h_match.group(2),
        )
        indicated_h_replicated = ",".join(
            f"{locant_int}{_format_prime(i)}H" for i in range(count)
        ) + "-"
        # Vowel elision for the -yl form: "indole" -> "indol".
        display_stem = ring_stem[:-1] if ring_stem.endswith('e') else ring_stem
        # Assembly base: "1H,1'H-2,2'-biindol"
        assembly_base = (
            f"{indicated_h_replicated}{connection_str}-{multiplier}{display_stem}"
        )
    else:
        # For heterocyclic rings, apply vowel elision: "pyridine" -> "pyridin" before -yl
        # (IUPAC P-31.1.3.4: terminal 'e' dropped before '-yl')
        display_name = ring_name
        if display_name.endswith('e'):
            display_name = display_name[:-1]

        # Assembly base: "1,1'-biphenyl" or "2,2'-bipyridin"
        assembly_base = f"{connection_str}-{multiplier}{display_name}"

    # Full prefix: "[1,1'-biphenyl]-4-yl" or "[1H,1'H-2,2'-biindol]-5-yl"
    return f"[{assembly_base}]-{attach_locant}{attach_prime}-yl"


def name_mixed_ring_prefix(
    mol,
    ring_systems_list: List[Set[int]],
    inter_system_bonds: List[Tuple[int, int, int, int]],
    attachment_atom_idx: int,
) -> Optional[str]:
    """Generate compound substituent prefix for non-identical connected rings.

    When two or more non-identical ring systems are connected by single bonds
    and appear as a substituent on a parent chain, the ring carrying the free
    valence (chain attachment) is the parent of the compound substituent prefix.
    The other ring(s) become simple substituents on it.

    For the parent ring of the compound prefix:
    - Carbocyclic: numbering gives chain attachment locant 1 (lowest locant rule)
    - Heterocyclic: standard IUPAC numbering (heteroatom at position 1)

    Args:
        mol: RDKit Mol object
        ring_systems_list: List of sets of atom indices, one per ring system
        inter_system_bonds: List of (atom_A, atom_B, system_A, system_B) tuples
            from _find_inter_system_bonds()
        attachment_atom_idx: Atom index where the multi-ring fragment connects
            to the parent chain

    Returns:
        Compound prefix string like ``(4-(pyridin-2-yl)phenyl)`` or None.
    """
    if len(ring_systems_list) < 2:
        return None

    from .ring_substituents import get_ring_substituent_name, identify_ring_system

    # Determine which ring the chain attaches to -- that becomes the parent
    # of the compound substituent prefix (it carries the free valence = -yl)
    parent_idx = None
    sub_idx = None
    for i, sys_atoms in enumerate(ring_systems_list):
        if attachment_atom_idx in sys_atoms:
            parent_idx = i
            break
    if parent_idx is None:
        return None

    # For 2-ring systems, the other ring is the substituent
    sub_idx = 1 - parent_idx if len(ring_systems_list) == 2 else None
    if sub_idx is None:
        # For 3+ non-identical rings, not yet supported
        return None

    parent_atoms = ring_systems_list[parent_idx]
    sub_atoms = ring_systems_list[sub_idx]

    # Get the parent ring's system name (for stem)
    parent_ring_tuple = tuple(sorted(parent_atoms))
    parent_ring_name = identify_ring_system(mol, parent_ring_tuple)
    if not parent_ring_name:
        return None

    # Get the substituent ring's -yl name
    sub_ring_tuple = tuple(sorted(sub_atoms))

    # For the substituent ring, get its prefix name with position if applicable
    # Find where the inter-ring bond attaches to the sub ring
    sub_attach_atom = None
    for a1, a2, s1, s2 in inter_system_bonds:
        if s1 == sub_idx:
            sub_attach_atom = a1
            break
        elif s2 == sub_idx:
            sub_attach_atom = a2
            break

    sub_yl_name = get_ring_substituent_name(
        mol, sub_ring_tuple, attachment_point=sub_attach_atom
    )
    if not sub_yl_name:
        return None

    # Check if sub ring name needs parenthesization (if it contains locants/hyphens)
    # e.g., "pyridin-2-yl" needs parentheses: "(pyridin-2-yl)"
    # but "phenyl" does not
    if '-' in sub_yl_name and not sub_yl_name.startswith('('):
        sub_display = f"({sub_yl_name})"
    else:
        sub_display = sub_yl_name

    # Parent ring stem: drop terminal 'e' before -yl (vowel elision per IUPAC)
    parent_stem = parent_ring_name
    if parent_stem.endswith('e'):
        parent_stem = parent_stem[:-1]

    # Determine locants on the parent ring
    # Find the inter-ring bond atom on the parent ring
    parent_conn_atom = None
    for a1, a2, s1, s2 in inter_system_bonds:
        if s1 == parent_idx:
            parent_conn_atom = a1
            break
        elif s2 == parent_idx:
            parent_conn_atom = a2
            break
    if parent_conn_atom is None:
        return None

    # For locant calculation, use _get_substituent_locant which numbers
    # carbocyclic rings from the chain attachment point (locant 1) and
    # heterocyclic rings from standard IUPAC numbering
    connection_locant = _get_substituent_locant(
        mol, parent_conn_atom, parent_conn_atom,
        parent_atoms, attachment_atom_idx
    )
    attach_locant = _get_substituent_locant(
        mol, attachment_atom_idx, attachment_atom_idx,
        parent_atoms, attachment_atom_idx
    )

    # For heterocyclic parent rings, use the standard IUPAC numbering
    # (not relative to attachment point). Check if parent is heterocyclic.
    parent_has_het = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in parent_atoms
    )
    if parent_has_het:
        # Standard IUPAC numbering from _get_connection_locant
        connection_locant = _get_connection_locant(mol, parent_conn_atom, parent_atoms)
        attach_locant = _get_connection_locant(mol, attachment_atom_idx, parent_atoms)

    # Build compound prefix
    # Format: "(connection_locant-sub_displaystem-attach_locant-yl)"
    # Example with phenyl parent: "(4-(pyridin-2-yl)phenyl)" -- attach_locant omitted when 1
    # Example with pyridine parent: "(2-phenylpyridin-3-yl)"

    # For benzene, use "phenyl" directly (retained substituent name)
    # For other rings, use "stem-attach_locant-yl"
    if parent_ring_name == 'benzene':
        # Benzene as parent: "phenyl" is the standard substituent name
        # Locant for connection: the position where sub ring attaches
        # When chain attachment = position 1, connection locant is meaningful
        if attach_locant == 1:
            # Simple case: chain at position 1
            return f"({connection_locant}-{sub_display}phenyl)"
        else:
            return f"({connection_locant}-{sub_display}phenyl)"
    else:
        # General case: stem-attach_locant-yl
        return f"({connection_locant}-{sub_display}{parent_stem}-{attach_locant}-yl)"


# ============================================================================
# Phase 151-03 D-21: cascade-step-6 supplier (get_ring_assembly_iupac_locants)
# ============================================================================

def get_ring_assembly_iupac_locants(mol) -> Optional[Dict[int, _Locant]]:
    """Phase 151-03 D-21 cascade-step-6 supplier for ring assemblies size >= 2.

    Returns the per-system IUPAC numbering of every ring atom merged into a
    single ``Dict[int, int]`` covering ALL ring atoms in the assembly.
    Primes are NAME-format-layer concerns only (in ``_format_prime``), so
    the supplier emits plain integer locants — the comparator in
    ``compare_locant_sets`` and ``_build_ring_pos`` sees the integer base.

    Coverage invariant per Pitfall 7: returns ``None`` when partial
    coverage would otherwise leak into the cascade-step-6 gate. The gate
    in ``candidate_pool.py:634::_has_iupac_locants`` checks dict
    truthiness only; a partial map would silently mis-rank candidates.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom_idx -> int locant covering all ring atoms in
        every system of the assembly. Returns ``None`` when:
          * the molecule is not a ring assembly per ``detect_ring_assembly``
            (size < 2, mixed signatures, branched topology, oversize, etc.),
          * any system's per-ring numbering produces partial coverage.

    Source: 151-CONTEXT.md D-18, D-19, D-21.
    Source: 151-PATTERNS.md Pattern S-3 (cascade-step-6 supplier contract).
    Source: 151-RESEARCH.md §"OPSIN Compatibility Evidence" (12 named cases).
    """
    from ..perception.rings import get_ring_systems

    if mol is None:
        return None
    ring_systems = get_ring_systems(mol, include_spiro=False)
    if len(ring_systems) < 2:
        return None

    info = detect_ring_assembly(mol, ring_systems)
    if info is None:
        return None

    per_system = _compute_per_system_ring_locants(mol, info)
    if per_system is None:
        return None

    # Merge per-system maps into a single atom_idx -> locant map.
    merged: Dict[int, int] = {}
    for sys_map in per_system:
        for atom_idx, loc in sys_map.items():
            merged[atom_idx] = loc

    # Coverage invariant (Pitfall 7): every ring atom in the assembly
    # must be covered. Partial coverage returns None so the cascade
    # falls through to the sorted-int proxy (no silent mis-ranking).
    ri = mol.GetRingInfo()
    all_ring_atoms: Set[int] = set()
    for r in ri.AtomRings():
        all_ring_atoms.update(r)
    if not (set(merged.keys()) >= all_ring_atoms):
        return None

    # Filter to ring atoms only (cascade-step-6 contract; no acyclic atoms).
    return {k: v for k, v in merged.items() if k in all_ring_atoms}
