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
}


def _system_signature(mol, system_atoms: Set[int]) -> Tuple:
    """
    Compute a ring system signature for identity comparison.

    The signature includes sorted element symbols, sorted ring sizes that
    are fully contained within the system, and aromaticity of the system.
    Two systems are identical if and only if their signatures match.

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in the ring system

    Returns:
        Tuple of (sorted_elements, sorted_ring_sizes, is_aromatic)
    """
    elements = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in system_atoms)

    ri = mol.GetRingInfo()
    ring_sizes = []
    for ring in ri.AtomRings():
        if set(ring) <= system_atoms:
            ring_sizes.append(len(ring))

    is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in system_atoms)

    return (tuple(elements), tuple(sorted(ring_sizes)), is_aromatic)


def _find_inter_system_bonds(
    mol, ring_systems: List[Set[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Find all single bonds connecting different ring systems.

    Returns list of (atom_idx_A, atom_idx_B, system_idx_A, system_idx_B).
    Only SINGLE and AROMATIC bond types are accepted (biphenyl's inter-ring
    bond may be typed as either depending on Kekulization).
    """
    connections = []
    acceptable_types = {
        rdchem.BondType.SINGLE,
        rdchem.BondType.AROMATIC,
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
            # Use the heterocycle naming infrastructure
            from .heterocycles import name_heterocycle
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
        """Walk ring from start through direction_neighbor, return ordered atoms."""
        visited = [start]
        current = direction_neighbor
        while current != start:
            visited.append(current)
            nexts = [n for n in adj[current] if n not in visited or n == start]
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
        connection_parts.append(f"{loc1}{prime1},{loc2}{prime2}")

    connection_str = ":".join(connection_parts)

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
        indicated_h_replicated = ",".join(
            f"{locant_int}{_format_prime(i)}H" for i in range(count)
        ) + "-"
        base_name = (
            f"{indicated_h_replicated}{connection_str}-{multiplier}{ring_stem}"
        )
    else:
        base_name = f"{connection_str}-{multiplier}{ring_name}"

    # Find substituents
    substituent_list = _get_substituent_info(mol, ring_systems, connections)

    if not substituent_list:
        return base_name

    # Build substituent prefix
    # Group by name for multipliers
    from collections import Counter
    from ..assembly.naming_utils import (
        get_multiplier_prefix,
        alpha_sort_key,
    )

    # Sort substituents alphabetically by name
    substituent_list.sort(key=lambda s: alpha_sort_key(s['name']))

    # Group identical substituents
    sub_groups = {}
    for sub in substituent_list:
        name = sub['name']
        if name not in sub_groups:
            sub_groups[name] = []
        prime = _format_prime(sub['system_idx'])
        sub_groups[name].append(f"{sub['locant']}{prime}")

    # Build prefix parts
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locants = sub_groups[name]
        n = len(locants)
        locant_str = ",".join(str(l) for l in locants)
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
