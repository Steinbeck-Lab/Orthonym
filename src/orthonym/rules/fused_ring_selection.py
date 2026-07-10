"""FR-2.3 base component selection for fused ring systems.

References:
- IUPAC 2013 Blue Book P-25.3.2.4
- https://iupac.qmul.ac.uk/fusedring/FR23.html
- HERITAGE-1990 §4 (Wisniewski J. Chem. Inf. Comput. Sci. 30, 324-332)
- V18_MILESTONE_PLAN.md Appendix A.6 lines 2280-2422

Complete 10-criterion cascade:
(a) Heteroatom type: N > F > Cl > Br > I > O > S > Se > Te > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Hg
(b) Greater number of rings
(c) Larger ring at first point of difference (decreasing size)
(d) Greater total heteroatom count
(e) Greater heteroatom variety
(f) Heteroatoms by alt priority order
(g) Preferred orientation (FR-5.2 fallback: symmetry lower-locant)
(h) Lower locants for heteroatoms
(i) Locant ordering by heteroatom type
(j) Lower bridgehead carbon locants

Implementation depth (Phase 149 D-03):
  - (a)-(f) FULL implementation per IUPAC P-25.3.2.4 + V18 Appendix A.6.
  - (g)-(j) deterministic stubs (returning constants); Phase 155 fills with
    FR-5.2 orientation + peripheral numbering.

DRY discipline (Phase 149 D-04 / D-15):
  - _HETEROATOM_SENIORITY (P-18(b) errata-correct, 20 entries) is REUSED via
    a relative import from ring_selection for FR-2.3(a) — see imports below.
  - _FR23_HETEROATOM_ALT (NEW, 19 entries) lives here for FR-2.3(f); it is
    genuinely different from _HETEROATOM_SENIORITY (Hg present, Al/Ga absent,
    N at rank 12 not 20) per V18 Appendix A.6 lines 2319-2325.
"""
from dataclasses import dataclass, field
from typing import FrozenSet, List, Set, Tuple

from rdkit import Chem

from .ring_selection import _HETEROATOM_SENIORITY  # P-18(b) errata-correct, 20 entries; D-04


# ============================================================================
# FR-2.3(f) Alt Heteroatom Order
# ============================================================================

# FR-2.3(f) alt heteroatom order (different from (a)).
# Source: V18_MILESTONE_PLAN Appendix A.6 lines 2318-2325; QMUL FR-2.3.
# Note: Hg present (rank 2); Al / Ga absent — differs from
# _HETEROATOM_SENIORITY by design (D-04 lock).
_FR23_HETEROATOM_ALT = {
    'F': 20, 'Cl': 19, 'Br': 18, 'I': 17,
    'O': 16, 'S': 15, 'Se': 14, 'Te': 13,
    'N': 12, 'P': 11, 'As': 10, 'Sb': 9, 'Bi': 8,
    'Si': 7, 'Ge': 6, 'Sn': 5, 'Pb': 4,
    'B': 3, 'Hg': 2,
}

# Tuple in alt-priority order, used by (f) computation in _rank.
# Keep parallel with _FR23_HETEROATOM_ALT keys (verify identical via test).
_FR23_ALT_ORDER = (
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N',
    'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Hg',
)


# ============================================================================
# ComponentRank Dataclass
# ============================================================================


@dataclass(frozen=True, order=True)
class ComponentRank:
    """FR-2.3 ranking tuple. Lower sort value = preferred component.

    All fields negated where IUPAC says "more / larger / lower-locant wins"
    so that ``min(sorted([...]))`` selects the IUPAC-preferred candidate.

    Order of fields matches FR-2.3 criteria (a)-(j); Python dataclass
    ``order=True`` generates lexicographic comparison in declaration order.

    Stubs (g)-(j) are deterministic constants in Phase 149 per D-03;
    Phase 155 fills with FR-5.2 orientation + peripheral numbering.

    Source: V18_MILESTONE_PLAN Appendix A.6 lines 2328-2341.
    Source: 149-CONTEXT.md D-02, D-03.
    """
    senior_het_neg: int                                                       # (a)
    ring_count_neg: int                                                       # (b)
    ring_sizes_neg: Tuple[int, ...]                                           # (c) descending sizes, negated
    het_count_neg: int                                                        # (d)
    het_variety_neg: int                                                      # (e)
    alt_het_tuple: Tuple[int, ...] = field(default_factory=tuple)             # (f)
    orient_stub: int = 0                                                      # (g) — Phase 155 fills
    het_locants_stub: Tuple[int, ...] = field(default_factory=tuple)          # (h) — Phase 155 fills
    het_type_locants_stub: Tuple[int, ...] = field(default_factory=tuple)     # (i) — Phase 155 fills
    bridgehead_locants_stub: Tuple[int, ...] = field(default_factory=tuple)   # (j) — Phase 155 fills


# ============================================================================
# Public API + Internal Helpers (Task 01-02 / 01-03 fill bodies)
# ============================================================================


def _rank(mol: Chem.Mol, atoms: Set[int]) -> ComponentRank:
    """Compute FR-2.3 ranking tuple for a component.

    Implements criteria (a)-(f) per V18 Appendix A.6 lines 2368-2421;
    (g)-(j) use dataclass-default constants per D-03 deterministic stubs.

    Source: V18_MILESTONE_PLAN Appendix A.6 lines 2368-2421.
    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
    Source: 149-CONTEXT.md D-02, D-03, D-04.
    """
    ri = mol.GetRingInfo()
    rings_in_component = [
        ring for ring in ri.AtomRings() if set(ring).issubset(atoms)
    ]

    # (a) most senior heteroatom — primary order via _HETEROATOM_SENIORITY
    # (REUSED from ring_selection per D-04 — NOT _FR23_HETEROATOM_ALT;
    # the alt order is the (f) tail-tiebreaker, not the (a) primary).
    senior_het = 0
    for idx in atoms:
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        rank = _HETEROATOM_SENIORITY.get(sym, 0)
        if rank > senior_het:
            senior_het = rank

    # (b) ring count
    ring_count = len(rings_in_component)

    # (c) ring sizes in decreasing order, negated
    sizes_desc = sorted((len(r) for r in rings_in_component), reverse=True)
    sizes_neg = tuple(-s for s in sizes_desc)

    # (d) heteroatom count
    het_count = sum(
        1 for idx in atoms
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
    )

    # (e) heteroatom variety
    # FR-2.3(e) is set cardinality per V18 Appendix A.6 + IUPAC text;
    # _HETEROATOM_VARIETY_ORDER from ring_selection.py is the P-44.2.1(g)
    # per-element-count vector and is NOT applicable here per CONTEXT D-15
    # audit (different IUPAC clause). FR-2.3(e) asks "how many distinct
    # heteroatom species" → set cardinality; P-44.2.1(g) asks "what's the
    # per-element-count vector" → tuple alignment. The two functions are
    # named similarly but answer different questions; reuse would be wrong.
    het_types = {
        mol.GetAtomWithIdx(idx).GetSymbol()
        for idx in atoms
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
    }
    variety = len(het_types)

    # (f) alt heteroatom order — count by each element in alt priority
    alt_counts = []
    for elem in _FR23_ALT_ORDER:
        n = sum(
            1 for idx in atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() == elem
        )
        alt_counts.append(-n)  # negate for min-sort

    # (g)-(j) P-25.3.2.4 tail tiebreaks (BlueBookV2.md:12317/12392/12407/12418).
    # These decide ONLY when (a)-(f) tie; they are per-component structural
    # descriptors so they are invariant to the input SMILES atom order.

    # (g) greatest number of rings in a horizontal row (preferred orientation).
    # For a single monocyclic component the horizontal-row count is 1; for a
    # multi-ring component, approximate the linear-fusion span by the longest
    # chain of ortho-fused rings sharing collinear fusion bonds. Fail-safe: use
    # the ring count as an upper bound (never mis-ranks a lone monocycle). Negate.
    horiz = _horizontal_row_count(mol, rings_in_component)

    # (h) lower locants for heteroatoms (kind-agnostic, as a set). Use the
    # component-internal canonical numbering (heteroatom-priority walk) so the
    # locant multiset is spelling-invariant. Lower is better -> store ascending
    # and compare directly (tuple min-sort).
    het_locants = _component_heteroatom_locants(mol, atoms, rings_in_component)

    # (i) lower locants for heteroatoms in the seniority order
    # F>Cl>Br>I>O>S>Se>Te>N>P>... (reuse _HETEROATOM_SENIORITY key ordering).
    het_type_locants = _component_het_type_locants(mol, atoms, rings_in_component)

    # (j) lower locants for peripheral fusion-carbon atoms.
    bridgehead_locants = _component_fusion_carbon_locants(mol, atoms,
                                                          rings_in_component)

    return ComponentRank(
        senior_het_neg=-senior_het,
        ring_count_neg=-ring_count,
        ring_sizes_neg=sizes_neg,
        het_count_neg=-het_count,
        het_variety_neg=-variety,
        alt_het_tuple=tuple(alt_counts),
        orient_stub=-horiz,                    # (g) more rows preferred -> negate
        het_locants_stub=tuple(het_locants),   # (h) lower set preferred -> ascending
        het_type_locants_stub=tuple(het_type_locants),  # (i)
        bridgehead_locants_stub=tuple(bridgehead_locants),  # (j)
    )


def _horizontal_row_count(mol, rings_in_component):
    """(g) P-25.3.2.4: rings in a horizontal row in the preferred orientation.
    Fail-safe approximation: the longest run of ortho-fused rings in the
    component (a lower bound on the true horizontal-row span; for a monocycle
    this is 1). Used only as an (a)-(f) tail tiebreak, never a primary."""
    if len(rings_in_component) <= 1:
        return len(rings_in_component)
    # ortho-fusion adjacency: two rings sharing exactly one bond (2 atoms)
    idx = list(range(len(rings_in_component)))
    sets = [set(r) for r in rings_in_component]
    adj = {i: [] for i in idx}
    for i in idx:
        for j in idx:
            if i < j and len(sets[i] & sets[j]) == 2:
                adj[i].append(j)
                adj[j].append(i)
    # longest simple path over the (small) fusion graph
    best = 1

    def dfs(u, seen):
        nonlocal best
        best = max(best, len(seen))
        for v in adj[u]:
            if v not in seen:
                dfs(v, seen | {v})

    for s in idx:
        dfs(s, {s})
    return best


def _component_numbering(mol, atoms, rings_in_component):
    """Spelling-invariant per-component locant map: number the component by a
    heteroatom-priority canonical walk (locant 1 -> most senior heteroatom or,
    for carbocycles, the RDKit-canonical-rank-lowest atom). Returns
    {atom_idx: locant}. Kept local + deterministic (uses canonical ranks, not
    input order) so (h)-(j) tuples do not depend on the SMILES spelling."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    ordered = sorted(atoms, key=lambda a: (
        -_HETEROATOM_SENIORITY.get(mol.GetAtomWithIdx(a).GetSymbol(), 0),
        ranks[a]))
    return {a: i + 1 for i, a in enumerate(ordered)}


def _component_heteroatom_locants(mol, atoms, rings_in_component):
    """(h) ascending locants of ALL heteroatoms (kind-agnostic)."""
    numb = _component_numbering(mol, atoms, rings_in_component)
    locs = [numb[a] for a in atoms
            if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    return sorted(locs)


def _component_het_type_locants(mol, atoms, rings_in_component):
    """(i) locants grouped by heteroatom seniority order (senior element's
    locants first, ascending within each element)."""
    numb = _component_numbering(mol, atoms, rings_in_component)
    out = []
    for elem in sorted({mol.GetAtomWithIdx(a).GetSymbol() for a in atoms
                        if mol.GetAtomWithIdx(a).GetAtomicNum() != 6},
                       key=lambda e: -_HETEROATOM_SENIORITY.get(e, 0)):
        out.extend(sorted(numb[a] for a in atoms
                          if mol.GetAtomWithIdx(a).GetSymbol() == elem))
    return out


def _component_fusion_carbon_locants(mol, atoms, rings_in_component):
    """(j) ascending locants of peripheral fusion carbons (carbons shared by
    >=2 rings of the component)."""
    numb = _component_numbering(mol, atoms, rings_in_component)
    shared = set()
    sets = [set(r) for r in rings_in_component]
    for a in atoms:
        if mol.GetAtomWithIdx(a).GetAtomicNum() != 6:
            continue
        if sum(1 for s in sets if a in s) >= 2:
            shared.add(a)
    return sorted(numb[a] for a in shared)


def _enumerate_components(mol: Chem.Mol) -> List[FrozenSet[int]]:
    """Enumerate fusion components per IUPAC P-25.3.1.3.

    Per D-05: SSSR rings as base; multi-piece decomposition deferred
    to Phase 151 (spiro / fused-polycyclic + side-ring scope).

    Each SSSR ring is one "component". MONOCYCLIC_COMPONENTS recognition
    happens INSIDE _rank when computing per-component descriptors; recognition
    is NOT a hard gate — FR-2.3 ranks any component, recognized or not (per
    RESEARCH §"Critical insight" line 274).

    Returns list of frozensets per CD-04 (frozenset for hashability +
    ordering stability per Phase 145.2 determinism doctrine).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.1.3
    Source: 149-CONTEXT.md D-05, CD-04.
    Source: 149-RESEARCH.md "Component Decomposition Algorithm".
    """
    ri = mol.GetRingInfo()
    return [frozenset(ring) for ring in ri.AtomRings()]


def select_base_component(
    mol: Chem.Mol,
    fused_components: List[Set[int]],
) -> Tuple[Set[int], List[Set[int]]]:
    """Select preferred base component per FR-2.3.

    Args:
        mol: Full RDKit molecule.
        fused_components: List of atom sets, each set is one fusion component.
            Must have >=2 components for fusion naming to apply.

    Returns:
        (base_component, other_components_ordered_by_rank)

    Raises:
        ValueError: if fewer than 2 components.

    Source: V18_MILESTONE_PLAN §6 Phase 149 SC #1.
    Source: 149-CONTEXT.md D-06.
    """
    if len(fused_components) < 2:
        raise ValueError("Need >=2 components for fusion naming")

    ranked = sorted(fused_components, key=lambda c: _rank(mol, c))
    return ranked[0], ranked[1:]
