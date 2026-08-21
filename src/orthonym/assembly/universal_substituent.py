"""Phase B.2 -- the unconditional recursive substitutive namer core.

``name_universal_substitutive(mol)`` is a NEW, self-contained T4/best-effort
producer: it NEVER declines a hard branch.  Where the existing recursive
substituent namer (``assembly/substituent_enumerator.py``) hits an unnameable
fragment and does ``return None`` / emits the ``'substituent'`` sentinel --
which fails the WHOLE enclosing candidate closed -- this module re-enters
ITSELF on the branch subgraph and always bottoms out at a valid (possibly
ugly, non-PIN) systematic token: a simple leaf prefix, a von-Baeyer/'a'-
replacement skeleton, or a deeper recursive call.  There is no depth cap.

Architecture (``name_subgraph``, NOT copied --
see ``):

    name(component) = parent(spine) + Sum(name(branch_i) rendered as -yl)

Termination (the shrinking-atom-set argument, verbatim from that research):
every recursive call is handed a component that is a PROPER SUBSET of its
caller's component -- the caller's spine is non-empty (>=1 atom) and is
removed before any branch is computed, and branch components are pairwise
disjoint (each BFS is bounded by a monotonically GROWING exclude set built
from already-claimed atoms).  So along any recursion path the component size
strictly decreases every call; recursion cannot be infinite.  Neither
reference bounds the WORK this can cost on a pathological input (has no budget at all; mitigation is a depth>=1 local decline, exactly
the failure mode this module must NOT reproduce), so this module adds an
explicit in-algorithm atom/node WORK BUDGET (``_Budget``, charged by
component size at every call) that fails CLOSED -- returns ``None`` -- when
exceeded.  This is a counter, not a signal/timeout: it can never hang.

Coverage-by-construction (the 0-wrong carry): every ``_ComponentResult``
carries ``covers`` -- the exact set of heavy-atom indices it accounts for --
and the top-level entry point asserts the union of the flat binding list
equals the FULL heavy-atom set of the input before returning anything.  Any
gap voids the candidate (``None``), never a partial name.

Scope of THIS module (be honest about what is built vs. deferred, per the
task brief): parent selection covers acyclic chains (own longest-path/tree
walk), single rings (own minimal cyclo-'a'-replacement numbering) and
polycyclic ring systems (reusing ``rules.vonbaeyer_universal.
analyze_cage_universal`` + ``rules.polycyclic._build_parent_with_unsaturation``
for the parent TEXT only -- NOT ``general_engine``'s substituent-recursion
tail, which is exactly the declining machinery this module replaces).
Net charge, isotopes, multi-fragment inputs and indicated hydrogen are
OUT OF SCOPE for this task and void the whole call (``None``) -- they are a
later-phase lift (design doc step 4), not a silent wrong emission.  This
module is a PURE PRODUCER: it never calls ``verify_or_none`` and never
ships a name; the caller (a later wiring task) is responsible for gating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ..perception.stereo import assign_stereochemistry
from ..perception.rings import get_ring_systems
from .naming_utils import (
    SIMPLE_MULTIPLIERS,
    alpha_sort_key,
    format_substituent_prefix,
)
from ..rules.skeletal_replacement import REPLACEMENT_TERMS, _A_CITATION_ORDER
from ..rules.vonbaeyer_universal import analyze_cage_universal
from ..rules.polycyclic import _build_parent_with_unsaturation


# ---------------------------------------------------------------------------
# Work budget -- mandatory in-algorithm backstop (NOT a signal/timeout).
# ---------------------------------------------------------------------------

#: Generous for any real organic molecule (dev500's largest is well under a
#: few hundred heavy atoms with modest branch nesting); a pathological giant
#: with deep alternating branch nesting re-charges the same atoms once per
#: ancestor call, so this trips in bounded, FAST (no I/O, no subprocess)
#: Python work long before it would hang.
DEFAULT_ATOM_WORK_BUDGET = 20_000


class _BudgetExceeded(Exception):
    """Raised internally when the work budget is exhausted; converted to a
    clean ``None`` at the top-level entry point. Never escapes this module."""


@dataclass
class _Budget:
    remaining: int

    def charge(self, n: int) -> None:
        self.remaining -= n
        if self.remaining < 0:
            raise _BudgetExceeded()


@dataclass
class _Ctx:
    mol: "Chem.Mol"
    ring_systems: List[FrozenSet[int]]
    ring_system_of: Dict[int, int]
    budget: _Budget


@dataclass(frozen=True)
class UniversalResult:
    """Public result: a name string + the atom->token coverage binding."""

    name: str
    bindings: Tuple[Tuple[str, FrozenSet[int]], ...]
    covers: FrozenSet[int]


@dataclass
class _ComponentResult:
    """Internal: one recursively-named component (parent OR substituent)."""

    name: str                                    # assembled name, NOT yet -yl
    bindings: List[Tuple[str, FrozenSet[int]]]    # flat, covers `covers`
    covers: FrozenSet[int]
    attach_locant: Optional[int]                  # this component's own spine
    #                                              locant of its attach atom,
    #                                              or None if not applicable
    #                                              (top-level call).
    is_prefix_ready: bool = False                 # True for a leaf shortcut
    #                                              (e.g. "carboxy", "oxo"):
    #                                              already a complete
    #                                              substituent prefix --
    #                                              _render_as_substituent
    #                                              must NOT mechanically
    #                                              append -yl/-ylidene to it.


# ===========================================================================
# Public entry point
# ===========================================================================

def name_universal_substitutive(
    mol, atom_work_budget: int = DEFAULT_ATOM_WORK_BUDGET,
) -> Optional[UniversalResult]:
    """Name *mol* unconditionally, or return ``None`` (void -- never partial).

    ``None`` happens for exactly three reasons, all fail-closed:
      1. a scope guard (charged species / isotopes / radicals / wildcard
         atoms / multi-fragment input) -- explicitly out of scope for this
         task, never silently mis-named;
      2. the work budget trips on a pathological input;
      3. the atom-coverage assertion finds a gap (should not happen given
         the construction, but is asserted rather than trusted).
    """
    if mol is None:
        return None
    # Cheap size guard BEFORE any perception work. This is not redundant with
    # the per-call recursive charging below: kekulization, CIP assignment and
    # ring-system perception all run UNCONDITIONALLY on the whole molecule,
    # before the first recursive call/charge ever happens, and are NOT
    # guaranteed safe at arbitrary size -- measured: a 25,000-atom linear
    # chain segfaults inside that pipeline (a C-level stack/recursion limit,
    # not a Python exception the budget's try/except could catch) well
    # before ``_name_component`` charges a single atom. An input already
    # bigger than the whole budget can never finish within it, so refuse it
    # before touching RDKit's heavier graph algorithms at all.
    if mol.GetNumHeavyAtoms() > atom_work_budget:
        return None
    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
    except Exception:
        return None

    if len(Chem.GetMolFrags(work)) != 1:
        return None  # multi-fragment: out of scope for this producer
    if any(a.GetFormalCharge() != 0 for a in work.GetAtoms()):
        # PER-ATOM, not just net, formal charge. Measured: nitroethane
        # (CC[N+](=O)[O-]) has NET charge 0 (a +1/-1 internal pair) but this
        # module does not model formal charge at all -- it fills implicit H
        # from plain valence, so an internally charge-separated atom (nitro,
        # azide, N-oxide, ...) that slipped past a net-charge-only guard was
        # constructed WRONG (measured: the emitted name did not round-trip).
        # Charge handling (P-73/P-74 suffixes) is an explicit later-phase
        # scope item (design doc step 4) -- refuse rather than mis-construct.
        return None
    if any(a.GetIsotope() for a in work.GetAtoms()):
        return None
    if any(a.GetNumRadicalElectrons() for a in work.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 0 for a in work.GetAtoms()):
        return None  # wildcard atom: unverifiable, never claim to name it

    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        return None  # fail closed rather than guess a bond order

    assign_stereochemistry(work)  # canonical CIP path (never raw rdCIPLabeler)

    heavy = frozenset(a.GetIdx() for a in work.GetAtoms() if a.GetAtomicNum() > 1)
    if not heavy:
        return None

    ring_systems = [frozenset(s) for s in get_ring_systems(work, include_spiro=True)]
    ring_system_of: Dict[int, int] = {}
    for i, atoms in enumerate(ring_systems):
        for a in atoms:
            ring_system_of[a] = i

    ctx = _Ctx(mol=work, ring_systems=ring_systems, ring_system_of=ring_system_of,
               budget=_Budget(atom_work_budget))

    try:
        comp = _name_component(ctx, heavy, attach_hint=None, is_top=True)
    except _BudgetExceeded:
        return None
    if comp is None:
        return None

    covers = frozenset(a for _tok, ids in comp.bindings for a in ids)
    if covers != heavy:
        return None  # void: coverage incomplete -- never ship a partial name

    return UniversalResult(name=comp.name, bindings=tuple(comp.bindings), covers=covers)


# ===========================================================================
# The recursive core
# ===========================================================================

def _name_component(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
) -> Optional[_ComponentResult]:
    """Name one component -- the whole molecule (``is_top``) or one branch.

    Termination: ``component`` shrinks strictly on every recursive call (see
    module docstring); the work budget is charged here, once per call, by
    component size, and fails closed on exhaustion.
    """
    ctx.budget.charge(len(component))
    if not component:
        return None

    mol = ctx.mol

    # ---- direct functional-group / leaf shortcuts (branches only) --------
    # Mirrors the reference architecture's "heteroatom-start shortcut" +
    # "direct-functional-group shortcut": a handful of common small groups
    # get their standard prefix directly rather than an ugly generic
    # skeletal-replacement rendering of a 1-3 atom "chain".
    if not is_top and attach_hint is not None:
        shortcut = _leaf_shortcut(mol, component, attach_hint)
        if shortcut is not None:
            token, atoms = shortcut
            return _ComponentResult(
                name=token, bindings=[(token, atoms)], covers=atoms,
                attach_locant=None, is_prefix_ready=True,
            )

    # ---- parent (spine) selection -----------------------------------------
    ring_here = _ring_system_for_component(ctx, component, attach_hint, is_top)
    if ring_here is not None:
        spine_result = _name_ring_spine(ctx, component, ring_here, attach_hint)
    else:
        spine_result = _name_chain_spine(ctx, component, attach_hint)
    if spine_result is None:
        return None
    spine_atoms, spine_core, spine_atom_to_locant, attach_locant = spine_result

    # ---- branches: every off-spine atom, named by RE-ENTERING this SAME
    # function on its own (strictly smaller) subgraph. NO depth cap. -------
    bindings: List[Tuple[str, FrozenSet[int]]] = [
        (spine_core, frozenset(spine_atoms)),
    ]
    prefix_entries: Dict[str, List[int]] = {}
    for s_atom, root, order, branch_atoms in _discover_branches(
        mol, spine_atoms, component,
    ):
        sub = _name_component(ctx, branch_atoms, attach_hint=root, is_top=False)
        if sub is None:
            return None  # never ship a partial name: whole call voids
        rendered = _render_as_substituent(sub, order)
        loc = spine_atom_to_locant[s_atom]
        prefix_entries.setdefault(rendered, []).append(loc)
        bindings.append((rendered, sub.covers))

    prefix_parts = []
    for name, locs in prefix_entries.items():
        locs = sorted(locs)
        prefix_parts.append((alpha_sort_key(name),
                              format_substituent_prefix(name, locs, len(locs))))
    prefix_parts.sort(key=lambda t: t[0])
    joined = "-".join(p for _k, p in prefix_parts)

    if joined:
        # P-16.5 hyphen glue: a joining '-' is needed only before a
        # locant-initial core (mirrors general_engine._emit_ring_from_analysis).
        glue = "-" if spine_core[:1].isdigit() else ""
        full_name = joined + glue + spine_core
    else:
        full_name = spine_core

    covers = frozenset(a for _t, ids in bindings for a in ids)
    return _ComponentResult(
        name=full_name, bindings=bindings, covers=covers,
        attach_locant=attach_locant,
    )


def _render_as_substituent(sub: _ComponentResult, bond_order: int) -> str:
    """Rewrite a component's parent name as a ``-yl``/``-ylidene``/``-ylidyne``
    substituent prefix, cited at ITS OWN attachment locant.

    Mechanical rule (uniform for chain, ring and polycyclic spines alike --
    always VALID, not always the shortest/PIN-preferred spelling): drop the
    trailing parent-hydride ``e`` and append ``-{locant}-yl`` (or
    ``-ylidene``/``-ylidyne`` for a double/triple exocyclic attachment).

    A ``_leaf_shortcut`` result (``is_prefix_ready``, e.g. ``carboxy``,
    ``hydroxy``, ``oxo``) is ALREADY a complete substituent prefix -- its
    bond order to the parent is already baked into which token was chosen
    (``hydroxy`` for a single-bonded leaf O, ``oxo`` for a double-bonded
    one) -- so it is returned unchanged rather than mechanically suffixed.
    """
    if sub.is_prefix_ready:
        return sub.name
    name = sub.name
    stem = name[:-1] if name.endswith("e") else name
    suffix = {1: "yl", 2: "ylidene", 3: "ylidyne"}.get(bond_order, "yl")
    loc = sub.attach_locant if sub.attach_locant is not None else 1
    return f"{stem}-{loc}-{suffix}"


# ===========================================================================
# Branch discovery (the termination primitive)
# ===========================================================================

def _discover_branches(
    mol, spine_atoms: FrozenSet[int], component: FrozenSet[int],
) -> List[Tuple[int, int, int, FrozenSet[int]]]:
    """Every off-spine branch of *spine_atoms* within *component*.

    Returns ``(spine_atom, branch_root, bond_order, branch_component)``
    tuples. ``branch_component`` is computed by BFS bounded by a
    monotonically GROWING exclude set (starts at ``spine_atoms``, gains each
    branch as it is claimed) -- the shrinking-atom-set termination argument.
    """
    growing_exclude = set(spine_atoms)
    roots: List[Tuple[int, int, int]] = []
    seen_roots = set()
    for s in sorted(spine_atoms):
        for nb in mol.GetAtomWithIdx(s).GetNeighbors():
            j = nb.GetIdx()
            if nb.GetAtomicNum() <= 1:
                continue
            if j in growing_exclude or j not in component or j in seen_roots:
                continue
            seen_roots.add(j)
            bond = mol.GetBondBetweenAtoms(s, j)
            order = round(bond.GetBondTypeAsDouble())
            roots.append((s, j, order))

    out: List[Tuple[int, int, int, FrozenSet[int]]] = []
    for s, j, order in roots:
        if j in growing_exclude:
            continue  # claimed by an earlier branch's BFS already
        branch_component = _bfs_component(mol, j, frozenset(growing_exclude))
        growing_exclude |= branch_component
        out.append((s, j, order, branch_component))
    return out


def _bfs_component(mol, start: int, exclude: FrozenSet[int]) -> FrozenSet[int]:
    """Heavy-atom BFS from *start*, never crossing *exclude*."""
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if nb.GetAtomicNum() <= 1 or j in exclude or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return frozenset(seen)


def _tree_neighbors(ctx: _Ctx, atom: int, component: FrozenSet[int]) -> List[int]:
    """Neighbors of *atom* usable for CHAIN-spine walking: heavy, in
    *component*, and NOT a ring atom (rings are branch ports, not chain
    continuations -- see module docstring on chain/ring composition)."""
    out = []
    for nb in ctx.mol.GetAtomWithIdx(atom).GetNeighbors():
        j = nb.GetIdx()
        if nb.GetAtomicNum() <= 1 or j not in component:
            continue
        if ctx.ring_system_of.get(j) is not None:
            continue
        out.append(j)
    return out


# ===========================================================================
# Ring-system selection
# ===========================================================================

def _ring_system_for_component(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
) -> Optional[FrozenSet[int]]:
    """Which (if any) ring system in *component* is this call's parent spine.

    Top level: ring beats chain whenever ANY ring atoms are present
    (conventional seniority simplification) -- the LARGEST ring system in
    the component, ties broken by lowest minimum atom index.
    Branch: a ring is the spine ONLY when the branch's own attachment atom
    IS a ring atom (P-29.2: substituent numbering starts at the free
    valence) -- otherwise the branch is a chain and any ring further inside
    is itself a deeper branch, discovered recursively.
    """
    candidates = []
    if is_top:
        seen = set()
        for a in component:
            idx = ctx.ring_system_of.get(a)
            if idx is not None and idx not in seen:
                seen.add(idx)
                sys_atoms = ctx.ring_systems[idx]
                if sys_atoms <= component:
                    candidates.append(sys_atoms)
        if not candidates:
            return None
        candidates.sort(key=lambda s: (-len(s), min(s)))
        return candidates[0]
    else:
        if attach_hint is None:
            return None
        idx = ctx.ring_system_of.get(attach_hint)
        if idx is None:
            return None
        sys_atoms = ctx.ring_systems[idx]
        if not (sys_atoms <= component):
            return None
        return sys_atoms


# ===========================================================================
# Ring spine construction (monocyclic own-built + polycyclic reuse)
# ===========================================================================

def _name_ring_spine(
    ctx: _Ctx, component: FrozenSet[int], ring_atoms: FrozenSet[int],
    attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, int], Optional[int]]]:
    mol = ctx.mol
    ri = mol.GetRingInfo()
    sssr_here = [set(r) for r in ri.AtomRings() if set(r) <= ring_atoms]
    if len(sssr_here) >= 2:
        cage = analyze_cage_universal(mol, cage_atoms=set(ring_atoms))
        if cage is None:
            return None
        parent_block = _build_parent_with_unsaturation(
            cage.total_atoms, cage.unsaturation, fg_suffix=None)
        core = cage.hetero_prefix + cage.descriptor + parent_block
        atom_to_locant = dict(cage.atom_to_locant)
        spine = frozenset(cage.cage_atoms)
        attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
        return spine, core, atom_to_locant, attach_locant

    # Monocyclic: build our own minimal cyclo/'a'-replacement name.
    if len(sssr_here) != 1:
        return None  # defensive: a ring system with 0 SSSR rings inside it
    ring_tuple = next((r for r in ri.AtomRings() if set(r) == ring_atoms), None)
    if ring_tuple is None:
        return None
    n = len(ring_tuple)
    base = list(ring_tuple)

    def candidates_for(start_first: Optional[int]):
        cands = []
        starts = range(n) if start_first is None else [
            i for i in range(n) if base[i] == start_first
        ]
        for start in starts:
            rotated = base[start:] + base[:start]
            cands.append(rotated)
            cands.append([rotated[0]] + list(reversed(rotated[1:])))
        return cands

    if attach_hint is not None:
        cands = candidates_for(attach_hint)
    else:
        cands = candidates_for(None)
    if not cands:
        return None

    def score(order):
        hetero = sorted(i + 1 for i, a in enumerate(order)
                        if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)
        unsat = []
        for i in range(n):
            a, b = order[i], order[(i + 1) % n]
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None:
                continue
            bt = round(bond.GetBondTypeAsDouble())
            if bt >= 2:
                unsat.append(i + 1)
        return (hetero, unsat)

    best = min(cands, key=score)
    atom_to_locant = {a: i + 1 for i, a in enumerate(best)}

    double_bonds, triple_bonds = [], []
    for i in range(n):
        a, b = best[i], best[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        bt = round(bond.GetBondTypeAsDouble())
        if bt == 2:
            double_bonds.append(i + 1)
        elif bt == 3:
            triple_bonds.append(i + 1)
    unsaturation = {"double_bonds": double_bonds, "triple_bonds": triple_bonds}

    hetero_prefix = _build_hetero_prefix(mol, best, atom_to_locant)
    parent_block = _build_parent_with_unsaturation(n, unsaturation, fg_suffix=None)
    core = hetero_prefix + "cyclo" + parent_block

    attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
    return frozenset(ring_atoms), core, atom_to_locant, attach_locant


# ===========================================================================
# Chain spine construction (acyclic; own tree-diameter / rooted walk)
# ===========================================================================

def _name_chain_spine(
    ctx: _Ctx, component: FrozenSet[int], attach_hint: Optional[int],
) -> Optional[Tuple[FrozenSet[int], str, Dict[int, int], Optional[int]]]:
    mol = ctx.mol

    if attach_hint is not None:
        path = _farthest_path_from(ctx, attach_hint, component)
    else:
        # Two-BFS tree-diameter technique: BFS from an arbitrary atom finds
        # one end (u) of a longest path; BFS from u finds the other end and,
        # via parent pointers, the path itself.
        u = _farthest_from(ctx, next(iter(sorted(component))), component)
        path = _farthest_path_from(ctx, u, component)
        # Free choice of numbering direction: lowest locants to heteroatoms,
        # then unsaturation, then (P-31 simplified) substituent attachment
        # points -- an all-carbon saturated chain ties on the first two, so
        # without this third tier a branch could be numbered from the wrong
        # end (e.g. 2-methylbutane emitted as "3-methylbutane").
        rev = list(reversed(path))
        if _chain_score(mol, rev, component) < _chain_score(mol, path, component):
            path = rev

    if not path:
        return None
    n = len(path)
    atom_to_locant = {a: i + 1 for i, a in enumerate(path)}

    double_bonds, triple_bonds = [], []
    for i in range(n - 1):
        bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
        if bond is None:
            continue
        bt = round(bond.GetBondTypeAsDouble())
        if bt == 2:
            double_bonds.append(i + 1)
        elif bt == 3:
            triple_bonds.append(i + 1)
    unsaturation = {"double_bonds": double_bonds, "triple_bonds": triple_bonds}

    hetero_prefix = _build_hetero_prefix(mol, path, atom_to_locant)
    parent_block = _build_parent_with_unsaturation(n, unsaturation, fg_suffix=None)
    core = hetero_prefix + parent_block

    attach_locant = atom_to_locant.get(attach_hint) if attach_hint is not None else None
    return frozenset(path), core, atom_to_locant, attach_locant


def _chain_score(mol, order: List[int], component: Optional[FrozenSet[int]] = None):
    hetero = [i + 1 for i, a in enumerate(order)
              if mol.GetAtomWithIdx(a).GetAtomicNum() != 6]
    unsat = []
    for i in range(len(order) - 1):
        bond = mol.GetBondBetweenAtoms(order[i], order[i + 1])
        if bond is not None and round(bond.GetBondTypeAsDouble()) >= 2:
            unsat.append(i + 1)
    branch_locants: List[int] = []
    if component is not None:
        spine_set = set(order)
        for i, a in enumerate(order):
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if nb.GetAtomicNum() > 1 and j in component and j not in spine_set:
                    branch_locants.append(i + 1)
        branch_locants.sort()
    return (hetero, unsat, branch_locants)


def _farthest_from(ctx: _Ctx, start: int, component: FrozenSet[int]) -> int:
    """BFS from *start* over chain-tree neighbors; returns the farthest atom
    reached (ties broken by lowest atom index, for determinism)."""
    dist = {start: 0}
    order_seen = [start]
    stack = [start]
    while stack:
        cur = stack.pop(0)
        for nb in _tree_neighbors(ctx, cur, component):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                order_seen.append(nb)
                stack.append(nb)
    maxd = max(dist.values())
    best = min(a for a in order_seen if dist[a] == maxd)
    return best


def _farthest_path_from(ctx: _Ctx, start: int, component: FrozenSet[int]) -> List[int]:
    """The longest chain-tree path STARTING at *start* (forced root -- this
    is how a branch's attachment atom always becomes its own locant 1)."""
    parent: Dict[int, Optional[int]] = {start: None}
    dist = {start: 0}
    order_seen = [start]
    stack = [start]
    while stack:
        cur = stack.pop(0)
        for nb in _tree_neighbors(ctx, cur, component):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                order_seen.append(nb)
                stack.append(nb)
    maxd = max(dist.values())
    far = min(a for a in order_seen if dist[a] == maxd)
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ===========================================================================
# Skeletal ('a') replacement prefix (P-15.4.3.1 citation order)
# ===========================================================================

def _build_hetero_prefix(mol, spine_order: List[int], atom_to_locant: Dict[int, int]) -> str:
    by_element: Dict[str, List[int]] = {}
    for a in spine_order:
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym in REPLACEMENT_TERMS:
            by_element.setdefault(sym, []).append(atom_to_locant[a])
    if not by_element:
        return ""
    elements = sorted(by_element, key=lambda e: _A_CITATION_INDEX_SAFE(e))
    parts = []
    for el in elements:
        locs = sorted(by_element[el])
        term = REPLACEMENT_TERMS[el]
        count = len(locs)
        mult = SIMPLE_MULTIPLIERS.get(count, "") if count > 1 else ""
        locant_str = ",".join(str(x) for x in locs)
        parts.append(f"{locant_str}-{mult}{term}")
    return "-".join(parts)


def _A_CITATION_INDEX_SAFE(el: str) -> int:
    try:
        return _A_CITATION_ORDER.index(el)
    except ValueError:
        return len(_A_CITATION_ORDER)


# ===========================================================================
# Direct functional-group / leaf shortcuts (branches only)
# ===========================================================================

_LEAF_SINGLE = {
    "F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo",
    "O": "hydroxy", "N": "amino", "S": "sulfanyl",
    "Se": "selanyl", "Te": "tellanyl",
    "P": "phosphanyl", "As": "arsanyl", "Sb": "stibanyl", "Bi": "bismuthanyl",
    "Si": "silyl", "Ge": "germyl", "Sn": "stannyl", "Pb": "plumbyl", "B": "boranyl",
}
_LEAF_DOUBLE = {
    "O": "oxo", "S": "sulfanylidene", "Se": "selanylidene", "Te": "tellanylidene",
    "N": "imino",
}


def _leaf_shortcut(mol, component: FrozenSet[int], attach_hint: int):
    """A handful of common small groups named directly rather than via the
    generic chain/replacement machinery. Returns ``(token, atom_ids)`` or
    ``None`` (fall through to generic construction)."""
    atom = mol.GetAtomWithIdx(attach_hint)
    others = [n.GetIdx() for n in atom.GetNeighbors()
              if n.GetIdx() in component and n.GetAtomicNum() > 1]

    # Single heavy atom leaf: F/Cl/Br/I/-OH/=O/-NH2/=NH/-SH/=S/... .
    # Disambiguate single vs. double attachment from the ACTUAL bond order to
    # the (excluded) outside parent atom -- never guessed from H count, which
    # would misclassify monovalent halogens (0 implicit H, always a single
    # bond).
    if len(component) == 1 and not others:
        outside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in component]
        sym = atom.GetSymbol()
        if len(outside) == 1:
            bond = mol.GetBondBetweenAtoms(attach_hint, outside[0])
            order = round(bond.GetBondTypeAsDouble()) if bond is not None else 1
            if order == 1 and sym in _LEAF_SINGLE:
                return _LEAF_SINGLE[sym], frozenset(component)
            if order == 2 and sym in _LEAF_DOUBLE:
                return _LEAF_DOUBLE[sym], frozenset(component)
        return None

    # cyano: C bonded only to a terminal triple-bonded N.
    if len(component) == 2 and attach_hint in component:
        other = next((a for a in component if a != attach_hint), None)
        if (other is not None and atom.GetSymbol() == "C"
                and mol.GetAtomWithIdx(other).GetSymbol() == "N"):
            bond = mol.GetBondBetweenAtoms(attach_hint, other)
            if bond is not None and round(bond.GetBondTypeAsDouble()) == 3:
                other_others = [n.GetIdx() for n in
                               mol.GetAtomWithIdx(other).GetNeighbors()
                               if n.GetIdx() in component]
                if other_others == [attach_hint]:
                    return "cyano", frozenset(component)

    # carboxy: C(=O)(OH) attached via the carbon, exactly 3 atoms.
    if len(component) == 3 and atom.GetSymbol() == "C" and len(others) == 2:
        o_atoms = [o for o in others if mol.GetAtomWithIdx(o).GetSymbol() == "O"]
        if len(o_atoms) == 2:
            kinds = []
            ok = True
            for o in o_atoms:
                o_atom = mol.GetAtomWithIdx(o)
                o_nbrs = [n.GetIdx() for n in o_atom.GetNeighbors() if n.GetIdx() in component]
                if o_nbrs != [attach_hint]:
                    ok = False
                    break
                bond = mol.GetBondBetweenAtoms(attach_hint, o)
                bt = round(bond.GetBondTypeAsDouble())
                kinds.append((bt, o_atom.GetTotalNumHs()))
            if ok and sorted(kinds) == sorted([(2, 0), (1, 1)]):
                return "carboxy", frozenset(component)

    # nitro: N bonded to exactly two terminal O's, nothing else.
    if len(component) == 3 and atom.GetSymbol() == "N" and len(others) == 2:
        if all(mol.GetAtomWithIdx(o).GetSymbol() == "O" for o in others):
            ok = True
            for o in others:
                o_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(o).GetNeighbors()
                         if n.GetIdx() in component]
                if o_nbrs != [attach_hint]:
                    ok = False
            if ok:
                return "nitro", frozenset(component)

    return None
