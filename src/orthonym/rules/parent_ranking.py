""" (P4-b): the Blue-Book-ranked parent-candidate SET.

WHY THIS MODULE EXISTS
======================
Production is **early parent commit**: each handler selects its own parent
internally and adds exactly ONE candidate to the pool, and the pool's *ranking*
path is dead (``selection_mode='first_applicable'``,
``assembly/candidate_pool.py:135``, and the module says so at ``:953-955``). So
there was never a candidate set to filter — building the ranked set is the work.
See `` Part A1/A3 and Part G.

WHAT THE BLUE BOOK ACTUALLY LICENSES (derivation Part B1 + B1a)
===============================================================
There is **no** escape clause conditioning the choice of parent on whether a
name for it can be constructed. Every P-44/P-45 criterion is a property of the
structure or of the candidate name string; nameability is never among them:

    BB:18873 — P-44.1 SENIORITY ORDER FOR PARENT STRUCTURES — "When there is a
    choice, the senior parent structure is chosen by applying the following
    criteria, in order, until a decision is reached. These criteria must always
    be applied before those applicable to rings and ring systems (see P-44.2)
    and to chains (see P-44.3)."

    BB:24096 — P-52.2.8 Selection between a ring and a chain as parent hydride —
    "Within the same heteroatom class and for the same number of characteristic
    groups cited as the principal characteristic group, **a ring is always
    selected as the parent hydride to construct a preferred IUPAC name. In
    general nomenclature, a ring or a chain can be the parent hydride.**"

Consequently a fall-through to a lower-ranked parent is **never** a PIN. Its
correct status is fixed by:

    BB:24623 — P-58.1 INTRODUCTION — "… **Preferred IUPAC names are generated
    under the condition that the name of the parent structure and the names of
    all or part of components are preferred IUPAC names. When this condition is
    not fulfilled** and when the names of components are acceptable for general
    nomenclature, **the resulting names of the compounds are acceptable only for
    general nomenclature.**"

PIN status is therefore *conditional and compositional*. A rank>0 emission is a
**general IUPAC name** (BB:1938) — T3/T4, never ``is_pin: True``. ``namer.
name_tiered`` already enforces that for every ``source == 'general_engine'``
emission; ``record_parent_fallthrough`` makes the reason auditable rather than
merely correct.

But tier 2 is not the floor. BB:1982 marks discarded names ``not``, and those
are "no longer recommended" — emitting one is an **accuracy defect, not a
label**. ``is_bluebook_discarded_name`` is the fail-closed veto for that.

CASCADE SHAPE, AND WHY P-44.4 IS NOT A CROSS-CLASS TERM HERE
============================================================
P-44.4 applies only "If the criteria of P-44.1 through P-44.3 … do not effect a
choice" (BB:21016). For a ring-vs-chain pair P-44.1.2.2 **always** effects a
choice (ring wins, BB:24096 "always"), so the cross-class P-44.4.1(a)–(l) list
can only ever fire *within* a class — where both existing scorers already
implement it (``ring_system_score`` [27]/[28]; ``chain_score`` elements 3/4).
That is why this module needs no new P-44.4 comparator, and it is a derivation
result, not an omission.

Ranking key (all terms "lower is senior", so ``sorted`` ⇒ most senior first):

    k0 -pcg_count P-44.1.1 (BB:18875) max principal-characteristic
                               groups — precedes everything, ring or chain
    k1 -senior_atom_rank P-44.1.2 (BB:18917) senior skeletal atom, in
                               P-44.1.2's OWN element order (N>P>…>O>S>…>C)
    k2 0 ring / 1 chain P-44.1.2.2 + P-52.2.8 (BB:24096) ring always wins
    k3 within-class score rings: P-44.2.1(a)–(g) then P-44.2.2 type then
                               P-44.4.1(a),(b) — ``ring_system_score``, verified
                               BB-faithful (derivation Part H1).
                               chains: ``chain_score``'s order (Part H2).
    k4 structural tiebreak P-45.5 alphanumerical order is the Blue Book's own
                               last resort (BB:22234/22250); the chain scorer
                               already applies it (``chains._p45_alpha_key``).
                               This final term makes the order TOTAL using
                               RDKit **canonical ranks**, which are invariant to
                               SMILES spelling — never atom-input or set/dict
                               iteration order (``determinism_new`` must stay 0).

⚠ FOUR DISTINCT ELEMENT SEQUENCES (derivation Part B8). P-44.1.2, P-44.2.1(c),
P-44.2.1(g)/P-44.4.1(f) and P-44.3(c) are four *different* orders; sharing one
table is a correctness bug. This module uses ``P_44_1_2_ELEMENT_RANK`` for k1
**only** — that table implements P-44.1.2 and nothing else.

RANK 1 IS THE INCUMBENT, BY CONSTRUCTION
========================================
Rank 0 of the returned list is always the parent production already committed to
(``select_principal_ring_system`` / ``features.principal_chain``). That makes
the fall-through *purely additive*: it can only ever try parents that today's
code never reaches, so best-effort output is byte-identical whenever rank 0
still passes its gates, and the PIN path never runs this code at all
(``general_fallback`` defaults False, ``namer.py:1815``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

# Enumeration guard. NOT a nameability pre-filter (the derivation forbids
# those) — a combinatorial one: chain DFS enumerates every simple path, which is
# exponential in branching. Exceeding it is LOGGED with the drop count so a cap
# can never read as "covered everything" (see ``rank_parent_candidates``).
MAX_CHAIN_CANDIDATES: int = 400
MAX_CHAIN_ENUM: int = 4000


@dataclass(frozen=True)
class ParentCandidate:
    """One candidate parent structure, with its Blue-Book rank.

    ``atoms`` is sorted for a ring system and **ordered/oriented** for a chain
    (the chain's numbering direction is part of the candidate).
    ``rank`` 0 is the incumbent — the parent production already chose.
    """

    kind: str                      # 'ring' | 'chain'
    atoms: Tuple[int, ...]
    rank: int
    key: Tuple[Any, ...]
    is_incumbent: bool


# --------------------------------------------------------------------------- #
# BB:1982 tier-3 veto #
# --------------------------------------------------------------------------- #
def is_bluebook_discarded_name(name: Optional[str]) -> bool:
    """True if ``name`` is one the Blue Book marks ``not`` (BB:1982).

    Exact match over the lowercase/whitespace-collapsed name, against the corpus
    extracted by ``. Exact-only is
    deliberate: this predicate may only ever *suppress* an emission, so a
    substring or prefix rule (which could veto a correct name) is unacceptable
    while a miss (the veto not firing) is merely a known limitation.
    """
    if not name:
        return False
    from ..data.bluebook_not_names import BLUEBOOK_NOT_NAMES
    return " ".join(str(name).strip().lower().split()) in BLUEBOOK_NOT_NAMES


# --------------------------------------------------------------------------- #
# Ranking terms #
# --------------------------------------------------------------------------- #
def _pcg_count(mol, features, atoms: Set[int]) -> int:
    """P-44.1.1 (BB:18875): principal-characteristic-group instances borne.

    An instance counts for a candidate when any of its match atoms lies on the
    candidate — the same "legacy whole-match" semantics
    ``chains.chain_score``'s ``fg_bearing_carbons`` fallback uses, so a ring and
    a chain candidate are counted on one rule.
    """
    matches = getattr(features, 'principal_group_atoms', None) or []
    seen: Set[Tuple[int, ...]] = set()
    n = 0
    for match in matches:
        key = tuple(sorted(match))
        if key in seen:
            continue
        seen.add(key)
        if set(match) & atoms:
            n += 1
    return n


def _senior_atom_rank(mol, atoms: Set[int]) -> int:
    """P-44.1.2 (BB:18917): rank of the most senior skeletal atom present.

    Returns a value where HIGHER is more senior, so the key negates it. Uses
    P-44.1.2's own element order (N > P > As > … > O > S > Se > Te > C) via
    ``candidate_pool.P_44_1_2_ELEMENT_RANK`` — the table that implements exactly
    this rule. Per derivation Part B8 it must NOT be reused for the ring or
    chain cascades, which have their own, different sequences.
    """
    from ..assembly.candidate_pool import (
        P_44_1_2_ELEMENT_RANK, _P_44_1_2_SENTINEL_RANK,
    )
    best = _P_44_1_2_SENTINEL_RANK
    for idx in atoms:
        r = P_44_1_2_ELEMENT_RANK.get(
            mol.GetAtomWithIdx(idx).GetSymbol(), _P_44_1_2_SENTINEL_RANK)
        if r < best:
            best = r
    # invert so "higher = more senior"
    return _P_44_1_2_SENTINEL_RANK - best


def _structural_key(canon_ranks: Sequence[int],
                    atoms: Sequence[int]) -> Tuple[int, ...]:
    """Total-order tiebreak that is invariant to SMILES spelling.

    RDKit canonical ranks are a function of the molecular graph, so this key is
    stable across respellings of the same structure — unlike atom indices, set
    iteration order, or enumeration order. Required by the ``determinism_new``
    invariant (derivation Part D item 6).
    """
    return tuple(sorted(canon_ranks[i] for i in atoms))


# --------------------------------------------------------------------------- #
# Enumeration #
# --------------------------------------------------------------------------- #
def _dedupe_chains(chains: List[List[int]]) -> List[List[int]]:
    """Collapse a chain and its reverse, and repeated atom sets, keeping the
    first-enumerated representative (so the incumbent's own orientation, which
    is applied later, is never perturbed by dedupe order)."""
    seen: Set[frozenset] = set()
    out: List[List[int]] = []
    for c in chains:
        k = frozenset(c)
        if len(k) != len(c) or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def enumerate_parent_candidates(mol, features) -> Tuple[List[Set[int]],
                                                        List[List[int]], int]:
    """All ring systems and all skeletal chains — NO nameability pre-filter.

    Returns ``(ring_systems, chains, n_chains_dropped)``. The Blue Book's
    cascade never consults nameability (derivation Part B1), so a candidate is
    dropped here only for combinatorial reasons, and the count is returned so
    the caller can log it rather than let a silent cap read as full coverage.
    """
    from ..perception.chains import find_all_skeletal_chains
    from ..perception.rings import get_ring_systems

    ring_systems = [set(rs) for rs in
                    (getattr(features, 'ring_systems', None)
                     or get_ring_systems(mol))]
    ring_atoms: Set[int] = set()
    for rs in ring_systems:
        ring_atoms |= rs

    raw = find_all_skeletal_chains(
        mol, min_length=2,
        exclude_atoms=ring_atoms or None,
        max_chains=MAX_CHAIN_ENUM,
    )
    chains = _dedupe_chains(raw)
    dropped = 0
    if len(chains) > MAX_CHAIN_CANDIDATES:
        # Keep the LONGEST chains: P-44.3(b) skeletal-atom count is the only
        # P-44.3 term that can be evaluated before scoring, and a shorter chain
        # can outrank a longer one only via P-44.3(a)/(c) heteroatom counts,
        # which the truncation preserves by sorting on (hetero, length).
        chains.sort(key=lambda c: (
            -sum(1 for i in c if mol.GetAtomWithIdx(i).GetSymbol() != 'C'),
            -len(c)))
        dropped = len(chains) - MAX_CHAIN_CANDIDATES
        chains = chains[:MAX_CHAIN_CANDIDATES]
    return ring_systems, chains, dropped


# --------------------------------------------------------------------------- #
# The ranked set #
# --------------------------------------------------------------------------- #
def rank_parent_candidates(mol, features) -> List[ParentCandidate]:
    """The Blue-Book-ranked parent-candidate set, most senior first.

    Rank 0 is the parent production already committed to (see the module
    docstring); ranks 1..N are the candidates today's code never reaches,
    ordered by the P-44 cascade with a spelling-invariant final tiebreak.
    """
    from ..perception.chains import find_principal_chain
    from .ring_selection import ring_system_score, select_principal_ring_system

    ring_systems, chains, dropped = enumerate_parent_candidates(mol, features)
    if dropped:
        logger.info(
            "parent_ranking: chain enumeration capped at %d, DROPPED %d "
            "candidates (not a nameability filter -- a combinatorial one)",
            MAX_CHAIN_CANDIDATES, dropped)

    canon_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    # ---- the incumbent -------------------------------------------------- #
    chain_is_parent = bool(getattr(features, 'chain_is_parent', False))
    incumbent_chain = tuple(getattr(features, 'principal_chain', None) or ())
    incumbent: Optional[Tuple[str, Tuple[int, ...]]] = None
    if ring_systems and not chain_is_parent:
        senior = select_principal_ring_system(mol, ring_systems)
        if senior:
            incumbent = ('ring', tuple(senior))
    if incumbent is None and len(incumbent_chain) >= 2:
        incumbent = ('chain', incumbent_chain)

    # ---- ranked chain candidates, reusing the production comparator ------ #
    ranked_chains: List[List[int]] = []
    if chains:
        try:
            ranked_chains = find_principal_chain(
                mol,
                getattr(features, 'functional_groups', None) or {},
                getattr(features, 'principal_group', None),
                exclude_atoms=_ring_atoms_of(ring_systems) or None,
                _rank_candidates=chains,
            ) or []
        except Exception as e:            # fail-closed: no chain candidates
            logger.info("parent_ranking: chain ranking failed (%s)", e)
            ranked_chains = []
    chain_order = {frozenset(c): i for i, c in enumerate(ranked_chains)}

    # ---- score every candidate ------------------------------------------ #
    scored: List[Tuple[Tuple[Any, ...], str, Tuple[int, ...]]] = []
    for rs in ring_systems:
        atoms = tuple(sorted(rs))
        scored.append((
            (-_pcg_count(mol, features, rs),
             -_senior_atom_rank(mol, rs),
             0,                                  # P-44.1.2.2: ring
             ring_system_score(mol, rs),
             _structural_key(canon_ranks, atoms)),
            'ring', atoms))
    for c in ranked_chains:
        cset = set(c)
        atoms = tuple(c)
        scored.append((
            (-_pcg_count(mol, features, cset),
             -_senior_atom_rank(mol, cset),
             1,                                  # P-44.1.2.2: chain
             (chain_order.get(frozenset(c), len(chain_order)),),
             _structural_key(canon_ranks, atoms)),
            'chain', atoms))

    scored.sort(key=lambda t: t[0])

    # ---- emit, incumbent forced to rank 0 -------------------------------- #
    out: List[ParentCandidate] = []
    if incumbent is not None:
        kind, atoms = incumbent
        key = next((k for k, kd, a in scored
                    if kd == kind and set(a) == set(atoms)), ())
        out.append(ParentCandidate(kind=kind, atoms=atoms, rank=0, key=key,
                                   is_incumbent=True))
    for key, kind, atoms in scored:
        if incumbent is not None and kind == incumbent[0] \
                and set(atoms) == set(incumbent[1]):
            continue
        out.append(ParentCandidate(kind=kind, atoms=atoms, rank=len(out),
                                   key=key, is_incumbent=False))
    return out


def _ring_atoms_of(ring_systems: Sequence[Set[int]]) -> Set[int]:
    atoms: Set[int] = set()
    for rs in ring_systems:
        atoms |= set(rs)
    return atoms


# --------------------------------------------------------------------------- #
# Applying a candidate #
# --------------------------------------------------------------------------- #
def features_for_candidate(features, candidate: ParentCandidate):
    """A shallow features copy whose parent is ``candidate``.

    The three ring producers all resolve their parent through
    ``select_principal_ring_system(mol, features.ring_systems)``, which returns
    its single element verbatim when the list has length 1
    (``ring_selection.py:785-786``) — so restricting ``ring_systems`` is enough
    to steer them, with no producer change at all. The chain producer reads
    ``principal_chain`` / ``chain_is_parent`` / ``atom_to_locant``.

    Returns ``features`` itself for the incumbent, so rank 0 is byte-identical.
    """
    if candidate.is_incumbent:
        return features
    import copy
    f = copy.copy(features)
    if candidate.kind == 'ring':
        f.ring_systems = [set(candidate.atoms)]
        f.chain_is_parent = False
    else:
        f.chain_is_parent = True
        f.principal_chain = list(candidate.atoms)
        f.atom_to_locant = None      # re-derived by the caller
    return f
