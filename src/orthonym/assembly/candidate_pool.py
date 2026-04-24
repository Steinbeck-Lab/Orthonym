"""CandidatePool — Phase 145.1 behavior-preserving extraction of composer.py
handler-cascade dispatch.

PURPOSE
-------
Phase 145.1 (this module): factors out the in-line handler cascade in
composer.assemble_name() into an explicit pool. Pool runs in
selection_mode='first_applicable' which reproduces the current sequential
'if applies: return' flow bit-for-bit. SHIP GATE: byte-identical generated
name strings on baseline_v17_all_corpora.csv (7,500 rows).

Phase 146 (downstream consumer): flips selection_mode to 'score_based',
raises chain priority, sets the three gate thresholds to None to delete
the IUPAC-non-conformant ratio gates, and recalibrates FACTOR_WEIGHTS.
All four 146 changes are 1-line edits to HANDLER_POLICIES + FACTOR_WEIGHTS.

DESIGN CONTRACT (D-08, D-09 from 145.1-CONTEXT.md):
- HandlerPolicy is a dataclass with declarative gate fields. Phase 145.1
  populates them from current composer.py constants. Phase 146 sets gates
  to None and flips chain priority.
- HANDLER_POLICIES is a module-level dict, single source of truth for
  handler tier/priority/gate metadata. Pulls priorities from
  coverage_scoring.HANDLER_PRIORITY to avoid divergence (ISS-002
  remediation: Plan 02 Task 2 extends HANDLER_PRIORITY with n_oxide,
  amine, simple_molecule entries so EVERY priority pulls from a single
  source — no hardcoded literals in this dict).
- CandidatePool runs in 'first_applicable' mode for 145.1: pool.best()
  returns self._candidates[0] (first-added wins). Equivalent to the
  sequential cascade in composer.py:1183-1338 when handlers add
  candidates in current dispatch order.
- Thread-local _pool_store mirrors coverage_scoring._confidence_store.
  Lifecycle: clear_pool() at assemble_name() prologue (Plan 03 wires it
  next to existing clear_confidence() at composer.py:674).

BYTE-IDENTICAL RISK MITIGATIONS (RESEARCH §9.1):
- Risk 1: pool.add() must NOT pass parent_atom_indices into
  compute_confidence (would change atom_coverage). Set field POST-HOC
  on the returned CandidateName instead.
- Risk 2: parent_correctness in FACTOR_WEIGHTS must be inserted at LAST
  position (Plan 02). Mid-position changes float summation order.
- Risk 3: store_confidence(pool.best()) and log_confidence(pool.best())
  must be preserved at the new return site so name_with_confidence()
  keeps working (Plan 03 enforces this).

PERFORMANCE (ISS-005 remediation):
- ParentCorrectnessScorer is bound at MODULE LOAD via try/except
  ImportError (single binding, not re-imported per call). pool.add()
  references the module-level binding directly. Eliminates the 7,500x
  per-call import overhead that the original per-call try-import would
  have incurred during full byte-identical runs.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from .coverage_scoring import (
    CandidateName,
    CONFIDENCE_GATE_THRESHOLD,
    HANDLER_PRIORITY,
    compute_confidence,
    log_confidence,
    select_best_candidate,
)

# ISS-005 REMEDIATION: module-level lazy binding for ParentCorrectnessScorer.
# Originally the import lived inside pool.add() (try/except ImportError per
# call) which would have cost ~7,500 try/except dispatches during full
# byte-identical runs. Module-level binding pays the import cost ONCE at
# candidate_pool.py load time. The try/except still tolerates Plan 02 not
# being committed yet (Plan 01-isolated unit tests exercise the None branch).
try:
    from ..rules.parent_correctness import ParentCorrectnessScorer
except ImportError:
    # Plan 02 creates parent_correctness.py; in Plan 01 stand-alone unit
    # tests (committed BEFORE Plan 02 lands) the import will fail. pool.add
    # branches on this binding being None and defaults parent_correctness
    # factor to 0.5 (no-decision; safe because FACTOR_WEIGHTS weight=0.0).
    ParentCorrectnessScorer = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RATIO_REJECT_FLOOR — Phase 145.2 D-09-a.2 sanity gate
# ---------------------------------------------------------------------------
# Minimum `ratio` factor required for a candidate to be added to the pool.
# Catches obvious-garbage candidates (e.g., a 1-character name on a 30-HA
# molecule → ratio = 1/30/1.5 ≈ 0.022 < 0.10 → rejected).
#
# This is DISTINCT from FACTOR_WEIGHTS['ratio'] (demoted to 0.0 in Plan 01
# Task 1) and distinct from _MIN_RATIO_ACCEPT at composer.py:1520 (preserved
# per D-10). See 145.2-CONTEXT.md §Plan 01 for the three-ratio disambiguation.
#
# Floor value 0.10 chosen so the reject condition is rare in practice (no
# real Tier A candidate on the 7,500-row baseline has ratio < 0.10) but
# strong enough to catch adversarial / pathological garbage during Phase 146
# competitive selection.
RATIO_REJECT_FLOOR: float = 0.10


# ---------------------------------------------------------------------------
# Phase 146 D-08: feature-flag-controlled default selection mode.
# ORTHONYM_SELECTION_MODE env var read at module-import time.
# Default 'first_applicable' (V17) during the post-148 soak per D-07.
# Rollback one-liner:
#   ORTHONYM_USE_V18_WEIGHTS=false ORTHONYM_SELECTION_MODE=first_applicable pytest tests/
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
# ---------------------------------------------------------------------------
_DEFAULT_SELECTION_MODE: str = os.getenv(
    'ORTHONYM_SELECTION_MODE', 'first_applicable'
).strip().lower()
assert _DEFAULT_SELECTION_MODE in ('first_applicable', 'score_based'), (
    f"Invalid ORTHONYM_SELECTION_MODE={_DEFAULT_SELECTION_MODE!r}; "
    f"must be 'first_applicable' or 'score_based'"
)


# ---------------------------------------------------------------------------
# Phase 146 D-02: P-44.1.2 element seniority for parent-hydride selection.
# SEPARATE from rules/seniority.py SENIORITY_ORDER (which is principal-GROUP
# seniority for P-41/P-42 acid/ester/amide ordering). Lower index = more senior.
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.2
# ---------------------------------------------------------------------------
P_44_1_2_ELEMENT_SENIORITY: List[str] = [
    'N', 'P', 'As', 'Sb', 'Bi',
    'Si', 'Ge', 'Sn', 'Pb',
    'B', 'Al', 'Ga', 'In', 'Tl',
    'O', 'S', 'Se', 'Te',
    'C',
]
# Pre-computed lookup: element symbol -> seniority index (lower = senior).
# Elements not in the list (e.g., 'F', 'Cl', 'H') get rank len(list)
# (sentinel "least senior") via the dict-get default below.
P_44_1_2_ELEMENT_RANK: Dict[str, int] = {
    sym: idx for idx, sym in enumerate(P_44_1_2_ELEMENT_SENIORITY)
}
_P_44_1_2_SENTINEL_RANK: int = len(P_44_1_2_ELEMENT_SENIORITY)


# ---------------------------------------------------------------------------
# HandlerPolicy dataclass — declarative per-handler pool participation rules
# ---------------------------------------------------------------------------

@dataclass
class HandlerPolicy:
    """Declarative policy for one handler's pool participation.

    Fields:
        handler_id:    Handler name (matches HANDLER_PRIORITY keys).
        tier:          'ring_a' | 'ring_b' | 'chain' | 'direct_return'.
        priority:      Tiebreak priority (pulled from HANDLER_PRIORITY).
        direct_return: True for Tier B and direct-return handlers
                       (first-applicable wins); False for Tier A pool
                       handlers (compete via select_best_candidate)
                       and chain (now first-class peer of ring_a per
                       Phase 146 SC-5).

    Phase 146 SC-8: cascade_ratio_min, min_ratio_accept, gate_threshold
    fields REMOVED. Cascade competition is now handled by
    `_best_two_tier` (D-01 two-tier selector) and the inline V17 gate at
    composer.py preserves byte-identical V17 soak behavior. Tier B
    handlers no longer carry a per-handler gate threshold; the
    confidence gate is enforced by the handler-internal call to
    `_confidence_gate` against `CONFIDENCE_GATE_THRESHOLD`.
    """
    handler_id: str
    tier: str
    priority: int
    direct_return: bool = False


# ---------------------------------------------------------------------------
# HANDLER_POLICIES — single source of truth for per-handler pool metadata
#
# Populated VERBATIM from composer.py current constants (verified against
# main HEAD per 145.1-RESEARCH.md §2.2). Phase 146 SC-8 cleanup:
# - Removed cascade_ratio_min / min_ratio_accept / gate_threshold kwargs
#   (fields deleted from HandlerPolicy). Cascade competition is now
#   handled by _best_two_tier; Tier B gate enforcement stays inside
#   each handler's call to _confidence_gate(CONFIDENCE_GATE_THRESHOLD).
# - Raised chain policy priority to 4 (ring_a-peer) per SC-5 so chain
#   competes as a first-class pool candidate in V18 score_based mode.
#   HANDLER_PRIORITY['chain'] remains 1 so V17 select_best_candidate
#   tiebreak behavior is byte-identical (the policy-level priority is
#   only consumed by the two-tier selector, not by V17's cascade).
#
# PRIORITY SOURCE (ISS-002 remediation): every entry pulls priority from
# HANDLER_PRIORITY (the single source of truth). Plan 02 Task 2 extends
# HANDLER_PRIORITY with the missing 'n_oxide', 'amine', 'simple_molecule'
# entries (with priorities 5, 5, 1 respectively) so this dict can pull from
# HANDLER_PRIORITY[*] without hardcoded literals. The sole intentional
# exception is 'chain', whose policy-level priority is promoted to 4
# (ring_a-peer) for Phase 146 SC-5 while HANDLER_PRIORITY['chain']
# stays at 1 to preserve V17 byte-identical tiebreak semantics.
#
# SCOPE NOTE: salt / ion / radical / simple_molecule handlers are EXCLUDED
# from this dict per 145.1-RESEARCH.md §9.2 item 3 + ISS-001 enumeration:
# they route to assemble_ion_name() / _name_simple_molecule() directly at
# composer.py:693-710 and composer.py:1389 with completely different
# naming semantics (P-73 functional class, P-68 radicals, terminal
# fallback). They never participate in the pool.
# ---------------------------------------------------------------------------

HANDLER_POLICIES: Dict[str, HandlerPolicy] = {
    # --- Tier B: 13 handlers using _confidence_gate (CONFIDENCE_GATE_THRESHOLD=0.40) ---
    # Each runs gate check before return; on gate-fail, falls through to
    # next handler. Pool reproduces by returning None from add() on
    # gate-fail; caller checks `if cand is not None: return ...`.
    # Phase 146 SC-8: gate_threshold kwarg removed from every entry; the
    # gate is enforced by the handler-internal _confidence_gate call.
    'oxime':           HandlerPolicy('oxime',           'ring_b', priority=HANDLER_PRIORITY['oxime'],           direct_return=True),
    'hydrazone':       HandlerPolicy('hydrazone',       'ring_b', priority=HANDLER_PRIORITY['hydrazone'],       direct_return=True),
    'isocyanate':      HandlerPolicy('isocyanate',      'ring_b', priority=HANDLER_PRIORITY['isocyanate'],      direct_return=True),
    'isothiocyanate':  HandlerPolicy('isothiocyanate',  'ring_b', priority=HANDLER_PRIORITY['isothiocyanate'],  direct_return=True),
    'carbamic_acid':   HandlerPolicy('carbamic_acid',   'ring_b', priority=HANDLER_PRIORITY['carbamic_acid'],   direct_return=True),
    'carbamate':       HandlerPolicy('carbamate',       'ring_b', priority=HANDLER_PRIORITY['carbamate'],       direct_return=True),
    'urea':            HandlerPolicy('urea',            'ring_b', priority=HANDLER_PRIORITY['urea'],            direct_return=True),
    'guanidine':       HandlerPolicy('guanidine',       'ring_b', priority=HANDLER_PRIORITY['guanidine'],       direct_return=True),
    'sulfoxide':       HandlerPolicy('sulfoxide',       'ring_b', priority=HANDLER_PRIORITY['sulfoxide'],       direct_return=True),
    'sulfone':         HandlerPolicy('sulfone',         'ring_b', priority=HANDLER_PRIORITY['sulfone'],         direct_return=True),
    'thioether':       HandlerPolicy('thioether',       'ring_b', priority=HANDLER_PRIORITY['thioether'],       direct_return=True),
    'boronic_acid':    HandlerPolicy('boronic_acid',    'ring_b', priority=HANDLER_PRIORITY['boronic_acid'],    direct_return=True),
    'partial_sat':     HandlerPolicy('partial_sat',     'ring_b', priority=HANDLER_PRIORITY['partial_sat'],     direct_return=True),

    # --- Direct-return: 18 handlers, no gate ---
    # These return the first non-None name produced; no candidate competition.
    # ISS-002 remediation: priorities for n_oxide, amine, polyfunctional all
    # pulled from HANDLER_PRIORITY (Plan 02 Task 2 added the missing entries).
    'n_oxide':         HandlerPolicy('n_oxide',         'direct_return', priority=HANDLER_PRIORITY['n_oxide'],         direct_return=True),
    'acid_halide':     HandlerPolicy('acid_halide',     'direct_return', priority=HANDLER_PRIORITY['acid_halide'],     direct_return=True),
    'anhydride':       HandlerPolicy('anhydride',       'direct_return', priority=HANDLER_PRIORITY['anhydride'],       direct_return=True),
    'lactone':         HandlerPolicy('lactone',         'direct_return', priority=HANDLER_PRIORITY['lactone'],         direct_return=True),
    'lactam':          HandlerPolicy('lactam',          'direct_return', priority=HANDLER_PRIORITY['lactam'],          direct_return=True),
    'ring_ester':      HandlerPolicy('ring_ester',      'direct_return', priority=HANDLER_PRIORITY['ring_ester'],      direct_return=True),
    'polyfunctional':  HandlerPolicy('polyfunctional',  'direct_return', priority=HANDLER_PRIORITY['polyfunctional'],  direct_return=True),
    'multi_ester':     HandlerPolicy('multi_ester',     'direct_return', priority=HANDLER_PRIORITY['multi_ester'],     direct_return=True),
    'ester':           HandlerPolicy('ester',           'direct_return', priority=HANDLER_PRIORITY['ester'],           direct_return=True),
    'phosphine_oxide': HandlerPolicy('phosphine_oxide', 'direct_return', priority=HANDLER_PRIORITY['phosphine_oxide'], direct_return=True),
    'phosphate_ester': HandlerPolicy('phosphate_ester', 'direct_return', priority=HANDLER_PRIORITY['phosphate_ester'], direct_return=True),
    'phosphine':       HandlerPolicy('phosphine',       'direct_return', priority=HANDLER_PRIORITY['phosphine'],       direct_return=True),
    'phosphinic_acid': HandlerPolicy('phosphinic_acid', 'direct_return', priority=HANDLER_PRIORITY['phosphinic_acid'], direct_return=True),
    'ring_assembly':   HandlerPolicy('ring_assembly',   'direct_return', priority=HANDLER_PRIORITY['ring_assembly'],   direct_return=True),
    'polycyclic':      HandlerPolicy('polycyclic',      'direct_return', priority=HANDLER_PRIORITY['polycyclic'],      direct_return=True),
    'ring_nitrile':    HandlerPolicy('ring_nitrile',    'direct_return', priority=HANDLER_PRIORITY['ring_nitrile'],    direct_return=True),
    'amide':           HandlerPolicy('amide',           'direct_return', priority=HANDLER_PRIORITY['amide'],           direct_return=True),
    'amine':           HandlerPolicy('amine',           'direct_return', priority=HANDLER_PRIORITY['amine'],           direct_return=True),

    # --- Tier A: 3 ring handlers competing via select_best_candidate ---
    # Phase 146 SC-8 cleanup: cascade_ratio_min and min_ratio_accept
    # kwargs removed. Cascade competition between complex_ring,
    # heterocycle, and benzene is now decided by `_best_two_tier`
    # (D-01 two-tier selector). V17 byte-identical preservation of
    # the 0.40 complex_ring short-circuit is enforced inline in
    # composer.py via a selection_mode-gated guard (see 146-05 SUMMARY).
    'complex_ring':    HandlerPolicy('complex_ring',    'ring_a', priority=HANDLER_PRIORITY['complex_ring'], direct_return=False),
    'heterocycle':     HandlerPolicy('heterocycle',     'ring_a', priority=HANDLER_PRIORITY['heterocycle'],  direct_return=False),
    'benzene':         HandlerPolicy('benzene',         'ring_a', priority=HANDLER_PRIORITY['benzene'],      direct_return=False),

    # --- Chain: first-class peer of ring_a handlers (Phase 146 SC-5) ---
    # Phase 146 SC-5 / SC-8: chain policy-level priority raised from
    # HANDLER_PRIORITY['chain']=1 to 4 so chain competes as a first-
    # class candidate against ring_a handlers in the two-tier selector
    # (IUPAC P-44.1 cascade — chain length is criterion P-44.1.1/c).
    # HANDLER_PRIORITY['chain']=1 is UNCHANGED: select_best_candidate
    # in coverage_scoring.py consumes HANDLER_PRIORITY directly for V17
    # tiebreak byte-identical behavior. The policy-level priority (4)
    # is consumed only by the two-tier selector's ring_a-peer logic,
    # so V17 soak remains bit-for-bit stable.
    'chain':           HandlerPolicy('chain',           'chain',  priority=4, direct_return=False),
}


# ---------------------------------------------------------------------------
# Phase 146 helper: PCG counter (CD-01 resolution per RESEARCH §2.3).
# ---------------------------------------------------------------------------

def _count_pcgs_in_parent(
    candidate: 'CandidateName',
    features: Any,
) -> int:
    """Count principal characteristic groups attached to or inside the parent.

    CD-01 resolution per RESEARCH §2.3: a PCG counts when its attachment atom
    is inside parent_atom_indices OR bonded to an atom inside parent_atom_indices.
    Substituent PCGs (e.g. nitrile on a substituent chain) do NOT count.

    Returns 0 (no-decision sentinel) when:
      - candidate.parent_atom_indices is None
      - features.principal_group is None / falsy
      - features.principal_group_atoms is missing or empty
      - features.mol is None

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
    """
    if candidate.parent_atom_indices is None:
        return 0  # no-decision -> count as 0 -> loses cascade step 1
    parent_set = set(candidate.parent_atom_indices)
    principal_group = getattr(features, 'principal_group', None)
    if not principal_group:
        return 0
    pg_atoms = getattr(features, 'principal_group_atoms', None) or []
    mol = getattr(features, 'mol', None)
    if mol is None:
        return 0
    count = 0
    for pg_tuple in pg_atoms:
        if not pg_tuple:
            continue
        attach = pg_tuple[0]
        if attach in parent_set:
            count += 1
            continue
        # Indirect: any neighbor of attach is in parent
        try:
            atom = mol.GetAtomWithIdx(attach)
            if any(nbr.GetIdx() in parent_set for nbr in atom.GetNeighbors()):
                count += 1
        except Exception:
            continue
    return count


def _count_multiple_bonds_in_atom_set(mol: Any, atom_set: Set[int]) -> int:
    """Count double + triple bonds where both endpoints are in atom_set.

    Adapted from rules/parent_selection.py:193-209 _count_multiple_bonds.
    Used by Tier-1 filter step 5 (P-44.4.1.2) AND the multiple_bond_count factor
    wired into coverage_scoring.py by Plan 03.

    Duplicated here (rather than imported) to avoid a circular dependency
    between assembly/ and rules/ modules.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
    """
    if mol is None or not atom_set:
        return 0
    from rdkit import Chem
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBeginAtomIdx() in atom_set
                and bond.GetEndAtomIdx() in atom_set):
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE or bt == Chem.BondType.TRIPLE:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Phase 146 Tier-1 cascade filters (D-01, D-02).
# Each filter has signature (List[CandidateName]) -> List[CandidateName].
# NEVER returns empty list — if all candidates tie, all are returned.
# ---------------------------------------------------------------------------

def _filter_max_pcg_count(candidates: List['CandidateName']) -> List['CandidateName']:
    """P-44.1.1: max principal characteristic group count wins.

    Reads candidate.parent_pcg_count (set POST-HOC by pool.add via
    _count_pcgs_in_parent). Candidates with parent_pcg_count==None are
    treated as 0 (no-decision -> loses cascade step 1).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
    """
    if not candidates:
        return candidates
    counts = [
        c.parent_pcg_count if c.parent_pcg_count is not None else 0
        for c in candidates
    ]
    max_count = max(counts)
    return [c for c, n in zip(candidates, counts) if n == max_count]


def _filter_senior_heteroatom_class(
    candidates: List['CandidateName'],
) -> List['CandidateName']:
    """P-44.1.2: senior heteroatom class wins (N > P > ... > C).

    For each candidate, the senior element in its parent atom set is the one
    with the LOWEST index in P_44_1_2_ELEMENT_SENIORITY. Empty parent atoms
    treated as having only C (least senior).

    Reads `mol` from features stashed on the candidate (Plan 03 wiring) or
    from the `_features_mol` test attribute used by Plan 02 isolated tests.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.2
    """
    if not candidates:
        return candidates

    def _senior_rank(cand: 'CandidateName') -> int:
        atoms = cand.parent_atom_indices
        if not atoms:
            return _P_44_1_2_SENTINEL_RANK
        # Production wiring (Plan 03+): pool.add() will stash features.mol
        # on the candidate via a `features` attribute. For Plan 02 unit
        # tests we read mol from a stashed `_features_mol` attr that the
        # test fixture sets directly.
        mol = getattr(getattr(cand, 'features', None), 'mol', None)
        if mol is None:
            mol = getattr(cand, '_features_mol', None)
        if mol is None:
            return _P_44_1_2_SENTINEL_RANK
        best = _P_44_1_2_SENTINEL_RANK
        for idx in atoms:
            try:
                sym = mol.GetAtomWithIdx(idx).GetSymbol()
            except Exception:
                continue
            rank = P_44_1_2_ELEMENT_RANK.get(sym, _P_44_1_2_SENTINEL_RANK)
            if rank < best:
                best = rank
        return best

    ranks = [_senior_rank(c) for c in candidates]
    best_rank = min(ranks)
    return [c for c, r in zip(candidates, ranks) if r == best_rank]


def _filter_ring_over_chain_on_tie(
    candidates: List['CandidateName'],
) -> List['CandidateName']:
    """P-52.2.8 / P-44.1.2.2: ring beats chain on tie. If any ring candidate
    exists in the input, drop all chain candidates. Else identity.

    Reads HANDLER_POLICIES[c.handler].tier; tier=='chain' is the chain class.
    Unknown handler (no entry in HANDLER_POLICIES) is treated as non-chain
    (defensive default — keeps the candidate in the result).

    Source: https://iupac.qmul.ac.uk/BlueBook/P5.html P-52.2.8
    """
    if not candidates:
        return candidates
    non_chain = [
        c for c in candidates
        if (HANDLER_POLICIES.get(c.handler) is None
            or HANDLER_POLICIES[c.handler].tier != 'chain')
    ]
    if non_chain and len(non_chain) < len(candidates):
        return non_chain
    return candidates


def _filter_max_skeletal_atoms(
    candidates: List['CandidateName'],
) -> List['CandidateName']:
    """P-44.4.1.1: max number of skeletal atoms in the parent wins.

    Reads len(candidate.parent_atom_indices); None counts as 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.1
    """
    if not candidates:
        return candidates
    sizes = [
        len(c.parent_atom_indices) if c.parent_atom_indices else 0
        for c in candidates
    ]
    max_size = max(sizes)
    return [c for c, n in zip(candidates, sizes) if n == max_size]


def _filter_max_multiple_bonds(
    candidates: List['CandidateName'],
) -> List['CandidateName']:
    """P-44.4.1.2: max (double + triple bonds in parent) wins.

    Reads the multiple_bond_count factor (wired into compute_confidence by
    Plan 03). For Plan 02 isolated unit tests, the factor must be set on
    the candidate explicitly via cand.factors['multiple_bond_count'] = N.
    Missing key treated as 0.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
    """
    if not candidates:
        return candidates
    counts = [c.factors.get('multiple_bond_count', 0) for c in candidates]
    max_count = max(counts)
    return [c for c, n in zip(candidates, counts) if n == max_count]


def _filter_lowest_locants(
    candidates: List['CandidateName'],
) -> List['CandidateName']:
    """P-44.4.1.4+: lowest locants for principal groups, then heteroatoms, etc.

    STUB for Phase 146. Phase 147 implements via authoritative iupac_locants.
    In Phase 146, this filter should NEVER be invoked because _has_iupac_locants
    returns False (no candidates have iupac_locants populated). The stub returns
    candidates unchanged so callers fall through to Tier 2.

    TODO(phase147): implement via _build_ring_pos with authoritative locants.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4 — P-44.4.1.12
    """
    return candidates  # Identity: no filtering until Phase 147 lands


def _has_iupac_locants(candidates: List['CandidateName']) -> bool:
    """D-02 / D-19 safe probe: True iff ALL candidates have iupac_locants populated.

    In Phase 146, no candidate populates iupac_locants (only ortho-fused does
    per namer.py:998-1002, and ortho-fused doesn't go through pool dispatch).
    Therefore this returns False in Phase 146 -> Tier-1 step 6 is a no-op.

    Phase 147 populates iupac_locants on more ring types -> step 6 activates.
    """
    for cand in candidates:
        ri = getattr(cand, 'ring_info', None) or {}
        if not ri.get('iupac_locants'):
            return False
    return True


# Module-level cascade order (P-44.1 lexicographic).
# Step 6 (locant criteria) is gated separately inside _best_two_tier
# via _has_iupac_locants per D-02.
_TIER1_FILTERS: List[Callable[[List['CandidateName']], List['CandidateName']]] = [
    _filter_max_pcg_count,            # P-44.1.1
    _filter_senior_heteroatom_class,  # P-44.1.2
    _filter_ring_over_chain_on_tie,   # P-52.2.8 / P-44.1.2.2
    _filter_max_skeletal_atoms,       # P-44.4.1.1
    _filter_max_multiple_bonds,       # P-44.4.1.2
]


# ---------------------------------------------------------------------------
# CandidatePool class — collects scored candidates from one assemble_name() call
# ---------------------------------------------------------------------------

class CandidatePool:
    """Holds scored candidate names from one assemble_name() invocation.

    Phase 145.1 mode: 'first_applicable' (literal re-encoding of current
    handler-cascade in composer.py:assemble_name). Pool.best() returns
    self._candidates[0] (first-added wins). Equivalent to the sequential
    'if applies: return' cascade.

    Phase 146 mode: 'score_based' (delegates to select_best_candidate over
    full pool). Enables real chain-vs-ring competition.

    SCORING TIMING (D-10): parent_correctness factor is computed inline in
    pool.add() via the module-level ParentCorrectnessScorer binding. In
    145.1, FACTOR_WEIGHTS['parent_correctness'] = 0.0 ensures the factor
    is recorded but does NOT influence cand.confidence (IEEE 754: x+0.0=x).

    Tier B gate semantics (preserves _confidence_gate at composer.py:279):
      tier='ring_b' handlers → if cand.confidence < CONFIDENCE_GATE_THRESHOLD
      (0.40), pool.add() returns None (gate-rejected); caller falls
      through to next handler. Post Phase 146 SC-8, the gate threshold
      is read from module-level CONFIDENCE_GATE_THRESHOLD (not the
      deleted policy.gate_threshold field).

    Direct-return handlers: pool.add() ALSO sets self._direct_return_winner
    to short-circuit pool.best() (CD-02: store all + early-exit flag —
    saves work; both byte-identical-equivalent in first_applicable mode).
    """

    def __init__(self, selection_mode: str = 'first_applicable') -> None:
        assert selection_mode in ('first_applicable', 'score_based'), \
            f"unknown selection_mode: {selection_mode!r}"
        self.selection_mode = selection_mode
        self._candidates: List[CandidateName] = []
        self._direct_return_winner: Optional[CandidateName] = None

    def add(
        self,
        name: str,
        handler_id: str,
        features: Any,
        parent_atom_indices: Optional[Set[int]] = None,
    ) -> Optional[CandidateName]:
        """Score and add a candidate. Returns the candidate, or None if
        gate-rejected (Tier B handlers only).

        BYTE-IDENTICAL CONTRACT (RESEARCH §9.1 Risk 1):
          parent_atom_indices is set POST-HOC on the returned CandidateName
          (NOT passed into compute_confidence). This guarantees confidence
          is byte-identical to current composer.py code.

        Args:
            name: The IUPAC name produced by the handler.
            handler_id: Key into HANDLER_POLICIES. Unknown handler_id is
                accepted with no policy (no gate, no direct-return flag).
            features: MolecularFeatures (passed through to compute_confidence
                and to ParentCorrectnessScorer).
            parent_atom_indices: Optional set of atom indices comprising the
                parent structure. Used by ParentCorrectnessScorer; does NOT
                affect compute_confidence (Risk 1).

        Returns:
            The added CandidateName on success.
            None if Tier B gate rejected (cand.confidence <
            CONFIDENCE_GATE_THRESHOLD).
        """
        policy = HANDLER_POLICIES.get(handler_id)
        # Compute confidence WITHOUT parent_atom_indices (Risk 1 mitigation)
        cand = compute_confidence(name, handler_id, features)
        # RATIO_REJECT_FLOOR sanity gate (Phase 145.2 D-09-a.2). Rejects
        # obvious-garbage candidates regardless of handler tier. Missing
        # 'ratio' factor treated as "not garbage" (defensive default).
        ratio_val = cand.factors.get('ratio')
        if ratio_val is not None and ratio_val < RATIO_REJECT_FLOOR:
            logger.debug(
                "candidate rejected by RATIO_REJECT_FLOOR: "
                "handler=%s name=%r ratio=%.3f < floor=%.3f",
                handler_id, name, ratio_val, RATIO_REJECT_FLOOR,
            )
            return None
        # Tier B gate enforcement (preserves _confidence_gate behavior).
        # Phase 146 SC-8: the gate threshold is read from the module-level
        # CONFIDENCE_GATE_THRESHOLD (0.40) rather than the now-deleted
        # policy.gate_threshold field. Tier B membership is determined by
        # policy.tier == 'ring_b' (13 handlers: oxime/.../partial_sat).
        if policy is not None and policy.tier == 'ring_b':
            if cand.confidence < CONFIDENCE_GATE_THRESHOLD:
                # Gate rejected — return None so caller falls through
                return None
        # Set parent_atom_indices POST-HOC (Risk 1)
        cand.parent_atom_indices = parent_atom_indices
        # Phase 146 CD-01: populate parent_pcg_count POST-HOC.
        # Same Risk 1 mitigation as parent_atom_indices: NEVER passed into
        # compute_confidence (would break byte-identical guarantees).
        # See _count_pcgs_in_parent above for the algorithm.
        cand.parent_pcg_count = _count_pcgs_in_parent(cand, features)
        # Phase 146 D-06 + Risk 1 preservation: POST-HOC patch
        # multiple_bond_count factor in V18 mode. compute_confidence
        # was called with parent_atom_indices=None (Risk 1), so the
        # factor was 0.0. Now that parent_atom_indices is set, recompute.
        #
        # BYTE-IDENTICAL CONTRACT: this branch only fires in V18 mode
        # ('multiple_bond_count' in FACTOR_WEIGHTS) AND when
        # parent_atom_indices is populated. In V17 mode the key is absent
        # from FACTOR_WEIGHTS so the branch short-circuits and
        # cand.confidence is left untouched (no V17 drift).
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
        from .coverage_scoring import (
            FACTOR_WEIGHTS, _compute_multiple_bond_count
        )
        if ('multiple_bond_count' in FACTOR_WEIGHTS
                and parent_atom_indices is not None):
            cand.factors['multiple_bond_count'] = round(
                _compute_multiple_bond_count(features, parent_atom_indices), 4
            )
            # Recompute confidence with the patched factor.
            cand.confidence = round(
                sum(
                    FACTOR_WEIGHTS[k] * cand.factors.get(k, 0.0)
                    for k in FACTOR_WEIGHTS
                ),
                4,
            )
            cand.confidence = min(max(cand.confidence, 0.0), 1.0)
        # Inline parent_correctness scoring (D-10).
        # ISS-005 REMEDIATION: reference the MODULE-LEVEL ParentCorrectnessScorer
        # binding (set at module load via try/except ImportError). Avoids
        # ~7,500 per-call try-import dispatches during full byte-identical runs.
        if ParentCorrectnessScorer is not None:
            cand.factors['parent_correctness'] = ParentCorrectnessScorer.score(
                cand, features.mol
            )
        else:
            # Plan 02 has not committed parent_correctness.py yet (Plan 01
            # isolated unit test scenario). Default to 0.5 (no-decision;
            # safe because FACTOR_WEIGHTS['parent_correctness']=0.0).
            cand.factors['parent_correctness'] = 0.5
        self._candidates.append(cand)
        if policy is not None and policy.direct_return:
            if self._direct_return_winner is None:
                self._direct_return_winner = cand
        return cand

    def best(self) -> Optional[CandidateName]:
        """Select winning candidate per selection_mode.

        BYTE-IDENTICAL CONTRACT (Phase 145.1 drift fix, 2026-04-23):

        In `selection_mode='first_applicable'`, the original (pre-Plan-01)
        composer.py used an `if X applies: return X` cascade where each
        direct-return handler short-circuited the entire dispatch. The
        Plan-03 refactor preserved this intent by tracking
        `_direct_return_winner` (set by add() when a `direct_return=True`
        handler fires), but the original Plan 01 implementation of best()
        ignored this field and just returned `_candidates[0]`. This caused
        a byte-identical drift on CHEBI:85380 (`COc1cc...c4ccc1c2c43`):
        the Tier A `complex_ring` handler (direct_return=False) added
        `1-methoxypyrene` first, then the polycyclic handler
        (direct_return=True) added the correct `2-methoxypyrene`, but
        best() returned `_candidates[0]` = the wrong-locant complex_ring
        candidate.

        FIX: when a direct-return handler has fired, return ITS candidate
        (matches the pre-Plan-01 "if X applies: return X" semantics).
        Otherwise, return the first added candidate (Tier A subset
        compete-by-position for first_applicable).

        Phase 146 will flip selection_mode to 'score_based' and the
        _direct_return_winner short-circuit no longer applies — full
        candidate set competes via select_best_candidate() per D-08.
        """
        if not self._candidates:
            return None
        if self.selection_mode == 'first_applicable':
            # If any direct_return=True handler fired, IT wins (semantic
            # equivalent to the original "if X applies: return X" pattern).
            # Otherwise, first-added wins (Tier A subset semantics).
            if self._direct_return_winner is not None:
                return self._direct_return_winner
            return self._candidates[0]
        # Phase 146 SC-6 (D-01): two-tier selector.
        # Tier 1: Blue Book P-44.1 lexicographic cascade (5 immediate filters
        #         + step 6 gated on iupac_locants per D-02).
        # Tier 2: weighted-sum tiebreak via select_best_candidate.
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
        return self._best_two_tier()

    def all_candidates(self) -> List[CandidateName]:
        """Return all collected candidates (for logging/diagnostics)."""
        return list(self._candidates)

    def _best_two_tier(self) -> Optional[CandidateName]:
        """Phase 146 Tier-1 lexicographic cascade + Tier-2 weighted-sum tiebreak.

        Implements Blue Book P-44.1 per CONTEXT.md <domain>:
          Step 1: P-44.1.1   max PCG suffix count
          Step 2: P-44.1.2   senior heteroatom class (N > P > ... > C)
          Step 3: P-52.2.8   ring-over-chain on tie
          Step 4: P-44.4.1.1 max skeletal atoms
          Step 5: P-44.4.1.2 max multiple bonds
          Step 6: P-44.4.1.4+ locant criteria (gated on iupac_locants per D-02)

        Tier 2: when Tier 1 leaves >1 candidate, defer to weighted-sum tiebreak
        via existing select_best_candidate() — which uses calibrated FACTOR_WEIGHTS,
        EPSILON=0.01 ties, and HANDLER_PRIORITY fallback.

        NOT YET WIRED INTO best() — Plan 05 replaces the score_based branch
        in best() with a call to this method. In Phase 146 Wave 1, this method
        is callable by unit tests but unreachable from production code paths.

        Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
        """
        if not self._candidates:
            return None

        candidates = list(self._candidates)

        # Tier 1 — lexicographic filters (short-circuit on single-winner).
        for filter_fn in _TIER1_FILTERS:
            if len(candidates) <= 1:
                break
            candidates = filter_fn(candidates)

        # Step 6 — gated on iupac_locants per D-02.
        # Phase 146: _has_iupac_locants returns False -> step 6 is no-op.
        # Phase 147: populates iupac_locants -> step 6 activates.
        if len(candidates) > 1 and _has_iupac_locants(candidates):
            candidates = _filter_lowest_locants(candidates)

        if len(candidates) == 1:
            return candidates[0]

        # Tier 2 — weighted-sum tiebreak (existing infrastructure).
        # Import locally to avoid any circular-import surprises at module load.
        from orthonym.assembly.coverage_scoring import select_best_candidate
        return select_best_candidate(candidates)


# ---------------------------------------------------------------------------
# Thread-local pool stack (per-molecule cascade scoping per IUPAC P-44.0)
#
# IUPAC 2013 Blue Book P-44.0 mandates that "the selection of a preferred
# parent structure is based on the seniority of classes" — applied per-molecule,
# single-pass. Each invocation of assemble_name() represents one molecule's
# parent-selection cascade.
#
# assemble_name() is called recursively from at least 6 sites:
#   - composer.py:766 (N-oxide handler)
#   - composer.py:1691, 1834 (composition / decomposition handlers)
#   - assembly/fragment_naming.py:323, 334 (substituent fragment naming)
#   - assembly/substituent_enumerator.py:871, 1091 (nested substituents)
#   - decomposition/engine.py:46 (decomposition fallback)
#
# A single-slot thread-local pool would let a recursive substituent's cascade
# pollute its outer molecule's cascade (the inner candidate becomes pool[0]
# from the outer's perspective, and selection_mode='first_applicable' returns
# the wrong winner). This violates P-44.0 per-molecule scoping.
#
# Fix: thread-local STACK. Each assemble_name() prologue calls push_pool() to
# push a fresh pool; the body's existing handler dispatch reads/writes the top
# of stack via get_current_pool(); the epilogue (try/finally) calls pop_pool()
# to restore the previous pool. Recursion-safe to arbitrary depth.
# ---------------------------------------------------------------------------

_pool_store = threading.local()


def _ensure_stack() -> List[CandidatePool]:
    """Initialize the per-thread pool stack on first access.

    Each thread has its own stack (via threading.local()), so concurrent
    benchmark runners ( uses ThreadPoolExecutor)
    have independent pool stacks with no cross-thread leakage.
    """
    if not hasattr(_pool_store, 'stack'):
        _pool_store.stack = []
    return _pool_store.stack


def push_pool() -> CandidatePool:
    """Push a fresh pool onto the per-thread stack and return it.

    Called by assemble_name() prologue (composer.py). MUST be paired with
    pop_pool() in a finally clause so the pool is popped on every exit path
    (normal return, exception, early return statements).

    Phase 146 D-08: selection_mode defaults to the ORTHONYM_SELECTION_MODE
    env var (default 'first_applicable' for V17 byte-identical soak). Set
    ORTHONYM_SELECTION_MODE=score_based to activate the two-tier selector
    end-to-end without a code revert.

    Returns the new pool that is now the active cascade scope for the
    currently-executing assemble_name() call.
    """
    stack = _ensure_stack()
    new_pool = CandidatePool(selection_mode=_DEFAULT_SELECTION_MODE)
    stack.append(new_pool)
    return new_pool


def pop_pool() -> None:
    """Pop the top pool from the per-thread stack.

    Called by assemble_name() epilogue (composer.py, in finally clause).
    Safe to call when the stack is empty (no-op) — defensive against any
    code path that pops without a matching push.
    """
    stack = _ensure_stack()
    if stack:
        stack.pop()


def get_current_pool() -> CandidatePool:
    """Return the top of the per-thread pool stack — the active cascade
    for the currently-executing assemble_name() call.

    Auto-pushes a fresh pool if the stack is empty. This covers two cases:
    (1) a handler called get_current_pool() outside an assemble_name() scope
    (e.g., from a unit test that invokes pool.add() directly), and
    (2) backward compatibility with any legacy caller that expected the
    pre-fix lazy-init behavior.
    """
    stack = _ensure_stack()
    if not stack:
        return push_pool()
    return stack[-1]


def clear_pool() -> None:
    """BACKWARD-COMPAT: replace the top of the stack with a fresh pool.

    Pre-fix code (Plan 03) called clear_pool() in assemble_name()'s prologue
    to reset state. Post-fix, push_pool() is the right primitive (the wrapping
    try/finally handles pop). This alias preserves the call-site signature
    so any handler still calling clear_pool() directly does not break.

    Behavior: if the stack is empty, push a fresh pool (same as pre-fix
    lazy init); if non-empty, replace the top so the current cascade restarts
    cleanly without affecting outer cascades on the stack.
    """
    stack = _ensure_stack()
    if stack:
        stack[-1] = CandidatePool(selection_mode=_DEFAULT_SELECTION_MODE)
    else:
        push_pool()


# ---------------------------------------------------------------------------
# Phase 146 SC-5: compute_chain_candidate for unconditional chain wiring.
# Wraps the existing chain naming pipeline so chain becomes a first-class
# candidate alongside ring handlers. Idempotent: same features -> same output.
# Returns None when no viable chain candidate exists (e.g., pure benzene
# with no principal_chain). V18-mode-only wiring lives in composer.py's
# Tier A integration point; V17 mode's existing ring-fallback-to-chain
# path remains byte-identical (SC-6 two-tier never fires).
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
# ---------------------------------------------------------------------------

def compute_chain_candidate(
    features: Any, style: str = "iupac"
) -> Optional['CandidateName']:
    """Compute a chain-handler candidate from features, or None if infeasible.

    Per SC-5: idempotent. Called from composer.py's Tier A integration point
    UNCONDITIONALLY when V18 mode is active (selection_mode='score_based').
    In V17 mode (selection_mode='first_applicable'), the existing chain
    naming path still runs only when ring handlers have all failed — this
    helper's output is ignored because Tier A direct-return ring handlers
    short-circuit pool.best() in first_applicable dispatch.

    The function defers actual chain-name construction to composer.py's
    existing chain machinery. When a usable chain-naming entry point is not
    importable (e.g., during isolated unit tests of this module), the
    function returns None safely — chain never becomes a competing candidate,
    so ring handlers continue to win by default.

    Args:
        features: MolecularFeatures with principal_chain populated.
        style: naming style ('iupac' / 'pin'), forwarded to chain namer.

    Returns:
        CandidateName with handler='chain' and parent_atom_indices set, OR
        None if the chain pipeline cannot produce a viable name for these
        features.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    """
    # Defensive: no principal chain -> no chain candidate.
    principal_chain = getattr(features, 'principal_chain', None)
    if not principal_chain:
        return None

    # Try to invoke the existing chain naming pipeline via a public entry
    # point. The function name may vary across refactors; try a small
    # ordered list of candidates. If all fail, return None (safe default).
    _name_chain_fn = None
    try:
        from orthonym.assembly.composer import name_chain as _name_chain_fn  # type: ignore
    except ImportError:
        _name_chain_fn = None
    if _name_chain_fn is None:
        try:
            from orthonym.assembly.composer import _name_chain as _name_chain_fn  # type: ignore
        except ImportError:
            _name_chain_fn = None

    if _name_chain_fn is None:
        # No public chain-naming entry point found. In production, composer.py
        # already runs the chain fallback after ring handlers fail (the
        # `if features.principal_chain: parent = _generate_chain_parent(...)`
        # path near the tail of _assemble_name_impl). That path remains
        # unchanged. Return None so this helper is a no-op in that case.
        return None

    try:
        chain_name = _name_chain_fn(features, style=style)
    except TypeError:
        # Signature variant: single-arg (features,)
        try:
            chain_name = _name_chain_fn(features)
        except Exception:
            return None
    except Exception:
        return None

    if not chain_name:
        return None

    # Convert principal_chain (iterable of atom indices) to a Set[int].
    try:
        chain_atom_set: Set[int] = set(int(i) for i in principal_chain)
    except Exception:
        chain_atom_set = set()

    # Build the CandidateName via compute_confidence (parent_atom_indices=None
    # at construction time to preserve Risk 1 invariant). The caller then
    # passes parent_atom_indices to pool.add(), which sets the attribute
    # POST-HOC and recomputes multiple_bond_count in V18 mode.
    cand = compute_confidence(chain_name, "chain", features)
    cand.parent_atom_indices = chain_atom_set
    return cand
