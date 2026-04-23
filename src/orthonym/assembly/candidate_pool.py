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
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .coverage_scoring import (
    CandidateName,
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
# HandlerPolicy dataclass — declarative per-handler pool participation rules
# ---------------------------------------------------------------------------

@dataclass
class HandlerPolicy:
    """Declarative policy for one handler's pool participation.

    Fields:
        handler_id:        Handler name (matches HANDLER_PRIORITY keys).
        tier:              'ring_a' | 'ring_b' | 'chain' | 'direct_return'.
        priority:          Tiebreak priority (pulled from HANDLER_PRIORITY).
        min_ratio_accept:  DELETE IN PHASE 146 — composer.py:1323 ratio gate
                           for Tier A pool exit (0.30 / 1.5 = 0.20 effective).
        cascade_ratio_min: DELETE IN PHASE 146 — composer.py:1199 cascade
                           threshold for complex_ring early-accept (0.40).
        gate_threshold:    DELETE IN PHASE 146 — Tier B _confidence_gate
                           threshold (0.40, mirrors CONFIDENCE_GATE_THRESHOLD).
        direct_return:     True for Tier B and direct-return handlers
                           (first-applicable wins); False for Tier A pool
                           handlers (compete via select_best_candidate)
                           and chain (fallback in 145.1).
    """
    handler_id: str
    tier: str
    priority: int
    min_ratio_accept:  Optional[float] = None  # DELETE IN PHASE 146
    cascade_ratio_min: Optional[float] = None  # DELETE IN PHASE 146
    gate_threshold:    Optional[float] = None  # DELETE IN PHASE 146 (Tier B gate)
    direct_return: bool = False


# ---------------------------------------------------------------------------
# HANDLER_POLICIES — single source of truth for per-handler pool metadata
#
# Populated VERBATIM from composer.py current constants (verified against
# main HEAD per 145.1-RESEARCH.md §2.2). Phase 146 sets gate fields to None
# and flips chain.priority to equal ring_a handlers.
#
# PRIORITY SOURCE (ISS-002 remediation): every entry pulls priority from
# HANDLER_PRIORITY (the single source of truth). Plan 02 Task 2 extends
# HANDLER_PRIORITY with the missing 'n_oxide', 'amine', 'simple_molecule'
# entries (with priorities 5, 5, 1 respectively) so this dict can pull from
# HANDLER_PRIORITY[*] without hardcoded literals.
#
# SCOPE NOTE: salt / ion / radical / simple_molecule handlers are EXCLUDED
# from this dict per 145.1-RESEARCH.md §9.2 item 3 + ISS-001 enumeration:
# they route to assemble_ion_name() / _name_simple_molecule() directly at
# composer.py:693-710 and composer.py:1389 with completely different
# naming semantics (P-73 functional class, P-68 radicals, terminal
# fallback). They never participate in the pool.
# ---------------------------------------------------------------------------

HANDLER_POLICIES: Dict[str, HandlerPolicy] = {
    # --- Tier B: 13 handlers using _confidence_gate (gate_threshold=0.40) ---
    # Each runs gate check before return; on gate-fail, falls through to
    # next handler. Pool reproduces by returning None from add() on
    # gate-fail; caller checks `if cand is not None: return ...`.
    'oxime':           HandlerPolicy('oxime',           'ring_b', priority=HANDLER_PRIORITY['oxime'],           gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'hydrazone':       HandlerPolicy('hydrazone',       'ring_b', priority=HANDLER_PRIORITY['hydrazone'],       gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'isocyanate':      HandlerPolicy('isocyanate',      'ring_b', priority=HANDLER_PRIORITY['isocyanate'],      gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'isothiocyanate':  HandlerPolicy('isothiocyanate',  'ring_b', priority=HANDLER_PRIORITY['isothiocyanate'],  gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'carbamic_acid':   HandlerPolicy('carbamic_acid',   'ring_b', priority=HANDLER_PRIORITY['carbamic_acid'],   gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'carbamate':       HandlerPolicy('carbamate',       'ring_b', priority=HANDLER_PRIORITY['carbamate'],       gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'urea':            HandlerPolicy('urea',            'ring_b', priority=HANDLER_PRIORITY['urea'],            gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'guanidine':       HandlerPolicy('guanidine',       'ring_b', priority=HANDLER_PRIORITY['guanidine'],       gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'sulfoxide':       HandlerPolicy('sulfoxide',       'ring_b', priority=HANDLER_PRIORITY['sulfoxide'],       gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'sulfone':         HandlerPolicy('sulfone',         'ring_b', priority=HANDLER_PRIORITY['sulfone'],         gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'thioether':       HandlerPolicy('thioether',       'ring_b', priority=HANDLER_PRIORITY['thioether'],       gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'boronic_acid':    HandlerPolicy('boronic_acid',    'ring_b', priority=HANDLER_PRIORITY['boronic_acid'],    gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)
    'partial_sat':     HandlerPolicy('partial_sat',     'ring_b', priority=HANDLER_PRIORITY['partial_sat'],     gate_threshold=0.40, direct_return=True),  # DELETE IN PHASE 146 (gate)

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
    # complex_ring has BOTH cascade_ratio_min (early-accept short-circuit
    # of subsequent ring handlers) AND min_ratio_accept (pool exit gate).
    # heterocycle and benzene only have min_ratio_accept (set on the
    # pool-exit-gate check, not on individual handler add).
    'complex_ring':    HandlerPolicy('complex_ring',    'ring_a', priority=HANDLER_PRIORITY['complex_ring'],
                                      cascade_ratio_min=0.40,  # DELETE IN PHASE 146 (composer.py:1199)
                                      min_ratio_accept=0.30,   # DELETE IN PHASE 146 (composer.py:1323)
                                      direct_return=False),
    'heterocycle':     HandlerPolicy('heterocycle',     'ring_a', priority=HANDLER_PRIORITY['heterocycle'],
                                      min_ratio_accept=0.30,   # DELETE IN PHASE 146 (composer.py:1323)
                                      direct_return=False),
    'benzene':         HandlerPolicy('benzene',         'ring_a', priority=HANDLER_PRIORITY['benzene'],
                                      min_ratio_accept=0.30,   # DELETE IN PHASE 146 (composer.py:1323)
                                      direct_return=False),

    # --- Chain: fallback in 145.1; raised to equal in 146 ---
    # priority=1 (HANDLER_PRIORITY['chain']) — wins only when pool is
    # otherwise empty or all higher-priority handlers gate-rejected.
    # PHASE 146: raise priority to 4 to enable chain-vs-ring competition
    # per IUPAC P-44.1 cascade (chain length is criterion (c)).
    'chain':           HandlerPolicy('chain',           'chain',  priority=HANDLER_PRIORITY['chain'],  # DELETE IN PHASE 146 (raise to ring_a equal)
                                      direct_return=False),
}


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
      policy.gate_threshold=0.40 → if cand.confidence < 0.40, pool.add()
      returns None (gate-rejected); caller falls through to next handler.

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
            None if Tier B gate rejected (cand.confidence < gate_threshold).
        """
        policy = HANDLER_POLICIES.get(handler_id)
        # Compute confidence WITHOUT parent_atom_indices (Risk 1 mitigation)
        cand = compute_confidence(name, handler_id, features)
        # Tier B gate enforcement (preserves _confidence_gate behavior)
        if policy is not None and policy.gate_threshold is not None:
            if cand.confidence < policy.gate_threshold:
                # Gate rejected — return None so caller falls through
                return None
        # Set parent_atom_indices POST-HOC (Risk 1)
        cand.parent_atom_indices = parent_atom_indices
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
        # Phase 146 path: score-based selection over the full pool.
        return select_best_candidate(self._candidates)

    def all_candidates(self) -> List[CandidateName]:
        """Return all collected candidates (for logging/diagnostics)."""
        return list(self._candidates)


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

    Returns the new pool that is now the active cascade scope for the
    currently-executing assemble_name() call.
    """
    stack = _ensure_stack()
    new_pool = CandidatePool(selection_mode='first_applicable')
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
        stack[-1] = CandidatePool(selection_mode='first_applicable')
    else:
        push_pool()
