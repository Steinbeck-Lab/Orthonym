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
        """Select winning candidate per selection_mode."""
        if not self._candidates:
            return None
        if self.selection_mode == 'first_applicable':
            # 145.1: first added wins. The cascade order IS the handler
            # dispatch order in composer.assemble_name(). When the first
            # added candidate is a direct_return handler, that's the
            # winner — pool.best() returns it.
            return self._candidates[0]
        # Phase 146 path: score-based selection over the full pool.
        return select_best_candidate(self._candidates)

    def all_candidates(self) -> List[CandidateName]:
        """Return all collected candidates (for logging/diagnostics)."""
        return list(self._candidates)


# ---------------------------------------------------------------------------
# Thread-local pool store (mirrors coverage_scoring._confidence_store)
# ---------------------------------------------------------------------------

_pool_store = threading.local()


def get_current_pool() -> CandidatePool:
    """Return thread-local pool, creating one on first access.

    Lazy-init pattern mirrors getattr(_confidence_store, 'last_candidate', None)
    at coverage_scoring.py:418. Pool defaults to selection_mode='first_applicable'
    (Phase 145.1 behavior). Phase 146 will instantiate with 'score_based'.
    """
    pool = getattr(_pool_store, 'pool', None)
    if pool is None:
        pool = CandidatePool(selection_mode='first_applicable')
        _pool_store.pool = pool
    return pool


def clear_pool() -> None:
    """Reset thread-local pool. Called at assemble_name() prologue
    (composer.py:674 next to existing clear_confidence() call)."""
    _pool_store.pool = CandidatePool(selection_mode='first_applicable')
