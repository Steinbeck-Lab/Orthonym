"""CandidatePool: the candidate names of one assemble_name call, the gates that void a candidate, and the choice of the winner.

WHAT IT DOES
------------
A handler that names the molecule calls ``pool.add(name, handler_id, features,...)``. ``add`` scores the name with
``coverage_scoring.compute_confidence``, applies the two gates that void a candidate, and stores the survivor:

- ``RATIO_REJECT_FLOOR`` (every handler): a candidate whose ``ratio`` factor is below the floor is voided, unless the
  producer measured a complete atom partition (``coverage_measured=True``).
- the Tier-B gate (the ``ring_b`` handlers of ``HANDLER_POLICIES``): a candidate whose confidence is below
  ``CONFIDENCE_GATE_THRESHOLD`` is voided.

A voided candidate returns ``None`` and the caller falls through to the next handler.

``best`` returns the candidate of the first handler whose policy is ``direct_return``; when none fired it returns the
first candidate added. The inner handlers run in dispatch order (``assembly/inner_dispatch.py``) and the first one that
succeeds returns its own result, so the winner of a pool is the first name that a handler produced. The three Tier-A ring
handlers (``complex_ring``, ``heterocycle``, ``benzene``) are the exception: ``handlers/tier_a_ring.py`` adds up to three
candidates and ranks them with ``coverage_scoring.select_best_candidate`` (highest confidence, then handler priority,
then alphanumerical order of the name).

The pool does not rank parent skeletons by the criteria. That ranking is ``rules/p44_scorer.py``
(``select_parent_unified``), and the ranking of finished names is ``assembly/offer_pool.py`` (``rank_offers``,
``select_offer``, ``select_rt_passing``), which ``namer.py`` consumes.

FIELDS SET AFTER SCORING
------------------------
``parent_atom_indices``, ``ring_info``, ``tree``, ``node_scores`` and ``tree_rewritten`` are set on the candidate after
``compute_confidence`` has returned and never enter it: the confidence is a function of the name, the handler and the
features alone, so passing an atom set or a tree cannot change it. ``factors['parent_correctness']`` is recorded after
scoring in the same way; ``ParentCorrectnessScorer`` is bound once at module load (not imported per ``add`` call).

POOL STACK
----------
``assemble_name`` is recursive (nested substituent and fragment namings, N-oxide, decomposition). Each call pushes a
fresh pool with ``push_pool`` and pops it in a ``finally`` clause with ``pop_pool``; ``get_current_pool`` returns
the top. A recursive call therefore never adds to its caller's pool. The stack is thread-local, so concurrent runners
keep independent stacks.
"""

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:  # pragma: no cover - typing-only import (avoids runtime cycle)
    from .name_tree import NameTreeNode

from .coverage_scoring import (
    CONFIDENCE_GATE_THRESHOLD,
    HANDLER_PRIORITY,
    CandidateName,
    compute_confidence,
)

# REMEDIATION: module-level lazy binding for ParentCorrectnessScorer.
# Originally the import lived inside pool.add (try/except ImportError per
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

# a phase SCORE-03: module-level PerNodeScorer binding (mirrors the
# ParentCorrectnessScorer guard above — pay the import once at load, tolerate
# the scorer module being absent in pre-Plan-02 isolated tests). The add
# attach below no-ops when this is None.
try:
    from .per_substring_scoring import PerNodeScorer
except ImportError:
    PerNodeScorer = None  # type: ignore[assignment, misc]

# single source of truth for the coarse/structured discriminator. name_tree
# is a leaf module (stdlib-only top-level imports), so this direct runtime import
# introduces no cycle — never re-derive the predicate (it would drift from the
# contract test).
# candidate ledger. ``metrics.candidate_ledger`` is a leaf module
# (threading + typing only), so this is cycle-free for the same reason the
# ``name_tree`` import above is. The ledger is OFF unless a consumer calls
# ``enable``, so the cost on the production path is one boolean test per
# ``add`` return; the byte-identical contract is asserted in
# tests/unit/metrics/test_candidate_ledger.py.
from ..metrics.candidate_ledger import Stage as _LedgerStage
from ..metrics.candidate_ledger import is_enabled as _ledger_on
from ..metrics.candidate_ledger import record_candidate as _ledger_record
from .name_tree import is_coarse_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RATIO_REJECT_FLOOR — a phase -a.2 sanity gate
# ---------------------------------------------------------------------------
# Minimum `ratio` factor required for a candidate to be added to the pool.
# Catches obvious-garbage candidates (e.g., a 1-character name on a 30-HA
# molecule → ratio = 1/30/1.5 ≈ 0.022 < 0.10 → rejected).
#
# This is DISTINCT from FACTOR_WEIGHTS['ratio'] (demoted to 0.0 in Plan 01
# Task 1) and distinct from _MIN_RATIO_ACCEPT at composer.py:1520 (preserved
# per). See 145.2-internal notes §Plan 01 for the three-ratio disambiguation.
#
# Floor value 0.10 chosen so the reject condition is rare in practice (no
# real Tier A candidate on the 7,500-row baseline has ratio < 0.10) but
# strong enough to catch pathological garbage.
RATIO_REJECT_FLOOR: float = 0.10


# ---------------------------------------------------------------------------
# HandlerPolicy dataclass — declarative per-handler pool participation rules
# ---------------------------------------------------------------------------

@dataclass
class HandlerPolicy:
    """Declarative policy for one handler's pool participation.

    Fields:
        handler_id: Handler name (matches HANDLER_PRIORITY keys).
        tier: 'ring_a' | 'ring_b' | 'chain' | 'direct_return'.
        priority: Tiebreak priority (pulled from HANDLER_PRIORITY).
        direct_return: True for Tier B and direct-return handlers
                       (best returns the first such candidate); False for
                       the Tier A ring handlers (ranked by
                       select_best_candidate) and chain.

    ``add`` applies the module constant CONFIDENCE_GATE_THRESHOLD to the
    ``ring_b`` handlers; a policy carries no threshold of its own.
    """
    handler_id: str
    tier: str
    priority: int
    direct_return: bool = False


# ---------------------------------------------------------------------------
# HANDLER_POLICIES — per-handler pool metadata
#
# ``tier`` chooses the gate ``add`` applies ('ring_b': the confidence gate);
# ``direct_return`` makes ``best`` return that handler's candidate. Every
# priority is HANDLER_PRIORITY[handler_id] (assembly/coverage_scoring.py), the
# single source of the handler priorities, except 'chain' (see its entry).
#
# The salt / ion / radical / simple_molecule handlers have no entry: they route
# to assemble_ion_name / _name_simple_molecule directly functional
# class, radicals, terminal fallback) and never use the pool.
# ---------------------------------------------------------------------------

HANDLER_POLICIES: Dict[str, HandlerPolicy] = {
    # --- Tier B: the 'ring_b' handlers (CONFIDENCE_GATE_THRESHOLD=0.40) ---
    # add returns None when the confidence is below the threshold; the caller
    # checks `if cand is not None: return...` and otherwise falls through to
    # the next handler.
    'oxime':           HandlerPolicy('oxime',           'ring_b', priority=HANDLER_PRIORITY['oxime'],           direct_return=True),
    'hydrazone':       HandlerPolicy('hydrazone',       'ring_b', priority=HANDLER_PRIORITY['hydrazone'],       direct_return=True),
    'isocyanate':      HandlerPolicy('isocyanate',      'ring_b', priority=HANDLER_PRIORITY['isocyanate'],      direct_return=True),
    'isothiocyanate':  HandlerPolicy('isothiocyanate',  'ring_b', priority=HANDLER_PRIORITY['isothiocyanate'],  direct_return=True),
    'carbamic_acid':   HandlerPolicy('carbamic_acid',   'ring_b', priority=HANDLER_PRIORITY['carbamic_acid'],   direct_return=True),
    'carbamate':       HandlerPolicy('carbamate',       'ring_b', priority=HANDLER_PRIORITY['carbamate'],       direct_return=True),
    'urea':            HandlerPolicy('urea',            'ring_b', priority=HANDLER_PRIORITY['urea'],            direct_return=True),
    'guanidine':       HandlerPolicy('guanidine',       'ring_b', priority=HANDLER_PRIORITY['guanidine'],       direct_return=True),
    'cyanamide':       HandlerPolicy('cyanamide',       'ring_b', priority=HANDLER_PRIORITY['cyanamide'],       direct_return=True),
    'sulfoxide':       HandlerPolicy('sulfoxide',       'ring_b', priority=HANDLER_PRIORITY['sulfoxide'],       direct_return=True),
    'sulfone':         HandlerPolicy('sulfone',         'ring_b', priority=HANDLER_PRIORITY['sulfone'],         direct_return=True),
    # -6I: Se/Te oxide handlers (mirror sulfoxide/sulfone).
    'selenoxide':      HandlerPolicy('selenoxide',      'ring_b', priority=HANDLER_PRIORITY['selenoxide'],      direct_return=True),
    'selenone':        HandlerPolicy('selenone',        'ring_b', priority=HANDLER_PRIORITY['selenone'],        direct_return=True),
    'telluroxide':     HandlerPolicy('telluroxide',     'ring_b', priority=HANDLER_PRIORITY['telluroxide'],     direct_return=True),
    'tellurone':       HandlerPolicy('tellurone',       'ring_b', priority=HANDLER_PRIORITY['tellurone'],       direct_return=True),
    'thioether':       HandlerPolicy('thioether',       'ring_b', priority=HANDLER_PRIORITY['thioether'],       direct_return=True),
    'boronic_acid':    HandlerPolicy('boronic_acid',    'ring_b', priority=HANDLER_PRIORITY['boronic_acid'],    direct_return=True),
    'partial_sat':     HandlerPolicy('partial_sat',     'ring_b', priority=HANDLER_PRIORITY['partial_sat'],     direct_return=True),

    # --- Direct-return: no confidence gate ---
    # These return the first non-None name produced; no candidate competition.
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

    # --- Tier A: 3 ring handlers; tier_a_ring.py ranks their candidates with
    # select_best_candidate (confidence, then HANDLER_PRIORITY, then name) ---
    'complex_ring':    HandlerPolicy('complex_ring',    'ring_a', priority=HANDLER_PRIORITY['complex_ring'], direct_return=False),
    'heterocycle':     HandlerPolicy('heterocycle',     'ring_a', priority=HANDLER_PRIORITY['heterocycle'],  direct_return=False),
    'benzene':         HandlerPolicy('benzene',         'ring_a', priority=HANDLER_PRIORITY['benzene'],      direct_return=False),

    # --- Chain: priority 4 here; select_best_candidate ranks by
    # HANDLER_PRIORITY['chain'] (1) ---
    'chain':           HandlerPolicy('chain',           'chain',  priority=4, direct_return=False),
}


def _label_chain_parent_prefix_seniority(name: str, features: Any) -> None:
    """Label a chain-parent candidate below the PIN when makes another chain
    of the same rank senior.

    Every producer that names a cyclic molecule on a chain parent (the chain
    catch-all, the amide, ester and polyfunctional producers,...) builds the name
    on ``features.principal_chain`` and adds it here with those features, so the
    check sits here once. ``perception.chains.chain_parent_prefix_seniority``
    counts, locates and orders a ring bonded to a candidate chain as a prefix
     to, the Blue Book,:21698,:21791), which the chain
    selector does not do. ``'senior_alternative'``: the name is recorded as a
    label-only non-PIN part (``systematic_verified``); ``'undecided'``: as an
    uncertified PIN-form name (``pin_unverified``). Fail closed and label only:
    the name stays, so the tiers below the PIN keep it.
    """
    if not name or (getattr(features, 'chain_is_parent', False) != getattr(features, 'is_cyclic', False)):
        return                  # a ring parent; an acyclic chain is read for the tie of an amine's nitrogen only
    from ..perception.chains import chain_parent_prefix_seniority
    from .spelling_offer import note_amine_chain_parent
    note_amine_chain_parent(features)       # the best-effort tier may prefer a ring parent's cleaner name
    status = chain_parent_prefix_seniority(features)
    # the record is read as a substring of the shipped name: an amine's stem survives its naming as the
    # cation, 'methanamine' -> 'methanaminium'), the whole name does not
    fragment = name[:-1] if name.endswith('amine') else name
    if status == 'senior_alternative':
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(fragment)
    elif status == 'undecided':
        from ..metrics.provenance import record_uncertified_pin_name
        record_uncertified_pin_name(fragment)


# ---------------------------------------------------------------------------
# CandidatePool class — collects scored candidates from one assemble_name call
# ---------------------------------------------------------------------------

class CandidatePool:
    """Holds scored candidate names from one assemble_name invocation.

    ``best`` returns the candidate of the first ``direct_return`` handler, else
    the first candidate added (module docstring).

    ``parent_correctness`` is recorded in ``add`` through the module-level
    ``ParentCorrectnessScorer`` binding; it is computed after the confidence, so
    it does not enter ``cand.confidence``.

    Tier B gate semantics: for a ``ring_b`` handler, ``add`` returns None
    (gate-rejected) when ``cand.confidence`` is below ``CONFIDENCE_GATE_THRESHOLD``
    (0.40), and the caller falls through to the next handler.

    Direct-return handlers: ``add`` also sets ``self._direct_return_winner``
    (the first direct-return candidate), which ``best`` returns.
    """

    def __init__(self) -> None:
        self._candidates: List[CandidateName] = []
        self._direct_return_winner: Optional[CandidateName] = None
        # __init__ takes no features: push_pool / clear_pool bind the controller flag
        # and oracle from the per-call features right after construction. A pool created
        # without features (get_current_pool auto-push, isolated tests) takes the safe
        # defaults below; add binds it off its own features on the first call.
        self._enable_triviality_controller: bool = False
        self._triv_oracle = None
        self._triv_flag_initialized: bool = False

    def _bind_triviality_features(self, features: Any) -> None:
        """Bind the a phase controller flag/oracle off the authoritative per-call ``features``.

         (code review 2026-05-30): the flag is a per-``Orthonym``-call constant (namer
        stashes it on ``features`` in ``_classify`` before ``assemble_name``), so it is bound at
        pool CONSTRUCTION via ``push_pool(features)`` / ``clear_pool(features)`` rather than
        lifted off whichever candidate happens to call ``add`` first. Idempotent: marks the
        pool initialized so the ``add`` fallback below does not re-lift.
        """
        self._enable_triviality_controller = getattr(
            features, '_enable_triviality_controller', False)
        self._triv_oracle = getattr(features, '_triv_oracle', None)
        self._triv_flag_initialized = True

    def add(
        self,
        name: str,
        handler_id: str,
        features: Any,
        parent_atom_indices: Optional[Set[int]] = None,
        ring_info: Optional[Dict[str, Any]] = None,
        tree: Optional["NameTreeNode"] = None,
        coverage_measured: bool = False,
    ) -> Optional[CandidateName]:
        """Score and add a candidate. Returns the candidate, or None if
        gate-rejected (Tier B handlers only).

        BYTE-IDENTICAL CONTRACT (RESEARCH Risk 1):
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
            coverage_measured: True when the producer MEASURED a complete atom
                partition for ``name`` (every heavy atom bound to one of its
                fragments; ``_w2_atom_coverage_verdict``). The RATIO_REJECT_FLOOR
                is a character-count stand-in for coverage, so it is skipped
                then: numerical-term chain names are short by design
                ('octacontane', ratio 0.092). Does not affect the confidence.

        Returns:
            The added CandidateName on success.
            None if Tier B gate rejected (cand.confidence <
            CONFIDENCE_GATE_THRESHOLD).
        """
        # (code review 2026-05-30): the controller flag/oracle are bound at pool
        # CONSTRUCTION from the authoritative per-call `features` (push_pool/clear_pool ->
        # _bind_triviality_features). This lazy lift is now only the FALLBACK for pools created
        # WITHOUT features — get_current_pool's auto-push and isolated unit tests — binding off
        # the first add's features idempotently. In production the pool is already initialized
        # at construction, so this no-ops (it never depends on add ordering).
        if not self._triv_flag_initialized:
            self._bind_triviality_features(features)
        policy = HANDLER_POLICIES.get(handler_id)
        # Compute confidence WITHOUT parent_atom_indices (Risk 1 mitigation)
        cand = compute_confidence(name, handler_id, features)
        # RATIO_REJECT_FLOOR sanity gate (a phase -a.2). Rejects
        # obvious-garbage candidates regardless of handler tier. Missing
        # 'ratio' factor treated as "not garbage" (defensive default).
        # A MEASURED complete atom partition (coverage_measured) supersedes this
        # length proxy (TRIAGE j3-long-alkanes, g1 C1); the OPSIN round-trip gate
        # still verifies the emitted name.
        ratio_val = cand.factors.get('ratio')
        if (ratio_val is not None and ratio_val < RATIO_REJECT_FLOOR
                and not coverage_measured):
            logger.debug(
                "candidate rejected by RATIO_REJECT_FLOOR: "
                "handler=%s name=%r ratio=%.3f < floor=%.3f",
                handler_id, name, ratio_val, RATIO_REJECT_FLOOR,
            )
            # ledger: a candidate WAS built and this floor threw it away.
            # Recorded after the decision so it cannot influence it.
            if _ledger_on():
                _ledger_record(handler_id, _LedgerStage.GATE_REJECTED, name,
                               detail=f"ratio_reject_floor:{ratio_val:.3f}")
            return None
        # Tier B gate: a candidate of a 'ring_b' handler (policy.tier) whose
        # confidence is below the module-level CONFIDENCE_GATE_THRESHOLD (0.40)
        # is voided.
        if policy is not None and policy.tier == 'ring_b':
            if cand.confidence < CONFIDENCE_GATE_THRESHOLD:
                # Gate rejected — return None so caller falls through
                if _ledger_on():
                    _ledger_record(handler_id, _LedgerStage.GATE_REJECTED, name,
                                   detail=f"tier_b_confidence:{cand.confidence:.3f}")
                return None
        # Set parent_atom_indices POST-HOC (Risk 1)
        cand.parent_atom_indices = parent_atom_indices
        # a phase: attach ring_info POST-HOC (Risk 1 pattern).
        # NEVER passed into compute_confidence — that would break the
        # byte-identical guarantee per a phase.
        cand.ring_info = ring_info
        # a phase SCORE-01: attach the structured Name-Tree POST-HOC (Risk 1
        # pattern). NEVER passed into compute_confidence — byte-identical
        # preserved (a phase). best surfaces the winner's tree.
        cand.tree = tree
        # a phase SCORE-03: attach per-node scores POST-HOC (Risk 1 pattern).
        # NEVER into compute_confidence; score_tree returns {} in production
        # (no reference name set) and on coarse trees, so cand.confidence stays
        # byte-identical (SCORE-05); node_scores never touches confidence.
        if (PerNodeScorer is not None and tree is not None
                and not is_coarse_node(tree)):
            cand.node_scores = PerNodeScorer.score_tree(tree, features.mol)
        # a phase /02/03: attach the rewritten tree POST-HOC (Risk-1 pattern; SACRED per
        # a phase + 166). NEVER passed into compute_confidence; byte-identical preserved
        # at Stage A per internal notes. Default (flag OFF): tree_rewritten == tree (no-op), the
        # controller block is skipped, production reads cand.tree unchanged. With the flag ON the
        # consumers read tree_rewritten.
        cand.tree_rewritten = tree
        cand.node_scores_rewritten = getattr(cand, 'node_scores', None)
        if (self._enable_triviality_controller and tree is not None
                and not is_coarse_node(tree)):
            try:
                from .retained_substitution import apply_triviality_controller  # Pattern S3 lazy
                principal_group = getattr(features, 'principal_group', None)
                cand.tree_rewritten = apply_triviality_controller(
                    tree, features.mol, principal_group,
                    opsin_oracle=self._triv_oracle, enabled=True,  # WARNING #9 oracle propagated
                )
                if (PerNodeScorer is not None
                        and not is_coarse_node(cand.tree_rewritten)):
                    cand.node_scores_rewritten = PerNodeScorer.score_tree(
                        cand.tree_rewritten, features.mol)
            except Exception as exc:
                logger.warning(
                    "Phase 168 controller error on cand %r: %s; falling back to original tree",
                    getattr(cand, 'name', '?'), exc,
                )
                cand.tree_rewritten = tree
        # a phase fallback: if ring_info wasn't passed explicitly, read
        # the transient attribute set by namer.py:_classify (allows existing
        # composer.py call sites to flow ring_info through without a
        # signature change at every site).
        if cand.ring_info is None:
            cand.ring_info = getattr(features, '_ring_info', None)
        # Inline parent_correctness scoring .
        # REMEDIATION: reference the MODULE-LEVEL ParentCorrectnessScorer
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
        # to counted with ring prefixes: a chain-parent name
        # whose chain is not the senior one is labelled below the PIN (label only;
        # the name, its read-back and every tier's name are unchanged).
        _label_chain_parent_prefix_seniority(name, features)
        self._candidates.append(cand)
        if policy is not None and policy.direct_return:
            if self._direct_return_winner is None:
                self._direct_return_winner = cand
        if _ledger_on():
            _ledger_record(handler_id, _LedgerStage.PRODUCED, name,
                           detail=f"confidence:{cand.confidence:.3f}")
        return cand

    def best(self) -> Optional[CandidateName]:
        """Return the winning candidate, or None when the pool is empty.

        The candidate of the first ``direct_return`` handler wins (the same result as
        the sequential "if X applies: return X" dispatch of ``assemble_name``);
        when no such handler added a candidate, the first candidate added wins.
        For example, the Tier-A ``complex_ring`` handler (``direct_return`` False)
        may add '1-methoxypyrene' and the ``polycyclic`` handler (``direct_return``
        True) then add '2-methoxypyrene' for the same molecule: ``best``
        returns the ``polycyclic`` candidate, not ``_candidates[0]``.
        """
        if not self._candidates:
            return None
        if self._direct_return_winner is not None:
            return self._direct_return_winner
        return self._candidates[0]

    def all_candidates(self) -> List[CandidateName]:
        """Return all collected candidates (for logging/diagnostics)."""
        return list(self._candidates)


# ---------------------------------------------------------------------------
# Thread-local pool stack: one pool per assemble_name call.
#
# assemble_name recurses (N-oxide handler, composition and decomposition
# handlers, fragment naming, substituent enumeration, decomposition/engine). A
# single-slot pool would let a recursive call's candidate become pool[0] of the
# outer call, and best would then return the inner name. So the prologue of
# assemble_name calls push_pool, the body reads and writes the top of the
# stack through get_current_pool, and the epilogue (try/finally) calls
# pop_pool. Each thread has its own stack.
# ---------------------------------------------------------------------------

_pool_store = threading.local()


def _ensure_stack() -> List[CandidatePool]:
    """Initialize the per-thread pool stack on first access.

    Each thread has its own stack (via threading.local), so concurrent
    benchmark runners (scripts/benchmark_multi_corpus.py uses ThreadPoolExecutor)
    have independent pool stacks with no cross-thread leakage.
    """
    if not hasattr(_pool_store, 'stack'):
        _pool_store.stack = []
    return _pool_store.stack


def push_pool(features: Any = None) -> CandidatePool:
    """Push a fresh pool onto the per-thread stack and return it.

    Called by assemble_name prologue (composer.py). MUST be paired with
    pop_pool in a finally clause so the pool is popped on every exit path
    (normal return, exception, early return statements).

     (code review 2026-05-30): when ``features`` is provided, the a phase
    controller flag/oracle are bound NOW (construction) off that authoritative
    per-call features object, instead of being lifted lazily off the first
    add call. ``features=None`` (get_current_pool auto-push, tests) leaves the
    pool's flag at its safe default and the add fallback binds it later.

    Returns the new pool that is now the active cascade scope for the
    currently-executing assemble_name call.
    """
    stack = _ensure_stack()
    new_pool = CandidatePool()
    if features is not None:
        new_pool._bind_triviality_features(features)
    stack.append(new_pool)
    return new_pool


def pop_pool() -> None:
    """Pop the top pool from the per-thread stack.

    Called by assemble_name epilogue (composer.py, in finally clause).
    Safe to call when the stack is empty (no-op) — defensive against any
    code path that pops without a matching push.
    """
    stack = _ensure_stack()
    if stack:
        stack.pop()


def get_current_pool() -> CandidatePool:
    """Return the top of the per-thread pool stack — the active cascade
    for the currently-executing assemble_name call.

    Auto-pushes a fresh pool if the stack is empty. This covers two cases:
    (1) a handler called get_current_pool outside an assemble_name scope
    (e.g., from a unit test that invokes pool.add directly), and
    (2) backward compatibility with any legacy caller that expected the
    pre-fix lazy-init behavior.
    """
    stack = _ensure_stack()
    if not stack:
        return push_pool()
    return stack[-1]


def clear_pool(features: Any = None) -> None:
    """BACKWARD-COMPAT: replace the top of the stack with a fresh pool.

    Pre-fix code (Plan 03) called clear_pool in assemble_name's prologue
    to reset state. Post-fix, push_pool is the right primitive (the wrapping
    try/finally handles pop). This alias preserves the call-site signature
    so any handler still calling clear_pool directly does not break.

    Behavior: if the stack is empty, push a fresh pool (same as pre-fix
    lazy init); if non-empty, replace the top so the current cascade restarts
    cleanly without affecting outer cascades on the stack.

     (code review 2026-05-30): binds the a phase controller flag/oracle
    off ``features`` at construction (see push_pool) when provided.
    """
    stack = _ensure_stack()
    if stack:
        new_pool = CandidatePool()
        if features is not None:
            new_pool._bind_triviality_features(features)
        stack[-1] = new_pool
    else:
        push_pool(features)
