"""CandidatePool — Phase 145.1 behavior-preserving extraction of composer.py
handler-cascade dispatch.

PURPOSE
-------
Phase 145.1 (this module): factors out the in-line handler cascade in
composer.assemble_name() into an explicit pool. Pool runs in
selection_mode='first_applicable' which reproduces the current sequential
'if applies: return' flow bit-for-bit. SHIP GATE: byte-identical generated
name strings on baseline_v17_all_corpora.csv (7,500 rows).

Phase 146 (downstream consumer) was DESIGNED to flip selection_mode to
'score_based', raise chain priority, set the three gate thresholds to None
to delete the IUPAC-non-conformant ratio gates, and recalibrate
FACTOR_WEIGHTS. NOTE (Phase 166 SCORE-04, audit §Stale-Comment Inventory):
that production flip never landed — production stays 'first_applicable'.
The RT-moving cutover is now the default-OFF 'score_based_per_substring'
mode, deferred to a downstream documented-delta phase.
All four 146 changes are 1-line edits to HANDLER_POLICIES + FACTOR_WEIGHTS.

DESIGN CONTRACT (, from 145.1-CONTEXT.md):
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:  # pragma: no cover - typing-only import (avoids runtime cycle)
    from .name_tree import NameTreeNode

from .coverage_scoring import (
    CONFIDENCE_GATE_THRESHOLD,
    HANDLER_PRIORITY,
    CandidateName,
    compute_confidence,
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

# Phase 166 SCORE-03: module-level PerNodeScorer binding (mirrors the
# ParentCorrectnessScorer guard above — pay the import once at load, tolerate
# the scorer module being absent in pre-Plan-02 isolated tests). The add()
# attach below no-ops when this is None.
try:
    from .per_substring_scoring import PerNodeScorer
except ImportError:
    PerNodeScorer = None  # type: ignore[assignment, misc]

# WR-4 single source of truth for the coarse/structured discriminator. name_tree
# is a leaf module (stdlib-only top-level imports), so this direct runtime import
# introduces no cycle — never re-derive the predicate (it would drift from the
# contract test).
# candidate ledger. ``metrics.candidate_ledger`` is a leaf module
# (threading + typing only), so this is cycle-free for the same reason the
# ``name_tree`` import above is. The ledger is OFF unless a consumer calls
# ``enable()``, so the cost on the production path is one boolean test per
# ``add()`` return; the byte-identical contract is asserted in
# tests/unit/metrics/test_candidate_ledger.py.
from ..metrics.candidate_ledger import Stage as _LedgerStage
from ..metrics.candidate_ledger import is_enabled as _ledger_on
from ..metrics.candidate_ledger import record_candidate as _ledger_record
from .name_tree import is_coarse_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RATIO_REJECT_FLOOR — Phase 145.2 -a.2 sanity gate
# ---------------------------------------------------------------------------
# Minimum `ratio` factor required for a candidate to be added to the pool.
# Catches obvious-garbage candidates (e.g., a 1-character name on a 30-HA
# molecule → ratio = 1/30/1.5 ≈ 0.022 < 0.10 → rejected).
#
# This is DISTINCT from FACTOR_WEIGHTS['ratio'] (demoted to 0.0 in Plan 01
# Task 1) and distinct from _MIN_RATIO_ACCEPT at composer.py:1520 (preserved
# per). See 145.2-CONTEXT.md §Plan 01 for the three-ratio disambiguation.
#
# Floor value 0.10 chosen so the reject condition is rare in practice (no
# real Tier A candidate on the 7,500-row baseline has ratio < 0.10) but
# strong enough to catch adversarial / pathological garbage during Phase 146
# competitive selection.
RATIO_REJECT_FLOOR: float = 0.10


# ---------------------------------------------------------------------------
# Phase 146: feature-flag-controlled default selection mode.
# ORTHONYM_SELECTION_MODE env var read at module-import time.
# Default 'first_applicable' (V17) during the post-148 soak per.
# Rollback one-liner:
#   ORTHONYM_USE_V18_WEIGHTS=false ORTHONYM_SELECTION_MODE=first_applicable pytest tests/
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
# ---------------------------------------------------------------------------
_DEFAULT_SELECTION_MODE: str = os.getenv(
    'ORTHONYM_SELECTION_MODE', 'first_applicable'
).strip().lower()
assert _DEFAULT_SELECTION_MODE in (
    'first_applicable', 'score_based', 'score_based_per_substring'
), (
    f"Invalid ORTHONYM_SELECTION_MODE={_DEFAULT_SELECTION_MODE!r}; must be "
    f"'first_applicable', 'score_based', or 'score_based_per_substring'"
)


# ---------------------------------------------------------------------------
# Phase 146: P-44.1.2 element seniority for parent-hydride selection.
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
    `_best_two_tier` (two-tier selector) and the inline V17 gate at
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
    'cyanamide':       HandlerPolicy('cyanamide',       'ring_b', priority=HANDLER_PRIORITY['cyanamide'],       direct_return=True),
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
    # (two-tier selector). V17 byte-identical preservation of
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
# Phase 146 Tier-1 cascade filters.
# Each filter has signature (List[CandidateName]) -> List[CandidateName].
# NEVER returns empty list — if all candidates tie, all are returned.
# ---------------------------------------------------------------------------

# ============================================================================
# UNREACHABLE IN PRODUCTION -- measured 2026-07-30 at d5f9ca96
#
# This function and the rest of the P-44 criteria cascade below (through
# _TIER1_FILTERS, _has_iupac_locants and _best_two_tier) CANNOT EXECUTE in the
# shipped configuration. Three independent measured reasons, any one sufficient:
#
#   1. _DEFAULT_SELECTION_MODE is 'first_applicable', so best() returns early and
#      never reaches its _best_two_tier call.
#   2. The pool never holds more than ONE candidate. Spied over all 309 dev500
#      failures: 517 best() calls, pool size == 1 in 517/517, _best_two_tier
#      invoked 0 times, 0 rows with >1 candidate. Seniority filters over a set of
#      one are a no-op by construction.
#   3. dispatch_inner never consults the pool -- composer.py:1121 returns the
#      winning handler's own NamingResult. Handlers RETURN rather than ADD (32 of
#      36 HANDLER_POLICIES are direct_return=True), so nothing is ever ranked.
#
# 'wins' (correct name available but not selected) = 1 of 309, and
# that one was a fragment name from an abandoned nested pool, not a rival
# whole-molecule candidate. Write-up: eval/LOG.md "2".
#
# DO NOT revive by flipping ORTHONYM_SELECTION_MODE=score_based -- measured, that
# changes 0 of 300 names, because of reason 3. DO NOT calibrate FACTOR_WEIGHTS
# against it (scripts/calibrate_phase_146.py tunes a selector that never runs).
# Kept rather than deleted because Plan-02 unit tests call these directly.
# ============================================================================
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
    """P-44.4.1.4+: lowest locants for principal groups, then multiple bonds, then substituents.

    Phase 147 implementation (replaces Phase 146 stub). For each candidate
    in the pool, extracts feature-specific locant lists (PG attachment
    positions, multiple-bond positions, substituent positions) using the
    same atom-walk patterns as ``parent_selection.py``:
    ``_compare_pg_locants`` / ``_compare_multiple_bond_locants`` /
    ``_compare_substituent_locants``. Each cascade step partitions the
    pool into winners (lowest locant set) and losers via
    ``compare_locant_sets`` (tuple-aware per Plan 01); winners advance,
    losers are dropped.

    Reuses (does NOT duplicate) the canonical comparison semantics from
    ``parent_selection.py`` — the comparators there are chain-vs-ring
    framed and cannot be called directly on a candidate pool, but their
    locant-extraction patterns at lines ~327-340 / 402-417 / 466-482 are
    reused inline. Per Phase 147 BL-3 /: a parallel comparison
    engine would violate "reuse > rebuild".

    Precondition: ``_has_iupac_locants(candidates)`` must return True
    before this function is invoked (enforced at ``_best_two_tier``
    step-6 dispatch). Defensive fall-through: if any candidate lacks
    ``parent_atom_indices``, ``_mol_ref``, or has incomplete ring_info,
    returns candidates unchanged (Tier-2 takes over).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4 - P-44.4.1.12
    Source: ``src/orthonym/rules/parent_selection.py:292-490`` (canonical
        P-44.1(f)/(g)/(i) implementations whose locant-extraction patterns
        are reused).
    Source: Phase 147 CONTEXT,, BL-3.
    """
    from rdkit import Chem

    from orthonym.rules.locants import compare_locant_sets
    from orthonym.rules.parent_selection import (
        _build_ring_pos,
        _pg_attachment_atoms,
    )

    if len(candidates) <= 1:
        return list(candidates)

    # Cascade extractor: returns (cand, pg_locants, mbond_locants, sub_locants)
    # for one candidate. Each list is the feature-locant list on that
    # candidate's parent. Returns None if the candidate lacks the data
    # required for cascade comparison (defensive bail-out).
    def _extract_locant_lists(cand):
        if cand.parent_atom_indices is None or not cand.ring_info:
            return None
        mol = getattr(cand, '_mol_ref', None)
        if mol is None:
            return None
        parent_set = set(cand.parent_atom_indices)
        if not parent_set:
            return None
        ring_pos = _build_ring_pos(parent_set, cand.ring_info)
        # Confirm ring_pos covers all parent atoms; if any are missing,
        # fall through (sorted fallback already handled inside
        # _build_ring_pos for partial coverage but the cascade requires
        # complete authoritative numbering).
        if any(idx not in ring_pos for idx in parent_set):
            return None

        # PG attachment locants (pattern: parent_selection.py:327-340).
        # IM-01: per-FG attachment indices via _pg_attachment_atoms; for
        # multi-atom PGs (disulfide), the per-instance locant is min over
        # the FG's attachment atoms (matches IUPAC P-31.1.4).
        pg_atoms_list = getattr(cand, '_principal_group_atoms', None) or []
        fg_name = getattr(cand, '_principal_group', None)
        pg_locants = []
        for pg_atoms in pg_atoms_list:
            if not pg_atoms:
                continue
            attachments = _pg_attachment_atoms(fg_name, pg_atoms)
            instance_locants = []
            for attachment in attachments:
                if attachment in parent_set:
                    instance_locants.append(ring_pos[attachment])
                    continue
                atom = mol.GetAtomWithIdx(attachment)
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in parent_set:
                        instance_locants.append(ring_pos[nbr.GetIdx()])
                        break
            if instance_locants:
                pg_locants.append(min(instance_locants))

        # Multiple-bond locants on parent (pattern: parent_selection.py:402-417).
        mbond_locants = []
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt != Chem.BondType.DOUBLE and bt != Chem.BondType.TRIPLE:
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in parent_set and b in parent_set:
                mbond_locants.append(min(ring_pos[a], ring_pos[b]))

        # Substituent locants on parent (pattern: parent_selection.py:466-482).
        sub_locants = []
        for idx in sorted(parent_set):
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if (nbr.GetIdx() not in parent_set
                        and nbr.GetSymbol() != 'H'):
                    sub_locants.append(ring_pos[idx])
                    break

        return (cand, sorted(pg_locants), sorted(mbond_locants),
                sorted(sub_locants))

    extracted = []
    for cand in candidates:
        row = _extract_locant_lists(cand)
        if row is None:
            # Defensive: incomplete data on any candidate -> bail out.
            return list(candidates)
        extracted.append(row)

    def _select_lowest(rows, locant_idx):
        """Partition rows by lowest locant-set at the given cascade step.

        rows: list of (cand, pg, mb, sub).
        locant_idx: 1=pg, 2=mb, 3=sub.
        """
        best = [rows[0]]
        best_locants = rows[0][locant_idx]
        for row in rows[1:]:
            cmp = compare_locant_sets(row[locant_idx], best_locants)
            if cmp < 0:
                best = [row]
                best_locants = row[locant_idx]
            elif cmp == 0:
                best.append(row)
            # cmp > 0: drop
        return best

    # Step 1: P-44.4.1.4 / P-44.1(f) PG locants.
    winners = _select_lowest(extracted, 1)
    if len(winners) == 1:
        return [winners[0][0]]
    # Step 2: P-44.1(g) multiple-bond locants.
    winners = _select_lowest(winners, 2)
    if len(winners) == 1:
        return [winners[0][0]]
    # Step 3: P-44.1(i) substituent locants.
    winners = _select_lowest(winners, 3)
    return [row[0] for row in winners]


def _has_iupac_locants(candidates: List['CandidateName']) -> bool:
    """ / safe probe: True iff ALL candidates have iupac_locants populated.

    Phase 147: ring_info is now a first-class CandidateName field (no longer
    getattr-defensive). Returns False when any candidate has ring_info=None
    OR ring_info['iupac_locants'] is None/empty (spiro/VB stubs per).

    Phase 146 mode: no candidate populates ring_info -> returns False ->
    Tier-1 step 6 is a no-op.
    Phase 147 mode: benzene/simple-hetero/PAH/fused-hetero candidates populate
    iupac_locants -> step 6 activates.
    """
    for cand in candidates:
        ri = cand.ring_info or {}
        if not ri.get('iupac_locants'):
            return False
    return True


# Module-level cascade order (P-44.1 lexicographic).
# Step 6 (locant criteria) is gated separately inside _best_two_tier
# via _has_iupac_locants per.
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

    SCORING TIMING: parent_correctness factor is computed inline in
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
        assert selection_mode in (
            'first_applicable', 'score_based', 'score_based_per_substring'), \
            f"unknown selection_mode: {selection_mode!r}"
        self.selection_mode = selection_mode
        self._candidates: List[CandidateName] = []
        self._direct_return_winner: Optional[CandidateName] = None
        # Phase 168 BLOCKER #9 fix (reviews iter 1): __init__ takes NO features (push_pool builds the
        # pool with selection_mode only). Initialize the flag+oracle to safe defaults; the REAL
        # set-site is add() below, which lifts them off the per-call `features` argument (idempotent
        # on first call). _best_two_tier (Plan-04 Stage B) reads self._enable_triviality_controller.
        self._enable_triviality_controller: bool = False
        self._triv_oracle = None
        self._triv_flag_initialized: bool = False

    def _bind_triviality_features(self, features: Any) -> None:
        """Bind the Phase 168 controller flag/oracle off the authoritative per-call ``features``.

        WR-06 (code review 2026-05-30): the flag is a per-``Orthonym()``-call constant (namer
        stashes it on ``features`` in ``_classify`` before ``assemble_name``), so it is bound at
        pool CONSTRUCTION via ``push_pool(features)`` / ``clear_pool(features)`` rather than
        lifted off whichever candidate happens to call ``add()`` first. Idempotent: marks the
        pool initialized so the ``add()`` fallback below does not re-lift.
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
        # WR-06 (code review 2026-05-30): the controller flag/oracle are bound at pool
        # CONSTRUCTION from the authoritative per-call `features` (push_pool/clear_pool ->
        # _bind_triviality_features). This lazy lift is now only the FALLBACK for pools created
        # WITHOUT features — get_current_pool()'s auto-push and isolated unit tests — binding off
        # the first add()'s features idempotently. In production the pool is already initialized
        # at construction, so this no-ops (it never depends on add() ordering).
        if not self._triv_flag_initialized:
            self._bind_triviality_features(features)
        policy = HANDLER_POLICIES.get(handler_id)
        # Compute confidence WITHOUT parent_atom_indices (Risk 1 mitigation)
        cand = compute_confidence(name, handler_id, features)
        # RATIO_REJECT_FLOOR sanity gate (Phase 145.2 -a.2). Rejects
        # obvious-garbage candidates regardless of handler tier. Missing
        # 'ratio' factor treated as "not garbage" (defensive default).
        ratio_val = cand.factors.get('ratio')
        if ratio_val is not None and ratio_val < RATIO_REJECT_FLOOR:
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
        # Tier B gate enforcement (preserves _confidence_gate behavior).
        # Phase 146 SC-8: the gate threshold is read from the module-level
        # CONFIDENCE_GATE_THRESHOLD (0.40) rather than the now-deleted
        # policy.gate_threshold field. Tier B membership is determined by
        # policy.tier == 'ring_b' (13 handlers: oxime/.../partial_sat).
        if policy is not None and policy.tier == 'ring_b':
            if cand.confidence < CONFIDENCE_GATE_THRESHOLD:
                # Gate rejected — return None so caller falls through
                if _ledger_on():
                    _ledger_record(handler_id, _LedgerStage.GATE_REJECTED, name,
                                   detail=f"tier_b_confidence:{cand.confidence:.3f}")
                return None
        # Set parent_atom_indices POST-HOC (Risk 1)
        cand.parent_atom_indices = parent_atom_indices
        # Phase 147: attach ring_info POST-HOC (Risk 1 pattern).
        # NEVER passed into compute_confidence() — that would break the
        # byte-identical guarantee per Phase 146.
        cand.ring_info = ring_info
        # Phase 165 SCORE-01: attach the structured Name-Tree POST-HOC (Risk 1
        # pattern). NEVER passed into compute_confidence() — byte-identical
        # preserved (Phase 146). best surfaces the winner's tree.
        cand.tree = tree
        # Phase 166 SCORE-03: attach per-node scores POST-HOC (Risk 1 pattern).
        # NEVER into compute_confidence(); score_tree returns {} in production
        # (no reference name set) and on coarse trees, so cand.confidence stays
        # byte-identical (SCORE-05). Contrast the :810-823 multiple_bond_count
        # recompute — that is the ANTI-model; node_scores never touches confidence.
        if (PerNodeScorer is not None and tree is not None
                and not is_coarse_node(tree)):
            cand.node_scores = PerNodeScorer.score_tree(tree, features.mol)
        # Phase 168 TRIV-01/02/03: attach the rewritten tree POST-HOC (Risk-1 pattern; SACRED per
        # Phase 165 + 166). NEVER passed into compute_confidence; byte-identical preserved
        # at Stage A per CONTEXT. Default (flag OFF): tree_rewritten == tree (no-op), the
        # controller block is skipped, production reads cand.tree unchanged. Stage B (Plan-04) flips
        # the comparator/serializer to read tree_rewritten via self._enable_triviality_controller.
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
        # Phase 147 fallback: if ring_info wasn't passed explicitly, read
        # the transient attribute set by namer.py:_classify (allows existing
        # composer.py call sites to flow ring_info through without a
        # signature change at every site).
        if cand.ring_info is None:
            cand.ring_info = getattr(features, '_ring_info', None)
        # Stash mol reference for _filter_lowest_locants atom walks.
        # Transient runtime attribute; not a CandidateName field.
        cand._mol_ref = features.mol
        # Stash principal_group_atoms for PG-locant cascade extraction
        # (used by Tier-1 step 6 / _filter_lowest_locants).
        cand._principal_group_atoms = list(
            getattr(features, 'principal_group_atoms', None) or []
        )
        # IM-01: stash FG name so the cascade can route through
        # ``_pg_attachment_atoms`` (per-FG attachment override). Without
        # this the disulfide bug (flanking-C as attachment) would persist
        # in the candidate-pool path even after parent_selection.py is fixed.
        cand._principal_group = getattr(features, 'principal_group', None)
        # Phase 146 CD-01: populate parent_pcg_count POST-HOC.
        # Same Risk 1 mitigation as parent_atom_indices: NEVER passed into
        # compute_confidence (would break byte-identical guarantees).
        # See _count_pcgs_in_parent above for the algorithm.
        cand.parent_pcg_count = _count_pcgs_in_parent(cand, features)
        # Phase 146 + Risk 1 preservation: POST-HOC patch
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
        from .coverage_scoring import FACTOR_WEIGHTS, _compute_multiple_bond_count
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
        # Inline parent_correctness scoring.
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
        if _ledger_on():
            _ledger_record(handler_id, _LedgerStage.PRODUCED, name,
                           detail=f"confidence:{cand.confidence:.3f}")
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

        Phase 146's planned flip to 'score_based' as the production default
        never landed (Phase 166 SCORE-04, audit §Stale-Comment Inventory):
        production stays 'first_applicable'. The default-OFF
        'score_based_per_substring' mode (the deferred cutover) competes
        the full candidate set via _best_two_tier(per_substring=True) ->
        select_best_candidate().
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
        # Phase 146 SC-6: two-tier selector.
        # Tier 1: Blue Book P-44.1 lexicographic cascade (5 immediate filters
        # + step 6 gated on iupac_locants per).
        # Tier 2: weighted-sum tiebreak via select_best_candidate.
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
        # Phase 166 SCORE-04: the default-OFF 'score_based_per_substring' mode
        # adds a lexicographic per-substring refinement (parent->locant->
        # substituent,) BEFORE the Tier-2 aggregate fallback. The existing
        # 'score_based' path is byte-identical (per_substring=False).
        return self._best_two_tier(
            per_substring=(self.selection_mode == 'score_based_per_substring')
        )

    def all_candidates(self) -> List[CandidateName]:
        """Return all collected candidates (for logging/diagnostics)."""
        return list(self._candidates)

    def _best_two_tier(self, *, per_substring: bool = False) -> Optional[CandidateName]:
        """Phase 146 Tier-1 lexicographic cascade + Tier-2 weighted-sum tiebreak.

        Implements Blue Book P-44.1 per CONTEXT.md <domain>:
          Step 1: P-44.1.1   max PCG suffix count
          Step 2: P-44.1.2   senior heteroatom class (N > P > ... > C)
          Step 3: P-52.2.8   ring-over-chain on tie
          Step 4: P-44.4.1.1 max skeletal atoms
          Step 5: P-44.4.1.2 max multiple bonds
          Step 6: P-44.4.1.4+ locant criteria (gated on iupac_locants per)

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

        # Step 6 — gated on iupac_locants per.
        # Phase 146: _has_iupac_locants returns False -> step 6 is no-op.
        # Phase 147: populates iupac_locants -> step 6 activates.
        if len(candidates) > 1 and _has_iupac_locants(candidates):
            candidates = _filter_lowest_locants(candidates)

        if len(candidates) == 1:
            return candidates[0]

        # Phase 166 SCORE-04: per-substring lexicographic refinement,
        # default-OFF — only 'score_based_per_substring' passes per_substring=True.
        # Reduce to the candidate(s) no other beats by parent->locant->
        # substituent first-point-of-difference; a remaining full tie (all
        # compare_by_node_scores == 0) falls through to the Tier-2 aggregate
        # (STRICT refinement). per_substring=False => byte-identical to today.
        if per_substring and len(candidates) > 1:
            from .per_substring_scoring import _candidate_scores, _scores_from, compare_scores
            # PHASE 168 STAGE B CUTOVER (CONTEXT; #1 + BLOCKER #4 fix, reviews iter 1):
            # read self._enable_triviality_controller from the pool's OWN state (set by add() per
            # Plan-02 2c-2; NOT a non-existent features_or_pool param). When ON, key each candidate
            # on (node_scores_rewritten, tree_rewritten) via the pure _scores_from helper — NEVER
            # mutate cand.node_scores (no aliasing, no restore). Flag-OFF: _key == _candidate_scores
            # and compare_scores == the prior compare_by_node_scores body -> byte-identical.
            def _key(cand):
                if self._enable_triviality_controller:
                    return _scores_from(
                        getattr(cand, 'node_scores_rewritten', None),
                        getattr(cand, 'tree_rewritten', None),
                    )
                return _candidate_scores(cand)
            best = candidates[0]
            best_key = _key(best)
            for cand in candidates[1:]:
                ck = _key(cand)
                if compare_scores(ck, best_key) < 0:
                    best, best_key = cand, ck
            candidates = [c for c in candidates if compare_scores(_key(c), best_key) == 0]
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
    benchmark runners (scripts/benchmark_multi_corpus.py uses ThreadPoolExecutor)
    have independent pool stacks with no cross-thread leakage.
    """
    if not hasattr(_pool_store, 'stack'):
        _pool_store.stack = []
    return _pool_store.stack


def push_pool(features: Any = None) -> CandidatePool:
    """Push a fresh pool onto the per-thread stack and return it.

    Called by assemble_name() prologue (composer.py). MUST be paired with
    pop_pool() in a finally clause so the pool is popped on every exit path
    (normal return, exception, early return statements).

    Phase 146: selection_mode defaults to the ORTHONYM_SELECTION_MODE
    env var (default 'first_applicable' for V17 byte-identical soak). Set
    ORTHONYM_SELECTION_MODE=score_based to activate the two-tier selector
    end-to-end without a code revert.

    WR-06 (code review 2026-05-30): when ``features`` is provided, the Phase 168
    controller flag/oracle are bound NOW (construction) off that authoritative
    per-call features object, instead of being lifted lazily off the first
    add() call. ``features=None`` (get_current_pool auto-push, tests) leaves the
    pool's flag at its safe default and the add() fallback binds it later.

    Returns the new pool that is now the active cascade scope for the
    currently-executing assemble_name() call.
    """
    stack = _ensure_stack()
    new_pool = CandidatePool(selection_mode=_DEFAULT_SELECTION_MODE)
    if features is not None:
        new_pool._bind_triviality_features(features)
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


def clear_pool(features: Any = None) -> None:
    """BACKWARD-COMPAT: replace the top of the stack with a fresh pool.

    Pre-fix code (Plan 03) called clear_pool() in assemble_name()'s prologue
    to reset state. Post-fix, push_pool() is the right primitive (the wrapping
    try/finally handles pop). This alias preserves the call-site signature
    so any handler still calling clear_pool() directly does not break.

    Behavior: if the stack is empty, push a fresh pool (same as pre-fix
    lazy init); if non-empty, replace the top so the current cascade restarts
    cleanly without affecting outer cascades on the stack.

    WR-06 (code review 2026-05-30): binds the Phase 168 controller flag/oracle
    off ``features`` at construction (see push_pool) when provided.
    """
    stack = _ensure_stack()
    if stack:
        new_pool = CandidatePool(selection_mode=_DEFAULT_SELECTION_MODE)
        if features is not None:
            new_pool._bind_triviality_features(features)
        stack[-1] = new_pool
    else:
        push_pool(features)


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

# UNREACHABLE IN PRODUCTION -- 2, 2026-07-30. A permanent stub: both import
# targets (composer.name_chain, composer._name_chain) are ABSENT from composer.py,
# verified in-process, so this returns None unconditionally even with the selection
# mode flipped. The V18 chain-vs-ring competitor it was written for never existed.
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
