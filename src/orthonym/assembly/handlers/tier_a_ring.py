"""Phase 160 Plan-06 composite handler: Tier-A ring cascade.

Per CONTEXT D-28 (gap-closure) + ADR-19-02 §3.2 Option A: encodes the
Tier-A pool-compete cascade as a SINGLE composite handler so the
dispatch_inner first-match-wins interface (CONTEXT D-22) stays untouched.

LIFT SOURCE: composer.py:996-1322 (verbatim, with ``return pool.best().name``
or ``return candidate_name`` replaced by ``return NamingResult(name=...,
tree=None, atom_to_locant_hint=None)``).

Internal cascade order (preserved verbatim from composer.py inline body):
1. complex_ring detection + assembly (composer.py:1014-1066) — pushed to pool
2. polycyclic fallback when complex_ring rejected (composer.py:1083-1098) — direct return
3. partial_sat fallback when complex_ring rejected (composer.py:1103-1110) — direct return
4. heterocycle + lactone safety net (composer.py:1122-1152)
5. benzene (composer.py:1159-1173)
6. Phase 146 chain push-to-pool (composer.py:1185-1192)
7. select_best_candidate selector + handler-level stereo injection (composer.py:1197-1305)

Returns None if all sub-paths fall through AND no Tier-A candidate ratio
gate accepted. The surrounding dispatch_inner loop then continues to
general_acyclic catch-all (Plan-08).

CRITICAL PRESERVATION CONSTRAINTS:
- The ``_tier_a_pool_count_before_complex`` slicing pattern at
  composer.py:1197 MUST be preserved verbatim per Phase 153 D-02 + the
  PHASE 146 PRESERVE block at composer.py:964-994.
- The two ``inject_stereo_from_locant_map`` call sites (composer.py:1301)
  are part of this composite per CONTEXT D-13 + Phase 152 D-04 layering.
- ``assemble_ion_name`` pre-pool bypass (composer.py:751-768) is NOT in
  this composite — that's the outer pre-pool path before dispatch_inner
  runs per CONTEXT D-09.

IUPAC cite: P-44.1 (principal chain vs ring competition).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)


def _is_tier_a_ring(features: Any) -> bool:
    """Predicate: matches if features.is_cyclic AND not chain_is_parent.

    Per CONTEXT D-25 + AP-160-26 predicate-purity. Pure read-only.
    """
    if not getattr(features, 'is_cyclic', False):
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    return True


def name_tier_a_ring(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Composite handler encoding the Tier-A pool-compete cascade.

    Per ADR-19-02 §3.2 + CONTEXT D-28: dispatch_inner cannot natively
    model pool-compete semantics; this composite encodes the full
    cascade in one entry while keeping dispatch_inner first-match-wins
    interface unchanged (CONTEXT D-22).

    Two early-return paths preserved verbatim from composer.py:996-1322:
    - The lactone safety-net path (composer.py:1126-1136) returns the
      lactone name directly.
    - The polycyclic / partial_sat post-rejection fallback paths
      (composer.py:1083-1110) return their handler name directly.

    Byte-identical preservation per DECOMP-03: every pool.add call, the
    ``_tier_a_pool_count_before_complex`` slicing pattern, the
    ``del _tier_a_pool._candidates[...]`` truncation, and the
    handler-level stereo-injection block are preserved verbatim.

    Returns:
        NamingResult: selector chose a candidate OR an early-return
            sub-path produced a name.
        None: all sub-paths fell through AND the selector rejected the
            Tier-A subset (low-ratio rescue did not fire).
    """
    from ..candidate_pool import get_current_pool
    from ..coverage_scoring import (
        select_best_candidate, store_confidence, log_confidence,
    )
    from ..composer import (
        _is_complex_ring_system,
        _assemble_complex_ring_name,
        _enrich_complex_ring_with_subs,
        _complex_ring_parent_atom_indices,
        _assemble_polycyclic_name,
        _try_partially_saturated_carbocycle,
        _enrich_handler_name,
        _assemble_heterocycle_name,
        _assemble_benzene_name,
        _ring_handler_parent_atom_indices,
        _ring_is_whole_molecule_for_complex,
    )

    # =========================================================================
    # TIER A RING COMPETITION — Phase 145.1 routes through CandidatePool
    # =========================================================================
    # PHASE 146 PRESERVE — DO NOT remove this comment block. The Tier A
    # subset slicing pattern below MUST survive Phase 146's gate deletion
    # per Phase 153 D-02 + the inline body's PRESERVE marker.
    # =========================================================================

    _complex_ring_accepted = False
    _complex_result_for_injection: Optional[Any] = None
    _tier_a_pool = get_current_pool()
    _tier_a_pool_count_before_complex = len(_tier_a_pool.all_candidates())

    # === Sub-path 1: complex_ring (composer.py:1025-1077 verbatim) ===
    if (
        features.is_cyclic
        and not getattr(features, 'chain_is_parent', False)
        and _is_complex_ring_system(features.mol)
    ):
        complex_result = _assemble_complex_ring_name(features.mol, features)
        if complex_result:
            complex_name = complex_result.name
            if (not complex_result.substituents_included
                    and complex_result.atom_to_locant):
                complex_name = _enrich_complex_ring_with_subs(
                    features.mol, complex_name,
                    complex_result.ring_atoms,
                    complex_result.atom_to_locant,
                )
            _complex_result_for_injection = complex_result
            _complex_cand = _tier_a_pool.add(
                complex_name, 'complex_ring', features,
                parent_atom_indices=_complex_ring_parent_atom_indices(
                    complex_result, features.mol,
                ),
            )
            if _complex_cand is not None:
                if _tier_a_pool.selection_mode == 'score_based':
                    _complex_ring_accepted = True
                elif _complex_cand.factors.get('ratio', 0) >= 0.40:
                    _complex_ring_accepted = True
        # If complex ring naming fails, fall through.

    # === Sub-path 2 fallback: polycyclic + partial_sat post-rejection
    # (composer.py:1083-1110 verbatim) ===
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        polycyclic_name = getattr(features, 'polycyclic_name', None)
        if polycyclic_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "polycyclic", _ha, (polycyclic_name or "")[:60],
                )
            poly_assembled = _assemble_polycyclic_name(features, style)
            pool = get_current_pool()
            pool.add(poly_assembled, "polycyclic", features)
            return NamingResult(
                name=pool.best().name, tree=None, atom_to_locant_hint=None,
            )

        if features.is_cyclic and not getattr(features, 'chain_is_parent', False):
            partial_sat_name = _try_partially_saturated_carbocycle(features.mol)
            if partial_sat_name:
                partial_sat_name = _enrich_handler_name(
                    features, partial_sat_name, "partial_sat",
                )
                pool = get_current_pool()
                cand = pool.add(partial_sat_name, "partial_sat", features)
                if cand is not None:
                    return NamingResult(
                        name=cand.name, tree=None, atom_to_locant_hint=None,
                    )

    # === Sub-paths 3 & 4: heterocycle + benzene (composer.py:1116-1173) ===
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        ring_type = getattr(features, 'ring_type', None)
        if ring_type and ring_type.startswith('heterocyclic'):
            # Lactone safety net.
            from ...rules.lactones import (
                is_monocyclic_lactone, name_monocyclic_lactone,
            )
            lactone_info = is_monocyclic_lactone(features.mol)
            if lactone_info:
                lactone_name = name_monocyclic_lactone(features.mol)
                if lactone_name:
                    pool = get_current_pool()
                    pool.add(lactone_name, "lactone", features)
                    return NamingResult(
                        name=pool.best().name, tree=None, atom_to_locant_hint=None,
                    )
            # Heterocycle candidate push.
            hetero_name = _assemble_heterocycle_name(features, style)
            if hetero_name:
                _tier_a_pool.add(
                    hetero_name, 'heterocycle', features,
                    parent_atom_indices=_ring_handler_parent_atom_indices(
                        features, 'heterocycle',
                    ),
                )

        if getattr(features, 'is_benzene', False):
            benzene_name = _assemble_benzene_name(features, style)
            if benzene_name:
                _tier_a_pool.add(
                    benzene_name, 'benzene', features,
                    parent_atom_indices=_ring_handler_parent_atom_indices(
                        features, 'benzene',
                    ),
                )

    # === Sub-path 5: Phase 146 chain push-to-pool (composer.py:1185-1192) ===
    if _tier_a_pool.selection_mode == 'score_based':
        from ..candidate_pool import compute_chain_candidate
        _chain_cand = compute_chain_candidate(features, style=style)
        if _chain_cand is not None:
            _tier_a_pool.add(
                _chain_cand.name, 'chain', features,
                parent_atom_indices=_chain_cand.parent_atom_indices,
            )

    # === Selector + handler-level stereo injection (composer.py:1197-1321) ===
    # PHASE 146 PRESERVE: the slicing pattern below MUST survive Phase 146's
    # gate deletion (see comment block at top of this section).
    _tier_a_candidates_added = (
        _tier_a_pool.all_candidates()[_tier_a_pool_count_before_complex:]
    )
    if _tier_a_candidates_added:
        best = select_best_candidate(_tier_a_candidates_added)
        log_confidence(best)
        from ..fragment_naming import is_top_level_naming
        if is_top_level_naming():
            store_confidence(best)
        # D-10 PRESERVED: low-heavy-atom rescue fallback.
        _MIN_RATIO_FALLBACK = 0.20
        total_heavy = features.mol.GetNumHeavyAtoms()
        if total_heavy <= 15 or best.factors.get('ratio', 0) >= _MIN_RATIO_FALLBACK:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    best.handler, total_heavy, best.name[:60],
                )
            candidate_name = best.name
            if best.handler in ('benzene', 'heterocycle', 'complex_ring'):
                _parent_set = best.parent_atom_indices
                _real_coverage = (
                    len(_parent_set) / max(features.mol.GetNumHeavyAtoms(), 1)
                    if _parent_set
                    else 0.0
                )
                if is_top_level_naming() and _real_coverage >= 0.99:
                    from ...rules.stereochemistry import (
                        needs_stereo_injection, inject_stereo_from_locant_map,
                    )
                    if needs_stereo_injection(features.mol, candidate_name):
                        if best.handler == 'benzene':
                            atom_to_locant = getattr(
                                features, 'benzene_atom_to_locant', None,
                            )
                            _inpe = True
                        elif best.handler == 'heterocycle':
                            atom_to_locant = getattr(
                                features, 'heterocycle_atom_to_locant', None,
                            )
                            _inpe = True
                        else:  # complex_ring
                            atom_to_locant = (
                                _complex_result_for_injection.atom_to_locant
                                if _complex_result_for_injection is not None
                                else None
                            )
                            _inpe = _ring_is_whole_molecule_for_complex(
                                _complex_result_for_injection, features.mol,
                            )
                        candidate_name = inject_stereo_from_locant_map(
                            candidate_name, features.mol, atom_to_locant,
                            include_near_parent_ez=_inpe,
                        )
            return NamingResult(
                name=candidate_name, tree=None, atom_to_locant_hint=None,
            )
        # Low ratio: fall through but store metadata for debugging.
        logger.debug(
            "Coverage gate: best candidate ratio too low, falling through "
            "to chain naming. best_handler=%s best_confidence=%.4f ratio=%.4f",
            best.handler, best.confidence, best.factors.get('ratio', 0),
        )
        # BYTE-IDENTICAL FIX: truncate rejected Tier-A candidates so the
        # outer fallback doesn't see them via pool.best().
        del _tier_a_pool._candidates[_tier_a_pool_count_before_complex:]

    # All sub-paths fell through; signal dispatch_inner to continue.
    return None


__all__ = ["name_tier_a_ring", "_is_tier_a_ring"]
