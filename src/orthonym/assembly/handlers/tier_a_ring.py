"""Phase 160 Plan-06 composite handler: Tier-A ring cascade.

Per CONTEXT D-28 (gap-closure) + ADR-19-02 §3.2 Option A: encodes the
Tier-A pool-compete cascade as a SINGLE composite handler so the
dispatch_inner first-match-wins interface (CONTEXT D-22) stays untouched.

LIFT SOURCE: composer.py:996-1322 (verbatim, with ``return pool.best().name``
or ``return candidate_name`` replaced by ``return NamingResult(name=...,
tree=<coarse NameTreeNode>, atom_to_locant_hint=None)`` — Phase 165 SCORE-01
attaches a coarse tree at EVERY return site, including the early returns).

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

from ..name_tree import NamingResult, NameTreeNode

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


def _benzene_is_phenol(features: Any) -> bool:
    """Phase 177 WSB-01 (D-05): True iff a hydroxyl (-OH) is directly bonded to
    an aromatic ring carbon of the benzene parent.

    Phenol benzene is EXCLUDED from the backstop inject allowlist (peptides /
    tyrosine-type compounds and ring-OH parents are already configured or carry
    no authoritative orientation for injection per D-05). Pure read-only.
    """
    mol = getattr(features, 'mol', None)
    ring = getattr(features, 'benzene_ring', None) or getattr(
        features, 'principal_ring', None)
    if mol is None or not ring:
        return False
    ring_set = set(ring)
    for idx in ring_set:
        atom = mol.GetAtomWithIdx(idx)
        if not atom.GetIsAromatic():
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring_set:
                continue
            if nbr.GetSymbol() == 'O' and nbr.GetTotalNumHs() >= 1 \
                    and nbr.GetDegree() == 1:
                return True
    return False


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

    # v23 IH-01 (Phase 2): an unsubstituted ring ketone on a mancude ring is the
    # added-indicated-hydrogen form (pyridin-2(1H)-one / naphthalen-1(2H)-one).
    # The default paths drop the C=O (carbocyclic) or name it as a '2-oxo'
    # prefix (heterocyclic). This recognizer is tightly scoped + fail-closed
    # (returns None for everything else), so it preempts only the cases it names
    # correctly and never touches the pool for any other molecule.
    from ...rules.partial_saturation import name_ring_ketone_with_added_indicated_h
    _kih_name = name_ring_ketone_with_added_indicated_h(features.mol)
    if _kih_name:
        return NamingResult(
            name=_kih_name,
            tree=NameTreeNode(
                parent_stem=_kih_name, class_id="tier_a_ring",
                iupac_section_cite="P-31.1.4.2.4", fragment_legacy=_kih_name,
            ),
            atom_to_locant_hint=None,
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

    # WS-A task 9 (A-i): the whole-molecule complex-ring path must not
    # preempt the P-44 parent decision. With >=2 ring systems, skip it when
    # the parent is NOT the fused/complex system:
    #   (a) the among-rings senior system (P-44.2) is a MONOCYCLE
    #       (2-(1-benzofuran-2-yl)pyridine: pyridine is the parent), or
    #   (b) the principal characteristic group sits on a ring system other
    #       than the senior one (P-44.1 — that system is the parent).
    _skip_complex_non_senior = False
    _ring_systems = getattr(features, 'ring_systems', None) or []
    if features.is_cyclic and len(_ring_systems) >= 2:
        _senior = getattr(features, 'senior_ring_system', None)
        # The among-rings P-44.2 decision applies to SEPARATE ring systems
        # only. Spiro rings share an ATOM (one ring SYSTEM per P-24) yet may
        # arrive as two entries here — skipping the complex path for them
        # dropped half the molecule ('(3R,6S)-oxan-3-ol' for a dioxaspiro
        # parent, self-test RT True->False). Any atom overlap between ring
        # systems disables the skip.
        _systems_overlap = False
        for _i in range(len(_ring_systems)):
            for _j in range(_i + 1, len(_ring_systems)):
                if set(_ring_systems[_i]) & set(_ring_systems[_j]):
                    _systems_overlap = True
                    break
            if _systems_overlap:
                break
        if _senior and not _systems_overlap:
            _senior_set = set(_senior)
            _ri = features.mol.GetRingInfo()
            _n_rings_in_senior = sum(
                1 for _r in _ri.AtomRings() if set(_r) <= _senior_set
            )
            if _n_rings_in_senior <= 1:
                _skip_complex_non_senior = True
            elif getattr(features, 'principal_group_atoms', None):
                from ...rules.parent_selection import is_principal_group_on_ring
                _pg_on_senior = is_principal_group_on_ring(
                    features.mol, _senior_set,
                    features.principal_group_atoms,
                    features.principal_group,
                )
                _pg_on_other = any(
                    is_principal_group_on_ring(
                        features.mol, _rs,
                        features.principal_group_atoms,
                        features.principal_group,
                    )
                    for _rs in _ring_systems
                    if not (set(_rs) & _senior_set)
                )
                if _pg_on_other and not _pg_on_senior:
                    _skip_complex_non_senior = True

    # === Sub-path 1: complex_ring (composer.py:1025-1077 verbatim) ===
    if (
        features.is_cyclic
        and not getattr(features, 'chain_is_parent', False)
        and not _skip_complex_non_senior
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
            _nm = pool.best().name
            return NamingResult(
                name=_nm,
                tree=NameTreeNode(parent_stem=_nm, class_id="tier_a_ring", iupac_section_cite="P-25", fragment_legacy=_nm),
                atom_to_locant_hint=None,
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
                        name=cand.name,
                        tree=NameTreeNode(parent_stem=cand.name, class_id="tier_a_ring", iupac_section_cite="P-25", fragment_legacy=cand.name),
                        atom_to_locant_hint=None,
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
                    _nm = pool.best().name
                    return NamingResult(
                        name=_nm,
                        tree=NameTreeNode(parent_stem=_nm, class_id="tier_a_ring", iupac_section_cite="P-25", fragment_legacy=_nm),
                        atom_to_locant_hint=None,
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
            # Phase 177 WSB-01 (D-04/D-05): thread the authoritative benzene
            # atom_to_locant + phenol flag onto the winning candidate as POST-HOC
            # fields (never into compute_confidence — byte-identity Risk 1) so
            # the namer backstop can inject a missed stereodescriptor on the
            # NON-PHENOL benzene cohort. complex_ring / heterocycle stay log-only
            # (no map threaded -> backstop sees atom_to_locant=None).
            if best.handler == 'benzene':
                _bz_map = getattr(features, 'benzene_atom_to_locant', None)
                if _bz_map:
                    best.atom_to_locant = dict(_bz_map)
                    # Phenol detection: an -OH directly on the benzene ring.
                    best.is_phenol_benzene = _benzene_is_phenol(features)
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
                name=candidate_name,
                tree=NameTreeNode(parent_stem=candidate_name, class_id="tier_a_ring", iupac_section_cite="P-25", fragment_legacy=candidate_name),
                atom_to_locant_hint=None,
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
