"""a phase Plan-06 composite handler: Tier-A ring cascade.

Per internal notes (gap-closure) + -02 Option A: encodes the
Tier-A pool-compete cascade as a SINGLE composite handler so the
dispatch_inner first-match-wins interface (internal notes) stays untouched.

LIFT SOURCE: composer.py:996-1322 (verbatim, with ``return pool.best.name``
or ``return candidate_name`` replaced by ``return NamingResult(name=...,
tree=<coarse NameTreeNode>, atom_to_locant_hint=None)`` — a phase SCORE-01
attaches a coarse tree at EVERY return site, including the early returns).

Internal cascade order (preserved verbatim from composer.py inline body):
1. complex_ring detection + assembly (composer.py:1014-1066) — pushed to pool
2. polycyclic fallback when complex_ring rejected (composer.py:1083-1098) — direct return
3. partial_sat fallback when complex_ring rejected (composer.py:1103-1110) — direct return
4. heterocycle + lactone safety net (composer.py:1122-1152)
5. benzene (composer.py:1159-1173)
6. a phase chain push-to-pool (composer.py:1185-1192)
7. select_best_candidate selector + handler-level stereo injection (composer.py:1197-1305)

Returns None if all sub-paths fall through AND no Tier-A candidate ratio
gate accepted. The surrounding dispatch_inner loop then continues to
general_acyclic catch-all (Plan-08).

CRITICAL PRESERVATION CONSTRAINTS:
- The ``_tier_a_pool_count_before_complex`` slicing pattern at
  composer.py:1197 MUST be preserved verbatim per a phase + the
  PHASE 146 PRESERVE block at composer.py:964-994.
- The two ``inject_stereo_from_locant_map`` call sites (composer.py:1301)
  are part of this composite per internal notes + a phase layering.
- ``assemble_ion_name`` pre-pool bypass (composer.py:751-768) is NOT in
  this composite — that's the outer pre-pool path before dispatch_inner
  runs per internal notes.

IUPAC cite: (principal chain vs ring competition).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _spiro_principal_suffix_preference(features: Any, current_name: str) -> Optional[str]:
    """ PIN conformance: return the general-engine SUFFIX-form name for
    a spiro parent whose principal characteristic group the complex_ring assembly
    demoted to a prefix (`9-carboxyspiro[5.5]undecane` -> `spiro[5.5]undecane-3-
    carboxylic acid`), or None to keep ``current_name``.

    Root cause (measured 2026-08-10): complex_ring names a suffixable PG as a
    detachable prefix on a spiro parent; von-Baeyer parents already suffix
    correctly, so only spiro is affected. The general engine builds the correct
    suffix form. Prefer it ONLY when OPSIN parses it back to the input structure
    (re-anchored + RT-gated => 0-wrong). Jar-absent -> None (fail closed, keep the
    existing name). Scoped to spiro so von-Baeyer/fused and PG-free spiro are
    byte-identical.
    """
    try:
        from ...rules.spiro import is_spiro_system
        if not is_spiro_system(features.mol):
            return None
        from ..general_engine import name_general
        eng = name_general(
            features.mol, features,
            allow_aromatic_general=False, allow_suffix_free=False)
        if eng is None or not eng.name or eng.name == current_name:
            return None
        from ...namer import _validity_gate_jar_present, _validity_gate_name_to_smiles
        if not _validity_gate_jar_present():
            return None  # fail closed: never ship an unverified rewrite
        smi = _validity_gate_name_to_smiles(eng.name)
        if smi is None:
            return None
        from rdkit import Chem
        if Chem.CanonSmiles(smi) == Chem.CanonSmiles(
                Chem.MolToSmiles(features.mol)):
            return eng.name
    except Exception as e:
        logger.debug("spiro principal-suffix preference skipped: %s", e)
    return None


def _complex_parent_is_spiro(mol: Any, complex_result: Any,
                             principal_group_atoms: Any = None) -> bool:
    """True when the complex-ring parent holds a spiro atom (a spiro union,,
    its assembly left the substituents to the prefix enricher, and a principal
    characteristic group has an atom outside the ring skeleton (the =O of a
    ketone, the -OH of an alcohol): that group is then cited as a prefix. A
    match lying wholly in the ring (a ring P-H or N-H) is part of the parent
    hydride, not a characteristic group of it."""
    if getattr(complex_result, 'substituents_included', True):
        return False
    try:
        from ...perception.rings import get_spiro_atoms
        ring_atoms = set(getattr(complex_result, 'ring_atoms', ()) or ())
        if not set(get_spiro_atoms(mol)) & ring_atoms:
            return False
        return any(
            any(a not in ring_atoms for a in match)
            for match in (principal_group_atoms or ()))
    except Exception:
        return False


def _is_tier_a_ring(features: Any) -> bool:
    """Predicate: matches if features.is_cyclic AND not chain_is_parent.

    Per internal notes + -26 predicate-purity. Pure read-only.
    """
    if not getattr(features, 'is_cyclic', False):
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    return True


def _benzene_is_phenol(features: Any) -> bool:
    """a phase -01 : True iff a hydroxyl (-OH) is directly bonded to
    an aromatic ring carbon of the benzene parent.

    Phenol benzene is EXCLUDED from the backstop inject allowlist (peptides /
    tyrosine-type compounds and ring-OH parents are already configured or carry
    no authoritative orientation for injection per). Pure read-only.
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

    Per -02 + internal notes: dispatch_inner cannot natively
    model pool-compete semantics; this composite encodes the full
    cascade in one entry while keeping dispatch_inner first-match-wins
    interface unchanged (internal notes).

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
    # (a phase): an unsubstituted ring ketone on a mancude ring is the
    # added-indicated-hydrogen form (pyridin-2(1H)-one / naphthalen-1(2H)-one).
    # The default paths drop the C=O (carbocyclic) or name it as a '2-oxo'
    # prefix (heterocyclic). This recognizer is tightly scoped + fail-closed
    # (returns None for everything else), so it preempts only the cases it names
    # correctly and never touches the pool for any other molecule.
    from ...rules.partial_saturation import (
        name_added_h_fused_carbocycle_suffix,
        name_hydro_mancude_fused_carbocycle,
        name_ring_ketone_with_added_indicated_h,
    )
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _assemble_benzene_name,
        _assemble_complex_ring_name,
        _assemble_heterocycle_name,
        _assemble_polycyclic_name,
        _complex_ring_parent_atom_indices,
        _enrich_complex_ring_with_subs,
        _enrich_handler_name,
        _is_complex_ring_system,
        _ring_handler_parent_atom_indices,
        _ring_is_whole_molecule_for_complex,
    )
    from ..coverage_scoring import (
        log_confidence,
        select_best_candidate,
        store_confidence,
    )
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

    # v52 P4 /: the -ol/-amine sibling of the KIH preempt. A
    # mancude naphthalene bearing an -ol/-amine (di-) suffix that requires 'added
    # indicated hydrogen' — naphthalen-4a(2H)-ol, naphthalene-2,4a(2H)-diamine,
    # naphthalene-4a,8a-diol. The default carbocyclic paths fail closed (the bare
    # hydro producer cannot express the suffix), so these otherwise abstain.
    # Tightly scoped + fail-closed (returns None for everything else), so it
    # preempts only the cases it names correctly and never touches the pool.
    _added_h_name = name_added_h_fused_carbocycle_suffix(features.mol)
    if _added_h_name:
        return NamingResult(
            name=_added_h_name,
            tree=NameTreeNode(
                parent_stem=_added_h_name, class_id="tier_a_ring",
                iupac_section_cite="P-14.7.2", fragment_legacy=_added_h_name,
            ),
            atom_to_locant_hint=None,
        )

    # B1b /: the suffix-free sibling of the KIH preempt above.
    # A bare, unsubstituted, all-carbon fused ring system that is a hydro form of
    # a mancude parent carrying INTRINSIC indicated hydrogen — e.g.
    # 10,11-dihydro-5H-dibenzo[a,d][7]annulene (the amitriptyline core),
    # 6,7-dihydro-5H-benzo[7]annulene. Tightly scoped + fail-closed (returns None
    # for everything else), so it preempts only the cases it names correctly and
    # never touches the pool for any other molecule.
    _hydro_name = name_hydro_mancude_fused_carbocycle(features.mol)
    if _hydro_name:
        return NamingResult(
            name=_hydro_name,
            tree=NameTreeNode(
                parent_stem=_hydro_name, class_id="tier_a_ring",
                iupac_section_cite="P-31.1.4", fragment_legacy=_hydro_name,
            ),
            atom_to_locant_hint=None,
        )

    # Lever C (, /: a ring-attached ANILIDE — a senior secondary/
    # tertiary amide whose acyl is a simple unsubstituted carbocycle (benzoyl,
    # cyclohexanecarbonyl) and whose only junior FGs live inside the N-aryl
    # substituent — must be named as the amide parent (benzamide /
    # cyclohexanecarboxamide), NOT with the junior N-aryl ring as parent. Tier-A
    # otherwise picks the phenol/aniline ring and demotes the amide to an
    # 'N-benzoyl' prefix ('N-benzoyl-4-aminophenol', a violation — the amide
    # outranks the phenol). _name_ring_attached_anilide is tightly scoped +
    # fail-closed (returns None for everything else — substituted/fused/hetero
    # acyl rings, defined stereo, un-contained junior FGs), so it preempts only
    # the cases it names correctly and never touches the pool for any other
    # molecule (the same shape as the KIH preempt above).
    from ...rules.polyfunctional import _name_ring_attached_anilide
    _anilide_name = _name_ring_attached_anilide(features)
    if _anilide_name:
        return NamingResult(
            name=_anilide_name,
            tree=NameTreeNode(
                parent_stem=_anilide_name, class_id="tier_a_ring",
                iupac_section_cite="P-66.1", fragment_legacy=_anilide_name,
            ),
            atom_to_locant_hint=None,
        )

    # =========================================================================
    # TIER A RING COMPETITION — a phase routes through CandidatePool
    # =========================================================================
    # PHASE 146 PRESERVE — DO NOT remove this comment block. The Tier A
    # subset slicing pattern below MUST survive a phase's gate deletion
    # per a phase + the inline body's PRESERVE marker.
    # =========================================================================

    _complex_ring_accepted = False
    _complex_result_for_injection: Optional[Any] = None
    _tier_a_pool = get_current_pool()
    _tier_a_pool_count_before_complex = len(_tier_a_pool.all_candidates())

    # task 9 (A-i): the whole-molecule complex-ring path must not
    # preempt the parent decision. With >=2 ring systems, skip it when
    # the parent is NOT the fused/complex system:
    # (a) the among-rings senior system is a MONOCYCLE
    # (2-(1-benzofuran-2-yl)pyridine: pyridine is the parent), or
    # (b) the principal characteristic group sits on a ring system other
    # than the senior one — that system is the parent).
    _skip_complex_non_senior = False
    _ring_systems = getattr(features, 'ring_systems', None) or []

    # / + (BB 33859 imido preferred prefix): classification
    # already chose a PAH parent (features.polycyclic_name set via the namer L1
    # among-rings exemption) BECAUSE the principal characteristic group sits ON
    # the PAH core while a senior fused-heterocyclic ring is only a SUBSTITUENT
    # (5-(1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl)naphthalene-1-carboxylic acid).
    # In that case the whole-molecule complex_ring path must NOT preempt the
    # already-correct polycyclic assembler — the L1 decision is authoritative.
    # The namer's PAH block early-returns before senior_ring_system is set, so
    # the senior-based skip below never fires here; gate directly on the PAH
    # decision instead. Determinism-safe: keyed on canonical PAH-core membership
    # + PCG locus (is_principal_group_on_ring), never ring/registration order.
    _pah_name = getattr(features, 'polycyclic_name', None)
    if (_pah_name
            and features.is_cyclic
            and not getattr(features, 'chain_is_parent', False)
            and getattr(features, 'principal_group_atoms', None)):
        from ...rules.parent_selection import is_principal_group_on_ring
        from ...rules.polycyclics import get_polycyclic_core_atoms
        _pah_core = get_polycyclic_core_atoms(features.mol, _pah_name)
        if _pah_core and is_principal_group_on_ring(
                features.mol, set(_pah_core),
                features.principal_group_atoms,
                features.principal_group):
            _skip_complex_non_senior = True

    if features.is_cyclic and len(_ring_systems) >= 2:
        _senior = getattr(features, 'senior_ring_system', None)
        # The among-rings decision applies to SEPARATE ring systems
        # only. Spiro rings share an ATOM (one ring SYSTEM per yet may
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
            # core-namer FAIL-CLOSED: the enricher returns None when a
            # RING-BEARING compound substituent on the complex-ring parent cannot
            # render. Emitting the bare parent would be an atom-incomplete
            # (wrong-molecule) partial, so fall through to the other tiers rather
            # than pool it -- exactly the "If complex ring naming fails, fall
            # through" contract already documented below.
            if complex_name is not None:
                # PIN conformance: the complex_ring assembly demotes a
                # suffixable principal characteristic group to a PREFIX on a SPIRO
                # parent (`9-carboxyspiro[5.5]undecane`), where the PIN cites it as
                # the principal SUFFIX (`spiro[5.5]undecane-3-carboxylic acid`).
                # von-Baeyer parents already suffix correctly -- only spiro demotes
                # (measured). The general engine builds the correct suffix form;
                # prefer it IFF it OPSIN-round-trips to the input (re-anchored +
                # RT-gated => 0-wrong; jar-absent keeps the existing name,
                # fail-closed). Scoped to spiro+PG, so von-Baeyer / fused parents
                # and PG-free spiro stay byte-identical.
                if getattr(features, 'principal_group', None) is not None:
                    _alt = _spiro_principal_suffix_preference(
                        features, complex_name)
                    if _alt is not None:
                        complex_name = _alt
                    elif _complex_parent_is_spiro(
                            features.mol, complex_result,
                            getattr(features, 'principal_group_atoms', None)):
                        # /: the principal characteristic group
                        # is cited as the suffix of the parent hydride, e.g.
                        # 'spiro[4.5]decane-1,7-dione (PIN)',
                        # the Blue Book). The spiro assembly above cites
                        # every group as a prefix, so with no suffix form this
                        # name ('3',4,4'-trihydroxy-...-spiro[...]') is not the
                        # PIN: it is labelled below it.
                        from ...metrics.provenance import record_non_pin_fragment
                        record_non_pin_fragment(complex_name)
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

    # === Wave2 fail-closed guard: pure-monocyclic spiro system ===
    # A molecule whose rings are monocycles joined ONLY at spiro atoms has no
    # valid fallback below this point: the generic paths name a single ring
    # and silently drop the others (branched trispiro -> 'cyclononane'), or
    # re-perceive a wrong core. If the spiro subsystem itself declined (tri+
    # polyspiro descriptors, compound-locant unsaturation,...), refuse via
    # the G0 UNSUPPORTED_RING_SYSTEM signal — jar-independent, never a wrong
    # name. Fused/bridged hybrids are untouched (their own paths run below).
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        from ...rules.spiro import is_spiro_system as _t6_is_pure_spiro
        if _t6_is_pure_spiro(features.mol):
            from ...errors import unsupported_ring_system
            raise unsupported_ring_system()

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
            # SECOND partial_sat enrichment site (the first is
            # handlers/partial_sat.py). It must inherit the producer's numbering
            # and its already-spelled atom set for the same reasons -- measured
            # on a 7,500-row corpus sweep, this site was the one emitting
            # `1,2-dimethyl-6,7-dimethyl-1,2,3,5,8,8a-hexahydronaphthalene` and
            # `1,2-dimethyl-2,3-dimethyl-perhydronaphthalene`, i.e. every ring
            # methyl cited TWICE, on two different numberings.
            from ..composer import (
                _try_partially_saturated_carbocycle_with_locants,
            )
            produced = _try_partially_saturated_carbocycle_with_locants(features.mol)
            if produced:
                partial_sat_name = _enrich_handler_name(
                    features, produced.name, "partial_sat",
                    atom_to_locant=produced.atom_to_locant or None,
                    already_spelled_atoms=produced.spelled_offring_atoms or None,
                )
                pool = get_current_pool()
                cand = pool.add(partial_sat_name, "partial_sat", features)
                if cand is not None:
                    return NamingResult(
                        name=cand.name,
                        tree=NameTreeNode(parent_stem=cand.name, class_id="tier_a_ring", iupac_section_cite="P-25", fragment_legacy=cand.name),
                        atom_to_locant_hint=produced.atom_to_locant or None,
                    )

    # === Sub-paths 3 & 4: heterocycle + benzene (composer.py:1116-1173) ===
    if not _complex_ring_accepted and not getattr(features, 'chain_is_parent', False):
        ring_type = getattr(features, 'ring_type', None)
        if ring_type and ring_type.startswith('heterocyclic'):
            # Lactone safety net.
            from ...rules.lactones import (
                is_monocyclic_lactone,
                name_monocyclic_lactone,
            )
            from .lactone import _has_separate_senior_group
            lactone_info = is_monocyclic_lactone(features.mol)
            # (the Blue Book) seniority guard: a lactone is a
            # pseudoketone and ranks below an acid or ester (Table 4.1,
            # the Blue Book). When a SEPARATE senior suffix-capable group is present
            # the lactone must NOT claim the parent; fall through to the
            # heterocycle assembler, which expresses the ring C=O as an ``oxo``
            # prefix and the senior group as the suffix (e.g.
            # 5-oxooxolane-2-carboxylic acid). Mirrors the _is_lactone predicate
            # guard so both lactone-as-parent entry points agree.
            if lactone_info and not _has_separate_senior_group(
                features.mol, lactone_info,
            ):
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

    # === Sub-path 5: a phase chain push-to-pool (composer.py:1185-1192) ===
    if _tier_a_pool.selection_mode == 'score_based':
        from ..candidate_pool import compute_chain_candidate
        _chain_cand = compute_chain_candidate(features, style=style)
        if _chain_cand is not None:
            _tier_a_pool.add(
                _chain_cand.name, 'chain', features,
                parent_atom_indices=_chain_cand.parent_atom_indices,
            )

    # === Selector + handler-level stereo injection (composer.py:1197-1321) ===
    # PHASE 146 PRESERVE: the slicing pattern below MUST survive a phase's
    # gate deletion (see comment block at top of this section).
    _tier_a_candidates_added = (
        _tier_a_pool.all_candidates()[_tier_a_pool_count_before_complex:]
    )
    if _tier_a_candidates_added:
        best = select_best_candidate(_tier_a_candidates_added)
        log_confidence(best)
        from ..fragment_naming import is_top_level_naming
        if is_top_level_naming():
            # a phase -01 (/): thread the authoritative benzene
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
        # PRESERVED: low-heavy-atom rescue fallback.
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
                        inject_stereo_from_locant_map,
                        inject_stereo_reanchored_rt_gated,
                        needs_stereo_injection,
                    )
                    if needs_stereo_injection(features.mol, candidate_name):
                        _is_complex = best.handler not in ('benzene', 'heterocycle')
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
                        if _is_complex:
                            # -C1C2C6 Pattern A: the complex_ring atom->locant
                            # map (e.g. a spiro-of-fused-component parent's
                            # combined_locants) can be numbered inconsistently
                            # with the printed descriptor, dropping the stereo
                            # descriptor on the wrong locant -> OPSIN-unparseable.
                            # RT-gate the numbering and re-anchor to OPSIN's own
                            # locants when the first candidate fails to round-trip
                            # (a project rule; byte-identical for every currently-
                            # round-tripping name).
                            candidate_name = inject_stereo_reanchored_rt_gated(
                                candidate_name, features.mol, atom_to_locant,
                                include_near_parent_ez=_inpe,
                            )
                        else:
                            candidate_name = inject_stereo_from_locant_map(
                                candidate_name, features.mol, atom_to_locant,
                                include_near_parent_ez=_inpe,
                            )
            # (the Blue Book): "All preferred IUPAC names for
            # esters are named by functional class nomenclature"; the acyloxy /
            # alkoxycarbonyl prefixes are for an ester beside a senior group
            #,:31698). No producer in this ring cascade builds
            # the functional class name, so when the whole molecule's principal
            # group is an exocyclic ester, this name cites it as a prefix: valid,
            # not preferred ('3-[(3-hydroxy-2-phenylpropanoyl)oxy]-8-methyl-8-
            # azabicyclo[3.2.1]octane'). It ships at the general tier, never as
            # pin_verified -- the same demotion as a general-only ring prefix.
            if (getattr(features, 'principal_group', None) == 'ester'
                    and is_top_level_naming()):
                from ...perception.smarts_cache import compiled as _smarts
                _mol = features.mol
                # an ester whose C(=O)-O bond is not a ring bond (a lactone is
                # named as a ring '-one', which IS its PIN form)
                if any(not _mol.GetBondBetweenAtoms(_m[0], _m[2]).IsInRing()
                       for _m in _mol.GetSubstructMatches(
                           _smarts("[CX3](=O)[OX2][#6]"))):
                    from ...metrics.provenance import record_general_ring_prefix
                    record_general_ring_prefix()
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
        # outer fallback doesn't see them via pool.best.
        del _tier_a_pool._candidates[_tier_a_pool_count_before_complex:]

    # All sub-paths fell through; signal dispatch_inner to continue.
    return None


__all__ = ["name_tier_a_ring", "_is_tier_a_ring"]
