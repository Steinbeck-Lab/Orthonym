"""Phase 160.2 general_acyclic catch-all handler (DECOMP-01 closure).

Per CONTEXT D-08 + ADR-19-02 §3.1 + CONTEXT D-12 RESEARCH §10: explicit
catch-all (priority 99999, predicate=lambda *_: True) closing the
Plan-02 fallthrough gap so dispatch_inner first-match-AND-succeeds-wins
(ADR-19-04) ALWAYS returns a non-None InnerDispatchResult.

LIFT SOURCE: composer.py:951-1055 chain-fallback section (verbatim, with
helper calls re-routed from local composer scope to
``from ._handler_shared import _generate_chain_parent, _generate_ring_parent,
_generate_suffix, _generate_prefixes, _generate_stereodescriptors,
_assemble_fragments``; final ``return candidate_name`` replaced by a
pool-carried ``NamingResult(name=best.name, tree=best.tree, ...)``).

IUPAC cite: P-14 + P-23 + P-44 (catch-all substitutive nomenclature).

Phase 165 SCORE-01: emits a STRUCTURED NameTreeNode derived from its own
fragment list via ``fragments_to_tree`` (Path A), attached through the
CandidatePool ``tree=`` carry. The returned tree is read off the winning
candidate (``best().tree``) so a higher-priority handler's win is never
mislabelled with a general_acyclic tree.

CRITICAL: this handler lifts ONLY the chain-fallback section. The
cycloalkane stereo backstop at composer.py:1057-1097 STAYS in
_assemble_name_impl (orchestrator applies it AFTER dispatch_inner returns)
per CONTEXT D-04.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)


def _is_general_acyclic(features: Any) -> bool:
    """Catch-all predicate per CONTEXT D-08 + AP-160-08.

    AP-160.2-06 CASE B refinement (Phase 160.2 Plan-02-03 honest-fail-on-data):
    the catch-all MUST mirror the chain-fallback section's effective domain
    in composer.py:_assemble_name_impl. The inline cascade still contains
    amide+amine branches (composer.py:917-931) that the handler-side
    _is_amide / _is_amine predicates REJECT (polyfunctional + Tier-A
    mutexes that the inline branches do NOT enforce). A pure
    True-always catch-all would preempt those inline branches and break
    byte-identical for polyfunctional amide / amine cases.

    Pure read-only per CONTEXT D-25 + AP-160-26.

    Inline-cascade-order mirror — returns False (defer to inline cascade) when:
    1. principal_group is amide AND pg_count == 1
       (inline amide branch at composer.py:917-931 handles this).
    2. principal_group is amine
       (inline amine branch at composer.py:919-931 handles this).
    Otherwise returns True (general acyclic catch-all fires).

    Refinement REVERTS automatically when Plan-02-04 ships in CASE A
    (predicate parity verified for amide / amine handlers); at that point
    the inline cascade is deleted and this predicate can collapse back to
    `return True` per the original CONTEXT D-08 catch-all spec.
    """
    pg = getattr(features, 'principal_group', None)
    # Mirror composer.py:917-919 inline amide branch guard.
    if pg in ('primary_amide', 'secondary_amide', 'tertiary_amide'):
        pg_atoms = getattr(features, 'principal_group_atoms', None)
        pg_count = len(pg_atoms) if pg_atoms else 1
        if pg_count == 1:
            return False
    # Mirror composer.py:919 inline amine branch guard.
    if pg in ('secondary_amine', 'tertiary_amine'):
        return False
    return True


def name_general_acyclic(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """General catch-all chain/ring fallback (DECOMP-01 closure; Phase 160.2).

    Verbatim lift of composer.py:951-1055 (chain-fallback section). Helper
    calls re-routed from composer.py local scope to ``_handler_shared``
    imports. String output MUST be byte-identical per DECOMP-03 contract.

    Returns NamingResult(name=..., tree=<structured NameTreeNode>,
    atom_to_locant_hint=None); the tree is read off the winning pool candidate
    (Phase 165 SCORE-01).
    """
    # Lazy imports per PATTERNS § Lazy Import — break the
    # composer.py <-> general_acyclic.py cycle that the inner_dispatch
    # registration will create at module load time.
    from ..candidate_pool import get_current_pool
    from ..composer import (
        HandlerResult,
        NameFragment,
        _get_parent_atom_count,
    )
    from ._handler_shared import (
        _assemble_fragments,
        _generate_chain_parent,
        _generate_prefixes,
        _generate_ring_parent,
        _generate_stereodescriptors,
        _generate_suffix,
        _w2_atom_coverage_declines,
    )

    # === Body: verbatim lift of composer.py:951-1055 (chain-fallback section) ===
    fragments = []

    # Generate parent name (chain or ring)
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    elif features.ring_systems:
        parent = _generate_ring_parent(features)
    else:
        parent = NameFragment(text="", fragment_type="parent")

    # Wave2 T3a conservation: an EMPTY parent stem cannot describe any
    # molecule — the assembler would glue suffixes onto the bare 'ane'
    # filler ('anedicarboxylic acid' for the Si-bridge witness once the
    # benzene handler learned to decline it). Per the ADR-19-04 handler
    # contract, return None to cascade (-> honest unknown) instead.
    if not parent.text:
        logger.debug(
            "fallback_chain_ring decline: empty parent stem (unnameable "
            "parent hydride) smiles=%s", features.canonical_smiles,
        )
        return None

    fragments.append(parent)

    # Generate suffix for principal group
    suffix = None
    if features.principal_group:
        suffix = _generate_suffix(features)
        if suffix:
            # Validate suffix locants against parent capacity
            parent_size = _get_parent_atom_count(features)
            from ...rules.locant_validation import validate_suffix_locants
            validated_locants, validated_count = validate_suffix_locants(
                list(suffix.locants), parent_size, suffix.count
            )
            if validated_locants != list(suffix.locants) or validated_count != suffix.count:
                # M3 Task 1: carry `atoms` through the rebuild -- omitting it
                # would silently re-null the just-populated suffix atoms and
                # revert this molecule to the coverage close's skip default.
                suffix = NameFragment(
                    text=suffix.text,
                    locants=tuple(validated_locants),
                    fragment_type="suffix",
                    count=validated_count,
                    atoms=suffix.atoms,
                )
            fragments.append(suffix)

    # Generate prefixes for substituents and non-principal groups
    prefixes = _generate_prefixes(features)
    fragments.extend(prefixes)

    # Stereodescriptors. P-91.3 (BB:44639): a parent-scope descriptor's locant is read in
    # the PARENT's numbering. That rule now lives in _generate_stereodescriptors itself
    # (it guards both ring maps on `principal_chain`), so the per-caller override this
    # site used to pass is redundant and was removed -- see the shared function.
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    # P-14.3.4.5 (BB:3007) at PARENT scope -- `heptafluorobutanoic acid` (BB:3017,
    # verbatim (PIN)). Decided HERE for the same reason as the mononuclear rule and
    # the P-14.3.4.3 licence below: the rule counts the PARENT COMPOUND's
    # substitutable hydrogens and only `features` carries the structure. The licence is
    # applied by REBUILDING THE PREFIX FRAGMENTS WITHOUT LOCANTS rather than by a
    # print-time flag, so every renderer downstream agrees -- see the helper's
    # docstring § "Why the fragments and not a flag".
    from ._handler_shared import (
        _l5_prefix_locants_omitted,
        _prefix_fragments_without_locants,
    )
    if _l5_prefix_locants_omitted(features, fragments):
        fragments = _prefix_fragments_without_locants(fragments)

    # Assemble in correct order.
    # P-16.5.1.3.1 mononuclear enclosing rule is keyed on a STRUCTURAL property:
    # the perceived parent skeleton has exactly ONE heavy atom (parent atom count
    # == 1), of ANY element. Detect it here — where `features` is available — and
    # thread the boolean into the assembler. This replaces the former
    # stem-string (`parent_frag.text == "meth"`) + no-suffix gate, which was a
    # molecule-class band-aid.
    is_mononuclear_parent = (_get_parent_atom_count(features) == 1)
    # P-14.3.4.3 (BB:2939) is decided HERE for the same reason as the mononuclear
    # rule above: the licence needs the STRUCTURE (the parent compound's
    # substitutable-hydrogen orbits), and `_assemble_fragments` receives only name
    # fragments. `_l3_prefix_locant_omitted` delegates the rule itself to
    # `assembly.locant_omission`; deny-by-default, so False whenever the licence
    # cannot be positively established.
    #
    # Applied by REBUILDING THE PREFIX FRAGMENTS, exactly like P-14.3.4.5 above and
    # for the same measured reason: it used to be a print-time flag threaded into
    # `_assemble_fragments` only, which the name-tree serializer -- the production
    # composition site for this class -- never saw, so the two renderers disagreed
    # on `chloropropanedioic acid` and `_serializer_flip_or_name` silently kept the
    # legacy string. See `_prefix_fragments_without_locants`.
    from ._handler_shared import _l3_prefix_locant_omitted
    if _l3_prefix_locant_omitted(features, fragments):
        fragments = _prefix_fragments_without_locants(fragments)

    # P-14.3.4.4 (BB:2953) at PARENT scope -- `diphenylethanedione` (BB:28338,
    # verbatim (PIN)) and `di(naphthalen-2-yl)ethanedione` (BB:28380). Decided HERE
    # for the same reason as the two licences above: the ISOMER-COUNT test needs the
    # STRUCTURE (the parent hydride's positions and the decorations' bond orders),
    # and `_assemble_fragments` receives only name fragments. `_l4_locants_omitted`
    # delegates the rule itself to `assembly.locant_omission`; deny-by-default.
    #
    # Applied by REBUILDING the fragments so every renderer downstream agrees, like
    # the two above. Unlike L3/L5, L4 empties the WHOLE scope, so the SUFFIX locants
    # are cleared too (`ethane-1,2-dione` -> `ethanedione`); the helper's
    # fail-closed check keeps a half-stripped name from shipping.
    from ._handler_shared import _l4_locants_omitted, _fragments_without_locants
    if _l4_locants_omitted(features, fragments):
        fragments = _fragments_without_locants(fragments, is_mononuclear_parent)
    assembled = _assemble_fragments(
        fragments, style, is_mononuclear_parent=is_mononuclear_parent,
    )

    # Observational coverage logging for fallback chain/ring path (ARCH-06)
    if logger.isEnabledFor(logging.DEBUG):
        _fb_total_ha = features.mol.GetNumHeavyAtoms()
        _fb_parent = set(features.principal_chain or []) | set(getattr(features, 'principal_ring', None) or [])
        _fb_accounted = set(_fb_parent)
        # Include FG atoms
        for _fb_fg_matches in getattr(features, 'functional_groups', {}).values():
            for _fb_m in _fb_fg_matches:
                _fb_accounted.update(_fb_m)
        _fb_hr = HandlerResult(
            name=assembled,
            handler_id="fallback_chain_ring",
            parent_atoms=_fb_parent,
            accounted_atoms=_fb_accounted,
            total_heavy_atoms=_fb_total_ha,
        )
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=%.2f accounted=%d/%d name=%s",
            _fb_hr.handler_id, _fb_hr.coverage, len(_fb_hr.accounted_atoms),
            _fb_hr.total_heavy_atoms, _fb_hr.name[:60],
        )

    # ASSEMBLY_AUDIT: detect FGs present in molecule but missing from final name.
    # Guarded by logger level check so there is no performance impact in production.
    if logger.isEnabledFor(logging.DEBUG):
        from ...rules.seniority import PREFIX_FORMS
        detected_fgs = set()
        fg_dict = getattr(features, 'functional_groups', {})
        pg = getattr(features, 'principal_group', None)
        for fg_name_audit, fg_matches in fg_dict.items():
            if fg_name_audit in ('alkene', 'alkyne'):
                continue
            if fg_name_audit == pg:
                continue  # principal group is the suffix, not a prefix
            if fg_matches:
                detected_fgs.add(fg_name_audit)
        missing_fgs = set()
        for fg_audit in detected_fgs:
            prefix = PREFIX_FORMS.get(fg_audit)
            if prefix is None:
                continue  # functional-class-only, no prefix form expected
            if prefix and prefix in assembled:
                continue
            missing_fgs.add(fg_audit)
        if missing_fgs:
            logger.debug(
                "ASSEMBLY_AUDIT: missing_fg=%s in name=%s smiles=%s",
                missing_fgs, assembled, getattr(features, 'canonical_smiles', '?'),
            )

    if _w2_atom_coverage_declines(features, fragments, assembled):
        return None  # task-W2 Witness-B: name silently DROPPED atoms → decline

    # Phase 145.1: route chain-naming through pool.
    # In first_applicable mode, pool.best() returns the FIRST added
    # candidate. If a higher-priority handler already added one above,
    # pool.best() is that one (chain naming computed but not returned).
    # If no other handler fired (this is the only candidate), pool.best()
    # is the chain candidate. D-02: chain has priority=fallback in 145.1
    # (preserves byte-identical); Phase 146 raises priority for competition.
    pool = get_current_pool()
    # Phase 165 SCORE-01: derive a structured tree from the SAME fragment list
    # built above and attach it to the chain candidate. Reading best().tree
    # (NOT the local `tree`) guarantees the returned tree corresponds to the
    # RETURNED name: if a higher-priority handler's candidate wins, best.tree is
    # that handler's tree (or None -> coarse-bucket counted in Plan 04), never a
    # mismatched general_acyclic tree.
    from ..name_tree_builder import fragments_to_tree
    tree = fragments_to_tree(
        fragments, class_id="general_acyclic", section_cite="P-14+P-23+P-44",
        is_mononuclear_parent=is_mononuclear_parent,
    )
    pool.add(assembled, "chain", features, tree=tree)
    best = pool.best()

    # G0 fail-closed safety (DD7): general_acyclic is the @99999 catch-all, but
    # the pool can still be empty when NO candidate could be built (e.g. an
    # all-aromatic-ring + two-metal species like the P-69 gold
    # c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1, which has no nameable
    # acyclic parent). Per the ADR-19-04 handler contract ("return None on
    # gate-fail, never raise"), return None to cascade rather than crash on
    # ``best.name`` — that AttributeError was masked by name_compound's broad
    # except but propagated raw (NoneType .name) through the direct .name() /
    # raise_on_limit API.
    if best is None:
        return None

    return NamingResult(
        name=best.name, tree=best.tree, atom_to_locant_hint=None,
    )


__all__ = ["name_general_acyclic", "_is_general_acyclic"]
