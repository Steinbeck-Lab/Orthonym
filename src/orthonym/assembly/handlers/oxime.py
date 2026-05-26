"""Phase 160 oxime handler — Tier B retained-name (gate 0.40).

Verbatim lift of the oxime dispatch logic from composer.py:771-782
(inline branch) + composer.py:1906-2042 (body of _name_oxime_or_hydrazone).
The body itself STAYS in composer.py per CONTEXT D-24 incremental-migration
discipline — this handler module is a thin wrapper that invokes the
existing composer.py logic via lazy import. Plan-03 commit 03-10
(composer.py thinning) deletes the inline body from composer.py once
the inner-dispatch substrate is fully wired.

Byte-identical lock per CONTEXT D-21 (DECOMP-03): the handler's behavior
on every canary fixture MUST equal the inline branch's behavior bit-for-bit;
verified by `python  --mode delta`
at the atomic commit gate.

IUPAC cite: P-66.6 (oxime functional class naming).

References:
- composer.py:771-782 (inline dispatch branch; REMOVED at this commit).
- composer.py:1906-2042 (_name_oxime_or_hydrazone body; STAYS until 03-10).
- 160-AUDIT-DECOMP.md § 1 row 'oxime' + § 2.2 predicate purity proof.
- 160-PATTERNS.md § 6 (Tier-B lift handler pattern).
- 160-CONTEXT.md D-21 (atomic-commit byte-identical canary lock).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_oxime(features: Any) -> bool:
    """Mirrors composer.py:771 (``features.principal_group == 'oxime'``).

    Pure read-only per CONTEXT D-25 / AP-160-26: reads
    ``features.principal_group`` attribute set by perception layer; no
    mutation of features, mol, or module-global state.
    """
    return getattr(features, 'principal_group', None) == 'oxime'


def name_oxime(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Phase 160 Tier-B oxime handler.

    Verbatim semantics of composer.py:771-782 (inline branch). Returns
    ``NamingResult(name=<final string>, tree=None, atom_to_locant_hint=None)``
    on success; ``None`` on gate-fail / not-applicable (pool gate
    threshold 0.40 per HANDLER_POLICIES['oxime']).

    Wave-1 strategy per CONTEXT D-24: lazy-import the composer.py body
    (``_name_oxime_or_hydrazone``) and the enrichment helper
    (``_enrich_handler_name``); call them with the same arguments the
    inline branch used; route through the same ``pool.add()`` call so
    Phase 145.1 byte-identical lock methodology is preserved.

    The ``mol`` parameter is accepted for API uniformity per CONTEXT
    D-05 but not used here (composer.py:_name_oxime_or_hydrazone reads
    features.mol directly). The ``style`` parameter is similarly unused
    for first-wave Tier-B handlers (style only affects Pass-2 assembly,
    not Tier-B handlers).

    Args:
        features: MolecularFeatures object.
        mol: RDKit Mol object (not used in first-wave; reserved per D-05).
        style: Naming style (not used in first-wave Tier-B; reserved per D-05).

    Returns:
        NamingResult on success, or None on gate-fail. Per CONTEXT D-05
        the ``tree`` field is None for first-wave handlers; the ``name``
        field is the byte-identical contract per DECOMP-03.
    """
    # Lazy imports per PATTERNS § Lazy Import (avoid composer.py -> handlers
    # -> composer.py cycle at module load).
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _name_oxime_or_hydrazone, _enrich_handler_name,
        _inject_stereo_if_missing,
    )

    oxime_name = _name_oxime_or_hydrazone(features, 'oxime')
    if not oxime_name:
        return None

    oxime_name = _enrich_handler_name(features, oxime_name, "oxime")

    # Phase 145.1: route through pool.add() — returns None on gate-fail.
    pool = get_current_pool()
    cand = pool.add(oxime_name, "oxime", features)
    if cand is None:
        return None

    # Per CONTEXT D-13 layering: this handler's inline branch at
    # composer.py:781 wrapped the name in _inject_stereo_if_missing — we
    # preserve that byte-identical behavior here. The NamingResult.name
    # field is the FINAL name (post-stereo-injection); the dispatch caller
    # returns it directly without further processing per the Plan-02
    # dispatch contract.
    final_name = _inject_stereo_if_missing(
        features, cand.name, atom_to_locant=None,
    )
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="oxime", iupac_section_cite="P-68.3.1.2", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_oxime", "_is_oxime"]
