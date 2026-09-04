"""Phase 160 acid_halide handler — direct-return shim.

1-line wrapper around ``rules.acid_halides.name_acid_halide`` per
CONTEXT + Phase 158 AP-5 (no-logic-in-shim discipline). Verbatim
move of the dispatch logic at composer.py:919-932.

IUPAC cite: P-66.5 (acyl halides; functional class naming
'ethanoyl chloride' / 'benzoyl bromide').

References:
- composer.py:919-932 (inline dispatch branch; REMOVED at this commit).
- rules.acid_halides.name_acid_halide — chemical-logic body (unchanged).
- 160-AUDIT-DECOMP.md § 1 row 'acid_halide' + § 2.11 purity proof.
- 160-PATTERNS.md § 8 (shim handler pattern).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_acid_halide(features: Any) -> bool:
    """Mirrors composer.py:919 (3-class principal_group check).

    Wave2 T6c: extended with the acyl pseudohalides (P-65.5.2.1) — the same
    two-word functional-class grammar via rules.acid_halides.HALIDE_WORDS.
    """
    return getattr(features, 'principal_group', None) in (
        'acid_chloride', 'acid_bromide', 'acid_fluoride',
        'acyl_azide', 'acyl_cyanide', 'acyl_isocyanate',
    )


def name_acid_halide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return acid-halide handler.

    Verbatim lift of composer.py:919-932. Lazy import keeps the
    handlers.acid_halide -> rules.acid_halides chain off the module-
    import-time graph (Phase 158 + PATTERNS § Lazy Import).
    """
    from ...rules.acid_halides import name_acid_halide as _name_acid_halide
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    # C1: the producer records which parent it numbered the name in; the
    # stereo injector below must not re-infer it (measured: both
    # features.principal_chain and features.chain_is_parent report "chain" on
    # ring-parented names -- see composer._inject_stereo_if_missing).
    _scope_out: dict = {}
    halide_name = _name_acid_halide(features, scope_out=_scope_out)
    if not halide_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "acid_halide", _ha, halide_name[:60],
        )

    pool = get_current_pool()
    pool.add(halide_name, "acid_halide", features)
    # composer.py:923 inline: _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
        parent_scope=_scope_out.get('parent_scope'),
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="acid_halide", iupac_section_cite="P-65.5", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_acid_halide", "_is_acid_halide"]
