"""a phase urea handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:876-885 (inline branch) +
composer.py:2672-2759 (_try_name_urea body). Per internal notes, body
stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (ureas; retained name with N-substitution).

References:
- composer.py:876-885 (inline dispatch branch; REMOVED at this commit).
- composer.py:2672-2759 (_try_name_urea body).
- internal notes-DECOMP.md row 'urea' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_urea(features: Any) -> bool:
    """Mirrors composer.py:876 (urea FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('urea')) and getattr(features, 'principal_group', None) is None


def name_urea(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B urea handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _try_name_semicarbazone,
        _try_name_urea,
    )

    # (W2E-P1FG Task 12): a semicarbazone (R2C=N-NH-CO-NH2) is a urea
    # FG with principal_group None; name it substitutively BEFORE the plain
    # urea path (which would drop the ylidene). Fail-closed -> falls through.
    urea_name = _try_name_semicarbazone(features)
    # `_try_name_urea` builds a COMPLETE name: the retained parent plus every
    # N-substituent it walked itself. Enrichment can therefore only spell an
    # atom a second time -- it turned `N-(1-methylcyclohexyl)urea` into
    # `1-methylN-(1-methylcyclohexyl)urea`, numbering the gem-methyl against a
    # parent (urea) that has no atom 1 at all. Only the semicarbazone branch,
    # which names a different parent, is enriched.
    _complete_name = False
    if not urea_name:
        urea_name = _try_name_urea(features)
        _complete_name = bool(urea_name)
    if not urea_name:
        return None

    if not _complete_name:
        urea_name = _enrich_handler_name(features, urea_name, "urea")

    pool = get_current_pool()
    cand = pool.add(urea_name, "urea", features)
    if cand is None:
        return None

    # The retained urea parent has NO numbered skeleton -- its only locants
    # are the italic letters N / N'. Declare that scope so a
    # numeric front-of-name stereo block, which could not resolve against it,
    # is never prepended. See _inject_stereo_if_missing for the measurement.
    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="urea", iupac_section_cite="P-66.4.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_urea", "_is_urea"]
