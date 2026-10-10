"""a phase guanidine handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:892-901 (inline branch) +
composer.py:2760-2906 (_try_name_guanidine body). Per internal notes,
body stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (guanidine and its derivatives; retained name with
N-substitution).

References:
- composer.py:892-901 (inline dispatch branch; REMOVED at this commit).
- composer.py:2760-2906 (_try_name_guanidine body).
- internal notes-DECOMP.md row 'guanidine' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_guanidine(features: Any) -> bool:
    """Mirrors composer.py:892 (guanidine FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('guanidine')) and getattr(features, 'principal_group', None) is None


def name_guanidine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B guanidine handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _try_name_guanidine,
    )

    guanidine_name = _try_name_guanidine(features)
    if not guanidine_name:
        return None

    guanidine_name = _enrich_handler_name(features, guanidine_name, "guanidine")

    pool = get_current_pool()
    cand = pool.add(guanidine_name, "guanidine", features)
    if cand is None:
        return None

    # Guanidine, like urea, has no numbered skeleton: its only locants are the
    # italic letters N, N', N'', the Blue Book "Guanidine and its
    # derivatives": "the locants N, N' and N'' are used in preferred IUPAC names.
    # The locants 1, 2, and 3 have been used but are no longer recommended").
    # A front-of-name block with NUMERIC locants therefore resolves to nothing in
    # the parent "Citation of locants", the Blue Book); the stereo of the
    # N-substituents is cited inside their own prefixes "NAMING OF
    # STEREOISOMERS", the Blue Book), which the substituent namer does. Without the
    # declared scope the ring and chain readings both agreed and the block was
    # prepended: '(1E)-N,N'-di[(1E)-prop-1-en-1-yl]guanidine', which OPSIN rejects.
    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="guanidine", iupac_section_cite="P-66.4.1.2.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_guanidine", "_is_guanidine"]
