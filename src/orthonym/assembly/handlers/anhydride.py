"""a phase anhydride handler — direct-return shim.

1-line wrapper around ``rules.anhydrides.name_anhydride``. Verbatim move
of composer.py:921-934 dispatch logic.

IUPAC cite: (anhydrides; functional class naming).

References:
- composer.py:921-934 (inline dispatch branch; REMOVED at this commit).
- rules.anhydrides.name_anhydride — chemical-logic body (unchanged).
- internal notes-DECOMP.md row 'anhydride' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_anhydride(features: Any) -> bool:
    """Mirrors composer.py:921 (``features.principal_group == 'anhydride'``)."""
    return getattr(features, 'principal_group', None) == 'anhydride'


def name_anhydride(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return anhydride handler."""
    from ...rules.anhydrides import name_anhydride as _name_anhydride
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    anhydride_name = _name_anhydride(features)
    if not anhydride_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "anhydride", _ha, anhydride_name[:60],
        )

    pool = get_current_pool()
    pool.add(anhydride_name, "anhydride", features)
    # composer.py:943 inline: _inject_stereo_if_missing(features, pool.best.name, atom_to_locant=None)
    # (the Blue Book): the acid groups of an anhydride are cited as separate words, each a name with
    # its own descriptors ('bis[(2S)-2-aminopropanoic] hexanedioic dianhydride'): no front block of the whole name
    #,:44643) is added to it. A mixed anhydride too: the front block put the descriptor of
    # '2-methylbutanoic' on 'acetic' ('(2S)-acetic 2-methylbutanoic anhydride', labelled pin_verified) or cited it
    # twice ('(2S)-methanesulfonic (2S)-2-methylbutanoic anhydride'); a descriptor an acid word lacks is caught by
    # the read-back, never fronted. A one-word name (a cyclic anhydride named as a ring dione,
    # '(3R)-3-methyloxolane-2,5-dione') is the parent's name and keeps the parent's block (review 2026-11-08)
    final_name = pool.best().name
    if " " not in final_name:
        final_name = _inject_stereo_if_missing(features, final_name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="anhydride", iupac_section_cite="P-65.7", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_anhydride", "_is_anhydride"]
