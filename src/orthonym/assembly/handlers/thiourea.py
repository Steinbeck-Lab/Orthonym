"""Chalcogen-urea handler — retained parent thiourea/selenourea/tellurourea.

Residue R3. Mirrors ``handlers/urea.py`` exactly; the body lives in
``composer._try_name_thiourea`` for the same reason urea's does.

IUPAC cite: P-66.1.6.1.3 "Chalcogen analogues of urea and isourea"
(``BlueBookV2/BlueBookV2.md:33437``), subsection P-66.1.6.1.3.1 (``:33439``):
*"Chalcogen analogues of urea are named by functional replacement nomenclature
using the prefixes 'thio', 'seleno', and 'telluro'. Preferred IUPAC names use
the letter locants N, and N'."*  Worked ``(PIN)`` examples: ``thiourea (PIN)``
(``:33444``) and ``N-(butan-2-yl)selenourea (PIN)`` (``:33451``).

Why the parent is the thiourea and not the ring the molecule also carries:
P-66.1.6.1.1.2 (``:33320``) ranks urea *"as an amide of carbonic acid"*; P-41
Table 4.1 puts amides at class 11 (``:18184``) against carbon rings at class 40
(``:18216``); P-44.1.1 (``:18875``) selects on that order; and the
"ring outranks chain" licence is gated on *"Within the same heteroatom class"*
(P-52.2.8, ``:24096``), which a class-40 ring does not share with an amide.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_thiourea(features: Any) -> bool:
    """Chalcogen-urea FG present AND no principal characteristic group.

    Mirrors ``_is_urea``. The ``principal_group is None`` clause is what keeps
    this handler off acid-bearing molecules such as
    ``CNC(=S)NCCC(=O)O`` — there the carboxylic acid is senior (P-41 Table 4.1
    class 5 vs the amide class 11), so the thiourea is demoted to the
    ``(methylcarbamothioyl)amino`` PREFIX built by
    ``substituent_prefix_forms.get_n_substituted_carbamothioylamino_prefix``.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return (bool(fg.get('thiourea'))
            and getattr(features, 'principal_group', None) is None)


def name_thiourea(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B chalcogen-urea retained-name handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _try_name_thiourea, _enrich_handler_name, _inject_stereo_if_missing,
    )

    name = _try_name_thiourea(features)
    if not name:
        return None

    name = _enrich_handler_name(features, name, "thiourea")

    pool = get_current_pool()
    cand = pool.add(name, "thiourea", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name,
            class_id="thiourea",
            iupac_section_cite="P-66.1.6.1.3",
            fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_thiourea", "_is_thiourea"]
