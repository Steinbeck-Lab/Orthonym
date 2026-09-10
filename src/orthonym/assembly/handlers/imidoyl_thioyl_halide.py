""" a phase imidoyl / carbothioyl / carboselenoyl halide handler.

Direct-return shim around ``rules.acid_halides.name_imidoyl_thioyl_halide``
(mirrors the sulfonyl_halide / acid_halide shim discipline). The acid halide of
the imido / chalcogeno analogue of a carboxylic acid is a two-word functional-
class name '{parent}carbo{imidoyl|thioyl|selenoyl} {halide}' (ring parent) or
'{chain-stem}{imidoyl|thioyl|selenoyl} {halide}' (chain parent).

IUPAC cite: P-65.5.1 (BB 31438 'cyclohexanecarboximidoyl chloride (PIN)',
31442 'cyclohexanecarbothioyl chloride (PIN)').
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_imidoyl_thioyl_halide(features: Any) -> bool:
    """True iff the principal group is an imidoyl/carbothioyl/carboselenoyl halide."""
    return getattr(features, 'principal_group', None) in (
        'imidoyl_halide', 'carbothioyl_halide', 'carboselenoyl_halide',
    )


def name_imidoyl_thioyl_halide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return imidoyl/thioyl/selenoyl-halide handler. Lazy import keeps
    the handlers -> rules.acid_halides chain off the import-time graph."""
    from ...rules.acid_halides import (
        name_imidoyl_thioyl_halide as _name_imidoyl_thioyl_halide,
    )
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    halide_name = _name_imidoyl_thioyl_halide(features)
    if not halide_name:
        return None

    pool = get_current_pool()
    pool.add(halide_name, "imidoyl_thioyl_halide", features)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="imidoyl_thioyl_halide",
            iupac_section_cite="P-65.5.1", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_imidoyl_thioyl_halide", "_is_imidoyl_thioyl_halide"]
