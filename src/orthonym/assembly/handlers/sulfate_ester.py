"""v50 B2 sulfate_ester handler — direct-return shim.

1-line wrapper around ``rules.sulfur_oxoacid.name_sulfate_ester``. Handles the
two sulfuric-acid ester variants (di / mono) via principal_group membership,
mirroring ``handlers/phosphate_ester.py`` for the phosphorus analogue.

IUPAC cite: (esters of mononuclear noncarbon oxoacids;
``the Blue Book Blue Book``; PIN example ``methyl hydrogen sulfate``:35968).

References:
- handlers/phosphate_ester.py — the phosphorus template.
- rules.sulfur_oxoacid.name_sulfate_ester — chemical-logic body.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

_SULFATE_ESTER_GROUPS = ("sulfate_diester", "sulfate_monoester")


def _is_sulfate_ester(features: Any) -> bool:
    """Mirrors phosphate_ester's predicate (principal_group in the sulfate set)."""
    return getattr(features, "principal_group", None) in _SULFATE_ESTER_GROUPS


def name_sulfate_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return sulfate-ester handler."""
    from ...rules.sulfur_oxoacid import name_sulfate_ester as _name_sulfate_ester
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    fg_key = features.principal_group
    matches = features.functional_groups.get(fg_key, [])
    if not matches:
        return None

    # The sulfur atom is the first atom of both sulfate-ester SMARTS matches.
    for idx in matches[0]:
        atom = features.mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "S":
            name = _name_sulfate_ester(features.mol, idx)
            if name:
                pool = get_current_pool()
                pool.add(name, "sulfate_ester", features)
                final_name = _inject_stereo_if_missing(features, pool.best().name)
                return NamingResult(
                    name=final_name,
                    tree=NameTreeNode(
                        parent_stem=final_name, class_id="sulfate_ester",
                        iupac_section_cite="P-67.1.3.2", fragment_legacy=final_name,
                    ),
                    atom_to_locant_hint=None,
                )
            break

    return None


__all__ = ["name_sulfate_ester", "_is_sulfate_ester"]
