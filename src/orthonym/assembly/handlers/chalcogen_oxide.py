"""-6I selenoxide / selenone / telluroxide / tellurone handlers.

Se/Te analogues of the sulfoxide/sulfone handlers (P-63.6, the Blue Book:
"selenium and tellurium... named in the same way"). The chemical-logic body
is the SHARED, element-generic ``rules.sulfur.name_chalcogen_oxide_substitutive``
(diaryl multiplicative / ring+chain / two-chain substitutive PINs) with the
matching functional-class fallback (``dimethyl selenoxide`` style) for shapes
the substitutive builder declines.

BB-verbatim PIN targets:
- the Blue Book ``(ethaneseleninyl)benzene`` (CC[Se](=O)c1ccccc1)
- the Blue Book ``1,1'-selenonyldibenzene`` (O=[Se](=O)c1ccccc1)

References:
- handlers/sulfoxide.py + handlers/sulfone.py — the S sibling shims cloned here.
- rules.sulfur.name_chalcogen_oxide_substitutive — the (now Se/Te-aware) body.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from ..name_tree import NameTreeNode, NamingResult


def _make_chalcogen_oxide_handler(
    fg_key: str, oxide_kind: str, class_id: str, n_oxo: int,
) -> Tuple[Callable[[Any], bool], Callable[..., Optional[NamingResult]]]:
    """Build (predicate, handler) for one Se/Te oxide FG, mirroring the S shims.

    Args:
        fg_key: functional-group key ('selenoxide', 'selenone',...).
        oxide_kind: substitutive prefix stem ('seleninyl'/'selenonyl'/...).
        class_id: handler/pool class id (== fg_key).
        n_oxo: 1 (seleninyl-type, one =O) or 2 (selenonyl-type, two =O), which
            selects the functional-class fallback (name_sulfoxide / name_sulfone,
            both element-generic since -6I).
    """

    def _predicate(features: Any) -> bool:
        # Mirror _is_sulfoxide/_is_sulfone: the FG has no suffix form, so the
        # handler fires when the FG is present and no senior group claimed PCG.
        return (
            getattr(features, 'principal_group', None) is None
            and bool(getattr(features, 'functional_groups', {}).get(fg_key))
        )

    def _handler(
        features: Any, mol: Any = None, style: str = "pin",
    ) -> Optional[NamingResult]:
        from ...rules.sulfur import (
            chalcogen_oxide_fc_covers_molecule,
            name_chalcogen_oxide_substitutive,
            name_sulfone as _name_sulfone,
            name_sulfoxide as _name_sulfoxide,
        )
        from ..candidate_pool import get_current_pool
        from ..composer import _enrich_handler_name, _inject_stereo_if_missing

        matches = features.functional_groups.get(fg_key, [])
        if not matches:
            return None

        # Conservation: both name forms describe EXACTLY R-Se(=O)x-R'. Decline
        # when atoms sit beyond that unit (parallel to the sulfoxide handler).
        if not chalcogen_oxide_fc_covers_molecule(features.mol, matches[0]):
            return None

        # P-63.6: substitutive is the PIN ('(ethaneseleninyl)benzene',
        # "1,1'-selenonyldibenzene"). Functional class ('dimethyl selenoxide')
        # stays for --trivial and as the fail-open fallback for shapes the
        # substitutive builder declines.
        name = None
        if style == "pin":
            name = name_chalcogen_oxide_substitutive(
                features.mol, matches[0], oxide_kind,
            )
        if not name:
            _fc = _name_sulfoxide if n_oxo == 1 else _name_sulfone
            name = _fc(features.mol, matches[0])
        if not name:
            return None

        name = _enrich_handler_name(features, name, class_id)

        pool = get_current_pool()
        cand = pool.add(name, class_id, features)
        if cand is None:
            return None

        final_name = _inject_stereo_if_missing(features, cand.name)
        return NamingResult(
            name=final_name,
            tree=NameTreeNode(
                parent_stem=final_name, class_id=class_id,
                iupac_section_cite="P-63.6", fragment_legacy=final_name,
            ),
            atom_to_locant_hint=None,
        )

    return _predicate, _handler


_is_selenoxide, name_selenoxide = _make_chalcogen_oxide_handler(
    'selenoxide', 'seleninyl', 'selenoxide', 1)
_is_selenone, name_selenone = _make_chalcogen_oxide_handler(
    'selenone', 'selenonyl', 'selenone', 2)
_is_telluroxide, name_telluroxide = _make_chalcogen_oxide_handler(
    'telluroxide', 'tellurinyl', 'telluroxide', 1)
_is_tellurone, name_tellurone = _make_chalcogen_oxide_handler(
    'tellurone', 'telluronyl', 'tellurone', 2)


__all__ = [
    "_is_selenoxide", "name_selenoxide",
    "_is_selenone", "name_selenone",
    "_is_telluroxide", "name_telluroxide",
    "_is_tellurone", "name_tellurone",
]
