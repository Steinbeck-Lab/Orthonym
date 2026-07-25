"""P-67 organo-oxoacid handlers for the heavier pnictogens (As, Sb).

Exact analogues of ``handlers.phosphonic_acid`` / ``handlers.phosphinic_acid``.
The Blue Book gives all four as PRESELECTED names (BB L36051-36054)::

    AsH(O)(OH)2  arsonic acid    AsH2(O)OH  arsinic acid
    SbH(O)(OH)2  stibonic acid   SbH2(O)OH  stibinic acid

and the substituent-prefix PIN forms verbatim -- ``ethylstibinic acid``
(BB L36064), ``methyl(phenyl)arsinic acid`` (BB L36066),
``(4-acetamido-3-methylphenyl)arsonic acid`` (BB L33010).

An arsonic acid is NOT an organometallic: P-69 never applies to it. Without
these handlers an organyl As/Sb oxoacid perceives no principal group at all,
falls through the whole cascade, and is reported by the descriptive fallback --
which is why ``C[As](=O)(O)O`` used to come back as "inorganic compound".

The four handlers are generated from one factory rather than copied four times,
so the arsenic and antimony paths cannot drift apart. Each delegates to the
element-generic namer in ``rules.phosphorus`` and fail-closes (returns ``None``,
cascade-continuation) for a substituent the simple namer cannot prove.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

# functional-group key -> (rules.phosphorus function name, IUPAC cite)
_PNICTOGEN_OXOACIDS = {
    "arsonic_acid": ("name_arsonic_acid", "P-67.1.1.2"),
    "arsinic_acid": ("name_arsinic_acid", "P-67.1.1.2"),
    "stibonic_acid": ("name_stibonic_acid", "P-67.1.1.2"),
    "stibinic_acid": ("name_stibinic_acid", "P-67.1.1.2"),
}


def _make_pnictogen_oxoacid_handlers(fg_name: str) -> Tuple[Callable, Callable]:
    """Build the (predicate, handler) pair for one P-67 pnictogen oxoacid."""
    rules_fn_name, cite = _PNICTOGEN_OXOACIDS[fg_name]

    def _predicate(features: Any) -> bool:
        """Mirrors the phosphonic predicate (``principal_group == <fg>``)."""
        return getattr(features, "principal_group", None) == fg_name

    def _handler(
        features: Any, mol: Any = None, style: str = "pin",
    ) -> Optional[NamingResult]:
        from ..candidate_pool import get_current_pool
        from ..composer import _inject_stereo_if_missing
        from ...rules import phosphorus as _phosphorus

        matches = features.functional_groups.get(fg_name, [])
        if not matches:
            return None

        name = getattr(_phosphorus, rules_fn_name)(features.mol, matches[0])
        if not name:
            # complex organyl -> defer to the generic path (cascade-continuation)
            return None

        pool = get_current_pool()
        pool.add(name, fg_name, features)
        final_name = _inject_stereo_if_missing(features, pool.best().name)
        return NamingResult(
            name=final_name,
            tree=NameTreeNode(
                parent_stem=final_name,
                class_id=fg_name,
                iupac_section_cite=cite,
                fragment_legacy=final_name,
            ),
            atom_to_locant_hint=None,
        )

    _predicate.__name__ = f"_is_{fg_name}"
    _predicate.__qualname__ = _predicate.__name__
    _handler.__name__ = f"name_{fg_name}"
    _handler.__qualname__ = _handler.__name__
    return _predicate, _handler


_is_arsonic_acid, name_arsonic_acid = _make_pnictogen_oxoacid_handlers("arsonic_acid")
_is_arsinic_acid, name_arsinic_acid = _make_pnictogen_oxoacid_handlers("arsinic_acid")
_is_stibonic_acid, name_stibonic_acid = _make_pnictogen_oxoacid_handlers("stibonic_acid")
_is_stibinic_acid, name_stibinic_acid = _make_pnictogen_oxoacid_handlers("stibinic_acid")


__all__ = [
    "_is_arsonic_acid", "name_arsonic_acid",
    "_is_arsinic_acid", "name_arsinic_acid",
    "_is_stibonic_acid", "name_stibonic_acid",
    "_is_stibinic_acid", "name_stibinic_acid",
]
