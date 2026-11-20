"""Benzene di- and polyamides whose nitrogens carry groups: each group cited with the N-locant of its own nitrogen.

 'N-Substitution', (the Blue Book): "In di- and polyamides, except for geminal
diamides, N locants with superscripted arabic numbers, which are the locants of the parent structure, are used to
differentiate the nitrogen atoms"; 'N1-methylbutanediamide (PIN)', 'N3-ethyl-N1-methylnaphthalene-1,3-disulfonamide
(PIN)'; 'benzene-1,4-dicarboxamide (PIN)' (:32718). Kept apart from ``rules.benzene`` (a module at its size budget),
whose benzene polyamine writer, the same superscript convention) builds the name with the amide suffix.
"""
from typing import Dict, List, Optional

#: the key of the per-locant N-group table of a suffix in ``n_substituents_map``
_BY_LOCANT = "__by_locant__:"
#: the multiplied amide suffixes this writer cites
_AMIDE_SUFFIXES = ("carboxamide", "sulfonamide")


def note_n_substituents(n_substituents_map: Dict, sub_info: Dict, locant: int) -> None:
    """Record the N-groups of a suffix group: under the suffix name (the one-group writers read it) and per locant."""
    n_subs = sub_info.get('n_substituents')
    if n_subs:
        n_substituents_map[sub_info['suffix_name']] = n_subs
        n_substituents_map.setdefault(_BY_LOCANT + sub_info['suffix_name'], {})[locant] = list(n_subs)


def name_benzene_polyamide(chosen_suffix: str, chosen_locants: List[int], prefix_groups: Dict[str, List[int]],
                           n_substituents_map: Dict) -> Optional[str]:
    """'N1-methylbenzene-1,4-dicarboxamide' for two or more ring amides of which at least one nitrogen carries a
    group; None for any other case (one amide: the 'benzamide' writer; no N-group: the plain multiplied suffix)."""
    if chosen_suffix not in _AMIDE_SUFFIXES or len(chosen_locants) < 2:
        return None
    by_locant = n_substituents_map.get(_BY_LOCANT + chosen_suffix) or {}
    if not by_locant:
        return None
    from .benzene import _name_substituted_benzenediamine
    return _name_substituted_benzenediamine(prefix_groups, list(chosen_locants), dict(by_locant), chosen_suffix)
