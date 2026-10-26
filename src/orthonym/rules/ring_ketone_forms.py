"""A ring ketone of a fused substituent group cites its added hydrogen and its hydro prefixes.

A ring atom with a double bond to an atom outside the ring (a ring C=O, C=S, C=N: the 'oxo',
'sulfanylidene' and 'imino' prefixes) has no ring double bond, so the ring is not the bare mancude
parent: the name cites added or indicated hydrogen and hydro prefixes, "'Added indicated
hydrogen'" (the Blue Book) and (:24695), 'pyridin-1(2H)-yl (preferred prefix)'.
'(4-oxoquinazolin-3(4H)-yl)acetic acid' and '3-(4-chlorophenyl)-4-oxo-3,4-dihydroquinazolin-2-yl',
not the catalogue's mancude stems '4-oxoquinazolin-3-yl' and '4-oxoquinazolin-2-yl'. The fusion name of
the ring as drawn is ``rules.fused_forms.fused_system_form``'s; this module routes such a ring to it
(best-effort tier only: the callers are gated on ``allow_mancude``).
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def has_exocyclic_multiple_bond(mol, ring_atoms) -> bool:
    """True when a ring atom carries a double or triple bond to an atom outside the ring."""
    from .monocycle_forms import _exocyclic_multiple
    ring_set = set(ring_atoms)
    return any(_exocyclic_multiple(mol, a, ring_set) for a in ring_set)


def ketone_core_numbering(mol, ring_atoms, attach, deco_carriers) -> Optional[Tuple[Dict[int, int], str]]:
    """``(pos, tail)`` of a fused ring core with a ring ketone, by its fusion name
    (``ring_substituents._fused_form_core_numbering``), else None."""
    if not has_exocyclic_multiple_bond(mol, ring_atoms):
        return None
    from .ring_substituents import _fused_form_core_numbering
    return _fused_form_core_numbering(mol, ring_atoms, attach, deco_carriers)


def decorated_ketone_fused_name(mol, ring_atoms, attach, decorations: List[Tuple[int, str]]) -> Optional[str]:
    """The prefix of a fused ring substituent group that has a ring ketone, from its fusion name:
    the decorations ``[(ring atom, prefix name)]`` alphabetized and multiplied, then the form's tail
    ('4-oxo' + 'quinazolin-3(4H)-yl'). None when the ring has no ketone or the form declines."""
    core = ketone_core_numbering(mol, ring_atoms, attach, [ra for ra, _ in decorations])
    if core is None:
        return None
    from ..assembly.naming_utils import (alpha_sort_key, begins_with_italic_designator,
                                         enclose_if_compound, multiplied_component)
    from .ring_substituents import _DECO_MULT
    pos, tail = core
    groups = defaultdict(list)
    for ra, name in decorations:
        groups[name].append(pos[ra])
    parts = []
    for name, locs in groups.items():
        locs = sorted(locs)
        if _DECO_MULT.get(len(locs)) is None:
            return None
        parts.append((alpha_sort_key(name), f"{','.join(str(loc) for loc in locs)}-"
                      f"{multiplied_component(len(locs), name, enclose_if_compound(name))}"))
    parts.sort(key=lambda part: part[0])
    sep = '-' if (tail[:1].isdigit() or begins_with_italic_designator(tail)) else ''
    return f"{'-'.join(text for _key, text in parts)}{sep}{tail}"
