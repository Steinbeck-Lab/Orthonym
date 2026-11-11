"""The numbering of a catalogued fused ring system that is named as a substituent prefix.

 "NUMBERING" (the Blue Book): "(c) principal characteristic groups and free
valences (suffixes)" (:3256) take low locants before "(f) detachable alphabetized prefixes,
all considered together" (:3301). The catalogue matcher
(``data.fused_heterocycles._select_lowest_locant_match``) chooses among the numberings a
symmetric core allows by the locant set of every atom that carries an outside bond, so the
attachment atom of a prefix counts as one more substituent: 'OC(=O)CCc1ccc2OCC(C)(C)Oc2c1'
was '3-(2,2-dimethyl-2,3-dihydro-1,4-benzodioxin-7-yl)propanoic acid' ({2,2,7} before
{3,3,6}), where the free valence takes 6 first: '3-(3,3-dimethyl-2,3-dihydro-1,4-
benzodioxin-6-yl)propanoic acid'. ``rules.fused_rings._renumber_core_for_suffix`` makes the
same choice for a suffix.
"""
from __future__ import annotations

from typing import Dict, Union


def _key(loc):
    return (loc, '') if isinstance(loc, int) else loc


def renumber_core_for_free_valence(mol, core_name: str, core_smiles: str,
                                   atom_mapping: Dict[int, Union[int, str]],
                                   attach_idx: int) -> Dict[int, Union[int, str]]:
    """The atom -> locant map of the numbering that gives ``attach_idx`` (the free valence)
    the lowest locant; among those the matcher's own criteria decide. The current map is
    kept when it already gives the lowest locant, and every other numbering must keep the
    core's name (``core_name``: the prefix stem is that name's spelling)."""
    from ..data.fused_heterocycles import (
        _coerce_locant_for_compare, core_numberings, select_lowest_locant_match)

    current = _coerce_locant_for_compare(atom_mapping.get(attach_idx))
    if current is None:
        return atom_mapping
    options = []
    for match, mapping, name in core_numberings(mol, core_smiles, atom_mapping):
        loc = _coerce_locant_for_compare(mapping.get(attach_idx))
        if name == core_name and loc is not None:
            options.append((_key(loc), match, mapping))
    if not options:
        return atom_mapping
    low = min(opt[0] for opt in options)
    if _key(current) <= low:
        return atom_mapping
    tied = [opt for opt in options if opt[0] == low]
    chosen = list(select_lowest_locant_match(mol, [opt[1] for opt in tied], core_smiles))
    for _loc, match, mapping in tied:
        if list(match) == chosen:
            return mapping
    return atom_mapping
