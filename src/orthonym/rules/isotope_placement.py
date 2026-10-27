"""The order of the placements of a nuclide descriptor that read back to the same molecule.

Kept apart from ``isotopes`` (a module at its size budget); ``_find_best_placement`` there sorts its placements with
:func:`placement_order`.
"""
from __future__ import annotations

import re

from rdkit import Chem


def labelled_group_locant(candidate: str, off: int) -> int:
    """The locant of the substituent group that holds the descriptor spliced at ``off`` of ``candidate``, else 99.

     (the Blue Book), "The preferred IUPAC name is based on the senior parent structure that has the
    lowest locant(s) for isotopically modified substituent groups": '2-bromo-1-{[2-(81Br)bromopentyl]oxy}pentane
    (PIN) [not 2-(81Br)bromo-1-[(2-bromopentyl)oxy]pentane]', "attached at '1' which is lower than '2'". Where two
    placements read back to the same molecule (the constitution without the nuclide is symmetric), the group is the
    outermost bracketed group that holds the descriptor, or the simple prefix it stands before, and its locant is the
    number in front of it."""
    opened, closing = [], {')': '(', ']': '[', '}': '{'}
    for i, ch in enumerate(candidate[:off]):
        if ch in '([{':
            opened.append((ch, i))
        elif ch in closing and opened and opened[-1][0] == closing[ch]:
            opened.pop()
    start = opened[0][1] if opened else off
    m = re.search(r"(\d+)[a-z]?'*-?$", candidate[:start])
    return int(m.group(1)) if m else 99


def nuclide_has_equivalent_site(original: Chem.Mol) -> bool:
    """True when a labelled atom of ``original`` has an equivalent atom without the labels (the constitution
    without the nuclides is symmetric at it): the case decides, two placements of one nuclide that read back."""
    plain = Chem.Mol(original)
    for atom in plain.GetAtoms():
        atom.SetIsotope(0)
    try:
        Chem.GetSymmSSSR(plain)
        ranks = list(Chem.CanonicalRankAtoms(plain, breakTies=False))
    except (RuntimeError, ValueError):
        return False
    return any(ranks.count(ranks[a.GetIdx()]) > 1 for a in original.GetAtoms() if a.GetIsotope())


def placement_order(w, equivalent_site: bool = False) -> tuple:
    """The order of the placements of ``_find_best_placement`` (``w`` = p4542, locant, subscript form, placement
    quality, offset, descriptor, candidate); the locant of the modified group (:func:`labelled_group_locant`) counts
    where the nuclide has an equivalent site."""
    return (w[0], w[1], w[2], w[3], labelled_group_locant(w[6], w[4]) if equivalent_site else 0, w[4])
