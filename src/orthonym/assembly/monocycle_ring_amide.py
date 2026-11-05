"""The N-substituents of a ring carboxamide / sulfonamide in the lone-monocycle producer of the general engine.

 (the Blue Book): the substituents R' and R'' of the nitrogen of R-CO-NR'R'' are cited as prefixes
preceded by the locant N. On a ring carboxamide ('pyrrolidine-2-carboxamide') they stand beside the ring prefixes, in
the one alphanumerical order of the detachable prefixes, "Prefixes... are arranged alphanumerically",
:3375): '1-acetyl-N-methylpyrrolidine-2-carboxamide'. The cage path of the engine builds them
(``general_engine._ring_amide_n_prefixes``); ``name_general_monocycle`` held the atoms of an N-substituent out of
nowhere and refused ('partition incomplete'), so a heteromonocycle with an N-substituted carboxamide was named on a
ring without the amide suffix ('1-acetyl-2-(methylcarbamoyl)pyrrolidine') or on another ring of the molecule.
"""
from typing import List, Optional, Tuple, Union

from ..rules.amide_parent import is_lone_ring


def senior_ring_system(mol, ring_systems, features) -> tuple:
    """The ring system of the molecule that ``name_general_monocycle`` takes as its parent monocycle: the senior one by
    (``rules.ring_selection.select_principal_ring_system``), or, where that is a lone ring too, the lone ring that bears the
    principal characteristic group: (the Blue Book) puts the group before the seniority of the rings, so the
    pyrrolidine of 'pyrrolidine-2-carboxamide' is the parent although the thiazole ring of a substituent is the senior
    ring. Only the atom that carries the locant counts as bearing the group (the first atoms of the N-substituents of an
    amide are in its match too, and a ring that holds one of them does not bear it). A senior ring that is fused, spiro or
    bridged is the cage producers' parent and stays the choice: the producer then declines, as it always did."""
    from ..rules.parent_selection import _pg_attachment_atoms
    from ..rules.ring_selection import select_principal_ring_system
    plain = select_principal_ring_system(mol, ring_systems)
    group = getattr(features, 'principal_group', None)
    if not plain or not is_lone_ring(mol, plain) or not group:
        return plain
    anchors = [a for match in getattr(features, 'principal_group_atoms', None) or []
               for a in _pg_attachment_atoms(group, tuple(match))]
    hinted = select_principal_ring_system(mol, ring_systems, anchors)
    return hinted if hinted and is_lone_ring(mol, hinted) else plain


def ring_amide_n_block(mol, suffix_core: Optional[str], pg_matches, ring_set, suffix_atoms: set,
                       allow_mancude: bool) -> Tuple[set, list, list, Optional[str]]:
    """``(atoms, entries, bindings, reason)``: the N-substituents of the ring carboxamide / sulfonamide that the suffix
    ``suffix_core`` stands for, or the ``reason`` that the producer must refuse. Empty for any other suffix. The atoms
    of the N-substituents are taken out of ``suffix_atoms`` in place (they are named as prefixes, not part of the suffix).

    Refused: an N-substituent that cannot be named (an atom is never dropped), and a group whose nitrogen is a ring atom
    (an acyl on a ring nitrogen is no carboxamide suffix: it is a prefix of the ring, 'hidden amide').
    The ring that bears the principal group is the parent, the Blue Book), whatever the seniority of
    the other rings: ``name_general_monocycle`` selects it with the principal group atoms as the hint, as the cage path
    does."""
    if suffix_core not in ('carboxamide', 'sulfonamide'):
        return set(), [], [], None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and i in ring_set for match in pg_matches for i in match):
        return set(), [], [], "ring amide/sulfonamide nitrogen is a ring atom"
    from .general_engine import _ring_amide_n_prefixes
    found = _ring_amide_n_prefixes(mol, pg_matches, allow_mancude)
    if found is None:
        return set(), [], [], "ring amide/sulfonamide N-substituent unnameable"
    suffix_atoms -= found[0]
    return found + (None,)


def prefix_parts(groups: dict, n_entries: List, omit_ring_locant: bool) -> Optional[List[str]]:
    """The detachable prefixes of the ring in one order: the ring substituents (``groups``: name -> locants)
    and the N-substituents (``n_entries``: (key, token)), by the citation key of the substituent name. None when a
    multiplicity is beyond the table of multiplying prefixes."""
    from .general_engine import _citation_key, _mult_prefix
    from ..data.locant_text import render_locants
    entries = list(n_entries)
    for prefix, locants in groups.items():
        text = _mult_prefix(len(locants), prefix)
        if text is None:
            return None
        entries.append((_citation_key(prefix), text if omit_ring_locant else render_locants(sorted(locants)) + '-' + text))
    return [token for _, token in sorted(entries, key=lambda e: e[0])]
