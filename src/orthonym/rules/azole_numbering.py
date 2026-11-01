"""The numbering of an azole whose indicated hydrogen holds a substituent: the heteroatoms fix it first.

``### **** NUMBERING`` (the Blue Book): "(a) fixed numbering in chains, rings, or ring systems" and "(b)
indicated hydrogen for unsubstituted compounds; a higher locant may be needed at another position to accommodate a
substituent suffix". The numbering of a Hantzsch-Widman ring is fixed by its heteroatoms, and the
indicated hydrogen takes the lowest locant among the numberings that keep them. ``****`` (:4916), sentence
:4918: "Type 1 retained names of parent hydrides described in Chapters and have unlimited substitution by
substituent groups cited either as suffixes or prefixes": the substituent that stands in the place of the indicated
hydrogen leaves the parent '4H-1,2,4-triazole' ('4-methyl-4H-1,2,4-triazole').

``heterocycles.orient_heterocycle_with_substituents`` made the pyrrole-type atom the start of the numbering whatever
the locants of the other heteroatoms were: '4-methyl-4H-1,2,4-triazole' was numbered from its N-substituted nitrogen
(heteroatom locants 1,3,4, the methyl cited at 1) and '2-methyl-2H-1,2,3-triazole' as 1,2,5; and the parent name
(``heterocycles._monocycle_indicated_h_prefix``) counted only a nitrogen that still held its hydrogen, so a
substituted one lost the 'H'. The final check withdrew both names. Here every heteroatom of the senior element starts a
candidate, the indicated hydrogen is the second key after the heteroatom locants, and the atom that a substituent
stands in for is the atom that holds it.

Scope: the molecule that is named is neutral and no cation path consumes its numbering. The neutral producer asks for
it (``orient_heterocycle_with_substituents(..., mancude_hydrogen=True)`` from ``namer``'s heterocycle assembly,
``name_heterocycle(..., substituent_holds_h=True)`` from ``composer._assemble_heterocycle_name``); the numbering the
ions read (``ions._ring_iupac_locants``, the parent names of ``azolium_cations``) is the one it always was, because
a cation's 'ium' and its indicated hydrogen compete for the locants, ``### ****``
the Blue Book; '1,4-diphenyl... 1H-1,2,4-triazol-4-ium',:42460; '1H-imidazol-3-ium (PIN)',:41396) and are
placed by the azolium namers, which read the numbering of the bare parent. A ring with a charge keeps it too.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple


def ring_has_charge(mol, ring_atoms) -> bool:
    """True when an atom of the ring carries a formal charge."""
    return any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in ring_atoms)


def numbering_starts(mol, ring_set: Set[int], heteroatoms: Sequence[Tuple[int, str]],
                     neutral_producer: bool) -> List[int]:
    """The ring atoms that may be numbered 1: the heteroatoms of the senior element; for a caller that is not the neutral
    producer, or in a ring with a charge, only those of the best indicated-hydrogen tier (the numbering of the ions)."""
    from ..data.hw_heteroatoms import get_heteroatom_priority
    from .heterocycles import _occupies_indicated_h_position
    best = min(get_heteroatom_priority(sym) for _idx, sym in heteroatoms)
    starts = [idx for idx, sym in heteroatoms if get_heteroatom_priority(sym) == best]
    if not neutral_producer or ring_has_charge(mol, ring_set):
        tier = {i: 0 if _occupies_indicated_h_position(mol, i, ring_set) else 1 for i in starts}
        starts = [i for i in starts if tier[i] == min(tier.values())]
    return starts


def indicated_locants(mol, ring_set: Set[int], heteroatoms: Sequence[Tuple[int, str]],
                      loc_map: Dict[int, int], neutral_producer: bool) -> List[int]:
    """The locants of the ring atoms that hold the indicated hydrogen, for the second key of the numbering; empty for a
    caller that is not the neutral producer and in a ring with a charge."""
    from .heterocycles import _occupies_indicated_h_position
    if not neutral_producer or ring_has_charge(mol, ring_set):
        return []
    return sorted(loc_map[i] for i, _ in heteroatoms if _occupies_indicated_h_position(mol, i, ring_set))


def pyrrole_type_atoms(mol, ring_set: Set[int], oriented: Sequence[int], substituent_holds_h: bool) -> List[int]:
    """The N, P, As and Sb atoms of a mancude ring that hold the indicated hydrogen, or (for the neutral producer) a
    substituent in its place; in a ring with a charge only an atom that still holds a hydrogen."""
    from .heterocycles import _occupies_indicated_h_position
    charged = ring_has_charge(mol, oriented) or not substituent_holds_h
    out = []
    for idx in oriented:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() in ('N', 'P', 'As', 'Sb') and (
                atom.GetTotalNumHs() >= 1 if charged else _occupies_indicated_h_position(mol, idx, ring_set)):
            out.append(idx)
    return out
