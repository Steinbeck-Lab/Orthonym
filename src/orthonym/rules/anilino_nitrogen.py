"""The anilino prefix of an amino group whose nitrogen carries a benzene ring (PB-W).

``rules.ring_substituents.anilino_prefix_from_n_branch`` derives the preferred prefix of a
C6H5-N< group from the GRAPH. The sites that compose an amino prefix from the names of the groups
on the nitrogen ('(2,6-dichlorophenyl)amino', 'methyl(phenyl)amino') call this module first, so
that the one rule has one home and a site cannot cite the general-nomenclature form as a
preferred prefix.

 "Retained names" (the Blue Book): "The prefix name 'anilino' is retained as
the preferred prefix for C6H5-NH– with full substitution allowed. The name 'phenylamino' may be
used in general nomenclature." Its examples (:26153) '4-chloroanilino (preferred prefix)
(4-chlorophenyl)amino'. (:26298), the nitrogen substituted as well: ':26308'
'3-(N-methylanilino)phenol (PIN)' against '3-[methyl(phenyl)amino]phenol'.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple


def anilino_for_branches(mol, n_idx: int,
                         branches: Sequence[Tuple[Sequence[int], str]],
                         *, enclose: bool = False) -> Optional[str]:
    """The preferred anilino prefix of the nitrogen ``n_idx``, or ``None``.

    ``branches`` holds one ``(atoms, name)`` pair per group on the nitrogen (the atoms of the
    group, the prefix name the caller gave it). One group: a benzene ring with its own
    decoration ('anilino', '2,6-dichloroanilino'). Two groups: one of them a benzene ring, the
    other any prefix ('N-methylanilino', 'N-phenylanilino'); the ring decoration and the
    N-substituent are cited in one alphanumerical sequence ('4-chloro-N-methylanilino',
    "full substitution allowed"). A substituted benzene ring on the other group declines: the more
    substituted ring is the aniline,:21604). ``None`` leaves the composition to the
    caller, so a shape this does not decide is never changed.
    """
    from .ring_substituents import anilino_prefix_from_n_branch
    if len(branches) == 1:
        return anilino_prefix_from_n_branch(mol, n_idx, branches[0][0], enclose=enclose)
    if len(branches) != 2:
        return None
    for ring, other in ((branches[0], branches[1]), (branches[1], branches[0])):
        found = anilino_prefix_from_n_branch(
            mol, n_idx, ring[0], enclose=enclose, n_substituent=other[1])
        if found is not None:
            return found
    return None
