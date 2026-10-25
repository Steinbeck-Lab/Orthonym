"""The locant of the free valence of a retained fused hydrocarbon substituent.

 "General methodology for naming substituent groups" gives the free valence of a ring the lowest
locant its numbering allows ('naphthalen-2-yl', never the equivalent '-6-yl'; the Blue Book heading,:15802). The
numbering of a retained fused hydrocarbon is fixed, so the locant is read off the tabulated
numbering of ``POLYCYCLIC_DATA`` over every automorphism of the ring system
(``polycyclics.get_polycyclic_iupac_locants``) and not off the alpha/beta heuristic of
``polycyclics._map_pah_atoms_to_iupac``, which numbers the ring positions of naphthalene only.
"""
from typing import Optional, Tuple


def pah_attachment_locant(mol, ring_atoms: Tuple[int, ...], attachment_atom: int,
                          pah_name: str) -> Optional[int]:
    """The lowest plain locant of ``attachment_atom`` in the retained PAH ``pah_name`` that is the
    ring system ``ring_atoms`` of ``mol``, or None when the numbering is not tabulated or the atom
    takes a fusion locant."""
    from .polycyclics import get_polycyclic_iupac_locants
    from .ring_substituents import _extract_ring_submol
    sub, attach = _extract_ring_submol(mol, ring_atoms, attachment_atom)
    if sub is None:
        return None
    numbering = get_polycyclic_iupac_locants(sub, pah_name, substituent_atoms={attach},
                                             pcg_atoms={attach})
    loc = numbering.get(attach) if numbering else None
    return loc if isinstance(loc, int) else None
