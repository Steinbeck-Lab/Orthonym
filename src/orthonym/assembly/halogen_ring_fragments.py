"""The best-effort floor for a branch whose fragment holds a ring halogen, in the order of its two writers.

``substituent_enumerator.name_substituent`` offers a branch the cascade declined first to the universal floor
(``universal_substituent``) and then to the general engine's route (``_route_fragment_to_general_engine``,
which names the fragment as a molecule with a hydroxy group on the attachment atom and reads the '-yl' off the
'-ol'). For a fragment with a ring halogen (``lambda_convention.FUSED_RING_LAMBDA_HALOGENS``) the order is
reversed: the route builds the Hantzsch-Widman name of the ring ('1lambda3-iodinan-4-yl',,
the Blue Book "Mancude and saturated heteromonocyclic compounds with up to and including ten ring members
are named by the extended Hantzsch-Widman system";:9158 the lambda form), the book's form, where the
floor has only the 'a' replacement spelling ('1lambda3-iodacyclohexan-4-yl'). Where the route has no name the floor
goes on with that 'a' spelling, which is a name of a ring of ten members or fewer that is not the one gives:
a free valence on the halogen or on another ring heteroatom has no '-ol' of a Hantzsch-Widman ring, a ring ion has no
cumulative '-ol' and 'ide' or 'ylium' in the route, and ``monocycle_forms`` (the book builder of the floor's ring
spine) spells no lambda ring. Those fragments are therefore best-effort 'a' names, none a PIN; the Hantzsch-Widman
spelling of them is open. The order of the two writers changed none of the Hantzsch-Widman names main already gave.
"""
from typing import Callable, Optional


def ring_halogen_fragment(mol, frag_atoms_set) -> bool:
    """The fragment holds a ring atom of ``lambda_convention.FUSED_RING_LAMBDA_HALOGENS``."""
    from ..rules.lambda_convention import FUSED_RING_LAMBDA_HALOGENS
    return any(mol.GetAtomWithIdx(a).GetSymbol() in FUSED_RING_LAMBDA_HALOGENS
               and mol.GetAtomWithIdx(a).IsInRing() for a in frag_atoms_set)


def name_fragment_floor(mol, frag_atoms, frag_atoms_set, attach_idx, free_valence,
                        route: Callable) -> Optional[str]:
    """The universal floor's prefix for the fragment, after the general engine's ``route`` for a fragment with a
    ring halogen and a single free valence; ``None`` when neither names it."""
    if free_valence == 1 and ring_halogen_fragment(mol, frag_atoms_set):
        routed = route(mol, frag_atoms, attach_idx)
        if routed and ' ' not in routed:
            return routed
    from .universal_substituent import name_universal_substituent_prefix
    return name_universal_substituent_prefix(
        mol, frag_atoms_set, attach_idx, bond_order=free_valence, cap_attachment=True)
