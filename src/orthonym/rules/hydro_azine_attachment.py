"""A free valence on the ring nitrogen of a hydro-azine is cited with added hydrogen.

 "'Added indicated hydrogen'" (the Blue Book): the nitrogen of quinoline and its analogues has no hydrogen to
give, so a free valence there is expressed by added hydrogen, '1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one (PIN)' (:29376,
under, not by the 'hydro' prefix of a retained saturated name ('1,2,3,4-tetrahydroquinolin-1-yl'). The producers of
the fusion names write the first form; ``ring_substituents.get_ring_substituent_name`` leaves the retained hydro stem alone
for such an attachment so that they are asked.
"""
import re
from typing import Optional

#: a retained hydro-azine name of the benzo-fused azines the spelling check of knows
_HYDRO_AZINE = re.compile(
    r"(?:\d+[a-z]?(?:,\d+[a-z]?)*-)?(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)hydro"
    r"(?:(?:iso)?quinoline|quinoxaline|quinazoline|phthalazine|cinnoline|acridine|phenanthridine)")


def hydro_azine_bonded_at_nitrogen(mol, retained: str, attachment_point: Optional[int]) -> bool:
    """True when ``retained`` is a hydro-azine name and the free valence is on its ring nitrogen."""
    return (attachment_point is not None and mol.GetAtomWithIdx(attachment_point).GetSymbol() == 'N'
            and _HYDRO_AZINE.fullmatch(retained) is not None)
