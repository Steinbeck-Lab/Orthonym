"""The 'spirobi' name of a monospiro ring system of two identical polycyclic components.

 (the Blue Book) "Monospiro ring systems consisting of two identical polycyclic ring
components are named by placing the nondetachable prefix 'spirobi' before the name of the component
ring system enclosed in square brackets. The established numbering system of the polycyclic ring system
component is retained with one system having primed locants. The location of the spiro atom is
indicated in the name by the appropriate locants (unprimed first) placed at the front of the name."
 (:10152): indicated hydrogen still to cite goes in front of the spiro atom locants, which the
separable form ``spiro[A-a,b'-B]`` the callers keep for that case does not build either.
"""
import re
from typing import Optional

_INDICATED_HYDROGEN = re.compile(r"^\d+[A-Za-z]*H-")


def monospiro_name(first_name: str, first_loc, second_name: str, second_loc, strip_indicated_h) -> str:
    """The name of the monospiro system of two components joined at ``first_loc`` and ``second_loc``
    (the second primed): ``9,9'-spirobi[fluorene]`` for two identical components joined at the same
    locant with no indicated hydrogen left to cite (the indicated hydrogen the spiro atom takes over
    is dropped by ``strip_indicated_h``), the separable ``spiro[A-a,b'-B]`` otherwise."""
    first_name = strip_indicated_h(first_name, first_loc)
    second_name = strip_indicated_h(second_name, second_loc)
    if first_name == second_name and first_loc == second_loc and not _INDICATED_HYDROGEN.match(first_name):
        return f"{first_loc},{second_loc}'-spirobi[{first_name}]"
    return f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"
