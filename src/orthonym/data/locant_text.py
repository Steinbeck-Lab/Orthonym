"""The text of a list of locants: ``1,2``, ``N,N'``, ``2,4a,8a``.

IUPAC 2013 "Commas", (a): commas are used to separate locants, numerals, or
italicized letters (``1,2-dibromoethane``, ``N,N-diethylfuran-2-carboxamide``).

``render_locants`` writes the locants in the order it is given, each with ``str``, with a
comma and no space between them. It never sorts: the order of the locants is a decision of
the caller, and a caller that needs ascending order sorts first, at the call.

It is a leaf: it imports nothing from ``orthonym``, so a module of any package may import it
at load time.
"""

from typing import Iterable


def render_locants(locants: Iterable[object]) -> str:
    """The locants joined by commas, in the order given, each written with ``str``."""
    return ",".join(map(str, locants))
