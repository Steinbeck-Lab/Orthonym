"""Spelling check (c): the terminal 'a' of a numerical multiplying prefix before a suffix in 'a' or 'o'
(registered with:func:`orthonym.validation.pin_spelling.register`).

 'Elision of vowels' (the Blue Book), "(c) the terminal letter 'a' in the names of numerical multiplicative
prefixes when followed by a suffix beginning with 'a' or 'o'": 'benzenehexol (not benzenehexaol)' (:7619);
"'tetramine', not 'tetraamine'"; 'tetrone'. A name with 'tetraol', 'hexaamine' or 'pentaone' is valid (OPSIN
reads it) but not the PIN. Name side only, no structure.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register

_UNELIDED = re.compile(r"(?:tetra|penta|hexa|hepta|octa|nona|deca|icosa)(?:ol|olate|one|amine|al|oic acid|oate)(?![a-z])")


@register('P-16.7.1')
def multiplier_elision_check(mol, name):
    """ (c): 'tetrol', 'tetramine', 'tetrone', never 'tetraol', 'tetraamine', 'tetraone'."""
    if _UNELIDED.search(name):
        return SpellingFailure('P-16.7.1', "the terminal 'a' of a multiplying prefix is elided before '-ol', "
                                           "'-amine' and '-one' ('benzenehexol', not 'benzenehexaol')")
    return None
