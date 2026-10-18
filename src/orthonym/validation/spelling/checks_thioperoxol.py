"""Spelling checks of the thioperoxol names, read lexically (registered with
:func:`orthonym.validation.pin_spelling.register`).

A twin read-back proves the structure of a thioperoxol name, never its spelling
(``validation.thioperoxol_twin``): OPSIN reads 'cyclohexane-1-sulfenic acid' and 'ethane-sulfenate' for the
structures of 'cyclohexane-1-SO-thioperoxol' and 'ethane-SO-thioperoxolate', neither of which is the PIN.

* (the Blue Book) "The locant '1' is omitted:... (c) in monosubstituted homogeneous
  monocyclic rings" (:2913), 'cyclohexanethiol (PIN)': a ring with a thioperoxol suffix and nothing else
  cites no locant, 'cyclohexane-SO-thioperoxol'.
* Anions derived from hydroxy compounds (:41011): 'CH3-CH2-S-O- ethane(SO-thioperoxolate) (PIN)'
  (:41035): the parent hydride that ends without a locant is followed by the enclosed suffix.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register

_MONOCYCLE_LOCANT = re.compile(r"^cyclo[a-z]*ane-1-(?:SO|OS)-thioperoxol(?:ate)?$")
_UNENCLOSED_ANION = re.compile(r"[a-z]ane-(?:SO|OS)-thioperoxolate$")


@register("P-14.3.4")
def monocycle_locant_check(mol, name):
    """A monosubstituted monocycle whose thioperoxol suffix cites the locant 1 (c),:2913)."""
    for word in (name or "").split():
        if _MONOCYCLE_LOCANT.match(word):
            return SpellingFailure("P-14.3.4.2", f"locant '1' cited on the monosubstituted ring in '{word}' "
                                                 f"(thioperoxol suffix, P-14.3.4.2 (c))")
    return None


@register("P-72.2.2.2.2")
def enclosed_anion_suffix_check(mol, name):
    """A thioperoxolate of a parent without a locant that does not enclose its suffix (:41035)."""
    for word in (name or "").split():
        if _UNENCLOSED_ANION.search(word):
            return SpellingFailure("P-72.2.2.2.2", f"'{word}': the anion suffix of a parent without a locant is "
                                                   f"enclosed, 'ethane(SO-thioperoxolate)' (thioperoxolate)")
    return None
