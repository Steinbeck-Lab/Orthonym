"""Spelling check of identical prefixes of one scope (registered with
:func:`orthonym.validation.pin_spelling.register`).

A round trip proves the molecule, never the spelling: OPSIN reads '2-(4-methoxyphenyl)-2-methyl-N-
methylpropanamide' back to the structure of the preferred '2-(4-methoxyphenyl)-N,2-dimethylpropanamide'.
A name that cites one substituent twice in one scope is the general spelling of a name the Blue Book
writes with a multiplying prefix.

* (the Blue Book), under "MULTIPLICATIVE PREFIXES 'DI', 'TRI', ETC. vs. 'BIS',
  'TRIS', ETC." (:7025): "Multiplicative prefixes... are the principal method for describing a
  multiplicity of identical features of a structure in chemical names." The group takes the locants of
  all its members, the italic N cited before the numerals,:3195, "Italic capital and
  lower-case letter locants are lower than Greek letter locants, which, in turn, are lower than
  numerals"): 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (:32879), 'N,N,2-trimethyl-3-{...}
  propanamide (PIN)' (:21624). (:3448): multiplicative prefixes "do not alter the alphabetical
  order already established", so the one group takes the place of its members.

The scope is a unit of the name as:mod:`.units` reads it: a word, or the content of one substituent
enclosure ('',:2869, "the parent structure or... a unit of structure as defined by its
appropriate enclosing marks"). Two prefixes with locants that name the same substituent in one unit
break the rule: a simple prefix by its word ('methyl', and 'tert-butyl' apart from 'butyl'), an
enclosed prefix by its complete text, stereodescriptors included, so '[(1R)-1-hydroxyethyl]' and
'[(1S)-1-hydroxyethyl]' are two substituents. A prefix without a locant is part of a compound word
('ethoxyethoxy') and is not counted. A unit with an isotope descriptor is not read:
(:43718) makes '(13C)methyl' another substituent than 'methyl' ('2-(13C)methyl-3-methylpyridine').
The check reads the name text only; the structure is not used.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from ..pin_spelling import SpellingFailure, register
from .lexer import UnbalancedMarksError
from .units import parse_name


def _identity(prefix) -> Optional[tuple]:
    """What makes two prefixes of one unit the same substituent, or None for one that is not
    cited with a locant (a part of a compound word)."""
    if not prefix.locs:
        return None
    if prefix.kind == "simple":
        return ("simple", prefix.key, prefix.ital)
    if prefix.kind == "enclosed":
        return ("enclosed", prefix.raw)
    return None


@register("P-16.3.1")
def identical_prefixes_check(mol, name) -> Optional[SpellingFailure]:
    """ (:7027): a substituent cited twice in one scope is one multiplied prefix
    ('N,2-dimethyl', '2,3-dihydroxy'), never '2-methyl... N-methyl'."""
    try:
        _s, _encs, parsed = parse_name(name)
    except UnbalancedMarksError:
        return None
    for unit in parsed:
        if "isotope_or_other" in unit.notes:
            continue            # an isotopically modified prefix is another substituent
        # a locanted word followed by an unlocanted one is the first part of a compound word
        # ('5-methoxycarbonyl'): the reader splits it, and it is not the simple prefix 'methoxy'
        seen = Counter(
            _identity(p) for p, nxt in zip(unit.prefixes, unit.prefixes[1:] + [None])
            if nxt is None or nxt.locs)
        for identity, count in seen.items():
            if identity is not None and count > 1:
                return SpellingFailure(
                    "P-16.3.1",
                    f"'{identity[1]}' is cited {count} times in one scope; identical prefixes are one "
                    f"multiplied prefix with all their locants")
    return None
