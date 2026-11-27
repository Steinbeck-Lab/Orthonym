"""Spelling check: the head locants of a multiplicative name are locants of the multiplied parent unit
(registered with:func:`orthonym.validation.pin_spelling.register`).

 "Multiplicative name formation" (the Blue Book), item (a): "the locants for the positions of substitution
of the linking multiplicative substituent atom or group to the identical parent structural unit"; '2,2'-{oxybis[(1E)-
ethene-2,1-diyl-4,1-phenylene]}dipropanoic acid (PIN)' (:50799): a ring between the link and the parent belongs to the
linker. So '4,4'-oxybis(1-phenylethan-1-one)', whose 4,4' are positions of the phenyl prefix that a two-carbon parent
does not have (PIN '1,1'-[oxydi(4,1-phenylene)]di(ethan-1-one)'), is no PIN.

Read only when the unit's parent is an acyclic chain the name spells ('ethan-1-one', 'propanoic acid', 'acetamide',
'formamide'); a ring parent, a nitrogen head ('N,N'-'), or a parent the reader does not know is not judged. Reads the
name only.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register
from .units import STEM_RE, STEMS

_HEAD = re.compile(r"^((?:\d+'*,)+\d+'*)-")
_RETAINED = (('form', 1), ('acet', 2))
_RETAINED_END = re.compile(r"(form|acet)(?:amide|ic|aldehyde|onitrile|ate|ohydrazide|imidamide)$")
_CHAIN = re.compile(r"(" + STEM_RE + r")(?=an(?:e\b|-\d|oic|amide|al\b|enitrile|ethioamide|oyl|edioic|ediamide)"
                    r"|a?-\d+(?:,\d+)*-(?:di|tri)?(?:en|yn))")
#: what may follow the chain stem in a chain parent: unsaturation endings, multipliers and suffixes
_TAIL_WORDS = {'a', 'an', 'ane', 'en', 'ene', 'yn', 'yne', 'dien', 'diene', 'trien', 'triene', 'diyn', 'diyne',
               'di', 'tri', 'tetra', 'one', 'ol', 'al', 'amine', 'imine', 'thiol', 'nitrile', 'amide', 'thione',
               'thial', 'thioamide', 'oic', 'oyl', 'anoic', 'anamide', 'anal', 'anenitrile', 'anethioamide',
               'anedioic', 'anediamide', 'anoyl', 'dione', 'diol', 'dial', 'diamine', 'dithiol', 'dinitrile',
               'diamide', 'dioic', 'edioic', 'ediamide', 'acid', 'peroxol', 'hydrazide', 'imidamide', 'enoic',
               'ynoic', 'dienoic', 'trienoic', 'enamide', 'ynamide', 'enal', 'ynal', 'enenitrile', 'ynenitrile',
               'anehydrazide', 'animidamide'}


def _chain_length(unit):
    """Length of the acyclic chain parent the unit spells, or None (a ring or an unread parent)."""
    u = unit.strip()
    if u.endswith(' acid'):
        u = u[:-5]
    m = _RETAINED_END.search(u)
    if m:
        return dict(_RETAINED)[m.group(1)]
    hits = list(_CHAIN.finditer(u))
    if not hits:
        return None
    last = hits[-1]
    if last.group(1).endswith('cont') and last.start() and u[last.start() - 1].isalpha():
        return None                     # a longer stem than the table holds ('hentetracont', 41)
    if not set(re.findall(r"[a-z]+", u[last.end():])) <= _TAIL_WORDS:
        return None
    return STEMS[last.group(1)]


def _unit(name):
    """The text of the multiplied unit, '(...)' at the end after 'bis', 'di', 'tris', 'tri' or 'tetrakis', else None."""
    if not name.endswith((')', ']')):
        return None
    close, open_ = name[-1], '(' if name[-1] == ')' else '['
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        depth += {close: 1, open_: -1}.get(name[i], 0)
        if depth == 0:
            break
    else:
        return None
    if not re.search(r"(?:bis|di|tris|tri|tetrakis)$", name[:i]):
        return None
    return name[i + 1:-1]


@register('P-15.3.1.3')
def multiplied_unit_locant_check(mol, name):
    """A head locant of a multiplicative name is higher than the length of the chain parent it should number."""
    if not name or ' ' in name.replace(' acid)', ')').replace(' acid]', ']'):
        return None
    n = name.strip()
    head = _HEAD.match(n)
    unit = _unit(n) if head else None
    if unit is None:
        return None
    length = _chain_length(unit)
    if length is None:
        return None
    top = max(int(x.rstrip("'")) for x in head.group(1).split(','))
    if top <= length:
        return None
    return SpellingFailure('P-15.3.1.3', f"head locant {top} of a multiplicative name is not a locant of the "
                                         f"{length}-carbon parent unit: a ring between the link and the parent is "
                                         f"part of the linker ('1,1'-[oxydi(4,1-phenylene)]di(ethan-1-one)')")
