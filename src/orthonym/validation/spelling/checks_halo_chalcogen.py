"""Spelling check: a halogen on a λ4/λ6 chalcogen written as a substituent prefix
('pentafluoro-λ6-sulfanyl', 'trichloro-λ4-sulfanyl') is a name the Blue Book prints as no PIN
(registered with:func:`orthonym.validation.pin_spelling.register`).

 "Halogen atoms attached to heteroatoms": the halo prefixes are for halogens on "B, Al, In, Ga,
Tl, Si, Ge, Sn, Pb, and Bi", the Blue Book); (:25876): "acid halides or
esters of inorganic acids may be created when the halogen atoms are linked to phosphorus or chalcogen
atoms. Names must be based on the seniority of classes". "Sulfanes, selanes, and tellanes"
(:39527) names such compounds "on the basis of parent hydrides such as λ4-sulfane, λ6-sulfane, and
λ4-selane" ('tetramethoxy-λ4-sulfane (PIN)', not '[(trimethoxy-λ4-sulfanyl)oxy]methane'). The book prints
no PIN with a halo-λ-sulfanyl, -selanyl or -tellanyl prefix, so a name that writes one is labelled below
the PIN; the name is kept. The parent spelling ('pentafluoro(methyl)-λ6-sulfane') is not checked: the
book has neutral halo-λ parents as PINs ('1-hydroxy-1,1-diiodo-1λ4-disulfane-2-carbonitrile (PIN)',
:36995). Charged centres ('tetrafluoro(methyl)-λ4-sulfanium (PIN)',:41384) and sulfinyl/sulfonyl
halides (a double bond on the chalcogen) are not this group. The name side is the prefix word, the
structure side a neutral, singly bonded S, Se or Te with more than two bonds and a halogen neighbour.
No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_HALOGENS = frozenset({'F', 'Cl', 'Br', 'I'})
_LAMBDA_PREFIX = re.compile(r'λ\d+-(?:sulfan|selan|tellan)(?:ediyl|ylidene|yl)')
_HALO_WORD = re.compile(r'(?:fluoro|chloro|bromo|iodo)')
_OPEN, _CLOSE = '([{', ')]}'


def _has_halo_lambda_chalcogen(mol) -> bool:
    """A neutral S, Se or Te with single bonds only, more than two bonds (a nonstandard bonding number) and a
    halogen neighbour."""
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() in ('S', 'Se', 'Te') and atom.GetFormalCharge() == 0
                and atom.GetDegree() + atom.GetTotalNumHs() > 2
                and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
                and any(n.GetSymbol() in _HALOGENS for n in atom.GetNeighbors())):
            return True
    return False


def _prefix_text(name: str, end: int) -> str:
    """The text of the substituent prefix that ends at ``end``, at its own bracket depth: from the opening mark
    that encloses it (or the start of the name, or a space) up to ``end``, with every enclosed part left out."""
    depth, out = 0, []
    for i in range(end - 1, -1, -1):
        ch = name[i]
        if ch in _CLOSE:
            depth += 1
        elif ch in _OPEN:
            if depth == 0:
                break
            depth -= 1
        elif ch == ' ' and depth == 0:
            break
        elif depth == 0:
            out.append(ch)
    return ''.join(reversed(out))


@register('P-61.3.2.2')
def halo_lambda_chalcogenyl_prefix_check(mol, name):
    """P-61.3.2.2 (:25876) / P-68.4.3.1 (:39527): the name writes a halogen-bearing λ4/λ6 chalcogen as a
    halo-λ-chalcogenyl prefix, for which the book prints no PIN."""
    if not name or 'λ' not in name or not _has_halo_lambda_chalcogen(mol):
        return None
    for m in _LAMBDA_PREFIX.finditer(name):
        word = _prefix_text(name, m.start())
        if _HALO_WORD.search(word):
            return SpellingFailure('P-61.3.2.2', f"'{word}{m.group(0)}': a halogen on a λ chalcogen written as a "
                                                 "substituent prefix; the book prints no PIN for it")
    return None
