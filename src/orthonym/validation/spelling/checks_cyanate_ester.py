"""Spelling check: an ester of cyanic, thiocyanic or selenocyanic acid written as a prefix where no
group senior to the esters is present (registered with:func:`orthonym.validation.pin_spelling.register`).

 "Cyanic acid" (the Blue Book): "Cyanic acid is classified as an acid, thus generating anhydrides... and
esters"; (:18182) ranks "9 Esters" above acid halides (10), amides (11), nitriles, aldehydes, ketones,
alcohols and amines. (:32035): "In the presence of a characteristic group having seniority for
citation as suffix, an ester group is indicated by appropriate prefixes" -- 'propan-2-yl thiocyanate (PIN)' (:32033),
'3-(thiocyanato)propanoic acid (PIN)' (:31013), 'S-ethyl 3-(thiocyanato)propanethioate (PIN)' (:32041). So with no
principal group, or one junior to the esters, the ester names the molecule ('2-hydroxyethyl thiocyanate', as
'3-oxobutyl bromate (PIN)',:35974), and a name that does not end in the ester word ('2-(thiocyanato)ethan-1-ol') is
correct but is no PIN. Two or more esters take the multiplied ester word ('ethane-1,2-diyl diacetate (PIN)',:31823),
so a name with the prefix AND the ester word ('2-(thiocyanato)ethyl thiocyanate') is no PIN either.

Not this group: an acyl thiocyanate (R-CO-S-CN, an anhydride), an -OCN on a heteroatom (an anhydride,,
:35690), ions and radicals (classes 1-6, senior to esters). The structure side is a carbon-X-C#N group (X = O, S, Se)
whose carbon carries no double bond to O, S or N; the name side is the end of the name and the prefix words. Reads
perception only, never naming code or OPSIN.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_ESTER = Chem.MolFromSmarts('[#6;!$([#6]=[O,S,N])]-[O,S,Se;X2]-[C;X2]#[N;X1]')
_NITRO = Chem.MolFromSmarts('[N+](=O)[O-]')
# A lactone is a pseudoketone, class 16: "A lactone, as a pseudoketone, ranks lower in the seniority of classes than
# an acid or an ester" "Lactones", the Blue Book); only open esters count below.
#: carbamic and carbonic acids, their esters: carbon acids senior to cyanic acid ("... carbonic acid > cyanic acid",
#:,:30746)
_CARBONIC = Chem.MolFromSmarts('[#7,#8][CX3](=[O,S])[OX2,SX2]')
_ESTER_WORD = re.compile(r'\s\S*cyanate\)?$')
_PREFIX_WORD = re.compile(r'(?<!iso)(?:thio|seleno)?cyanato')


def _ester_count(mol) -> int:
    return len(mol.GetSubstructMatches(_ESTER))


def _charged_or_radical(mol) -> bool:
    """A charge outside a nitro group, or a radical: an ion or radical class senior to the esters."""
    nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}
    return any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in nitro)
               for a in mol.GetAtoms())


#: groups of a class senior to the esters of cyanic acid, or an ester ranked by its acid,:18172-:18182;
#: "... carbonic acid > cyanic acid",:30746): acids (carboxylic, chalcogen, peroxy, and the oxoacids of S, Se, P, As,
#: B, N), anhydrides, open carboxylic and noncarbon oxoacid esters. A lactone (class 16) is not among them. Read from the
#: structure only (the checks import no naming code); anything else leaves the cyanate ester senior.
_SENIOR_TO_CYANATE_ESTERS = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se,N])[OX2H1,SX2H1,SeX2H1]',          # carboxylic, thio, seleno, imidic acids
    '[CX3](=O)O[OX2H1]',                               # peroxy acids
    '[!#6;!#1;!#7](=O)[OX2H1]',                        # oxoacids of S, Se, P, As, B,...
    '[CX3](=[O,S])[OX2,SX2][CX3]=[O,S]',               # anhydrides
    '[#6][CX3](=[O,S,Se])[OX2,SX2,SeX2;!R][#6]',        # open carboxylic and thio esters
    '[!#6;!#1;!#7](=O)[OX2;!R][#6]',                   # esters of noncarbon oxoacids (sulfonate, phosphate,...)
    '[O,S]=[N+]([O-])[OX2][#6]', '[OX1]=[NX2][OX2][#6]',   # nitrates and nitrites
))


def _junior_to_esters(mol) -> bool:
    """No group of a class senior to a cyanic acid ester (``_SENIOR_TO_CYANATE_ESTERS``, or a carbamate or carbonate)."""
    if mol.HasSubstructMatch(_CARBONIC):
        return False
    return not any(mol.HasSubstructMatch(p) for p in _SENIOR_TO_CYANATE_ESTERS if p is not None)


@register('P-65.6.3.3.7.2.2')
def cyanate_ester_prefix_check(mol, name):
    """The name writes a cyanate, thiocyanate or selenocyanate ester as a prefix where the ester outranks the
    principal group, or writes one as a prefix beside the ester word of another."""
    if not name:
        return None
    n = _ester_count(mol)
    if not n or _charged_or_radical(mol):
        return None
    ends_in_ester = bool(_ESTER_WORD.search(name.strip()))
    has_prefix = bool(_PREFIX_WORD.search(name))
    if ends_in_ester:
        if n > 1 and has_prefix:
            return SpellingFailure('P-65.6.3.3.7.2.2', "an ester named beside another as a '(thio)cyanato' prefix; "
                                                       "the esters take one multiplied ester word")
        return None
    if _junior_to_esters(mol):
        return SpellingFailure('P-65.6.3.3.7.2.2', "a (thio)cyanate ester outranks the principal group (P-41 class 9); "
                                                   "the functional-class ester name is the PIN")
    return None
