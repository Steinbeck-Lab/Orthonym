"""Spelling check: an ester of a carboxylic acid on a ring nitrogen written as an acyloxy prefix
(registered with:func:`orthonym.validation.pin_spelling.register`).

 "Pseudoesters", (the Blue Book): "When 'E', in R-CO-O-E, is an element from the following
list: B, Al, In, Ga, Tl, Si, Ge, Sn, Pb, N(cyclic), P, As, Sb, Bi, the pseudoester is named as a traditional ester, unless
other names must be selected in accordance with the seniority order of classes, in decreasing order: salts > acids >
anhydrides > esters" -- 'borinan-1-yl acetate (PIN)', 'phosphinan-1-yl benzoate (PIN)'. So with no salt, acid,
anhydride or other ester in the molecule, the ester name ('2,5-dioxopyrrolidin-1-yl acetate') is the PIN, and a name on
the ring ('1-(acetyloxy)pyrrolidine-2,5-dione') is correct but no PIN. An open-chain N is another class
('1-[(dimethylamino)oxy]ethan-1-one (PIN)',:32068) and is not matched. A molecule with another ester is left alone
(which ester names it is not decided here). Reads the structure and the name only.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_RING_N_ESTER = Chem.MolFromSmarts('[CX3](=O)[OX2;!R][#7;R;+0]')
#: neutral groups drawn with charges (nitro, azido, diazo): no ion
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
#: salts, acids, anhydrides and other esters "salts > acids > anhydrides > esters")
_SENIOR_OR_OTHER_ESTER = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se,N])[OX2H1,SX2H1,SeX2H1]',          # carboxylic and chalcogen acids
    '[!#6;!#1;!#7](=O)[OX2H1]',                        # oxoacids of S, Se, P, As, B,...
    '[CX3](=[O,S])[OX2,SX2][CX3]=[O,S]',               # anhydrides
    '[CX3](=[O,S,Se])[OX2,SX2,SeX2;!R][#6]',            # open carboxylic and thio esters
    '[!#6;!#1](=O)[OX2;!R][#6]',                       # esters of noncarbon oxoacids
    '[CX3](=O)[OX2][Si,Ge,Sn,Pb,B,Al,Ga,In,Tl,P,As,Sb,Bi]',   # the other pseudoesters
))


def _ends_in_ester_word(name: str) -> bool:
    last = name.strip().rsplit(' ', 1)
    return len(last) == 2 and last[1].rstrip(')]}').endswith('ate')


@register('P-65.6.3.4.1')
def ring_n_pseudoester_prefix_check(mol, name):
    """The name writes an ester on a ring N as an acyloxy prefix where it names the molecule."""
    if not name or not mol.HasSubstructMatch(_RING_N_ESTER):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if len(mol.GetSubstructMatches(_RING_N_ESTER)) == 1 and any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER):
        return None
    if len(mol.GetSubstructMatches(_RING_N_ESTER)) > 1 and any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER[:3]):
        return None
    if _ends_in_ester_word(name):
        return None
    return SpellingFailure('P-65.6.3.4.1', "an ester on a ring nitrogen is named as a traditional ester "
                                           "('2,5-dioxopyrrolidin-1-yl acetate'), not as an acyloxy prefix")
