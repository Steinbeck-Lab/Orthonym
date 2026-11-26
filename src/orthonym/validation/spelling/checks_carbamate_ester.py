"""Spelling check: a carbamate ester named on a junior parent with the ester as an '(alkoxycarbonyl)' prefix
(registered with:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book): "All preferred IUPAC names for esters are named by functional class nomenclature";
 (:18182) ranks "9 Esters" above acid halides (10), amides and ureas (11), hydrazides, imides, nitriles,
aldehydes, ketones, alcohols, amines, imines and the classes of the heteroatom parents (21-40); '2-hydroxypropyl
(2-aminoethyl)carbamate (PIN)' (:30766). So a molecule whose senior class is one carbamate ester (an open O on carbon) is
named as the ester ('tert-butyl piperidine-1-carboxylate', 'benzyl phenylcarbamate', '2-imidodicarbonate' esters,:31049),
and a name on a ring, urea, sulfonyl halide or silane parent ('2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)ethane-1-
sulfonyl chloride') is correct but no PIN.

Not this group: an acid (carboxylic, chalcogen, or of B, P, As, Sb, S, Se), an anhydride, another open ester (carboxylic or
carbonate on carbon, thio, noncarbon oxoacid, pseudoester incl. ring N), an ion, a radical, a second fragment; a cyclic
carbamate (ring O) or an oxime carbamate (O on N). Two or more carbamate esters are left to their own lead unless the name
has no ester word at all. Reads the structure and the name only.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

# the N of an azide or an isocyanate on the carbonyl makes another acid: 'carbonazidic acid (PIN)',:30834),
# 'ethyl carbonazidate', not a carbamate
_CARBAMATE_ESTER = Chem.MolFromSmarts('[#7;!$(N=[N+]=[N-]);!$(N=N#N);!$(N=C=[O,S,Se,Te])][CX3](=O)[OX2;!R][#6]')
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
_SENIOR_OR_OTHER_ESTER = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se,N])[OX2H1,SX2H1,SeX2H1]',               # carboxylic, chalcogen and imidic acids
    '[B,P,As,Sb,S,Se,Te,Si,N;!$([N+](=O)[O-])][OX2H1,SX2H1]',  # acids of noncarbon centres (boronic, arsonic,...)
    '[CX3](=[O,S])[OX2][OX2H1]', '[!#6;!#1](=O)[OX2][OX2H1]',   # peroxy acids
    '[CX3](=[O,S])[OX2,SX2][CX3]=[O,S]',                    # anhydrides
    '[!#6;!#1](=O)[OX2][!#6;!#1]=O', '[!#6;!#1](=O)[OX2][CX3]=[O,S]',   # anhydrides of noncarbon acids
    '[CX3;!$(C[#7])](=[O,S,Se])[OX2,SX2,SeX2;!R][#6]',       # other open esters on carbon (carboxylic, carbonate, thio)
    '[CX3;$(C[#7])](=S)[OX2,SX2;!R][#6]',                   # thiocarbamate esters
    '[CX3](=[NX2])[OX2,SX2;!R][#6]',                        # imidic esters ('ethyl...propanimidate')
    '[!#6;!#1;!#7](=O)[OX2;!R][#6]',                        # esters of noncarbon oxoacids
    '[CX3](=O)[OX2][Si,Ge,Sn,Pb,B,Al,Ga,In,Tl,P,As,Sb,Bi,$([#7;R;+0])]',   # pseudoesters
))
_ESTER_WORD = re.compile(r'(?:carbamate|carboxylate|imidodicarbonate)[)\]}]*$')


@register('P-65.6.3.2.1')
def carbamate_ester_name_check(mol, name):
    """A molecule whose senior class is a carbamate ester is named on another parent."""
    if not name or not mol.HasSubstructMatch(_CARBAMATE_ESTER):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER_ESTER):
        return None
    if _ESTER_WORD.search(name.strip()):
        return None
    return SpellingFailure('P-65.6.3.2.1', "a carbamate ester is the senior class (P-41 class 9); its functional-class "
                                           "ester name is the PIN, not a name with an '(alkoxycarbonyl)' prefix")
