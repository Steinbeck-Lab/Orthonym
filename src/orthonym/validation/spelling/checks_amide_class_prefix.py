"""Spelling check /: an amide-class group (amide, thioamide, amidine, sulfonamide) cited only as a
prefix in a name with no suffix (registered with:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book): "11 Amides" (thioamides, the imidamides or amidines, and the amides of sulfonic acids
rank with them). The prefixes 'carbamoyl', 'carbamothioyl', 'carbamimidoyl' and 'sulfamoyl' are used only "In the presence of
a function having priority for citation as suffix" "Names of substituent groups derived from chalcogen
derivatives of amides",:33196; "Substituents of the types -SO2-NH2, -SO-NH2...",:32971); 'benzenecarbothioamide
(PIN)' (:33155), 'pyrrolidine-1-sulfonamide (PIN)' (:32758), 'piperidine-1-carboxamide (PIN)' (:32685); amidines on a ring or a
heteroatom take the 'carboximidamide' suffix,:34200). So a single-word name that ends in a bare parent hydride and
cites such a group as a prefix ('(dimethylcarbamothioyl)benzene', '1-ethyl-4-(sulfamoylmethyl)benzene',
'4-carbamimidoylmorpholine') is correct but no PIN.

Not this group: a molecule with a senior class (an acid or ester of any centre, an anhydride, an acyl or sulfonyl halide or
pseudohalide, a thiuram or peroxy ester), an ion, a radical, a second fragment; a name ending in a suffix or a guanidine.
Reads the structure and the name only.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_TRIGGERS = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S])[#7;!R]',                    # amides and thioamides (open N)
    '[CX3;!R](=[O,S])[#7;R]',                  # carbamoyl / thiocarbamoyl on a ring N
    '[CX3;!R](=[NX2])[#7]',                    # amidines (acyclic carbon)
    '[SX4](=O)(=O)[#7]',                       # sulfonamides
))
_SENIOR = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se])[OX2H1,SX2H1,SeX2H1]',               # carboxylic and chalcogen acids
    '[!#6;!#1;!#7](=O)[OX2H1,SX2H1]',                     # acids of S, P, As, B,...
    '[CX3](=[O,S,Se])[O,S,Se;X2][!#1]',                    # esters, anhydrides, thiurams, peroxy esters (O/S/Se on C)
    '[!#6;!#1;!#7](=O)[OX2][#6]',                         # esters of noncarbon oxoacids (sulfamates, phosphates,...)
    '[CX3](=[O,S])[F,Cl,Br,I]', '[!#6;!#1](=O)[F,Cl,Br,I]',   # acyl, sulfonyl, sulfinyl, phosphoryl halides
    '[CX3](=[O,S])[$(N=[N+]=[N-]),$(C#N),$(N=C=[O,S])]',   # acyl pseudohalides (azide, cyanide, isocyanate)
))
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
_HYDRIDE_END = re.compile(r'(?:ane|ene|yne|ole|ine)[)\]}]*$')
_NOT_A_HYDRIDE_END = re.compile(r'(?:amine|imine|aniline|guanidine|hydroxylamine|hydrazine|amidine|toluidine)[)\]}]*$')
_PREFIX_WORD = re.compile(r'(?<!thio)(?:carbamoyl|carbamothioyl|carbamimidoyl|sulfamoyl)')


@register('P-66.1.1.4.2')
def amide_class_as_prefix_check(mol, name):
    """An amide-class group is the senior class and is cited only as a prefix on a bare parent hydride."""
    if not name or ' ' in name.strip():
        return None
    end = name.strip()
    if not _HYDRIDE_END.search(end) or _NOT_A_HYDRIDE_END.search(end) or not _PREFIX_WORD.search(end):
        return None
    if not any(mol.HasSubstructMatch(p) for p in _TRIGGERS):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if any(mol.HasSubstructMatch(p) for p in _SENIOR):
        return None
    return SpellingFailure('P-66.1.1.4.2', "an amide-class group is the senior class and takes the suffix "
                                           "('-carboxamide', '-carbothioamide', '-carboximidamide', '-sulfonamide'); "
                                           "a name citing it only as a prefix on a bare parent hydride is no PIN")
