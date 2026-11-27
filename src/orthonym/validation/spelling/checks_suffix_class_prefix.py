"""Spelling check: a thiol, hydroperoxide, alcohol or ketone cited only as a prefix in a name with no suffix
(registered with:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book-18191): "16 Ketones", "17 Hydroxy compounds and chalcogen analogues", "18 Hydroperoxides
(peroxols)" are classes expressed as suffixes, and (:18875): "The senior parent structure has the maximum number of
substituents corresponding to the principal characteristic group (suffix)" -- 'cyclohexanethiol (PIN)' (:2917),
'ethaneperoxol (PIN)' (:24316), '1,2,3,4-tetrahydronaphthalene-1-peroxol (PIN)' (:27948). So when such a group is the senior
class, a name that ends in a bare parent hydride and cites the group as 'sulfanyl', 'hydroperoxy', 'hydroxy' or 'oxo'
('1-methyl-2-sulfanyl-4,5-dihydro-1H-imidazole', PIN '1-methyl-4,5-dihydro-1H-imidazole-2-thiol') is correct but no PIN.

Not this group: a molecule with a class senior to the group or a suffix of its own -- an acid of carbon or of a noncarbon
centre, an ester, anhydride, amide, lactone, lactam, carbamate, acyl halide (any C(=O|S) on O, S, N or a halogen), a
sulfonamide, a guanidine, an acyclic amidine carbon, an N-cyano group, a nitrile or an aldehyde; an ion, a radical, a second
fragment; a name that ends in a suffix, an amine-type parent or a retained name (only bare parent-hydride endings are read).
Reads the structure and the name only.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_TRIGGERS = tuple(Chem.MolFromSmarts(s) for s in (
    '[#6][SX2H1]',                                          # thiols (on any carbon, C=N included)
    '[#6][OX2][OX2H1]',                                     # hydroperoxides
    '[#6;!$([#6]=[O,S,N]);!$([#6]#*)][OX2H1]',             # alcohols and phenols
    '[#6][CX3](=O)[#6]',                                    # ketones
    '[#6][CX3;!R](=[O,S])[#7;R]',                           # an acyl on a ring N: a pseudoketone,:33125)
))
_SENIOR_OR_OTHER = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S,Se])[O,S,Se,F,Cl,Br,I]',                  # acids, esters, anhydrides, lactones, acyl halides
    '[CX3](=[O,S,Se])[#7;!R]', '[CX3;R](=[O,S,Se])[#7]',   # amides, carbamates, lactams (an acyl on a ring N is none)
    '[CX3](=[O,S,Se])([#7;R])[!#6]',                       # ureas and carbamates of a ring N
    '[!#6;!#1;!#7;!#8][OX2H1,SX2H1]',                      # acids of noncarbon centres
    '[SX4](=O)(=O)[#7;!R]', '[SX3](=O)[#7;!R]',            # sulfonamides, sulfinamides (a sulfonyl on a ring N is none)
    '[CX3](=[NX2])([#7])[#7]',                              # guanidines
    '[CX3;!R](=[NX2])[#7]', '[CX3;!R](=[NX2])[OX2H1]',     # acyclic amidine and imidic acid carbons
    '[#7]C#N', '[CX2]#[NX1]', '[CX3H1](=O)[#6]',           # N-cyano groups, nitriles, aldehydes
))
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))
#: bare parent-hydride endings; the amine-type and retained parents among them are listed apart
_HYDRIDE_END = re.compile(r'(?:ane|ene|yne|ole|ine)[)\]}]*$')
_NOT_A_HYDRIDE_END = re.compile(r'(?:amine|imine|aniline|guanidine|hydroxylamine|hydrazine|amidine|toluidine)[)\]}]*$')
_PREFIX_WORD = re.compile(r'(?<![a-z])(?:di|tri|tetra|penta|hexa|bis|tris)?(?:sulfanyl|hydroperoxy|hydroxy|oxo)'
                          r'|(?:acetyl|benzoyl|oyl|carbonyl)(?!oxy|amino|sulfanyl|imino)')   # acyl words: a ketone


@register('P-44.1.1')
def suffix_class_as_prefix_check(mol, name):
    """A senior thiol, hydroperoxide, alcohol or ketone is cited only as a prefix and the name ends in a parent hydride."""
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
    if any(mol.HasSubstructMatch(p) for p in _SENIOR_OR_OTHER):
        return None
    return SpellingFailure('P-44.1.1', "a thiol, hydroperoxide, alcohol or ketone is the senior class and takes the suffix "
                                       "(P-44.1.1); a name citing it only as a prefix on a bare parent hydride is no PIN")
