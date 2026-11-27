"""Spelling check: a name on an amine or aniline parent for a molecule that holds an amide (registered with
:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book): "11 Amides" (with thioamides, ureas and the amides of sulfonic acids), 12 hydrazides,
13 imides and 16 ketones and pseudoketones (a ring-N acyl group, a lactam) all rank above "19 Amines". So a molecule with
such a group is never named on an amine parent: "Anilides" (:32849), "names expressing N-substitution by a
phenyl group on an amide are preferred IUPAC names" ('N-(3-bromophenyl)-2H-1,3-benzodioxole-5-carboxamide', not
'N-(2H-1,3-benzodioxole-5-carbonyl)-3-bromoaniline'); "'Hidden' amides" (:33125), '1-(piperidin-1-yl)ethan-1-one
(PIN)' (:33129). A name that ends in 'amine' or 'aniline' is then correct but no PIN.

Only a sulfonamide whose S and N are both acyclic counts: a sulfonyl on a ring N ('1-(methanesulfonyl)piperidin-4-amine')
or a ring sulfonamide (a sultam) is no amide suffix the book prints, so those names are left alone. Ions, radicals and
mixtures are left alone. Reads the structure and the name only.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_SENIOR_TO_AMINES = tuple(Chem.MolFromSmarts(s) for s in (
    '[CX3](=[O,S])[#7]',                       # amides, thioamides, ureas, carbamates, lactams, ring-N acyl groups
    '[SX4;!R](=O)(=O)[#7;!R]',                 # acyclic sulfonamides
))
_CHARGE_PAIRS = tuple(Chem.MolFromSmarts(s) for s in ('[N+](=O)[O-]', '[#7]=[N+]=[N-]', '[#6]=[N+]=[N-]'))


@register('P-41')
def amine_parent_beside_amide_check(mol, name):
    """The name ends in 'amine' or 'aniline' while an amide (or a class above the amines named here) is present."""
    if not name or ' ' in name.strip() or not name.rstrip(')]}').endswith(('amine', 'aniline')):
        return None
    paired = {i for p in _CHARGE_PAIRS for m in mol.GetSubstructMatches(p) for i in m}
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() or (a.GetFormalCharge() and a.GetIdx() not in paired)
                                              for a in mol.GetAtoms()):
        return None
    if not any(mol.HasSubstructMatch(p) for p in _SENIOR_TO_AMINES):
        return None
    return SpellingFailure('P-41', "an amide (P-41 class 11) or a pseudoketone outranks the amines (class 19); "
                                   "a name on an amine parent is no PIN")
