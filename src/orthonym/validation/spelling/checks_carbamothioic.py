"""Spelling check: the chalcogen analogues of carbamic and carbonic acid (registered with
:func:`orthonym.validation.pin_spelling.register`).

 (the Blue Book-:30832), "functional replacement nomenclature is applied to the retained name
'carbamic acid'": 'H2N-CS-OH carbamothioic O-acid (PIN)' (:30774), 'H2N-CO-SH carbamothioic S-acid (PIN)'
(:30826), 'HO-CO-SH carbonothioic S-acid (PIN) (not sulfanylformic acid)' (:30778), 'carbonothioic O,O-acid
(PIN)' (:30861); "Substitution of formic acid is not recommended". Two spellings are therefore not the PIN:

* the formic-acid spelling of an N-substituted one, '(methylamino)methanethioic O-acid' or '(diethylamino)methane-
  (dithioic acid)', where the PIN is 'methylcarbamothioic O-acid', 'diethylcarbamodithioic acid';
* an acid word without the italic O-/S- locant ('carbamothioic acid', 'carbonothioic acid') when the chalcogen atoms
  of the acid carbon are not all of one element: the PIN cites the locant, and OPSIN 2.9.0 reads no such word, so
  every name the engine can ship for it is valid but not the PIN.

The structure side is an acyclic carbon with one double bond to O, S, Se or Te, an -OH/-SH/-SeH/-TeH and, as its
third neighbour, a nitrogen (carbamic) or a second hydroxy-type group (carbonic). No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_CHALCOGENS = {'O', 'S', 'Se', 'Te'}
_CHALCOGEN_ACID_WORD = re.compile(r"(?:thio|seleno|telluro)ic\)? acid|(?:thioic|selenoic|telluroic) [OS]")


def _acid_centres(mol):
    """``(has_nitrogen, chalcogen elements)`` of each acyclic carbamic/carbonic acid carbon of ``mol``."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or atom.IsInRing() or atom.GetDegree() != 3 or atom.GetFormalCharge():
            continue
        double = [b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
        single = [b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE]
        if len(double) != 1 or double[0].GetSymbol() not in _CHALCOGENS or len(single) != 2:
            continue
        acidic = [a for a in single if a.GetSymbol() in _CHALCOGENS and a.GetDegree() == 1 and a.GetTotalNumHs() == 1]
        nitrogens = [a for a in single if a.GetSymbol() == 'N' and not a.GetIsAromatic()]
        if len(acidic) == 1 and len(nitrogens) == 1:
            yield True, {double[0].GetSymbol(), acidic[0].GetSymbol()}, double[0].GetSymbol(), [acidic[0].GetSymbol()]
        elif len(acidic) == 2:
            yield (False, {double[0].GetSymbol()} | {a.GetSymbol() for a in acidic}, double[0].GetSymbol(),
                   sorted((a.GetSymbol() for a in acidic), key=['O', 'S', 'Se', 'Te'].index))


def _expected_locant(centre):
    """The book's italic locant for one acid carbon (this check's own reading of, independent of the
    producer's ``rules.chalcogen_acid_locant``): the acidic atom of a carbamic acid ('O-acid'); for a carbonic acid two
    acidic atoms of one element other than O ('Se,Se-acid'), one beside -OH and =O ('S-acid'), two -OH ('O,O-acid')."""
    has_n, _elements, double, acidic = centre
    if has_n:
        return acidic[0]
    if acidic[0] == acidic[1] != 'O':
        return f"{acidic[0]},{acidic[1]}"
    if acidic[0] == 'O' and acidic[1] != 'O' and double == 'O':
        return acidic[1]
    return 'O,O' if acidic == ['O', 'O'] else None


@register('P-65.2.1.2')
def carbamothioic_acid_check(mol, name):
    """: a formic-acid spelling of a carbamic analogue, or a mixed-chalcogen acid word with no O-/S- locant."""
    for has_n, elements, _double, _acidic in _acid_centres(mol):
        if elements == {'O'}:
            continue
        if has_n and 'carbam' not in name and re.search(r"methane(?:\(di)?(?:thio|seleno|telluro)ic", name):
            return SpellingFailure('P-65.2.1.2', "a carbamic acid analogue written as a substituted formic acid; "
                                                 "the PIN is a 'carbamo...ic acid' (substitution of formic acid is "
                                                 "not recommended)")
    centres = list(_acid_centres(mol))
    for has_n, elements, _double, _acidic in centres:
        if len(elements) > 1 and _CHALCOGEN_ACID_WORD.search(name):
            loc = _expected_locant(centres[0]) if len(centres) == 1 else None
            if not loc or f" {loc}-acid" not in name:
                return SpellingFailure('P-65.2.1.2', "a mixed-chalcogen carbamic/carbonic acid needs the italic "
                                                     "element locant of its acidic atoms ('carbamothioic S-acid')")
    return None
