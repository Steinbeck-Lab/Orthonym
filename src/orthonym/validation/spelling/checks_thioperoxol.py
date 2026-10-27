"""Spelling checks of the thioperoxol names, read lexically (registered with
:func:`orthonym.validation.pin_spelling.register`).

A twin read-back proves the structure of a thioperoxol name, never its spelling
(``validation.thioperoxol_twin``): OPSIN reads 'cyclohexane-1-sulfenic acid' and 'ethane-sulfenate' for the
structures of 'cyclohexane-1-SO-thioperoxol' and 'ethane-SO-thioperoxolate', neither of which is the PIN.

* (the Blue Book) "The locant '1' is omitted:... (c) in monosubstituted homogeneous
  monocyclic rings" (:2913), 'cyclohexanethiol (PIN)': a ring with a thioperoxol suffix and nothing else
  cites no locant, 'cyclohexane-SO-thioperoxol'.
* Anions derived from hydroxy compounds (:41011): 'CH3-CH2-S-O- ethane(SO-thioperoxolate) (PIN)'
  (:41035): the parent hydride that ends without a locant is followed by the enclosed suffix.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

_MONOCYCLE_LOCANT = re.compile(r"^cyclo[a-z]*ane-1-(?:SO|OS)-thioperoxol(?:ate)?$")
_UNENCLOSED_ANION = re.compile(r"[a-z]ane-(?:SO|OS)-thioperoxolate$")


@register("P-14.3.4")
def monocycle_locant_check(mol, name):
    """A monosubstituted monocycle whose thioperoxol suffix cites the locant 1 (c),:2913)."""
    for word in (name or "").split():
        if _MONOCYCLE_LOCANT.match(word):
            return SpellingFailure("P-14.3.4.2", f"locant '1' cited on the monosubstituted ring in '{word}' "
                                                 f"(thioperoxol suffix, P-14.3.4.2 (c))")
    return None


@register("P-72.2.2.2.2")
def enclosed_anion_suffix_check(mol, name):
    """A thioperoxolate of a parent without a locant that does not enclose its suffix (:41035)."""
    for word in (name or "").split():
        if _UNENCLOSED_ANION.search(word):
            return SpellingFailure("P-72.2.2.2.2", f"'{word}': the anion suffix of a parent without a locant is "
                                                   f"enclosed, 'ethane(SO-thioperoxolate)' (thioperoxolate)")
    return None


_DITHIOPEROXOL_GROUP = Chem.MolFromSmarts("[#6][SX2][SX2H1]")
_DISULFANYL_PREFIX = re.compile(r"disulfanyl")
_MULTIPLIED_DITHIOPEROXOL = re.compile(r"(?:di|tri|tetra|penta|hexa)dithioperoxol")
_PREFIX_ATOMS = frozenset({6, 9, 17, 35, 53})        # carbon and the halogens: prefixes only


def _only_dithioperoxol_groups(mol) -> bool:
    """True when the molecule's only characteristic groups are '-S-SH' groups on carbon: every other atom is carbon or a
    halogen, no bond between them is multiple outside a ring, so no group of a higher class is there to be the suffix."""
    groups = {m[2] for m in mol.GetSubstructMatches(_DITHIOPEROXOL_GROUP)} | {m[1] for m in
                                                                              mol.GetSubstructMatches(_DITHIOPEROXOL_GROUP)}
    if not groups:
        return False
    for atom in mol.GetAtoms():
        if atom.GetIdx() in groups or atom.GetAtomicNum() in _PREFIX_ATOMS:
            continue
        return False
    return not any(b.GetBondTypeAsDouble() > 1.0 and not b.IsInRing() and not b.GetIsAromatic() for b in mol.GetBonds())


@register("P-63.4.2")
def ring_dithioperoxol_check(mol, name):
    """A '-S-SH' group that is the principal characteristic group, cited as 'disulfanyl'.

     "Chalcogen analogues of hydroperoxides" (the Blue Book), (:27963): the compounds R-S-SH
    "When representing the principal function, they are named... (1) by substitutive nomenclature and the appropriate
    suffix listed in Table 6.1... Method (1) generates preferred IUPAC names": '-dithioperoxol' (Table 6.1,:27981),
    'ethanedithioperoxol (PIN)' (:27998), 'benzene-1,4-bis(dithioperoxolate) (PIN)' (:41041); 'disulfanyl' is the prefix
    of a group that is not the principal one (:28006, '3,4-bis(disulfanyl)benzamide (PIN)',:28021). Decided only
    where the group is the one characteristic group of the molecule (carbon and halogens besides)."""
    if not _DISULFANYL_PREFIX.search(name or "") or not _only_dithioperoxol_groups(mol):
        return None
    return SpellingFailure("P-63.4.2.1", "the -S-SH group is the principal characteristic group and is cited as the "
                                         "prefix 'disulfanyl' ('benzenedithioperoxol')")


@register("P-16.3.6")
def multiplied_dithioperoxol_check(mol, name):
    """'didithioperoxol': a multiplied suffix that begins with 'di' takes 'bis' (e)).

     (:7134) "The prefixes 'bis', 'tris', 'tetrakis', etc. are also used to avoid ambiguity:... (e) before
    names beginning with a multiplicative prefix 'di'": 'bis(diazonium)' (:7192), 'bis(disulfanyl) (preselected
    prefix) (not didisulfanyl)' (:7196); the same for the suffixes 'bis(dithioic acid)' (:7108, (b)):
    'ethane-1,2-bis(dithioperoxol)', not '...didithioperoxol'."""
    if _MULTIPLIED_DITHIOPEROXOL.search(name or ""):
        return SpellingFailure("P-16.3.6", "the multiplied suffix 'dithioperoxol' takes 'bis', as 'bis(dithioperoxol)' (P-16.3.6 (e))")
    return None
