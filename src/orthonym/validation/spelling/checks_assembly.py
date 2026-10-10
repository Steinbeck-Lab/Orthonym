"""Spelling check: a molecule that holds a ring assembly is named as one (registered with
:func:`orthonym.validation.pin_spelling.register`).

``### **** DEFINITIONS`` (the Blue Book),:15542: "Two or more cyclic systems (single
rings or fused systems, alicyclic von Baeyer systems, spiro systems, phane systems, fullerenes)
that are directly joined to each other by single or double bonds are called 'ring assemblies' when
the number of such direct ring junctions is one less than the number of cyclic systems involved.";
``## ****`` (:24070) ``### ****`` (:24072) "Preferred IUPAC names for assemblies of
two or more identical cyclic systems joined by a single bond are formed using the names of parent
hydrides rather than the names of substituent groups"; ``## ****`` (:24151),:24159
'1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)' (not '2-(piperidin-2-yl)pyridine'); ``###
**** Ring assemblies composed of monocyclic components`` (:17077),:17083
'2,3-dihydro-1,1'-biphenyl (PIN)' (not '(cyclohexa-1,3-dien-1-yl)benzene'); ``### ****``
(:16107):16118 '[1,1'-biphenyl]-4-yl (preferred prefix)'.

A round trip proves the molecule, never that the name takes the assembly as one parent: a
substitutive producer that takes two identical ring systems apart ('5-(benzyloxy)-2-(pyridin-2-yl)
pyridine' for 5-(benzyloxy)-2,2'-bipyridine) round-trips exactly and ships ``pin_verified``. Seven
producers do this, and every one of them passes the call site of the spelling checks, so one check
there decides the class.

Structure side (RDKit only, and a reader of its own: the checks import no naming or rules code,
see ``test_the_checks_import_no_naming_code_and_no_opsin``): the ring systems (rings that share an
atom are one system) and the groups of two or more systems with the same skeleton joined by a bond.
Identity is that of the skeleton, so the mancude and the saturated forms of one ring are one group
 :24153 "When assemblies of otherwise identical rings contain both mancude and saturated
rings, the use of hydro prefixes is preferred, except in the case of a two ring assembly
consisting of one benzene ring and a cyclohexane ring"), a ring-nitrogen or a double-bond junction
counts:15593, '1,1'-bipyrrole (PIN)':15603), and that one benzene + cyclohexane
assembly is no group ('cyclohexylbenzene (PIN)':24157). The reader agrees with
``orthonym.rules.ring_assembly_screen._assemblies`` that the drug-lane producers consult
(``test_leads_l2_assembly_label.py`` compares the two over the eval molecules).

Name side: a lexical count of the assembly texts, a primed (or composite) junction locant set
directly before ``bi``, ``ter``, ``quater``, ``quinque``, ``sexi`` or ``septi``. No naming code, no
OPSIN, so the check cannot inherit a defect of the code that built the name.

Verdict: the molecule holds at least one group and the name cites no assembly text. The check
abstains (returns None) for the names and structures the rule does not decide: a linear phane name
 :23901;:24088 "Phane names are preferred IUPAC names rather than ring
assembly names when seven or more rings or ring systems are present") and a group of seven or more
ring systems (the phane parent hydride is the PIN there, ``checks_phane``).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .lexer import normalise
from .linear_phane import is_linear_phane_name

#: a primed (two rings) or composite (three or more rings; ':' separates the junction pairs, '^'
#: the superscript of a composite locant) junction locant set directly before the multiplying
#: prefix of an assembly: '2,2'-bipyridine', '1,1':4',1''-terphenyl', '1^1,2^1:2^2,3^1-tercyclopropane'.
#: 'bis' before an enclosing mark (a multiplied substituent: '3,3'-bis(...)'), 'tert-' and
#: 'terephth...' are not assemblies; 'bi' before a ring name that starts with 's' is one
#: ('1,1'-bistibinane',; the Blue Book).
ASSEMBLY_HEAD = (
    r"(?<![\w'^])\d[\d,:'^a-zλ]*(?:'|:)[\d,:'^a-zλ]*-"
    r"(?:bi(?!s[(\[{])|ter(?!t-|ephth)|quater|quinque|sexi|septi|octi|novi|deci)")
_ASSEMBLY_TEXT = re.compile(ASSEMBLY_HEAD + r"(?=[a-z(\[{\-\d])")

#: seven or more ring systems: (:24088) names them with a phane parent hydride
_PHANE_RINGS = 7


def count_assembly_texts(name: str) -> int:
    """How many ring assembly descriptors the name cites (see ``_ASSEMBLY_TEXT``)."""
    return len(_ASSEMBLY_TEXT.findall(normalise(name)))


# ------------------------------------------------------------------ the structure side
def _ring_systems(mol: Chem.Mol) -> List[Set[int]]:
    """Rings that share an atom (fused, bridged, spiro) merged into one ring system."""
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for other in [s for s in systems if s & merged]:
            merged |= other
            systems.remove(other)
        systems.append(merged)
    return systems


def _skeleton(mol: Chem.Mol, atoms: Set[int]) -> str:
    """The ring system's skeleton as a canonical SMILES: its elements and connectivity, every bond
    single, no aromatic flag, hydrogens, charges and isotopes ignored (pyridine = piperidine,
    benzene = cyclohexene)."""
    rw = Chem.RWMol()
    index: Dict[int, int] = {}
    for a in sorted(atoms):
        atom = Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum())
        atom.SetNoImplicit(True)
        index[a] = rw.AddAtom(atom)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in index and j in index:
            rw.AddBond(index[i], index[j], Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw.GetMol(), canonical=True)


def _lone_carbon_six_ring(mol: Chem.Mol, atoms: Set[int]) -> bool:
    bonds = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    return (len(atoms) == 6 and len(bonds) == 6
            and all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in atoms))


def _benzene(mol: Chem.Mol, atoms: Set[int]) -> bool:
    return (_lone_carbon_six_ring(mol, atoms)
            and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms))


def _cyclohexane(mol: Chem.Mol, atoms: Set[int]) -> bool:
    return (_lone_carbon_six_ring(mol, atoms)
            and not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms)
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds()
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms))


def ring_assembly_groups(mol: Chem.Mol) -> List[List[Set[int]]]:
    """The ring assemblies of ``mol``: each a list of two or more ring systems (atom sets) with the
    same skeleton, joined by bonds. The two-ring assembly of one benzene ring and one cyclohexane
    ring:24153) is no group."""
    systems = _ring_systems(mol)
    if len(systems) < 2:
        return []
    owner = {a: k for k, s in enumerate(systems) for a in s}
    skeletons: Dict[int, str] = {}
    root = list(range(len(systems)))

    def find(k: int) -> int:
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for b in mol.GetBonds():
        ki, kj = owner.get(b.GetBeginAtomIdx()), owner.get(b.GetEndAtomIdx())
        if ki is None or kj is None or ki == kj:
            continue
        for k in (ki, kj):
            if k not in skeletons:
                skeletons[k] = _skeleton(mol, systems[k])
        if skeletons[ki] == skeletons[kj]:
            root[find(ki)] = find(kj)
    grouped: Dict[int, List[int]] = {}
    for k in range(len(systems)):
        grouped.setdefault(find(k), []).append(k)
    groups = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        if len(members) == 2:
            a, b = (systems[k] for k in members)
            if (_benzene(mol, a) and _cyclohexane(mol, b)) or (_benzene(mol, b) and _cyclohexane(mol, a)):
                continue
        groups.append([systems[k] for k in members])
    return groups


# ------------------------------------------------------------------ the check
@register("P-28.1")
def ring_assembly_named_as_such_check(mol, name) -> Optional[SpellingFailure]:
    """ (:15542) with (:24072), (:24159) and (:17083): a
    molecule whose ring systems form a ring assembly is named with an assembly parent hydride
    (or an assembly prefix,:16118); a name that cites no assembly takes the assembly
    apart into a parent ring and a ring substituent, and is not the PIN."""
    try:
        groups = ring_assembly_groups(mol)
    except Exception:  # noqa: BLE001 - the structure could not be read: abstain
        return None
    if not groups:
        return None
    if max(len(g) for g in groups) >= _PHANE_RINGS or is_linear_phane_name(name):
        return None
    if count_assembly_texts(name) >= 1:
        return None
    sizes = sorted((len(g) for g in groups), reverse=True)
    return SpellingFailure(
        "P-28.1", f"the structure holds {len(groups)} ring assembly group(s) of {sizes} ring "
                  f"systems, and the name cites no ring assembly")
