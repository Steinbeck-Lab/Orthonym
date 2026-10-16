"""Spelling check of the prefix 'oxophosphanyl': a heterone is a principal characteristic group
(registered with:func:`orthonym.validation.pin_spelling.register`).

A round trip proves the molecule, never the spelling: OPSIN reads '{4-[(oxophosphanyl)oxy]phenyl}
methanol' back to the structure of the PIN '[4-(hydroxymethyl)phenoxy]phosphanone'. The check reads the
RDKit structure and the name text only.

* "HETERONES" (the Blue Book): "Compounds containing the -PO, -PO2, -AsO or -AsO2 are
  called heterones (see,. In the presence of a more senior characteristic group
  they are described by the compound prefixes oxophosphanyl,...". The prefix is therefore correct
  only when a characteristic group senior to the heterone is present.
* (:18162; items:18174-:18189): 'ketones, pseudoketones and heterones' are class 16,
  below the radicals and ions (1-4), acids, anhydrides, esters, acid halides, amides, hydrazides,
  imides, nitriles and aldehydes (5-15), and above hydroxy compounds (17), hydroperoxides (18),
  amines (19) and imines (20). (:29390): an acyclic heterone takes the suffix 'one' and
  that "leads to preferred IUPAC names": 'phenylphosphanone (PIN)' (:28291).
* Within class 16, (:29628): "There is no seniority order difference between ketones and
  pseudoketones. When necessary, the maximum number of carbonyl groups or doubly bonded oxygen atoms,
  the seniority order between chains and rings, and between rings and ring systems, are considered".
  The count first,:18875): a parent that holds two or more ketone groups is senior to the
  phosphanone, which holds one. Then (:18917): "The senior parent structure, whether cyclic
  or acyclic, has the senior atom... N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In >
  Tl > O > S > Se > Te > C", "A single senior atom is sufficient to give seniority to the parent
  hydride". So a ketone whose parent hydride holds nitrogen keeps the prefix; a ketone on a carbon,
  oxygen or sulfur parent does not, the phosphanone is the parent.
* "Pseudoketones" (:28258): "(a) cyclic compounds in which a carbonyl group in a ring is
  bonded to one or two skeletal heteroatoms", '1,3-dioxan-2-one (PIN)': a lactone, a cyclic carbonate
  and a cyclic urea are class-16 ketones with the ring as the parent hydride (the lactam, the imide
  and the cyclic urea of an N ring therefore keep the prefix; the lactone and the carbonate of an
  all-O ring do not). ``_CYCLIC_ESTER_IS_PSEUDOKETONE`` reads this once.

A name that cites the prefix for a structure whose senior class is below class 16, or is class 16
on a carbon, oxygen or sulfur parent, is a correct name that is not the PIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

#: (a): the carbonyl of a lactone or a cyclic carbonate is a class-16 pseudoketone
_CYCLIC_ESTER_IS_PSEUDOKETONE = True

#: the group the prefix names: P with its =O and one more neighbour (bare -P=O or -PH=O)
_HETERONE_P = Chem.MolFromSmarts("[P;D2;+0](=[O;D1;+0])-[!#1]")

_CHALC = "O,S,Se,Te"        # the doubly bonded chalcogen of a carbonyl or its analogues

#: classes 5-15 of Table 4.1 (acids to aldehydes), senior to the heterone. The ring carbonyl
#: bonded to ring heteroatoms (lactone, lactam, cyclic carbonate, imide, cyclic anhydride) is a
#: pseudoketone, class 16, and is not here (``_RING_CARBONYL``).
_SENIOR = [Chem.MolFromSmarts(s) for s in (
    "[CX3](=[O,S,Se,Te])[OX2H1,SX2H1,SeX2H1,TeX2H1]",          # carboxylic acids, chalcogen analogues
    "[CX3](=O)[OX2][OX2H1]",                                    # peroxoic acids
    "[SX4](=O)(=O)[OX2H1]", "[SX3](=O)[OX2H1]", "[PX4](=O)[OX2H1]",
    "[$([CX3]=O),$([SX4]=O),$([PX4]=O)]-!@[OX2]-!@[$([CX3]=O),$([SX4]=O),$([PX4]=O)]",   # anhydrides
    "[CX3](=[O,S,Se,Te])-!@[OX2,SX2,SeX2,TeX2][#6]",            # esters (acyclic ester bond)
    "[SX4](=O)(=O)-!@[OX2][#6]", "[PX4](=O)-!@[OX2][#6]",
    "[CX3,SX4](=O)[F,Cl,Br,I]",                                 # acid halides
    "[CX3](=[O,S,Se,Te])-!@[NX3;!R]",                           # amides, hydrazides
    "[SX4](=O)(=O)-!@[NX3;!R]",                                 # sulfonamides
    "[CX3;!a](=[NX2;!R])-!@[NX3;!R]",                           # amidines, guanidines
    "[CX3](=O)[NX3][CX3](=O)",                                  # imides
    "[#6][CX2]#[NX1]",                                          # nitriles
    "[CX3H1](=[O,S,Se,Te])[#6]",                                # aldehydes
)]

#: the carbonyl carbon of a ketone between two carbons (acyclic or in a ring)
_KETONE_C = Chem.MolFromSmarts(f"[#6X3;$([#6](=[{_CHALC};D1])([#6])[#6])]")
#: the carbonyl carbon of a pseudoketone: in a ring, bonded to one or two ring heteroatoms
_RING_CARBONYL_C = Chem.MolFromSmarts(f"[#6X3;R;$([#6](=[{_CHALC};D1])[#7,#8,#16,#34,#52;R])]")
#: the carbonyl of a lactone or a cyclic carbonate (a class-9 ester if ``_CYCLIC_ESTER_IS_PSEUDOKETONE``
#: were False)
_CYCLIC_ESTER_C = Chem.MolFromSmarts(f"[#6X3;R;$([#6](=[{_CHALC};D1])[#8,#16;R])]")

#: the N+ / O- pair of a nitro group or an N-oxide, and a diazo or azide pair: no ion
_NEUTRAL_PAIR = Chem.MolFromSmarts("[$([N+,n+]=O),$([N+,n+]-[O-]),$([O-]-[N+,n+]),$([N+]=[N-]),"
                                   "$([N-]=[N+]),$([N+]=[N+]-[N-]),$([N-]-[N+]#N)]")


def _is_ion_or_radical(frag) -> bool:
    """Classes 1-4: a radical, an anion, a cation or a zwitterion (not a nitro group, an N-oxide,
    a diazo or an azide, whose charges cancel inside the group)."""
    if any(a.GetNumRadicalElectrons() for a in frag.GetAtoms()):
        return True
    paired = {i for m in frag.GetSubstructMatches(_NEUTRAL_PAIR) for i in m}
    return any(a.GetFormalCharge() and a.GetIdx() not in paired for a in frag.GetAtoms())


def _ring_systems(frag) -> list:
    """The atom sets of the ring systems: rings that share an atom form one (fused, spiro, bridged)."""
    systems: list = []
    for ring in frag.GetRingInfo().AtomRings():
        ring = set(ring)
        merged = [s for s in systems if s & ring]
        for s in merged:
            systems.remove(s)
            ring |= s
        systems.append(ring)
    return systems


def _ketone_parent_is_senior(frag) -> bool:
    """True when class 16 is the senior class present and a ketone group holds the parent that the
    prefix is cited on: a ring system with nitrogen, or one with two or more ketone groups
    ,. A ketone on a chain, a hidden amide (b), the acyl chain is the
    parent) and one on a carbon, oxygen or sulfur ring system are not: the phosphanone is the parent."""
    carbonyls = {m[0] for pat in (_KETONE_C, _RING_CARBONYL_C) for m in frag.GetSubstructMatches(pat)}
    for system in _ring_systems(frag):
        if (len(carbonyls & system) >= 2
                or (carbonyls & system and any(frag.GetAtomWithIdx(i).GetSymbol() == "N" for i in system))):
            return True
    return False


def _senior_than_heterone(frag) -> bool:
    """True when the principal characteristic group of ``frag`` is senior to the heterone, or ties
    with it and its parent is senior (the prefix is then the PIN spelling)."""
    if _is_ion_or_radical(frag) or any(frag.HasSubstructMatch(p) for p in _SENIOR):
        return True
    if not _CYCLIC_ESTER_IS_PSEUDOKETONE and frag.HasSubstructMatch(_CYCLIC_ESTER_C):
        return True
    return _ketone_parent_is_senior(frag)


@register("P-61.6")
def heterone_prefix_check(mol, name):
    """ (:25977): 'oxophosphanyl' is the prefix of a heterone only in the presence of a more
    senior characteristic group. A name that cites it for a component in which the heterone is the
    principal characteristic group is not the PIN (the PIN is the phosphanone)."""
    if "oxophosphanyl" not in name:
        return None
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        if not frag.HasSubstructMatch(_HETERONE_P):
            continue
        if not _senior_than_heterone(frag):
            return SpellingFailure(
                "P-61.6", "the prefix 'oxophosphanyl' names a heterone, which is class 16 (Table 4.1) and "
                          "the principal characteristic group here: the PIN is the phosphanone")
    return None
