""": the molecules whose senior parent hydride is a hydrazine or a diazene.

One home of the structural question "is the hydrazine (or the diazene) unit the parent hydride of
this molecule?", asked by the producer that names the polyazane (``rules.polyazane``) and by the
spelling check that declines a 'hydrazinyl' ('diazenyl') prefix for such a molecule
(``validation.spelling.checks_parent``,. Asked in one place, the producer names exactly the
molecules for which the check calls the prefix name wrong, and no others.

 "Seniority of classes": "The senior parent structure, whether cyclic or acyclic, has the
senior atom in accordance with the seniority of classes (see expressed by the following
decreasing element order: N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S >
Se > Te > C. This criterion is applied to select the senior atom in parents and to choose between
rings and chains" (the Blue Book);: "A single senior atom is sufficient to give
seniority to the parent hydride" (:18919), with '1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine (PIN)
(N > Si > O)' (:18950). class 21 (:18197): "Nitrogen compounds: heterocycles,
polyazanes, hydrazines (except for hydrazides), diazenes, hydroxylamines, azanes (except for
amides, amines, and imines)". A hydrazine unit is therefore the parent hydride of a molecule with no
characteristic group of the classes 1 to 20 and no nitrogen ring (a heterocycle precedes it in class
21): 'phenylhydrazine (PIN)' (:38535). A diazene unit is in the same class, and is the parent of an azo
compound that carries no such group: "Symmetrical monoazo compounds" (:38759) names it
"by substituting the parent diazene, HN=NH, by the appropriate substituent groups" (:38761), '(3-chlorophenyl)
(4-chlorophenyl)diazene (PIN)' (:38774), and (:38778) "Method (1) leads to preferred IUPAC
names" (:38783) for the unsymmetrical ones; with a principal characteristic group the group is the suffix and
the azo unit a 'diazenyl' prefix ('4-(phenyldiazenyl)benzene-1-sulfonic acid (PIN)',:38798).

RDKit only: no naming code, no OPSIN. The module is in the spelling package because the checks of that
package read RDKit and the name only (``tests/unit/validation/test_pin_spelling_interface.py``); the
producer imports it from here, so the rule has the one home.
"""
from rdkit import Chem

_NN_UNIT = Chem.MolFromSmarts("[#7;!R]-[#7;!R]")
_NN_DOUBLE = Chem.MolFromSmarts("[#7;!R]=[#7;!R]")
_RING_N = Chem.MolFromSmarts("[#7;R]")
_N_THREE = Chem.MolFromSmarts("[#7]~[#7]~[#7]")
#: groups that are no characteristic group of the classes 1 to 20 of (:18162-:18219):
#: * the nitro group and the nitroso group on carbon, which are prefixes only: (:25607) "Diazo compounds,
#: nitro and nitroso compounds... are also included in this Section", with the characteristic groups "denoted,
#: in substitutive nomenclature, only as prefixes" (:25609), and (:25933) "Compounds containing the
#: -NO2 or -NO group are named by means of the prefixes 'nitro' and 'nitroso'" (the nitroso group on N or O is
#: a derivative of nitrous acid,, and is not read here);
#: * the sulfoxide and the sulfone, which are classes 41 (:18217);
#: * the peroxide R-O-O-R' and its chalcogen analogues, R-S-S-R', R-Se-Se-R', R-Te-Te-R' and the mixed ones
#: R-O-S-R'...: class 42 (:18218) "Peroxides, then chalcogen analogues with the greater number of the senior
#: chalcogen atom, where O > S > Se > Te", below the class of nitrogen, 21 (:18197). A chain of more than two
#: chalcogen atoms and a chalcogen atom with a hydrogen (-OOH is class 18, a hydroperoxide) are no such group
_NO_CLASS_GROUPS = [Chem.MolFromSmarts(x) for x in (
    "[#6][NX3](=O)=O", "[#6][N+](=O)[O-]", "[#6][NX2;+0]=[OX1;+0]",
    "[#6][SX3](=O)[#6]", "[#6][SX4](=O)(=O)[#6]",
    "[#6][O,S,Se,Te;X2;+0][O,S,Se,Te;X2;+0][#6]")]
#: B, Al, Si, P, Ga, Ge, As, In, Sn, Sb, Tl, Pb, Bi: the atoms of a parent hydride (a borane, a silane, a
#: phosphane, a stannane, a ring of them), which the class of nitrogen outranks in: "N > P > As >
#: Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C" (:18917), and (:18919)
#: '1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine (PIN) (N > Si > O)' (:18950). They are no characteristic group
#: while they have single (or aromatic) bonds to carbon, to hydrogen and to one another only
_PARENT_HYDRIDE_ATOMS = frozenset({5, 13, 14, 15, 31, 32, 33, 49, 50, 51, 81, 82, 83})


def _is_hydrazine_n(a) -> bool:
    """A neutral nitrogen of a hydrazine unit: three single bonds (hydrogen counted), in no ring, one
    of them to another nitrogen, the rest to carbon or hydrogen."""
    return not (a.IsInRing() or a.GetDegree() + a.GetTotalNumHs() != 3
                or any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds())
                or not any(n.GetAtomicNum() == 7 for n in a.GetNeighbors())
                or any(n.GetAtomicNum() not in (6, 7) for n in a.GetNeighbors()))


def _is_diazene_n(a) -> bool:
    """A neutral nitrogen of a diazene unit R-N=N-R': in no ring, a double bond to the other nitrogen
    and one more valence, a single bond to carbon or a hydrogen (R-N=NH, the diazene itself)."""
    if a.IsInRing():
        return False
    to_n = [b for b in a.GetBonds() if b.GetOtherAtom(a).GetAtomicNum() == 7]
    to_c = [b for b in a.GetBonds() if b.GetOtherAtom(a).GetAtomicNum() == 6]
    if (len(to_n) != 1 or to_n[0].GetBondType() != Chem.BondType.DOUBLE
            or a.GetDegree() != len(to_n) + len(to_c) or len(to_c) + a.GetTotalNumHs() != 1):
        return False
    return all(b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC) for b in to_c)


def _only_unit_is_functional(mol, unit_n) -> bool:
    """True when every atom of the molecule belongs to a hydrocarbon skeleton, a halogen on carbon,
    an ether or sulfide (class 41, with selenides and tellurides), a no-class group of
    ``_NO_CLASS_GROUPS``, an atom of ``_PARENT_HYDRIDE_ATOMS`` (neutral, single or aromatic bonds to
    carbon and to atoms of that set only, hydrogen for the rest), or a nitrogen of the unit that
    ``unit_n`` (``_is_hydrazine_n`` or ``_is_diazene_n``) recognises. Then no characteristic group of
    the classes 1 to 20 of is present: every one of them has a carbonyl, thiocarbonyl,
    imino, hydroxy, sulfanyl, peroxy, nitrile, acid (P=O, B-OH, Si-OH included) or amino (an N-C bond
    without an N-N bond) atom, none of which this allows. Stated by what is allowed, so a group the
    reader does not know can only make it decline."""
    claimed = {i for pat in _NO_CLASS_GROUPS for m in mol.GetSubstructMatches(pat) for i in m}
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons():
            return False
        if a.GetIdx() in claimed:
            continue
        if a.GetFormalCharge() or a.GetAtomicNum() == 0:
            return False
        z, nbrs = a.GetAtomicNum(), a.GetNeighbors()
        if z == 6:
            continue
        if z in (9, 17, 35, 53):
            if any(n.GetAtomicNum() != 6 for n in nbrs):
                return False
        elif z in (8, 16, 34, 52):
            if a.GetDegree() != 2 or any(n.GetAtomicNum() != 6 for n in nbrs):
                return False
        elif z in _PARENT_HYDRIDE_ATOMS:
            if (any(b.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC)
                    for b in a.GetBonds())
                    or any(n.GetAtomicNum() != 6 and n.GetAtomicNum() not in _PARENT_HYDRIDE_ATOMS
                           for n in nbrs)):
                return False
        elif z == 7:
            if not unit_n(a):
                return False
        else:
            return False

    def bond_ok(b):
        ends = (b.GetBeginAtom(), b.GetEndAtom())
        return (b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC)
                or {x.GetAtomicNum() for x in ends} == {6}
                or {x.GetIdx() for x in ends} <= claimed
                # the N=N bond of a diazene unit: both nitrogens passed ``_is_diazene_n`` above
                or (unit_n is _is_diazene_n and {x.GetAtomicNum() for x in ends} == {7}))
    return all(bond_ok(b) for b in mol.GetBonds())


def only_hydrazine_is_functional(mol) -> bool:
    """:func:`_only_unit_is_functional` for the hydrazine unit (an acyclic N-N bond, one or more units)."""
    return _only_unit_is_functional(mol, _is_hydrazine_n)


def hydrazine_is_senior_parent(mol) -> bool:
    """True when ``mol`` is one component with an acyclic N-N bond, no nitrogen ring (a ring of
    nitrogen is a heterocycle, which precedes the hydrazines in class 21), no N-N-N chain (a
    polyazane precedes a hydrazine) and no characteristic group of the classes 1 to 20
    (:func:`only_hydrazine_is_functional`): the hydrazine is then the parent hydride."""
    return (mol.HasSubstructMatch(_NN_UNIT)
            and not mol.HasSubstructMatch(_RING_N)
            and not mol.HasSubstructMatch(_N_THREE)
            and len(Chem.GetMolFrags(mol)) == 1
            and only_hydrazine_is_functional(mol))


def diazene_is_senior_parent(mol) -> bool:
    """True when ``mol`` is one component whose only nitrogen atoms are the two of ONE acyclic azo unit
    R-N=N-R' (carbon on both), with no nitrogen ring, no N-N-N chain and no characteristic group of the
    classes 1 to 20: the diazene is then the parent hydride;,:38759). A
    molecule with two azo groups is the bis(azo) case of (:38815), which is not decided
    here."""
    if (not mol.HasSubstructMatch(_NN_DOUBLE) or mol.HasSubstructMatch(_RING_N)
            or mol.HasSubstructMatch(_N_THREE) or len(Chem.GetMolFrags(mol)) != 1
            or not _only_unit_is_functional(mol, _is_diazene_n)):
        return False
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and _is_diazene_n(a)) == 2


def polyazane_is_senior_parent(mol) -> bool:
    """True when the hydrazine or the diazene is the parent hydride of ``mol``."""
    return hydrazine_is_senior_parent(mol) or diazene_is_senior_parent(mol)


__all__ = ["diazene_is_senior_parent", "hydrazine_is_senior_parent",
           "only_hydrazine_is_functional", "polyazane_is_senior_parent"]
