"""Suffix groups and prefixes of a phane,,.

the Blue Book, each line read with ``sed -n``:

* (:18875): "The senior parent structure has the maximum number of substituents
  corresponding to the principal characteristic group (suffix)": the principal groups are the
  occurrences of the senior suffix class that ``validation.spelling.linear_phane.principal_groups``
  reads (one home of which atoms form which suffix class).
* (c) (:3256): the principal characteristic groups take the lowest locants; (a)
  (:7595) the terminal 'e' is elided before a suffix that starts with a vowel.
* (:15111): "locants for amplificant atoms are constructed by citing the actual locants of
  the amplificant as superscripts to the locant of the superatom that represents the amplificant in
  the simplified skeleton."

A suffix group is built only when it is its functional atoms and nothing more (a hydroxy group, a
primary amine or amide, a carboxylic acid, a nitrile, an aldehyde, a ketone on a chain atom). Every
atom outside the skeleton and the suffix groups belongs to a substituent prefix named by the engine's
substituent namer. A structure with a principal group that the phane does not express is no phane
parent and is not named here.

A hydroxy or amino group on a skeletal atom other than carbon is a principal group like any other
 "Heterols",:27230: "When the hydroxy group is attached to an atom other than carbon, hydroxy
compounds belong to a compound class called heterols. They are classified as hydroxy compounds and named
using the suffix 'ol'... Names formed using a suffix are preferred to those formed by means of the prefix
'hydroxy'";,:26213: "When attached to heteroatoms, amino groups are expressed as suffixes when
representing the principal characteristic group"). The suffix is not built: such a group is named as a prefix
only when a group of a senior class is the principal one, and the structure is not named otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple

from rdkit import Chem

from ...validation.spelling.linear_phane import principal_groups
from .skeleton import PhaneSkeleton


@dataclass(frozen=True)
class Suffix:
    core: str                   # 'one', 'ol', 'amine', 'carboxylic acid', 'carbonitrile', 'carbaldehyde', 'carboxamide'
    atom: int                   # the skeleton atom whose locant the suffix takes
    group: FrozenSet[int]       # the atoms of the group outside the skeleton


@dataclass(frozen=True)
class Prefix:
    name: str
    atom: int                   # the skeleton atom the prefix is attached to


@dataclass(frozen=True)
class Plan:
    suffixes: Tuple[Suffix, ...]
    prefixes: Tuple[Prefix, ...]


def _terminal(atom: Chem.Atom, symbol: str, order: Chem.BondType, hydrogens: int) -> bool:
    """A heavy atom bonded to one atom only, by a bond of type ``order``, with ``hydrogens`` hydrogens."""
    if atom.GetSymbol() != symbol or atom.GetDegree() != 1 or atom.GetTotalNumHs() != hydrogens:
        return False
    return atom.GetBonds()[0].GetBondType() == order


def _carbon_group(mol: Chem.Mol, centre: int, skeleton: Set[int], wanted: List[Tuple[str, Chem.BondType, int]]
                  ) -> Optional[Tuple[int, FrozenSet[int]]]:
    """A carbon atom whose other neighbours are exactly the terminal atoms ``wanted`` (symbol, bond type,
    hydrogens) and one skeleton atom: the skeleton atom and the atoms of the group (the carbon included)."""
    atom = mol.GetAtomWithIdx(centre)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() or atom.GetDegree() != len(wanted) + 1:
        return None
    rest = list(atom.GetNeighbors())
    group = {centre}
    for symbol, order, hydrogens in wanted:
        for n in rest:
            bond = mol.GetBondBetweenAtoms(centre, n.GetIdx())
            if (_terminal(n, symbol, order, hydrogens) and bond.GetBondType() == order
                    and n.GetIdx() not in group):
                group.add(n.GetIdx())
                break
        else:
            return None
    hold = [n.GetIdx() for n in rest if n.GetIdx() not in group]
    if len(hold) != 1 or hold[0] not in skeleton:
        return None
    if mol.GetBondBetweenAtoms(centre, hold[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    return hold[0], frozenset(group)


_S, _D, _T = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE


def _suffix_of(mol: Chem.Mol, cls: str, centre: int, skeleton: Set[int], chain: Set[int]) -> Optional[Suffix]:
    """The suffix an occurrence of the principal class gives, or None when it is no plain group of a
    skeleton atom."""
    atom = mol.GetAtomWithIdx(centre)
    if cls == "ketone":
        if centre not in chain or atom.GetSymbol() != "C":
            return None
        oxo = [n for n in atom.GetNeighbors() if _terminal(n, "O", _D, 0)]
        if len(oxo) != 1:
            return None
        return Suffix("one", centre, frozenset({oxo[0].GetIdx()}))
    if cls in ("hydroxy", "amine"):
        symbol, hydrogens, core = ("O", 1, "ol") if cls == "hydroxy" else ("N", 2, "amine")
        if not _terminal(atom, symbol, _S, hydrogens) or atom.GetFormalCharge():
            return None
        hold = atom.GetNeighbors()[0].GetIdx()
        if hold not in skeleton or mol.GetAtomWithIdx(hold).GetSymbol() != "C":
            return None
        return Suffix(core, hold, frozenset({centre}))
    table = {
        "acid": ("carboxylic acid", [("O", _D, 0), ("O", _S, 1)]),
        "nitrile": ("carbonitrile", [("N", _T, 0)]),
        "aldehyde": ("carbaldehyde", [("O", _D, 0)]),
        "amide": ("carboxamide", [("O", _D, 0), ("N", _S, 2)]),
    }
    if cls not in table:
        return None
    core, wanted = table[cls]
    if cls == "aldehyde" and atom.GetTotalNumHs() != 1:
        return None
    found = _carbon_group(mol, centre, skeleton, wanted)
    if found is None:
        return None
    hold, group = found
    return Suffix(core, hold, group)


#: Table 4.1 class numbers of the hydroxy (alcohols,:18187) and the amino (amines,:18192) classes, as the reader ranks them
_HYDROXY_CLASS, _AMINO_CLASS = 17, 19
#: the skeletal atoms whose hydroxy or amino group may be a prefix when a senior class decides
_HETERO_PREFIX_OK = frozenset({"N", "Si", "Ge", "Sn", "Pb"})


def _heteroatom_groups_junior(mol: Chem.Mol, skeleton: Set[int], senior_class: Optional[int]) -> bool:
    """False when a hydroxy or amino group sits on a skeletal atom other than carbon and could be a principal
    group (the reader does not see these groups); True when there is none, or when a class senior to theirs
    is the principal one and the atom is one whose group is then a plain prefix."""
    for a in skeleton:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() == "C":
            continue
        for n in atom.GetNeighbors():
            if n.GetIdx() in skeleton or n.GetDegree() != 1:
                continue
            if mol.GetBondBetweenAtoms(a, n.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                continue
            if n.GetSymbol() in ("O", "S", "Se", "Te") and n.GetTotalNumHs() == 1:
                group_class = _HYDROXY_CLASS
            elif n.GetSymbol() == "N" and n.GetTotalNumHs() == 2:
                group_class = _AMINO_CLASS
            else:
                continue
            if senior_class is None or senior_class >= group_class or atom.GetSymbol() not in _HETERO_PREFIX_OK:
                return False
    return True


def plan_substituents(skel: PhaneSkeleton) -> Optional[Plan]:
    """The suffix groups and the named prefixes of ``skel``, or None when a group or a prefix is not built
    or an atom is left over."""
    from ...assembly.substituent_enumerator import discover_substituents, name_substituent
    from ...errors import is_failure_name, is_refusal_sentinel
    mol = skel.mol
    skeleton = {a for amp in skel.amplificants for a in amp.atoms} | {n[1] for n in skel.nodes if n[0] == "atom"}
    chain = {n[1] for n in skel.nodes if n[0] == "atom"}
    suffixes: List[Suffix] = []
    cls, occurrences = principal_groups(mol)
    if not _heteroatom_groups_junior(mol, skeleton, occurrences[0][0][0] if occurrences else None):
        return None
    if cls is not None:
        for occurrence in occurrences:
            suffix = _suffix_of(mol, cls, occurrence[3], skeleton, chain)
            if suffix is None:
                return None
            suffixes.append(suffix)
        if len({s.core for s in suffixes}) != 1 or len({s.atom for s in suffixes}) != len(suffixes):
            return None
    group = set().union(*(s.group for s in suffixes)) if suffixes else set()
    prefixes: List[Prefix] = []
    covered = set(group)
    if skel.other - group:
        parent = skeleton | group
        found = discover_substituents(mol, parent, parent_type="chain", principal_chain=sorted(parent),
                                      atom_to_locant={a: a for a in parent}, general_fallback=True)
        if found is None:
            return None
        for sub in found:
            fragment = set(sub.frag_atoms)
            bonded = [n.GetIdx() for n in mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                      if n.GetIdx() in fragment]
            if not bonded or sub.attach_mol_idx not in skeleton:
                return None
            name = name_substituent(mol, fragment, bonded[0])
            if not name or is_refusal_sentinel(name) or is_failure_name(name):
                return None
            prefixes.append(Prefix(name, sub.attach_mol_idx))
            covered |= fragment
    if covered != {a for a in skel.other if mol.GetAtomWithIdx(a).GetAtomicNum() != 1}:
        return None
    return Plan(tuple(suffixes), tuple(prefixes))
