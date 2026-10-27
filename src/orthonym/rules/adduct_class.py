"""The class of an adduct component that has no principal characteristic group, and the order it gives.

 Organic adducts (the Blue Book): "the individual components are cited in the order of seniority of classes
(see ". A component with no suffix class (an acid, an amine...) belongs to a class "denoted by the senior atom in
heterane nomenclature" of (:18158): 21 nitrogen compounds ("heterocycles, polyazanes, hydrazines...,
diazenes, hydroxylamines, azanes",:18197), 22 to 34 the compounds of P, As, Sb, Bi, Si (26, "silanes",:18202), Ge, Sn,
Pb, B, Al, Ga, In, Tl, 35 to 38 the compounds of O, S, Se, Te ("heterocycles, polyoxidanes... but not peroxides or
ethers",:18211), 40 "Carbon compounds: rings, chains" (:18216), 41 "Ethers, then sulfides, sulfoxides, sulfones" (:18217)
and 42 peroxides (:18218). Within a class the heterocycles and carbocycles stand before the chains, and the rings by
 (a)-(g) (:19408), then, when those tie, by "The senior ring, ring system or principal chain has the
greater number of multiple bonds;... mancude rings or ring systems are considered as consisting of noncumulative double
bonds" (:21033), which puts 'benzene' before 'cyclohexane'.
"""
from __future__ import annotations

from typing import List, Set

from rdkit import Chem

#: the skeletal atoms of classes 21 to 38 of (:18158), in the order of the classes
_HETERO_ORDER = ("N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl", "O", "S", "Se", "Te")
_CLASS = {sym: 21 + i for i, sym in enumerate(_HETERO_ORDER)}
_CARBON, _ETHER, _PEROXIDE, _NONE = 40, 41, 42, 99
_CHALCOGENS = ("O", "S", "Se", "Te")


def _ring_systems(mol) -> List[Set[int]]:
    """The ring systems of ``mol`` as atom sets: rings that share an atom are one system."""
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        rest = []
        for system in systems:
            if system & merged:
                merged |= system
            else:
                rest.append(system)
        systems = rest + [merged]
    return systems


def _acyclic_parent_classes(mol, ring_atoms: Set[int]) -> List[int]:
    """The classes of the acyclic heteroatom parents of ``mol``: a neutral nitrogen bonded to a neutral nitrogen (a
    hydrazine, a diazene), a neutral atom of P... Tl (a phosphane, a silane, a borane), and for the chalcogens 41
    where one links two atoms that are not chalcogens (an ether, a sulfide), 42 for a peroxide."""
    classes: List[int] = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if atom.GetIdx() in ring_atoms or atom.GetFormalCharge():
            continue
        if sym == "N":
            if any(n.GetSymbol() == "N" and not n.GetFormalCharge() for n in atom.GetNeighbors()):
                classes.append(_CLASS["N"])
        elif sym in _CHALCOGENS:
            neighbours = [n.GetSymbol() for n in atom.GetNeighbors()]
            if sym == "O" and neighbours.count("O") and len(neighbours) > 1:
                classes.append(_PEROXIDE)
            elif len(neighbours) > 1 and not any(s in _CHALCOGENS for s in neighbours):
                classes.append(_ETHER)
        elif sym in _CLASS:
            classes.append(_CLASS[sym])
    return classes


def _multiple_bonds(mol, system: Set[int]) -> int:
    """The number of multiple bonds of a ring system, the mancude rings as noncumulative double bonds."""
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        kekule = mol
    return sum(1 for b in kekule.GetBonds()
               if b.GetBondTypeAsDouble() >= 2 and b.GetBeginAtomIdx() in system and b.GetEndAtomIdx() in system)


def component_class_key(frag_mol) -> tuple:
    """The key of an organic component that has no principal characteristic group (smaller = cited first):
    ``(class of, 0 for a ring parent and 1 for none, the ring key)``. An ion stands first, ``(-1,)``: the
    cations (class 6) before the neutral classes, ordered by size as before."""
    from .bridged_fused_pin.selection import p44_2_1_key
    if Chem.GetFormalCharge(frag_mol):
        return (-1,)
    systems = _ring_systems(frag_mol)
    ring_atoms = set().union(*systems) if systems else set()

    def ring_class(system) -> int:
        classes = [_CLASS[frag_mol.GetAtomWithIdx(a).GetSymbol()] for a in system
                   if frag_mol.GetAtomWithIdx(a).GetSymbol() in _CLASS]
        return min(classes) if classes else _CARBON

    acyclic = _acyclic_parent_classes(frag_mol, ring_atoms)
    ring_classes = [ring_class(s) for s in systems]
    hetero = [c for c in ring_classes + acyclic if c < _CARBON]
    if hetero:
        cls = min(hetero)
    elif acyclic:
        cls = min(acyclic)
    elif any(a.GetAtomicNum() == 6 for a in frag_mol.GetAtoms()):
        cls = _CARBON
    else:
        cls = _NONE
    rings = [s for s in systems if ring_class(s) == cls]
    if not rings:
        return (cls, 1, ())
    return (cls, 0, min((p44_2_1_key(frag_mol, s), -_multiple_bonds(frag_mol, s)) for s in rings))
