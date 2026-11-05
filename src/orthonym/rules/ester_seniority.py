"""Which ester of a polyester is the one its functional-class name is built on.

 "Polyester names formed by using substitutive nomenclature, and/or multiplicative nomenclature, and
functional class nomenclature" (the Blue Book): "When required, seniority of rings, ring systems, and chains;
number and location of substituents; and alphanumerical order is applied in forming the alcoholic component of a
functional class ester name (see through ". The examples fix the order of the criteria:

- the acid component: 'phenyl 3-(benzoyloxy)benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate; the substituted
  benzoic acid is senior to the unsubstituted one; see ]' (more substituents); '4-[(3-ethoxy-3-oxopropanoyl)oxy]
  phenyl methyl butanedioate (PIN) (not ethyl 4-[(4-methoxy-4-oxobutanoyl)oxy]phenyl propanedioate; butanedioic acid is
  preferred to propanedioic acid' (the longer chain), with "Systems composed of rings and chains" (a ring is
  senior to a chain);
- then the alcohol component: '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN) (not 2-[2-(acetyloxy)phenyl]ethyl acetate;
  the ring is senior to the chain)', '2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN) (not 2-{2-[(acetyloxy)
  ethyl]phenyl}methyl acetate; the ethyl chain is senior to the methyl chain)'. The chain of a substituent group is
  its longest one, "THE PRINCIPAL SUBSTITUENT CHAIN" (b): "the greater number of skeletal atoms, i.e., the longest
  chain" (:22632), so the alcohol chain runs through the attachment carbon ('propan-2-yl', not 'ethyl').

Alphanumerical order decides only what these leave tied; it is the caller's last criterion. Of two rings the senior is
that of "SENIORITY ORDER ONLY FOR RINGS AND RING SYSTEMS" (:19402), (:19408) "The senior ring or ring system:
(a) is a heterocycle; (b) has at least one nitrogen atom; (c) has at least one heteroatom (in the absence of nitrogen) that
occurs earlier in the following sequence: F > Cl > Br > I > O > S >...; (d) has the greater number of rings; (e) has the
greater number of skeletal atoms; (f) has the greater number of heteroatoms of any kind; (g) has the greater number of
heteroatoms occurring earlier in the sequence", then the criteria of that decide between rings
(``rules.ring_selection.ring_system_score``); rings that tie on it go by their substituents:
'6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate (PIN) (not 4-[5-(acetyloxy)pyridin-2-yl]phenyl acetate; the nitrogenous ring is
senior to the carbocyclic ring)' (:31895). What this does not model: the heteroatoms of a chain and its unsaturation.
"""
from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

#: (skeletal atoms, sum of the heavy-atom degrees of those atoms) of an acyclic carbon chain: longer, then more branches
_Arm = Tuple[int, int]


def _heavy_degree(atom) -> int:
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)


def _arm(mol, idx: int, parent: int) -> _Arm:
    """The best acyclic carbon chain that starts at ``idx`` and runs away from ``parent``. The acyclic carbon atoms of
    a molecule form a forest, so a walk away from its start cannot meet itself."""
    atom = mol.GetAtomWithIdx(idx)
    best: _Arm = (0, 0)
    for nb in atom.GetNeighbors():
        if nb.GetIdx() != parent and nb.GetAtomicNum() == 6 and not nb.IsInRing():
            best = max(best, _arm(mol, nb.GetIdx(), idx))
    return (1 + best[0], _heavy_degree(atom) + best[1])


def _ring_system(mol, idx: int) -> Set[int]:
    """The atoms of the ring system that holds ``idx`` (the atoms joined to it by ring bonds)."""
    system, frontier = {idx}, [idx]
    while frontier:
        here = frontier.pop()
        for bond in mol.GetAtomWithIdx(here).GetBonds():
            other = bond.GetOtherAtomIdx(here)
            if bond.IsInRing() and other not in system:
                system.add(other)
                frontier.append(other)
    return system


def _ring_component(mol, ring_atom: int) -> tuple:
    """``(0, score of the ring system, -substituents)``: a ring ranks before every chain (1), the Blue Book:
    19336: "Within the same class, a ring or ring system has seniority over a chain"); two rings by the engine's reading of
    " SENIORITY ORDER ONLY FOR RINGS AND RING SYSTEMS" (:19402), (a) to (g) (:19408: a heterocycle, a
    nitrogen atom, the senior heteroatom, the greater number of rings, of skeletal atoms, of heteroatoms, of the senior
    heteroatoms), then the kinds of ring system of and the criteria of that decide between rings
    (``rules.ring_selection.ring_system_score``, the one the principal amide of ``rules.amide_parent`` is ranked by); rings
    of one score by the number of their substituents. The parent hydride comes before the number of
    substituents INTRODUCTION,:21551: "Two or more names may result based on the same senior parent structure
    selected according to, because of different substitution patterns"), so a benzoic acid with more substituents
    is not senior to a naphthalenecarboxylic acid with fewer ('naphthalene-2-carboxylate' before 'benzoate': the greater
    number of rings, (d), is element [3] of the score). The ester group itself is one of the heavy atoms bonded
    to the ring system, so it is not counted."""
    from .ring_selection import ring_system_score
    system = _ring_system(mol, ring_atom)
    outside = sum(1 for i in system for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                  if nb.GetIdx() not in system and nb.GetAtomicNum() > 1)
    return (0, ring_system_score(mol, system), -(outside - 1))


def _chain_component(length: int, degrees: int, own_atoms: int) -> tuple:
    """``(1, -skeletal atoms, -substituents)``: each bond of the chain takes two of the degrees, the ``own_atoms`` heavy
    atoms bonded to the chain are those of the ester group itself, and every other neighbour is a substituent."""
    return (1, -length, -(degrees - 2 * (length - 1) - own_atoms))


def _acid_component(mol, acyl_c: int, ester_o: int) -> tuple:
    """The acid half: a ring acid when the acyl carbon is bonded to a ring carbon, else the acyl chain, whose first
    skeletal atom is the acyl carbon (its =O and the ester oxygen are the ester group)."""
    acyl = mol.GetAtomWithIdx(acyl_c)
    ring_nb = next((nb.GetIdx() for nb in acyl.GetNeighbors() if nb.GetAtomicNum() == 6 and nb.IsInRing()), None)
    if ring_nb is not None:
        return _ring_component(mol, ring_nb)
    length, degrees = _arm(mol, acyl_c, ester_o)
    return _chain_component(length, degrees, 2)


def _alcohol_component(mol, alkyl_c: int, ester_o: int) -> tuple:
    """The alcohol half: the ring system of the attachment atom, else the longest acyclic carbon chain THROUGH the
    attachment carbon (b)), which takes its two best arms."""
    attach = mol.GetAtomWithIdx(alkyl_c)
    if attach.IsInRing():
        return _ring_component(mol, alkyl_c)
    arms = sorted((_arm(mol, nb.GetIdx(), alkyl_c) for nb in attach.GetNeighbors()
                   if nb.GetIdx() != ester_o and nb.GetAtomicNum() == 6 and not nb.IsInRing()), reverse=True)[:2]
    length = 1 + sum(a[0] for a in arms)
    degrees = _heavy_degree(attach) + sum(a[1] for a in arms)
    return _chain_component(length, degrees, 1)


def ester_seniority_key(mol, ester_match: Sequence[int]) -> tuple:
    """The seniority of one ester of ``mol`` as the parent of its functional-class name; the smaller key is the senior
    ester. ``ester_match`` is the ester SMARTS match ``(carbonyl C, =O, ester O, alkyl C)``. The key is
    ``(acid component, alcohol component)``, each ``(ring before chain, longer chain, more substituents)``."""
    acyl_c, _carbonyl_o, ester_o, alkyl_c = ester_match[:4]
    return (_acid_component(mol, acyl_c, ester_o), _alcohol_component(mol, alkyl_c, ester_o))


def senior_ester(mol, ester_matches: Sequence[Sequence[int]]) -> Optional[Sequence[int]]:
    """The most senior of ``ester_matches``; the first of those that tie (the atom order of the input). A lactone (its
    C(=O)-O bond is a ring bond) is no candidate beside an acyclic ester: "Cyclic esters"
    (the Blue Book) "For these compounds heterocyclic names are preferred IUPAC names", so it has no alcohol
    component to be the parent of a functional class name and stays a heterocyclic pseudoketone (1)); the
    key ranked its ring acid before every chain acid."""
    acyclic = [m for m in ester_matches if not mol.GetBondBetweenAtoms(m[0], m[2]).IsInRing()]
    return min(acyclic or ester_matches, key=lambda m: ester_seniority_key(mol, m), default=None)
