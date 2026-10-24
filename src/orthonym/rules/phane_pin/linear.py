"""The skeleton of a linear phane (2),.

the Blue Book, each line read with ``sed -n``: (:14847) the numerical term counts the nodes, superatoms
included, and "No prefix is used to name linear phanes" (:14857); (:23901) "Phane nomenclature is used
to generate preferred IUPAC names for ring assemblies and linear acyclic/cyclic compounds that include a minimum of
seven nodes including at least four rings or ring systems, two of which must be terminal, even though the compounds
could also be named by substitutive or multiplicative nomenclature."

The chain is the one of the reader (``validation.spelling.linear_phane``): its ring systems are the amplificants,
its acyclic atoms the other nodes, every other atom of the molecule a substituent or a group of a suffix.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ...validation.spelling.linear_phane import Node as ReaderNode
from ...validation.spelling.linear_phane import ring_system_atoms
from .amplificants import amplificant_hydride
from .skeleton import Amplificant, Node, PhaneSkeleton, SkeletonKind, chain_atom_ok


def perceive_linear(mol: Chem.Mol, chain: Tuple[ReaderNode, ...]) -> Optional[PhaneSkeleton]:
    """The path skeleton of ``chain``, or None when a node is no allowed amplificant, a chain atom is not one a
    phane name expresses, two nodes are bonded twice, a node is bonded to a node of the chain that does not
    follow it (the chain closes a cycle: a cyclophane), an amplificant is attached twice to one atom or by a
    multiple bond, or an end node is not a ring system."""
    if len(chain) < 7 or chain[0][0] != "R" or chain[-1][0] != "R":
        return None
    atoms_of: List[frozenset] = []
    for node in chain:
        atoms_of.append(ring_system_atoms(mol, node[1]) if node[0] == "R" else frozenset({node[1]}))
    owner: Dict[int, int] = {a: i for i, atoms in enumerate(atoms_of) for a in atoms}
    if len(owner) != sum(len(a) for a in atoms_of):
        return None
    between: Dict[Tuple[int, int], Tuple[int, int, Chem.BondType]] = {}
    for bond in mol.GetBonds():
        a, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in owner or c not in owner or owner[a] == owner[c]:
            continue
        i, j = owner[a], owner[c]
        if abs(i - j) != 1 or (min(i, j), max(i, j)) in between:
            return None
        if i > j:
            i, j, a, c = j, i, c, a
        between[(i, j)] = (a, c, bond.GetBondType())
    if len(between) != len(chain) - 1:
        return None
    nodes: List[Node] = []
    amplificants: List[Amplificant] = []
    for i, node in enumerate(chain):
        if node[0] == "R":
            hydride = amplificant_hydride(mol, atoms_of[i])
            if hydride is None:
                return None
            nodes.append(("amp", len(amplificants)))
            amplificants.append(Amplificant(len(amplificants), atoms_of[i], hydride))
        elif chain_atom_ok(mol.GetAtomWithIdx(node[1])):
            nodes.append(("atom", node[1]))
        else:
            return None
    edge: Dict[Tuple[Node, Node], Tuple[int, int, Chem.BondType]] = {}
    for (i, j), (a, c, bond_type) in between.items():
        if bond_type not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return None
        if nodes[i][0] == "atom" and nodes[j][0] == "atom" and mol.GetBondBetweenAtoms(a, c).GetIsAromatic():
            return None
        if (nodes[i][0] == "amp" or nodes[j][0] == "amp") and bond_type != Chem.BondType.SINGLE:
            return None
        edge[(nodes[i], nodes[j])] = (a, c, bond_type)
        edge[(nodes[j], nodes[i])] = (c, a, bond_type)
    for i, node in enumerate(nodes):
        if node[0] == "amp" and 0 < i < len(nodes) - 1:
            if between[(i - 1, i)][1] == between[(i, i + 1)][0]:
                return None            # both bonds of the amplificant leave from one atom
    adj = {nodes[i]: tuple(nodes[j] for j in (i - 1, i + 1) if 0 <= j < len(nodes)) for i in range(len(nodes))}
    return PhaneSkeleton(mol, SkeletonKind.PATH, tuple(nodes), adj, edge, tuple(amplificants),
                         frozenset(range(mol.GetNumAtoms())) - set(owner))
