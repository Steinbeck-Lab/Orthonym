"""The C60 template (``data/fullerene_c60.py``) against the locant facts the Blue Book prints.

Each fact is checked by a walk over the table's own bonds, not by a second copy of the table, and the
sources are cited: NUMBERING OF FULLERENES (the Blue Book) and "Fullerenes and
modified fullerenes ortho fused to organic rings or ring systems" (:15464-15480).
"""
import collections

import pytest
from rdkit import Chem

from orthonym.data.fullerene_c60 import (C60_ATOMS, C60_DOUBLE_BONDS, C60_NEIGHBOURS,
                                         neighbour_pairs, template_mol)
from orthonym.perception.automorphisms import skeleton_automorphisms

pytestmark = pytest.mark.unit

BONDS = {frozenset(p) for p in neighbour_pairs()}
#: [60]fullerene as the aromatic SMILES the census rows use (an independent spelling of the same graph)
C60_PLAIN = ("c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7"
             "c3c3c8c4c4c9c5c1c1c6c2c3c41")


def _pentagons():
    found = set()

    def walk(path):
        if len(path) == 5:
            if path[0] in C60_NEIGHBOURS[path[-1]]:
                found.add(frozenset(path))
            return
        for n in C60_NEIGHBOURS[path[-1]]:
            if n not in path:
                walk(path + [n])

    for a in C60_NEIGHBOURS:
        walk([a])
    return found


def _is_56(a, b):
    """The bond lies in a pentagon: a [5,6] bond (a [6,6] bond lies in two hexagons)."""
    return any({a, b} <= p for p in PENTAGONS)


PENTAGONS = _pentagons()


def _distances(source):
    dist, queue = {source: 0}, collections.deque([source])
    while queue:
        cur = queue.popleft()
        for nb in C60_NEIGHBOURS[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def test_graph_is_a_truncated_icosahedron():
    # Fullerenes (the Blue Book): "twelve five-membered rings and the rest six-membered";
    # [60]fullerene: "a truncated icosahedron" (60 atoms, 90 bonds, 12 pentagons, 20 hexagons)
    assert C60_ATOMS == len(C60_NEIGHBOURS) == 60
    assert sorted(C60_NEIGHBOURS) == list(range(1, 61))
    assert all(len(set(ns)) == 3 and a not in ns for a, ns in C60_NEIGHBOURS.items())
    assert all(a in C60_NEIGHBOURS[b] for a, ns in C60_NEIGHBOURS.items() for b in ns)
    assert len(BONDS) == 90
    assert len(PENTAGONS) == 12
    assert sum(1 for b in BONDS if _is_56(*b)) == 60       # every pentagon edge once: 12 * 5
    # every atom lies in exactly one pentagon (the isolated-pentagon rule of the truncated icosahedron)
    assert all(sum(1 for p in PENTAGONS if a in p) == 1 for a in C60_NEIGHBOURS)


def test_numbering_is_continuous_and_starts_at_a_pentagon():
    #: "The objectives are to achieve a continuous numbering and use a well defined starting point"
    assert all(i + 1 in C60_NEIGHBOURS[i] for i in range(1, 60))
    assert frozenset({1, 2, 3, 4, 5}) in PENTAGONS and all(b in C60_NEIGHBOURS[a] for a, b in
                                                          zip((1, 2, 3, 4, 5), (2, 3, 4, 5, 1)))


@pytest.mark.parametrize("bond", [(1, 9), (2, 12), (7, 8)])
def test_bonds_the_book_fuses_are_66_bonds(bond):
    # (:15464-15480): 'cyclopropa[1,9]', '[1,3]dioxolo[4',5':1,9]', 'tris(oxireno)[2',3':1,9;
    # 2'',3'':2,12;2''',3''':7,8]' fuse on [6,6] bonds ("the fusion bond" of the fullerene is a double
    # bond of the cage)
    assert frozenset(bond) in BONDS and not _is_56(*bond)


def test_the_bonds_the_book_numbers_around_the_first_pentagon():
    # 'decanor' examples: '1,2,3,4,5,6,9,12,15,18-decanor' removes the pentagon 1-5 and the five
    # atoms bonded to it, 6, 9, 12, 15, 18; the [6,6] bond 7,8 stays beside them
    around = {n for a in (1, 2, 3, 4, 5) for n in C60_NEIGHBOURS[a]} - {1, 2, 3, 4, 5}
    assert around == {6, 9, 12, 15, 18}
    assert not _is_56(7, 8) and 7 in C60_NEIGHBOURS[6] and 8 in C60_NEIGHBOURS[9]
    for bond in [(3, 15), (4, 18), (5, 6)]:
        assert frozenset(bond) in BONDS and not _is_56(*bond)
    assert _is_56(1, 2) and _is_56(2, 3) and _is_56(1, 5)


def test_the_antipode_of_the_first_66_bond():
    # the bond 52,60 lies opposite 1,9: each atom's farthest atom (graph distance 9) is the other end's
    # partner (1 -> 60, 9 -> 52), and the bond is a [6,6] bond (trans-1 of the 1,9 adduct)
    far = {a: [b for b, d in _distances(a).items() if d == 9] for a in C60_NEIGHBOURS}
    assert all(len(v) == 1 for v in far.values())
    assert far[1] == [60] and far[9] == [52]
    assert frozenset((52, 60)) in BONDS and not _is_56(52, 60)


def test_kekule_structure_is_the_66_bonds():
    #: "each component retains its own bonding pattern": the cage's double bonds are the [6,6] bonds
    expected = {b for b in BONDS if not _is_56(*tuple(b))}
    assert {frozenset(b) for b in C60_DOUBLE_BONDS} == expected and len(expected) == 30
    assert sorted(a for b in C60_DOUBLE_BONDS for a in b) == list(range(1, 61))


def test_template_mol_agrees_with_the_tables():
    mol = template_mol()
    assert mol.GetNumAtoms() == 60 and all(a.GetAtomMapNum() == a.GetIdx() + 1 for a in mol.GetAtoms())
    bonds = {frozenset((b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1)): b.GetBondType()
             for b in mol.GetBonds()}
    assert set(bonds) == BONDS
    assert {b for b, t in bonds.items() if t == Chem.BondType.DOUBLE} == {frozenset(b) for b in C60_DOUBLE_BONDS}
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())
    plain = Chem.MolFromSmiles(C60_PLAIN)
    stripped = Chem.Mol(mol)
    for a in stripped.GetAtoms():
        a.SetAtomMapNum(0)
    assert Chem.MolToInchiKey(stripped) == Chem.MolToInchiKey(plain)    # C60, XMWRBQBLMFGWIX


def test_ih_has_120_symmetries():
    # (C60-Ih): the point group has 120 operations (60 rotations and 60 improper ones); each is a way to
    # number the cage, and the lowest locants decide among them
    auts, exhaustive = skeleton_automorphisms(template_mol(), set(range(60)), element_blind=False, cap=120)
    assert exhaustive and len(auts) == 120
    assert len({tuple(sorted(a.items())) for a in auts}) == 120


def test_the_66_bond_stabiliser_has_four_operations():
    # a [6,6] bond lies on the fixed points of four operations (the identity, a mirror holding the bond,
    # the mirror across it, the half turn), so 120 / 30 bonds = 4 numberings put it on 1,9
    auts, _ = skeleton_automorphisms(template_mol(), set(range(60)), element_blind=False, cap=120)
    index = {a: a + 1 for a in range(60)}
    onto = [s for s in auts if {index[s[0]], index[s[8]]} == {1, 9}]
    assert len(onto) == 4
    ends = collections.Counter((index[s[0]], index[s[8]]) for s in onto)
    assert ends == {(1, 9): 2, (9, 1): 2}
