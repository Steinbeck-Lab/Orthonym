"""The shared indicated / added hydrogen rule for a ring system with =X groups
(``orthonym.rules.ring_hydrogen``), checked on the Blue Book's own PIN examples.

Each structure is written with atom-map numbers that are the PIN locants (fusion
locants such as 4a are mapped through ``LETTERS``), so the test needs no engine
numbering. The expected text is the hydrogen part of the printed PIN:

- '4H-pyran-4-one (PIN) pyran-4-one' (the Blue Book), '1H-inden-1-one (PIN)'
  (:28416), 'naphthalen-1(2H)-one (PIN)' (:28418), 'pyridin-2(1H)-one (PIN)' (:21316),
  '1H-pyrrole-2,5-dione (PIN)' (:33843), 'naphthalene-1,2-dione' (:3753,,
  '7H-1-benzopyran-7-one (PIN)' (:24776), '2H,7H-pyrano[2,3-b]pyran-2,7-dione (PIN)'
  (:24782), '1,2-dihydro-3H-indol-3-one (PIN)' (:29342), '1H-cyclopenta[a]naphthalene-
  1,2(3H)-dione (PIN)' (:24822), '4-iminonaphthalen-1(4H)-one (PIN)' (:24870).
- Derived by (:24768) / (:24794) / (:24689), the
  book prints no PIN of these parents with a suffix: '6H-pyrido[1,2-b]pyridazin-6-one',
  '4H-quinolizin-4-one', 'indolizin-3(2H)-one' (the bridgehead N4 carries no hydrogen,
   :8320 "carrying one or more hydrogen atoms"), '2H-1,3-dioxol-2-one',
  'oxepin-3(2H)-one', '2H,4H-1,3-dioxin-4-one', '7,8-dihydro-2H,5H-pyrano[4,3-b]pyran-2-one',
  '5-methylidene-5H-pyrrolo[3,2-b]pyridine' (a prefix =X is an ordinary saturated position,
   :24864).
"""
import pytest
from rdkit import Chem

from orthonym.rules import ring_hydrogen as rh

LETTERS = {31: "3a", 41: "4a", 51: "5a", 71: "7a", 81: "8a", 91: "9a", 92: "9b"}


def _read(smiles):
    mol = Chem.MolFromSmiles(smiles)
    a2l = {}
    for at in mol.GetAtoms():
        m = at.GetAtomMapNum()
        if m:
            a2l[at.GetIdx()] = LETTERS.get(m, m)
    for at in mol.GetAtoms():
        at.SetAtomMapNum(0)
    return mol, a2l


def _place(smiles, suffix_locants):
    mol, a2l = _read(smiles)
    ring = set(a2l)
    suffix = {a for a, loc in a2l.items() if loc in suffix_locants}
    got = rh.ring_hydrogen(mol, ring, suffix, a2l)
    assert got is not None, smiles
    return (rh.parent_prefix(got, a2l), rh.added_text(got, a2l))


CASES = [
    # (structure with PIN locants as atom maps, suffix locants, front text, added text)
    ("O=[C:4]1[CH:3]=[CH:2][O:1][CH:6]=[CH:5]1", {4}, "4H-", ""),                  #:28414
    ("O=[C:1]1[CH:2]=[CH:3][c:31]2[cH:4][cH:5][cH:6][cH:7][c:71]12", {1}, "1H-", ""),  #:28416
    ("O=[C:1]1[CH2:2][CH:3]=[CH:4][c:41]2[cH:5][cH:6][cH:7][cH:8][c:81]12", {1}, "", "2H"),  #:28418
    ("O=[c:2]1[cH:3][cH:4][cH:5][cH:6][nH:1]1", {2}, "", "1H"),                    #:21316
    ("O=[C:2]1[CH:3]=[CH:4][C:5](=O)[NH:1]1", {2, 5}, "1H-", ""),                  #:33843
    ("O=[C:1]1[C:2](=O)[CH:3]=[CH:4][c:41]2[cH:5][cH:6][cH:7][cH:8][c:81]12", {1, 2}, "", ""),  #:3753
    ("[O:1]1[CH:2]=[CH:3][CH:4]=[C:41]2[CH:5]=[CH:6][C:7](=O)[CH:8]=[C:81]12", {7}, "7H-", ""),  #:24776
    ("O=[c:2]1[cH:3][cH:4][c:41]2[cH:5][cH:6][c:7](=O)[o:8][c:81]2[o:1]1", {2, 7}, "2H,7H-", ""),  #:24782
    ("O=[C:3]1[CH2:2][NH:1][c:71]2[cH:7][cH:6][cH:5][cH:4][c:31]12", {3}, "1,2-dihydro-3H-", ""),  #:29342
    ("O=[C:1]1[C:2](=O)[CH2:3][C:31]2=[CH:4][CH:5]=[C:51]3[CH:6]=[CH:7][CH:8]=[CH:9][C:91]3=[C:92]12",
     {1, 2}, "1H-", "3H"),                                                          #:24822
    ("O=[C:1]1[CH:2]=[CH:3][C:4](=N)[c:41]2[cH:5][cH:6][cH:7][cH:8][c:81]12", {1}, "", "4H"),  #:24870
    ("O=[c:6]1[cH:7][cH:8][n:9]2[n:1][cH:2][cH:3][cH:4][c:41]2[cH:5]1", {6}, "6H-", ""),
    ("O=[c:4]1[cH:3][cH:2][cH:1][c:91]2[cH:9][cH:8][cH:7][cH:6][n:5]12", {4}, "4H-", ""),
    ("O=[C:3]1[CH2:2][CH:1]=[C:81]2[CH:8]=[CH:7][CH:6]=[CH:5][N:4]12", {3}, "", "2H"),
    ("O=[c:2]1[o:1][cH:5][cH:4][o:3]1", {2}, "2H-", ""),
    ("O=[C:3]1[CH2:2][O:1][CH:7]=[CH:6][CH:5]=[CH:4]1", {3}, "", "2H"),
    ("O=[C:4]1[CH:5]=[CH:6][O:1][CH2:2][O:3]1", {4}, "2H,4H-", ""),
    ("O=[c:2]1[cH:3][cH:4][c:41]2[CH2:5][O:6][CH2:7][CH2:8][c:81]2[o:1]1", {2}, "7,8-dihydro-2H,5H-", ""),
    ("[CH2]=[C:5]1[CH:6]=[CH:7][C:71]2=[N:1][CH:2]=[CH:3][C:31]2=[N:4]1", set(), "5H-", ""),
]


@pytest.mark.parametrize("smiles,suffix,front,added", CASES)
def test_the_book_hydrogen_of_a_ring_system_with_a_group(smiles, suffix, front, added):
    assert _place(smiles, suffix) == (front, added)


def test_a_charged_ring_atom_declines():
    mol = Chem.MolFromSmiles("O=c1ccc2cccc[n+]2[n-]1")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    a2l = {a: i + 1 for i, a in enumerate(sorted(ring))}
    carbonyl = {a.GetIdx() for a in mol.GetAtoms()
                if a.IsInRing() and any(n.GetSymbol() == "O" for n in a.GetNeighbors())}
    assert rh.ring_hydrogen(mol, ring, carbonyl, a2l) is None


def test_exocyclic_double_atoms_reads_the_kekule_bond():
    mol = Chem.MolFromSmiles("O=c1ccn2ncccc2c1")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert rh.exocyclic_double_atoms(mol, ring) == frozenset({1})
