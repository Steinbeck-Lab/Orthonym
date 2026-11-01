"""The skeleton of (C60-Ih)[5,6]fullerene with its systematic locants, as a fact table.

 NUMBERING OF FULLERENES (the Blue Book): "Systematic numbering is not yet a fully solved
issue in the nomenclature of fullerenes. The objectives are to achieve a continuous numbering and use a
well defined starting point for all fullerenes.... the systematic numbering used with IUPAC systematic
names is derived from the system developed by the Chemical Abstracts Service and that the two systems are
identical for these two specific fullerenes." The numbering is drawn in the book's figures only, so this
table is read off the contiguous spiral numbering (locant i bonded to locant i + 1, one pentagon first)
that reproduces every locant fact the book prints, and the tests pin each of them
(``tests/unit/data/test_fullerene_c60.py``):

* (the Blue Book-15480) "Fullerenes and modified fullerenes ortho fused to organic rings or
  ring systems": the [6,6] bonds 1,9, 2,12 and 7,8 of 'tris(oxireno)[2',3':1,9;2'',3'':2,12;2''',3''':7,8]
  (C60-Ih)[5,6]fullerene', and the bond 1,9 of '3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene' and
  '[1,3]dioxolo[4',5':1,9](C60-Ih)[5,6]fullerene';
* the 'decanor' example: the pentagon 1,2,3,4,5 and the atoms next to it, 6, 9, 12, 15 and 18
  ('1,2,3,4,5,6,9,12,15,18-decanor'), with the [6,6] bond 7,8 left beside it;
* (:25834) '1-(trifluoromethyl)-1,9-dihydro(C60-Ih)[5,6]fullerene' and (:26870)
  '(C60-Ih)[5,6]fulleren-1(9H)-ol': the bond 1,9 is a double bond of the mancude cage.

The Kekule structure is the one whose 30 double bonds are the 30 [6,6] bonds (the bonds that lie in no
pentagon); every cage atom has exactly one of them.
"""
from functools import lru_cache
from typing import Dict, Tuple

#: Systematic names (the Blue Book): the parent name of the bare fullerene
C60_NAME = "(C60-Ih)[5,6]fullerene"
#: the token every name of this skeleton holds: the parenthetical prefix, the ring sizes, the stem
C60_CORE_TOKEN = "(C60-Ih)[5,6]fulleren"
C60_ATOMS = 60

#: locant -> its three neighbours' locants
C60_NEIGHBOURS: Dict[int, Tuple[int, int, int]] = {
    1: (2, 5, 9), 2: (1, 3, 12), 3: (2, 4, 15), 4: (3, 5, 18), 5: (1, 4, 6),
    6: (5, 7, 20), 7: (6, 8, 22), 8: (7, 9, 25), 9: (1, 8, 10), 10: (9, 11, 26),
    11: (10, 12, 29), 12: (2, 11, 13), 13: (12, 14, 30), 14: (13, 15, 33), 15: (3, 14, 16),
    16: (15, 17, 34), 17: (16, 18, 37), 18: (4, 17, 19), 19: (18, 20, 38), 20: (6, 19, 21),
    21: (20, 22, 40), 22: (7, 21, 23), 23: (22, 24, 42), 24: (23, 25, 44), 25: (8, 24, 26),
    26: (10, 25, 27), 27: (26, 28, 45), 28: (27, 29, 47), 29: (11, 28, 30), 30: (13, 29, 31),
    31: (30, 32, 48), 32: (31, 33, 50), 33: (14, 32, 34), 34: (16, 33, 35), 35: (34, 36, 51),
    36: (35, 37, 53), 37: (17, 36, 38), 38: (19, 37, 39), 39: (38, 40, 54), 40: (21, 39, 41),
    41: (40, 42, 55), 42: (23, 41, 43), 43: (42, 44, 57), 44: (24, 43, 45), 45: (27, 44, 46),
    46: (45, 47, 58), 47: (28, 46, 48), 48: (31, 47, 49), 49: (48, 50, 59), 50: (32, 49, 51),
    51: (35, 50, 52), 52: (51, 53, 60), 53: (36, 52, 54), 54: (39, 53, 55), 55: (41, 54, 56),
    56: (55, 57, 60), 57: (43, 56, 58), 58: (46, 57, 59), 59: (49, 58, 60), 60: (52, 56, 59),
}

#: the 30 [6,6] bonds, the double bonds of the Kekule structure of the bare fullerene
C60_DOUBLE_BONDS = (
    (1, 9), (2, 12), (3, 15), (4, 18), (5, 6), (7, 8), (10, 11), (13, 14), (16, 17), (19, 20),
    (21, 40), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39),
    (41, 42), (43, 57), (44, 45), (46, 58), (47, 48), (49, 59), (50, 51), (52, 60), (53, 54), (55, 56),
)

#: the skeleton with atom-map numbers equal to the locants and the Kekule structure above
C60_MAPPED_SMILES = (
    "[C:1]12=[C:9]3[C:8]4=[C:7]5[C:6]6=[C:5]1[C:4]1=[C:18]7[C:17]8=[C:16]9[C:15]%10=[C:3]1[C:2]2=[C:12]1"
    "[C:11]2=[C:10]3[C:26]3=[C:27]%11[C:28]%12=[C:29]2[C:30]2=[C:31]%13[C:32]%14=[C:33]([C:14]%10=[C:13]12)"
    "[C:34]9=[C:35]1[C:36]2=[C:37]8[C:38]8=[C:39]9[C:40]%10=[C:21]([C:20]6=[C:19]78)[C:22]5=[C:23]5"
    "[C:24](=[C:25]43)[C:44]3=[C:45]%11[C:46]4=[C:58]6[C:57]7=[C:43]3[C:42]5=[C:41]%10[C:55]3=[C:56]7"
    "[C:60]5=[C:52]([C:51]1=[C:50]%14[C:49](=[C:59]65)[C:48]%13=[C:47]%124)[C:53]2=[C:54]93"
)


def neighbour_pairs() -> Tuple[Tuple[int, int], ...]:
    """Every bond of the skeleton once, as (lower locant, higher locant)."""
    return tuple(sorted((a, b) for a, ns in C60_NEIGHBOURS.items() for b in ns if a < b))


@lru_cache(maxsize=1)
def template_mol():
    """The skeleton as an RDKit molecule (not aromatised: the Kekule structure of ``C60_DOUBLE_BONDS``),
    atom index ``i - 1`` carrying locant ``i`` and the atom-map number ``i``. Shared and read-only:
    copy it before editing."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(C60_MAPPED_SMILES, sanitize=False)
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
    order = sorted(range(mol.GetNumAtoms()), key=lambda i: mol.GetAtomWithIdx(i).GetAtomMapNum())
    return Chem.RenumberAtoms(mol, order)
