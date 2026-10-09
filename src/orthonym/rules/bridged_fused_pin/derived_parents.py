"""Fused ring parents the Blue Book does not print, derived step by step by the
rules: name -> ``DerivedParent`` (the derivation, OPSIN input name, OPSIN SMILES, OPSIN
locants in SMILES atom order).

 (the Blue Book): fused mancude systems "that have no accepted retained or
systematic name described in sections and are named by prefixing to the name
of a component ring or ring system (the parent component) designations of the other
component(s) (attached components)". ``parent_table.BB_PARENTS`` holds the names the book
prints; this table holds a name the book does not print but its rules give, one entry per
skeleton a bridged fused PIN needs (user ruling D6, 2026-10-02: a second small table of
rule-derived parents, each with a written derivation and an OPSIN 2.9.0 test of its
structure and numbering; no general widening). The structure and the numbering are OPSIN
2.9.0's own reading of the name (``<name> -o extendedsmi``, its ``$_AV`` locants), as for
``BB_PARENTS``; the derivation below is what makes the name and its numbering the
ones, and ``tests/unit/rules/test_bf_s4_derived_parents.py`` checks each step against the
entry and reads every entry back with OPSIN.

'[1]benzofuro[3,2-e]isoquinoline' -- the fused parent of the 4,5-epoxymorphinans
(morphine, codeine, thebaine, nalbuphine,...; their bridged fused name is
'4,12-methano[1]benzofuro[3,2-e]isoquinoline'). Rings, with the morphinan atom labels of
 (the Blue Book): A = benzene (C1-C4, C11, C12), E = furan (O, C4, C12, C13,
C5), C = C5-C8, C14, C13, D = C9, C14, C13, C15, C16, N17 (C10, the methano bridge, is not
part of the parent).

1. Components. (:12137) "If there is a choice for selecting the parent
   component, the following criteria are considered, in order": (a) (:12139) "a component
   containing at least one of the heteroatoms occurring earlier in the following order:
   N > F >... > O" -- the nitrogen components are pyridine (D) and the retained
   isoquinoline (C + D; N17 is bonded to C9, which is bonded to the fusion atom C14, so
   the nitrogen is at the 2-position: isoquinoline, not quinoline); (b) (:12163) "a
   component containing the greater number of rings" -- isoquinoline. The benzene ring A
   is fused only to the furan E, so A + E is one benzoheterocycle unit, (:13437)
   "Heterobicyclic compounds consisting of a heteromonocycle fused to a benzene ring in
   which the benzene ring is not part of a system having a retained name... are treated
   as a single component unit (a 'benzoheterocycle'...). They may be treated as a parent
   component or an attached component", named '1-benzofuran (PIN)' (:11827).
2. Prefix. (:11907) "Locants that describe structural features of components,
   such as positions of heteroatoms, are kept with the name of the component and are
   enclosed within square brackets"; the book's '[1]benzopyrano[2,3-c]pyrrole (PIN)'
   (:12157) spells a benzoheterocycle prefix so: '[1]benzofuro'. No elision of the final
   'o' (Note:11909).
3. Descriptor. (:11911) "each peripheral side of the parent component...
   using the italic letters a, b, c, etc., beginning with a for the side numbered '1,2',
   b for '2,3' etc..... These numbers are chosen to be as low as is consistent with the
   numbering of the compound and their order conforms to the direction of lettering of
   the parent component". Isoquinoline 1 (C9), 2 (N17), 3 (C16), 4 (C15), 4a (C13), 5
   (C5), 6, 7, 8 (C6, C7, C8), 8a (C14): sides a 1-2, b 2-3, c 3-4, d 4-4a, e 4a-5. The
   furan is fused on C13-C5 = side e; 1-benzofuran's C3 lies on 4a and its C2 on 5, so in
   the direction of lettering (4a -> 5) the attached locants are '3,2': '[3,2-e]'. The
   attached locants 3 and 2 are nonfusion atoms of 1-benzofuran, and no locant of the
   parent component is cited: side e (4a-5) is a peripheral side of isoquinoline, lettered
   as:11911 letters "each peripheral side of the parent component (including sides whose
   locants are distinguished by letters, for example, 2a,3a)". The fusion atom C13
   (isoquinoline 4a) then carries four ring bonds, as naphthalene 8a does in
   'naphtho[1,8a-b]azirine (PIN)',:14007,:14011); the book prints another
   fused PIN with such an atom, 'cyclopenta[1,2-b:5,1-b']difuran (PIN)' (:13401).
4. Orientation. (a) (:12083) "maximum number of rings in a horizontal row"
   with "vertical common bonds... always those furthest apart" (:12085): ring C cannot
   be the middle of a row (its two fusion bonds, to E and to D, share C13); the furan E
   can (its two fusion bonds are not adjacent), so A-E-C is the row (3 rings, the
   maximum). (b) (:12093) "maximum number of rings in upper right quadrant": D is fused
   on the C13-C14 side of C next to the C/E bond, so with C at the right end and D above
   the row D lies in the upper right quadrant; the mirror images put D in the upper left
   or below. One orientation.
5. Numbering. (:12501) "The numbering of peripheral atoms in the preferred
   orientation starts from the uppermost ring... from the nonfused atom most
   counterclockwise in the ring selected and proceeds in a clockwise direction around the
   system, including fusion heteroatoms but not fusion carbon atoms. Each fusion carbon
   atom is given the same number as the immediately preceding nonfusion skeletal atom,
   modified by a Roman letter". The uppermost ring is D; clockwise from C13 its nonfusion
   atoms are C15, C16, N17, C9: 1, 2, 3, 4; then 4a (C14), 5, 6, 7 (C8, C7, C6), 7a (C5),
   8 (O), 8a (C4), 9, 10, 11 (C3, C2, C1), 12 (C11), 12a (C12), 12b (C13). Heteroatoms at
   3 and 8. OPSIN 2.9.0 numbers the name the same way (the entry's locants are its
   ``$_AV`` output).
6. Bridges on this parent. (a) (:14261) "contain the maximum number of
   rings", (b) (:14271) "include the maximum number of skeletal atoms": removing C10
   leaves this 17-atom four-ring parent; removing -CH2-CH2-N< (C15, C16, N17) leaves
   phenanthro[4,5-bcd]furan (15 atoms) and loses (b); removing the oxygen leaves rings
   B, C, D sharing the bond C13-C14, not a fused parent. (:14181) bridge
   locants in numerical order: C10 joins 4 (C9) and 12 (C11): '4,12-methano';
   (:14403) "Bridge atoms are numbered continuing from the highest locant of the fused
   ring system": C10 is 13.
"""
from typing import Dict, NamedTuple, Tuple


class DerivedParent(NamedTuple):
    #: (step, the Blue Book line) for each step of the module docstring's derivation
    derivation: Tuple[Tuple[str, int], ...]
    opsin_name: str
    smiles: str
    locants: Tuple[str, ...]


#: name (indicated hydrogen dropped) -> DerivedParent
DERIVED_PARENTS: Dict[str, DerivedParent] = {
    "[1]benzofuro[3,2-e]isoquinoline": DerivedParent(
        derivation=(
            ("P-25.3.2.4 (a) nitrogen component", 12139),
            ("P-25.3.2.4 (b) more rings: isoquinoline", 12163),
            ("P-25.3.5 benzoheterocycle unit 1-benzofuran", 13437),
            ("P-25.3.1.3 component locants in square brackets", 11907),
            ("P-25.3.1.3 side letters and attached locants", 11911),
            ("P-25.3.8.4 a four-bonded fusion atom: naphtho[1,8a-b]azirine (PIN)", 14011),
            ("P-25.3.2.3.3 (a) rings in a horizontal row", 12083),
            ("P-25.3.2.3.3 (b) rings in the upper right quadrant", 12093),
            ("P-25.3.3.1.1 peripheral numbering", 12501),
        ),
        opsin_name="1H-[1]benzofuro[3,2-e]isoquinoline",
        smiles="C1C=NC=C2C=CC=C3C12C1=C(O3)C=CC=C1",
        locants=("1", "2", "3", "4", "4a", "5", "6", "7", "7a", "12b", "12a", "8a", "8",
                 "9", "10", "11", "12"),
    ),
    "cyclopenta[a]phenanthrene": DerivedParent(
        derivation=(
            ("P-25.3.2.4 (b) more rings: phenanthrene is the parent component", 12163),
            ("P-25.3.2.2.1 monocyclic hydrocarbon prefix cyclopenta", 12000),
            ("P-25.3.1.3 side letter of the attached component", 11911),
            ("P-25.3.3 traditional numbering retained", 12493),
        ),
        opsin_name="cyclopenta[a]phenanthrene",
        smiles="C1=CC=CC2=CC=C3C=4CC=CC4C=CC3=C12",
        locants=("1", "2", "3", "4", "5", "6", "7", "8", "14", "15", "16", "17", "13",
                 "12", "11", "9", "10"),
    ),
}
