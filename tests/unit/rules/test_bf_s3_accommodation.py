"""Slice S3: the rule that accommodates a suffix on an atom without enough hydrogen, checked
against every name of the class the Blue Book prints on a fused or monocyclic parent.

``hydro.accommodate`` chooses the indicated hydrogen, the 'added indicated hydrogen' and the
hydro positions for one numbering:24766-:24841,:24693-:24721;
(b):3246 "a higher locant may be needed at another position to accommodate a substituent
suffix"). Each case is OPSIN 2.9.0's structure and numbering of the printed name (`-o
extendedsmi`, ``$_AV`` locants), the suffix atoms with the hydrogen each needs (two for =O
or =S, one for -OH, -COOH or a free valence), and the printed indicated hydrogen, added
hydrogen and hydro locants ('' = none; a printed name without hydro locants lists every
remaining saturated position). The bridged fused rows of the class are in
``test_bf_s3_targets.py``. Not tested here: the Table 3.2 row '2,3-dihydro-1H-isoindol-2-yl'
(:17380), which and the book's own '1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl'
(:25477) contradict (S3 TRIAGE)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import hydro
from orthonym.rules.bridged_fused_pin.selection import Split, ring_system

# (printed name, OPSIN extended SMILES, the Blue Book line, [(suffix locant, H needed)],
# indicated hydrogen, added hydrogen, hydro locants)
CASES = [
    ('5H-inden-5-one',
     'C=1C=CC2=CC(C=CC12)=O |$_AV:1;2;3;3a;4;5;6;7;7a;O$|',
     ':3254', [('5', 2)], '5', '', ''),
    ('1,2-dihydro-3H-indol-3-one',
     'N1CC(C2=CC=CC=C12)=O |$_AV:1;2;3;3a;4;5;6;7;7a;O$|',
     ':29342', [('3', 2)], '3', '', '1,2'),
    ('hexahydro-1H-isoindole-1,3(2H)-dione',
     "C1(NC(C2CCCCC12)=O)=O |$_AV:1;2;3;3a;4;5;6;7;7a;O';O$|",
     ':33849', [('1', 2), ('3', 2)], '1', '2', '3a,4,5,6,7,7a'),
    ('2-phenyl-1H-isoindole-1,3(2H)-dione',
     "C1(=CC=CC=C1)N1C(C2=CC=CC=C2C1=O)=O |$_AV:1;2;3;4;5;6;2;1;7a;7;6;5;4;3a;3;O';O$|",
     ':33853', [('1', 2), ('3', 2)], '1', '2', ''),
    ('1H-cyclopenta[a]naphthalene-1,2(3H)-dione',
     "C1(C(CC=2C1=C1C=CC=CC1=CC2)=O)=O |$_AV:1;2;3;3a;9b;9a;9;8;7;6;5a;5;4;O';O$|",
     ':24822', [('1', 2), ('2', 2)], '1', '3', ''),
    ('3,3a-dihydro-1H-indene-1,4(2H)-dione',
     "C1(CCC2C(C=CC=C12)=O)=O |$_AV:1;2;3;3a;4;5;6;7;7a;O';O$|",
     ':24820', [('1', 2), ('4', 2)], '1', '2', '3,3a'),
    ("9,10-dihydro-2H,4H-benzo[1,2-b:4,3-c']dipyran-2,6(8H)-dione",
     "C1=C2C(COC1=O)=CC(C=1OCCCC12)=O |$_AV:1;10b;4a;4;3;2;O;5;6;6a;7;8;9;10;10a;O'$|",
     ':24850', [('2', 2), ('6', 2)], '2,4', '8', '9,10'),
    ("4,4a-dihydro-2H,5H-benzo[1,2-b:4,3-c']dipyran-5,6(6aH)-dione",
     "C1=C2C(COC1)C(C(C1OC=CC=C12)=O)=O |$_AV:1;10b;4a;4;3;2;5;6;6a;7;8;9;10;10a;O';O$|",
     ':24854', [('5', 2), ('6', 2)], '2,5', '6a', '4,4a'),
    ('1H,5H-pyrido[3,2,1-ij]quinoline-6,8-dione',
     "C1C=CN2C=3C(C(C=CC13)=O)=CC(C2)=O |$_AV:1;2;3;4;10b;7a;8;9;10;10a;O';7;6;5;O$|",
     ':24858', [('8', 2), ('6', 2)], '1,5', '', ''),
    ('1,3,4,5-tetrahydronaphthalene-4a(2H)-carboxylic acid',
     "C1CCCC2(CC=CC=C12)C(=O)O |$_AV:1;2;3;4;4a;5;6;7;8;8a;C;O';O$|",
     ':24719', [('4a', 1)], '', '2', '1,3,4,5'),
    ('1,4-dihydro-3aH-indene-3a-carboxylic acid',
     "C1C=CC2(CC=CC=C12)C(=O)O |$_AV:1;2;3;3a;4;5;6;7;7a;C;O';O$|",
     ':24778', [('3a', 1)], '3a', '', '1,4'),
    ('1,3b,4,5,6,6a,7,7a-octahydro-3aH-cyclopenta[a]pentalene-3a,4-diol',
     "C1C2CC3C(C2(C=C1)O)C(CC3)O |$_AV:1;7a;7;6a;3b;3a;3;2;O;4;5;6;O'$|",
     ':24790', [('3a', 1), ('4', 1)], '3a', '', '1,3b,4,5,6,6a,7,7a'),
    ('2H,7H-pyrano[2,3-b]pyran-2,7-dione',
     "O1C(C=CC2=C1OC(C=C2)=O)=O |$_AV:1;2;3;4;4a;8a;8;7;6;5;O';O$|",
     ':24782', [('2', 2), ('7', 2)], '2,7', '', ''),
    ('7H-1-benzopyran-7-one',
     'O1C=CC=C2C1=CC(C=C2)=O |$_AV:1;2;3;4;4a;8a;8;7;6;5;O$|',
     ':24776', [('7', 2)], '7', '', ''),
    ("2,3,7,8-tetrahydro-4H,6H-benzo[1,2-b:5,4-b']dipyran-4,6-dione",
     "O1C2=C(C(CC1)=O)C=C1C(OCCC1=O)=C2 |$_AV:1;10a;4a;4;3;2;O;5;5a;9a;9;8;7;6;O';10$|",
     ':24788', [('4', 2), ('6', 2)], '4,6', '', '2,3,7,8'),
    ("7,8-dihydro-2H,6H-benzo[1,2-b:5,4-b']dipyran-6-one",
     'O1C2=C(C=CC1)C=C1C(OCCC1=O)=C2 |$_AV:1;10a;4a;4;3;2;5;5a;9a;9;8;7;6;O;10$|',
     ':24804', [('6', 2)], '2,6', '', '7,8'),
    ('naphthalen-1(2H)-one',
     'C1(CC=CC2=CC=CC=C12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':24699', [('1', 2)], '', '2', ''),
    ('5,6,7,8-tetrahydronaphthalen-2(4aH)-one',
     'C=1C(C=CC2CCCCC12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':24717', [('2', 2)], '', '4a', '5,6,7,8'),
    ('pyrimidine-4,6(1H,5H)-dione',
     "N1C=NC(CC1=O)=O |$_AV:1;2;3;4;5;6;O';O$|",
     ':24709', [('4', 2), ('6', 2)], '', '1,5', ''),
    ('anthracene-1,9,10(2H)-trione',
     "C1(CC=CC=2C(C3=CC=CC=C3C(C12)=O)=O)=O |$_AV:1;2;3;4;4a;10;10a;5;6;7;8;8a;9;9a;O';O'';O$|",
     ':24758', [('1', 2), ('10', 2), ('9', 2)], '', '2', ''),
    ('2,3-dihydronaphthalene-1,4-dione',
     "C1(CCC(C2=CC=CC=C12)=O)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O';O$|",
     ':24884', [('1', 2), ('4', 2)], '', '', '2,3'),
    ('pyrene-1,3,6,8(2H,7H)-tetrone',
     "C1(CC(C2=CC=C3C(CC(C4=CC=C1C2=C34)=O)=O)=O)=O |$_AV:1;2;3;3a;4;5;5a;6;7;8;8a;9;10;10a;10b;10c;O''';O'';O';O$|",
     ':28932', [('1', 2), ('3', 2), ('6', 2), ('8', 2)], '', '2,7', ''),
    ('1H-cyclopenta[b]naphthalene-1,5,8-trione',
     "C1(C=CC=2C1=CC=1C(C=CC(C1C2)=O)=O)=O |$_AV:1;2;3;3a;9a;9;8a;8;7;6;5;4a;4;O';O'';O$|",
     ':24748', [('1', 2), ('8', 2), ('5', 2)], '1', '', ''),
    ('anthracene-9,10-dione',
     "C1=CC=CC=2C(C3=CC=CC=C3C(C12)=O)=O |$_AV:1;2;3;4;4a;10;10a;5;6;7;8;8a;9;9a;O;O'$|",
     ':24740', [('10', 2), ('9', 2)], '', '', ''),
    ('pyrazine-2,3-dione',
     "N=1C(C(N=CC1)=O)=O |$_AV:1;2;3;4;5;6;O';O$|",
     ':24728', [('2', 2), ('3', 2)], '', '', ''),
    ('5,7-dihydro-6H-dibenzo[a,c][7]annulen-6-one',
     'C1=CC=CC2=C1C1=C(CC(C2)=O)C=CC=C1 |$_AV:1;2;3;4;4a;11b;11a;7a;7;6;5;O;8;9;10;11$|',
     ':49606', [('6', 2)], '6', '', '5,7'),
    ('tetrahydro-4H-pyran-4-one',
     'O1CCC(CC1)=O |$_AV:1;2;3;4;5;6;O$|',
     ':24772', [('4', 2)], '4', '', '2,3,5,6'),
    ('1H-phenalen-4-ol',
     'C1C=CC=2C(=CC=C3C=CC=C1C23)O |$_AV:1;2;3;3a;4;5;6;6a;7;8;9;9a;9b;O$|',
     ':3250', [('4', 1)], '1', '', ''),
    ('2H-pyran-6-carboxylic acid',
     "O1CC=CC=C1C(=O)O |$_AV:1;2;3;4;5;6;C;O';O$|",
     ':3252', [('6', 1)], '2', '', ''),
    ('1H-inden-1-one',
     'C1(C=CC2=CC=CC=C12)=O |$_AV:1;2;3;3a;4;5;6;7;7a;O$|',
     ':28416', [('1', 2)], '1', '', ''),
    ('naphthalen-1(4H)-one',
     'C1(C=CCC2=CC=CC=C12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':24880', [('1', 2)], '', '4', ''),
    ('quinolin-2(1H)-one',
     'N1C(C=CC2=CC=CC=C12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':29336', [('2', 2)], '', '1', ''),
    ('isoquinolin-1(2H)-one',
     'C1(NC=CC2=CC=CC=C12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':29338', [('1', 2)], '', '2', ''),
    ('hexahydro-1H-2-benzopyran-1,3(4H)-dithione',
     "C1(OC(CC2C1CCCC2)=S)=S |$_AV:1;2;3;4;4a;8a;8;7;6;5;S';S$|",
     ':32551', [('1', 2), ('3', 2)], '1', '4', '4a,5,6,7,8,8a'),
    ('3,4-dihydronaphthalen-1(2H)-one',
     'C1(CCCC2=CC=CC=C12)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;O$|',
     ':3276', [('1', 2)], '', '2', '3,4'),
    ('2-(1,3,4,5-tetrahydro-2H-2-benzazepin-2-yl)ethan-1-ol',
     'C1N(CCCC2=C1C=CC=C2)CCO |$_AV:1;2;3;4;5;5a;9a;9;8;7;6;2;1;O$|',
     ':24774', [('2', 1)], '2', '', '1,3,4,5'),
    ('1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl (as its 2-methyl compound)',
     "CN1C(C2=CC=CC=C2C1=O)=O |$_AV:1;2;1;7a;7;6;5;4;3a;3;O';O$|",
     ':25477', [('2', 1)], '2', '', '1,3'),
    ('quinoline-1(2H)-carboxylic acid',
     "N1(CC=CC2=CC=CC=C12)C(=O)O |$_AV:1;2;3;4;4a;5;6;7;8;8a;C;O';O$|",
     ':24701', [('1', 1)], '', '2', ''),
    ('1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one',
     'N1(CCCC2=CC=CC=C12)C(C)=O |$_AV:1;2;3;4;4a;5;6;7;8;8a;1;2;O$|',
     ':7467', [('1', 1)], '', '2', '3,4'),
    ('2,3-dihydro-1H-inden-2-yl (as its 2-methyl compound)',
     'CC1CC2=CC=CC=C2C1 |$_AV:1;2;1;7a;7;6;5;4;3a;3$|',
     ':17374', [('2', 1)], '1', '', '2,3'),
]


def _locs(atoms, a2l):
    key = lambda v: (int("".join(c for c in str(v) if c.isdigit())), str(v).lstrip("0123456789"))
    return ",".join(str(x) for x in sorted((a2l[a] for a in atoms), key=key))


@pytest.mark.parametrize("name,ext,line,groups,ih,added,hyd", CASES, ids=[c[0] for c in CASES])
def test_the_book_rows_of_the_class_are_rebuilt(name, ext, line, groups, ih, added, hyd):
    smi, av = ext.split(" |$_AV:")
    labels = av.rstrip("$|").split(";")
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    ring = set(ring_system(mol))
    double = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
              and b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
    unsat = frozenset(a for b in double for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    state = hydro.hydro_state(mol, Split(frozenset(ring), (), (), None, len(double), unsat))
    assert state is not None
    a2l = {a: int(labels[a]) if labels[a].isdigit() else labels[a] for a in ring}
    by_label = {str(v): a for a, v in a2l.items()}
    need = frozenset(by_label[l] for l, n in groups if hydro.lacks_hydrogen(state, by_label[l], n))
    got_ih, got_added, got_hydro = hydro.accommodate(state, a2l, need)
    assert (_locs(got_ih, a2l), _locs(got_added, a2l), _locs(got_hydro, a2l)) == (ih, added, hyd), line


def test_without_a_lacking_group_the_s2_choice_is_kept():
    # no suffix atom lacks hydrogen: the indicated hydrogen is the one the builder chose
    # before slice S3:24641, choose_indicated_hydrogen) on every numbering
    # of an indicated-hydrogen parent (3a,4,5,6,7,7a-hexahydro-1H-indene, 2,3,4,5-tetrahydro-
    # 1H-2-benzazepine)
    for smi in ("C1C=CC2CCCCC12", "C1NCCCc2ccccc21"):
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        ring = set(ring_system(mol))
        double = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                  and b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
        unsat = frozenset(a for b in double for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        state = hydro.hydro_state(mol, Split(frozenset(ring), (), (), None, len(double), unsat))
        assert state is not None and state.ih_sets[0]
        order = sorted(ring)
        for shift in range(len(order)):
            a2l = {a: (i + shift) % len(order) + 1 for i, a in enumerate(order)}
            ih, added, rest = hydro.accommodate(state, a2l, frozenset())
            assert (ih, rest) == hydro.choose_indicated_hydrogen(state, a2l) and not added


def _bridged_state(smiles):
    from orthonym.rules.bridged_fused_pin import numbering, prefixes, selection
    mol = Chem.MolFromSmiles(smiles)
    sp = selection.best_splits(mol, selection.ring_system(mol))[0]
    bps = [prefixes.bridge_prefix(mol, sp, i) for i in range(len(sp.bridges))]
    nb = numbering.numberings(mol, sp, [b.name for b in bps], [b.first for b in bps])[0]
    return mol, hydro.hydro_state(mol, sp), nb.atom_to_locant


def test_three_indicated_hydrogen_atoms():
    # 3a,6a-methanocyclopenta[c]furan needs 1H, 3H and one of 4H/6H (the Blue Book)
    from orthonym.rules.bridged_fused_pin import build
    mol, state, a2l = _bridged_state("C1OCC23CCCC12C3")
    assert state is not None and len(state.ih_sets[0]) == 3
    assert build(mol)[0] == "dihydro-1H,3H,4H-3a,6a-methanocyclopenta[c]furan"


def test_total_hydrogenation_keeps_its_locants_only_in_the_printed_case():
    #:24800 '5,6-dihydro-1H,3H,4H-3a,6a-methanocyclopenta[c]furan-1,3-dione (PIN)': the
    # indicated hydrogen holds the two diones (1H,3H) and one (4H) holds none;:33849
    # 'hexahydro-1H-isoindole-1,3(2H)-dione (PIN)': every indicated hydrogen holds a group
    mol, state, a2l = _bridged_state("O=C1OC(=O)C23CCCC12C3")
    need = frozenset(a for a in state.eligible if hydro.lacks_hydrogen(state, a, 2)
                     and any(n.GetAtomicNum() == 8 and n.GetDegree() == 1
                             for n in mol.GetAtomWithIdx(a).GetNeighbors()))
    ih, added, rest = hydro.accommodate(state, a2l, need)
    assert (hydro.ih_text(ih, a2l), added) == ("1H,3H,4H", frozenset())
    assert hydro.hydro_text(state, rest, a2l, ih, need) == "5,6-dihydro"
    assert hydro.hydro_text(state, rest, a2l, ih, frozenset()) == "dihydro"
    assert hydro.hydro_text(state, rest, a2l, frozenset(need), need) == "dihydro"
