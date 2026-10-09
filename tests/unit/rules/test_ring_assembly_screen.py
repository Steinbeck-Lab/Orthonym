""",: does a molecule join two identical ring systems by a bond? (``rules.ring_assembly_screen``).

 (the Blue Book) "Two or more cyclic systems... that are directly joined to each
other by single or double bonds are called 'ring assemblies'...";:15550 "Ring assemblies are
composed of identical cyclic systems (rings or ring systems); assemblies of nonidentical cyclic
systems (rings or ring systems) are not called ring assemblies". (:24153) "When
assemblies of otherwise identical rings contain both mancude and saturated rings, the use of
hydro prefixes is preferred, except in the case of a two ring assembly consisting of one benzene
ring and a cyclohexane ring": identity is that of the ring skeleton. A principal characteristic
group on one component is the suffix of the assembly ('(1P)-2',5'-dimethoxy-6-nitro[1,1'-
biphenyl]-2-carboxylic acid (PIN)',:49805), so the drug lane's producers decline such a
molecule. The screen fails closed: True on error.
"""
import pytest
from rdkit import Chem

from orthonym.rules import ring_assembly_screen
from orthonym.rules.ring_assembly_screen import joins_identical_ring_systems, ring_system_key

NILOTINIB = "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1"
ENASIDENIB = "CC(C)(O)CNc1nc(Nc2ccnc(C(F)(F)F)c2)nc(-c2cccc(C(F)(F)F)n2)n1"

ROWS = [
    #:15569 'The name biphenyl is retained as 1,1'-biphenyl.'
    ("c1ccc(cc1)-c1ccccc1", True),
    #:49805 '(1P)-2',5'-dimethoxy-6-nitro[1,1'-biphenyl]-2-carboxylic acid (PIN)' (no axis given)
    ("OC(=O)c1cccc([N+](=O)[O-])c1-c1cc(OC)ccc1OC", True),
    # N-(pyridin-2-yl)[1,1'-biphenyl]-4-carboxamide, not '4-phenyl-N-(pyridin-2-yl)benzamide'
    ("O=C(Nc1ccccn1)c1ccc(cc1)-c1ccccc1", True),
    # 5-[(pyrimidin-2-yl)amino][1,1'-biphenyl]-2-carboxamide
    ("NC(=O)c1ccc(cc1-c1ccccc1)Nc1ncccn1", True),
    # two identical heterocycles: a bipyridine
    ("c1ccc(nc1)-c1ccccn1", True),
    # a double-bond junction (:15542 "by single or double bonds")
    ("C1CCC(CC1)=C1CCCCC1", True),
    # a mancude and a saturated form of one ring are "otherwise identical rings",:24153):
    # '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)', not '2-(piperidin-2-yl)pyridine' (:24159)
    ("C1CCNC(C1)c1ccccn1", True),
    # '2,3-dihydro-1,1'-biphenyl (PIN)', not '(cyclohexa-1,3-dien-1-yl)benzene' (:17083)
    ("C1(=CC=CCC1)c1ccccc1", True),
    # a benzene ring and a cyclohexene ring: hydro prefixes (:24153; (a),:17079)
    ("C1=C(CCCC1)c1ccccc1", True),
    # three rings: only the two-ring benzene + cyclohexane assembly is substitutive (:24153);
    # '1²,1⁵,2²,2³-tetrahydro-1¹,2¹:2⁴,3¹-terphenyl (PIN)' (:17091), '...-dodecahydro-1¹,2¹:2⁴,3¹-
    # terphenyl (PIN)' over '...-4-(4-ethylphenyl)-1,1'-bi(cyclohexane)' (:50157)
    ("c1ccc(cc1)C1CCC(CC1)c1ccccc1", True),
    ("C1CCC(CC1)C1CCC(CC1)c1ccccc1", True),
    # a cyclohexane and a cyclohexene: '[1,1'-bi(cyclohexan)]-3'-en-4-yl' (:50452)
    ("C1CCC(CC1)C1CC=CCC1", True),
    # a pyridine and a piperidine inside a substituent or bearing the suffix
    ("O=C(Nc1ccc(cn1)C1CCNCC1)c1ccccc1", True),
    ("OC(=O)c1ccc(Nc2ccc(cn2)C2CCNCC2)cc1", True),
    ("Nc1ccc(nc1)C1CCNCC1", True),
    # a junction at a ring nitrogen,:15593): '1,1′-bipyrrole (PIN)' (:15603),
    # '2H-1,2′-bipyridine (PIN)' (:15634); with (:24153) a pyridine bonded to the nitrogen
    # of a piperidine is '3,4,5,6-tetrahydro-2H-1,2′-bipyridine', not '2-(piperidin-1-yl)pyridine',
    # and two piperidines joined at their nitrogens are '1,1′-bipiperidine'
    ("c1ccc(nc1)N1CCCCC1", True),
    ("C1CCN(CC1)N1CCCCC1", True),
    # the exception (:24153): one benzene ring and one cyclohexane ring, 'cyclohexylbenzene
    # (PIN)' (:24157), also with a substituent on the cyclohexane ring ('N-[(1r,4S)-4-phenyl
    # cyclohexyl]' in a PIN,:50452)
    ("C1CCC(CC1)c1ccccc1", False),
    ("NC1CCC(CC1)c1ccccc1", False),
    ("O=C(Nc1ccccn1)c1ccc(cc1)C1CCCCC1", False),
    # nonidentical ring skeletons (:15550): substitutive names
    ("c1ccc(cc1)-c1ccccn1", False),
    ("C1CCC(CC1)c1ccccn1", False),
    ("C1CC1c1ccccc1", False),
    ("c1ccc(cc1)N1CCCCC1", False),
    # rings joined through an atom, not a bond
    ("c1ccc(Oc2ccccc2)cc1", False),
    # nilotinib (five ring systems), enasidenib (a triazine bonded to a pyridine)
    (NILOTINIB, False),
    (ENASIDENIB, False),
    # one ring system: fused, spiro
    ("c1ccc2ccccc2c1", False),
    ("C1CCC2(CC1)CCCCC2", False),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_identical_ring_systems_joined_by_a_bond(smiles, expected):
    assert joins_identical_ring_systems(Chem.MolFromSmiles(smiles)) is expected


@pytest.mark.parametrize("a,b,same", [
    ("c1ccncc1", "C1CCNCC1", True),        # pyridine, piperidine
    ("c1ccccc1", "C1=CCCCC1", True),       # benzene, cyclohexene
    ("c1ccccc1", "c1ccncc1", False),       # benzene, pyridine
    ("c1ccc2[nH]ccc2c1", "C1CCC2NCCC2C1", True),   # indole, octahydroindole
])
def test_the_key_is_the_ring_skeleton(a, b, same):
    ka = ring_system_key(Chem.MolFromSmiles(a), range(Chem.MolFromSmiles(a).GetNumAtoms()))
    kb = ring_system_key(Chem.MolFromSmiles(b), range(Chem.MolFromSmiles(b).GetNumAtoms()))
    assert (ka == kb) is same


def test_an_error_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("perception failed")
    monkeypatch.setattr(ring_assembly_screen, "_ring_systems", boom)
    assert joins_identical_ring_systems(Chem.MolFromSmiles("c1ccc(cc1)-c1ccccn1")) is True


def test_the_answer_is_memoised_per_molecule_object():
    mol = Chem.MolFromSmiles("c1ccc(cc1)-c1ccccc1")
    assert joins_identical_ring_systems(mol) is True
    assert ring_assembly_screen._MEMO_MOL is mol and ring_assembly_screen._MEMO == {"v": True}
