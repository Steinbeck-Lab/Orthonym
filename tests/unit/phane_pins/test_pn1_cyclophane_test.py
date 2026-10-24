import pytest
from rdkit import Chem

from orthonym.validation.spelling.cyclophane import is_p52_cyclophane, p52_cyclophane_systems

from ._helpers import gold, mol, substituted


@pytest.mark.parametrize("line,name,smiles", gold("cycle"), ids=lambda v: str(v)[:24])
def test_book_cyclophanes_are_cyclophanes(line, name, smiles):
    # (1): the book prints these as phane PINs
    assert is_p52_cyclophane(mol(smiles)), name


@pytest.mark.parametrize("line,name,smiles", gold("path"), ids=lambda v: str(v)[:24])
def test_linear_phanes_are_no_cyclophanes(line, name, smiles):
    assert not is_p52_cyclophane(mol(smiles)), name


def test_a_conjugated_macrocycle_is_a_cyclophane():
    # RDKit leaves these furans non-aromatic in the macrocycle (1) names the system mancude,
    # not RDKit-aromatic); the test reads the isolated ring system (:29005)
    (_, name, smiles), = [r for r in substituted("suffix") if "furana" in r[1]]
    assert is_p52_cyclophane(mol(smiles)), name


@pytest.mark.parametrize("smiles", [
    "C1CC2=CC=C(C=C2)CCC2=CC=C1C=C2",          # [2.2]paracyclophane
    "C1Cc2cccc(c2)CCc2cccc1c2",                # [2.2]metacyclophane
    "C1CC2=CC3=CC(=C2)CCC=2C=C(C=C1C2)CC3",    # [2.2.2](1,3,5)cyclophane
])
def test_other_cyclophanes(smiles):
    assert is_p52_cyclophane(mol(smiles))


@pytest.mark.parametrize("smiles", [
    "C1CCCCCc2ccccc2CCCCC1",                   # (:23839): dodecahydrobenzo[14]annulene is a fusion name
    "C1CC2CCCC(C1)CCC1CCCC(C1)CC2",            # (:23867): no mancude ring, a von Baeyer PIN
    "C1Cc2cccc3cccc1c23",                      # acenaphthene: peri-fused
    "Oc1ccc2c3c1O[C@H]1[C@@H](O)CC[C@@]4(O)[C@@H](C2)N(CC2CCC2)CC[C@]314",   # a morphinan: bridged fused
    "CC1CCCC(=O)CCCC=Cc2cc(O)cc(O)c2C(=O)O1",  # zearalenone: benzene on adjacent atoms of the macrocycle
    "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3",   # porphyrin: a retained parent
    "C1CC2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1N2",    # a chlorin, the same class
    "C1=CC2=CC3=CC=C(N3)C4=CC=C(N4)C=C5C=CC(=N5)C=C1N2",     # a corrole
    "c1cc2nc3ccc(nc4ccc(nc5ccc(nc1n2)[nH]5)n4)[nH]3",         # porphyrazine: nitrogen at the meso positions
    "C1=CC=C2C(=C1)C3=NC4=C5C=CC=CC5=C(N4)N=C6C7=CC=CC=C7C(=N6)N=C8C9=CC=CC=C9C(=N8)N=C2N3",   # phthalocyanine
    "c1ccc(-c2ccccc2)cc1",
    # (:23833,:23835): a ring on adjacent atoms of each of two separate macrocycles is fused or
    # bridged fused, and the attachment atoms of different macrocycles are not compared
    "c12cc3c(cc1CCCCCCCCCC2)CCCCCCCCCC3",
    "c12ccc3c(c1CCCCCCCCCC2)CCCCCCCCCC3",
    "C1CCCCCCCCc2ccc3cc4CCCCCCCCCc4cc3c2C1",
])
def test_not_cyclophanes(smiles):
    assert not is_p52_cyclophane(mol(smiles))


def test_a_bridged_fused_system_is_no_cyclophane():
    # (:14902) lists bridged fused ring systems apart from the mancude ones, so a ring system
    # with a methano bridge is no mancude system (1)), and (:23831) makes the bridged
    # fused name the PIN: the book prints 1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine (PIN) at:19904
    assert not is_p52_cyclophane(mol("C1=CC2=C3N=Cc4cc5ccc6cc5cc4C(=CC3=C1C2)CCCCC6"))


def test_a_ring_system_with_a_saturated_ring_is_not_a_mancude_system():
    # (1): the same chain on the benzene ring of a tetralin and on a benzene ring
    assert not is_p52_cyclophane(mol("C1CCCCCCCc2cc3CCCCc3c1c2"))
    assert is_p52_cyclophane(mol("C1CCCCCCCc2cccc1c2"))


@pytest.mark.parametrize("smiles", [
    "C1CCCCCCCc2cc3C=CC=Cc3c1c2",              # naphthalene on two nonadjacent positions
    "C1CCCCCCCC2=CC=CC(=C2)C1",                # a Kekule drawing of the benzene ring
])
def test_a_mancude_system_in_any_drawing_is_a_mancude_system(smiles):
    assert is_p52_cyclophane(mol(smiles))


@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC=CC=CC=C1", 4), ("c1ccoc1", 2), ("C1=CC=COC=C1", 3), ("C1=CC=C[SiH]=C1", 3),
])
def test_the_maximum_number_of_ring_double_bonds(smiles, expected):
    from orthonym.validation.spelling.cyclophane import _max_ring_double_bonds
    m = mol(smiles)
    assert _max_ring_double_bonds(m, list(m.GetRingInfo().AtomRings()[0])) == expected


def test_the_systems_are_atom_sets_of_the_macrocycle():
    systems = p52_cyclophane_systems(mol("C1CC2=CC=C(C=C2)CCC2=CC=C1C=C2"))
    assert len(systems) == 1 and len(systems[0]) == 16


def test_the_producers_read_the_same_test():
    from orthonym.rules import phane_pin
    assert phane_pin.is_p52_cyclophane is is_p52_cyclophane
    assert phane_pin.p52_cyclophane_systems is p52_cyclophane_systems


def test_ring_system_hydride_keeps_the_identical_parent():
    from orthonym.validation.spelling.linear_phane import _identical_parent, ring_system_hydride
    m = mol("Cc1ccncc1")
    atoms = frozenset(range(1, 7))
    hydride, atom_map = ring_system_hydride(m, atoms)
    assert Chem.MolToSmiles(hydride, isomericSmiles=False) == _identical_parent(m, atoms) == "c1ccncc1"
    assert set(atom_map) == set(atoms) and sorted(atom_map.values()) == list(range(6))


@pytest.mark.parametrize("smiles", ["c1ccccc1", "c1ccoc1", "c1ccncc1", "c1cnccn1", "C1=CC=CNC=C1", "C1=CC=COC=C1"])
def test_the_local_matching_agrees_with_the_engine(smiles):
    # the reader may not import the engine (the package's import guard), so it carries its own ring matching:
    # an O, S, Se or Te atom of valence two never carries a mancude ring double bond
    from orthonym.rules.heterocycles import _mancude_max_matching
    from orthonym.validation.spelling.cyclophane import _max_ring_double_bonds
    m = mol(smiles)
    ring = list(m.GetRingInfo().AtomRings()[0])
    assert _max_ring_double_bonds(m, ring) == _mancude_max_matching(m, ring)[0]


def test_a_ring_in_two_macrocycles_is_compared_within_each_one():
    # [2.2.2](1,3,5)cyclophane and the spiro book skeleton (:19659) keep their rings attached at nonadjacent atoms
    assert is_p52_cyclophane(mol("C1CC2=CC3=CC(=C2)CCC=2C=C(C=C1C2)CC3"))
    assert is_p52_cyclophane(mol("C1CCC2=CC(=CC=C2)CCCC12CCCC1=CC(=CC=C1)CCC2"))


@pytest.mark.parametrize("n_nodes,edges,blocks", [
    (3, [(0, 1), (1, 2), (2, 0)], {0: 0, 1: 0, 2: 0}),                       # a triangle
    (4, [(0, 1), (1, 2), (2, 0), (2, 3)], {0: 0, 1: 0, 2: 0}),               # a bridge is in no block
    (5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}),   # two cycles on one node
    (2, [(0, 1), (0, 1)], {0: 0, 1: 0}),                                     # two parallel edges
    (2, [(0, 0), (0, 1)], {0: 0}),                                           # a loop and a bridge
])
def test_cycle_edges_are_grouped_by_block(n_nodes, edges, blocks):
    from orthonym.validation.spelling.cyclophane import _cycle_blocks
    got = _cycle_blocks(n_nodes, edges)
    assert set(got) == set(blocks)
    for a in blocks:
        for b in blocks:
            assert (got[a] == got[b]) == (blocks[a] == blocks[b])


@pytest.mark.parametrize("smiles", [
    # a porphyrin, and a phthalocyanine, with a [2.2]paracyclophane bonded to it
    "C1=Cc2cc3ccc([nH]3)c(-c3cc4ccc3CCc3ccc(cc3)CC4)c3nc(cc4ccc(cc1n2)[nH]4)C=C3",
    "c1ccc2c(c1)-c1nc3nc(nc4[nH]c(nc5[nH]c(nc-2n1)c1ccccc51)c1cc(-c2cc5ccc2CCc2ccc(cc2)CC5)ccc41)-c1ccccc1-3",
])
def test_a_tetrapyrrole_and_a_separate_cyclophane_are_judged_on_the_cyclophane(smiles):
    m = mol(smiles)
    systems = p52_cyclophane_systems(m)
    assert len(systems) == 1 and len(systems[0]) == 16
    assert all(m.GetAtomWithIdx(a).GetSymbol() == "C" for a in systems[0])


@pytest.mark.parametrize("smiles,judged", [
    ("C1CCCCc2cccc1c2", False),               # [5]metacyclophane: the ring has 8 atoms and joins the benzene's system
    ("C1CCCCCc2cccc1c2", False),              # [6]metacyclophane: a macrocycle of 9 atoms, smaller than any the book prints
    ("C1CCCCCCc2cccc1c2", True),              # [7]metacyclophane: 10 atoms, as the ring of the:15021 name
    ("C1CCCCCc2ccc1cc2", True),               # [6]paracyclophane: 10 atoms
])
def test_the_test_judges_macrocycles_the_book_prints_a_phane_pin_for(smiles, judged):
    # (1) states no size limit; the smallest macrocycle of the book's cyclophane PIN rows has 10 atoms
    assert is_p52_cyclophane(mol(smiles)) is judged


#: book PIN rows the structure test judges and the producers do not name: the peri-fused fluoranthene is no amplificant of
#: the producers; the SMILES is the one OPSIN main reads the book name to (VERIFIED 2026-10-24)
STRUCTURE_ONLY = [
    ("15039", "1(3,10)-fluoranthenacyclononaphane", "C1CCCCCCCC=2C=CC=C3C4=CC=CC5=C1C=CC(C23)=C54"),
]


@pytest.mark.parametrize("line,name,smiles", STRUCTURE_ONLY, ids=lambda v: str(v)[:24])
def test_a_book_cyclophane_with_one_polycyclic_amplificant_is_a_cyclophane(line, name, smiles):
    # (:15033) '1(3,10)-fluoranthenacyclononaphane (I) (PIN)' (:15039): one polycyclic amplificant, a chain,
    # no other ring; (1) (:23828) is met, so the test must judge it
    assert is_p52_cyclophane(mol(smiles))


def test_a_chain_across_the_2_6_positions_of_naphthalene_is_a_cyclophane():
    # the same shape as:15039 on naphthalene joined at the nonadjacent positions 2 and 6 by a macrocycle of 12 atoms
    assert is_p52_cyclophane(mol("C1CCCc2ccc3cc(ccc3c2)CC1"))
