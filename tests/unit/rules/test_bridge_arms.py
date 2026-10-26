"""``rules.bridge_arms.arm_sizes_differ``: the graph test that spares the cut of a bridge atom.

``_try_acyclic_heteroatom_bridge`` cuts the molecule at an O/S/NH bridge and asks whether the two pieces have
one canonical SMILES: two identical arms). Two pieces with one canonical SMILES hold the same atoms
but for hydrogens, so a bridge whose arms hold unlike numbers of atoms that are not hydrogen is settled
without the cut. The test pins the claim on bridges of many molecules: wherever the count says "unlike", the
cut gives no piece pair with one SMILES.
"""
import pytest
from rdkit import Chem

from orthonym.perception.molcache import canon_smiles
from orthonym.rules import multiplicative as multiplicative_rules
from orthonym.rules.bridge_arms import arm_sizes_differ

pytestmark = pytest.mark.unit

# ethers, thioethers, secondary amines, amides, peptides, polyethers, salts, rings: acyclic O/S/NH bridges of
# every kind the namer meets, arms alike (the cases) and unlike
MOLECULES = [
    "OC(=O)COCC(=O)O", "OCCSCCO", "OC(=O)CNCC(=O)O", "OCCNCCO", "CCOCC", "CCOC", "COCCOC",
    "OCCOCCOCCO", "OCCOCCOCCOCCOCCO", "CCSCC", "CNC", "CCNCC", "CC(C)OC(C)C", "c1ccccc1OCCOc1ccccc1",
    "NCC(=O)NCC(=O)NCC(=O)O", "CC(=O)NCC(=O)NC(C)C(=O)NCC(=O)O",
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCCN)N)C(=O)O",
    "C(CNCCN)N", "NCCNCCNCCN", "OCCN(CCO)CCO", "CCOC(=O)COCC(=O)OCC", "OCC(O)COCC(O)CO",
    "ClCCOCCCl", "BrCCSCCBr", "N#CCCNCCC#N", "[Na+].[O-]C(=O)COCC(=O)[O-]", "OC(=O)CSCC(=O)O",
    "CCOCCOCCOCCOC", "CCCOCCC.CC", "COc1ccc(OC)cc1", "CN(C)CCOCCN(C)C", "OCCOCCO.OCCO",
    "O=C(O)CCOCCC(=O)O", "OC(=O)CCNCCC(=O)O", "C1CCOCC1", "C1CNCCN1", "c1ccc2c(c1)OCCO2",
    "CC(C)Oc1ccccc1OC(C)C", "OCCCOCCCO", "OCCCSCCCO", "NCCOCCN", "CC(O)COC(C)CO",
    "C=CCOCC=C", "C#CCOCC#C", "CCOCC(=O)OCC", "FC(F)(F)OC(F)(F)F", "OC[C@H](O)COC[C@H](O)CO",
    "[2H]OC([2H])([2H])C", "CC(C)(C)OC(C)(C)C", "NC(=O)CNCC(N)=O", "O=C(NCCO)NCCO",
]


def _bridges(mol):
    """Every acyclic O/S/NH atom with two acyclic carbon neighbours, as ``_try_acyclic_heteroatom_bridge``
    selects them: ``(bridge index, neighbour indices)``."""
    ring_atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
    found = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in ring_atoms:
            continue
        if multiplicative_rules._classify_single_atom_bridge(atom) not in ("O", "S", "NH") or atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetIdx() in ring_atoms or n.GetSymbol() != "C" for n in heavy):
            continue
        found.append((atom.GetIdx(), [n.GetIdx() for n in heavy]))
    return found


def _cut_pieces_alike(mol, idx, nbrs):
    fragments = multiplicative_rules._split_at_bridge(mol, idx, nbrs)
    return fragments is not None and canon_smiles(fragments[0]) == canon_smiles(fragments[1])


def test_unlike_arms_never_cut_into_one_smiles():
    """ (the Blue Book: multiplicative nomenclature needs identical units): the count never says "unlike"
    for arms the cut finds alike."""
    checked = unlike = 0
    for smiles in MOLECULES:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        for idx, nbrs in _bridges(mol):
            checked += 1
            if arm_sizes_differ(mol, idx, nbrs):
                unlike += 1
                assert not _cut_pieces_alike(mol, idx, nbrs), (smiles, idx)
    assert checked >= 40 and unlike >= 20      # the molecules above do exercise both sides


@pytest.mark.parametrize("smiles", ["OC(=O)COCC(=O)O", "OCCSCCO", "OC(=O)CNCC(=O)O", "OCCNCCO", "CCOCC",
                                    "OCCOCCO", "CC(C)OC(C)C"])
def test_alike_arms_are_left_to_the_cut(smiles):
    """The arms of the examples hold as many atoms: the answer is "not decided" and the cut runs."""
    mol = Chem.MolFromSmiles(smiles)
    bridges = _bridges(mol)
    assert bridges
    assert all(not arm_sizes_differ(mol, idx, nbrs) for idx, nbrs in bridges)
    assert any(_cut_pieces_alike(mol, idx, nbrs) for idx, nbrs in bridges)


@pytest.mark.parametrize("smiles", ["CCOC", "OCCOCC", "CCOC(=O)C", "CNCC", "CC(=O)NCC(=O)O"])
def test_unlike_arms_are_decided(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bridges = _bridges(mol)
    assert bridges
    assert all(arm_sizes_differ(mol, idx, nbrs) for idx, nbrs in bridges)


def test_answer_does_not_depend_on_the_order_of_the_neighbours_or_of_the_calls():
    """No state is kept: the neighbours in either order and the calls in any order give one answer."""
    cases = []
    for smiles in MOLECULES:
        mol = Chem.MolFromSmiles(smiles)
        cases += [(mol, idx, nbrs) for idx, nbrs in _bridges(mol)]
    forward = [arm_sizes_differ(m, i, n) for m, i, n in cases]
    backward = [arm_sizes_differ(m, i, n) for m, i, n in reversed(cases)][::-1]
    swapped = [arm_sizes_differ(m, i, n[::-1]) for m, i, n in cases]
    assert forward == backward == swapped


def test_a_bridge_in_a_ring_is_not_decided():
    """The second neighbour is reached from the first: the molecule is not cut into two arms."""
    mol = Chem.MolFromSmiles("C1CCOCC1")
    o = next(a for a in mol.GetAtoms() if a.GetSymbol() == "O")
    assert arm_sizes_differ(mol, o.GetIdx(), [n.GetIdx() for n in o.GetNeighbors()]) is False


def test_explicit_hydrogens_do_not_count():
    """Hydrogen atoms are not counted: with an explicit hydrogen atom on one arm only, the arms of 'CCOCC' hold
    as many atoms that are not hydrogen, the answer is "not decided", and the cut finds the pieces alike."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles("[H]CCOCC", params)
    assert mol.GetNumAtoms() == 6
    (idx, nbrs), = _bridges(mol)
    assert arm_sizes_differ(mol, idx, nbrs) is False
    plain = Chem.MolFromSmiles("CCOCC")
    assert _cut_pieces_alike(plain, *_bridges(plain)[0])


@pytest.mark.parametrize("smiles, expected", [
    ("OC(=O)COCC(=O)O", "2,2'-oxydiacetic acid"),
    ("OCCSCCO", "2,2'-sulfanediyldi(ethan-1-ol)"),
    ("OC(=O)CNCC(=O)O", "2,2'-azanediyldiacetic acid"),
    ("OCCNCCO", "2,2'-azanediyldi(ethan-1-ol)"),
    ("CCOCC", None),
    ("OCCOCC", None),
])
def test_multiplicative_name_of_a_bridged_pair_is_unchanged(smiles, expected):
    """ (the docstring examples of ``_try_acyclic_heteroatom_bridge``): the names are what they were."""
    assert multiplicative_rules._try_acyclic_heteroatom_bridge(Chem.MolFromSmiles(smiles)) == expected
