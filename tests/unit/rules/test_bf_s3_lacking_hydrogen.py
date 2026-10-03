"""Slice S3 (Task S3.5): a monovalent suffix or a free valence on an atom with no hydrogen in
the mancude bridged parent (a bridgehead, a fusion atom).

 (the Blue Book): 'added indicated hydrogen' is cited "in the absence of, or
lack of, sufficient hydrogen atoms"; (:24719) '1,3,4,5-tetrahydronaphthalene-
4a(2H)-carboxylic acid (PIN)'; (:17340) '1,3,4,5-tetrahydronaphthalen-4a(2H)-yl
(preferred prefix)';:7467 '1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one'. Indicated hydrogen
that can hold the group does so:24768; '1,4-dihydro-3aH-indene-3a-carboxylic
acid (PIN)':24778). A group on a =CH- atom needs nothing ('1H-phenalen-4-ol (PIN)':3250,
``test_bf_s3_protected.py``). Every name was read back by OPSIN 2.9.0 to the input's full
InChIKey (S3 planning notes, ledger)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build, build_substituent

LACKING = [
    ("OC12CCC(C1)c1ccccc12", "3,4-dihydro-1,4-methanonaphthalen-1(2H)-ol"),
    ("OC(=O)C12CCC(C1)c1ccccc12", "3,4-dihydro-1,4-methanonaphthalene-1(2H)-carboxylic acid"),
    ("OC(=O)C12CCCCC1C1CCC2C1", "octahydro-1,4-methanonaphthalene-4a(2H)-carboxylic acid"),
    ("C1CC[C@@]2([C@H](C1)C[C@H]3CCC[C@@H]2[C@H]3O)O",
     "(4aS,5R,9R,10aR,11S)-decahydro-5,9-methanobenzo[8]annulene-4a,11(2H)-diol"),
]


@pytest.mark.parametrize("smiles,name", LACKING)
def test_lacking_hydrogen_name(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


def test_free_valence_on_a_bridgehead():
    # 9,10-ethanoanthracene: the bridgehead C9 has no hydrogen in the mancude parent
    mol = Chem.MolFromSmiles("C1CC2c3ccccc3C1c1ccccc21")
    attach = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 3 and not a.GetIsAromatic()][0]
    assert build_substituent(mol, attach) == "9,10-ethanoanthracen-9(10H)-yl"
