"""task-123: the 2-component ortho-fusion descriptor path must be
deterministic -- the same molecule must get the same fusion PIN regardless
of input SMILES atom order/spelling.

Root cause (measured, HEAD): a parent ring with a symmetric heteroatom
arrangement (e.g. pyrimidine's N1<->N3 mirror) has two equally-valid IUPAC
numberings tied on heteroatom locants; the old single-pick
``_get_iupac_ring_order_for_fusion(is_child=False)`` chose whichever
physical atom happened to sort first by RDKit atom index -- an artifact of
input SMILES atom order, not chemistry (~4/12 random respellings of
c1ncc2sccc2n1 gave the wrong 'thieno[2,3-e]pyrimidine' instead of the
correct 'thieno[3,2-d]pyrimidine',: lowest letter).

Governing rule: "General principles" (the Blue Book).
"""
import pytest
from rdkit import Chem

from orthonym import name_compound


@pytest.mark.parametrize("smi,expected", [
    ("c1ncc2sccc2n1", "thieno[3,2-d]pyrimidine"),
    ("c1ncc2occc2n1", "furo[3,2-d]pyrimidine"),
])
def test_fusion_descriptor_is_spelling_independent(smi, expected):
    m = Chem.MolFromSmiles(smi)
    names = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(20)}
    assert names == {expected}, f"non-deterministic or wrong: {names}"
