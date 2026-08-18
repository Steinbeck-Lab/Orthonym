import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

@pytest.fixture(scope="module")
def namer(): return Orthonym()

def _rt(smi, name):
    o = opsin_parse(name)
    return bool(o) and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == inchi.MolToInchiKey(Chem.MolFromSmiles(o))

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("O=C1CCCCCCC/C=C/CCO1", "(10E)-oxacyclotridec-10-en-2-one"),
    ("O=C1OC=CC1", "furan-2(3H)-one"),
    ("O=C1CCC=CO1", "3,4-dihydro-2H-pyran-2-one"),
])
def test_unsaturated_lactone_names(namer, smi, expected):
    n = namer.name(smi)
    assert n == expected, n
    assert _rt(smi, n), n

@pytest.mark.opsin_gate
def test_saturated_lactone_unchanged(namer):
    # decline must be scoped to unsaturated/dione — saturated lactone still names
    assert namer.name("O=C1CCCCCCCCCCCO1") and "unknown" not in namer.name("O=C1CCCCCCCCCCCO1")
    assert _rt("O=C1CCCCCCCCCCCO1", namer.name("O=C1CCCCCCCCCCCO1"))
