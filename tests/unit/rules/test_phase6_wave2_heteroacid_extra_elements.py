import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("CCCCCCCC(CC[Se](=O)(=O)O)CCCCCC", "3-hexyldecane-1-selenonic acid"),
    ("CCCCCCCC(CC[Se](=O)O)CCCCCC", "3-hexyldecane-1-seleninic acid"),
    ("CCC(=O)NCC[Se](=O)(=O)O", "2-propanamidoethane-1-selenonic acid"),
    ("CCCCCCCC(CC[Te](=O)(=O)O)CCCCCC", "3-hexyldecane-1-telluronic acid"),
    ("CCC(=O)NCC[Te](=O)(=O)O", "2-propanamidoethane-1-telluronic acid"),
])
def test_se_te_acid_parent_chain(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "CCCCC[Se](=O)(=O)O",   # plain selenonic — must stay correct
    "CCCCC[As](=O)(O)O",    # plain arsonic — must stay correct
])
def test_plain_heteroacid_unchanged(namer, smi):
    name = namer.name(smi)
    assert name and "unknown" not in name and "not supported" not in name, name
    assert _full_rt(smi, name), name
