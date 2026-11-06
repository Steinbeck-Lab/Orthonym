import pytest
from tests.support.read_back import same_key


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
    assert same_key(smi, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "CCCCC[Se](=O)(=O)O",   # plain selenonic — must stay correct
    "CCCCC[As](=O)(O)O",    # plain arsonic — must stay correct
])
def test_plain_heteroacid_unchanged(namer, smi):
    name = namer.name(smi)
    assert name and "unknown" not in name and "not supported" not in name, name
    assert same_key(smi, name), name
