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
def test_brace_escalation_oxirane(namer):
    smi = "CCC(C)OCC1CO1"
    name = namer.name(smi)
    assert name == "{[(butan-2-yl)oxy]methyl}oxirane", name
    assert "[[" not in name and "]]" not in name, name
    assert _full_rt(smi, name), name
