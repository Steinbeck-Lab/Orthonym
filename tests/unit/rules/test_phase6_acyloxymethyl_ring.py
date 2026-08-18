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
def test_acetoxymethyl_cyclohexanecarboxylic_acid(namer):
    smi = "CC(=O)OCC1CCCCC1C(=O)O"
    name = namer.name(smi)
    assert name == "2-[(acetyloxy)methyl]cyclohexane-1-carboxylic acid", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_hydroxymethyl_control_unchanged(namer):
    # same skeleton, -CH2OH instead of the ester: must stay correct.
    smi = "OCC1CCCCC1C(=O)O"
    assert namer.name(smi) == "2-(hydroxymethyl)cyclohexane-1-carboxylic acid"
