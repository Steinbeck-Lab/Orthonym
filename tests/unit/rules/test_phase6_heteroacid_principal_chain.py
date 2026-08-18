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
    ("CCC(=O)NCCS(=O)(=O)O", "2-propanamidoethane-1-sulfonic acid"),
    ("CCCCC(=O)NCCS(=O)(=O)O", "2-pentanamidoethane-1-sulfonic acid"),
])
def test_taurine_amide_sulfonic_acid_parent(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_short_acyl_boundary_unchanged(namer):
    # competing acyl chain <= the 2-C bearing chain: bug does not bite, still names.
    smi = "CC(=O)NCCS(=O)(=O)O"
    name = namer.name(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name
