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
def test_nitroethoxymethyl_oxirane(namer):
    smi = "[O-][N+](=O)CCOCC1CO1"
    name = namer.name(smi)
    assert name == "[(2-nitroethoxy)methyl]oxirane", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_nitropropyl_analog_rescued(namer):
    # currently abstains (OPSIN-unparseable "1-nitropropyl"); after fix names or still RT-safe
    smi = "[O-][N+](=O)CCCOCC1CO1"
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert "1-nitropropyl" not in name, name
        assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "[O-][N+](=O)CCc1ccccc1",   # (2-nitroethyl)benzene — locant must be 2
    "O=[N+]([O-])CCCC",          # 1-nitrobutane parent (nitro as suffix-less)
])
def test_nitro_substituent_locant_general(namer, smi):
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert _full_rt(smi, name), name
