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
def test_nitro_on_norbornane_names(namer):
    smi = "[O-][N+](=O)C1CC2CCC1C2"
    name = namer.name(smi)
    assert name == "2-nitrobicyclo[2.2.1]heptane", name
    assert _full_rt(smi, name), name


def test_no_silent_drop_returns_name_or_abstains(namer):
    # whichever way, never a wrong constitution: emitted name (if any) must RT.
    smi = "[O-][N+](=O)C1CC2CCC1C2"
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert _full_rt(smi, name), name
