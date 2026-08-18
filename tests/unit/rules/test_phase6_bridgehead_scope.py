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
    ("c1ccccc1C1CC2CCC1C2", "2-phenylbicyclo[2.2.1]heptane"),
    ("C1CCC(CC1)C1CC2CCC1C2", "2-cyclohexylbicyclo[2.2.1]heptane"),
])
def test_pendant_ring_on_norbornane_names(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert _full_rt(smi, name), name


def test_bridgehead_count_is_component_scoped():
    from orthonym.perception.rings import find_ring_bridgeheads
    mol = Chem.MolFromSmiles("c1ccccc1C1CC2CCC1C2")  # 2-phenylnorbornane
    # the aliphatic norbornane cage = the non-aromatic ring atoms.
    scoped = {a.GetIdx() for a in mol.GetAtoms()
              if a.IsInRing() and not a.GetIsAromatic()}
    # whole-molecule default wrongly yields 4 (the pendant phenyl junction);
    # scoped to the cage it is exactly 2.
    assert len(find_ring_bridgeheads(mol, scoped)) == 2
    assert len(find_ring_bridgeheads(mol)) != 2  # documents the HEAD bug
