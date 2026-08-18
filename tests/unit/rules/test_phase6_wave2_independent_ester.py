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
def test_acyloxymethyl_ring_methyl_ester(namer):
    smi = "COC(=O)C1CCCCC1COC(C)=O"
    name = namer.name(smi)
    assert name == "methyl 2-[(acetyloxy)methyl]cyclohexane-1-carboxylate", name
    assert " methyl " not in name  # no stray-space token
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_ring_acyloxy_direct(namer):
    # acyloxy directly on the ring (spy's confirmed clean win)
    smi = "COC(=O)c1ccc(OC(C)=O)cc1"
    name = namer.name(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_topological_acyl_never_wrong(namer):
    # a macrolactone / long topological acyl must abstain or RT -- never a wrong flattening
    smi = "O=C1CCCCCCCCCCCOC1"  # a macrolactone (oxacyclotridecan-2-one class)
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert _full_rt(smi, name), name
