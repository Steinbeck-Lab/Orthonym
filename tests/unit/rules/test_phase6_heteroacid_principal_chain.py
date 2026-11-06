import pytest
from tests.support.read_back import same_key


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("CCC(=O)NCCS(=O)(=O)O", "2-propanamidoethane-1-sulfonic acid"),
    ("CCCCC(=O)NCCS(=O)(=O)O", "2-pentanamidoethane-1-sulfonic acid"),
])
def test_taurine_amide_sulfonic_acid_parent(namer, smi, expected):
    name = namer.name(smi)
    assert name == expected, name
    assert same_key(smi, name), name


@pytest.mark.opsin_gate
def test_short_acyl_boundary_unchanged(namer):
    # competing acyl chain <= the 2-C bearing chain: bug does not bite, still names.
    smi = "CC(=O)NCCS(=O)(=O)O"
    name = namer.name(smi)
    assert name and "unknown" not in name, name
    assert same_key(smi, name), name
