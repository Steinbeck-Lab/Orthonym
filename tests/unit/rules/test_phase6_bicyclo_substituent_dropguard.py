import pytest
from tests.support.read_back import same_key


@pytest.mark.opsin_gate
def test_nitro_on_norbornane_names(namer):
    smi = "[O-][N+](=O)C1CC2CCC1C2"
    name = namer.name(smi)
    assert name == "2-nitrobicyclo[2.2.1]heptane", name
    assert same_key(smi, name), name


def test_no_silent_drop_returns_name_or_abstains(namer):
    # whichever way, never a wrong constitution: emitted name (if any) must RT.
    smi = "[O-][N+](=O)C1CC2CCC1C2"
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert same_key(smi, name), name
