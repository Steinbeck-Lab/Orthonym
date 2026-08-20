import pytest
from orthonym.namer import Orthonym
from orthonym.errors import OrthonymLimitError

WILDCARD_MSG = "compound with wildcard atoms (not supported)"

def _name(smiles):
    return Orthonym().name(smiles)   # default raise_on_limit=False, as the CLI calls it

def test_simple_wildcard_does_not_ship_ethane():
    out = _name("CC*")
    assert out == WILDCARD_MSG
    assert out != "ethane"

def test_double_wildcard_does_not_ship_pentane():
    assert _name("*CCCCC*") == WILDCARD_MSG

def test_wildcard_ester_does_not_ship_ethyl_formate():
    assert _name("CCOC(=O)*") == WILDCARD_MSG

def test_isotope_over_wildcard_abstains():
    # [2H]CC* : the isotope decorator runs first (2847) and its recursive skeleton
    # name is now the wildcard sentinel -> decorator fails closed -> abstain.
    assert _name("[2H]CC*") == WILDCARD_MSG

def test_raise_on_limit_still_raises():
    with pytest.raises(OrthonymLimitError) as exc:
        Orthonym().name("CC*", raise_on_limit=True)
    assert exc.value.code == "WILDCARD_ATOMS"

@pytest.mark.parametrize("smiles,expected_contains", [
    ("CC[CH2+]", "propylium"),                     # charged, RT-valid — unchanged
    ("[O-]C(=O)CC[N+](C)(C)C", "azaniumyl"),       # zwitterion — unchanged
])
def test_charged_probes_unchanged(smiles, expected_contains):
    assert expected_contains in _name(smiles)

def test_plain_alcohol_unchanged():
    assert _name("CCO") == "ethanol"
