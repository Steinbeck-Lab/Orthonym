import pytest
from tests.support.read_back import same_key


@pytest.mark.opsin_gate
def test_acetoxymethyl_cyclohexanecarboxylic_acid(namer):
    smi = "CC(=O)OCC1CCCCC1C(=O)O"
    name = namer.name(smi)
    assert name == "2-[(acetyloxy)methyl]cyclohexane-1-carboxylic acid", name
    assert same_key(smi, name), name


@pytest.mark.opsin_gate
def test_hydroxymethyl_control_unchanged(namer):
    # same skeleton, -CH2OH instead of the ester: must stay correct.
    smi = "OCC1CCCCC1C(=O)O"
    assert namer.name(smi) == "2-(hydroxymethyl)cyclohexane-1-carboxylic acid"


@pytest.mark.opsin_gate
def test_mirror_shape_methoxycarbonyl_ring_diacid_monoester(namer):
    # The direction-agnostic guard also defers the MIRROR shape
    # Ring-C(=O)-O-R (a mono-alkyl ester of a ring diacid), not just the
    # Ring-CH2-O-C(=O)R acyloxy shape above -- desired breadth, not a leak.
    smi = "COC(=O)C1CCCCC1C(=O)O"
    name = namer.name(smi)
    assert name == "2-(methoxycarbonyl)cyclohexane-1-carboxylic acid", name
    assert same_key(smi, name), name


@pytest.mark.opsin_gate
def test_mirror_shape_ethyl_isopropyl_fail_closed(namer):
    # Same mirror shape with longer O-alkyl groups currently abstains rather
    # than emitting a name; lock that it NEVER goes wrong (0-wrong is
    # absolute) even though it isn't yet rescued.
    for smi in ["CCOC(=O)C1CCCCC1C(=O)O", "CC(C)OC(=O)C1CCCCC1C(=O)O"]:
        name = namer.name(smi)
        assert (not name) or ("unknown" in name) or same_key(smi, name), name
