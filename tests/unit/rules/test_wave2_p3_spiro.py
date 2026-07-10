import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P3SpiroVerify:
    """Rows already correct at HEAD — lock them against regression before
    the engine + scorer edits in this plan."""

    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCC2(CC1)CC=CC2", "spiro[4.5]dec-2-ene"),   # P-31.1.5.1
        ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),    # heterospiro control
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected
