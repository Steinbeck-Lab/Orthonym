import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P1ChainsAVerify:
    """Rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        ("[PH3]", "phosphane"),                              # P-52.1.1
        ("[SnH3]O[SnH2]O[SnH3]", "tristannoxane"),           # P-52.1.3
        ("N1CCOCCCCCCCC1", "1-oxa-4-azacyclododecane"),      # P-22.2.3.2.3
        ("C[C@H](Cl)[C@@H](Cl)C", "(2S,3S)-2,3-dichlorobutane"),  # P-44.4.1.12
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected
