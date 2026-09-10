"""W2E-P1FC Task 7 — (BB 18114): complex prefixes by concatenation.
"(benzyloxy)carbonyl (preferred prefix)" for -CO-O-CH2-C6H5.
benzyl carbonochloridate = the acyl chloride of mono-benzyl carbonic acid."""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP3542Concatenation:
    @pytest.mark.parametrize("smiles", [
        "O=C(Cl)OCc1ccccc1",
        "O=C(OCc1ccccc1)Cl",
    ])
    def test_benzyl_carbonochloridate(self, smiles):
        assert name_compound(smiles) == "benzyl carbonochloridate"

    def test_ethyl_carbonochloridate(self):
        # The recognizer generalizes to any nameable O-alkyl. OPSIN-RT verified.
        assert name_compound("CCOC(=O)Cl") == "ethyl carbonochloridate"
