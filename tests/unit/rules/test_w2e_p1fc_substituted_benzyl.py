"""W2E-P1FC Task 4 — (BB 16304): ring-substituted benzyl is the
systematic (substituted-phenyl)methyl in PINs, not (n-halobenzyl).
BB example: 2-[(4-bromophenyl)methyl]pyridine (PIN)."""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP29621SubstitutedBenzyl:
    @pytest.mark.parametrize("smiles,expected", [
        ("Clc1ccc(Cc2ccccn2)cc1", "2-[(4-chlorophenyl)methyl]pyridine"),
        ("Brc1ccc(Cc2ccccn2)cc1", "2-[(4-bromophenyl)methyl]pyridine"),
    ])
    def test_substituted_benzyl_on_pyridine(self, smiles, expected):
        assert name_compound(smiles) == expected
