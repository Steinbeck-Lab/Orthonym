"""W2E-P1FC Task 1 — verify-only pins for P-35.2.2 (BB 18015).

"Substituents formed by subtracting one or more hydrogen atoms from mono-
and dinuclear parent hydrides (see P-21.1, P-21.2)": systematic sulfanyl
(not mercapto), sulfanylidene (not thioxo), selanyl (not selenyl), tellanyl,
selanylidene. All outputs OPSIN-RT verified at HEAD 2026-07-09.
"""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP3522SystematicChalcogenPrefixes:
    @pytest.mark.parametrize("smiles,expected", [
        ("S=C1CCC(CC1)C(=O)O", "4-sulfanylidenecyclohexane-1-carboxylic acid"),
        ("[SeH]CCC(=O)O", "3-selanylpropanoic acid"),
        ("[TeH]CCC(=O)O", "3-tellanylpropanoic acid"),
        ("[Se]=C1CCC(CC1)C(=O)O", "4-selanylidenecyclohexane-1-carboxylic acid"),
    ])
    def test_pinned(self, smiles, expected):
        assert name_compound(smiles) == expected
