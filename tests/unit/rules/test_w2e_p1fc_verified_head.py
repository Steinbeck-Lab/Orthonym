"""W2E-P1FC Task 1 — verify-only pins for (BB 18015).

"Substituents formed by subtracting one or more hydrogen atoms from mono-
and dinuclear parent hydrides (see, ": systematic sulfanyl
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


@pytest.mark.unit
class TestP59218SubstitutedSubstituent:
    """ (BB 25269): subsidiary substituents named as prefixes;
    attachment point takes the lowest locant. Verified healed at HEAD."""

    def test_dichlorocyclohexanecarboxylic_acid(self):
        assert name_compound("ClC1(C(CCCC1)C(=O)O)Cl") == \
            "2,2-dichlorocyclohexane-1-carboxylic acid"
