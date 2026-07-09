"""Wave-2 plan P1AM (2026-07-09) — C1 amide/amidine/polyfunctional rows.

Every expected PIN in this file was OPSIN-2.9.0-parse-verified and
canonical-matched against the evidence SMILES during planning.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestT2CarbonicFamilyParents:
    """P-66.1.1.1.1.3 (BB 32675) + P-68.3.1.2.4 (BB 38623: 'The systematic
    name is the preferred IUPAC name') + P-66.4.2.2 (BB 34480)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NNC(=O)N", "hydrazinecarboxamide"),          # was 'semicarbazide'
        ("NNC(=N)NN", "hydrazinecarboximidohydrazide"),  # was unknown
        ("NC(=N)NN", "hydrazinecarboximidamide"),      # was unknown
    ])
    def test_pins(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_protect_carbonohydrazonic_diamide_unchanged(self):
        # existing exact row in the same dict must keep working
        assert name_compound("NC(=NN)N") == "carbonohydrazonic diamide"
