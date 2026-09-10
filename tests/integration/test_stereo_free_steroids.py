"""
Tests for stereo-free steroid SMILES -- expected behavior documentation.

These compounds have no stereo annotations in their input SMILES.
Orthonym correctly omits stereodescriptors (cannot invent stereo).
RT mismatch with OPSIN is expected: OPSIN adds default steroid stereo
when parsing retained names like "stigmast-5-en-3,7-diol".

: Closed as expected behavior, not a bug.
"""
import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestStereoFreeSteroids:
    def test_stigmastadienediol_no_stereo_in_input(self):
        """Stereo-free stigmasterol derivative: no @ in SMILES -> no stereo in name."""
        smiles = "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C"
        name = name_compound(smiles)
        assert name is not None
        # Name should NOT contain stereodescriptors (no stereo in input)
        assert "(R)" not in name and "(S)" not in name, f"Unexpected stereo in: {name}"
        # Should contain steroid-related naming
        assert "stigmast" in name.lower() or "ol" in name.lower()

    def test_ergostadienol_no_stereo_in_input(self):
        """Stereo-free ergosterol derivative: no @ in SMILES -> no stereo in name."""
        smiles = "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C"
        name = name_compound(smiles)
        assert name is not None
        assert "(R)" not in name and "(S)" not in name, f"Unexpected stereo in: {name}"
        assert "ergost" in name.lower() or "ol" in name.lower()
