"""
End-to-end integration tests for a phase Foundation.

This file validates ALL a phase success criteria and requirements.
"""

import pytest
from orthonym import name_compound


class TestPhase1Requirements:
    """Test representative compounds for each FOUND requirement."""

    # FOUND-01: Substituent prefix generation
    @pytest.mark.integration
    def test_found01_substituent_prefix(self):
        """FOUND-01: Substituent prefix generation."""
        # 2-methylbutane should have "methyl" prefix
        result = name_compound("CC(C)CC")
        assert "methyl" in result

    # FOUND-02: Locant calculation
    @pytest.mark.integration
    def test_found02_locant_calculation(self):
        """FOUND-02: Locant calculation."""
        # 2-methylbutane should have "2-" locant
        result = name_compound("CC(C)CC")
        assert result == "2-methylbutane"
        assert "2-methyl" in result

    # FOUND-04: Alphabetization
    @pytest.mark.integration
    def test_found04_alphabetization(self):
        """FOUND-04: Alphabetization of substituents.

        Ethyl comes before methyl alphabetically.
        """
        # 3-ethyl-4-methylhexane
        result = name_compound("CCC(CC)C(C)CC")
        assert result == "3-ethyl-4-methylhexane"
        # Verify ethyl comes before methyl
        assert result.index("ethyl") < result.index("methyl")

    # FOUND-05: Multiplicative prefixes
    @pytest.mark.integration
    def test_found05_multiplicative_prefixes(self):
        """FOUND-05: Multiplicative prefixes (di-, tri-)."""
        # 2,2-dimethylbutane should have "di" prefix
        result = name_compound("CC(C)(C)CC")
        assert result == "2,2-dimethylbutane"
        assert "dimethyl" in result

    # FOUND-15: Vowel elision
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected,no_double_e", [
        ("CCCO", "propan-1-ol", True),  # propan not propane
        ("CCC=O", "propanal", True),  # not propaneal
    ])
    def test_found15_vowel_elision(self, smiles, expected, no_double_e):
        """FOUND-15: Vowel elision applied correctly."""
        result = name_compound(smiles)
        assert result == expected
        if no_double_e:
            assert "ee" not in result  # No double 'e' from elision


class TestOPSINRoundTrip:
    """Round-trip validation with OPSIN.

    Generate name -> OPSIN parse -> compare canonical SMILES.
    This test is marked with roundtrip and will skip if OPSIN unavailable.
    """

    @pytest.mark.roundtrip
    def test_pentane_roundtrip(self, opsin_to_smiles, canonical):
        """Test pentane round-trip."""
        smiles = "CCCCC"
        name = name_compound(smiles)
        assert name == "pentane"

        # Parse back through OPSIN
        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_2_methylbutane_roundtrip(self, opsin_to_smiles, canonical):
        """Test 2-methylbutane round-trip."""
        smiles = "CC(C)CC"
        name = name_compound(smiles)
        assert name == "2-methylbutane"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_propan_1_ol_roundtrip(self, opsin_to_smiles, canonical):
        """Test propan-1-ol round-trip."""
        smiles = "CCCO"
        name = name_compound(smiles)
        assert name == "propan-1-ol"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_butan_2_one_roundtrip(self, opsin_to_smiles, canonical):
        """Test butan-2-one round-trip."""
        smiles = "CCC(C)=O"
        name = name_compound(smiles)
        assert name == "butan-2-one"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_propanoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        """Test propanoic acid round-trip."""
        smiles = "CCC(=O)O"
        name = name_compound(smiles)
        assert name == "propanoic acid"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_but_1_ene_roundtrip(self, opsin_to_smiles, canonical):
        """Test but-1-ene round-trip."""
        smiles = "C=CCC"
        name = name_compound(smiles)
        assert name == "but-1-ene"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_but_1_yne_roundtrip(self, opsin_to_smiles, canonical):
        """Test but-1-yne round-trip."""
        smiles = "C#CCC"
        name = name_compound(smiles)
        assert name == "but-1-yne"

        parsed_smiles = opsin_to_smiles(name)
        assert parsed_smiles is not None, f"OPSIN could not parse the emitted name {name!r}"
        assert canonical(parsed_smiles) == canonical(smiles)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_methane_single_carbon(self):
        """Methane (single carbon)."""
        assert name_compound("C") == "methane"


