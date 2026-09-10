"""
End-to-end integration tests for a phase Foundation.

This file validates ALL a phase success criteria and requirements.
"""

import pytest
from orthonym import name_compound


class TestPhase1SuccessCriteria:
    """Validate all 5 a phase success criteria."""

    @pytest.mark.integration
    def test_sc1_simple_alkane(self):
        """SC1: CCCC -> butane"""
        assert name_compound("CCCC") == "butane"

    @pytest.mark.integration
    def test_sc2_branched_alkane(self):
        """SC2: CC(C)C -> 2-methylpropane"""
        assert name_compound("CC(C)C") == "2-methylpropane"

    @pytest.mark.integration
    def test_sc3_functional_group_with_locant(self):
        """SC3: CCCO -> propan-1-ol"""
        assert name_compound("CCCO") == "propan-1-ol"

    @pytest.mark.integration
    def test_sc4_multiplicative_prefix(self):
        """SC4: CC(C)(C)CC -> 2,2-dimethylbutane"""
        assert name_compound("CC(C)(C)CC") == "2,2-dimethylbutane"


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

    # FOUND-03: First-point-of-difference
    @pytest.mark.integration
    def test_found03_first_point_of_difference(self):
        """FOUND-03: First-point-of-difference rule for locant sets.

        For 2,3-dimethylbutane vs other numberings,
        the [2,3] set wins over alternatives.
        """
        result = name_compound("CC(C)C(C)C")
        assert result == "2,3-dimethylbutane"

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

    # FOUND-06: Simple alkanes
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCCCC", "pentane"),
        ("CCCCCC", "hexane"),
        ("CCCCCCC", "heptane"),
    ])
    def test_found06_simple_alkanes(self, smiles, expected):
        """FOUND-06: Simple alkane naming."""
        assert name_compound(smiles) == expected

    # FOUND-07: Branched alkanes
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)CC", "2-methylbutane"),
        ("CCC(CC)CC", "3-ethylpentane"),
    ])
    def test_found07_branched_alkanes(self, smiles, expected):
        """FOUND-07: Branched alkane naming."""
        assert name_compound(smiles) == expected

    # FOUND-08: Alcohols
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCCO", "propan-1-ol"),
        ("CCC(O)C", "butan-2-ol"),
    ])
    def test_found08_alcohols(self, smiles, expected):
        """FOUND-08: Alcohol naming with locants."""
        assert name_compound(smiles) == expected

    # FOUND-09: Aldehydes
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCC=O", "propanal"),
        ("CCCC=O", "butanal"),
    ])
    def test_found09_aldehydes(self, smiles, expected):
        """FOUND-09: Aldehyde naming (no locant for terminal group)."""
        assert name_compound(smiles) == expected

    # FOUND-10: Ketones
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCC(C)=O", "butan-2-one"),
        ("CCC(CC)=O", "pentan-3-one"),
    ])
    def test_found10_ketones(self, smiles, expected):
        """FOUND-10: Ketone naming with locants."""
        assert name_compound(smiles) == expected

    # FOUND-11: Carboxylic acids
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCC(=O)O", "propanoic acid"),
        ("CCCC(=O)O", "butanoic acid"),
    ])
    def test_found11_carboxylic_acids(self, smiles, expected):
        """FOUND-11: Carboxylic acid naming."""
        assert name_compound(smiles) == expected

    # FOUND-12: Alkenes
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C=CCC", "but-1-ene"),
        ("CC=CC", "but-2-ene"),
    ])
    def test_found12_alkenes(self, smiles, expected):
        """FOUND-12: Alkene naming with locants."""
        assert name_compound(smiles) == expected

    # FOUND-13: Alkynes
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C#CCC", "but-1-yne"),
        ("CC#CC", "but-2-yne"),
    ])
    def test_found13_alkynes(self, smiles, expected):
        """FOUND-13: Alkyne naming with locants."""
        assert name_compound(smiles) == expected

    # FOUND-14: Enynes
    @pytest.mark.integration
    def test_found14_enynes(self):
        """FOUND-14: Enyne naming with double bond priority."""
        result = name_compound("C=CC#C")
        assert result == "but-1-en-3-yne"

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

    @pytest.mark.integration
    def test_ethane_two_carbons(self):
        """Ethane (two carbons)."""
        assert name_compound("CC") == "ethane"

    @pytest.mark.integration
    def test_decane_ten_carbons(self):
        """Decane (ten carbons)."""
        assert name_compound("CCCCCCCCCC") == "decane"

    @pytest.mark.integration
    def test_retained_name_benzene(self):
        """Benzene (retained name)."""
        assert name_compound("c1ccccc1") == "benzene"

    @pytest.mark.integration
    def test_retained_name_methanol(self):
        """Methanol (retained name)."""
        assert name_compound("CO") == "methanol"

    @pytest.mark.integration
    def test_retained_name_ethanol(self):
        """Ethanol (retained name)."""
        assert name_compound("CCO") == "ethanol"

    @pytest.mark.integration
    def test_propan_2_one_pin(self):
        """Propan-2-one is the IUPAC 2013 PIN (not acetone retained name)."""
        assert name_compound("CC(C)=O") == "propan-2-one"
