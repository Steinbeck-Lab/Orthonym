"""
End-to-end tests for functional group suffix naming.

Tests the complete naming pipeline for:
- Alcohols: with locants (propan-1-ol, butan-2-ol)
- Aldehydes: without locants (propanal, butanal)
- Ketones: with locants (butan-2-one, pentan-3-one)
- Carboxylic acids: without locants (propanoic acid, butanoic acid)
- Vowel elision: propan-1-ol not propane-1-ol
"""

import pytest
from orthonym import name_compound


# ============================================================================
# Alcohol Naming Tests
# ============================================================================

class TestAlcoholNaming:
    """Test alcohol naming with correct locants and vowel elision."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CO", "methanol"),          # Retained name
        ("CCO", "ethanol"),          # Retained name
        ("CCCO", "propan-1-ol"),     # Systematic, PIN style
        ("CC(O)C", "propan-2-ol"),   # Locant 2
        ("CCCCO", "butan-1-ol"),     # Locant 1
        ("CCC(O)C", "butan-2-ol"),   # Locant 2
    ])
    def test_simple_alcohols(self, smiles, expected):
        """Test simple alcohol naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_methanol_retained(self):
        """Methanol is a retained name, not metan-1-ol."""
        assert name_compound("CO") == "methanol"

    @pytest.mark.unit
    def test_ethanol_retained(self):
        """Ethanol is a retained name, not ethan-1-ol."""
        assert name_compound("CCO") == "ethanol"

    @pytest.mark.unit
    def test_propan_1_ol_systematic(self):
        """Propan-1-ol uses PIN-style infix locant."""
        result = name_compound("CCCO")
        assert result == "propan-1-ol"
        # Verify vowel elision (not propane-1-ol)
        assert "propane" not in result

    @pytest.mark.unit
    def test_propan_2_ol_locant(self):
        """Propan-2-ol has locant 2 for secondary alcohol."""
        result = name_compound("CC(O)C")
        assert result == "propan-2-ol"
        assert "-2-" in result

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCCCCO", "pentan-1-ol"),
        ("CCCCC(O)C", "hexan-2-ol"),
        ("CCC(O)CC", "pentan-3-ol"),
    ])
    def test_longer_alcohols(self, smiles, expected):
        """Test alcohols with longer chains."""
        assert name_compound(smiles) == expected


# ============================================================================
# Aldehyde Naming Tests
# ============================================================================

class TestAldehydeNaming:
    """Test aldehyde naming without locants (terminal groups)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C=O", "formaldehyde"),      # Retained name
        ("CC=O", "acetaldehyde"),     # Retained name
        ("CCC=O", "propanal"),        # Systematic
        ("CCCC=O", "butanal"),        # Systematic
        ("CCCCC=O", "pentanal"),      # Systematic
    ])
    def test_simple_aldehydes(self, smiles, expected):
        """Test simple aldehyde naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_formaldehyde_retained(self):
        """Formaldehyde is a retained name."""
        assert name_compound("C=O") == "formaldehyde"

    @pytest.mark.unit
    def test_acetaldehyde_retained(self):
        """Acetaldehyde is a retained name."""
        assert name_compound("CC=O") == "acetaldehyde"

    @pytest.mark.unit
    def test_propanal_no_locant(self):
        """Propanal has no locant (terminal group, always at position 1)."""
        result = name_compound("CCC=O")
        assert result == "propanal"
        # Verify no locant in name
        assert "-1-" not in result
        assert result == "propanal"  # Not propan-1-al

    @pytest.mark.unit
    def test_butanal_vowel_elision(self):
        """Butanal shows vowel elision (not butaneal)."""
        result = name_compound("CCCC=O")
        assert result == "butanal"
        # Not "butaneal" or "butane-al"
        assert "butaneal" not in result


# ============================================================================
# Ketone Naming Tests
# ============================================================================

class TestKetoneNaming:
    """Test ketone naming with correct locants."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)=O", "acetone"),       # Retained name
        ("CCC(C)=O", "butan-2-one"),  # Systematic with locant
        ("CCCC(C)=O", "pentan-2-one"),
        ("CCC(CC)=O", "pentan-3-one"),
    ])
    def test_simple_ketones(self, smiles, expected):
        """Test simple ketone naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_acetone_retained(self):
        """Acetone is a retained name, not propan-2-one."""
        assert name_compound("CC(C)=O") == "acetone"

    @pytest.mark.unit
    def test_butan_2_one_locant(self):
        """Butan-2-one requires locant in PIN style."""
        result = name_compound("CCC(C)=O")
        assert result == "butan-2-one"
        assert "-2-" in result

    @pytest.mark.unit
    def test_pentan_3_one_locant(self):
        """Pentan-3-one has ketone at position 3."""
        result = name_compound("CCC(CC)=O")
        assert result == "pentan-3-one"
        assert "-3-" in result

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CCCCC(C)=O", "hexan-2-one"),
        ("CCCC(CC)=O", "hexan-3-one"),
    ])
    def test_longer_ketones(self, smiles, expected):
        """Test ketones with longer chains."""
        assert name_compound(smiles) == expected


# ============================================================================
# Carboxylic Acid Naming Tests
# ============================================================================

class TestCarboxylicAcidNaming:
    """Test carboxylic acid naming without locants."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C(=O)O", "formic acid"),        # Retained name
        ("CC(=O)O", "acetic acid"),       # Retained name
        ("CCC(=O)O", "propanoic acid"),   # Systematic
        ("CCCC(=O)O", "butanoic acid"),   # Systematic
        ("CCCCC(=O)O", "pentanoic acid"), # Systematic
    ])
    def test_simple_acids(self, smiles, expected):
        """Test simple carboxylic acid naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_formic_acid_retained(self):
        """Formic acid is a retained name."""
        assert name_compound("C(=O)O") == "formic acid"

    @pytest.mark.unit
    def test_acetic_acid_retained(self):
        """Acetic acid is a retained name."""
        assert name_compound("CC(=O)O") == "acetic acid"

    @pytest.mark.unit
    def test_propanoic_acid_no_locant(self):
        """Propanoic acid has no locant (terminal group)."""
        result = name_compound("CCC(=O)O")
        assert result == "propanoic acid"
        # No locant in name
        assert "-1-" not in result

    @pytest.mark.unit
    def test_butanoic_acid_vowel_elision(self):
        """Butanoic acid shows vowel elision (not butaneoic acid)."""
        result = name_compound("CCCC(=O)O")
        assert result == "butanoic acid"


# ============================================================================
# Vowel Elision Tests (End-to-End)
# ============================================================================

class TestVowelElisionEndToEnd:
    """Test vowel elision through the complete naming pipeline."""

    @pytest.mark.unit
    def test_propanol_elision(self):
        """propan-1-ol not propane-1-ol."""
        result = name_compound("CCCO")
        assert result == "propan-1-ol"
        assert "propane-1-ol" != result

    @pytest.mark.unit
    def test_propanal_elision(self):
        """propanal not propaneal or propane-al."""
        result = name_compound("CCC=O")
        assert result == "propanal"
        assert "propaneal" != result

    @pytest.mark.unit
    def test_butanone_elision(self):
        """butan-2-one not butane-2-one."""
        result = name_compound("CCC(C)=O")
        assert result == "butan-2-one"
        assert "butane-2-one" != result

    @pytest.mark.unit
    def test_propanoic_acid_elision(self):
        """propanoic acid shows elision (not propaneoic acid)."""
        result = name_compound("CCC(=O)O")
        assert result == "propanoic acid"
        assert "propaneoic" not in result


# ============================================================================
# Retained Name Precedence Tests
# ============================================================================

class TestRetainedNamePrecedence:
    """Test that retained names take precedence over systematic names."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,retained_name", [
        ("CO", "methanol"),
        ("CCO", "ethanol"),
        ("C=O", "formaldehyde"),
        ("CC=O", "acetaldehyde"),
        ("CC(C)=O", "acetone"),
        ("C(=O)O", "formic acid"),
        ("CC(=O)O", "acetic acid"),
    ])
    def test_retained_names_checked_first(self, smiles, retained_name):
        """Retained names should be returned instead of systematic names."""
        result = name_compound(smiles)
        assert result == retained_name


# ============================================================================
# Combined Integration Tests
# ============================================================================

class TestFunctionalGroupNamingIntegration:
    """Integration tests for functional group naming."""

    @pytest.mark.integration
    def test_alcohol_locant_always_included_in_pin(self):
        """In PIN style, alcohol locants are always included."""
        # Even when locant seems "obvious"
        assert name_compound("CCCO") == "propan-1-ol"  # Not just propanol
        assert name_compound("CC(O)C") == "propan-2-ol"

    @pytest.mark.integration
    def test_terminal_groups_no_locant(self):
        """Terminal groups (aldehyde, acid) don't include locant."""
        assert name_compound("CCC=O") == "propanal"  # Not propan-1-al
        assert name_compound("CCC(=O)O") == "propanoic acid"  # Not propan-1-oic acid

    @pytest.mark.integration
    def test_non_terminal_groups_include_locant(self):
        """Non-terminal groups (alcohol, ketone) include locant."""
        # Alcohol at non-terminal position
        assert "-2-" in name_compound("CC(O)C")  # propan-2-ol
        # Ketone always non-terminal
        assert "-2-" in name_compound("CCC(C)=O")  # butan-2-one
