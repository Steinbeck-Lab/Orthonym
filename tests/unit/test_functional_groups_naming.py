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
        ("CCC(C)=O", "butan-2-one"),  # Systematic with locant
        ("CCCC(C)=O", "pentan-2-one"),
        ("CCC(CC)=O", "pentan-3-one"),
    ])
    def test_simple_ketones(self, smiles, expected):
        """Test simple ketone naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_propan_2_one_pin(self):
        """Propan-2-one is the IUPAC 2013 PIN (acetone removed from retained names)."""
        assert name_compound("CC(C)=O") == "propan-2-one"

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
        ("CCC(=O)O", "propanoic acid"),   # Systematic
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


# ============================================================================
# Combined Integration Tests
# ============================================================================

class TestFunctionalGroupNamingIntegration:
    """Integration tests for functional group naming."""

    @pytest.mark.integration
    def test_non_terminal_groups_include_locant(self):
        """Non-terminal groups (alcohol, ketone) include locant."""
        # Alcohol at non-terminal position
        assert "-2-" in name_compound("CC(O)C")  # propan-2-ol
        # Ketone always non-terminal
        assert "-2-" in name_compound("CCC(C)=O")  # butan-2-one


# ============================================================================
# Multiplier Prefix Tests (Simple and Complex, including >20)
# ============================================================================

class TestMultiplierPrefix:
    """Test get_multiplier_prefix for all count ranges including >20."""

    @pytest.mark.unit
    @pytest.mark.parametrize("count,name,expected", [
        # Regression: standard simple multipliers
        (1, "fluoro", ""),
        (2, "fluoro", "di"),
        (3, "fluoro", "tri"),
        (4, "fluoro", "tetra"),
        (5, "fluoro", "penta"),
        (10, "fluoro", "deca"),
        (15, "fluoro", "pentadeca"),
        (20, "fluoro", "icosa"),
        # New: simple multipliers >20
        (21, "fluoro", "henicosa"),
        (22, "fluoro", "docosa"),
        (25, "fluoro", "pentacosa"),
        (30, "fluoro", "triaconta"),
    ])
    def test_simple_multiplier_prefix(self, count, name, expected):
        """Simple multiplier prefixes (di/tri/tetra/henicosa/docosa/...)."""
        from orthonym.assembly.naming_utils import get_multiplier_prefix
        assert get_multiplier_prefix(count, name) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("count,name,expected", [
        # Regression: standard complex multipliers
        (2, "1-methylethyl", "bis"),
        (3, "1-methylethyl", "tris"),
        (4, "1-methylethyl", "tetrakis"),
        (10, "1-methylethyl", "decakis"),
        (20, "1-methylethyl", "icosakis"),
        # New: complex multipliers >20
        (21, "1-methylethyl", "henicosakis"),
        (22, "1-methylethyl", "docosakis"),
    ])
    def test_complex_multiplier_prefix(self, count, name, expected):
        """Complex multiplier prefixes (bis/tris/tetrakis/henicosakis/...)."""
        from orthonym.assembly.naming_utils import get_multiplier_prefix
        assert get_multiplier_prefix(count, name) == expected


@pytest.mark.parametrize(
    "smiles, expected, style",
    [
        pytest.param("OCC[SeH]", "2-selanylethan-1-ol", "pin", id="OCC[SeH]"),
        pytest.param("ClCC", "chloroethane", None, id="ClCC"),
        pytest.param("OCCc1ccc(O)c(O)c1", "4-(2-hydroxyethyl)benzene-1,2-diol", None, id="OCCc1ccc(O)c(O)c1"),
        pytest.param("CCN(CC)c1ccc(N=Nc2ccccc2)cc1", "N,N-diethyl-4-(phenyldiazenyl)aniline", None, id="CCN(CC)c1ccc(N=Nc2ccccc2)cc1"),
        pytest.param("Cc1ccc(O)c(C(C)C)c1", "4-methyl-2-(propan-2-yl)phenol", None, id="Cc1ccc(O)c(C(C)C)c1"),
        pytest.param("CCCCCCCO", "heptan-1-ol", None, id="CCCCCCCO"),
        pytest.param("C(Br)(Cl)F", "bromo(chloro)(fluoro)methane", None, id="C(Br)(Cl)F"),
        pytest.param("CCC#CCC", "hex-3-yne", None, id="CCC#CCC"),
        pytest.param("C=CCCO", "but-3-en-1-ol", None, id="C=CCCO"),
        pytest.param("CC(C)(C)O", "2-methylpropan-2-ol", None, id="CC(C)(C)O"),
    ],
)
def test_merged_functional_group_names(smiles, expected, style):
    """Names that development-era snapshot tests pinned and that no other test asserts.

    These (SMILES, name) pairs were asserted only by test files removed in the test
    slimming change; they move here so the names stay pinned by a rule-level test.
    Removed tests this replaces:
      - tests/unit/rules/test_w2e_p1fg_verified_head.py::TestW2EP1FGVerifiedAtHead::test_selanyl_prefix_with_senior_ol (1)
      - tests/unit/test_perception_171_gold_tripwires.py::test_protect_rows_hold (1)
      - tests/integration/test_calibrated_gate.py::test_canary_subset_stable (3)
      - tests/unit/routing/test_dispatcher.py::TestPerStoutClassDispatch::test_per_stoutclass_dispatch (1)
      - tests/unit/assembly/test_serializer_flip_tripwire.py::test_flipped_class_explicit_field_byte_identical (4)
    Rules cited by those tests:,.
    """
    from orthonym import name_compound
    kwargs = {} if style is None else {"style": style}
    assert name_compound(smiles, **kwargs) == expected
