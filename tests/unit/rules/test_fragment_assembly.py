"""
Tests for the fragment assembly module.

Tests cover:
- Ester assembly (alkyl alkanoate pattern)
- Amide assembly (N-substituent + acid-amide pattern)
- Glycoside assembly (basic concatenation)
- Acid-to-ate conversion helper
- Acid-to-amide conversion helper
- Acid-to-acyl conversion helper
- Alcohol-to-alkyl conversion helper
- Amine-to-prefix conversion helper
- Edge cases (missing inputs, unknown bond types)
"""

import pytest

from orthonym.decomposition.fragment_assembly import (
    assemble_fragment_name,
    _acid_to_ate,
    _acid_to_amide,
    _acid_to_acyl,
    _alcohol_to_alkyl,
    _amine_to_prefix,
)


# ============================================================================
# Ester assembly tests
# ============================================================================

class TestEsterAssembly:
    """Tests for ester fragment assembly: 'alkyl alkanoate' pattern."""

    def test_ethyl_acetate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "acetic acid", "alkyl": "ethanol"}
        )
        assert result == "ethyl acetate"

    def test_methyl_propanoate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "propanoic acid", "alkyl": "methanol"}
        )
        assert result == "methyl propanoate"

    def test_methyl_benzoate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "benzoic acid", "alkyl": "methanol"}
        )
        assert result == "methyl benzoate"

    def test_propyl_butanoate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "butanoic acid", "alkyl": "propan-1-ol"}
        )
        assert result is not None
        assert "propyl" in result
        assert "butanoate" in result

    def test_ethyl_hexanoate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "hexanoic acid", "alkyl": "ethanol"}
        )
        assert result == "ethyl hexanoate"

    def test_methyl_formate(self):
        result = assemble_fragment_name(
            "ester", {"acid": "formic acid", "alkyl": "methanol"}
        )
        assert result == "methyl formate"

    def test_cyclohexanecarboxylate(self):
        result = assemble_fragment_name(
            "ester",
            {"acid": "cyclohexanecarboxylic acid", "alkyl": "ethanol"},
        )
        assert result is not None
        assert "ethyl" in result
        assert "cyclohexanecarboxylate" in result

    def test_ester_missing_acid(self):
        result = assemble_fragment_name("ester", {"alkyl": "ethanol"})
        assert result is None

    def test_ester_missing_alkyl(self):
        result = assemble_fragment_name("ester", {"acid": "acetic acid"})
        assert result is None


# ============================================================================
# Acid-to-ate conversion tests
# ============================================================================

class TestAcidToAte:
    """Tests for _acid_to_ate() helper."""

    def test_acetic_acid(self):
        assert _acid_to_ate("acetic acid") == "acetate"

    def test_propanoic_acid(self):
        assert _acid_to_ate("propanoic acid") == "propanoate"

    def test_benzoic_acid(self):
        assert _acid_to_ate("benzoic acid") == "benzoate"

    def test_butanoic_acid(self):
        assert _acid_to_ate("butanoic acid") == "butanoate"

    def test_formic_acid(self):
        assert _acid_to_ate("formic acid") == "formate"

    def test_cyclohexanecarboxylic_acid(self):
        assert _acid_to_ate("cyclohexanecarboxylic acid") == "cyclohexanecarboxylate"

    def test_hexanoic_acid(self):
        assert _acid_to_ate("hexanoic acid") == "hexanoate"

    def test_pentanoic_acid(self):
        assert _acid_to_ate("pentanoic acid") == "pentanoate"


# ============================================================================
# Alcohol-to-alkyl conversion tests
# ============================================================================

class TestAlcoholToAlkyl:
    """Tests for _alcohol_to_alkyl() helper."""

    def test_methanol(self):
        assert _alcohol_to_alkyl("methanol") == "methyl"

    def test_ethanol(self):
        assert _alcohol_to_alkyl("ethanol") == "ethyl"

    def test_propan_1_ol(self):
        assert _alcohol_to_alkyl("propan-1-ol") == "propyl"

    def test_butan_1_ol(self):
        assert _alcohol_to_alkyl("butan-1-ol") == "butyl"

    def test_phenol(self):
        assert _alcohol_to_alkyl("phenol") == "phenyl"

    def test_methane_to_methyl(self):
        assert _alcohol_to_alkyl("methane") == "methyl"

    def test_ethane_to_ethyl(self):
        assert _alcohol_to_alkyl("ethane") == "ethyl"

    def test_already_alkyl(self):
        assert _alcohol_to_alkyl("methyl") == "methyl"

    def test_cyclohexanol(self):
        assert _alcohol_to_alkyl("cyclohexanol") == "cyclohexyl"


# ============================================================================
# Amide assembly tests
# ============================================================================

class TestAmideAssembly:
    """Tests for amide fragment assembly."""

    def test_n_methylacetamide(self):
        result = assemble_fragment_name(
            "amide", {"acid": "acetic acid", "amine": "methylamine"}
        )
        assert result is not None
        assert "N-methyl" in result
        assert "acetamide" in result

    def test_n_ethylpropanamide(self):
        result = assemble_fragment_name(
            "amide", {"acid": "propanoic acid", "amine": "ethylamine"}
        )
        assert result is not None
        assert "N-ethyl" in result
        assert "propanamide" in result

    def test_n_phenylbenzamide(self):
        result = assemble_fragment_name(
            "amide", {"acid": "benzoic acid", "amine": "aniline"}
        )
        assert result is not None
        assert "N-phenyl" in result
        assert "benzamide" in result

    def test_amide_missing_acid(self):
        result = assemble_fragment_name("amide", {"amine": "methylamine"})
        assert result is None

    def test_amide_missing_amine(self):
        result = assemble_fragment_name("amide", {"acid": "acetic acid"})
        assert result is None


# ============================================================================
# Acid-to-amide conversion tests
# ============================================================================

class TestAcidToAmide:
    """Tests for _acid_to_amide() helper."""

    def test_acetic_acid(self):
        assert _acid_to_amide("acetic acid") == "acetamide"

    def test_propanoic_acid(self):
        assert _acid_to_amide("propanoic acid") == "propanamide"

    def test_benzoic_acid(self):
        assert _acid_to_amide("benzoic acid") == "benzamide"

    def test_butanoic_acid(self):
        assert _acid_to_amide("butanoic acid") == "butanamide"

    def test_formic_acid(self):
        assert _acid_to_amide("formic acid") == "formamide"

    def test_cyclohexanecarboxylic_acid(self):
        assert _acid_to_amide("cyclohexanecarboxylic acid") == "cyclohexanecarboxamide"


# ============================================================================
# Acid-to-acyl conversion tests
# ============================================================================

class TestAcidToAcyl:
    """Tests for _acid_to_acyl() helper."""

    def test_acetic_acid(self):
        assert _acid_to_acyl("acetic acid") == "acetyl"

    def test_propanoic_acid(self):
        assert _acid_to_acyl("propanoic acid") == "propanoyl"

    def test_benzoic_acid(self):
        assert _acid_to_acyl("benzoic acid") == "benzoyl"

    def test_formic_acid(self):
        assert _acid_to_acyl("formic acid") == "formyl"

    def test_butanoic_acid(self):
        assert _acid_to_acyl("butanoic acid") == "butanoyl"


# ============================================================================
# Amine-to-prefix conversion tests
# ============================================================================

class TestAmineToPrefix:
    """Tests for _amine_to_prefix() helper."""

    def test_methylamine(self):
        assert _amine_to_prefix("methylamine") == "methyl"

    def test_ethylamine(self):
        assert _amine_to_prefix("ethylamine") == "ethyl"

    def test_aniline(self):
        assert _amine_to_prefix("aniline") == "phenyl"

    def test_ethanamine(self):
        assert _amine_to_prefix("ethanamine") == "ethyl"

    def test_propan_1_amine(self):
        assert _amine_to_prefix("propan-1-amine") == "propyl"

    def test_cyclohexanamine(self):
        assert _amine_to_prefix("cyclohexanamine") == "cyclohexyl"


# ============================================================================
# Glycoside assembly tests
# ============================================================================

class TestGlycosideAssembly:
    """Tests for glycoside fragment assembly (basic concatenation)."""

    def test_basic_glycoside(self):
        result = assemble_fragment_name(
            "glycosidic", {"sugar": "glucopyranose", "aglycone": "methanol"}
        )
        assert result == "glucopyranose methanol"

    def test_glycoside_missing_sugar(self):
        result = assemble_fragment_name(
            "glycosidic", {"aglycone": "methanol"}
        )
        assert result is None

    def test_glycoside_missing_aglycone(self):
        result = assemble_fragment_name(
            "glycosidic", {"sugar": "glucopyranose"}
        )
        assert result is None


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_bond_type(self):
        result = assemble_fragment_name(
            "unknown", {"acid": "acetic acid", "alkyl": "ethanol"}
        )
        assert result is None

    def test_empty_fragment_names(self):
        result = assemble_fragment_name("ester", {})
        assert result is None

    def test_none_bond_type(self):
        result = assemble_fragment_name(None, {"acid": "acetic acid"})
        assert result is None

    def test_none_fragment_names(self):
        result = assemble_fragment_name("ester", None)
        assert result is None

    def test_empty_string_bond_type(self):
        result = assemble_fragment_name("", {"acid": "acetic acid"})
        assert result is None

    def test_carbamate_assembly(self):
        result = assemble_fragment_name(
            "carbamate",
            {"alkyl": "ethanol", "amine": "methylamine"},
        )
        assert result is not None
        assert "ethyl" in result
        assert "carbamate" in result
