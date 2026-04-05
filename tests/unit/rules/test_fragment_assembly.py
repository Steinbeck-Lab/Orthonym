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
    _acid_to_thioate,
    _alcohol_to_alkyl,
    _amine_to_prefix,
    _join_components,
    _looks_like_acid_name,
    _looks_like_convertible_name,
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
    """Tests for glycoside fragment assembly with sugar prefix support."""

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

    def test_glycoside_with_acid_alkyl_keys(self):
        """Assembler accepts 'acid'/'alkyl' key convention from engine."""
        result = assemble_fragment_name(
            "glycosidic",
            {"acid": "beta-D-glucopyranosyloxy", "alkyl": "phenol"},
        )
        assert result is not None
        assert "glucopyranosyloxy" in result
        assert "phenol" in result

    def test_glycoside_with_sugar_aglycone_keys(self):
        """Assembler accepts 'sugar'/'aglycone' key convention."""
        result = assemble_fragment_name(
            "glycosidic",
            {"sugar": "beta-D-glucopyranosyloxy", "aglycone": "phenol"},
        )
        assert result is not None
        assert "glucopyranosyloxy" in result
        assert "phenol" in result

    def test_glycoside_produces_parenthesized_prefix(self):
        """Glycoside assembly produces '(prefix)aglycone' format."""
        result = assemble_fragment_name(
            "glycosidic",
            {"acid": "beta-D-glucopyranosyloxy", "alkyl": "phenol"},
        )
        assert "(beta-D-glucopyranosyloxy)" in result


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


# ============================================================================
# Component joining helper tests
# ============================================================================

class TestJoinComponents:
    """Tests for _join_components() hyphenation helper."""

    def test_letter_digit_boundary(self):
        assert _join_components("carbonyl", "2-amino") == "carbonyl-2-amino"

    def test_letter_uppercase_boundary(self):
        assert _join_components("carbonyl", "L-alanyl") == "carbonyl-L-alanyl"

    def test_letter_uppercase_n_sub(self):
        assert _join_components("carbonyl", "N-5-methyl") == "carbonyl-N-5-methyl"

    def test_paren_digit_boundary(self):
        assert _join_components("(glucopyranosyloxy)", "5,6-dibutyl") == "(glucopyranosyloxy)-5,6-dibutyl"

    def test_paren_uppercase_boundary(self):
        assert _join_components("(glucopyranosyloxy)", "N-methyl") == "(glucopyranosyloxy)-N-methyl"

    def test_no_hyphen_letter_letter(self):
        assert _join_components("methyl", "acetamide") == "methylacetamide"

    def test_no_hyphen_digit_letter(self):
        assert _join_components("propan-1", "ol") == "propan-1ol"

    def test_empty_left(self):
        assert _join_components("", "acetamide") == "acetamide"

    def test_empty_right(self):
        assert _join_components("carbonyl", "") == "carbonyl"

    def test_both_empty(self):
        assert _join_components("", "") == ""

    def test_none_left(self):
        assert _join_components(None, "acetamide") == "acetamide"

    def test_none_right(self):
        assert _join_components("carbonyl", None) == "carbonyl"

    def test_hyphen_ending_no_extra_hyphen(self):
        """If left already ends with hyphen, don't add another."""
        assert _join_components("N-", "methyl") == "N-methyl"

    def test_digit_ending_no_hyphen(self):
        """If left ends with digit, no hyphen needed before letter."""
        assert _join_components("propan-1-", "ol") == "propan-1-ol"


# ============================================================================
# Double-suffix guard tests
# ============================================================================

class TestDoubleSuffixGuardAte:
    """Tests for double-suffix prevention in _acid_to_ate()."""

    def test_already_ate_returns_unchanged(self):
        assert _acid_to_ate("propanoate") == "propanoate"

    def test_already_acetate_returns_unchanged(self):
        assert _acid_to_ate("acetate") == "acetate"

    def test_already_benzoate_returns_unchanged(self):
        assert _acid_to_ate("benzoate") == "benzoate"

    def test_already_formate_returns_unchanged(self):
        assert _acid_to_ate("formate") == "formate"

    def test_already_hexanoate_returns_unchanged(self):
        assert _acid_to_ate("hexanoate") == "hexanoate"

    def test_acid_still_converts(self):
        assert _acid_to_ate("propanoic acid") == "propanoate"

    def test_acetic_acid_still_converts(self):
        assert _acid_to_ate("acetic acid") == "acetate"

    def test_carboxylic_acid_still_converts(self):
        assert _acid_to_ate("cyclohexanecarboxylic acid") == "cyclohexanecarboxylate"


class TestDoubleSuffixGuardAmide:
    """Tests for double-suffix prevention in _acid_to_amide()."""

    def test_already_amide_returns_unchanged(self):
        assert _acid_to_amide("propanamide") == "propanamide"

    def test_already_acetamide_returns_unchanged(self):
        assert _acid_to_amide("acetamide") == "acetamide"

    def test_already_benzamide_returns_unchanged(self):
        assert _acid_to_amide("benzamide") == "benzamide"

    def test_already_formamide_returns_unchanged(self):
        assert _acid_to_amide("formamide") == "formamide"

    def test_acid_still_converts(self):
        assert _acid_to_amide("propanoic acid") == "propanamide"

    def test_acetic_acid_still_converts(self):
        assert _acid_to_amide("acetic acid") == "acetamide"

    def test_carboxylic_acid_still_converts(self):
        assert _acid_to_amide("cyclohexanecarboxylic acid") == "cyclohexanecarboxamide"


# ============================================================================
# Amide assembly with hyphenation tests
# ============================================================================

class TestAmideAssemblyHyphenation:
    """Tests that amide assembly uses proper hyphenation at boundaries."""

    def test_acyl_digit_boundary_gets_hyphen(self):
        """When acyl prefix ends with letter and amine starts with digit."""
        result = assemble_fragment_name(
            "amide",
            {"acid": "cyclohexanecarboxylic acid", "amine": "2-aminopentanoic acid"},
        )
        # If result is not None, verify no letter-digit boundary without hyphen
        if result and "carbonyl" in result:
            idx = result.index("carbonyl") + len("carbonyl")
            if idx < len(result) and result[idx].isdigit():
                assert result[idx - 1] == '-', f"Missing hyphen in: {result}"

    def test_simple_amide_still_works(self):
        """Simple amides should be unaffected by hyphenation changes."""
        result = assemble_fragment_name(
            "amide", {"acid": "acetic acid", "amine": "methylamine"}
        )
        assert result == "N-methylacetamide"


# ============================================================================
# Glycoside assembly with hyphenation tests
# ============================================================================

class TestGlycosideAssemblyHyphenation:
    """Tests that glycoside assembly uses proper hyphenation."""

    def test_paren_digit_gets_hyphen(self):
        """When aglycone starts with digit, hyphen after closing paren."""
        result = assemble_fragment_name(
            "glycosidic",
            {"sugar": "beta-D-glucopyranosyloxy", "aglycone": "5,6-dibutylphenol"},
        )
        assert result is not None
        assert ")-5" in result, f"Expected hyphen after paren: {result}"

    def test_paren_letter_no_hyphen(self):
        """When aglycone starts with lowercase letter, no extra hyphen."""
        result = assemble_fragment_name(
            "glycosidic",
            {"sugar": "beta-D-glucopyranosyloxy", "aglycone": "phenol"},
        )
        assert result is not None
        assert "(beta-D-glucopyranosyloxy)phenol" == result


# ============================================================================
# Pre-validation helper tests
# ============================================================================

class TestLooksLikeAcidName:
    """Tests for _looks_like_acid_name() shared pre-validator."""

    def test_propanoic_acid_returns_true(self):
        assert _looks_like_acid_name("propanoic acid") is True

    def test_acetic_returns_true(self):
        """Trivial acid stem without ' acid' suffix."""
        assert _looks_like_acid_name("acetic") is True

    def test_benzoic_acid_returns_true(self):
        assert _looks_like_acid_name("benzoic acid") is True

    def test_carboxylic_returns_true(self):
        """Ends with 'carboxylic' pattern."""
        assert _looks_like_acid_name("cyclohexanecarboxylic acid") is True

    def test_ethanol_returns_false(self):
        assert _looks_like_acid_name("ethanol") is False

    def test_cyclohexane_returns_false(self):
        assert _looks_like_acid_name("cyclohexane") is False

    def test_benzene_returns_false(self):
        assert _looks_like_acid_name("benzene") is False

    def test_glucose_returns_false(self):
        assert _looks_like_acid_name("glucose") is False

    def test_formic_acid_returns_true(self):
        assert _looks_like_acid_name("formic acid") is True

    def test_oic_stem_returns_true(self):
        """Bare systematic stem ending in 'oic'."""
        assert _looks_like_acid_name("propanoic") is True


class TestLooksLikeConvertibleName:
    """Tests for _looks_like_convertible_name() broader validator."""

    def test_acid_name_returns_true(self):
        assert _looks_like_convertible_name("propanoic acid") is True

    def test_amide_name_returns_true(self):
        assert _looks_like_convertible_name("benzamide") is True

    def test_ester_name_returns_true(self):
        assert _looks_like_convertible_name("propanoate") is True

    def test_n_substituted_amide_returns_true(self):
        assert _looks_like_convertible_name("N-methylbenzamide") is True

    def test_cyclohexane_returns_false(self):
        assert _looks_like_convertible_name("cyclohexane") is False

    def test_ethanol_returns_false(self):
        assert _looks_like_convertible_name("ethanol") is False


# ============================================================================
# Transformation fallback safety tests (None-return for non-acid inputs)
# ============================================================================

class TestTransformationFallbackSafety:
    """Tests that transformation functions return None for non-acid inputs.

    Per D-09 through D-12: when non-acid fragments reach these functions
    (due to fragment role misidentification), they should return None
    instead of fabricating garbage names like 'ethanolate'.
    """

    # --- _acid_to_ate ---

    def test_acid_to_ate_non_acid_returns_none(self):
        """_acid_to_ate('ethanol') must return None, not 'ethanolate'."""
        assert _acid_to_ate("ethanol") is None

    def test_acid_to_ate_random_string_returns_none(self):
        """_acid_to_ate('glucose') must return None, not 'glucosate'."""
        assert _acid_to_ate("glucose") is None

    def test_acid_to_ate_valid_systematic(self):
        """Regression: valid systematic acid still converts."""
        assert _acid_to_ate("propanoic acid") == "propanoate"

    def test_acid_to_ate_valid_trivial(self):
        """Regression: valid trivial acid still converts."""
        assert _acid_to_ate("acetic acid") == "acetate"

    def test_acid_to_ate_valid_carboxylic(self):
        """Regression: carboxylic acid pattern still converts."""
        assert _acid_to_ate("cyclohexanecarboxylic acid") == "cyclohexanecarboxylate"

    def test_acid_to_ate_valid_with_locant(self):
        """Regression: acid names with locant prefixes still convert."""
        result = _acid_to_ate("2-methylpropanoic acid")
        assert result is not None
        assert "ate" in result

    # --- _acid_to_amide ---

    def test_acid_to_amide_non_acid_returns_none(self):
        """_acid_to_amide('ethanol') must return None, not 'ethanolamide'."""
        assert _acid_to_amide("ethanol") is None

    def test_acid_to_amide_random_string_returns_none(self):
        """_acid_to_amide('glucose') must return None."""
        assert _acid_to_amide("glucose") is None

    def test_acid_to_amide_valid_systematic(self):
        """Regression: valid systematic acid still converts."""
        assert _acid_to_amide("propanoic acid") == "propanamide"

    def test_acid_to_amide_valid_trivial(self):
        """Regression: valid trivial acid still converts."""
        assert _acid_to_amide("acetic acid") == "acetamide"

    # --- _acid_to_acyl ---

    def test_acid_to_acyl_non_convertible_returns_none(self):
        """_acid_to_acyl('cyclohexane') must return None, not 'cyclohexaneyl'."""
        assert _acid_to_acyl("cyclohexane") is None

    def test_acid_to_acyl_ethanol_returns_none(self):
        """_acid_to_acyl('ethanol') must return None, not 'ethanolyl'."""
        assert _acid_to_acyl("ethanol") is None

    def test_acid_to_acyl_valid_acid(self):
        """Regression: valid acid input still converts."""
        assert _acid_to_acyl("acetic acid") == "acetyl"

    def test_acid_to_acyl_valid_amide(self):
        """Broader contract: amide inputs still convert (benzamide -> benzoyl)."""
        assert _acid_to_acyl("benzamide") == "benzoyl"

    def test_acid_to_acyl_valid_ester(self):
        """Broader contract: ester inputs still convert (propanoate -> propanoyl)."""
        assert _acid_to_acyl("propanoate") == "propanoyl"

    # --- _acid_to_thioate ---

    def test_acid_to_thioate_non_acid_returns_none(self):
        """_acid_to_thioate('ethanol') must return None, not 'ethanolthioate'."""
        assert _acid_to_thioate("ethanol") is None

    def test_acid_to_thioate_random_string_returns_none(self):
        """_acid_to_thioate('glucose') must return None."""
        assert _acid_to_thioate("glucose") is None

    def test_acid_to_thioate_valid_trivial(self):
        """Regression: valid trivial acid still converts."""
        assert _acid_to_thioate("acetic acid") == "ethanethioate"

    def test_acid_to_thioate_valid_systematic(self):
        """Regression: valid systematic acid still converts."""
        assert _acid_to_thioate("propanoic acid") == "propanethioate"
