"""
Unit tests for orthonym.assembly.naming_utils module.

Tests all pure naming utility functions:
- Alkyl substituent names
- Substituent prefix formatting
- Alphabetization sort keys
- Complex substituent detection
- Multiplier prefix selection
- Vowel elision
- Suffix with locants formatting (PIN style)
"""

import pytest

from orthonym.assembly.naming_utils import (
    ALKYL_NAMES,
    get_alkyl_name,
    format_substituent_prefix,
    alpha_sort_key,
    is_complex_substituent,
    get_multiplier_prefix,
    apply_vowel_elision,
    format_suffix_with_locants,
)


# ============================================================================
# TestAlkylNames
# ============================================================================

@pytest.mark.unit
class TestAlkylNames:
    """Tests for get_alkyl_name function."""

    @pytest.mark.parametrize("carbon_count,expected", [
        (1, "methyl"),
        (2, "ethyl"),
        (3, "propyl"),
        (4, "butyl"),
        (5, "pentyl"),
        (6, "hexyl"),
        (7, "heptyl"),
        (8, "octyl"),
        (9, "nonyl"),
        (10, "decyl"),
    ])
    def test_all_alkyl_names(self, carbon_count, expected):
        """Test all 10 alkyl names from methyl to decyl."""
        assert get_alkyl_name(carbon_count) == expected

    def test_alkyl_names_dict_has_10_entries(self):
        """Ensure ALKYL_NAMES has exactly 10 entries."""
        assert len(ALKYL_NAMES) == 10

    def test_unsupported_zero_raises_error(self):
        """Carbon count 0 is invalid."""
        with pytest.raises(ValueError, match="Unsupported carbon count 0"):
            get_alkyl_name(0)

    def test_unsupported_11_raises_error(self):
        """Carbon count 11 is beyond supported range."""
        with pytest.raises(ValueError, match="Unsupported carbon count 11"):
            get_alkyl_name(11)

    def test_unsupported_negative_raises_error(self):
        """Negative carbon count is invalid."""
        with pytest.raises(ValueError):
            get_alkyl_name(-1)

    def test_unsupported_large_raises_error(self):
        """Very large carbon count is invalid."""
        with pytest.raises(ValueError):
            get_alkyl_name(100)


# ============================================================================
# TestFormatSubstituentPrefix
# ============================================================================

@pytest.mark.unit
class TestFormatSubstituentPrefix:
    """Tests for format_substituent_prefix function."""

    def test_single_substituent(self):
        """Single methyl at position 2."""
        result = format_substituent_prefix("methyl", [2], 1)
        assert result == "2-methyl"

    def test_single_ethyl(self):
        """Single ethyl at position 3."""
        result = format_substituent_prefix("ethyl", [3], 1)
        assert result == "3-ethyl"

    def test_di_different_positions(self):
        """Dimethyl at positions 2 and 4."""
        result = format_substituent_prefix("methyl", [2, 4], 2)
        assert result == "2,4-dimethyl"

    def test_di_same_position(self):
        """Dimethyl at same position (geminal)."""
        result = format_substituent_prefix("methyl", [2, 2], 2)
        assert result == "2,2-dimethyl"

    def test_tri_substituent(self):
        """Trimethyl at positions 2, 3, 4."""
        result = format_substituent_prefix("methyl", [2, 3, 4], 3)
        assert result == "2,3,4-trimethyl"

    def test_diethyl(self):
        """Diethyl at positions 3 and 5."""
        result = format_substituent_prefix("ethyl", [3, 5], 2)
        assert result == "3,5-diethyl"

    def test_complex_substituent_bis(self):
        """Complex substituent with bis and parentheses."""
        result = format_substituent_prefix("1-methylethyl", [4, 7], 2)
        assert result == "4,7-bis(1-methylethyl)"

    def test_complex_substituent_tris(self):
        """Complex substituent with tris and parentheses."""
        result = format_substituent_prefix("1-methylethyl", [2, 4, 7], 3)
        assert result == "2,4,7-tris(1-methylethyl)"

    def test_complex_single_no_parentheses(self):
        """Single complex substituent - no parentheses needed (count=1)."""
        result = format_substituent_prefix("1-methylethyl", [4], 1)
        assert result == "4-1-methylethyl"

    def test_tetra_substituent(self):
        """Tetra- prefix for four simple substituents."""
        result = format_substituent_prefix("methyl", [2, 3, 4, 5], 4)
        assert result == "2,3,4,5-tetramethyl"


# ============================================================================
# TestAlphaSortKey
# ============================================================================

@pytest.mark.unit
class TestAlphaSortKey:
    """Tests for alpha_sort_key function."""

    def test_simple_name_unchanged(self):
        """Simple names return unchanged."""
        assert alpha_sort_key("methyl") == "methyl"

    def test_ethyl_unchanged(self):
        """ethyl returns unchanged."""
        assert alpha_sort_key("ethyl") == "ethyl"

    def test_strip_di_prefix(self):
        """di- multiplicative prefix is stripped."""
        assert alpha_sort_key("dimethyl") == "methyl"

    def test_strip_tri_prefix(self):
        """tri- multiplicative prefix is stripped."""
        assert alpha_sort_key("triethyl") == "ethyl"

    def test_strip_tetra_prefix(self):
        """tetra- multiplicative prefix is stripped."""
        assert alpha_sort_key("tetramethyl") == "methyl"

    def test_keep_iso_prefix(self):
        """iso- is a non-detachable prefix, kept for alphabetization."""
        assert alpha_sort_key("isopropyl") == "isopropyl"

    def test_keep_neo_prefix(self):
        """neo- is a non-detachable prefix, kept for alphabetization."""
        assert alpha_sort_key("neopentyl") == "neopentyl"

    def test_keep_cyclo_prefix(self):
        """cyclo- is a non-detachable prefix, kept for alphabetization."""
        assert alpha_sort_key("cyclopropyl") == "cyclopropyl"

    def test_strip_tert_prefix(self):
        """tert- is a detachable prefix, stripped for alphabetization."""
        assert alpha_sort_key("tert-butyl") == "butyl"

    def test_strip_sec_prefix(self):
        """sec- is a detachable prefix, stripped for alphabetization."""
        assert alpha_sort_key("sec-butyl") == "butyl"

    def test_strip_bis_prefix(self):
        """bis- complex multiplier is stripped."""
        assert alpha_sort_key("bismethyl") == "methyl"

    def test_strip_tris_prefix(self):
        """tris- complex multiplier is stripped."""
        assert alpha_sort_key("trisethyl") == "ethyl"

    def test_case_insensitive(self):
        """Sort keys are lowercase."""
        assert alpha_sort_key("Methyl") == "methyl"
        assert alpha_sort_key("ETHYL") == "ethyl"

    def test_sort_order_ethyl_before_methyl(self):
        """Verify correct IUPAC alphabetical ordering: ethyl before methyl.
        Both 'methyl' and 'dimethyl' have sort key 'methyl', so they
        come after 'ethyl'. Their relative order is determined by stable
        sort (input order preserved for equal keys)."""
        names = ["ethyl", "methyl", "dimethyl"]
        sorted_names = sorted(names, key=alpha_sort_key)
        # ethyl sorts first; methyl and dimethyl have same key
        assert sorted_names[0] == "ethyl"
        assert set(sorted_names[1:]) == {"methyl", "dimethyl"}

    def test_sort_order_with_iso(self):
        """iso- is included, so isopropyl sorts under 'i'."""
        names = ["methyl", "isopropyl", "ethyl"]
        sorted_names = sorted(names, key=alpha_sort_key)
        # ethyl (e) < isopropyl (i) < methyl (m)
        assert sorted_names == ["ethyl", "isopropyl", "methyl"]

    def test_sort_order_with_cyclo(self):
        """cyclo- is included, so cyclopropyl sorts under 'c'."""
        names = ["methyl", "cyclopropyl", "ethyl"]
        sorted_names = sorted(names, key=alpha_sort_key)
        # cyclopropyl (c) < ethyl (e) < methyl (m)
        assert sorted_names == ["cyclopropyl", "ethyl", "methyl"]


# ============================================================================
# TestIsComplexSubstituent
# ============================================================================

@pytest.mark.unit
class TestIsComplexSubstituent:
    """Tests for is_complex_substituent function."""

    def test_methyl_is_simple(self):
        """methyl is a simple substituent."""
        assert is_complex_substituent("methyl") is False

    def test_ethyl_is_simple(self):
        """ethyl is a simple substituent."""
        assert is_complex_substituent("ethyl") is False

    def test_propyl_is_simple(self):
        """propyl is a simple substituent."""
        assert is_complex_substituent("propyl") is False

    def test_name_with_digits_is_complex(self):
        """1-methylethyl has digits and hyphen, so it's complex."""
        assert is_complex_substituent("1-methylethyl") is True

    def test_name_with_digit_prefix_is_complex(self):
        """2-propyl has a digit, so it's complex."""
        assert is_complex_substituent("2-propyl") is True

    def test_name_with_hyphen_is_complex(self):
        """Names with hyphens are complex."""
        assert is_complex_substituent("tert-butyl") is True

    def test_isopropyl_is_simple(self):
        """isopropyl has no digits or hyphens, so it's simple."""
        assert is_complex_substituent("isopropyl") is False

    def test_neopentyl_is_simple(self):
        """neopentyl has no digits or hyphens, so it's simple."""
        assert is_complex_substituent("neopentyl") is False

    def test_cyclopropyl_is_simple(self):
        """cyclopropyl has no digits or hyphens, so it's simple."""
        assert is_complex_substituent("cyclopropyl") is False


# ============================================================================
# TestGetMultiplierPrefix
# ============================================================================

@pytest.mark.unit
class TestGetMultiplierPrefix:
    """Tests for get_multiplier_prefix function."""

    def test_count_1_returns_empty(self):
        """Count of 1 needs no multiplier."""
        assert get_multiplier_prefix(1, "methyl") == ""

    def test_di_for_simple(self):
        """di- for 2 simple substituents."""
        assert get_multiplier_prefix(2, "methyl") == "di"

    def test_tri_for_simple(self):
        """tri- for 3 simple substituents."""
        assert get_multiplier_prefix(3, "methyl") == "tri"

    def test_tetra_for_simple(self):
        """tetra- for 4 simple substituents."""
        assert get_multiplier_prefix(4, "methyl") == "tetra"

    def test_penta_for_simple(self):
        """penta- for 5 simple substituents."""
        assert get_multiplier_prefix(5, "methyl") == "penta"

    def test_bis_for_complex(self):
        """bis- for 2 complex substituents."""
        assert get_multiplier_prefix(2, "1-methylethyl") == "bis"

    def test_tris_for_complex(self):
        """tris- for 3 complex substituents."""
        assert get_multiplier_prefix(3, "1-methylethyl") == "tris"

    def test_tetrakis_for_complex(self):
        """tetrakis- for 4 complex substituents."""
        assert get_multiplier_prefix(4, "1-methylethyl") == "tetrakis"

    def test_pentakis_for_complex(self):
        """pentakis- for 5 complex substituents."""
        assert get_multiplier_prefix(5, "1-methylethyl") == "pentakis"

    def test_count_0_returns_empty(self):
        """Count of 0 returns empty."""
        assert get_multiplier_prefix(0, "methyl") == ""


# ============================================================================
# TestVowelElision
# ============================================================================

@pytest.mark.unit
class TestVowelElision:
    """Tests for apply_vowel_elision function."""

    def test_elision_before_ol(self):
        """Terminal 'e' elided before 'ol' (starts with 'o')."""
        assert apply_vowel_elision("propane", "ol") == "propanol"

    def test_elision_before_al(self):
        """Terminal 'e' elided before 'al' (starts with 'a')."""
        assert apply_vowel_elision("propane", "al") == "propanal"

    def test_elision_before_one(self):
        """Terminal 'e' elided before 'one' (starts with 'o')."""
        assert apply_vowel_elision("propane", "one") == "propanone"

    def test_elision_before_amine(self):
        """Terminal 'e' elided before 'amine' (starts with 'a')."""
        assert apply_vowel_elision("propane", "amine") == "propanamine"

    def test_no_elision_before_consonant(self):
        """No elision before 'diol' (starts with 'd', a consonant)."""
        assert apply_vowel_elision("butane", "diol") == "butanediol"

    def test_elision_before_oic_acid(self):
        """Terminal 'e' elided before 'oic acid' (starts with 'o')."""
        assert apply_vowel_elision("ethane", "oic acid") == "ethanoic acid"

    def test_no_trailing_e_no_elision(self):
        """No trailing 'e' means nothing to elide."""
        assert apply_vowel_elision("propan", "ol") == "propanol"

    def test_no_elision_before_e(self):
        """'e' before 'e' is NOT elided (IUPAC convention)."""
        assert apply_vowel_elision("butane", "ene") == "butaneene"

    def test_elision_before_yl(self):
        """Terminal 'e' elided before 'yl' (starts with 'y')."""
        assert apply_vowel_elision("propane", "yl") == "propanyl"

    def test_empty_suffix(self):
        """Empty suffix returns parent stem unchanged."""
        assert apply_vowel_elision("propane", "") == "propane"

    def test_empty_parent(self):
        """Empty parent returns suffix unchanged."""
        assert apply_vowel_elision("", "ol") == "ol"

    def test_both_empty(self):
        """Both empty returns empty string."""
        assert apply_vowel_elision("", "") == ""

    def test_methane_ol(self):
        """methane + ol -> methanol."""
        assert apply_vowel_elision("methane", "ol") == "methanol"

    def test_no_elision_before_nitrile(self):
        """No elision before 'nitrile' (starts with 'n')."""
        assert apply_vowel_elision("propane", "nitrile") == "propanenitrile"


# ============================================================================
# TestFormatSuffixWithLocants
# ============================================================================

@pytest.mark.unit
class TestFormatSuffixWithLocants:
    """Tests for format_suffix_with_locants function."""

    def test_propan_1_ol(self):
        """prop + an + ol at [1] -> propan-1-ol."""
        result = format_suffix_with_locants("prop", "an", "ol", [1])
        assert result == "propan-1-ol"

    def test_butan_2_one(self):
        """but + an + one at [2] -> butan-2-one."""
        result = format_suffix_with_locants("but", "an", "one", [2])
        assert result == "butan-2-one"

    def test_propane_1_2_diol(self):
        """prop + an + ol at [1,2] with di -> propane-1,2-diol."""
        result = format_suffix_with_locants("prop", "an", "ol", [1, 2], "di")
        assert result == "propane-1,2-diol"

    def test_pentanoic_acid_no_locant(self):
        """pent + an + oic acid with no locant (terminal acid)."""
        result = format_suffix_with_locants("pent", "an", "oic acid", [])
        assert result == "pentanoic acid"

    def test_propanal_no_locant(self):
        """prop + an + al with no locant (terminal aldehyde)."""
        result = format_suffix_with_locants("prop", "an", "al", [])
        assert result == "propanal"

    def test_hexan_3_ol(self):
        """hex + an + ol at [3] -> hexan-3-ol."""
        result = format_suffix_with_locants("hex", "an", "ol", [3])
        assert result == "hexan-3-ol"

    def test_butane_2_3_diol(self):
        """but + an + ol at [2,3] with di -> butane-2,3-diol."""
        result = format_suffix_with_locants("but", "an", "ol", [2, 3], "di")
        assert result == "butane-2,3-diol"

    def test_ethanoic_acid_no_locant(self):
        """eth + an + oic acid with no locant -> ethanoic acid."""
        result = format_suffix_with_locants("eth", "an", "oic acid", [])
        assert result == "ethanoic acid"

    def test_pentan_2_one(self):
        """pent + an + one at [2] -> pentan-2-one."""
        result = format_suffix_with_locants("pent", "an", "one", [2])
        assert result == "pentan-2-one"

    def test_butane_1_4_diol(self):
        """but + an + ol at [1,4] with di -> butane-1,4-diol."""
        result = format_suffix_with_locants("but", "an", "ol", [1, 4], "di")
        assert result == "butane-1,4-diol"

    def test_with_unsaturation_infix(self):
        """but + en + ol at [2] -> buten-2-ol (unsaturation preserved)."""
        result = format_suffix_with_locants("but", "en", "ol", [2])
        assert result == "buten-2-ol"

    def test_no_unsaturation_terminal(self):
        """pentan + empty + oic acid with no locant -> pentanoic acid."""
        result = format_suffix_with_locants("pentan", "", "oic acid", [])
        assert result == "pentanoic acid"

    def test_propanal_no_unsaturation_no_locant(self):
        """propan + empty + al with no locant -> propanal."""
        result = format_suffix_with_locants("propan", "", "al", [])
        assert result == "propanal"
