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
- Enclosing marks depth cycling (P-16.5.1.1)
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
    apply_enclosing_marks,
    get_bracket_depth,
    needs_brackets,
    _ALKYL_ROOTS_FULL,
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
        with pytest.raises(ValueError):
            get_alkyl_name(0)

    def test_supported_11_now_works(self):
        """Carbon count 11 is now supported via centralized chain_names module."""
        name = get_alkyl_name(11)
        assert name == "undecyl"

    def test_unsupported_negative_raises_error(self):
        """Negative carbon count is invalid."""
        with pytest.raises(ValueError):
            get_alkyl_name(-1)

    def test_large_count_now_works(self):
        """Large carbon count (100) is now supported via centralized chain_names module."""
        name = get_alkyl_name(100)
        assert isinstance(name, str)
        assert len(name) > 0
        assert name.endswith("yl")


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

    def test_complex_single_with_parentheses(self):
        """Single complex substituent with numeric locants gets parentheses (IUPAC P-16.5.1.1)."""
        result = format_substituent_prefix("1-methylethyl", [4], 1)
        assert result == "4-(1-methylethyl)"

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

    def test_internal_hyphen_is_complex(self):
        """A genuine compound substituent (locant + hyphen) is complex (P-16.3.3)."""
        assert is_complex_substituent("1-methylpropyl") is True

    def test_sec_tert_retained_names_are_simple(self):
        """Phase 171 BBR-ASM (P-16.3.3(b)/P-16.2.4.1(d)): the leading italic sec-/tert- prefix on an
        otherwise-simple retained name does NOT make it complex — di-tert-butyl /
        the bare N-substituent, NOT bis(tert-butyl) (BB 3465 / BB 7070). The earlier blanket
        'any hyphen -> complex' rule over-parenthesised these."""
        assert is_complex_substituent("tert-butyl") is False
        assert is_complex_substituent("sec-butyl") is False

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


# ============================================================================
# TestEnclosingMarksDepthCycling
# ============================================================================

@pytest.mark.unit
class TestEnclosingMarksDepthCycling:
    """Tests for IUPAC P-16.5.1.1 enclosing marks cycling through all 7 depths."""

    @pytest.mark.parametrize("depth,expected", [
        (0, "(x)"),
        (1, "[x]"),
        (2, "{x}"),
        (3, "(x)"),
        (4, "[x]"),
        (5, "{x}"),
        (6, "(x)"),
    ])
    def test_enclosing_marks_depth_0_through_6(self, depth, expected):
        """Enclosing marks cycle () -> [] -> {} -> () at each depth level."""
        assert apply_enclosing_marks("x", depth) == expected

    def test_enclosing_marks_preserves_content(self):
        """Content inside enclosing marks is preserved at all depth levels."""
        content = "2-(methylamino)ethyl"
        for depth in range(7):
            result = apply_enclosing_marks(content, depth)
            # Content must appear unchanged inside the marks
            assert content in result
            # Result is exactly open_mark + content + close_mark
            assert result[1:-1] == content

    @pytest.mark.parametrize("name,expected_depth", [
        ("methyl", 0),
        ("(2-methylpropyl)", 1),
        ("[2-(methylamino)ethyl]", 2),
        ("{complex}", 3),
    ])
    def test_bracket_depth_detects_all_types(self, name, expected_depth):
        """get_bracket_depth correctly identifies nesting depth from bracket type."""
        assert get_bracket_depth(name) == expected_depth


# ============================================================================
# TestC11PlusAlkylRootCoverage
# ============================================================================

@pytest.mark.unit
class TestC11PlusAlkylRootCoverage:
    """Tests for C11-C20 alkyl root recognition in utility functions.

    These test edge cases where alkyl roots beyond C10 (undecyl through icosyl)
    must be recognized by needs_brackets(), is_complex_substituent(), and the
    regex patterns _HALOALKYL_RE and _ALKYLAMINO_RE.
    """

    # --- _ALKYL_ROOTS_FULL module-level constant ---

    def test_alkyl_roots_full_has_20_entries(self):
        """_ALKYL_ROOTS_FULL must contain C1-C20 alkyl names."""
        assert len(_ALKYL_ROOTS_FULL) == 20

    def test_alkyl_roots_full_contains_undecyl(self):
        assert "undecyl" in _ALKYL_ROOTS_FULL

    def test_alkyl_roots_full_contains_icosyl(self):
        assert "icosyl" in _ALKYL_ROOTS_FULL

    # --- needs_brackets() C11+ tests ---

    def test_needs_brackets_hydroxyundecyl(self):
        """C11: hydroxy + undecyl is a compound substituent."""
        assert needs_brackets("hydroxyundecyl") is True

    def test_needs_brackets_hydroxydodecyl(self):
        """C12: hydroxy + dodecyl is a compound substituent."""
        assert needs_brackets("hydroxydodecyl") is True

    def test_needs_brackets_hydroxyicosyl(self):
        """C20: hydroxy + icosyl is a compound substituent."""
        assert needs_brackets("hydroxyicosyl") is True

    def test_needs_brackets_hydroxymethyl_still_works(self):
        """C1: hydroxy + methyl still works after extension."""
        assert needs_brackets("hydroxymethyl") is True

    def test_needs_brackets_methylsulfinyl_still_works(self):
        """C1: methyl + sulfinyl still works after extension."""
        assert needs_brackets("methylsulfinyl") is True

    def test_needs_brackets_undecylsulfinyl(self):
        """C11: undecyl + sulfinyl is a compound substituent."""
        assert needs_brackets("undecylsulfinyl") is True

    # --- is_complex_substituent() C11+ tests ---

    def test_is_complex_undecylsulfinyl(self):
        """C11: undecylsulfinyl is a complex substituent."""
        assert is_complex_substituent("undecylsulfinyl") is True

    def test_is_complex_icosylsulfonyl(self):
        """C20: icosylsulfonyl is a complex substituent."""
        assert is_complex_substituent("icosylsulfonyl") is True

    def test_is_complex_methylsulfinyl_still_works(self):
        """C1: methylsulfinyl still works after extension."""
        assert is_complex_substituent("methylsulfinyl") is True

    # --- Regex patterns C11+ tests ---

    def test_haloalkyl_re_fluoroundecyl(self):
        """C11: _HALOALKYL_RE should match fluoroundecyl."""
        from orthonym.assembly.naming_utils import _HALOALKYL_RE
        assert _HALOALKYL_RE.match("fluoroundecyl") is not None

    def test_haloalkyl_re_trifluoroicosyl(self):
        """C20: _HALOALKYL_RE should match trifluoroicosyl."""
        from orthonym.assembly.naming_utils import _HALOALKYL_RE
        assert _HALOALKYL_RE.match("trifluoroicosyl") is not None

    def test_haloalkyl_re_fluoromethyl_still_works(self):
        """C1: _HALOALKYL_RE should still match fluoromethyl."""
        from orthonym.assembly.naming_utils import _HALOALKYL_RE
        assert _HALOALKYL_RE.match("fluoromethyl") is not None

    def test_multiplier_syllables_match_simple_multipliers(self):
        """_MULTIPLIER_SYLLABLES is a forward declaration of SIMPLE_MULTIPLIERS.

        It exists only because the regexes compile at import time, ahead of the
        table. If the two drift apart, a whole band of multipliers silently stops
        being recognised as a compound haloalkyl prefix -- which is exactly the bug
        v29 Phase C Task 5a hit: the alternation stopped at ``hexa``, so
        ``heptafluoropropyl`` lost its enclosing marks the moment P-14.3.4.5
        (``:3007``) removed its locants.
        """
        from orthonym.assembly.naming_utils import (
            _MULTIPLIER_SYLLABLES, SIMPLE_MULTIPLIERS,
        )
        assert set(_MULTIPLIER_SYLLABLES) == set(SIMPLE_MULTIPLIERS.values())

    @pytest.mark.parametrize("name", [
        "heptafluoropropyl", "octafluorobutyl", "nonafluorobutyl",
        "undecafluoropentyl", "pentadecafluorooctyl", "tridecafluorohexyl",
    ])
    def test_haloalkyl_re_covers_multipliers_above_hexa(self, name):
        """P-14.3.4.5 emits locant-free names, so this regex -- not the digit check
        in is_complex_substituent -- becomes the load-bearing enclosing-mark gate."""
        from orthonym.assembly.naming_utils import _HALOALKYL_RE
        assert _HALOALKYL_RE.match(name) is not None, name

    @pytest.mark.parametrize("name", ["ethyl", "methyl", "propyl", "butyl",
                                      "phenyl", "cyclohexyl"])
    def test_haloalkyl_re_does_not_overmatch_plain_alkyls(self, name):
        from orthonym.assembly.naming_utils import _HALOALKYL_RE
        assert _HALOALKYL_RE.match(name) is None, name

    def test_alkylamino_re_undecylamino(self):
        """C11: _ALKYLAMINO_RE should match undecylamino."""
        from orthonym.assembly.naming_utils import _ALKYLAMINO_RE
        assert _ALKYLAMINO_RE.match("undecylamino") is not None

    def test_alkylamino_re_icosylamino(self):
        """C20: _ALKYLAMINO_RE should match icosylamino."""
        from orthonym.assembly.naming_utils import _ALKYLAMINO_RE
        assert _ALKYLAMINO_RE.match("icosylamino") is not None

    def test_alkylamino_re_methylamino_still_works(self):
        """C1: _ALKYLAMINO_RE should still match methylamino."""
        from orthonym.assembly.naming_utils import _ALKYLAMINO_RE
        assert _ALKYLAMINO_RE.match("methylamino") is not None

    # --- alpha_sort_key("trioxo") IUPAC P-14.4 behavior ---

    def test_alpha_sort_key_trioxo_returns_oxo(self):
        """Per IUPAC P-14.4, 'tri' is a multiplicative prefix stripped for
        alphabetization. alpha_sort_key('trioxo') correctly returns 'oxo'.
        This is used ONLY as a sort key, never for text reconstruction."""
        assert alpha_sort_key("trioxo") == "oxo"

    def test_alpha_sort_key_dioxo_returns_oxo(self):
        """'di' is stripped: dioxo -> oxo."""
        assert alpha_sort_key("dioxo") == "oxo"


@pytest.mark.unit
class TestPhase8MultipliedSuffixAndSilyl:
    """v23 Phase 8: amine-suffix elision (BB P-62.2.4.1.2: 'tetramine', not
    'tetraamine') + silyl/germyl-with-prefix complexity (P-16.3.3)."""

    def test_join_multiplied_suffix_amine_elision(self):
        from orthonym.assembly.naming_utils import _join_multiplied_suffix
        # tetra/penta/hexa final 'a' elides before 'amine'; di/tri unaffected.
        assert _join_multiplied_suffix("tetra", "amine") == "tetramine"
        assert _join_multiplied_suffix("penta", "amine") == "pentamine"
        assert _join_multiplied_suffix("di", "amine") == "diamine"
        assert _join_multiplied_suffix("tri", "amine") == "triamine"
        # -ol elision preserved; -one elides its 'a' too (Blue Book P-64.2.2.1(1):
        # 'tetrone', e.g. pentacosane-7,9,17,19-tetrone (PIN)). Wave2 T1b.
        assert _join_multiplied_suffix("tetra", "ol") == "tetrol"
        assert _join_multiplied_suffix("tetra", "one") == "tetrone"
        assert _join_multiplied_suffix("penta", "one") == "pentone"
        # 'di'/'tri' carry no terminal 'a' -> unaffected.
        assert _join_multiplied_suffix("di", "one") == "dione"
        assert _join_multiplied_suffix("tri", "one") == "trione"

    def test_is_complex_silyl_with_prefixes(self):
        # silyl/germyl carrying their own prefixes are compound -> enclosing marks.
        assert is_complex_substituent("trihydroxysilyl") is True
        assert is_complex_substituent("hydroxydimethylsilyl") is True
        assert is_complex_substituent("trimethylsilyl") is True
        # bare stems stay simple.
        assert is_complex_substituent("silyl") is False
        assert is_complex_substituent("germyl") is False
