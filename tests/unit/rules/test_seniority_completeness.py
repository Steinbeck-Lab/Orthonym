"""
Invariant tests for seniority table completeness.

Ensures every FG in SENIORITY_ORDER has explicit entries in both
SUFFIX_FORMS and PREFIX_FORMS. Acts as a guardrail preventing future
additions without corresponding dict entries.

Phase 93-01: seniority completeness invariant tests.
"""

import pytest
from orthonym.rules.seniority import (
    SENIORITY_ORDER,
    SUFFIX_FORMS,
    PREFIX_FORMS,
    get_suffix,
    get_prefix,
)


class TestSeniorityOrderCompleteness:
    """Every FG in SENIORITY_ORDER must have entries in both parallel dicts."""

    def test_all_seniority_entries_have_suffix_forms(self):
        """Every FG in SENIORITY_ORDER must have an entry in SUFFIX_FORMS."""
        missing = [fg for fg in SENIORITY_ORDER if fg not in SUFFIX_FORMS]
        assert not missing, (
            f"FGs in SENIORITY_ORDER missing from SUFFIX_FORMS: {missing}"
        )

    def test_all_seniority_entries_have_prefix_forms(self):
        """Every FG in SENIORITY_ORDER must have an entry in PREFIX_FORMS."""
        missing = [fg for fg in SENIORITY_ORDER if fg not in PREFIX_FORMS]
        assert not missing, (
            f"FGs in SENIORITY_ORDER missing from PREFIX_FORMS: {missing}"
        )

    def test_no_duplicate_entries_in_seniority_order(self):
        """SENIORITY_ORDER must not contain duplicate FG names."""
        seen = set()
        duplicates = []
        for fg in SENIORITY_ORDER:
            if fg in seen:
                duplicates.append(fg)
            seen.add(fg)
        assert not duplicates, (
            f"Duplicate entries in SENIORITY_ORDER: {duplicates}"
        )


class TestSuffixFormsStructure:
    """SUFFIX_FORMS entries must be either None or 2-tuples of strings."""

    def test_suffix_forms_values_are_none_or_2tuple(self):
        """Every SUFFIX_FORMS value is None or a 2-tuple of strings."""
        for fg, value in SUFFIX_FORMS.items():
            if value is not None:
                assert isinstance(value, tuple), (
                    f"SUFFIX_FORMS['{fg}'] is {type(value).__name__}, "
                    f"expected tuple or None"
                )
                assert len(value) == 2, (
                    f"SUFFIX_FORMS['{fg}'] has {len(value)} elements, "
                    f"expected 2 (chain_suffix, ring_suffix)"
                )
                chain_suffix, ring_suffix = value
                assert isinstance(chain_suffix, str), (
                    f"SUFFIX_FORMS['{fg}'][0] (chain_suffix) is "
                    f"{type(chain_suffix).__name__}, expected str"
                )
                assert isinstance(ring_suffix, str), (
                    f"SUFFIX_FORMS['{fg}'][1] (ring_suffix) is "
                    f"{type(ring_suffix).__name__}, expected str"
                )


class TestPrefixFormsStructure:
    """PREFIX_FORMS entries must be either None or strings."""

    def test_prefix_forms_values_are_none_or_string(self):
        """Every PREFIX_FORMS value is None or a string."""
        for fg, value in PREFIX_FORMS.items():
            if value is not None:
                assert isinstance(value, str), (
                    f"PREFIX_FORMS['{fg}'] is {type(value).__name__}, "
                    f"expected str or None"
                )


class TestGetSuffixHandlesNone:
    """get_suffix() must return None for functional-class-only FGs."""

    @pytest.mark.parametrize("fg_name", [
        "thioester",
        "sulfoxide",
        "sulfone",
        "thioether",
        "phosphine_oxide",
        "phosphate_triester",
        "phosphate_diester",
        "phosphate_monoester",
        "tertiary_phosphine",
        "secondary_phosphine",
        "primary_phosphine",
        "isocyanide",
        "azido",
        "azo",
        "cyanate",
        "thiocyanate",
    ])
    def test_get_suffix_returns_none_for_functional_class_only(self, fg_name):
        """get_suffix() returns None for FGs with no substitutive suffix."""
        assert get_suffix(fg_name) is None
        assert get_suffix(fg_name, is_ring=True) is None


class TestGetSuffixReturnsCorrectValues:
    """get_suffix() returns correct tuple values for FGs with real suffixes."""

    def test_hydrazide_chain_suffix(self):
        """Hydrazide chain suffix is 'ohydrazide' (IUPAC P-66.3)."""
        assert get_suffix("hydrazide") == "ohydrazide"

    def test_hydrazide_ring_suffix(self):
        """Hydrazide ring suffix is 'carbohydrazide' (IUPAC P-66.3)."""
        assert get_suffix("hydrazide", is_ring=True) == "carbohydrazide"

    def test_imide_chain_suffix(self):
        """Imide chain suffix is 'imide' (IUPAC P-66.2)."""
        assert get_suffix("imide") == "imide"

    def test_imide_ring_suffix(self):
        """Imide ring suffix is 'dicarboximide' (IUPAC P-66.2)."""
        assert get_suffix("imide", is_ring=True) == "dicarboximide"


class TestGetPrefixReturnsCorrectValues:
    """get_prefix() returns correct string values for newly-added entries."""

    def test_hydrazide_prefix(self):
        """Hydrazide prefix is 'hydrazinecarbonyl' (IUPAC P-66.3.5)."""
        assert get_prefix("hydrazide") == "hydrazinecarbonyl"

    def test_thioaldehyde_prefix(self):
        """Thioaldehyde prefix is 'thioxo' (IUPAC P-63.1.5)."""
        assert get_prefix("thioaldehyde") == "thioxo"

    def test_sulfoxide_prefix(self):
        """Sulfoxide prefix is 'sulfinyl' (IUPAC P-63.6)."""
        assert get_prefix("sulfoxide") == "sulfinyl"

    def test_sulfone_prefix(self):
        """Sulfone prefix is 'sulfonyl' (IUPAC P-63.6)."""
        assert get_prefix("sulfone") == "sulfonyl"

    def test_anhydride_prefix_is_none(self):
        """Anhydride has no substitutive prefix (functional class only)."""
        assert get_prefix("anhydride") is None

    def test_secondary_amide_prefix_is_none(self):
        """Secondary amide uses acylamino pathway, no simple prefix."""
        assert get_prefix("secondary_amide") is None

    def test_tertiary_amide_prefix_is_none(self):
        """Tertiary amide uses acylamino pathway, no simple prefix."""
        assert get_prefix("tertiary_amide") is None

    def test_imide_prefix_is_none(self):
        """Imide named as heterocyclic ring substituent, no simple prefix."""
        assert get_prefix("imide") is None

    def test_thioester_prefix_is_none(self):
        """Thioester named via decomposition pathway, no simple prefix."""
        assert get_prefix("thioester") is None


class TestNewSeniorityOrderPositions:
    """Prefix-only FGs are positioned after hydrazone in SENIORITY_ORDER."""

    def test_azido_after_hydrazone(self):
        """azido is positioned after hydrazone in seniority order."""
        hz_idx = SENIORITY_ORDER.index("hydrazone")
        az_idx = SENIORITY_ORDER.index("azido")
        assert az_idx > hz_idx

    def test_azo_after_hydrazone(self):
        """azo is positioned after hydrazone in seniority order."""
        hz_idx = SENIORITY_ORDER.index("hydrazone")
        az_idx = SENIORITY_ORDER.index("azo")
        assert az_idx > hz_idx

    def test_cyanate_after_hydrazone(self):
        """cyanate is positioned after hydrazone in seniority order."""
        hz_idx = SENIORITY_ORDER.index("hydrazone")
        cy_idx = SENIORITY_ORDER.index("cyanate")
        assert cy_idx > hz_idx

    def test_thiocyanate_after_hydrazone(self):
        """thiocyanate is positioned after hydrazone in seniority order."""
        hz_idx = SENIORITY_ORDER.index("hydrazone")
        tc_idx = SENIORITY_ORDER.index("thiocyanate")
        assert tc_idx > hz_idx

    def test_prefix_only_fgs_before_sulfoxide(self):
        """All prefix-only FGs are before sulfoxide in seniority order."""
        sx_idx = SENIORITY_ORDER.index("sulfoxide")
        for fg in ["azido", "azo", "cyanate", "thiocyanate"]:
            fg_idx = SENIORITY_ORDER.index(fg)
            assert fg_idx < sx_idx, (
                f"{fg} at index {fg_idx} should be before "
                f"sulfoxide at index {sx_idx}"
            )


class TestPrefixOnlyFGsSuffixFormsNone:
    """Prefix-only FGs must have None in SUFFIX_FORMS."""

    @pytest.mark.parametrize("fg_name", [
        "azido",
        "azo",
        "cyanate",
        "thiocyanate",
    ])
    def test_prefix_only_fg_has_none_suffix(self, fg_name):
        """Prefix-only FG has None in SUFFIX_FORMS."""
        assert fg_name in SUFFIX_FORMS, (
            f"{fg_name} not in SUFFIX_FORMS"
        )
        assert SUFFIX_FORMS[fg_name] is None, (
            f"SUFFIX_FORMS['{fg_name}'] should be None (prefix-only), "
            f"got {SUFFIX_FORMS[fg_name]}"
        )
