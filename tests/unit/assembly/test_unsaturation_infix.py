"""Unit tests for unsaturation infix builder and hydrocarbon name builder.

Tests _build_unsaturation_infix and _build_hydrocarbon_name to ensure:
1. All branch combinations produce correct IUPAC compliant output
2. No output ever contains doubled hyphens (--)
3. Special cases (2C, mono-cycloalkene) handled correctly

a phase Plan 02 Task 1: Written before rewrite to serve as characterization tests.
"""

import pytest

from orthonym.assembly.composer import _build_unsaturation_infix, _build_hydrocarbon_name


# ===========================================================================
# _build_unsaturation_infix tests
# ===========================================================================


class TestBuildUnsaturationInfix:
    """Test _build_unsaturation_infix for all branch combinations."""

    def test_saturated(self):
        """Saturated: no double or triple bonds -> 'an'."""
        assert _build_unsaturation_infix([], []) == "an"

    def test_single_double(self):
        """Single double bond at position 1 -> '-1-en'."""
        assert _build_unsaturation_infix([1], []) == "-1-en"

    def test_single_double_position_2(self):
        """Single double bond at position 2 -> '-2-en'."""
        assert _build_unsaturation_infix([2], []) == "-2-en"

    def test_single_triple(self):
        """Single triple bond at position 4 -> '-4-yn'."""
        assert _build_unsaturation_infix([], [4]) == "-4-yn"

    def test_single_triple_position_1(self):
        """Single triple bond at position 1 -> '-1-yn'."""
        assert _build_unsaturation_infix([], [1]) == "-1-yn"

    def test_enyne(self):
        """Single double + single triple -> '-1-en-4-yn'."""
        assert _build_unsaturation_infix([1], [4]) == "-1-en-4-yn"

    def test_diene(self):
        """Two double bonds -> 'a-1,3-dien'."""
        assert _build_unsaturation_infix([1, 3], []) == "a-1,3-dien"

    def test_diyne(self):
        """Two triple bonds -> 'a-1,3-diyn'."""
        assert _build_unsaturation_infix([], [1, 3]) == "a-1,3-diyn"

    def test_diene_yne(self):
        """Two double + one triple -> 'a-1,3-dien-5-yn'."""
        assert _build_unsaturation_infix([1, 3], [5]) == "a-1,3-dien-5-yn"

    def test_triene(self):
        """Three double bonds -> 'a-1,3,5-trien'."""
        assert _build_unsaturation_infix([1, 3, 5], []) == "a-1,3,5-trien"

    def test_no_doubled_hyphens_any_combo(self):
        """No output from _build_unsaturation_infix should contain '--'."""
        combos = [
            ([], []),
            ([1], []),
            ([], [4]),
            ([1], [4]),
            ([1, 3], []),
            ([], [1, 3]),
            ([1, 3], [5]),
            ([1, 3, 5], []),
            ([1, 3], [5, 7]),
            ([2], [5]),
            ([1, 3, 5, 7], []),
        ]
        for dl, tl in combos:
            result = _build_unsaturation_infix(dl, tl)
            assert "--" not in result, (
                f"Doubled hyphen in _build_unsaturation_infix({dl}, {tl}) = '{result}'"
            )


# ===========================================================================
# _build_hydrocarbon_name tests
# ===========================================================================


class TestBuildHydrocarbonName:
    """Test _build_hydrocarbon_name for all special cases."""

    def test_saturated_butane(self):
        """Saturated: 'butane'."""
        assert _build_hydrocarbon_name("but", [], []) == "butane"

    def test_saturated_ethane(self):
        """Saturated 2C: 'ethane'."""
        assert _build_hydrocarbon_name("eth", [], []) == "ethane"

    def test_ethene_no_locant(self):
        """2C alkene: 'ethene' (no locant for 2C)."""
        assert _build_hydrocarbon_name("eth", [1], []) == "ethene"

    def test_ethyne_no_locant(self):
        """2C alkyne: 'ethyne' (no locant for 2C)."""
        assert _build_hydrocarbon_name("eth", [], [1]) == "ethyne"

    def test_but_1_ene(self):
        """Single double bond with locant: 'but-1-ene'."""
        assert _build_hydrocarbon_name("but", [1], []) == "but-1-ene"

    def test_but_2_ene(self):
        """Single double bond at position 2: 'but-2-ene'."""
        assert _build_hydrocarbon_name("but", [2], []) == "but-2-ene"

    def test_buta_1_3_diene(self):
        """Diene: 'buta-1,3-diene'. Stem 'but' + 'a' connector added by function."""
        assert _build_hydrocarbon_name("but", [1, 3], []) == "buta-1,3-diene"

    def test_pent_1_en_4_yne(self):
        """Enyne: 'pent-1-en-4-yne'."""
        assert _build_hydrocarbon_name("pent", [1], [4]) == "pent-1-en-4-yne"

    def test_cyclohexane(self):
        """Cyclic saturated: 'cyclohexane'."""
        assert _build_hydrocarbon_name("cyclohex", [], []) == "cyclohexane"

    def test_cyclohexene_no_locant(self):
        """Mono-cycloalkene: 'cyclohexene' (no locant per IUPAC convention)."""
        assert _build_hydrocarbon_name("cyclohex", [1], []) == "cyclohexene"

    def test_cyclohexa_1_3_diene(self):
        """Cyclic diene: 'cyclohexa-1,3-diene'. Stem 'cyclohex' + 'a' added by function."""
        assert _build_hydrocarbon_name("cyclohex", [1, 3], []) == "cyclohexa-1,3-diene"

    def test_but_1_yne(self):
        """Single triple bond with locant: 'but-1-yne'."""
        assert _build_hydrocarbon_name("but", [], [1]) == "but-1-yne"

    def test_no_doubled_hyphens_any_combo(self):
        """No output from _build_hydrocarbon_name should contain '--'."""
        combos = [
            ("eth", [], []),
            ("eth", [1], []),
            ("eth", [], [1]),
            ("but", [], []),
            ("but", [1], []),
            ("but", [], [1]),
            ("but", [1], [3]),
            ("pent", [1], [4]),
            ("pent", [1, 3], []),
            ("cyclohex", [], []),
            ("cyclohex", [1], []),
            ("cyclohex", [1, 3], []),
        ]
        for stem, dl, tl in combos:
            result = _build_hydrocarbon_name(stem, dl, tl)
            assert "--" not in result, (
                f"Doubled hyphen in _build_hydrocarbon_name('{stem}', {dl}, {tl}) = '{result}'"
            )
