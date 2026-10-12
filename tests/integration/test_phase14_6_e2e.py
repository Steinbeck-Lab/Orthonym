"""
End-to-end tests for a phase accuracy fixes.

Tests the 6 bugs fixed in this phase:
-: Aromatic ring as alkyl chain in ion naming
-: Nitrile on benzene not named
-: Fused heterocycle substituent dropping
-: Chromene/coumarin naming failure
-: Carboxylate anion with aromatic parent (same as)
-: Thiazolidine/saturated heterocycle misidentification
"""

import pytest
from orthonym import name_compound


class TestBug1And5AromaticCarboxylate:
    """ &: Aromatic ring counted as alkyl chain in carboxylate naming."""

    def test_2_chlorobenzoate(self):
        result = name_compound('Clc1ccccc1C(=O)[O-]')
        assert result == '2-chlorobenzoate', f"Got {result}"

    def test_3_chlorobenzoate(self):
        result = name_compound('Clc1cccc(C(=O)[O-])c1')
        assert result == '3-chlorobenzoate', f"Got {result}"

    def test_4_methylbenzoate(self):
        result = name_compound('Cc1ccc(C(=O)[O-])cc1')
        assert result == '4-methylbenzoate', f"Got {result}"


class TestBug2BenzeneNitrile:
    """: Nitrile on benzene not named as benzonitrile."""

    def test_2_fluorobenzonitrile(self):
        result = name_compound('Fc1ccccc1C#N')
        assert result == '2-fluorobenzonitrile', f"Got {result}"


class TestBug3FusedHeterocycleSubstituents:
    """: Fused heterocycle substituent dropping."""

    def test_indole_acetonitrile(self):
        """Critical success criteria #3 from internal notes.

        a phase Plan 02 Task 03: post-148 (a) chain-as-parent yields
        `2-(1H-indol-3-yl)ethanenitrile` (suffix form) instead of v17
        ring-as-parent `cyanomethyl/acetonitrile` prefix. Both are
        IUPAC-acceptable; is correct per (a).
        """
        result = name_compound('N#CCc1c[nH]c2ccccc12')
        # Should contain indol AND cyanomethyl/acetonitrile/ethanenitrile in some form
        result_lower = result.lower()
        has_indol = 'indol' in result_lower
        has_nitrile_token = (
            'cyanomethyl' in result_lower
            or 'acetonitrile' in result_lower
            or 'ethanenitrile' in result_lower  # a phase (a) chain-as-parent
        )
        assert has_indol and has_nitrile_token, (
            f"Expected indol and nitrile token (cyanomethyl/acetonitrile/ethanenitrile), got {result}"
        )

    def test_indole_acetic_acid(self):
        """a phase Plan 02 Task 03: post-148 chain-as-parent suffix
        `ethanoic acid` accepted alongside v17 `carboxymethyl/acetic` prefix.
        """
        result = name_compound('OC(=O)Cc1c[nH]c2ccccc12')
        result_lower = result.lower()
        has_indol = 'indol' in result_lower
        has_acid_token = (
            'carboxymethyl' in result_lower
            or 'acetic' in result_lower
            or 'ethanoic' in result_lower  # a phase (a) chain-as-parent
        )
        assert has_indol and has_acid_token, (
            f"Expected indol and acid token (carboxymethyl/acetic/ethanoic), got {result}"
        )

    def test_plain_indole(self):
        """Ensure plain indole still works."""
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-indole' in result or 'indole' in result.lower(), f"Got {result}"


class TestBug4ChromeneCoumarin:
    """: Chromene/benzopyran naming failure."""

    def test_dihydrobenzofuran(self):
        result = name_compound('C1Cc2ccccc2O1')
        result_lower = result.lower()
        assert 'benzofuran' in result_lower or 'dihydro' in result_lower, f"Got {result}"

    def test_chromane(self):
        """: chromane PIN is 3,4-dihydro-2H-1-benzopyran."""
        result = name_compound('c1ccc2OCCCc2c1')
        assert result == '3,4-dihydro-2H-1-benzopyran', f"Got {result}"


class TestBug6Thiazolidine:
    """: Thiazolidine/saturated heterocycle misidentification."""

    def test_isothiazolidine(self):
        """Isothiazolidine: 5-membered ring with N and S adjacent."""
        result = name_compound('C1CSNC1')  # S-N adjacent (canonicalizes to C1CNSC1)
        # (the Blue Book): 'isothiazolidine... 1,2-thiazolidine (PIN)'.
        assert result == '1,2-thiazolidine', f"Got {result}"


class TestRegressionExistingFunctionality:
    """Ensure existing functionality still works after a phase changes."""

    def test_indole(self):
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-indole' in result or 'indole' in result.lower(), f"Got {result}"


class TestAllSuccessCriteria:
    """
    Explicit test class for all internal notes success criteria.
    These tests are the primary validation for a phase completion.
    """

    def test_success_criteria_3_indole_acetonitrile(self):
        """: indole-3-acetonitrile contains indol and a nitrile token.

        a phase Plan 02 Task 03: post-148 (a) chain-as-parent yields
        `2-(1H-indol-3-yl)ethanenitrile` (suffix form). Accepting all three
        renderings (v17 cyanomethyl/acetonitrile prefix + ethanenitrile
        suffix) preserves the spirit of while accommodating the
        IUPAC-correct cascade.
        """
        result = name_compound('N#CCc1c[nH]c2ccccc12')
        result_lower = result.lower()
        assert 'indol' in result_lower
        assert (
            'cyanomethyl' in result_lower
            or 'acetonitrile' in result_lower
            or 'ethanenitrile' in result_lower  # a phase (a) chain-as-parent
        )


class TestAdditionalCoverage:
    """Additional E2E tests for a phase coverage."""

    def test_multiple_substituents_on_benzoate(self):
        """Test benzoate with multiple substituents."""
        result = name_compound('[O-]C(=O)c1cc(Cl)cc(Cl)c1')
        assert 'chlorobenzoate' in result.lower(), f"Got {result}"

    def test_benzofuran(self):
        """Test benzofuran naming."""
        result = name_compound('c1ccc2occc2c1')
        assert '1-benzofuran' in result or 'benzofuran' in result.lower(), f"Got {result}"

    def test_quinoline(self):
        """Test quinoline naming (fused heterocycle)."""
        result = name_compound('c1ccc2ncccc2c1')
        assert 'quinoline' in result.lower(), f"Got {result}"
