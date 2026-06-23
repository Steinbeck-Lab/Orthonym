"""
End-to-end tests for Phase 14.6 accuracy fixes.

Tests the 6 bugs fixed in this phase:
- BUG-1: Aromatic ring as alkyl chain in ion naming
- BUG-2: Nitrile on benzene not named
- BUG-3: Fused heterocycle substituent dropping
- BUG-4: Chromene/coumarin naming failure
- BUG-5: Carboxylate anion with aromatic parent (same as BUG-1)
- BUG-6: Thiazolidine/saturated heterocycle misidentification
"""

import pytest
from orthonym import name_compound


class TestBug1And5AromaticCarboxylate:
    """BUG-1 & BUG-5: Aromatic ring counted as alkyl chain in carboxylate naming."""

    def test_4_chlorobenzoate(self):
        """Critical success criteria #1 from CONTEXT.md"""
        result = name_compound('O=C([O-])c1ccc(Cl)cc1')
        assert result == '4-chlorobenzoate', f"Got {result}"

    def test_simple_benzoate(self):
        result = name_compound('O=C([O-])c1ccccc1')
        assert result == 'benzoate', f"Got {result}"

    def test_2_chlorobenzoate(self):
        result = name_compound('Clc1ccccc1C(=O)[O-]')
        assert result == '2-chlorobenzoate', f"Got {result}"

    def test_3_chlorobenzoate(self):
        result = name_compound('Clc1cccc(C(=O)[O-])c1')
        assert result == '3-chlorobenzoate', f"Got {result}"

    def test_4_methylbenzoate(self):
        result = name_compound('Cc1ccc(C(=O)[O-])cc1')
        assert result == '4-methylbenzoate', f"Got {result}"

    def test_acetate_regression(self):
        """Ensure acyclic carboxylates still work."""
        result = name_compound('CC(=O)[O-]')
        assert result == 'acetate', f"Got {result}"

    def test_propanoate_regression(self):
        """Ensure longer chain carboxylates still work."""
        result = name_compound('CCC(=O)[O-]')
        assert result == 'propanoate', f"Got {result}"


class TestBug2BenzeneNitrile:
    """BUG-2: Nitrile on benzene not named as benzonitrile."""

    def test_benzonitrile(self):
        """Critical success criteria #2 from CONTEXT.md"""
        result = name_compound('c1ccccc1C#N')
        assert result == 'benzonitrile', f"Got {result}"

    def test_4_chlorobenzonitrile(self):
        result = name_compound('Clc1ccc(C#N)cc1')
        assert result == '4-chlorobenzonitrile', f"Got {result}"

    def test_3_methylbenzonitrile(self):
        result = name_compound('Cc1cccc(C#N)c1')
        assert result == '3-methylbenzonitrile', f"Got {result}"

    def test_2_fluorobenzonitrile(self):
        result = name_compound('Fc1ccccc1C#N')
        assert result == '2-fluorobenzonitrile', f"Got {result}"


class TestBug3FusedHeterocycleSubstituents:
    """BUG-3: Fused heterocycle substituent dropping."""

    def test_indole_acetonitrile(self):
        """Critical success criteria #3 from CONTEXT.md.

        Phase 148 Plan 02 Task 03: post-148 P-44.1(a) chain-as-parent yields
        `2-(1H-indol-3-yl)ethanenitrile` (suffix form) instead of v17
        ring-as-parent `cyanomethyl/acetonitrile` prefix. Both are
        IUPAC-acceptable; v18 is correct per P-44.1(a).
        """
        result = name_compound('N#CCc1c[nH]c2ccccc12')
        # Should contain indol AND cyanomethyl/acetonitrile/ethanenitrile in some form
        result_lower = result.lower()
        has_indol = 'indol' in result_lower
        has_nitrile_token = (
            'cyanomethyl' in result_lower
            or 'acetonitrile' in result_lower
            or 'ethanenitrile' in result_lower  # Phase 148 P-44.1(a) chain-as-parent
        )
        assert has_indol and has_nitrile_token, (
            f"Expected indol and nitrile token (cyanomethyl/acetonitrile/ethanenitrile), got {result}"
        )

    def test_indole_acetic_acid(self):
        """Phase 148 Plan 02 Task 03: post-148 chain-as-parent suffix
        `ethanoic acid` accepted alongside v17 `carboxymethyl/acetic` prefix.
        """
        result = name_compound('OC(=O)Cc1c[nH]c2ccccc12')
        result_lower = result.lower()
        has_indol = 'indol' in result_lower
        has_acid_token = (
            'carboxymethyl' in result_lower
            or 'acetic' in result_lower
            or 'ethanoic' in result_lower  # Phase 148 P-44.1(a) chain-as-parent
        )
        assert has_indol and has_acid_token, (
            f"Expected indol and acid token (carboxymethyl/acetic/ethanoic), got {result}"
        )

    def test_plain_indole(self):
        """Ensure plain indole still works."""
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-indole' in result or 'indole' in result.lower(), f"Got {result}"


class TestBug4ChromeneCoumarin:
    """BUG-4: Chromene/benzopyran naming failure."""

    def test_coumarin(self):
        """v23 IH-01f: PIN is 2H-1-benzopyran-2-one (1-benzopyran is the PIN ring
        parent per P-19(d); coumarin/chromene are general-nomenclature only)."""
        result = name_compound('O=c1ccc2ccccc2o1')
        assert result == '2H-1-benzopyran-2-one', f"Got {result}"

    def test_dihydrobenzofuran(self):
        result = name_compound('C1Cc2ccccc2O1')
        result_lower = result.lower()
        assert 'benzofuran' in result_lower or 'dihydro' in result_lower, f"Got {result}"

    def test_chromane(self):
        """Chromane (3,4-dihydro-2H-chromene) should be named correctly."""
        result = name_compound('c1ccc2OCCCc2c1')
        result_lower = result.lower()
        assert 'chromane' in result_lower or 'chromen' in result_lower, f"Got {result}"


class TestBug6Thiazolidine:
    """BUG-6: Thiazolidine/saturated heterocycle misidentification."""

    def test_thiazolidine(self):
        """Thiazolidine: 5-membered ring with N and S not adjacent."""
        result = name_compound('C1CSCN1')  # S-C-C-N (not adjacent)
        assert result == 'thiazolidine', f"Got {result}"

    def test_isothiazolidine(self):
        """Isothiazolidine: 5-membered ring with N and S adjacent."""
        result = name_compound('C1CSNC1')  # S-N adjacent (canonicalizes to C1CNSC1)
        assert result == 'isothiazolidine', f"Got {result}"


class TestRegressionExistingFunctionality:
    """Ensure existing functionality still works after Phase 14.6 changes."""

    def test_benzene(self):
        assert name_compound('c1ccccc1') == 'benzene'

    def test_indole(self):
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-indole' in result or 'indole' in result.lower(), f"Got {result}"

    def test_benzoic_acid(self):
        result = name_compound('c1ccc(C(=O)O)cc1')
        assert result == 'benzoic acid', f"Got {result}"

    def test_ethanol(self):
        assert name_compound('CCO') == 'ethanol'

    def test_pyridine(self):
        assert name_compound('c1ccncc1') == 'pyridine'

    def test_furan(self):
        assert name_compound('c1ccoc1') == 'furan'

    def test_thiophene(self):
        assert name_compound('c1ccsc1') == 'thiophene'

    def test_naphthalene(self):
        assert name_compound('c1ccc2ccccc2c1') == 'naphthalene'


class TestAllSuccessCriteria:
    """
    Explicit test class for all CONTEXT.md success criteria.
    These tests are the primary validation for Phase 14.6 completion.
    """

    def test_success_criteria_1_aromatic_carboxylate(self):
        """SC-1: 4-chlorobenzoate names correctly."""
        result = name_compound('O=C([O-])c1ccc(Cl)cc1')
        assert result == '4-chlorobenzoate'

    def test_success_criteria_2_benzonitrile(self):
        """SC-2: benzonitrile names correctly."""
        result = name_compound('c1ccccc1C#N')
        assert result == 'benzonitrile'

    def test_success_criteria_3_indole_acetonitrile(self):
        """SC-3: indole-3-acetonitrile contains indol and a nitrile token.

        Phase 148 Plan 02 Task 03: post-148 P-44.1(a) chain-as-parent yields
        `2-(1H-indol-3-yl)ethanenitrile` (suffix form). Accepting all three
        renderings (v17 cyanomethyl/acetonitrile prefix + v18 ethanenitrile
        suffix) preserves the spirit of SC-3 while accommodating the
        IUPAC-correct cascade.
        """
        result = name_compound('N#CCc1c[nH]c2ccccc12')
        result_lower = result.lower()
        assert 'indol' in result_lower
        assert (
            'cyanomethyl' in result_lower
            or 'acetonitrile' in result_lower
            or 'ethanenitrile' in result_lower  # Phase 148 P-44.1(a) chain-as-parent
        )

    def test_success_criteria_4_coumarin(self):
        """SC-4: coumarin names as the PIN 2H-1-benzopyran-2-one (v23 IH-01f, P-19(d))."""
        result = name_compound('O=c1ccc2ccccc2o1')
        assert result == '2H-1-benzopyran-2-one', f"Got {result}"


class TestAdditionalCoverage:
    """Additional E2E tests for Phase 14.6 coverage."""

    def test_multiple_substituents_on_benzoate(self):
        """Test benzoate with multiple substituents."""
        result = name_compound('[O-]C(=O)c1cc(Cl)cc(Cl)c1')
        assert 'chlorobenzoate' in result.lower(), f"Got {result}"

    def test_naphthyl_carboxylate(self):
        """Test naphthalene carboxylate naming."""
        result = name_compound('O=C([O-])c1ccc2ccccc2c1')
        assert 'naphthoate' in result.lower(), f"Got {result}"

    def test_benzofuran(self):
        """Test benzofuran naming."""
        result = name_compound('c1ccc2occc2c1')
        assert '1-benzofuran' in result or 'benzofuran' in result.lower(), f"Got {result}"

    def test_quinoline(self):
        """Test quinoline naming (fused heterocycle)."""
        result = name_compound('c1ccc2ncccc2c1')
        assert 'quinoline' in result.lower(), f"Got {result}"
