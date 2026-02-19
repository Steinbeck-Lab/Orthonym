"""
Phase 64: Charge and Ion Naming - Regression Tests

Tests for compounds fixed in Phase 64 Plan 01:
- A1: Single anion naming (-oate suffix via neutralize-then-name)
- A3: Salt naming (guanidinium, hydrogen prefix, neutral fragment handling)

Each test verifies a specific compound produces the expected name pattern.
When names improve in future phases, update the expected values.
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# A1: Single Anion Compounds
# These should produce -oate or -ate names for carboxylate anions.
# ---------------------------------------------------------------------------

class TestSingleAnionNaming:
    """Test single carboxylate anion compounds produce -oate names."""

    def test_a1_2_amino_acid_anion(self):
        """A1#2: amino acid carboxylate should produce -oate (not aminide)."""
        result = name_compound('CC[C@H](C)[C@H](N)C(=O)[O-]')
        assert 'oate' in result, f"Expected -oate suffix, got: {result}"
        assert 'aminide' not in result, f"Should not be aminide: {result}"

    def test_a1_3_ester_bearing_anion(self):
        """A1#3: ester-bearing carboxylate should produce -oate."""
        result = name_compound('C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O')
        assert 'oate' in result, f"Expected -oate suffix, got: {result}"
        assert result != '', "Should not be empty"

    def test_a1_4_phenoxy_anion(self):
        """A1#4: phenoxybutanoate anion."""
        result = name_compound('O=C([O-])CCCOc1ccc(Cl)cc1Cl')
        assert 'oate' in result or 'ate' in result, \
            f"Expected -oate/-ate suffix, got: {result}"

    def test_a1_6_poly_anion_dioate(self):
        """A1#6: poly-anion should produce -dioate (not -dioic acid)."""
        result = name_compound('O=C([O-])CC=CC(=O)C(=O)[O-]')
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"

    def test_a1_7_poly_anion_pentanedioate(self):
        """A1#7: poly-anion pentanedioate."""
        result = name_compound('O=C([O-])C(=O)C[C@H](O)C(=O)[O-]')
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"

    def test_a1_8_poly_anion_hexanedioate(self):
        """A1#8: poly-anion hexanedioate with stereo."""
        result = name_compound(
            'O=C([O-])[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)C(=O)[O-]'
        )
        assert 'dioate' in result, f"Expected -dioate suffix, got: {result}"
        assert 'dioic acid' not in result, \
            f"Should be -dioate not -dioic acid: {result}"


# ---------------------------------------------------------------------------
# A3: Salt Compounds
# These should produce correct "cation anion" format names.
# ---------------------------------------------------------------------------

class TestSaltNaming:
    """Test salt compounds produce correct IUPAC salt names."""

    def test_a3_1_sodium_hydrogen_fumarate(self):
        """A3#1: partial salt should have 'hydrogen' prefix."""
        result = name_compound('O=C([O-])/C=C/C(=O)O.[Na+]')
        assert 'sodium' in result, f"Expected 'sodium', got: {result}"
        assert 'hydrogen' in result, \
            f"Expected 'hydrogen' prefix for partial salt, got: {result}"

    def test_a3_2_guanidinium_salt(self):
        """A3#2: guanidinium should be correctly identified."""
        result = name_compound('NC(N)=[NH2+].O=C([O-])C(=O)O')
        assert 'guanidinium' in result, \
            f"Expected 'guanidinium', got: {result}"

    def test_a3_3_sodium_amino_acid_salt(self):
        """A3#3: sodium + amino acid anion salt."""
        result = name_compound(
            '[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-].[Na+]'
        )
        assert 'sodium' in result, f"Expected 'sodium', got: {result}"
        assert 'amino' in result or 'oate' in result, \
            f"Expected amino acid anion name, got: {result}"

    def test_a3_4_hydrochloride_salt(self):
        """A3#4: neutral organic + H+ + Cl- = hydrochloride."""
        result = name_compound(
            'COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]'
        )
        assert 'hydrochloride' in result, \
            f"Expected 'hydrochloride', got: {result}"

    def test_basic_sodium_acetate(self):
        """Basic salt: sodium acetate must not regress."""
        result = name_compound('[Na+].[O-]C(C)=O')
        assert result == 'sodium acetate', f"Got: {result}"

    def test_basic_potassium_chloride(self):
        """Basic salt: potassium chloride must not regress."""
        result = name_compound('[K+].[Cl-]')
        assert result == 'potassium chloride', f"Got: {result}"

    def test_basic_ammonium_chloride(self):
        """Basic salt: ammonium chloride must not regress."""
        result = name_compound('[NH4+].[Cl-]')
        assert result == 'ammonium chloride', f"Got: {result}"

    def test_guanidinium_retained_name(self):
        """Guanidinium cation lookup works correctly."""
        from orthonym.data.ion_retained_names import get_cation_name
        assert get_cation_name('NC(N)=[NH2+]') == 'guanidinium'


# ---------------------------------------------------------------------------
# Retained ion name stability
# ---------------------------------------------------------------------------

class TestRetainedIonNames:
    """Ensure retained ion names are stable after changes."""

    def test_acetate(self):
        assert name_compound('CC(=O)[O-]') == 'acetate'

    def test_benzoate(self):
        assert name_compound('O=C([O-])c1ccccc1') == 'benzoate'

    def test_naphthoate(self):
        result = name_compound('O=C([O-])c1ccc2ccccc2c1')
        assert result == 'naphthoate', f"Got: {result}"

    def test_4_chlorobenzoate(self):
        result = name_compound('O=C([O-])c1ccc(Cl)cc1')
        assert result == '4-chlorobenzoate', f"Got: {result}"

    def test_oxalate(self):
        result = name_compound('O=C([O-])C(=O)[O-]')
        assert result == 'oxalate', f"Got: {result}"
