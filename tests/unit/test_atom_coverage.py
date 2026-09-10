"""
Unit tests for the atom coverage validator.

Tests CoverageResult construction, threshold logic, and validate_atom_coverage
with both OPSIN-available and OPSIN-unavailable code paths.
"""

import shutil

import pytest
from rdkit import Chem

from orthonym.validation.atom_coverage import (
    CoverageResult,
    find_opsin_jar,
    validate_atom_coverage,
)


# ---------------------------------------------------------------------------
# CoverageResult dataclass tests
# ---------------------------------------------------------------------------


class TestCoverageResult:
    """Tests for the CoverageResult dataclass."""

    def test_construction_all_fields(self):
        """CoverageResult has all 8 expected fields."""
        result = CoverageResult(
            total_heavy_atoms=10,
            claimed_atoms=8,
            unclaimed_atoms=2,
            coverage_ratio=0.8,
            is_complete=True,
            method="parse_back",
            claimed_atom_indices={0, 1, 2, 3, 4, 5, 6, 7},
            unclaimed_atom_indices={8, 9},
        )
        assert result.total_heavy_atoms == 10
        assert result.claimed_atoms == 8
        assert result.unclaimed_atoms == 2
        assert result.coverage_ratio == 0.8
        assert result.is_complete is True
        assert result.method == "parse_back"
        assert result.claimed_atom_indices == {0, 1, 2, 3, 4, 5, 6, 7}
        assert result.unclaimed_atom_indices == {8, 9}

    def test_is_complete_field_roundtrips_true(self):
        """The dataclass stores is_complete=True as given.

        NOTE: this asserts only the value passed in -- it does NOT exercise
        any completeness RULE. It used to be named for a "0.80 ratio
        threshold" that no longer exists: completeness is now decided by
        constitution, in validate_atom_coverage. The real rule is tested in
        tests/unit/validation/test_atom_coverage_structural.py.
        """
        result = CoverageResult(
            total_heavy_atoms=10,
            claimed_atoms=8,
            unclaimed_atoms=2,
            coverage_ratio=0.80,
            is_complete=True,
            method="parse_back",
        )
        assert result.is_complete is True

    def test_is_complete_field_roundtrips_false(self):
        """The dataclass stores is_complete=False as given.

        As above: a field round-trip, not a rule. See
        tests/unit/validation/test_atom_coverage_structural.py.
        """
        result = CoverageResult(
            total_heavy_atoms=10,
            claimed_atoms=7,
            unclaimed_atoms=3,
            coverage_ratio=0.70,
            is_complete=False,
            method="parse_back",
        )
        assert result.is_complete is False

    def test_default_atom_indices_empty(self):
        """claimed/unclaimed atom indices default to empty sets."""
        result = CoverageResult(
            total_heavy_atoms=5,
            claimed_atoms=5,
            unclaimed_atoms=0,
            coverage_ratio=1.0,
            is_complete=True,
            method="parse_back",
        )
        assert result.claimed_atom_indices == set()
        assert result.unclaimed_atom_indices == set()


# ---------------------------------------------------------------------------
# validate_atom_coverage function tests
# ---------------------------------------------------------------------------

# Detect OPSIN availability once
_OPSIN_JAR = find_opsin_jar()
_JAVA_AVAILABLE = shutil.which("java") is not None
_OPSIN_AVAILABLE = _OPSIN_JAR is not None and _JAVA_AVAILABLE


class TestValidateAtomCoverage:
    """Tests for the validate_atom_coverage function."""

    def test_unavailable_when_no_opsin(self):
        """Returns method='unavailable' when OPSIN JAR is explicitly None."""
        mol = Chem.MolFromSmiles("CCO")
        result = validate_atom_coverage(mol, "ethanol", opsin_jar="/nonexistent/path.jar")
        # Should not crash; since JAR doesn't exist, subprocess will fail
        # and fall back to unavailable
        assert result.method in ("unavailable", "parse_back")
        assert result.total_heavy_atoms == 3

    def test_trivial_molecule_zero_heavy_atoms(self):
        """Molecule with zero heavy atoms gets trivial coverage."""
        mol = Chem.MolFromSmiles("[H][H]")
        result = validate_atom_coverage(mol, "dihydrogen", opsin_jar="/nonexistent.jar")
        assert result.total_heavy_atoms == 0
        assert result.coverage_ratio == 1.0
        assert result.is_complete is True
        assert result.method == "trivial"

    @pytest.mark.skipif(not _OPSIN_AVAILABLE, reason="OPSIN JAR or Java not available")
    def test_ethanol_parse_back(self):
        """Ethanol should get 100% coverage via parse-back."""
        mol = Chem.MolFromSmiles("CCO")
        result = validate_atom_coverage(mol, "ethanol", opsin_jar=_OPSIN_JAR)
        assert result.method == "parse_back"
        assert result.total_heavy_atoms == 3
        assert result.claimed_atoms == 3
        assert result.unclaimed_atoms == 0
        assert result.coverage_ratio == 1.0
        assert result.is_complete is True

    @pytest.mark.skipif(not _OPSIN_AVAILABLE, reason="OPSIN JAR or Java not available")
    def test_acetic_acid_parse_back(self):
        """Acetic acid should get 100% coverage via parse-back."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = validate_atom_coverage(mol, "acetic acid", opsin_jar=_OPSIN_JAR)
        assert result.method == "parse_back"
        assert result.total_heavy_atoms == 4
        assert result.claimed_atoms == 4
        assert result.coverage_ratio == 1.0
        assert result.is_complete is True

    @pytest.mark.skipif(not _OPSIN_AVAILABLE, reason="OPSIN JAR or Java not available")
    def test_incomplete_name_low_coverage(self):
        """A deliberately incomplete name should yield low coverage.

        We name a large molecule with just 'methanol' -- OPSIN will parse
        methanol (1 C + 1 O = 2 heavy atoms) while the input molecule has
        many more, so coverage should be well below 0.80.
        """
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")  # 16 carbons
        result = validate_atom_coverage(mol, "methanol", opsin_jar=_OPSIN_JAR)
        assert result.method == "parse_back"
        assert result.total_heavy_atoms == 16
        # methanol = 2 heavy atoms, so ratio = 2/16 = 0.125
        assert result.coverage_ratio < 0.80
        assert result.is_complete is False

    def test_empty_name_returns_unavailable(self):
        """Empty name string yields unavailable result."""
        mol = Chem.MolFromSmiles("CCO")
        result = validate_atom_coverage(mol, "", opsin_jar=_OPSIN_JAR)
        assert result.method == "unavailable"
        assert result.total_heavy_atoms == 3
