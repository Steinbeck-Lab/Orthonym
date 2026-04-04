"""Tests for malformed suffix form corrections (ASML-17).

Verifies that parent_to_prefix produces correct IUPAC substituent prefix
forms for retained names, not naive -yl appendages like 'adenineyl'.
"""
import pytest
from orthonym.assembly.substituent_naming import parent_to_prefix


@pytest.mark.unit
class TestRetainedNamePrefixLookup:
    """ASML-17: Retained names use lookup table for correct prefix forms."""

    @pytest.mark.parametrize("parent_name,expected", [
        ("adenine", "adenin-9-yl"),
        ("guanine", "guanin-9-yl"),
        ("indole", "1H-indol-3-yl"),
        ("purine", "purin-9-yl"),
        ("uracil", "uracil-1-yl"),
        ("thymine", "thymin-1-yl"),
        ("cytosine", "cytosin-1-yl"),
    ])
    def test_heterocyclic_retained_names(self, parent_name, expected):
        result = parent_to_prefix(parent_name, 5)
        assert result == expected, f"{parent_name} -> {result}, expected '{expected}'"

    @pytest.mark.parametrize("parent_name,expected", [
        ("glutaric acid", "glutaryl"),
        ("succinic acid", "succinyl"),
        ("malonic acid", "malonyl"),
    ])
    def test_acid_retained_names(self, parent_name, expected):
        result = parent_to_prefix(parent_name, 5)
        assert result == expected, f"{parent_name} -> {result}, expected '{expected}'"


@pytest.mark.unit
class TestAmideFallbackFix:
    """ASML-17: -amide names produce correct prefix forms, not 'amideyl'."""

    def test_formamide(self):
        result = parent_to_prefix("formamide", 1)
        assert "amideyl" not in result, f"Got malformed: {result}"
        assert "amido" in result or "amino" in result or "carbamoyl" in result, \
            f"Expected amido/amino/carbamoyl form, got: {result}"

    def test_acetamide(self):
        result = parent_to_prefix("acetamide", 2)
        assert "amideyl" not in result, f"Got malformed: {result}"


@pytest.mark.unit
class TestExistingBehaviorPreserved:
    """ASML-17 regression: existing parent_to_prefix behavior unchanged."""

    def test_propane(self):
        result = parent_to_prefix("propane", 3)
        assert result == "propyl", f"Expected propyl, got: {result}"

    def test_pyridine(self):
        result = parent_to_prefix("pyridine", 5)
        assert "pyridinyl" in result, f"Expected pyridinyl, got: {result}"

    def test_propanol(self):
        result = parent_to_prefix("propan-2-ol", 3)
        assert "hydroxy" in result, f"Expected hydroxy, got: {result}"
        assert "propyl" in result or "prop" in result, f"Expected prop stem, got: {result}"

    def test_butanoic_acid(self):
        result = parent_to_prefix("butanoic acid", 4)
        assert "carboxy" in result, f"Expected carboxy, got: {result}"
