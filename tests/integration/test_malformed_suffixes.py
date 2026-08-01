"""Tests for malformed suffix form corrections (ASML-17).

Verifies that parent_to_prefix produces correct IUPAC substituent prefix
forms for retained names, not naive -yl appendages like 'adenineyl'.
"""
import pytest
from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN, parent_to_prefix)


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
        result = parent_to_prefix(parent_name, 5, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result == expected, f"{parent_name} -> {result}, expected '{expected}'"

    @pytest.mark.parametrize("parent_name,expected", [
        ("glutaric acid", "glutaryl"),
        ("succinic acid", "succinyl"),
        ("malonic acid", "malonyl"),
    ])
    def test_acid_retained_names(self, parent_name, expected):
        result = parent_to_prefix(parent_name, 5, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result == expected, f"{parent_name} -> {result}, expected '{expected}'"


@pytest.mark.unit
class TestAmideFallbackFix:
    """ASML-17: -amide names produce correct prefix forms, not 'amideyl'."""

    def test_formamide(self):
        result = parent_to_prefix("formamide", 1, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert "amideyl" not in result, f"Got malformed: {result}"
        assert "amido" in result or "amino" in result or "carbamoyl" in result, \
            f"Expected amido/amino/carbamoyl form, got: {result}"

    def test_acetamide(self):
        result = parent_to_prefix("acetamide", 2, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert "amideyl" not in result, f"Got malformed: {result}"


@pytest.mark.unit
class TestExistingBehaviorPreserved:
    """ASML-17 regression: existing parent_to_prefix behavior unchanged."""

    def test_propane(self):
        result = parent_to_prefix("propane", 3, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert result == "propyl", f"Expected propyl, got: {result}"

    def test_pyridine(self):
        result = parent_to_prefix("pyridine", 5, attach_locant=ATTACH_LOCANT_UNKNOWN)
        assert "pyridinyl" in result, f"Expected pyridinyl, got: {result}"

    def test_propanol(self):
        # v29 residue Task A: the locanted -ol branch borrowed its locant from
        # the CAPPED molecule's numbering and now declines. The unlocanted,
        # one-position form still converts, which is what "existing behavior
        # preserved" means for this converter.
        assert parent_to_prefix(
            "propan-2-ol", 3, attach_locant=ATTACH_LOCANT_UNKNOWN) is None
        assert parent_to_prefix(
            "methanol", 1, attach_locant=ATTACH_LOCANT_UNKNOWN) == "hydroxymethyl"

    def test_butanoic_acid(self):
        # Task A: the carboxy locant came from a carbon COUNT; declines now.
        # The locant-free one-position form still converts.
        assert parent_to_prefix(
            "butanoic acid", 4, attach_locant=ATTACH_LOCANT_UNKNOWN) is None
        assert parent_to_prefix(
            "ethanoic acid", 2, attach_locant=ATTACH_LOCANT_UNKNOWN) == "carboxymethyl"
