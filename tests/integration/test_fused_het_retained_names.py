"""Tests for fused heterocycle retained names (post-Phase 148).

After Phase 148 deleted the strict-PG-count fused guard, these tests verify
the post-deletion stable behavior: known fused-hetero retained names are
preserved by the retained-name lookup path (NOT by the deleted guard, which
was protecting them via early bypass; cascade non-entry now protects them
because chain_len < 2 for all bare fused heterocycles).

Source: Phase 148 SC-1 cleanup; CONTEXT D-01.
Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
"""
import pytest

from orthonym import name_compound


class TestRetainedNamesPreserved:
    """Guard bypass does not break retained fused heterocycle names."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("CCc1nc2ccccc2[nH]1", "2-ethyl-1H-benzimidazole"),
        ("Nc1ccc2ncccc2c1", "quinolin-6-amine"),
        ("OC(=O)c1cnc2ccccc2c1", "quinoline-3-carboxylic acid"),
    ])
    def test_retained_name_preserved(self, smiles, expected):
        """Known fused heterocycle names must be preserved."""
        result = name_compound(smiles)
        assert result == expected, f"Expected {expected}, got {result}"


class TestTokenFilterExpansion:
    """DROP-04 token filter accepts fused heterocycle name stems."""

    def test_indole_prefix_not_dropped(self):
        """'indol' token should pass DROP-04 validation."""
        # Indole as substituent on ring-as-parent: ring naming path
        name = name_compound("N#CCc1c[nH]c2ccccc12")
        assert "indol" in name.lower(), f"Expected indol in name, got: {name}"

    def test_quinoline_prefix_not_dropped(self):
        """'quinolin' token should pass DROP-04 validation."""
        name = name_compound("Clc1ccc2ncccc2c1")
        assert "quinolin" in name.lower(), f"Expected quinolin in name, got: {name}"
