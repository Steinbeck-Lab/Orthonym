"""Tests for PEP-04: Ring-containing acyl fragment naming in acylamino prefixes.

Bug: _check_for_acylamino() calls _count_carbon_chain() which traverses ring C-C bonds,
linearizing ring carbons. N-benzoylglycine (6 ring C + 1 carbonyl C = 7) becomes
"heptanoylamino" instead of a benzoyl-based name.

Fix: Detect rings in acyl fragment and use name_substituent_fragment() for ring-containing
acyl groups, preserving _count_carbon_chain() for linear chains.
"""

import pytest
from orthonym import name_compound


class TestAcylaminoRingFix:
    """Tests for ring-containing acyl fragments not being linearized."""

    def test_n_benzoylglycine_no_linearization(self):
        """N-benzoylglycine should NOT linearize benzene ring to 'heptanoyl'."""
        result = name_compound("OC(=O)CNC(=O)c1ccccc1")
        assert "heptanoyl" not in result.lower(), (
            f"Ring linearization bug: got '{result}', should not contain 'heptanoyl'"
        )
        assert "benz" in result.lower(), (
            f"Expected benzoyl-based name, got '{result}'"
        )

    def test_n_naphthoylglycine_no_linearization(self):
        """N-(1-naphthoyl)glycine should NOT linearize naphthalene to 'undecanoyl'."""
        result = name_compound("OC(=O)CNC(=O)c1cccc2ccccc12")
        assert "undecanoyl" not in result.lower(), (
            f"Ring linearization bug: got '{result}', should not contain 'undecanoyl'"
        )
        assert "naphth" in result.lower(), (
            f"Expected naphthoyl-based name, got '{result}'"
        )

    def test_n_toluoylglycine_no_linearization(self):
        """N-(4-methylbenzoyl)glycine should NOT linearize to 'octanoyl'."""
        result = name_compound("OC(=O)CNC(=O)c1ccc(C)cc1")
        assert "octanoyl" not in result.lower(), (
            f"Ring linearization bug: got '{result}', should not contain 'octanoyl'"
        )

    def test_linear_acylamino_still_works(self):
        """N-hexanoylglycine (linear chain) -> 'hexanamido' (P-66.1.1.4.3)."""
        result = name_compound("OC(=O)CNC(=O)CCCCC")
        assert "hexanamido" in result.lower(), (
            f"Expected 'hexanamido' in result, got '{result}'"
        )

    def test_n_acetylglycine_still_works(self):
        """N-acetylglycine (simplest acyl) -> 'acetamido' (P-66.1.1.4.3)."""
        result = name_compound("OC(=O)CNC(=O)C")
        assert "acetamido" in result.lower(), (
            f"Expected 'acetamido' in result, got '{result}'"
        )

    def test_n_cyclopentylcarbonylglycine_no_linearization(self):
        """N-cyclopentanecarbonylglycine should NOT linearize to 'hexanoyl'."""
        result = name_compound("OC(=O)CNC(=O)C1CCCC1")
        assert "hexanoyl" not in result.lower(), (
            f"Ring linearization bug: got '{result}', should not contain 'hexanoyl'"
        )
