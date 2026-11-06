"""Tests for: Ring-containing acyl fragment naming in acylamino prefixes.

Bug: _check_for_acylamino calls _count_carbon_chain which traverses ring C-C bonds,
linearizing ring carbons. N-benzoylglycine (6 ring C + 1 carbonyl C = 7) becomes
"heptanoylamino" instead of a benzoyl-based name.

Fix: Detect rings in acyl fragment and use name_substituent_fragment for ring-containing
acyl groups, preserving _count_carbon_chain for linear chains.
"""

from orthonym import name_compound


class TestAcylaminoRingFix:
    """Tests for ring-containing acyl fragments not being linearized."""

    def test_linear_acylamino_still_works(self):
        """N-hexanoylglycine (linear chain) -> 'hexanamido'."""
        result = name_compound("OC(=O)CNC(=O)CCCCC")
        assert "hexanamido" in result.lower(), (
            f"Expected 'hexanamido' in result, got '{result}'"
        )

    def test_n_acetylglycine_still_works(self):
        """N-acetylglycine (simplest acyl) -> 'acetamido'."""
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
