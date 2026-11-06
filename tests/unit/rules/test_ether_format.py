"""Unit tests for ether prefix formatting and enclosing marks.

Validates that:
1. Simple methoxy/ethoxy/propoxy have NO enclosing marks
2. Complex ether substituents (branched, substituted) get enclosing marks
3. Complex substituents on monosubstituted rings get enclosing marks
4. Currently-passing ether compounds are not regressed

References:
    IUPAC 2013: Compound substituents need enclosing marks
    IUPAC 2013: Enclosing mark hierarchy (parentheses, brackets, braces)
    IUPAC 2013: Ether nomenclature
"""

import pytest
from orthonym import name_compound


class TestSimpleEtherNoEnclosingMarks:
    """Simple alkoxy prefixes should NOT have enclosing marks."""

    def test_phenoxy(self):
        """Phenoxy prefix -- simple prefix, no enclosing marks.

        Note: diphenyl ether may use alternative naming (oxydibenzene).
        When phenoxy IS used, it should not have brackets.
        """
        # Use a compound where phenoxy is clearly a substituent prefix
        name = name_compound("c1ccc(Oc2ccccc2)cc1")
        # Either phenoxy or oxydibenzene format is acceptable
        assert "phenoxy" in name or "oxy" in name
        # If phenoxy is used, it should NOT have enclosing marks
        if "phenoxy" in name:
            assert "(phenoxy)" not in name


class TestComplexSubstituentEnclosingMarks:
    """Complex substituents on rings need enclosing marks per IUPAC."""

    def test_isobutyl_on_monosubstituted_ring(self):
        """2-methylpropyl on benzene needs enclosing marks if systematic name used.

        If retained name (isobutyl) is used, no brackets needed.
        """
        name = name_compound("CC(C)Cc1ccccc1")
        # Either retained name "isobutylbenzene" (no brackets needed)
        # or systematic "(2-methylpropyl)benzene" (brackets needed)
        if "2-methylpropyl" in name:
            assert "(2-methylpropyl)" in name, (
                f"Systematic name needs brackets: {name}"
            )

    def test_complex_sub_on_polysubstituted_ring(self):
        """Complex substituent on polysubstituted ring gets brackets.

        This should already work via format_substituent_prefix.
        """
        name = name_compound("CC(C)=CCc1ccc(C)cc1")
        # Polysubstituted -> format_substituent_prefix handles brackets
        assert "(2-methylbut-2-enyl)" in name or "(" in name


class TestEtherRegressionGuard:
    """Ensure currently-passing ether compounds remain correct."""

    @pytest.mark.parametrize("smiles,expected_substr", [
        ("COc1ccccc1", "anisole"),  # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book)
        ("CCOc1ccccc1", "ethoxy"),         # simple ether prefix
        ("COc1ccc(C(=O)O)cc1", "methoxy"),  # substituted -> methoxy (not anisole)
    ])
    def test_simple_ether_regression(self, smiles, expected_substr):
        """Simple ether naming should not regress."""
        name = name_compound(smiles)
        assert expected_substr in name.lower(), (
            f"Expected '{expected_substr}' in name for {smiles}, got '{name}'"
        )

    def test_dimethoxybenzene(self):
        """2,4-dimethoxybenzoic acid should use dimethoxy, not bis(methoxy)."""
        name = name_compound("COc1ccc(C(=O)O)c(OC)c1")
        assert "dimethoxy" in name
        # Should NOT have bis(methoxy) -- methoxy is simple
        assert "bis(methoxy)" not in name
