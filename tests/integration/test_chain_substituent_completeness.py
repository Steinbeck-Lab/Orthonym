"""
Integration tests for chain substituent completeness (a phase, Plan 01).

Tests that heteroatom-containing substituent branches on acyclic chain systems
are correctly named, not dropped, and not double-counted. Covers bugs A-F:
  : Haloalkyl substituent naming
  : FG-on-branch double-counting
  : Aminomethyl/hydroxyalkyl locant correctness
  : Unrecognized heteroatom fallback

These tests assert that the generated name CONTAINS the expected fragment,
rather than matching an exact name, to be robust against minor locant or
formatting differences.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# / BUG-A: Haloalkyl substituent naming
# Previously: halogens on branches were named as standalone fluoro/chloro
# prefixes with wrong locants, instead of compound substituent names like
# (trifluoromethyl) or (fluoromethyl).
# ---------------------------------------------------------------------------

class TestHaloalkylChainSubstituents:
    """Haloalkyl branches produce correct compound substituent names."""

    @pytest.mark.integration
    def test_trifluoromethyl_on_acid(self):
        """CCCC(C(F)(F)F)CC(=O)O -> contains 'trifluoromethyl'."""
        name = name_compound('CCCC(C(F)(F)F)CC(=O)O')
        assert 'trifluoromethyl' in name, f"Expected 'trifluoromethyl' in '{name}'"

    @pytest.mark.integration
    def test_fluoromethyl_on_acid(self):
        """CCCC(CF)CC(=O)O -> contains 'fluoromethyl'."""
        name = name_compound('CCCC(CF)CC(=O)O')
        assert 'fluoromethyl' in name, f"Expected 'fluoromethyl' in '{name}'"

    @pytest.mark.integration
    def test_dichloromethyl_on_acid(self):
        """CCC(C(Cl)Cl)CC(=O)O -> contains 'dichloromethyl'."""
        name = name_compound('CCC(C(Cl)Cl)CC(=O)O')
        assert 'dichloromethyl' in name, f"Expected 'dichloromethyl' in '{name}'"

    @pytest.mark.integration
    def test_chloromethyl_on_acid(self):
        """CCC(CCl)CC(=O)O -> contains 'chloromethyl'."""
        name = name_compound('CCC(CCl)CC(=O)O')
        assert 'chloromethyl' in name, f"Expected 'chloromethyl' in '{name}'"

    @pytest.mark.integration
    def test_difluoromethyl_on_acid(self):
        """CCC(C(F)F)CC(=O)O -> contains 'difluoromethyl'."""
        name = name_compound('CCC(C(F)F)CC(=O)O')
        assert 'difluoromethyl' in name, f"Expected 'difluoromethyl' in '{name}'"

    @pytest.mark.integration
    def test_mixed_halogen_chlorofluoromethyl(self):
        """CCC(C(Cl)F)CC(=O)O -> contains both 'chloro' and 'fluoro' in a methyl."""
        name = name_compound('CCC(C(Cl)F)CC(=O)O')
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"
        assert 'fluoro' in name, f"Expected 'fluoro' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"

    @pytest.mark.integration
    def test_fluoroethyl_on_acid(self):
        """CCCC(CCF)CC(=O)O -> contains '2-fluoroethyl' (multi-carbon haloalkyl)."""
        name = name_compound('CCCC(CCF)CC(=O)O')
        assert 'fluoroethyl' in name, f"Expected 'fluoroethyl' in '{name}'"

    @pytest.mark.integration
    def test_trifluoroethyl_on_acid(self):
        """CCCC(CC(F)(F)F)CC(=O)O -> contains 'trifluoroethyl'."""
        name = name_compound('CCCC(CC(F)(F)F)CC(=O)O')
        assert 'trifluoroethyl' in name, f"Expected 'trifluoroethyl' in '{name}'"


# ---------------------------------------------------------------------------
# / BUG-E: Haloalkyl names get parentheses
# Previously: compound haloalkyl names were not recognized as needing brackets.
# ---------------------------------------------------------------------------

class TestHaloalkylBrackets:
    """Compound haloalkyl substituent names are wrapped in parentheses."""

    @pytest.mark.integration
    def test_trifluoromethyl_has_brackets(self):
        """CCCC(C(F)(F)F)CC(=O)O -> contains '(trifluoromethyl)' with parens."""
        name = name_compound('CCCC(C(F)(F)F)CC(=O)O')
        assert '(trifluoromethyl)' in name, f"Expected '(trifluoromethyl)' in '{name}'"

    @pytest.mark.integration
    def test_fluoromethyl_has_brackets(self):
        """CCCC(CF)CC(=O)O -> contains '(fluoromethyl)' with parens."""
        name = name_compound('CCCC(CF)CC(=O)O')
        assert '(fluoromethyl)' in name, f"Expected '(fluoromethyl)' in '{name}'"


# ---------------------------------------------------------------------------
# / BUG-B: No double-counted FG prefixes for branch-located FGs
# Previously: a hydroxymethyl branch produced both "(hydroxymethyl)" and a
# standalone "hydroxy" prefix, doubling the count.
# ---------------------------------------------------------------------------

class TestNoDoubleFGPrefix:
    """FGs on small branches are only named once (in compound substituent name)."""

    @pytest.mark.integration
    def test_hydroxy_branch_no_duplicate(self):
        """CCCC(CO)CC(=O)O -> 'hydroxy' appears exactly once (in hydroxymethyl)."""
        name = name_compound('CCCC(CO)CC(=O)O')
        assert name.count('hydroxy') == 1, (
            f"Expected exactly 1 'hydroxy' in '{name}', found {name.count('hydroxy')}"
        )

    @pytest.mark.integration
    def test_amino_branch_no_duplicate(self):
        """CCCC(CN)CC(=O)O -> 'amino' appears exactly once (in aminomethyl)."""
        name = name_compound('CCCC(CN)CC(=O)O')
        assert name.count('amino') == 1, (
            f"Expected exactly 1 'amino' in '{name}', found {name.count('amino')}"
        )

    @pytest.mark.integration
    def test_fluoro_branch_no_duplicate(self):
        """CCCC(CF)CC(=O)O -> 'fluoro' appears exactly once (in fluoromethyl)."""
        name = name_compound('CCCC(CF)CC(=O)O')
        assert name.count('fluoro') == 1, (
            f"Expected exactly 1 'fluoro' in '{name}', found {name.count('fluoro')}"
        )


# ---------------------------------------------------------------------------
# / BUG-C: Aminomethyl locant correctness
# Previously: aminomethyl had incorrect "1-amino" locant on 1-carbon chains.
# ---------------------------------------------------------------------------

class TestAminomethylLocant:
    """Amino locant: omitted for 1-carbon, present for 2+ carbon chains."""

    @pytest.mark.integration
    def test_aminomethyl_no_locant(self):
        """CCCC(CN)CC(=O)O -> 'aminomethyl' with no locant prefix."""
        name = name_compound('CCCC(CN)CC(=O)O')
        assert 'aminomethyl' in name, f"Expected 'aminomethyl' in '{name}'"
        assert '1-aminomethyl' not in name, f"Should NOT have '1-aminomethyl' in '{name}'"
        assert '1-amino' not in name, f"Should NOT have '1-amino' in '{name}'"

    @pytest.mark.integration
    def test_aminoethyl_with_locant(self):
        """CCCC(CCN)CC(=O)O -> '2-aminoethyl' (locant included for 2-carbon)."""
        name = name_compound('CCCC(CCN)CC(=O)O')
        assert '2-aminoethyl' in name, f"Expected '2-aminoethyl' in '{name}'"


# ---------------------------------------------------------------------------
# / BUG-D: Hydroxyalkyl locant correctness
# Previously: 2-hydroxyethyl was missing the locant (just 'hydroxyethyl').
# ---------------------------------------------------------------------------

class TestHydroxyalkylLocant:
    """Hydroxy locant: omitted for 1-carbon, present for 2+ carbon chains."""

    @pytest.mark.integration
    def test_hydroxymethyl_no_locant(self):
        """CCCC(CO)CC(=O)O -> 'hydroxymethyl' with no locant prefix."""
        name = name_compound('CCCC(CO)CC(=O)O')
        assert 'hydroxymethyl' in name, f"Expected 'hydroxymethyl' in '{name}'"

    @pytest.mark.integration
    def test_hydroxyethyl_with_locant(self):
        """CCCC(CCO)CC(=O)O -> '2-hydroxyethyl' (locant included for 2-carbon)."""
        name = name_compound('CCCC(CCO)CC(=O)O')
        assert '2-hydroxyethyl' in name, f"Expected '2-hydroxyethyl' in '{name}'"

    @pytest.mark.integration
    def test_hydroxyethyl_no_plain_form(self):
        """CCCC(CCO)CC(=O)O -> should NOT contain plain 'hydroxyethyl' without locant."""
        name = name_compound('CCCC(CCO)CC(=O)O')
        # Must not have plain 'hydroxyethyl' -- should be '2-hydroxyethyl'
        assert '2-hydroxy' in name, f"Expected '2-hydroxy' locant in '{name}'"


# ---------------------------------------------------------------------------
# / BUG-F: Generic fallback for unrecognized heteroatom substituents
# Previously: unrecognized heteroatom branches were silently dropped.
# Now: logged warning + fallback to carbon-count-based alkyl name.
# ---------------------------------------------------------------------------

class TestGenericFallback:
    """Heteroatom substituents produce a name with the substituent present."""

    @pytest.mark.integration
    def test_thiol_branch_produces_name(self):
        """CCCC(CS)CC(=O)O -> name includes sulfanyl substituent."""
        name = name_compound('CCCC(CS)CC(=O)O')
        assert name and len(name) > 0, "Expected non-empty name for thiol branch compound"
        assert 'hexanoic acid' in name, f"Expected 'hexanoic acid' base in '{name}'"
        # a phase: sulfanyl substituent is now present (was previously dropped)
        assert 'sulfanyl' in name, f"Expected 'sulfanyl' in '{name}'"

    @pytest.mark.integration
    def test_thiol_branch_not_crashing(self):
        """Various unusual heteroatom branches should not crash."""
        # Thioether on branch
        result = name_compound('CCCC(CS)CC(=O)O')
        assert result is not None and isinstance(result, str)

    @pytest.mark.integration
    def test_thioether_branch_has_sulfanyl(self):
        """CCCC(SC)CC(=O)O -> name includes methylsulfanyl substituent."""
        name = name_compound('CCCC(SC)CC(=O)O')
        assert 'sulfanyl' in name, f"Expected 'sulfanyl' in '{name}'"
        assert 'hexanoic acid' in name, f"Expected 'hexanoic acid' base in '{name}'"
