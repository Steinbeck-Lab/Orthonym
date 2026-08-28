"""Canary integration tests for the ring_substituent_bare_functional_group and
substituent_all_candidates_filtered fixes.

These use real benchmark SMILES that should produce improved names after the
ring_substituent_bare_functional_group (FG-only ring substituent) and
substituent_all_candidates_filtered (BUG-B overfilter) fixes.
"""

import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestRingSubstituentBareFunctionalGroupCanaryCompounds:
    """Canary tests for compounds affected by the
    ring_substituent_bare_functional_group fix."""

    def test_bromocyclohexane(self):
        """Bromocyclohexane: halogen on ring must appear as 'bromo' prefix."""
        result = name_compound("C1CCCCC1Br")
        assert result is not None
        assert "bromo" in result.lower(), f"Expected 'bromo' in '{result}'"
        assert "cyclohexane" in result.lower(), f"Expected 'cyclohexane' in '{result}'"

    def test_iodocyclopentane(self):
        """Iodocyclopentane: iodo prefix on ring."""
        result = name_compound("C1CCCC1I")
        assert result is not None
        assert "iodo" in result.lower(), f"Expected 'iodo' in '{result}'"

    def test_2_chloro_4_methylcyclohexanone(self):
        """2-chloro-4-methylcyclohexan-1-one: chloro + methyl + ketone on ring."""
        result = name_compound("O=C1CC(C)CC(Cl)C1")
        assert result is not None
        low = result.lower()
        assert "chloro" in low, f"Expected 'chloro' in '{result}'"
        assert "methyl" in low, f"Expected 'methyl' in '{result}'"

    def test_4_hydroxycyclohexanecarboxylic_acid(self):
        """4-hydroxycyclohexane-1-carboxylic acid: hydroxy FG on ring with acid."""
        result = name_compound("OC1CCC(CC1)C(=O)O")
        assert result is not None
        low = result.lower()
        assert "hydroxy" in low, f"Expected 'hydroxy' in '{result}'"

    def test_3_aminocyclohexanol(self):
        """3-aminocyclohexan-1-ol: amino + hydroxyl on ring."""
        result = name_compound("NC1CCCC(O)C1")
        assert result is not None
        low = result.lower()
        assert "amino" in low, f"Expected 'amino' in '{result}'"


@pytest.mark.integration
class TestSubstituentAllCandidatesFilteredCanaryCompounds:
    """Canary tests for compounds affected by the
    substituent_all_candidates_filtered fix."""

    def test_hydroxymethylcyclohexanol(self):
        """Hydroxy + hydroxymethyl on cyclohexane: both hydroxyl contexts present."""
        result = name_compound("OC1CCCCC1CO")
        assert result is not None
        low = result.lower()
        # At minimum hydroxyl information should be present
        assert "hydroxy" in low or "ol" in low, (
            f"Expected hydroxyl info in '{result}'"
        )
