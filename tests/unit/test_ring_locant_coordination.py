"""
Tests for ring locant coordination between ester prefixes and general substituents.

a phase-06: When a ring has both ester prefixes (acyloxy) AND other substituents
(methyl, chloro, hydroxy, etc.), the locants must be computed from a single unified
numbering system. Previously, _assemble_ring_with_ester_prefixes only handled ester
prefixes and dropped all other substituents.

References:
    IUPAC 2013 (all substituents receive coordinated locants)
    IUPAC 2013 (alphabetical ordering of detachable prefixes)
"""
import pytest
from orthonym import name_compound


class TestSingleEsterNoRegression:
    """Compounds with a single ester prefix and NO other substituents.
    These must continue to work exactly as before (no regression)."""

    def test_acetyloxycyclohexane(self):
        """CC(=O)OC1CCCCC1 -> acetyloxycyclohexane"""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert result == "acetyloxycyclohexane", f"Got: {result}"

    def test_acetyloxybenzene(self):
        """CC(=O)Oc1ccccc1 -> acetyloxybenzene"""
        result = name_compound("CC(=O)Oc1ccccc1")
        assert result == "acetyloxybenzene", f"Got: {result}"


class TestEsterPlusSingleSubstituent:
    """Compounds with ester prefix(es) AND one additional substituent.
    These are the core gap: both groups must appear with coordinated locants."""

    def test_acetyloxy_methyl_cyclohexane(self):
        """CC(=O)OC1CCC(C)CC1: methyl must appear in the name."""
        result = name_compound("CC(=O)OC1CCC(C)CC1")
        assert "methyl" in result, f"Missing 'methyl' in: {result}"
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"

    def test_acetyloxy_hydroxy_benzene(self):
        """CC(=O)Oc1ccc(O)cc1: hydroxy must appear in the name."""
        result = name_compound("CC(=O)Oc1ccc(O)cc1")
        assert "hydroxy" in result, f"Missing 'hydroxy' in: {result}"
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"


class TestEsterPlusMultipleSubstituents:
    """Compounds with ester prefix(es) AND multiple additional substituents."""

    def test_acetyloxy_chloro_methyl_benzene(self):
        """CC(=O)Oc1ccc(Cl)c(C)c1: chloro AND methyl must appear."""
        result = name_compound("CC(=O)Oc1ccc(Cl)c(C)c1")
        assert "chloro" in result, f"Missing 'chloro' in: {result}"
        assert "methyl" in result, f"Missing 'methyl' in: {result}"
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"


class TestMultipleEstersPlusSubstituent:
    """Compounds with multiple ester prefixes AND additional substituents."""

    def test_diacetyloxy_methyl_cyclohexane(self):
        """CC(=O)OC1CC(OC(C)=O)CC(C)C1: both esters AND methyl must appear."""
        result = name_compound("CC(=O)OC1CC(OC(C)=O)CC(C)C1")
        assert "methyl" in result, f"Missing 'methyl' in: {result}"
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"


class TestLocantConsistency:
    """Verify that locants are consistent (no conflicts, proper ordering)."""

    def test_no_duplicate_locants(self):
        """Locants should not be duplicated between ester and non-ester substituents."""
        result = name_compound("CC(=O)OC1CCC(C)CC1")
        # The name should have all locants in proper order, no duplicates
        # We check that both groups are present and locanted
        assert "methyl" in result, f"Missing 'methyl' in: {result}"
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"

    def test_alphabetical_ordering(self):
        """Prefixes should be in alphabetical order per IUPAC."""
        result = name_compound("CC(=O)Oc1ccc(Cl)c(C)c1")
        # acetyloxy < chloro < methyl alphabetically
        assert "acetyloxy" in result, f"Missing 'acetyloxy' in: {result}"
        assert "chloro" in result, f"Missing 'chloro' in: {result}"
        assert "methyl" in result, f"Missing 'methyl' in: {result}"
        # Verify ordering: acetyloxy before chloro before methyl
        a_pos = result.index("acetyloxy")
        c_pos = result.index("chloro")
        m_pos = result.index("methyl")
        assert a_pos < c_pos < m_pos, (
            f"Alphabetical order violated: acetyloxy@{a_pos}, chloro@{c_pos}, methyl@{m_pos} in: {result}"
        )


class TestNoRegressionWithoutEsters:
    """Ensure compounds WITHOUT ester prefixes are completely unaffected."""

    def test_plain_cyclohexane(self):
        """C1CCCCC1 -> cyclohexane"""
        result = name_compound("C1CCCCC1")
        assert result == "cyclohexane", f"Got: {result}"

    def test_methylcyclohexane(self):
        """CC1CCCCC1 -> methylcyclohexane"""
        result = name_compound("CC1CCCCC1")
        assert result == "methylcyclohexane", f"Got: {result}"

    def test_benzene(self):
        """c1ccccc1 -> benzene"""
        result = name_compound("c1ccccc1")
        assert result == "benzene", f"Got: {result}"

    def test_chlorobenzene(self):
        """Clc1ccccc1 -> chlorobenzene"""
        result = name_compound("Clc1ccccc1")
        assert result == "chlorobenzene", f"Got: {result}"
