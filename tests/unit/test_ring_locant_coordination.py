"""
Tests for ring locant coordination between ester prefixes and general substituents.

a phase-06: When a ring has both ester prefixes (acyloxy) AND other substituents
(methyl, chloro, hydroxy, etc.), the locants must be computed from a single unified
numbering system. Previously, _assemble_ring_with_ester_prefixes only handled ester
prefixes and dropped all other substituents.

References:
    IUPAC 2013 (all substituents receive coordinated locants)
    IUPAC 2013 (alphabetical ordering of detachable prefixes)

Since the PIN of a monoester is the functional class name, the single-ester rows below
assert it, not the 'acyloxy' prefix form these tests were written for (a phase-06):
 "Preferred IUPAC names for esters", Monoesters (the Blue Book)
"Monoesters formed from a monobasic acid and a 'monohydroxylic' component are named
systematically by placing the 'hydroxylic' component denoted by an organyl group (alkyl,
aryl, etc.) in front of the name of the acid component expressed as an anion" ('ethyl
acetate (PIN)'; the aryl case is the Blue Book's '6-[4-(acetyloxy)phenyl]pyridin-3-yl
acetate (PIN)',:31895, whose acetate ester is the principal group). The acyloxy prefix
is for an ester that is not the principal characteristic group "Esters cited
as prefixes",:31696). Every name asserted here reads back to its input's full InChIKey
with OPSIN 2.9.0.
"""
import pytest
from orthonym import name_compound


class TestSingleEsterNoRegression:
    """Compounds with a single ester prefix and NO other substituents.
    These must continue to work exactly as before (no regression)."""

    def test_acetyloxycyclohexane(self):
        """CC(=O)OC1CCCCC1 -> cyclohexyl acetate (PIN; was acetyloxycyclohexane)"""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert result == "cyclohexyl acetate", f"Got: {result}"

    def test_acetyloxybenzene(self):
        """CC(=O)Oc1ccccc1 -> phenyl acetate (PIN; was acetyloxybenzene)"""
        result = name_compound("CC(=O)Oc1ccccc1")
        assert result == "phenyl acetate", f"Got: {result}"


class TestEsterPlusSingleSubstituent:
    """Compounds with ester prefix(es) AND one additional substituent.
    These are the core gap: both groups must appear with coordinated locants."""

    def test_acetyloxy_methyl_cyclohexane(self):
        """CC(=O)OC1CCC(C)CC1: the methyl must appear, at its locant, in the ester name."""
        result = name_compound("CC(=O)OC1CCC(C)CC1")
        assert result == "4-methylcyclohexyl acetate", f"Got: {result}"

    def test_acetyloxy_hydroxy_benzene(self):
        """CC(=O)Oc1ccc(O)cc1: the hydroxy must appear, at its locant, in the ester name
        (the ester outranks the alcohol, so it is the principal group)."""
        result = name_compound("CC(=O)Oc1ccc(O)cc1")
        assert result == "4-hydroxyphenyl acetate", f"Got: {result}"


class TestEsterPlusMultipleSubstituents:
    """Compounds with ester prefix(es) AND multiple additional substituents."""

    def test_acetyloxy_chloro_methyl_benzene(self):
        """CC(=O)Oc1ccc(Cl)c(C)c1: chloro AND methyl must appear, at their locants."""
        result = name_compound("CC(=O)Oc1ccc(Cl)c(C)c1")
        assert result == "4-chloro-3-methylphenyl acetate", f"Got: {result}"


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
        # The ester is the principal group, so its ring carbon is C-1 (the free valence of
        # the 'cyclohexyl' substituent) and the methyl gets 4; one locant per group, none
        # duplicated
        assert result == "4-methylcyclohexyl acetate", f"Got: {result}"

    def test_alphabetical_ordering(self):
        """Prefixes should be in alphabetical order per IUPAC."""
        result = name_compound("CC(=O)Oc1ccc(Cl)c(C)c1")
        # The ester is no longer a prefix, so the detachable prefixes left to order are
        # chloro < methyl alphabetically
        assert result == "4-chloro-3-methylphenyl acetate", f"Got: {result}"
        assert "chloro" in result and "methyl" in result
        # Verify ordering: chloro before methyl
        c_pos = result.index("chloro")
        m_pos = result.index("methyl")
        assert c_pos < m_pos, (
            f"Alphabetical order violated: chloro@{c_pos}, methyl@{m_pos} in: {result}"
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
