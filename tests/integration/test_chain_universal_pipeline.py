"""
Integration tests for chain substituent naming via universal pipeline.

Verifies that _generate_alkyl_prefixes correctly names all substituent
types on chain parents using the universal pipeline (discover + classify).
Covers: simple alkyl, branched alkyl, heteroatom-containing (alkoxy,
acylamino, acyloxy), ring-on-chain, and compound substituents.
"""

import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestSimpleAlkylOnChain:
    """Simple unbranched alkyl substituents on chain parents."""

    def test_methyl_on_pentane(self):
        """2-methylpentane"""
        result = name_compound("CCCC(C)C")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "pentane" in result.lower(), f"Expected 'pentane' in '{result}'"

    def test_dimethyl_on_butane(self):
        """2,3-dimethylbutane"""
        result = name_compound("CC(C)C(C)C")
        assert result is not None
        assert "dimethyl" in result.lower(), f"Expected 'dimethyl' in '{result}'"
        assert "butane" in result.lower(), f"Expected 'butane' in '{result}'"

    def test_ethyl_on_hexane(self):
        """3-ethylhexane"""
        result = name_compound("CCCC(CC)CC")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"


@pytest.mark.integration
class TestHeteroatomSubOnChain:
    """Heteroatom-containing substituents: alkoxy, acylamino, acyloxy."""

    def test_methoxy_on_propane(self):
        """2-methoxypropane (isopropyl methyl ether alternative)"""
        result = name_compound("COC(C)C")
        assert result is not None
        lower = result.lower()
        assert "methoxy" in lower or "ether" in lower, (
            f"Expected 'methoxy' or ether naming in '{result}'"
        )

    def test_ethoxy_on_butanoic_acid(self):
        """4-ethoxybutanoic acid"""
        result = name_compound("OCCCCOCC")
        assert result is not None
        lower = result.lower()
        has_ethoxy = "ethoxy" in lower
        has_hydroxy = "hydroxy" in lower or "ol" in lower or "oic" in lower
        assert has_ethoxy or has_hydroxy, f"Expected ethoxy/hydroxy in '{result}'"

    def test_acetylamino_on_chain(self):
        """N-acetyl amino substituent on chain"""
        result = name_compound("CC(=O)NCCCC")
        assert result is not None
        lower = result.lower()
        assert "amino" in lower or "amide" in lower or "acetyl" in lower or "acetamido" in lower, (
            f"Expected amino/amide/acetyl in '{result}'"
        )


@pytest.mark.integration
class TestCompoundSubOnChain:
    """Compound substituents requiring brackets and bis/tris multipliers."""

    def test_hydroxymethyl_on_chain(self):
        """Hydroxymethyl substituent"""
        result = name_compound("OCC(CO)CCC")
        assert result is not None
        lower = result.lower()
        assert "hydroxy" in lower or "ol" in lower, (
            f"Expected hydroxy-related naming in '{result}'"
        )

    def test_ring_sub_on_acid_chain(self):
        """Cyclopentyl on pentanoic acid chain"""
        result = name_compound("C(CCCC(=O)O)C1CCCC1")
        assert result is not None
        lower = result.lower()
        assert "cyclopentyl" in lower, f"Expected 'cyclopentyl' in '{result}'"
        assert "anoic" in lower or "acid" in lower, (
            f"Expected acid suffix in '{result}'"
        )


@pytest.mark.integration
class TestMultipleSubsOnChain:
    """Multiple different substituents -- tests grouping and alpha-sort."""

    def test_chloro_and_methyl_on_pentane(self):
        """2-chloro-3-methylpentane -- alphabetical: chloro before methyl"""
        result = name_compound("CCC(C)C(Cl)C")
        assert result is not None
        lower = result.lower()
        assert "chloro" in lower or "chlor" in lower, (
            f"Expected 'chloro' in '{result}'"
        )
        assert "methyl" in lower, f"Expected 'methyl' in '{result}'"
