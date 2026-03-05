"""Integration tests for seniority-based fragment selection in decomposition.

Tests that P-44.1.1 seniority ranking correctly influences which fragment
becomes parent in ester, amide, and ether decomposition, and that the
quality gate detects dropped fragments.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.decomposition.engine import _name_quality_is_acceptable


# ---------------------------------------------------------------------------
# Amide seniority tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAmideSeniority:
    """Test seniority-based fragment selection for amide bonds."""

    def test_amide_with_senior_amine_includes_both_fragments(self):
        """COc1ccc(C(=O)N2CCCC2=O)cc1: name must reference both ring systems.

        The molecule has a methoxybenzene (acid side) and a pyrrolidinone
        (amine side). A correct name should reference both, not just
        produce 'benzamide' which drops the pyrrolidinone.
        """
        name = name_compound("COc1ccc(C(=O)N2CCCC2=O)cc1")
        assert name is not None and name != "unknown"

        name_lower = name.lower()
        # Name must reference BOTH structural fragments
        # Acid side: methoxyphenyl / methoxybenz / benzoyl / benzamid
        has_acid_ref = any(
            tok in name_lower
            for tok in ["methoxy", "benz", "phenyl", "benzoyl"]
        )
        # Amine side: pyrrolidin / pyrrolidone / oxopyrrolidin
        has_amine_ref = any(
            tok in name_lower
            for tok in ["pyrrolidin", "pyrrolidone", "oxopyrrolidin", "oxoprolin"]
        )
        assert has_acid_ref, (
            f"Name '{name}' missing acid-side reference "
            f"(methoxy/benz/phenyl/benzoyl)"
        )
        assert has_amine_ref, (
            f"Name '{name}' missing amine-side reference "
            f"(pyrrolidin/pyrrolidone)"
        )

    def test_simple_amide_unchanged(self):
        """N-methylacetamide: acid side is clearly more senior, no change."""
        name = name_compound("CC(=O)NC")
        assert name is not None
        # Should remain "N-methylacetamide" or similar
        assert "acetamide" in name.lower() or "ethanamide" in name.lower(), (
            f"Expected acetamide/ethanamide in '{name}'"
        )

    def test_amide_acid_more_senior_preserves_pattern(self):
        """N-methylbenzamide: acid side has ring + COOH, clearly more senior."""
        name = name_compound("O=C(NC)c1ccccc1")
        assert name is not None
        assert "benzamide" in name.lower(), (
            f"Expected 'benzamide' in '{name}'"
        )


# ---------------------------------------------------------------------------
# Ether seniority tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEtherSeniority:
    """Test seniority-based fragment selection for ether bonds."""

    def test_ether_seniority_over_size(self):
        """Ether where smaller side has COOH: COOH side should be parent.

        O=C(O)COc1ccccc1 (phenoxyacetic acid):
        - Small side: -CH2-COOH (carboxylic acid, higher seniority)
        - Large side: phenyl (no FG besides ether)
        The acid side should be parent, producing '...ethanoic acid'.
        """
        name = name_compound("O=C(O)COc1ccccc1")
        assert name is not None
        name_lower = name.lower()
        # The acid side must be the parent (produces "...oic acid" or "...ethanoic acid")
        assert "acid" in name_lower or "oic" in name_lower, (
            f"Expected acid suffix in '{name}' (COOH side should be parent)"
        )
        # The phenyl should appear as a substituent prefix
        assert "phenoxy" in name_lower or "phenyl" in name_lower, (
            f"Expected phenoxy/phenyl reference in '{name}'"
        )

    def test_simple_ether_unchanged(self):
        """Diethyl ether: symmetric-ish, existing behavior preserved."""
        name = name_compound("CCOCC")
        assert name is not None
        assert "eth" in name.lower(), f"Expected 'eth' in '{name}'"


# ---------------------------------------------------------------------------
# Canary stability tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDecompositionCanaryStability:
    """Ensure decomposition changes don't regress known-good names."""

    CANARY_COMPOUNDS = [
        ("CCOC(=O)C", "ethyl acetate"),
        ("CC(=O)NC", "N-methylacetamide"),
        ("CC(=O)OCC", "ethyl acetate"),
        ("O=C(O)COc1ccccc1", "2-phenoxyethanoic acid"),
        ("c1ccccc1", "benzene"),
        ("CCO", "ethanol"),
        ("CC(=O)O", "acetic acid"),
        ("CCCC", "butane"),
        ("CC(C)C", "2-methylpropane"),
        ("c1ccncc1", "pyridine"),
        ("C1CCCC1", "cyclopentane"),
        ("OCC(O)CO", "glycerol"),
        ("CCOCC", "ethoxyethane"),
    ]

    @pytest.mark.parametrize("smiles,expected", CANARY_COMPOUNDS)
    def test_canary_stability(self, smiles, expected):
        """Canary compound must produce exact expected name."""
        result = name_compound(smiles)
        assert result == expected, (
            f"Canary regression for {smiles}: "
            f"expected '{expected}', got '{result}'"
        )


# ---------------------------------------------------------------------------
# Quality gate fragment coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQualityGateFragmentCoverage:
    """Test that quality gate detects names that drop major fragments."""

    def test_rejects_name_dropping_ring_fragment(self):
        """Quality gate should reject a name that drops a ring system.

        COc1ccc(C(=O)N2CCCC2=O)cc1 has 16 heavy atoms, two ring systems
        separated by a cleavable amide bond. A name like '4-methoxybenzamide'
        (18 chars / 16 HA = 1.125) passes char/atom ratio but drops pyrrolidinone.

        The enhanced quality gate should detect that the molecule has two ring
        systems separated by cleavable bonds and that the name only covers one.
        """
        mol = Chem.MolFromSmiles("COc1ccc(C(=O)N2CCCC2=O)cc1")
        # This name drops the pyrrolidinone ring
        result = _name_quality_is_acceptable("4-methoxybenzamide", mol)
        assert result is False, (
            "Quality gate should reject '4-methoxybenzamide' which drops "
            "the pyrrolidinone ring system"
        )

    def test_accepts_complete_name(self):
        """Quality gate should accept a name that covers all fragments."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        # N-methylacetamide covers both sides of the amide
        result = _name_quality_is_acceptable("N-methylacetamide", mol)
        assert result is True, (
            "Quality gate should accept 'N-methylacetamide' which covers "
            "both fragments"
        )

    def test_accepts_simple_retained_name(self):
        """Quality gate should accept a correct retained name."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = _name_quality_is_acceptable("benzene", mol)
        assert result is True

    def test_rejects_when_two_ring_systems_but_only_one_named(self):
        """Molecule with two ring systems separated by cleavable bond.

        If name only references one ring system and ratio is low,
        quality gate should reject.
        """
        # Construct a molecule with two ring systems and a cleavable bond
        # phenyl benzoate: PhOC(=O)Ph - ester between two ring systems
        mol = Chem.MolFromSmiles("O=C(Oc1ccccc1)c1ccccc1")
        if mol is not None:
            # A name that only covers one ring system
            result = _name_quality_is_acceptable("benzoate", mol)
            # "benzoate" is 8 chars / 14 HA = 0.57 ratio, should be rejected
            assert result is False, (
                "Quality gate should reject 'benzoate' for PhOC(=O)Ph "
                "(drops one ring system)"
            )
