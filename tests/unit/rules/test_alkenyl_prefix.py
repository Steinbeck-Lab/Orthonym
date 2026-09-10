"""Tests for alkenyl/alkynyl substituent prefix naming (IUPAC.

When a substituent contains C=C or C#C unsaturation, it must be named
using systematic alkenyl/alkynyl forms, not saturated alkyl names:
  -CH=CH2 -> ethenyl (not ethyl or vinyl)
  -CH2-CH=CH2 -> prop-2-en-1-yl (not propyl or allyl)
  -C#CH -> ethynyl (not ethyl)
  -C(=CH2)(CH3) -> prop-1-en-2-yl (not isopropyl)

References:
    IUPAC 2013 Blue Book (naming of substituent groups)
    IUPAC 2013 Blue Book (numbering priority: free valence > unsaturation)
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# E2E: SMILES -> name containing alkenyl/alkynyl prefix
# ============================================================================


class TestAlkenylE2E:
    """End-to-end tests: SMILES -> IUPAC name with alkenyl/alkynyl prefix."""

    def test_ethenyl_on_cyclohexane(self):
        """Vinyl group on cyclohexane -> ethenylcyclohexane."""
        result = name_compound("C=CC1CCCCC1")
        assert "ethenyl" in result.lower()
        assert "cyclohexane" in result.lower()

    def test_ethynyl_on_benzene(self):
        """Ethynyl group on benzene."""
        result = name_compound("C#Cc1ccccc1")
        assert "ethynyl" in result.lower()
        assert "benzene" in result.lower()

    def test_prop_2_en_1_yl_on_benzene(self):
        """Allyl on benzene -> prop-2-en-1-ylbenzene."""
        result = name_compound("C=CCc1ccccc1")
        assert "prop-2-en-1-yl" in result.lower()

    def test_prop_1_en_2_yl_on_benzene(self):
        """Isopropenyl on benzene -> prop-1-en-2-ylbenzene."""
        result = name_compound("C=C(C)c1ccccc1")
        assert "prop-1-en-2-yl" in result.lower()

    def test_prop_1_en_1_yl_on_benzene(self):
        """1-Propenyl on benzene -> prop-1-en-1-ylbenzene."""
        result = name_compound("CC=Cc1ccccc1")
        assert "prop-1-en-1-yl" in result.lower()


# ============================================================================
# Negative / guard tests
# ============================================================================


class TestAlkenylNegative:
    """Tests that saturated chains still produce correct alkyl names."""

    def test_ethyl_still_named_ethyl(self):
        """Ethylcyclohexane: no double bond, must stay 'ethyl'."""
        result = name_compound("CCC1CCCCC1")
        assert "ethyl" in result.lower()
        assert "ethenyl" not in result.lower()

    def test_methyl_still_named_methyl(self):
        """Methylcyclohexane: no double bond, must stay 'methyl'."""
        result = name_compound("CC1CCCCC1")
        assert "methyl" in result.lower()
        assert "ethenyl" not in result.lower()

    def test_ring_unsaturation_not_alkenyl(self):
        """Cyclohexene: ring C=C, not a substituent -> not alkenyl."""
        result = name_compound("C1=CCCCC1")
        assert "ethenyl" not in result.lower()
        assert "cyclohex" in result.lower()

    def test_isopropyl_still_retained(self):
        """Saturated isopropyl is named as the located alkyl 'propan-2-yl' (F-T9/DD6
        ), NOT as an alkenyl — the point of this test is that no spurious 'en'
        unsaturation appears."""
        result = name_compound("CC(C)C1CCCCC1")
        assert "propan-2-yl" in result.lower()
        assert "en" not in result.lower()


# ============================================================================
# CI regression guards
# ============================================================================


class TestAlkenylCIRegression:
    """Compounds from CI/stereo benchmarks that must match updated values."""

    def test_ci033_ethenyl(self):
        """ci-033: C=C substituent is ethenyl, not ethyl."""
        result = name_compound(
            "C=C[C@@H]1C(=C)CC[C@H]2[C@H]1C[C@H]1OC(=O)"
            "[C@@]3(C)[C@H](O)CC[C@@]2(C)[C@@]13O"
        )
        assert "ethenyl" in result.lower()
        assert "-ethyl-" not in result.lower()

    def test_heptenyl_on_ring(self):
        """Hept-1-en-1-yl substituent on cyclohexenone ring."""
        result = name_compound("CCCCCC=CC1=C(CO)C(=O)C[C@H](O)[C@@H]1O")
        assert "hept-1-en-1-yl" in result.lower()
        assert "-heptyl-" not in result.lower()

    def test_prop_1_en_2_yl_on_tricyclic(self):
        """Prop-1-en-2-yl on tricyclic terpene, not isopropyl."""
        result = name_compound(
            "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)"
            "C/C=C(\\C)CC/C=C(\\C)CC[C@H]12"
        )
        assert "prop-1-en-2-yl" in result.lower()
        assert "15-isopropyl" not in result.lower()


class TestBranchedAlkenylPrefix:
    """ Fix 4 (1)): a BRANCHED acyclic alkenyl substituent
    must cite the free-valence locant ('-1-yl'), not the locant-dropped '-enyl'
    the recursive parent_to_prefix fallback produced. The principal chain runs
    through the free valence (longest, then max unsaturation)."""

    @pytest.mark.parametrize("smiles,expected", [
        # -CH=C(CH3)2 on benzene
        ("CC(C)=Cc1ccccc1", "(2-methylprop-1-en-1-yl)benzene"),
        # prenyl -CH2-CH=C(CH3)2 on benzene (was mis-named -> unknown)
        ("CC(C)=CCc1ccccc1", "(3-methylbut-2-en-1-yl)benzene"),
    ])
    def test_branched_alkenyl_on_benzene(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_target_hepta_diene(self):
        # The PIN-backlog target: 4-(2-methylprop-1-en-1-yl) branch.
        assert (name_compound("C=CCC(C=C(C)C)C(C)=CC")
                == "5-methyl-4-(2-methylprop-1-en-1-yl)hepta-1,5-diene")

    @pytest.mark.parametrize("smiles,frag", [
        # Regression: UNBRANCHED alkenyl substituents (handled by the linear
        # namer) must stay byte-identical -- the branched namer must NOT fire.
        ("C=CCc1ccccc1", "prop-2-en-1-yl"),
        ("CC=Cc1ccccc1", "prop-1-en-1-yl"),
        ("C=C(C)c1ccccc1", "prop-1-en-2-yl"),
        ("C=Cc1ccccc1", "ethenyl"),
    ])
    def test_unbranched_alkenyl_unchanged(self, smiles, frag):
        assert frag in name_compound(smiles)
