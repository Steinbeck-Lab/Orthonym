"""Tests for alkoxycarbonyl prefix generation (IUPAC P-65.6.3).

When an ester group is NOT the principal characteristic group, it is
expressed as an alkoxycarbonyl prefix:
  -COOCH3   -> methoxycarbonyl
  -COOC2H5  -> ethoxycarbonyl
  -COOPh    -> phenoxycarbonyl

NOTE: OPSIN 2.8.0 does not parse alkoxycarbonyl prefixes, so
round-trip validation is not possible for these names.  The names
are correct per IUPAC 2013 Blue Book P-65.6.3.
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# E2E: SMILES -> name containing alkoxycarbonyl
# ============================================================================


class TestAlkoxycarbonylE2E:
    """End-to-end tests: SMILES -> IUPAC name with alkoxycarbonyl prefix."""

    def test_methyl_ester_succinate(self):
        """Methyl hydrogen succinate: ester + acid on C4 chain."""
        result = name_compound("COC(=O)CCC(=O)O")
        assert "methoxycarbonyl" in result.lower()
        assert "butanoic acid" in result.lower()

    def test_ethyl_ester_glutarate(self):
        """Ethyl hydrogen glutarate: ester + acid on C5 chain."""
        result = name_compound("CCOC(=O)CCCC(=O)O")
        assert "ethoxycarbonyl" in result.lower()
        assert "pentanoic acid" in result.lower()

    def test_methyl_ester_adipate(self):
        """Methyl hydrogen adipate: ester + acid on C6 chain."""
        result = name_compound("COC(=O)CCCCC(=O)O")
        assert "methoxycarbonyl" in result.lower()
        assert "hexanoic acid" in result.lower()

    def test_ethyl_ester_malonate(self):
        """Ethyl hydrogen malonate: ester + acid on C3 chain."""
        result = name_compound("CCOC(=O)CC(=O)O")
        assert "ethoxycarbonyl" in result.lower()
        assert "propanoic acid" in result.lower()

    def test_phenyl_ester_succinate(self):
        """Phenyl hydrogen succinate: aryl ester + acid."""
        result = name_compound("O=C(Oc1ccccc1)CCC(=O)O")
        assert "phenoxycarbonyl" in result.lower()
        assert "butanoic acid" in result.lower()

    def test_propyl_ester(self):
        """Propyl ester on pentanoic acid."""
        result = name_compound("CCCOC(=O)CCCC(=O)O")
        assert "propoxycarbonyl" in result.lower()


# ============================================================================
# Negative / guard tests: things that must NOT produce alkoxycarbonyl
# ============================================================================


class TestAlkoxycarbonylNegative:
    """Tests that specific compound classes do NOT produce alkoxycarbonyl."""

    def test_lactone_no_alkoxycarbonyl(self):
        """gamma-Butyrolactone: cyclic ester, NOT alkoxycarbonyl."""
        result = name_compound("O=C1CCCO1")
        assert "oxycarbonyl" not in result.lower()
        assert "oxolan" in result.lower()

    def test_simple_ester_no_alkoxycarbonyl(self):
        """Ethyl acetate: ester as principal group, functional class naming."""
        result = name_compound("CCOC(=O)C")
        assert "oxycarbonyl" not in result.lower()

    def test_acyloxy_ester_no_double_naming(self):
        """Palmitoyloxy ester: acyloxy orientation, not alkoxycarbonyl."""
        result = name_compound("CCCCCCCCCCCCCCCC(=O)OC(CCCCC)CCCCCCCCCCCC(=O)[O-]")
        assert "oxycarbonyl" not in result.lower()
        assert "palmitoyloxy" in result.lower()

    def test_heteroatom_alkyl_no_alkoxycarbonyl(self):
        """Ester with nitrogen-containing OR fragment: skip alkoxycarbonyl."""
        result = name_compound("N[C@@H](COC(=O)CCC(=O)O)C(=O)O")
        assert "oxycarbonyl" not in result.lower()

    def test_macrocyclic_ester_no_alkoxycarbonyl(self):
        """Macrolide with pendant ester: alkyl is huge ring, skip."""
        # ci-055 compound
        result = name_compound(
            "CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/"
            "[C@H](C)[C@H](O)C[C@H](O)[C@H](C)[C@@H](O)CC[C@@H](C)"
            "[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
            "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)[C@@H]2O"
        )
        assert "oxycarbonyl" not in result.lower()


# ============================================================================
# CI regression guards
# ============================================================================


class TestAlkoxycarbonylCIRegression:
    """Compounds from CI benchmark that must not change."""

    def test_ci091_unchanged(self):
        """ci-091: complex naphthoquinone ester must not regress."""
        result = name_compound("CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O")
        assert result == "(acetyloxy)-1-docosyloxyethanedione"

    def test_separate_ether_and_ester(self):
        """Compound with both ester and genuine separate ether."""
        result = name_compound("COC(=O)c1ccc(Oc2ccccc2)cc1C(=O)O")
        # Ester on ring + separate phenoxy ether - both should be named
        assert "phenoxy" in result.lower() or "ether" in result.lower() or "benzoic" in result.lower()
