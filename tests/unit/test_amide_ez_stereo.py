"""
Tests for E/Z stereodescriptors in chain amide naming.

Covers requirements:
- STER-01: E/Z descriptors preserved with locants in chain amides
- STER-02: Correct unsaturation markers (-en-, -dien-)
- STER-03: Multiple E/Z descriptors formatted correctly
- STER-04: No R/S vs E/Z collision (coexistence)

Also includes regression guards for ring-attached and saturated amides.
"""

import pytest
from orthonym import name_compound


class TestAmideEZStereo:
    """STER-01 + STER-02: E/Z descriptors preserved with locants in chain amides."""

    def test_pent_2_enamide_z(self):
        """CC/C=C\\C(=O)N -> (2Z)-pent-2-enamide"""
        result = name_compound("CC/C=C\\C(=O)N")
        assert result == "(2Z)-pent-2-enamide", f"Got: {result}"

    def test_but_2_enamide_e(self):
        """C/C=C/C(=O)N -> (2E)-but-2-enamide"""
        result = name_compound("C/C=C/C(=O)N")
        assert result == "(2E)-but-2-enamide", f"Got: {result}"

    def test_oleamide_9z(self):
        """CCCCCCCC/C=C\\CCCCCCCC(=O)N -> (9Z)-octadec-9-enamide"""
        result = name_compound("CCCCCCCC/C=C\\CCCCCCCC(=O)N")
        assert result == "(9Z)-octadec-9-enamide", f"Got: {result}"


class TestAmideEZWithNSubstitution:
    """STER-01: E/Z preserved through N-substitution prefix."""

    def test_n_methyl_oleamide(self):
        """N-methyl oleamide has N-prefix, Z descriptor, and enamide suffix."""
        result = name_compound("CCCCCCCC/C=C\\CCCCCCCC(=O)NC")
        assert "N-methyl" in result, f"Missing N-methyl in: {result}"
        assert "(9Z)" in result, f"Missing (9Z) in: {result}"
        assert "enamide" in result, f"Missing enamide in: {result}"
        # Verify correct ordering: N-prefix before stereo block
        n_pos = result.index("N-methyl")
        z_pos = result.index("(9Z)")
        assert n_pos < z_pos, f"N-prefix should come before stereo: {result}"


class TestAmideMultipleEZ:
    """STER-03: Multiple E/Z descriptors formatted correctly."""

    def test_linoleamide_9z_12z(self):
        """Linoleamide has two Z double bonds: (9Z,12Z)-octadeca-9,12-dienamide"""
        result = name_compound("CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)N")
        assert result == "(9Z,12Z)-octadeca-9,12-dienamide", f"Got: {result}"


class TestAmideEZNoCollisionWithRS:
    """STER-04: R/S and E/Z descriptors can coexist in amide names."""

    def test_stereo_amide_with_rs_and_ez(self):
        """Amide with both a stereocenter and E/Z bond should contain both descriptors."""
        # Use a simple molecule with both chiral center and E/Z
        smiles = "[C@@H](O)(C/C=C/C(=O)N)CC"
        result = name_compound(smiles)
        # Should contain at least one E/Z label and one R/S label
        has_ez = "E" in result and ("enamide" in result or "en" in result)
        has_rs = "S" in result or "R" in result
        # The combined stereo block should contain both, e.g. (2E,5S)-
        assert has_ez, f"Missing E/Z descriptor in: {result}"
        assert has_rs, f"Missing R/S descriptor in: {result}"


class TestAmideRegressionGuards:
    """Regression guards: ring-attached and saturated amides must NOT change."""

    def test_cyclohexanecarboxamide_unchanged(self):
        """Ring-attached amide: C1CCCCC1C(=O)N -> cyclohexanecarboxamide"""
        result = name_compound("C1CCCCC1C(=O)N")
        assert result == "cyclohexanecarboxamide", f"Got: {result}"

    def test_pentanamide_unchanged(self):
        """Saturated chain amide: CCCCC(=O)N -> pentanamide"""
        result = name_compound("CCCCC(=O)N")
        assert result == "pentanamide", f"Got: {result}"

    def test_acetamide_retained(self):
        """Retained name: CC(=O)N -> acetamide"""
        result = name_compound("CC(=O)N")
        assert result == "acetamide", f"Got: {result}"

    def test_formamide_retained(self):
        """Retained name: C(=O)N -> formamide"""
        result = name_compound("C(=O)N")
        assert result == "formamide", f"Got: {result}"

    def test_n_methylacetamide_unchanged(self):
        """N-substituted retained: CNC(C)=O -> N-methylacetamide"""
        result = name_compound("CNC(C)=O")
        assert result == "N-methylacetamide", f"Got: {result}"

    def test_nn_dimethylformamide_unchanged(self):
        """N,N-disubstituted retained: CN(C)C=O -> N,N-dimethylformamide"""
        result = name_compound("CN(C)C=O")
        assert result == "N,N-dimethylformamide", f"Got: {result}"

    def test_nn_dipropylformamide_unchanged(self):
        """Tertiary formamide: CCCN(C=O)CCC -> N,N-dipropylformamide"""
        result = name_compound("CCCN(C=O)CCC")
        assert result == "N,N-dipropylformamide", f"Got: {result}"
