"""
Integration tests for Phase 41-03: Fatty acid retained names and phospholipid FG detection.

Tests:
- Fatty acid trivial name lookups (acylate, acyloxy)
- Fatty acid ester naming (methyl palmitate, methyl stearate)
- Phospholipid FG detection (phosphate_monoester vs phosphonic_acid)
- Regression: existing simple esters unchanged
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.data.trivial_acids import get_acylate_name
from orthonym.rules.esters import get_acyloxy_prefix
from orthonym.perception.functional_groups import detect_functional_groups


# ===========================================================================
# Fatty acid trivial name lookups
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidLookups:
    """Fatty acid retained names in acylate and acyloxy tables."""

    def test_get_acylate_palmitic(self):
        """Palmitic -> palmitate."""
        assert get_acylate_name("palmitic") == "palmitate"

    def test_get_acylate_stearic(self):
        """Stearic -> stearate."""
        assert get_acylate_name("stearic") == "stearate"

    def test_get_acylate_lauric(self):
        """Lauric -> laurate."""
        assert get_acylate_name("lauric") == "laurate"

    def test_get_acylate_myristic(self):
        """Myristic -> myristate."""
        assert get_acylate_name("myristic") == "myristate"

    def test_get_acylate_oleic(self):
        """Oleic -> oleate."""
        assert get_acylate_name("oleic") == "oleate"

    def test_get_acylate_arachidic(self):
        """Arachidic -> arachidate."""
        assert get_acylate_name("arachidic") == "arachidate"

    def test_get_acyloxy_palmitic(self):
        """Palmitic -> palmitoyloxy."""
        assert get_acyloxy_prefix("palmitic") == "palmitoyloxy"

    def test_get_acyloxy_stearic(self):
        """Stearic -> stearoyloxy."""
        assert get_acyloxy_prefix("stearic") == "stearoyloxy"

    def test_get_acyloxy_lauric(self):
        """Lauric -> lauroyloxy."""
        assert get_acyloxy_prefix("lauric") == "lauroyloxy"

    def test_get_acyloxy_myristic(self):
        """Myristic -> myristoyloxy."""
        assert get_acyloxy_prefix("myristic") == "myristoyloxy"

    def test_get_acyloxy_oleic(self):
        """Oleic -> oleoyloxy."""
        assert get_acyloxy_prefix("oleic") == "oleoyloxy"

    def test_get_acyloxy_arachidic(self):
        """Arachidic -> arachidoyloxy."""
        assert get_acyloxy_prefix("arachidic") == "arachidoyloxy"


# ===========================================================================
# Fatty acid ester naming (end-to-end)
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidEsterNaming:
    """Methyl esters of fatty acids use trivial acid names."""

    def test_methyl_palmitate_name(self):
        """Methyl palmitate (16C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "palmitate" in result, f"Expected 'palmitate' in: {result!r}"

    def test_methyl_stearate_name(self):
        """Methyl stearate (18C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "stearate" in result, f"Expected 'stearate' in: {result!r}"

    def test_methyl_laurate_name(self):
        """Methyl laurate (12C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "laurate" in result, f"Expected 'laurate' in: {result!r}"

    def test_methyl_myristate_name(self):
        """Methyl myristate (14C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "myristate" in result, f"Expected 'myristate' in: {result!r}"

    def test_regression_methyl_acetate(self):
        """Methyl acetate is unchanged."""
        assert name_compound("COC(C)=O") == "methyl acetate"

    def test_regression_ethyl_propanoate(self):
        """Ethyl propanoate is unchanged."""
        assert name_compound("CCOC(=O)CC") == "ethyl propanoate"


# ===========================================================================
# Phospholipid functional group detection
# ===========================================================================


@pytest.mark.integration
class TestPhospholipidFGDetection:
    """Phosphate monoester detection suppresses phosphonic_acid for C-O-P bonds."""

    def test_glycerol_phosphate_detects_monoester(self):
        """Glycerol phosphate: C-O-P(=O)(OH)2 detects phosphate_monoester.
        Note: phosphonic_acid also matches (same P atom, different bond interpretation)
        and is intentionally kept for backward-compatible naming (phosphono prefix)."""
        mol = Chem.MolFromSmiles("OCC(O)COP(=O)(O)O")
        fgs = detect_functional_groups(mol)
        assert "phosphate_monoester" in fgs, (
            f"Expected phosphate_monoester in FGs: {list(fgs.keys())}"
        )

    def test_phospholipid_diacetate_detects_monoester(self):
        """Phospholipid diacetate: detects phosphate_monoester for C-O-P bond."""
        mol = Chem.MolFromSmiles("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        fgs = detect_functional_groups(mol)
        assert "phosphate_monoester" in fgs, (
            f"Expected phosphate_monoester in FGs: {list(fgs.keys())}"
        )

    def test_phosphonic_acid_direct_cp_bond(self):
        """True phosphonic acid: direct C-P bond is still detected as phosphonic_acid."""
        mol = Chem.MolFromSmiles("CP(=O)(O)O")  # methylphosphonic acid
        fgs = detect_functional_groups(mol)
        assert "phosphonic_acid" in fgs, (
            f"Expected phosphonic_acid for direct C-P bond: {list(fgs.keys())}"
        )
        assert "phosphate_monoester" not in fgs, (
            f"phosphate_monoester should NOT match direct C-P bond: {list(fgs.keys())}"
        )

    def test_phosphate_diester_detected(self):
        """Phosphate diester: C-O-P(=O)(OH)(O-C) is phosphate_diester."""
        mol = Chem.MolFromSmiles("COP(=O)(O)OC")
        fgs = detect_functional_groups(mol)
        assert "phosphate_diester" in fgs, (
            f"Expected phosphate_diester in FGs: {list(fgs.keys())}"
        )

    def test_phosphate_triester_detected(self):
        """Phosphate triester: all three O's have C attached."""
        mol = Chem.MolFromSmiles("COP(=O)(OC)OC")
        fgs = detect_functional_groups(mol)
        assert "phosphate_triester" in fgs, (
            f"Expected phosphate_triester in FGs: {list(fgs.keys())}"
        )
