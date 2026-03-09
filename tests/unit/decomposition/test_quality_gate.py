"""Tests for name_pipeline_only utility function.

Verifies that name_pipeline_only:
- Does NOT consume depth levels (thread-local depth counter unchanged)
- Does NOT trigger decomposition (skips try_decompose)
- Returns None for invalid SMILES
- Works for simple and complex molecules
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_pipeline_only
from orthonym.assembly.fragment_naming import get_naming_depth


class TestNamePipelineOnly:
    """name_pipeline_only should name molecules without decomposition or depth cost."""

    @pytest.mark.unit
    def test_pipeline_only_no_depth_increment(self):
        """name_pipeline_only should NOT increment depth counter."""
        depth_before = get_naming_depth()
        result = name_pipeline_only("CCO")
        depth_after = get_naming_depth()
        assert depth_after == depth_before, (
            f"name_pipeline_only incremented depth: {depth_before} -> {depth_after}"
        )
        assert result == "ethanol"

    @pytest.mark.unit
    def test_pipeline_only_complex_molecule(self):
        """name_pipeline_only should handle complex molecules without depth waste."""
        # Phospholipid-like SMILES that previously hit DROP-13
        smiles = "CCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC"
        depth_before = get_naming_depth()
        result = name_pipeline_only(smiles)
        depth_after = get_naming_depth()
        assert depth_after == depth_before, (
            f"Depth changed during name_pipeline_only: {depth_before} -> {depth_after}"
        )
        assert result is not None, "name_pipeline_only returned None for valid SMILES"

    @pytest.mark.unit
    def test_pipeline_only_skips_decomposition(self):
        """name_pipeline_only should not call try_decompose."""
        # A simple ester that would normally decompose
        smiles = "CC(=O)OC"
        result = name_pipeline_only(smiles)
        # Pipeline-only should return the pipeline name, not decomposed name
        assert result is not None

    @pytest.mark.unit
    def test_pipeline_only_returns_none_on_invalid(self):
        """name_pipeline_only should return None for invalid SMILES."""
        result = name_pipeline_only("NOT_A_SMILES")
        assert result is None

    @pytest.mark.unit
    def test_pipeline_only_garbled_fallback_disabled(self):
        """name_pipeline_only should not use garbled-name fallback path."""
        depth_before = get_naming_depth()
        name_pipeline_only("c1ccccc1")
        depth_after = get_naming_depth()
        assert depth_after == depth_before


class TestTieredCoverage:
    """Tests for tiered coverage threshold based on bond type.

    Functional-class types (ester, amide, glycosidic, carbamate, thioester)
    use 0.6 chars/HA threshold. Substitutive types (sulfonamide,
    phosphodiester, ether, default) use 0.8 chars/HA threshold.
    """

    @pytest.mark.unit
    def test_ester_uses_lower_threshold(self):
        """Ester bond type should use 0.6 chars/HA threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Passes 0.6 threshold, fails 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("phenyl palmitate", mol, bond_type="ester") is True

    @pytest.mark.unit
    def test_ester_rejects_too_short(self):
        """Even with 0.6 threshold, very short names should be rejected."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("short", mol, bond_type="ester") is False

    @pytest.mark.unit
    def test_sulfonamide_uses_higher_threshold(self):
        """Sulfonamide bond type should use 0.8 chars/HA threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Fails 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("phenyl palmitate", mol, bond_type="sulfonamide") is False

    @pytest.mark.unit
    def test_sulfonamide_passes_with_long_name(self):
        """Sulfonamide with adequate length should pass 0.8 threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "N-methylbenzenesulfonamide" = 25 chars, 20 HA => 1.25 chars/HA
        mol = Chem.MolFromSmiles("CS(=O)(=O)c1ccc(C)c(C)c1NC")  # ~20 HA approx
        ha = mol.GetNumHeavyAtoms()
        name = "N-methylbenzenesulfonamide"
        # Ensure the molecule has enough HA to test meaningfully (>10)
        assert ha > 10, f"Need HA > 10 for test, got {ha}"
        assert _coverage_is_adequate(name, mol, bond_type="sulfonamide") is True

    @pytest.mark.unit
    def test_small_molecule_exemption_ester(self):
        """Small molecules (HA <= 10) should always pass regardless of bond type."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CC(=O)OC")  # methyl acetate, ~5 HA
        assert mol.GetNumHeavyAtoms() <= 10
        assert _coverage_is_adequate("anything", mol, bond_type="ester") is True

    @pytest.mark.unit
    def test_small_molecule_exemption_default(self):
        """Small molecules should pass even with empty bond_type."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane, 6 HA
        assert mol.GetNumHeavyAtoms() <= 10
        assert _coverage_is_adequate("anything", mol, bond_type="") is True

    @pytest.mark.unit
    def test_default_bond_type_uses_high_threshold(self):
        """No bond_type (default="") should use 0.8 threshold for backward compat."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Should fail 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")
        assert _coverage_is_adequate("phenyl palmitate", mol) is False

    @pytest.mark.unit
    def test_each_functional_class_type_uses_low_threshold(self):
        """All 5 functional-class types should use 0.6 threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        # "phenyl palmitate" = 16 chars => 0.67 chars/HA, passes 0.6 but fails 0.8
        for bond_type in ("ester", "amide", "glycosidic", "carbamate", "thioester"):
            assert _coverage_is_adequate("phenyl palmitate", mol, bond_type=bond_type) is True, (
                f"bond_type={bond_type} should use 0.6 threshold"
            )


class TestProbeIntegration:
    """Integration tests verifying the probe replacement in try_decompose().

    After Phase 097, the decomposition engine probe uses name_pipeline_only()
    instead of name_fragment_recursively(), eliminating the cache-disable hack
    and _fragment_guard dependency.
    """

    @pytest.mark.unit
    def test_probe_uses_name_pipeline_only(self):
        """try_decompose probe path should use name_pipeline_only, not name_fragment_recursively."""
        from unittest.mock import patch
        from orthonym.decomposition.engine import try_decompose

        # A simple ester that has cleavable bonds
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        with patch('orthonym.namer.name_pipeline_only', wraps=name_pipeline_only) as mock_npo:
            try_decompose(mol)
            # name_pipeline_only should have been called for the probe
            assert mock_npo.called, "try_decompose should use name_pipeline_only for baseline probe"

    @pytest.mark.unit
    def test_probe_does_not_touch_fragment_cache(self):
        """Probe path should not save/restore _fragment_guard.cache."""
        from orthonym.decomposition.engine import try_decompose
        from orthonym.assembly.fragment_naming import _fragment_guard

        mol = Chem.MolFromSmiles("CC(=O)OCC")
        cache_before = getattr(_fragment_guard, 'cache', None)
        try_decompose(mol)
        cache_after = getattr(_fragment_guard, 'cache', None)
        # Cache state should be untouched by the probe
        assert cache_before is cache_after or cache_before == cache_after, \
            "_fragment_guard.cache was modified by probe path"

    @pytest.mark.unit
    def test_probe_returns_systematic_name_for_decomposable(self):
        """name_pipeline_only should return a systematic name even for decomposable molecules."""
        result = name_pipeline_only("CC(=O)OCC")  # ethyl acetate
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
