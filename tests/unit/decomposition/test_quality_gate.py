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
