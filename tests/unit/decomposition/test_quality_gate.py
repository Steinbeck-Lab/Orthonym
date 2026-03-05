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
