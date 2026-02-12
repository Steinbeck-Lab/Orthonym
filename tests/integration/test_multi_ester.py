"""
Integration tests for multi-ester (dicarboxylic acid diester) naming.

Tests the detection and naming of compounds with two ester groups sharing
a dicarboxylic acid backbone, producing names in the format:
  "[multiplier]alkyl [parent]anedioate"

Examples:
  COC(=O)CC(=O)OC  -> "dimethyl propanedioate"  (dimethyl malonate)
  CCOC(=O)CCC(=O)OCC -> "diethyl butanedioate"  (diethyl succinate)
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# Dicarboxylic acid diester tests (NEW - should initially FAIL)
# ============================================================================

@pytest.mark.integration
def test_dimethyl_malonate():
    """Dimethyl malonate: dimethyl ester of malonic acid (propanedioic acid)."""
    result = name_compound("COC(=O)CC(=O)OC")
    assert result == "dimethyl propanedioate", f"Got: {result}"


@pytest.mark.integration
def test_diethyl_succinate():
    """Diethyl succinate: diethyl ester of succinic acid (butanedioic acid)."""
    result = name_compound("CCOC(=O)CCC(=O)OCC")
    assert result == "diethyl butanedioate", f"Got: {result}"


@pytest.mark.integration
def test_dimethyl_adipate():
    """Dimethyl adipate: dimethyl ester of adipic acid (hexanedioic acid)."""
    result = name_compound("COC(=O)CCCCC(=O)OC")
    assert result == "dimethyl hexanedioate", f"Got: {result}"


@pytest.mark.integration
def test_asymmetric_methyl_ethyl_malonate():
    """Asymmetric diester: different alkyl groups listed alphabetically."""
    result = name_compound("COC(=O)CC(=O)OCC")
    assert result == "ethyl methyl propanedioate", f"Got: {result}"


@pytest.mark.integration
def test_dimethyl_glutarate():
    """Dimethyl glutarate: dimethyl ester of glutaric acid (pentanedioic acid)."""
    result = name_compound("COC(=O)CCCC(=O)OC")
    assert result == "dimethyl pentanedioate", f"Got: {result}"


@pytest.mark.integration
def test_diethyl_oxalate():
    """Diethyl oxalate: diethyl ester of oxalic acid (2-carbon diacid, trivial name)."""
    result = name_compound("CCOC(=O)C(=O)OCC")
    assert result == "diethyl oxalate", f"Got: {result}"


# ============================================================================
# Regression tests (must STILL pass -- no regressions)
# ============================================================================

@pytest.mark.integration
def test_regression_ethyl_acetate():
    """Simple single ester: ethyl acetate must remain correct."""
    result = name_compound("CCOC(C)=O")
    assert result == "ethyl acetate", f"Got: {result}"


@pytest.mark.integration
def test_regression_methyl_propanoate():
    """Simple single ester: methyl propanoate must remain correct."""
    result = name_compound("COC(=O)CC")
    assert result == "methyl propanoate", f"Got: {result}"


@pytest.mark.integration
def test_regression_glycerol_diacetate():
    """Polyfunctional ester+alcohol: glycerol diacetate stays on polyfunctional path."""
    result = name_compound("CC(=O)OCC(O)COC(=O)C")
    assert "ol" in result, f"Expected 'ol' in result, got: {result}"
    assert "acetyloxy" in result, f"Expected 'acetyloxy' in result, got: {result}"


# ============================================================================
# Polyol polyester tests (NEW - should initially FAIL)
# ============================================================================

class TestPolyolPolyester:
    """Tests for fully-esterified polyol naming (triacetin, triglycerides)."""

    @pytest.mark.integration
    def test_triacetin(self):
        """Triacetin: glycerol triacetate -- 3 acetyloxy groups on propane backbone."""
        result = name_compound("CC(=O)OCC(COC(C)=O)OC(C)=O")
        assert result is not None, "Got None"
        # Should contain acyloxy prefix (acetyloxy or ethanoyloxy)
        has_acyloxy = "acetyloxy" in result or "ethanoyloxy" in result
        assert has_acyloxy, f"Expected 'acetyloxy' or 'ethanoyloxy' in result, got: {result}"
        # Should contain propane parent
        assert "propane" in result, f"Expected 'propane' in result, got: {result}"
        # Should have tri multiplier for 3 identical groups
        assert "tri" in result, f"Expected 'tri' in result, got: {result}"

    @pytest.mark.integration
    def test_glycerol_tripropanoate(self):
        """Glycerol tripropanoate: 3 propanoyloxy groups on propane backbone."""
        result = name_compound("CCC(=O)OCC(COC(=O)CC)OC(=O)CC")
        assert result is not None, "Got None"
        assert "propanoyloxy" in result, f"Expected 'propanoyloxy' in result, got: {result}"
        assert "propane" in result, f"Expected 'propane' in result, got: {result}"

    @pytest.mark.integration
    def test_glycerol_diacetate_mono_propanoate(self):
        """Mixed-acid triester: 2 acetyl + 1 propanoyl on glycerol backbone."""
        result = name_compound("CC(=O)OCC(COC(=O)CC)OC(C)=O")
        assert result is not None, "Got None"
        assert "acetyloxy" in result, f"Expected 'acetyloxy' in result, got: {result}"
        assert "propanoyloxy" in result, f"Expected 'propanoyloxy' in result, got: {result}"

    @pytest.mark.integration
    def test_regression_glycerol_diacetate_polyol(self):
        """Glycerol diacetate with free -OH: must still use polyfunctional path."""
        result = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert "ol" in result, f"Expected 'ol' in result, got: {result}"
        assert "acetyloxy" in result, f"Expected 'acetyloxy' in result, got: {result}"

    @pytest.mark.integration
    def test_regression_dimethyl_malonate_unaffected(self):
        """Dicarboxylic diester from 41-01: must still produce correct name."""
        result = name_compound("COC(=O)CC(=O)OC")
        assert result == "dimethyl propanedioate", f"Got: {result}"
