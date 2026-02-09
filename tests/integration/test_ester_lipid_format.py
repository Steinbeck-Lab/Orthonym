"""
Integration tests for Phase 31: Ester & Lipid Format fixes.
Tests EL-01 (ring ester prefix joining) and EL-02 (polyfunctional ester demotion).
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.integration
class TestRingEsterPrefixJoining:
    """EL-01: Ring ester prefixes use hyphen joining with locants."""

    def test_single_acyloxy_benzene(self):
        """Single acyloxy on benzene -- no locant needed."""
        result = name_compound("CC(=O)Oc1ccccc1")
        assert result is not None
        assert "acetyloxy" in result
        assert "benzene" in result

    def test_single_acyloxy_cyclohexane(self):
        """Single acyloxy on cyclohexane."""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert result is not None
        assert "acetyloxy" in result
        assert "cyclohexane" in result

    def test_different_acyloxy_benzene(self):
        """Two different acyloxy prefixes on benzene have locants and hyphens."""
        result = name_compound("CC(=O)Oc1ccc(OC(=O)CC)cc1")
        assert result is not None
        assert "acetyloxy" in result
        assert "propanoyloxy" in result
        # Should NOT have direct concatenation without hyphens
        assert "acetyloxypropanoyloxy" not in result

    def test_different_acyloxy_benzene_format(self):
        """Two different acyloxy prefixes produce locant-(prefix) format."""
        result = name_compound("CC(=O)Oc1ccc(OC(=O)CC)cc1")
        # Should be like: 1-(acetyloxy)-4-(propanoyloxy)benzene
        assert "(" in result  # Parenthesized prefixes

    def test_same_acyloxy_cyclohexane(self):
        """Same acyloxy prefix twice uses multiplier."""
        result = name_compound("CC(=O)OC1CCCCC1OC(C)=O")
        assert result is not None
        assert "acetyloxy" in result
        assert "cyclohexane" in result


@pytest.mark.integration
class TestPolyfunctionalEsterDemotion:
    """EL-02: Esters demoted to acyloxy prefixes in polyfunctional compounds."""

    def test_glycerol_diacetate_no_dioate(self):
        """Glycerol diacetate: acyloxy prefix + -ol suffix, NOT -dioate."""
        result = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert result is not None
        assert "oate" not in result, f"Got '-oate' suffix in polyfunctional ester: {result}"
        assert "ol" in result, f"Expected '-ol' suffix for alcohol principal group: {result}"

    def test_glycerol_diacetate_has_acyloxy(self):
        """Glycerol diacetate should contain acyloxy prefix."""
        result = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert "acetyloxy" in result or "ethanoyloxy" in result

    def test_diglyceride_mixed_acids(self):
        """Mixed-acid diglyceride: two different acyloxy + -ol suffix."""
        result = name_compound(
            "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC"
        )
        assert result is not None
        assert "oate" not in result, f"Got '-oate' suffix: {result}"
        assert "ol" in result, f"Expected '-ol' suffix: {result}"

    def test_diglyceride_has_acyloxy_prefixes(self):
        """Mixed-acid diglyceride should have acyloxy prefixes for both acids."""
        result = name_compound(
            "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC"
        )
        assert "decanoyloxy" in result
        assert "docosanoyloxy" in result

    def test_monoglyceride_no_oate(self):
        """Monoglyceride: one acyloxy, NOT -oate suffix."""
        result = name_compound("OCC(O)COC(=O)C")
        assert result is not None
        assert "oate" not in result, f"Got '-oate' suffix: {result}"

    def test_simple_ester_unaffected_ethyl_acetate(self):
        """Simple mono-esters still use functional class naming."""
        assert name_compound("CCOC(C)=O") == "ethyl acetate"

    def test_simple_ester_unaffected_methyl_propanoate(self):
        """Methyl propanoate is NOT polyfunctional -- no demotion."""
        assert name_compound("COC(=O)CC") == "methyl propanoate"

    def test_simple_ester_unaffected_propyl_acetate(self):
        """Propyl acetate is NOT polyfunctional -- no demotion."""
        assert name_compound("CCCOC(C)=O") == "propyl acetate"

    def test_ester_plus_amine(self):
        """Ester + amine: ester becomes acyloxy prefix, amine becomes suffix."""
        result = name_compound("CC(=O)OCC(N)C")
        assert result is not None
        assert "oate" not in result

    def test_ester_plus_alcohol_produces_ol(self):
        """Ester + alcohol: alcohol becomes -ol suffix."""
        result = name_compound("CC(=O)OCCO")
        assert result is not None
        assert "ol" in result
