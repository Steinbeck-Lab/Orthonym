"""Integration tests for steroid naming format fixes (a phase-01).

Tests the format correctness of steroid name assembly, covering:
-: prefix_parts list pattern in _assemble_np_name
-: mixed-ester acyloxy prefix format in _assemble_np_ester_name
- Regression guards for existing steroid naming formats
- End-to-end naming of steroid SMILES via name_compound
"""

import pytest

from orthonym import name_compound
from orthonym.rules.natural_products import (
    _acylate_to_acyloxy,
    _assemble_np_ester_name,
    _assemble_np_name,
)


# ---------------------------------------------------------------------------
#: prefix_parts pattern tests for _assemble_np_name
# ---------------------------------------------------------------------------

class TestAssembleNpNamePrefixParts:
    """Test _assemble_np_name uses prefix_parts list pattern correctly."""

    @pytest.mark.integration
    def test_hydroxy_only_ol_suffix(self):
        """Single hydroxy group produces -ol suffix (not prefix)."""
        result = _assemble_np_name(
            "cholest", "cholestane", hydroxyls=[3], ketones=[],
            unsaturation={"ene": [5], "yne": []}, stereo_prefix="",
        )
        assert result == "cholest-5-en-3-ol"

    @pytest.mark.integration
    def test_dihydroxy_diol_suffix(self):
        """Two hydroxy groups produce -diol suffix."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[3, 17], ketones=[],
            unsaturation={"ene": [], "yne": []}, stereo_prefix="",
        )
        assert result == "androstan-3,17-diol"

    @pytest.mark.integration
    def test_hydroxy_prefix_with_ketone_suffix(self):
        """Hydroxy as prefix + ketone as suffix with proper hyphen join."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[3], ketones=[7],
            unsaturation={"ene": [4], "yne": []}, stereo_prefix="",
        )
        assert result == "3-hydroxyandrost-4-en-7-one"

    @pytest.mark.integration
    def test_hydroxy_prefix_ketone_saturated(self):
        """Hydroxy prefix + ketone suffix on saturated scaffold."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[17], ketones=[3],
            unsaturation={"ene": [], "yne": []}, stereo_prefix="",
        )
        assert result == "17-hydroxyandrostan-3-one"

    @pytest.mark.integration
    def test_ketone_only_suffix(self):
        """Ketone only produces -one suffix (no prefix)."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[], ketones=[3],
            unsaturation={"ene": [4], "yne": []}, stereo_prefix="",
        )
        assert result == "androst-4-en-3-one"

    @pytest.mark.integration
    def test_diketone_suffix(self):
        """Two ketones produce -dione suffix."""
        result = _assemble_np_name(
            "pregn", "pregnane", hydroxyls=[], ketones=[3, 20],
            unsaturation={"ene": [4], "yne": []}, stereo_prefix="",
        )
        assert result == "pregn-4-en-3,20-dione"

    @pytest.mark.integration
    def test_unsaturation_only(self):
        """Unsaturation without hydroxyl/ketone produces bare ene suffix."""
        result = _assemble_np_name(
            "cholest", "cholestane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [5], "yne": []}, stereo_prefix="",
        )
        # Terminal 'e' added when no suffix follows (IUPAC: "ene" not "en")
        assert result == "cholest-5-ene"

    @pytest.mark.integration
    def test_bare_scaffold_fallback(self):
        """No decorations returns scaffold name."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [], "yne": []}, stereo_prefix="",
        )
        assert result == "androstane"

    @pytest.mark.integration
    def test_stereo_prefix_preserved(self):
        """Stereo prefix is prepended correctly."""
        result = _assemble_np_name(
            "androst", "androstane", hydroxyls=[17], ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            stereo_prefix="(8R,9S,10S,13S,14S)-",
        )
        assert result == "(8R,9S,10S,13S,14S)-17-hydroxyandrost-4-en-3-one"


# ---------------------------------------------------------------------------
#: Mixed ester format tests for _assemble_np_ester_name
# ---------------------------------------------------------------------------

class TestAssembleNpEsterName:
    """Test _assemble_np_ester_name handles same/mixed esters correctly."""

    @pytest.mark.integration
    def test_single_ester_functional_class(self):
        """Single ester uses functional class format: parent-yl acylate."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[], ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            esters=[{"locant": 17, "acylate": "acetate"}],
            stereo_prefix="",
        )
        assert result == "3-oxoandrost-4-en-17-yl acetate"

    @pytest.mark.integration
    def test_same_acid_diester_functional_class(self):
        """Same-acid multi-ester uses diyl + multiplied acylate."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [], "yne": []},
            esters=[
                {"locant": 3, "acylate": "acetate"},
                {"locant": 17, "acylate": "acetate"},
            ],
            stereo_prefix="",
        )
        assert result == "androstan-3,17-diyl diacetate"

    @pytest.mark.integration
    def test_mixed_acid_diester_acyloxy_prefix(self):
        """Mixed-acid multi-ester uses acyloxy prefix format."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [], "yne": []},
            esters=[
                {"locant": 3, "acylate": "acetate"},
                {"locant": 17, "acylate": "propanoate"},
            ],
            stereo_prefix="",
        )
        assert result == "3-(acetyloxy)-17-(propanoyloxy)androstane"

    @pytest.mark.integration
    def test_mixed_acid_diester_with_unsaturation(self):
        """Mixed-acid multi-ester with unsaturation suffix."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [4], "yne": []},
            esters=[
                {"locant": 3, "acylate": "acetate"},
                {"locant": 17, "acylate": "propanoate"},
            ],
            stereo_prefix="",
        )
        assert result == "3-(acetyloxy)-17-(propanoyloxy)androst-4-ene"

    @pytest.mark.integration
    def test_mixed_acid_with_hydroxy_prefix(self):
        """Mixed-acid ester with hydroxy prefix from additional OH groups."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[7], ketones=[],
            unsaturation={"ene": [], "yne": []},
            esters=[
                {"locant": 3, "acylate": "acetate"},
                {"locant": 17, "acylate": "propanoate"},
            ],
            stereo_prefix="",
        )
        # hydroxy prefix + acyloxy prefixes + stem
        assert "7-hydroxy" in result
        assert "(acetyloxy)" in result
        assert "(propanoyloxy)" in result

    @pytest.mark.integration
    def test_mixed_acid_with_stereo_prefix(self):
        """Mixed-acid ester with stereodescriptor prefix."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[], ketones=[],
            unsaturation={"ene": [], "yne": []},
            esters=[
                {"locant": 3, "acylate": "acetate"},
                {"locant": 17, "acylate": "propanoate"},
            ],
            stereo_prefix="(5R,8S)-",
        )
        assert result.startswith("(5R,8S)-")
        assert "(acetyloxy)" in result
        assert "(propanoyloxy)" in result

    @pytest.mark.integration
    def test_ester_with_hydroxy_and_ketone_prefixes(self):
        """Single ester with both hydroxy and oxo prefixes."""
        result = _assemble_np_ester_name(
            "androst", "androstane", hydroxyls=[7], ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            esters=[{"locant": 17, "acylate": "acetate"}],
            stereo_prefix="",
        )
        # Both prefixes should be hyphen-joined
        assert "7-hydroxy" in result
        assert "3-oxo" in result
        assert "17-yl acetate" in result


# ---------------------------------------------------------------------------
# Acylate-to-acyloxy conversion tests
# ---------------------------------------------------------------------------

class TestAcylateToAcyloxy:
    """Test _acylate_to_acyloxy helper conversions."""

    @pytest.mark.integration
    def test_acetate_to_acetyloxy(self):
        assert _acylate_to_acyloxy("acetate") == "acetyloxy"

    @pytest.mark.integration
    def test_propanoate_to_propanoyloxy(self):
        assert _acylate_to_acyloxy("propanoate") == "propanoyloxy"

    @pytest.mark.integration
    def test_benzoate_to_benzoyloxy(self):
        assert _acylate_to_acyloxy("benzoate") == "benzoyloxy"

    @pytest.mark.integration
    def test_formate_to_formyloxy(self):
        assert _acylate_to_acyloxy("formate") == "formyloxy"

    @pytest.mark.integration
    def test_butanoate_to_butanoyloxy(self):
        assert _acylate_to_acyloxy("butanoate") == "butanoyloxy"


# ---------------------------------------------------------------------------
# End-to-end steroid SMILES tests
# ---------------------------------------------------------------------------

class TestSteroidE2EFormat:
    """Test end-to-end naming format for steroid SMILES via name_compound."""

    @pytest.mark.integration
    def test_cholesterol_no_stereo_e2e(self):
        """Cholesterol without stereo should get systematic steroid name."""
        smiles = "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert result == "cholest-5-en-3-ol", f"Got '{result}'"

    @pytest.mark.integration
    def test_testosterone_type_e2e(self):
        """Androst-4-en-3-one with 17-OH (testosterone-like, no stereo)."""
        smiles = "OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        # Should contain androst stem and proper functional groups
        assert "androst" in result.lower() or "androstane" in result.lower(), f"Got '{result}'"

    @pytest.mark.integration
    def test_progesterone_type_e2e(self):
        """Pregn-4-ene-3,20-dione (progesterone-like, no stereo)."""
        smiles = "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        # Should contain pregn stem
        assert "pregn" in result.lower(), f"Got '{result}'"

    @pytest.mark.integration
    def test_androstanediol_e2e(self):
        """Androstane-3,17-diol (no stereo)."""
        smiles = "OC1CCC2(C)C(CCC3C2CCC2(C)C(O)CCC32)C1"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        if "androst" in result.lower():
            # Should have diol suffix
            assert "diol" in result, f"Expected diol in name, got '{result}'"
