"""Phase 27: Comprehensive isoindoline postprocessor tests.

Tests for the _postprocess_name() regex patterns that normalize
isoindoline-related names to OPSIN-compatible formats:
- isoindoline-X,Y-dione -> isoindoline-1,3-dione (any locant pair)
- isoindolin-2-one -> 2,3-dihydro-1H-isoindol-1-one (N-position ketone fix)
"""

import pytest

from orthonym.namer import _postprocess_name
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Section 1: Direct postprocessor function tests -- dione normalization
# ---------------------------------------------------------------------------


class TestIsoindolineDioneNormalization:
    """Verify isoindoline-X,Y-dione is normalized to isoindoline-1,3-dione."""

    @pytest.mark.unit
    def test_original_exact_case(self):
        """The original 2,4 locant pair that triggered the postprocessor."""
        assert _postprocess_name("isoindoline-2,4-dione") == "isoindoline-1,3-dione"

    @pytest.mark.unit
    def test_already_correct_locants(self):
        """Already-correct 1,3 locants should pass through unchanged."""
        assert _postprocess_name("isoindoline-1,3-dione") == "isoindoline-1,3-dione"

    @pytest.mark.unit
    def test_different_wrong_locants(self):
        """Any other locant pair should also be normalized to 1,3."""
        assert _postprocess_name("isoindoline-3,5-dione") == "isoindoline-1,3-dione"

    @pytest.mark.unit
    def test_prefixed_form(self):
        """Prefixed names should have only the dione locants corrected."""
        assert (
            _postprocess_name("6-hydroxyisoindoline-2,4-dione")
            == "6-hydroxyisoindoline-1,3-dione"
        )

    @pytest.mark.unit
    def test_multi_prefix_form(self):
        """Multiple prefixes should be preserved, only dione locants corrected."""
        assert (
            _postprocess_name("4,5,6,7-tetrachloroisoindoline-2,4-dione")
            == "4,5,6,7-tetrachloroisoindoline-1,3-dione"
        )

    @pytest.mark.unit
    def test_prefixed_correct_locants(self):
        """Prefixed form with already-correct locants should be unchanged."""
        assert (
            _postprocess_name("5-methylisoindoline-1,3-dione")
            == "5-methylisoindoline-1,3-dione"
        )


# ---------------------------------------------------------------------------
# Section 2: Direct postprocessor function tests -- mono-one conversion
# ---------------------------------------------------------------------------


class TestIsoindolinMonoOneConversion:
    """Verify isoindolin-2-one is converted to 2,3-dihydro-1H-isoindol-1-one."""

    @pytest.mark.unit
    def test_bare_mono_one(self):
        """Bare isoindolin-2-one should convert to dihydro form."""
        assert (
            _postprocess_name("isoindolin-2-one")
            == "2,3-dihydro-1H-isoindol-1-one"
        )

    @pytest.mark.unit
    def test_prefixed_mono_one_with_hyphen(self):
        """Prefix ending in digit+hyphen should connect directly."""
        result = _postprocess_name("5,7-dihydroxyisoindolin-2-one")
        assert result == "5,7-dihydroxy-2,3-dihydro-1H-isoindol-1-one"

    @pytest.mark.unit
    def test_prefixed_mono_one_letter_preceded(self):
        """Prefix ending in letter should get hyphen inserted before replacement."""
        result = _postprocess_name("dimethylisoindolin-2-one")
        assert result == "dimethyl-2,3-dihydro-1H-isoindol-1-one"


# ---------------------------------------------------------------------------
# Section 3: Edge cases and negative tests
# ---------------------------------------------------------------------------


class TestIsoindolineEdgeCases:
    """Negative tests and edge cases for isoindoline postprocessing."""

    @pytest.mark.unit
    def test_isoindoline_without_dione(self):
        """Name containing 'isoindoline' but not 'dione' passes through unchanged."""
        assert _postprocess_name("isoindoline") == "isoindoline"

    @pytest.mark.unit
    def test_dione_without_isoindoline(self):
        """Name with 'dione' but not 'isoindoline' passes through unchanged."""
        assert (
            _postprocess_name("cyclohexane-1,3-dione") == "cyclohexane-1,3-dione"
        )

    @pytest.mark.unit
    def test_empty_string(self):
        """Empty string should pass through unchanged."""
        assert _postprocess_name("") == ""

    @pytest.mark.unit
    def test_non_isoindoline_name(self):
        """Completely unrelated names should pass through unchanged."""
        assert _postprocess_name("ethanol") == "ethanol"
        assert _postprocess_name("benzene") == "benzene"

    @pytest.mark.unit
    def test_isoindolin_4_one_not_modified(self):
        """isoindolin-4-one should NOT be modified (OPSIN-parseable as-is)."""
        assert _postprocess_name("isoindolin-4-one") == "isoindolin-4-one"


# ---------------------------------------------------------------------------
# Section 4: End-to-end naming tests
# ---------------------------------------------------------------------------


class TestIsoindolineEndToEnd:
    """End-to-end tests calling name_compound on actual SMILES."""

    @pytest.mark.unit
    def test_phthalimide_substituent(self):
        """Phthalimide with pyrazole substituent gives isoindoline-1,3-dione."""
        name = name_compound("Cc1cc(N2C(=O)c3ccccc3C2=O)n(C)n1")
        assert "isoindoline-1,3-dione" in name

    @pytest.mark.unit
    def test_hydroxy_phthalimide_substituent(self):
        """Hydroxy phthalimide gives 6-hydroxyisoindoline-1,3-dione."""
        name = name_compound("O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)C(=O)N1")
        assert "6-hydroxyisoindoline-1,3-dione" in name

    @pytest.mark.unit
    def test_phthalimide_bare(self):
        """Bare phthalimide SMILES gives isoindoline-1,3-dione."""
        name = name_compound("O=C1NC(=O)c2ccccc21")
        assert "isoindoline-1,3-dione" in name


# ---------------------------------------------------------------------------
# Section 5: Root cause limitation documentation (xfail)
# ---------------------------------------------------------------------------


class TestIsoindolineRootCauseLimitation:
    """Document known limitation: prefix locants not fixable by postprocessor."""

    @pytest.mark.unit
    @pytest.mark.xfail(
        reason=(
            "Root cause locant mapping bug in fused_heterocycles.py -- "
            "prefix locants (e.g., 1-methoxy should be 7-methoxy) are not "
            "fixable by postprocessor regex. Deferred to future phase."
        ),
        strict=True,
    )
    def test_index_200_prefix_locant_bug(self):
        """Index 200: OPSIN fails because prefix locant 1-methoxy is wrong."""
        import subprocess

        name = name_compound("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C")
        # The name should be OPSIN-parseable if locants are correct
        result = subprocess.run(
            [
                "java", "-jar",
                "/home/kohulan/OpenSTOUT/Orthonym/opsin-cli-2.8.0-jar-with-dependencies.jar",
                "-osmi",
            ],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() != "", (
            f"OPSIN cannot parse: {name} (prefix locant mapping is wrong)"
        )
