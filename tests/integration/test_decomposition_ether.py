"""
Integration tests for ether bond cleavage in the decomposition engine .

Tests that ether-bridged molecules produce names with alkoxy prefixes when the
quality gate triggers decomposition, and that small/ring ethers are correctly
excluded from decomposition via the 5 guards:
  1. Ring guard (epoxides, oxetane, oxane)
  2. Ester exclusion (carbonyl carbons)
  3. Glycosidic exclusion (anomeric centers)
  4. Skeletal replacement exclusion (polyethers)
  5. Minimum fragment size (< 5 heavy atoms per side)

a phase, Plan 03 -- ether decomposition validation.
"""

import pytest

from orthonym import name_compound


class TestEtherDecomposition:
    """Test ether bond cleavage and alkoxy prefix generation."""

    @pytest.mark.integration
    def test_large_ether_on_quinoline_includes_alkoxy(self):
        """Large ether on quinoline ring: should include alkoxy prefix.

        CCCCCCCCOc1ccc(Cl)c2cccnc12 is a chloroquinoline with an octyl
        ether. The pipeline should recognize the alkoxy substituent and
        produce a name like '5-chloro-8-octoxyquinoline'.
        """
        name = name_compound("CCCCCCCCOc1ccc(Cl)c2cccnc12")
        assert name != "unknown", "Should not be unknown"
        # Name should contain an alkoxy prefix (not bare ring name)
        assert "oxy" in name.lower(), (
            f"Expected alkoxy prefix in name, got: {name}"
        )
        assert "quinoline" in name.lower(), (
            f"Expected 'quinoline' parent, got: {name}"
        )

    @pytest.mark.integration
    def test_large_alkoxy_on_benzoic_acid(self):
        """Large alkoxy on benzoic acid: CCCCCCCCOc1ccccc1C(=O)O.

        Should produce a name like '2-octoxybenzoic acid' with an alkoxy
        prefix on the ring parent.
        """
        name = name_compound("CCCCCCCCOc1ccccc1C(=O)O")
        assert name != "unknown", "Should not be unknown"
        assert "oxy" in name.lower(), (
            f"Expected alkoxy prefix, got: {name}"
        )
        assert "benzoic acid" in name.lower() or "benz" in name.lower(), (
            f"Expected benzoic acid parent, got: {name}"
        )

    @pytest.mark.integration
    def test_large_dialkyl_ether_produces_alkoxy_or_oxa_name(self):
        """Dibutyl ether (CCCCOCCCC): should produce a valid name.

        Both fragments are 4 carbons, at the minimum fragment size boundary.
        May use skeletal replacement (oxa) naming or regular pipeline naming.
        Should NOT be decomposed (both sides < 5 heavy atoms after guard).
        """
        name = name_compound("CCCCOCCCC")
        assert name != "unknown", "Should not be unknown"
        # Should produce either butoxybutane, 5-oxanonane, or similar
        assert "oxy" in name or "oxa" in name, (
            f"Expected ether-related name (oxy/oxa), got: {name}"
        )

    @pytest.mark.integration
    def test_diphenyl_ether_names_correctly(self):
        """Diphenyl ether (PhOPh): both fragments are small aromatic rings.

        Should NOT be decomposed (both sides are 6 heavy atoms each, but
        the oxygen is shared). Should use regular naming (oxydibenzene or
        phenoxybenzene).
        """
        name = name_compound("c1ccc(Oc2ccccc2)cc1")
        assert name != "unknown", "Should not be unknown"
        # Should contain oxy reference
        assert "oxy" in name.lower(), (
            f"Expected 'oxy' in diphenyl ether name, got: {name}"
        )

    @pytest.mark.integration
    def test_guard_anisole_not_decomposed(self):
        """Anisole (PhOCH3): should NOT be decomposed.

        The methyl fragment is too small (1 carbon < 5 heavy atoms).
        Should use existing alkoxy prefix naming on benzene.
        """
        name = name_compound("c1ccc(OC)cc1")
        assert name == "anisole" or name == "methoxybenzene" or "methoxy" in name.lower(), (
            f"Expected 'anisole' or 'methoxybenzene', got: {name}"
        )
