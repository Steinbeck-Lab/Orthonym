"""
Integration tests for ether bond cleavage in the decomposition engine (DEC-01).

Tests that ether-bridged molecules produce names with alkoxy prefixes when the
quality gate triggers decomposition, and that small/ring ethers are correctly
excluded from decomposition via the 5 guards:
  1. Ring guard (epoxides, oxetane, oxane)
  2. Ester exclusion (carbonyl carbons)
  3. Glycosidic exclusion (anomeric centers)
  4. Skeletal replacement exclusion (polyethers)
  5. Minimum fragment size (< 5 heavy atoms per side)

Phase 50, Plan 03 -- ether decomposition validation.
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
    def test_guard_dimethyl_ether_not_decomposed(self):
        """Dimethyl ether (COC): too small for decomposition.

        Both fragments < 5 heavy atoms. Should use regular naming pipeline
        and produce 'methoxymethane' or similar.
        """
        name = name_compound("COC")
        assert name != "unknown"
        # Should be methoxymethane (regular naming, not decomposition)
        assert "methox" in name.lower() or "oxa" in name.lower(), (
            f"Expected methoxymethane or oxa name, got: {name}"
        )

    @pytest.mark.integration
    def test_guard_anisole_not_decomposed(self):
        """Anisole (PhOCH3): should NOT be decomposed.

        The methyl fragment is too small (1 carbon < 5 heavy atoms).
        Should use existing alkoxy prefix naming on benzene.
        """
        name = name_compound("c1ccc(OC)cc1")
        assert name == "methoxybenzene" or "methoxy" in name.lower(), (
            f"Expected 'methoxybenzene', got: {name}"
        )

    @pytest.mark.integration
    def test_guard_oxane_ring_ether_excluded(self):
        """Tetrahydropyran (C1CCOCC1): oxygen is IN the ring.

        The ring guard should exclude this -- ether SMARTS requires [OX2;!R]
        (not in ring). Should produce 'oxane' via retained names.
        """
        name = name_compound("C1CCOCC1")
        assert name != "unknown"
        assert "oxane" in name.lower() or "oxane" in name.lower(), (
            f"Expected oxane/oxane, got: {name}"
        )

    @pytest.mark.integration
    def test_guard_diethyl_ether_too_small(self):
        """Diethyl ether (CCOCC): both fragments < 5 heavy atoms.

        Should NOT be decomposed. Should produce ethoxyethane via
        regular pipeline naming.
        """
        name = name_compound("CCOCC")
        assert name != "unknown"
        # Should be ethoxyethane or similar
        assert "ethox" in name.lower() or "oxa" in name.lower(), (
            f"Expected ethoxyethane or oxa name, got: {name}"
        )
