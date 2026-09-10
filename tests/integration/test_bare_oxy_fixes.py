"""
Integration tests for bare oxy / format fix verification .

Tests that benzene.py and polyfunctional.py no longer produce bare "oxy"
prefixes for compounds where a nameable alkyl chain exists. Also verifies
that O-S, O-P neighbor handling produces proper prefixes (sulfanyloxy,
phosphonooxy) and that legitimate alkoxy names still work.

Bare oxy detection: a name has "bare oxy" if it starts with "oxy" or has
"oxy" preceded by a non-letter character (space, hyphen, parenthesis)
without a qualifying alkyl/aryl prefix. Compounds like "methoxybenzene"
or "sulfanyloxybenzene" are NOT bare oxy.

a phase, Plan 03 -- bare oxy elimination validation.
"""

import pytest

from orthonym import name_compound


def _has_bare_oxy(name: str) -> bool:
    """Check if a name contains bare 'oxy' without a qualifying prefix.

    Returns True if 'oxy' appears:
    - At the start of the name
    - Preceded by space, hyphen, or open paren (no alkyl prefix)

    Returns False for legitimate alkoxy names (methoxy, ethoxy, etc.)
    where 'oxy' is preceded by alphabetic characters.
    """
    if name.startswith("oxy"):
        return True
    for pattern in [" oxy", "-oxy-", "(oxy"]:
        if pattern in name:
            return True
    return False


class TestBareOxyElimination:
    """Verify that bare 'oxy' is eliminated for nameable alkyl chains."""

    @pytest.mark.integration
    def test_benzene_nonyloxy_no_bare_oxy(self):
        """Benzene with C9 alkyl ether: should produce nonoxybenzene, not bare oxy."""
        name = name_compound("c1ccc(OCCCCCCCCC)cc1")
        assert name != "unknown"
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"
        assert "oxy" in name.lower(), f"Expected alkoxy prefix in: {name}"

    @pytest.mark.integration
    def test_benzene_hexyloxy_no_bare_oxy(self):
        """Benzene with C6 alkyl ether: should produce hexoxybenzene."""
        name = name_compound("c1ccc(OCCCCCC)cc1")
        assert name != "unknown"
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"
        assert "oxy" in name.lower(), f"Expected alkoxy prefix in: {name}"

    @pytest.mark.integration
    def test_benzene_undecyloxy_no_bare_oxy(self):
        """Benzene with C11 alkyl ether: should produce undecoxybenzene."""
        name = name_compound("c1ccc(OCCCCCCCCCCC)cc1")
        assert name != "unknown"
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"

    @pytest.mark.integration
    def test_benzene_dodecyloxy_no_bare_oxy(self):
        """Benzene with C12 alkyl ether: should produce dodecoxybenzene."""
        name = name_compound("c1ccc(OCCCCCCCCCCCC)cc1")
        assert name != "unknown"
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"

    @pytest.mark.integration
    def test_benzene_branched_alkoxy_no_bare_oxy(self):
        """Benzene with branched alkyl ether: no bare oxy."""
        # isobutoxybenzene = OCC(C)C on benzene
        name = name_compound("c1ccc(OCC(C)C)cc1")
        assert name != "unknown"
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"
        assert "oxy" in name.lower(), f"Expected alkoxy prefix in: {name}"


class TestOSNeighborNaming:
    """Verify O-S neighbors produce sulfanyloxy/sulfinyloxy/sulfonyloxy."""

    @pytest.mark.integration
    def test_benzene_o_sulfanyl(self):
        """Benzene O-S: should produce sulfanyloxybenzene (S without =O)."""
        name = name_compound("c1ccc(OS)cc1")
        assert "sulfanyloxy" in name.lower(), (
            f"Expected 'sulfanyloxy', got: {name}"
        )

    @pytest.mark.integration
    def test_benzene_o_sulfinyl(self):
        """Benzene O-S(=O): should produce sulfinyloxybenzene."""
        name = name_compound("c1ccc(OS(=O)C)cc1")
        assert "sulfinyloxy" in name.lower(), (
            f"Expected 'sulfinyloxy', got: {name}"
        )

    @pytest.mark.integration
    def test_benzene_o_sulfonyl(self):
        """Benzene O-SO2: should produce sulfonyloxybenzene."""
        name = name_compound("c1ccc(OS(=O)(=O)C)cc1")
        assert "sulfonyloxy" in name.lower(), (
            f"Expected 'sulfonyloxy', got: {name}"
        )


class TestOPNeighborNaming:
    """Verify O-P neighbors produce phosphonooxy."""

    @pytest.mark.integration
    def test_benzene_o_phosphono(self):
        """Benzene O-P(=O)(OH)2: should produce phosphonooxybenzene."""
        name = name_compound("c1ccc(OP(=O)(O)O)cc1")
        assert "phosphonooxy" in name.lower(), (
            f"Expected 'phosphonooxy', got: {name}"
        )


class TestLegitimateAlkoxyPreserved:
    """Verify that legitimate alkoxy names are NOT broken by the fixes."""

    @pytest.mark.integration
    def test_methoxybenzene_still_works(self):
        """Methoxybenzene: now returns 'anisole' (retained name, PIN)."""
        name = name_compound("c1ccc(OC)cc1")
        assert name == "anisole" or "methoxy" in name.lower(), (
            f"Expected 'anisole' or 'methoxy' in name, got: {name}"
        )

    @pytest.mark.integration
    def test_ethoxybenzene_still_works(self):
        """Ethoxybenzene: should still produce 'ethoxybenzene'."""
        name = name_compound("c1ccc(OCC)cc1")
        assert "ethoxy" in name.lower(), (
            f"Expected 'ethoxy' in name, got: {name}"
        )

    @pytest.mark.integration
    def test_isobutoxybenzene_still_works(self):
        """Isobutoxybenzene: should produce isobutoxybenzene."""
        name = name_compound("c1ccc(OCC(C)C)cc1")
        assert "oxy" in name.lower(), (
            f"Expected alkoxy in name, got: {name}"
        )
        assert not _has_bare_oxy(name), f"Bare 'oxy' found in: {name}"


class TestEdgeCases:
    """Edge cases for oxy naming."""

    @pytest.mark.integration
    def test_o_n_on_benzene_documented_bare_oxy(self):
        """O-N on benzene without multiple O on N: documented bare oxy case.

        This is the one case where bare 'oxy' is acceptable -- O-N linkage
        without standard IUPAC prefix. Verifying it produces a name and
        doesn't crash.
        """
        name = name_compound("c1ccc(ON)cc1")
        assert name != "unknown", "Should not be unknown"
        assert isinstance(name, str) and len(name) > 0

    @pytest.mark.integration
    def test_nitrooxy_on_benzene(self):
        """O-N(=O)=O on benzene: should produce (nitrooxy)benzene."""
        name = name_compound("c1ccc(O[N+](=O)[O-])cc1")
        assert name != "unknown"
        assert "nitrooxy" in name.lower() or "nitr" in name.lower(), (
            f"Expected nitrooxy reference, got: {name}"
        )
