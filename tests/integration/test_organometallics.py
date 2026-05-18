"""Phase 161 integration tests directly anchored to ROADMAP success criterion 1.

These tests verify that Orthonym names the most prominent organometallic
compounds (ferrocene, ruthenocene, cobaltocene, etc.) correctly per IUPAC
P-69 + Salzer 1999 + IR-10. Provides a quick smoke-test surface for the
ORGM-01 (sandwich complexes) + ORGM-02 (seniority cascade) requirements.

NEVER uses @pytest.mark.xfail (CONTEXT D-29) — honest-fail-on-data.
"""
import pytest

from orthonym import name_compound


@pytest.mark.integration
class TestRoadmapOrgm01:
    """Direct verification of ROADMAP success criterion 1 anchors.

    ROADMAP SC-1: 'Orthonym correctly names ferrocene as
    bis(η⁵-cyclopentadienyl)iron' (systematic) and 'ferrocene' (retained PIN).
    """

    def test_ferrocene_pin(self):
        """ROADMAP SC-1: ferrocene retained PIN."""
        assert name_compound(
            '[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'ferrocene'

    def test_ferrocene_systematic_matches_roadmap(self):
        """ROADMAP SC-1 verbatim: bis(η⁵-cyclopentadienyl)iron(II).

        Note: ROADMAP says "bis(η⁵-cyclopentadienyl)iron"; per Salzer §3.2 + D-07
        the systematic form INCLUDES Stock (II): "bis(η⁵-cyclopentadienyl)iron(II)"
        Both forms are acceptable per IUPAC PIN practice; CONTEXT D-03 locks the
        Stock-inclusive form as default systematic emission.
        """
        result = name_compound(
            '[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1', style='systematic'
        )
        assert result in (
            'bis(η⁵-cyclopentadienyl)iron(II)',
            'bis(η⁵-cyclopentadienyl)iron',
        )

    def test_ruthenocene_pin(self):
        """Ru(C5H5)2."""
        assert name_compound(
            '[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'ruthenocene'

    def test_osmocene_pin(self):
        """Os(C5H5)2."""
        assert name_compound(
            '[Os+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'osmocene'

    def test_cobaltocene_pin(self):
        """Co(C5H5)2."""
        assert name_compound(
            '[Co+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'cobaltocene'

    def test_nickelocene_pin(self):
        """Ni(C5H5)2."""
        assert name_compound(
            '[Ni+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'nickelocene'

    def test_chromocene_pin(self):
        """Cr(C5H5)2."""
        assert name_compound(
            '[Cr+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'chromocene'

    def test_vanadocene_pin(self):
        """V(C5H5)2."""
        assert name_compound(
            '[V+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'vanadocene'

    def test_manganocene_pin(self):
        """Mn(C5H5)2."""
        assert name_compound(
            '[Mn+2].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'manganocene'

    def test_ferrocenium_pin(self):
        """Fe(C5H5)2(+1) — ferrocenium cation."""
        assert name_compound(
            '[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1', style='pin'
        ) == 'ferrocenium'


@pytest.mark.integration
class TestOrgm03MetalCarbonyls:
    """Tier-2 metal carbonyls per Salzer §6."""

    def test_pentacarbonyliron(self):
        """Fe(CO)5."""
        assert name_compound(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]',
            style='pin',
        ) == 'pentacarbonyliron'

    def test_tetracarbonylnickel(self):
        """Ni(CO)4."""
        assert name_compound(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Ni]',
            style='pin',
        ) == 'tetracarbonylnickel'


@pytest.mark.integration
class TestOrgm04SigmaBonded:
    """Tier-3 σ-bonded main-group organometallics per P-69.3 + P-69.2."""

    def test_methyllithium(self):
        """CH3-Li."""
        assert name_compound('[Li][CH3]', style='pin') == 'methyllithium'

    def test_dimethylzinc(self):
        """(CH3)2Zn."""
        assert name_compound('C[Zn]C', style='pin') == 'dimethylzinc'

    def test_tetramethylstannane(self):
        """(CH3)4Sn — hydride-parent system."""
        assert name_compound('C[Sn](C)(C)C', style='pin') == 'tetramethylstannane'

    def test_tetramethylsilane(self):
        """(CH3)4Si — hydride-parent system."""
        assert name_compound('C[Si](C)(C)C', style='pin') == 'tetramethylsilane'
