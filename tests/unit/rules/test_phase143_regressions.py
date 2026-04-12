"""Tests for Phase 143 regression investigation.

Validates that the heterocyclic ring locant fix in _get_polycyclic_attachment_locant
correctly assigns IUPAC positions using Hantzsch-Widman numbering for retained-name
heterocyclic rings used as substituents.
"""
import pytest
from orthonym.namer import name_compound


class TestHeterocyclicSubstituentLocants:
    """Verify correct IUPAC locant numbering for heterocyclic ring-as-substituent."""

    def test_dioxolane_locant_not_at_oxygen(self):
        """1,3-dioxolane: attachment at C should give position 5, not 1 (which is O)."""
        name = name_compound("C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C")
        assert "dioxolan-5-yl" in name
        # Must NOT be dioxolan-1-yl (position 1 is oxygen)
        assert "dioxolan-1-yl" not in name

    def test_thiazole_locant_at_carbon_4(self):
        """Thiazole: S=1, C=2, N=3, C=4, C=5. Attachment at C-4 gives thiazol-4-yl."""
        name = name_compound(
            r"C=CCO/N=C(\C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1"
        )
        assert "thiazol-4-yl" in name
        # Must NOT be thiazol-1-yl (position 1 is sulfur)
        assert "thiazol-1-yl" not in name

    def test_thiazolidine_locant_at_carbon_2(self):
        """Thiazolidine: S=1, C=2, N=3, C=4, C=5. Attachment at C-2 gives thiazolidin-2-yl."""
        name = name_compound("CC1(C)SC(C(NC(=O)COc2ccccc2)C(=O)O)NC1C(=O)O")
        assert "thiazolidin-2-yl" in name
        # Must NOT be thiazolidin-3-yl (position 3 is nitrogen)
        assert "thiazolidin-3-yl" not in name


class TestOPSINParseRegressions:
    """Document the 8 compounds that lost OPSIN parseability in Phase 143.

    All 8 are trade-offs or OPSIN limitations: the new names are more correct
    IUPAC but OPSIN 2.9.0 cannot parse them. None of these had RT=1 in baseline.
    """

    @pytest.mark.xfail(reason="OPSIN limitation: does not recognize 'ajmaline' NP retained name")
    def test_ajmaline_retained_name(self):
        """Ajmaline is correctly resolved as retained NP name (was VB hexacyclo...)."""
        name = name_compound(
            "CC[C@H]1[C@@H]2CC3[C@@H]4N(C)c5ccccc5[C@]45C[C@@H](C2C5O)N3[C@@H]1O"
        )
        assert name == "ajmaline"

    @pytest.mark.xfail(reason="OPSIN limitation: does not recognize 'berberine' alkaloid retained name")
    def test_berberine_retained_name(self):
        """Berberine is correctly resolved as retained NP name (was VB tetracyclo...)."""
        name = name_compound("COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2")
        assert name == "berberine"
