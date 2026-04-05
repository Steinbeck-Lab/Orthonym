"""
Tests for IUPAC 2013 PIN cleanup of retained names (Phase 133-01).

Verifies:
- Acetone removed from RETAINED_NAMES (PIN is propan-2-one per P-31.1.3)
- Cresol isomers removed (PINs are 2/3/4-methylphenol)
- Isobutyric/isovaleric/pivalic acid removed (PINs are systematic names)
- Adamantane added to BICYCLO_RETAINED_NAMES
- Cubane added to BICYCLO_RETAINED_NAMES
"""

import pytest
from orthonym.data.retained_names import RETAINED_NAMES
from orthonym.data.bicyclo_systems import BICYCLO_RETAINED_NAMES
from orthonym import name_compound


class TestAcetoneRemoved:
    """Acetone should NOT be a retained name -- PIN is propan-2-one."""

    @pytest.mark.unit
    def test_acetone_not_in_retained_names_keys(self):
        """CC(C)=O should not be a key in RETAINED_NAMES."""
        assert "CC(C)=O" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_acetone_not_in_retained_names_values(self):
        """'acetone' should not appear as a value in RETAINED_NAMES."""
        # acetophenone contains 'acetone' substring, so check exact match
        for val in RETAINED_NAMES.values():
            assert val != "acetone", f"Found 'acetone' as retained name value"

    @pytest.mark.unit
    def test_acetone_smiles_produces_propan_2_one(self):
        """name_compound('CC(C)=O') should produce 'propan-2-one'."""
        result = name_compound("CC(C)=O")
        assert result == "propan-2-one", f"Got '{result}'"


class TestCresolsRemoved:
    """Cresol isomers should NOT be retained names -- PINs are methylphenols."""

    @pytest.mark.unit
    def test_4_cresol_not_in_retained_names(self):
        """Cc1ccc(O)cc1 should not be a key in RETAINED_NAMES."""
        assert "Cc1ccc(O)cc1" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_2_cresol_not_in_retained_names(self):
        """Cc1ccccc1O should not be a key in RETAINED_NAMES."""
        assert "Cc1ccccc1O" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_3_cresol_not_in_retained_names(self):
        """Cc1cccc(O)c1 should not be a key in RETAINED_NAMES."""
        assert "Cc1cccc(O)c1" not in RETAINED_NAMES

    @pytest.mark.unit
    def test_no_cresol_values_in_retained_names(self):
        """No value in RETAINED_NAMES should contain 'cresol'."""
        for val in RETAINED_NAMES.values():
            assert "cresol" not in val.lower(), f"Found cresol in value: {val}"

    @pytest.mark.unit
    def test_4_methylphenol_naming(self):
        """name_compound('Cc1ccc(O)cc1') should produce '4-methylphenol'."""
        result = name_compound("Cc1ccc(O)cc1")
        assert result == "4-methylphenol", f"Got '{result}'"

    @pytest.mark.unit
    def test_2_methylphenol_naming(self):
        """name_compound('Cc1ccccc1O') should produce '2-methylphenol'."""
        result = name_compound("Cc1ccccc1O")
        assert result == "2-methylphenol", f"Got '{result}'"

    @pytest.mark.unit
    def test_3_methylphenol_naming(self):
        """name_compound('Cc1cccc(O)c1') should produce '3-methylphenol'."""
        result = name_compound("Cc1cccc(O)c1")
        assert result == "3-methylphenol", f"Got '{result}'"


class TestBranchedAcidsRemoved:
    """Isobutyric/isovaleric/pivalic should NOT be retained names."""

    @pytest.mark.unit
    def test_isobutyric_not_in_retained_names(self):
        """'isobutyric acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "isobutyric acid", "Found 'isobutyric acid' as retained name"

    @pytest.mark.unit
    def test_isovaleric_not_in_retained_names(self):
        """'isovaleric acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "isovaleric acid", "Found 'isovaleric acid' as retained name"

    @pytest.mark.unit
    def test_pivalic_not_in_retained_names(self):
        """'pivalic acid' should not appear as a retained name value."""
        for val in RETAINED_NAMES.values():
            assert val != "pivalic acid", "Found 'pivalic acid' as retained name"


class TestAdamantaneAdded:
    """Adamantane should be in BICYCLO_RETAINED_NAMES."""

    @pytest.mark.unit
    def test_adamantane_in_bicyclo_retained_names(self):
        """Adamantane canonical SMILES should be a key in BICYCLO_RETAINED_NAMES."""
        assert "C1C2CC3CC1CC(C2)C3" in BICYCLO_RETAINED_NAMES

    @pytest.mark.unit
    def test_adamantane_value_correct(self):
        """BICYCLO_RETAINED_NAMES should map adamantane SMILES to 'adamantane'."""
        assert BICYCLO_RETAINED_NAMES.get("C1C2CC3CC1CC(C2)C3") == "adamantane"

    @pytest.mark.unit
    def test_adamantane_naming_e2e(self):
        """name_compound('C1C2CC3CC1CC(C2)C3') should produce 'adamantane'."""
        result = name_compound("C1C2CC3CC1CC(C2)C3")
        assert result == "adamantane", f"Got '{result}'"


class TestCubaneAdded:
    """Cubane should be in BICYCLO_RETAINED_NAMES."""

    @pytest.mark.unit
    def test_cubane_in_bicyclo_retained_names(self):
        """Cubane canonical SMILES should be a key in BICYCLO_RETAINED_NAMES."""
        assert "C12C3C4C1C1C2C3C41" in BICYCLO_RETAINED_NAMES

    @pytest.mark.unit
    def test_cubane_value_correct(self):
        """BICYCLO_RETAINED_NAMES should map cubane SMILES to 'cubane'."""
        assert BICYCLO_RETAINED_NAMES.get("C12C3C4C1C1C2C3C41") == "cubane"
