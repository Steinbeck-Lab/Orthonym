"""
Tests for terminal multi-suffix compounds: diacids, dialdehydes, dinitriles, diamides.

Validates that validate_suffix_locants preserves the multiplier count
when locants are empty (terminal groups that don't need explicit locants).

IUPAC 2013 convention:
- Terminal groups use compact form WITHOUT locants:
    pentanedioic acid, pentanedial, hexanedinitrile, pentanediamide
- Position-variable groups use explicit locants:
    butane-1,4-diol, pentane-2,4-dione, hexane-1,6-diamine

Reference: IUPAC 2013 Blue Book,,
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Diacids : terminal -oic acid at both ends
# ---------------------------------------------------------------------------

class TestDiacids:
    """Diacids: two carboxylic acid groups at chain termini."""

    @pytest.mark.unit
    def test_propanedioic_acid(self):
        # malonic acid (systematic PIN)
        assert name_compound("OC(=O)CC(=O)O") == "propanedioic acid"

    @pytest.mark.unit
    def test_butanedioic_acid(self):
        # succinic acid (systematic PIN)
        assert name_compound("OC(=O)CCC(=O)O") == "butanedioic acid"

    @pytest.mark.unit
    def test_pentanedioic_acid(self):
        # glutaric acid (systematic PIN)
        assert name_compound("OC(=O)CCCC(=O)O") == "pentanedioic acid"

    @pytest.mark.unit
    def test_hexanedioic_acid(self):
        # adipic acid (systematic PIN)
        assert name_compound("OC(=O)CCCCC(=O)O") == "hexanedioic acid"

    @pytest.mark.unit
    def test_heptanedioic_acid(self):
        assert name_compound("OC(=O)CCCCCC(=O)O") == "heptanedioic acid"


# ---------------------------------------------------------------------------
# Dialdehydes : terminal -al at both ends
# ---------------------------------------------------------------------------

class TestDialdehydes:
    """Dialdehydes: two aldehyde groups at chain termini."""

    @pytest.mark.unit
    def test_propanedial(self):
        assert name_compound("O=CCC=O") == "propanedial"

    @pytest.mark.unit
    def test_butanedial(self):
        assert name_compound("O=CCCC=O") == "butanedial"

    @pytest.mark.unit
    def test_pentanedial(self):
        assert name_compound("O=CCCCC=O") == "pentanedial"

    @pytest.mark.unit
    def test_hexanedial(self):
        assert name_compound("O=CCCCCC=O") == "hexanedial"


# ---------------------------------------------------------------------------
# Dinitriles : terminal -nitrile at both ends
# ---------------------------------------------------------------------------

class TestDinitriles:
    """Dinitriles: two nitrile groups at chain termini."""

    @pytest.mark.unit
    def test_propanedinitrile(self):
        assert name_compound("N#CCC#N") == "propanedinitrile"

    @pytest.mark.unit
    def test_butanedinitrile(self):
        assert name_compound("N#CCCC#N") == "butanedinitrile"

    @pytest.mark.unit
    def test_pentanedinitrile(self):
        assert name_compound("N#CCCCC#N") == "pentanedinitrile"

    @pytest.mark.unit
    def test_hexanedinitrile(self):
        assert name_compound("N#CCCCCC#N") == "hexanedinitrile"


# ---------------------------------------------------------------------------
# Diamides : terminal -amide at both ends
# ---------------------------------------------------------------------------

class TestDiamides:
    """Diamides: two primary amide groups at chain termini."""

    @pytest.mark.unit
    def test_propanediamide(self):
        assert name_compound("NC(=O)CC(=O)N") == "propanediamide"

    @pytest.mark.unit
    def test_butanediamide(self):
        assert name_compound("NC(=O)CCC(=O)N") == "butanediamide"

    @pytest.mark.unit
    def test_pentanediamide(self):
        assert name_compound("NC(=O)CCCC(=O)N") == "pentanediamide"

    @pytest.mark.unit
    def test_hexanediamide(self):
        assert name_compound("NC(=O)CCCCC(=O)N") == "hexanediamide"


# ---------------------------------------------------------------------------
# Position-variable multi-suffix (REGRESSION): must still include locants
# ---------------------------------------------------------------------------

class TestPositionVariableRegression:
    """Position-variable groups: locants required, multiplier preserved."""

    @pytest.mark.unit
    def test_butane_1_4_diol(self):
        assert name_compound("OCCCCO") == "butane-1,4-diol"

    @pytest.mark.unit
    def test_pentane_2_4_dione(self):
        assert name_compound("CC(=O)CC(=O)C") == "pentane-2,4-dione"

    @pytest.mark.unit
    def test_hexane_1_6_diamine(self):
        assert name_compound("NCCCCCCN") == "hexane-1,6-diamine"


# ---------------------------------------------------------------------------
# validate_suffix_locants unit tests (direct function validation)
# ---------------------------------------------------------------------------

class TestValidateSuffixLocantsDirect:
    """Direct tests for the validate_suffix_locants function."""

    @pytest.mark.unit
    def test_empty_locants_preserves_count_2(self):
        from orthonym.rules.locant_validation import validate_suffix_locants
        assert validate_suffix_locants([], 5, 2) == ([], 2)

    @pytest.mark.unit
    def test_empty_locants_preserves_count_3(self):
        from orthonym.rules.locant_validation import validate_suffix_locants
        assert validate_suffix_locants([], 5, 3) == ([], 3)

    @pytest.mark.unit
    def test_valid_locants_unchanged(self):
        from orthonym.rules.locant_validation import validate_suffix_locants
        assert validate_suffix_locants([1, 3], 6, 2) == ([1, 3], 2)

    @pytest.mark.unit
    def test_out_of_range_locants_filtered(self):
        from orthonym.rules.locant_validation import validate_suffix_locants
        assert validate_suffix_locants([1, 3, 8], 6, 3) == ([1, 3], 2)

    @pytest.mark.unit
    def test_zero_locant_filtered(self):
        from orthonym.rules.locant_validation import validate_suffix_locants
        assert validate_suffix_locants([0, 2, 4], 6, 3) == ([2, 4], 2)
