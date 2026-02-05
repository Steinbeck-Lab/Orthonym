"""
Tests for centralized IUPAC chain naming module.

Validates chain prefix generation for chains from 1 to 999 carbons,
covering standard prefixes (1-20), compositional naming (21+),
and derived forms (alkyl, acid, anoate).
"""

import pytest

from orthonym.data.chain_names import (
    get_chain_prefix,
    get_chain_name,
    get_alkyl_name,
    get_acid_name,
    get_acid_stem,
    get_anoate_name,
)


# ============================================================================
# Standard prefixes (1-20)
# ============================================================================

@pytest.mark.unit
class TestStandardPrefixes:
    """Test all 20 standard IUPAC chain prefixes."""

    @pytest.mark.parametrize("n,expected", [
        (1, "meth"),
        (2, "eth"),
        (3, "prop"),
        (4, "but"),
        (5, "pent"),
        (6, "hex"),
        (7, "hept"),
        (8, "oct"),
        (9, "non"),
        (10, "dec"),
        (11, "undec"),
        (12, "dodec"),
        (13, "tridec"),
        (14, "tetradec"),
        (15, "pentadec"),
        (16, "hexadec"),
        (17, "heptadec"),
        (18, "octadec"),
        (19, "nonadec"),
        (20, "icos"),
    ])
    def test_standard_prefix(self, n, expected):
        assert get_chain_prefix(n) == expected


# ============================================================================
# Compositional naming (21+)
# ============================================================================

@pytest.mark.unit
class TestCompositionalPrefixes:
    """Test IUPAC compositional naming system for 21+ chains."""

    @pytest.mark.parametrize("n,expected", [
        # Twenties (cos series)
        (21, "henicos"),
        (22, "docos"),
        (23, "tricos"),
        (24, "tetracos"),
        (25, "pentacos"),
        (26, "hexacos"),
        (27, "heptacos"),
        (28, "octacos"),
        (29, "nonacos"),
        # Thirties (triacont series)
        (30, "triacont"),
        (31, "hentriacont"),
        (32, "dotriacont"),
        (33, "tritriacont"),
        (34, "tetratriacont"),
        (39, "nonatriacont"),
        # Forties
        (40, "tetracont"),
        (41, "hentetracont"),
        (42, "dotetracont"),
        # Fifties
        (50, "pentacont"),
        (53, "tripentacont"),
        # Other decades
        (60, "hexacont"),
        (70, "heptacont"),
        (80, "octacont"),
        (90, "nonacont"),
        # Hundreds
        (100, "hect"),
        (200, "dict"),
        (300, "trict"),
    ])
    def test_compositional_prefix(self, n, expected):
        assert get_chain_prefix(n) == expected

    @pytest.mark.parametrize("n,expected", [
        # Hundreds + tens
        (110, "decahect"),
        (120, "cosahect"),
        (130, "triacontahect"),
        (140, "tetracontahect"),
        (150, "pentacontahect"),
        # Hundreds + tens + units
        (111, "undecahect"),
        (121, "henicosahect"),
        (132, "dotriacontahect"),
        (199, "nonanonacontahect"),
        # Multi-hundreds
        (210, "decadict"),
        (250, "pentacontadict"),
        (321, "henicosatrict"),
    ])
    def test_hundreds_compositional(self, n, expected):
        assert get_chain_prefix(n) == expected


# ============================================================================
# Full chain names (alkane)
# ============================================================================

@pytest.mark.unit
class TestChainNames:
    """Test full alkane chain names."""

    @pytest.mark.parametrize("n,expected", [
        (1, "methane"),
        (2, "ethane"),
        (10, "decane"),
        (20, "icosane"),
        (21, "henicosane"),
        (22, "docosane"),
        (25, "pentacosane"),
        (27, "heptacosane"),
        (30, "triacontane"),
        (32, "dotriacontane"),
        (41, "hentetracontane"),
        (50, "pentacontane"),
        (53, "tripentacontane"),
        (100, "hectane"),
        (132, "dotriacontahectane"),
    ])
    def test_chain_name(self, n, expected):
        assert get_chain_name(n) == expected


# ============================================================================
# Alkyl names
# ============================================================================

@pytest.mark.unit
class TestAlkylNames:
    """Test alkyl substituent names."""

    @pytest.mark.parametrize("n,expected", [
        (1, "methyl"),
        (2, "ethyl"),
        (5, "pentyl"),
        (10, "decyl"),
        (20, "icosyl"),
        (21, "henicosyl"),
        (41, "hentetracontyl"),
        (53, "tripentacontyl"),
        (100, "hectyl"),
    ])
    def test_alkyl_name(self, n, expected):
        assert get_alkyl_name(n) == expected


# ============================================================================
# Acid names
# ============================================================================

@pytest.mark.unit
class TestAcidNames:
    """Test carboxylic acid names."""

    @pytest.mark.parametrize("n,expected", [
        (1, "methanoic acid"),
        (2, "ethanoic acid"),
        (3, "propanoic acid"),
        (10, "decanoic acid"),
        (20, "icosanoic acid"),
        (28, "octacosanoic acid"),
        (100, "hectanoic acid"),
    ])
    def test_acid_name(self, n, expected):
        assert get_acid_name(n) == expected


# ============================================================================
# Acid stem
# ============================================================================

@pytest.mark.unit
class TestAcidStem:
    """Test acid stem generation (prefix + anoic)."""

    @pytest.mark.parametrize("n,expected", [
        (2, "ethanoic"),
        (3, "propanoic"),
        (28, "octacosanoic"),
    ])
    def test_acid_stem(self, n, expected):
        assert get_acid_stem(n) == expected


# ============================================================================
# Anoate names
# ============================================================================

@pytest.mark.unit
class TestAnoateNames:
    """Test ester suffix (anoate) names."""

    @pytest.mark.parametrize("n,expected", [
        (1, "methanoate"),
        (2, "ethanoate"),
        (3, "propanoate"),
        (20, "icosanoate"),
        (100, "hectanoate"),
    ])
    def test_anoate_name(self, n, expected):
        assert get_anoate_name(n) == expected


# ============================================================================
# Edge cases and error handling
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test boundary conditions and error handling."""

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            get_chain_prefix(0)

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            get_chain_prefix(-1)

    def test_1000_raises_value_error(self):
        with pytest.raises(ValueError):
            get_chain_prefix(1000)

    def test_min_value(self):
        assert get_chain_prefix(1) == "meth"

    def test_max_value(self):
        """999 is the maximum supported chain length."""
        prefix = get_chain_prefix(999)
        assert isinstance(prefix, str)
        assert len(prefix) > 0

    def test_all_prefixes_1_to_200_are_strings(self):
        """Every prefix from 1 to 200 should be a non-empty string."""
        for n in range(1, 201):
            prefix = get_chain_prefix(n)
            assert isinstance(prefix, str), f"n={n}: not a string"
            assert len(prefix) > 0, f"n={n}: empty string"

    def test_all_alkyl_names_end_in_yl(self):
        """Every alkyl name should end in 'yl'."""
        for n in range(1, 201):
            name = get_alkyl_name(n)
            assert name.endswith("yl"), f"n={n}: {name} doesn't end in 'yl'"

    def test_all_chain_names_end_in_ane(self):
        """Every chain name should end in 'ane'."""
        for n in range(1, 201):
            name = get_chain_name(n)
            assert name.endswith("ane"), f"n={n}: {name} doesn't end in 'ane'"

    def test_all_acid_names_end_in_acid(self):
        """Every acid name should end in 'anoic acid'."""
        for n in range(1, 201):
            name = get_acid_name(n)
            assert name.endswith("anoic acid"), f"n={n}: {name} doesn't end in 'anoic acid'"

    def test_all_anoate_names_end_in_anoate(self):
        """Every anoate name should end in 'anoate'."""
        for n in range(1, 201):
            name = get_anoate_name(n)
            assert name.endswith("anoate"), f"n={n}: {name} doesn't end in 'anoate'"
