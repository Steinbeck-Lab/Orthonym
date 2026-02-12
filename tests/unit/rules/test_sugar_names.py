"""
Tests for sugar retained names lookup and glycosyloxy prefix formatting.

Tests cover:
- Sugar lookup by canonical SMILES (stereo-specific)
- N-acetyl and glucuronic acid modified sugar lookup
- Non-stereo fallback lookup
- Glycosyloxy prefix formatting for various sugar types
- Edge cases (unknown SMILES, empty inputs)
"""

import pytest

from orthonym.data.sugar_names import (
    lookup_sugar,
    sugar_to_glycosyloxy_prefix,
    SUGAR_RETAINED_NAMES,
    NACETYL_SUGAR_NAMES,
    URONIC_ACID_NAMES,
)


# ============================================================================
# Sugar lookup tests
# ============================================================================

class TestSugarLookup:
    """Tests for lookup_sugar() -- canonical SMILES to retained name."""

    def test_alpha_d_glucose_lookup(self):
        """alpha-D-glucopyranose lookup returns correct tuple."""
        result = lookup_sugar("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result == ("alpha", "D", "glucopyranose")

    def test_beta_d_glucose_lookup(self):
        """beta-D-glucopyranose lookup returns correct tuple."""
        result = lookup_sugar("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert result == ("beta", "D", "glucopyranose")

    def test_alpha_d_galactose_lookup(self):
        """alpha-D-galactopyranose lookup returns correct tuple."""
        result = lookup_sugar("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O")
        assert result == ("alpha", "D", "galactopyranose")

    def test_beta_l_rhamnose_lookup(self):
        """beta-L-rhamnopyranose lookup returns correct tuple."""
        result = lookup_sugar("C[C@@H]1O[C@H](O)[C@H](O)[C@H](O)[C@H]1O")
        assert result == ("beta", "L", "rhamnopyranose")

    def test_alpha_l_fucose_lookup(self):
        """alpha-L-fucopyranose lookup returns correct tuple."""
        result = lookup_sugar("C[C@@H]1O[C@@H](O)[C@@H](O)[C@H](O)[C@@H]1O")
        assert result == ("alpha", "L", "fucopyranose")

    def test_beta_d_ribofuranose_lookup(self):
        """beta-D-ribofuranose lookup returns correct tuple."""
        result = lookup_sugar("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O")
        assert result == ("beta", "D", "ribofuranose")

    def test_unknown_smiles_returns_none(self):
        """Non-sugar SMILES returns None."""
        result = lookup_sugar("CCCC")
        assert result is None

    def test_nacetyl_glucosamine_lookup(self):
        """alpha-D-GlcNAc lookup returns correct tuple."""
        result = lookup_sugar(
            "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O"
        )
        assert result == (
            "alpha", "D", "2-(acetylamino)-2-deoxy-glucopyranose"
        )

    def test_glucuronic_acid_lookup(self):
        """beta-D-glucuronic acid lookup returns correct tuple."""
        result = lookup_sugar(
            "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert result == ("beta", "D", "glucuronopyranose")

    def test_nonstereo_fallback(self):
        """Sugar SMILES without @ characters returns base name only."""
        # Non-stereo glucose-like: strip all stereo from glucose SMILES
        # OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O -> OCC1OC(O)C(O)C(O)C1O
        from rdkit import Chem
        stereo_smi = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        mol = Chem.MolFromSmiles(stereo_smi)
        # Remove stereo
        Chem.RemoveStereochemistry(mol)
        nonstereo = Chem.MolToSmiles(mol)
        result = lookup_sugar(nonstereo)
        assert result is not None
        # Should have empty anomer and config, but base name present
        anomer, config, base_name = result
        assert anomer == ""
        assert config == ""
        assert "glucopyranose" in base_name


# ============================================================================
# Glycosyloxy prefix tests
# ============================================================================

class TestGlycosyloxyPrefix:
    """Tests for sugar_to_glycosyloxy_prefix() formatting."""

    def test_glucose_prefix(self):
        """beta-D-glucopyranose -> beta-D-glucopyranosyloxy."""
        result = sugar_to_glycosyloxy_prefix("beta", "D", "glucopyranose")
        assert result == "beta-D-glucopyranosyloxy"

    def test_rhamnose_prefix(self):
        """alpha-L-rhamnopyranose -> alpha-L-rhamnopyranosyloxy."""
        result = sugar_to_glycosyloxy_prefix("alpha", "L", "rhamnopyranose")
        assert result == "alpha-L-rhamnopyranosyloxy"

    def test_furanose_prefix(self):
        """beta-D-ribofuranose -> beta-D-ribofuranosyloxy."""
        result = sugar_to_glycosyloxy_prefix("beta", "D", "ribofuranose")
        assert result == "beta-D-ribofuranosyloxy"

    def test_nacetyl_prefix(self):
        """Modified sugar with deoxy prefix."""
        result = sugar_to_glycosyloxy_prefix(
            "beta", "D", "2-(acetylamino)-2-deoxy-glucopyranose"
        )
        assert result == "beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy"

    def test_no_anomer_config_prefix(self):
        """Empty anomer/config gives prefix without leading hyphens."""
        result = sugar_to_glycosyloxy_prefix("", "", "glucopyranose")
        assert result == "glucopyranosyloxy"


# ============================================================================
# Data completeness tests
# ============================================================================

class TestDataCompleteness:
    """Tests for lookup table completeness."""

    def test_sugar_retained_names_count(self):
        """At least 44 entries in the main sugar lookup."""
        assert len(SUGAR_RETAINED_NAMES) >= 44

    def test_nacetyl_names_count(self):
        """At least 4 N-acetyl sugar entries."""
        assert len(NACETYL_SUGAR_NAMES) >= 4

    def test_uronic_acid_names_count(self):
        """At least 2 uronic acid entries."""
        assert len(URONIC_ACID_NAMES) >= 2

    def test_all_entries_are_tuples_of_three(self):
        """All lookup values are (anomer, config, base_name) tuples."""
        all_names = {**SUGAR_RETAINED_NAMES, **NACETYL_SUGAR_NAMES, **URONIC_ACID_NAMES}
        for smiles, value in all_names.items():
            assert isinstance(value, tuple), f"Value for {smiles} is not a tuple"
            assert len(value) == 3, f"Value for {smiles} has {len(value)} elements"
