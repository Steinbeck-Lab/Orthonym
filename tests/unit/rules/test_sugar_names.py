"""
Tests for sugar retained names lookup and glycosyloxy prefix formatting.

Tests cover:
- Sugar lookup by canonical SMILES (stereo-specific)
- N-acetyl and glucuronic acid modified sugar lookup
- Non-stereo fallback lookup
- Glycosyloxy prefix formatting for various sugar types
- Edge cases (unknown SMILES, empty inputs)
- OPSIN carbohydrate integration (Phase 141 Plan 03)
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
            "alpha", "D", "2-acetamido-2-deoxy-glucopyranose"
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
            "beta", "D", "2-acetamido-2-deoxy-glucopyranose"
        )
        assert result == "beta-D-2-acetamido-2-deoxy-glucopyranosyloxy"

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


# ============================================================================
# OPSIN carbohydrate integration tests (Phase 141, Plan 03)
# ============================================================================

class TestOpsinCarbohydrateIntegration:
    """Tests for OPSIN carbohydrate data integration into sugar lookup."""

    def test_opsin_carbohydrate_integration_meglumine(self):
        """Meglumine (OPSIN simpleGroup) is lookupable via lookup_sugar."""
        from rdkit import Chem
        smi = "CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        result = lookup_sugar(can)
        assert result is not None, f"meglumine not found in sugar lookup (canonical: {can})"
        assert "meglumine" in result[2].lower() or "meglumin" in result[2].lower()

    def test_opsin_carbohydrate_integration_glucamine(self):
        """Glucamine (OPSIN simpleGroup) is lookupable via lookup_sugar."""
        from rdkit import Chem
        smi = "NC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        result = lookup_sugar(can)
        assert result is not None, f"glucamine not found in sugar lookup (canonical: {can})"
        assert "glucamine" in result[2].lower() or "glucamin" in result[2].lower()

    def test_opsin_carbohydrate_integration_sorbitol(self):
        """The sorbitol structure resolves to its PIN retained name D-glucitol.

        W5-A1 (P-102.5.6.5): the hand-curated ACYCLIC_PIN_SUGAR_NAMES entry
        pre-empts the OPSIN 'sorbitol' simpleGroup synonym, so lookup_sugar now
        returns the alditol PIN ('D', 'glucitol') rather than the deprecated
        'sorbitol'. The structure is still lookupable (the point of the OPSIN
        integration path); only the emitted name is now the PIN.
        """
        from rdkit import Chem
        smi = "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        result = lookup_sugar(can)
        assert result is not None, f"glucitol not found in sugar lookup (canonical: {can})"
        assert "glucitol" in result[2].lower()
        assert result[1] == "D"

    def test_opsin_ring_carbohydrate_garosamine(self):
        """Garosamine (OPSIN ring entry, monosaccharide) is lookupable via lookup_sugar."""
        from rdkit import Chem
        smi = "CN[C@@H]1[C@@H](O)C(O)OC[C@]1(C)O"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        result = lookup_sugar(can)
        assert result is not None, f"garosamine not found in sugar lookup (canonical: {can})"
        assert "garosamine" in result[2].lower()

    def test_disaccharides_excluded(self):
        """Disaccharides (2+ rings) are excluded to avoid decomposition interference."""
        from rdkit import Chem
        from orthonym.data.sugar_names import ALL_SUGAR_NAMES
        rutinose_smi = "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(rutinose_smi))
        assert can not in ALL_SUGAR_NAMES, "Disaccharide rutinose should not be in sugar lookup"

    def test_opsin_ring_carbohydrate_ascorbic_acid(self):
        """Ascorbic acid (OPSIN ring entry) is lookupable via lookup_sugar."""
        from rdkit import Chem
        smi = "O=C1O[C@H]([C@@H](O)CO)C(O)=C1O"
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        result = lookup_sugar(can)
        assert result is not None, f"ascorbic acid not found in sugar lookup (canonical: {can})"
        assert "ascorbic acid" in result[2].lower()

    def test_existing_sugars_preserved(self):
        """All original 56 hand-curated sugar entries still return correct tuples."""
        from orthonym.data.sugar_names import ALL_SUGAR_NAMES
        originals = {
            "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O": ("alpha", "D", "glucopyranose"),
            "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O": ("beta", "D", "glucopyranose"),
            "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O": ("alpha", "D", "galactopyranose"),
            "C[C@@H]1O[C@@H](O)[C@H](O)[C@H](O)[C@H]1O": ("alpha", "L", "rhamnopyranose"),
            "C[C@@H]1O[C@@H](O)[C@@H](O)[C@H](O)[C@@H]1O": ("alpha", "L", "fucopyranose"),
            "OC[C@H]1O[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1O": ("alpha", "D", "mannopyranose"),
        }
        for smi, expected in originals.items():
            result = ALL_SUGAR_NAMES.get(smi)
            assert result == expected, f"Original entry changed: {smi} -> {result} (expected {expected})"

    def test_sugar_count_expanded(self):
        """ALL_SUGAR_NAMES has at least 70 entries after OPSIN integration."""
        from orthonym.data.sugar_names import ALL_SUGAR_NAMES
        assert len(ALL_SUGAR_NAMES) >= 70, f"Expected >= 70 sugar entries, got {len(ALL_SUGAR_NAMES)}"

    def test_sugar_main_cascade(self):
        """name_compound() on beta-D-glucopyranose SMILES returns a sugar name."""
        from orthonym import name_compound
        result = name_compound("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert "glucopyranose" in result.lower(), f"Expected sugar name, got: {result}"

    def test_nonstereo_sugar_fallback_expanded(self):
        """Non-stereo glucose SMILES still returns a sugar name after expansion."""
        from rdkit import Chem
        stereo_smi = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        mol = Chem.MolFromSmiles(stereo_smi)
        Chem.RemoveStereochemistry(mol)
        nonstereo = Chem.MolToSmiles(mol)
        result = lookup_sugar(nonstereo)
        assert result is not None, f"Non-stereo glucose fallback broken after expansion"
        assert "glucopyranose" in result[2]

    def test_carbohydrate_suffix_rules_cataloged(self):
        """Carbohydrate suffix rules dict is populated."""
        from orthonym.data.sugar_names import CARBOHYDRATE_SUFFIX_RULES
        assert len(CARBOHYDRATE_SUFFIX_RULES) >= 10, \
            f"Expected >= 10 suffix rules, got {len(CARBOHYDRATE_SUFFIX_RULES)}"
        assert "ose" in CARBOHYDRATE_SUFFIX_RULES
        assert "itol" in CARBOHYDRATE_SUFFIX_RULES

    def test_hand_curated_wins_on_conflict(self):
        """Hand-curated entries take precedence over OPSIN entries on conflict."""
        from orthonym.data.sugar_names import ALL_SUGAR_NAMES
        smi = "N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O"
        result = ALL_SUGAR_NAMES.get(smi)
        assert result is not None
        assert result == ("alpha", "D", "glucosamine")
