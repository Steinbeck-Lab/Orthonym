"""
Tests for amide prefix forms (carbamoyl) in non-principal group context.

IUPAC 2013 P-66.1.1.4 method 2: When a primary amide (-CONH2) is not the
principal group, it is expressed as the prefix "carbamoyl".

Note: Secondary and tertiary amides (N-acyl bonds) are already handled
via the acylamino naming pathway in the pipeline. Adding carbamoyl for
sec/tert would cause double-naming (e.g., "ethanoylamino" + "carbamoyl").
Only primary_amide gets the carbamoyl prefix entry.

Covers:
- Unit tests for get_prefix() returning carbamoyl for primary_amide
- Unit tests confirming secondary/tertiary amide prefixes remain None
- E2E regression tests for compounds with primary amide as non-principal FG
- OPSIN round-trip validation for carbamoyl-containing names
- Double-counting guard (single amide -> exactly one carbamoyl)
- Negative test (amide as principal -> suffix, no carbamoyl prefix)
- Regression guard: N-acyl amides must NOT produce spurious carbamoyl
"""

import pytest
from orthonym import name_compound
from orthonym.rules.seniority import get_prefix


# ============================================================================
# Unit Tests: get_prefix behavior for amide types
# ============================================================================

class TestAmidePrefixUnit:
    """Verify PREFIX_FORMS dict entries for amide types."""

    def test_primary_amide_prefix_is_carbamoyl(self):
        """Primary amide (-CONH2) as non-principal FG gets carbamoyl prefix."""
        assert get_prefix("primary_amide") == "carbamoyl"

    def test_secondary_amide_prefix_is_none(self):
        """Secondary amide prefix remains None; handled by acylamino pathway."""
        assert get_prefix("secondary_amide") is None

    def test_tertiary_amide_prefix_is_none(self):
        """Tertiary amide prefix remains None; handled by acylamino pathway."""
        assert get_prefix("tertiary_amide") is None


# ============================================================================
# E2E Regression Tests: Compounds with primary amide as non-principal group
# ============================================================================

class TestCarbamoylPrefixE2E:
    """End-to-end tests: primary amide as non-principal FG produces carbamoyl."""

    def test_4_carbamoylbutanoic_acid(self):
        """NC(=O)CCC(=O)O -> 4-carbamoylbutanoic acid"""
        result = name_compound("NC(=O)CCC(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_3_carbamoylpropanoic_acid(self):
        """NC(=O)CC(=O)O -> 3-carbamoylpropanoic acid"""
        result = name_compound("NC(=O)CC(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_5_carbamoylpentanoic_acid(self):
        """NC(=O)CCCC(=O)O -> 5-carbamoylpentanoic acid"""
        result = name_compound("NC(=O)CCCC(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_6_carbamoylhexanoic_acid(self):
        """NC(=O)CCCCC(=O)O -> 6-carbamoylhexanoic acid"""
        result = name_compound("NC(=O)CCCCC(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_4_carbamoylbenzoic_acid(self):
        """OC(=O)c1ccc(C(N)=O)cc1 -> 4-carbamoylbenzoic acid"""
        result = name_compound("OC(=O)c1ccc(C(N)=O)cc1")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_2_carbamoylbenzoic_acid(self):
        """NC(=O)c1ccccc1C(=O)O -> 2-carbamoylbenzoic acid"""
        result = name_compound("NC(=O)c1ccccc1C(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_3_carbamoylbenzoic_acid(self):
        """NC(=O)c1cccc(C(=O)O)c1 -> 3-carbamoylbenzoic acid"""
        result = name_compound("NC(=O)c1cccc(C(=O)O)c1")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_2_carbamoylacetic_acid(self):
        """NC(=O)C(=O)O -> carbamoyl on short chain acid"""
        result = name_compound("NC(=O)C(=O)O")
        assert result is not None
        # This is oxamic acid (2-amino-2-oxoacetic acid or carbamoylformic acid)
        # Either systematic name or retained name is acceptable

    def test_carbamoyl_on_pyridine_acid(self):
        """NC(=O)c1ccncc1C(=O)O -> carbamoyl on pyridine with acid"""
        result = name_compound("NC(=O)c1ccncc1C(=O)O")
        assert result is not None
        assert "carbamoyl" in result

    def test_4_carbamoylcyclohexanecarboxylic_acid(self):
        """NC(=O)C1CCC(C(=O)O)CC1 -> carbamoyl on cyclohexane acid"""
        result = name_compound("NC(=O)C1CCC(C(=O)O)CC1")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result

    def test_carbamoylpentanedioic_acid(self):
        """NC(=O)CC(C(=O)O)CC(=O)O -> carbamoyl on diacid chain"""
        result = name_compound("NC(=O)CC(C(=O)O)CC(=O)O")
        assert result is not None
        assert "carbamoyl" in result

    def test_carbamoyl_with_hydroxy(self):
        """NC(=O)CC(O)C(=O)O -> carbamoyl + hydroxy on acid chain"""
        result = name_compound("NC(=O)CC(O)C(=O)O")
        assert result is not None
        assert "carbamoyl" in result
        assert "acid" in result


# ============================================================================
# OPSIN Round-Trip Tests
# ============================================================================

class TestCarbamoylRoundTrip:
    """OPSIN parse round-trip for carbamoyl-containing names."""

    @pytest.mark.roundtrip
    def test_4_carbamoylbutanoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)CCC(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_3_carbamoylpropanoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)CC(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_5_carbamoylpentanoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)CCCC(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_6_carbamoylhexanoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)CCCCC(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_4_carbamoylbenzoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "OC(=O)c1ccc(C(N)=O)cc1"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_2_carbamoylbenzoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)c1ccccc1C(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_3_carbamoylbenzoic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)c1cccc(C(=O)O)c1"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_carbamoyl_pyridine_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)c1ccncc1C(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_4_carbamoylcyclohexanecarboxylic_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)C1CCC(C(=O)O)CC1"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)

    @pytest.mark.roundtrip
    def test_carbamoyl_hydroxy_acid_roundtrip(self, opsin_to_smiles, canonical):
        smiles = "NC(=O)CC(O)C(=O)O"
        name = name_compound(smiles)
        assert name is not None
        assert "carbamoyl" in name
        parsed = opsin_to_smiles(name)
        if parsed:
            assert canonical(parsed) == canonical(smiles)


# ============================================================================
# Double-Counting Guard
# ============================================================================

class TestCarbamoylDoubleCounting:
    """Ensure single primary amide produces exactly one carbamoyl in name."""

    def test_single_amide_single_carbamoyl(self):
        """A compound with one primary amide group should have exactly one carbamoyl."""
        result = name_compound("NC(=O)CCC(=O)O")
        assert result is not None
        assert result.count("carbamoyl") == 1, (
            f"Expected exactly 1 'carbamoyl' in name, got {result.count('carbamoyl')}: {result}"
        )

    def test_single_amide_on_ring_single_carbamoyl(self):
        """A compound with one primary amide on a ring should have exactly one carbamoyl."""
        result = name_compound("OC(=O)c1ccc(C(N)=O)cc1")
        assert result is not None
        assert result.count("carbamoyl") == 1, (
            f"Expected exactly 1 'carbamoyl' in name, got {result.count('carbamoyl')}: {result}"
        )


# ============================================================================
# Negative Test: Amide as Principal Group (no carbamoyl prefix)
# ============================================================================

class TestAmideAsPrincipalGroup:
    """When amide IS the principal group, carbamoyl prefix should NOT appear."""

    def test_acetamide_no_carbamoyl(self):
        """CC(=O)N -> acetamide (amide is suffix, not prefix)"""
        result = name_compound("CC(=O)N")
        assert result is not None
        assert "carbamoyl" not in result, (
            f"Amide as principal group should not produce carbamoyl prefix: {result}"
        )

    def test_propanamide_no_carbamoyl(self):
        """CCC(=O)N -> propanamide (amide is suffix, not prefix)"""
        result = name_compound("CCC(=O)N")
        assert result is not None
        assert "carbamoyl" not in result, (
            f"Amide as principal group should not produce carbamoyl prefix: {result}"
        )

    def test_benzamide_no_carbamoyl(self):
        """NC(=O)c1ccccc1 -> benzamide (amide is suffix, not prefix)"""
        result = name_compound("NC(=O)c1ccccc1")
        assert result is not None
        assert "carbamoyl" not in result, (
            f"Amide as principal group should not produce carbamoyl prefix: {result}"
        )


# ============================================================================
# Regression Guard: N-acyl amides must NOT get spurious carbamoyl
# ============================================================================

class TestNAcylAmideNoCarbamoyl:
    """N-acyl amides (secondary/tertiary) handled by acylamino pathway.

    These compounds must NOT produce carbamoyl prefix, which would be
    double-naming the same structural feature.
    """

    def test_n_acetylglycine_no_carbamoyl(self):
        """CC(=O)NCC(=O)O -> 2-acetamidoethanoic acid, no carbamoyl (P-66.1.1.4.3)."""
        result = name_compound("CC(=O)NCC(=O)O")
        assert result is not None
        assert "carbamoyl" not in result, (
            f"N-acyl amide should not produce carbamoyl prefix: {result}"
        )

    def test_n_octadecanoyl_serine_no_carbamoyl(self):
        """Long N-acyl amide should not produce spurious carbamoyl."""
        result = name_compound("CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)C(=O)O")
        assert result is not None
        assert "carbamoyl" not in result, (
            f"N-acyl amide should not produce carbamoyl prefix: {result}"
        )
