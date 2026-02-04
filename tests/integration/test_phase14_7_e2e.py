"""
E2E integration tests for Phase 14.7: Ester Routing and Lactone Architecture Fix.

Tests cover:
1. Monocyclic lactone naming (oxolan-2-one, oxan-2-one, etc.)
2. Simple ester preservation (methyl acetate, ethyl acetate)
3. Ring-attached ester naming (acyloxy prefixes on ring parents)
4. Regression guards for existing compound classes
5. Success criteria from all three Phase 14.7 plans

Reference: IUPAC 2013 Blue Book, P-25.5.2 (Lactones), P-65.6.3.2.2 (Acyloxy prefixes)
"""

import pytest
from orthonym import name_compound


# ============================================================================
# Monocyclic Lactone Naming
# ============================================================================


class TestMonocyclicLactones:
    """Test that monocyclic lactones are named as heterocyclic ketones."""

    @pytest.mark.integration
    def test_gamma_butyrolactone(self):
        """gamma-Butyrolactone: 5-membered lactone -> oxolan-2-one."""
        result = name_compound("O=C1CCCO1")
        assert result == "oxolan-2-one", f"Expected 'oxolan-2-one', got '{result}'"

    @pytest.mark.integration
    def test_delta_valerolactone(self):
        """delta-Valerolactone: 6-membered lactone -> oxan-2-one."""
        result = name_compound("O=C1CCCCO1")
        assert result == "oxan-2-one", f"Expected 'oxan-2-one', got '{result}'"

    @pytest.mark.integration
    def test_beta_propiolactone(self):
        """beta-Propiolactone: 4-membered lactone -> oxetan-2-one."""
        result = name_compound("O=C1CCO1")
        assert result == "oxetan-2-one", f"Expected 'oxetan-2-one', got '{result}'"

    @pytest.mark.integration
    def test_epsilon_caprolactone(self):
        """epsilon-Caprolactone: 7-membered lactone -> oxepan-2-one."""
        result = name_compound("O=C1CCCCCO1")
        assert result == "oxepan-2-one", f"Expected 'oxepan-2-one', got '{result}'"

    @pytest.mark.integration
    def test_tetrahydrofuran_not_lactone(self):
        """THF has no C=O, so it is NOT a lactone - should be oxolane/tetrahydrofuran."""
        result = name_compound("C1CCCO1")
        assert "one" not in result.lower(), (
            f"THF should not have '-one' suffix: got '{result}'"
        )

    @pytest.mark.integration
    def test_oxetane_not_lactone(self):
        """Oxetane has no C=O, so it is NOT a lactone."""
        result = name_compound("C1CCO1")
        assert "one" not in result.lower(), (
            f"Oxetane should not have '-one' suffix: got '{result}'"
        )


# ============================================================================
# Simple Ester Preservation
# ============================================================================


class TestSimpleEstersPreserved:
    """Test that simple acyclic esters continue to use 'alkyl alkanoate' naming."""

    @pytest.mark.integration
    def test_methyl_acetate(self):
        result = name_compound("COC(C)=O")
        assert result == "methyl acetate", f"Expected 'methyl acetate', got '{result}'"

    @pytest.mark.integration
    def test_ethyl_acetate(self):
        result = name_compound("CCOC(C)=O")
        assert result == "ethyl acetate", f"Expected 'ethyl acetate', got '{result}'"

    @pytest.mark.integration
    def test_methyl_formate(self):
        result = name_compound("COC=O")
        assert result == "methyl formate", f"Expected 'methyl formate', got '{result}'"

    @pytest.mark.integration
    def test_methyl_propanoate(self):
        result = name_compound("COC(=O)CC")
        assert result == "methyl propanoate", f"Expected 'methyl propanoate', got '{result}'"

    @pytest.mark.integration
    def test_ethyl_butanoate(self):
        result = name_compound("CCOC(=O)CCC")
        assert result == "ethyl butanoate", f"Expected 'ethyl butanoate', got '{result}'"


# ============================================================================
# Ring-Attached Ester Naming (Acyloxy Prefixes)
# ============================================================================


class TestRingAttachedEsters:
    """Test that esters on ring systems use acyloxy prefix naming."""

    @pytest.mark.integration
    def test_cyclohexyl_acetate_not_hexyl(self):
        """Cyclohexyl acetate should NOT produce 'hexyl acetate'."""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert "hexyl acetate" not in result.lower(), (
            f"Should not produce 'hexyl acetate': got '{result}'"
        )
        assert "acetyloxy" in result.lower(), (
            f"Should contain 'acetyloxy' prefix: got '{result}'"
        )
        assert "cyclohexane" in result.lower(), (
            f"Should contain 'cyclohexane' parent: got '{result}'"
        )

    @pytest.mark.integration
    def test_phenyl_acetate(self):
        """Phenyl acetate should use acyloxy prefix on benzene parent."""
        result = name_compound("CC(=O)Oc1ccccc1")
        assert "hexyl" not in result.lower(), (
            f"Should not contain 'hexyl': got '{result}'"
        )
        assert "acetyloxy" in result.lower(), (
            f"Should contain 'acetyloxy' prefix: got '{result}'"
        )

    @pytest.mark.integration
    def test_cyclohexyl_acetate_full_name(self):
        """Full expected name for cyclohexyl acetate."""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert result == "acetyloxycyclohexane", (
            f"Expected 'acetyloxycyclohexane', got '{result}'"
        )

    @pytest.mark.integration
    def test_phenyl_acetate_full_name(self):
        """Full expected name for phenyl acetate with benzene parent."""
        result = name_compound("CC(=O)Oc1ccccc1")
        assert result == "acetyloxybenzene", (
            f"Expected 'acetyloxybenzene', got '{result}'"
        )


# ============================================================================
# Existing Compounds Preserved (Regression Guards)
# ============================================================================


class TestExistingCompoundsPreserved:
    """Regression guards: existing compound naming must not change."""

    @pytest.mark.integration
    def test_benzene(self):
        assert name_compound("c1ccccc1") == "benzene"

    @pytest.mark.integration
    def test_cyclohexane(self):
        assert name_compound("C1CCCCC1") == "cyclohexane"

    @pytest.mark.integration
    def test_ethanol(self):
        result = name_compound("CCO")
        assert result in ("ethanol", "ethan-1-ol"), f"Got '{result}'"

    @pytest.mark.integration
    def test_acetic_acid(self):
        assert name_compound("CC(=O)O") == "acetic acid"

    @pytest.mark.integration
    def test_benzoic_acid(self):
        assert name_compound("OC(=O)c1ccccc1") == "benzoic acid"

    @pytest.mark.integration
    def test_coumarin(self):
        """Coumarin should use retained name."""
        result = name_compound("O=c1ccc2ccccc2o1")
        assert result == "coumarin" or "coumarin" in result.lower(), (
            f"Expected 'coumarin', got '{result}'"
        )

    @pytest.mark.integration
    def test_bicyclo_octane(self):
        result = name_compound("C1CC2CCCC1C2")
        assert "bicyclo" in result.lower(), f"Expected bicyclo name, got '{result}'"

    @pytest.mark.integration
    def test_adamantane(self):
        """Adamantane: pre-existing naming (not a retained name in current data)."""
        result = name_compound("C1C2CC3CC1CC(C2)C3")
        # Adamantane is a complex polycyclic - verify it doesn't crash
        assert result is not None and len(result) > 0

    @pytest.mark.integration
    def test_pyridine(self):
        assert name_compound("c1ccncc1") == "pyridine"

    @pytest.mark.integration
    def test_morpholine(self):
        assert name_compound("C1COCCN1") == "morpholine"


# ============================================================================
# Success Criteria (Direct Tests of Plan Truth Statements)
# ============================================================================


class TestSuccessCriteria:
    """Tests mapping directly to must_haves truths from Phase 14.7 plans."""

    @pytest.mark.integration
    def test_truth_gamma_butyrolactone_is_oxolan_2_one(self):
        """Truth: name_compound('O=C1CCCO1') returns 'oxolan-2-one'."""
        assert name_compound("O=C1CCCO1") == "oxolan-2-one"

    @pytest.mark.integration
    def test_truth_delta_valerolactone_is_oxan_2_one(self):
        """Truth: name_compound('O=C1CCCCO1') returns 'oxan-2-one'."""
        assert name_compound("O=C1CCCCO1") == "oxan-2-one"

    @pytest.mark.integration
    def test_truth_cyclohexyl_acetate_uses_ring_parent(self):
        """Truth: CC(=O)OC1CCCCC1 names correctly with ring parent (not 'hexyl acetate')."""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert "hexyl acetate" not in result, f"Got '{result}'"

    @pytest.mark.integration
    def test_truth_methyl_acetate_preserved(self):
        """Truth: COC(C)=O still returns 'methyl acetate'."""
        assert name_compound("COC(C)=O") == "methyl acetate"

    @pytest.mark.integration
    def test_truth_ethyl_acetate_preserved(self):
        """Truth: CCOC(C)=O still returns 'ethyl acetate'."""
        assert name_compound("CCOC(C)=O") == "ethyl acetate"

    @pytest.mark.integration
    def test_truth_esters_on_complex_rings_use_acyloxy(self):
        """Truth: Esters on complex ring systems use acyloxy prefix naming."""
        result = name_compound("CC(=O)OC1CCCCC1")
        assert "acetyloxy" in result, f"Expected acyloxy prefix, got '{result}'"

    @pytest.mark.integration
    def test_truth_zero_regressions(self):
        """Truth: All core compounds still name correctly."""
        # Spot check a few key compounds across different classes
        assert name_compound("C") == "methane"
        assert name_compound("CC") == "ethane"
        assert name_compound("CCC") == "propane"
        assert name_compound("CCO") in ("ethanol", "ethan-1-ol")
        assert name_compound("CC=O") in ("acetaldehyde", "ethanal")
        assert name_compound("CC(=O)O") == "acetic acid"
        assert name_compound("c1ccccc1") == "benzene"
        assert name_compound("C1CCCCC1") == "cyclohexane"

    @pytest.mark.integration
    def test_truth_non_lactone_heterocycles_unchanged(self):
        """Non-lactone heterocycles (THF, oxetane) must remain unchanged."""
        thf = name_compound("C1CCCO1")
        assert "one" not in thf, f"THF got -one suffix: '{thf}'"

        oxetane = name_compound("C1CCO1")
        assert "one" not in oxetane, f"Oxetane got -one suffix: '{oxetane}'"
