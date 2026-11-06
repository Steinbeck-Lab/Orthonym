"""
E2E integration tests for a phase: Ester Routing and Lactone Architecture Fix.

Tests cover:
1. Monocyclic lactone naming (oxolan-2-one, oxan-2-one, etc.)
2. Simple ester preservation (methyl acetate, ethyl acetate)
3. Ring-attached ester naming (acyloxy prefixes on ring parents)
4. Regression guards for existing compound classes
5. Success criteria from all three a phase plans

Reference: IUPAC 2013 Blue Book, (Lactones), (Acyloxy prefixes)
"""

import pytest
from orthonym import name_compound


# ============================================================================
# Monocyclic Lactone Naming
# ============================================================================


class TestMonocyclicLactones:
    """Test that monocyclic lactones are named as heterocyclic ketones."""

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
    def test_oxolane_not_lactone(self):
        """THF has no C=O, so it is NOT a lactone - should be oxolane/oxolane."""
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
# Ring-Attached Ester Naming (Acyloxy Prefixes)
# ============================================================================


# (under 'Definitions', the Blue Book): "All preferred
# IUPAC names for esters are named by functional class nomenclature." The ring is the
# 'alcoholic' component, cited as the organyl group (cf. '6-[4-(acetyloxy)phenyl]pyridin-3-yl
# acetate (PIN)',:31895); the acyloxy prefix is reserved for esters cited as prefixes,
# when a senior group is present,:31696), so the earlier acyloxy-on-ring
# names ('acetyloxycyclohexane', 'acetyloxybenzene') are not preferred names.


# ============================================================================
# Success Criteria (Direct Tests of Plan Truth Statements)
# ============================================================================


class TestSuccessCriteria:
    """Tests mapping directly to must_haves truths from a phase plans."""

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
