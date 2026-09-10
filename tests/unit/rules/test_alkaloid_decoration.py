"""Tests for alkaloid decoration enumeration (IUPAC natural product nomenclature).

When a morphinan-class alkaloid is detected via scaffold matching,
the system should enumerate its decorations (epoxy bridges, hydroxyls,
methoxys, unsaturation, N-alkyls) using the morphinan numbering system
(positions 1-17, N at 17).

References:
    IUPAC 2013 Blue Book (natural product nomenclature)
    WHO INN numbering for morphinan skeleton
"""

import pytest
from orthonym.namer import name_compound


# ============================================================================
# E2E: SMILES -> decorated morphinan name
# ============================================================================


class TestMorphinanE2E:
    """End-to-end tests: SMILES -> decorated morphinan IUPAC name."""

    def test_morphine_natural_exact_lookup(self):
        """Natural (-)-morphine hits exact derivative lookup."""
        result = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        )
        assert result == "morphine"

    def test_morphine_enantiomer_scaffold_decoration(self):
        """(+)-morphine hits scaffold detection and gets full decorations."""
        result = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        # Expect: epoxy, N-methyl, unsaturation, hydroxyls
        assert "4,5-epoxy" in result
        assert "17-methyl" in result
        assert "morphin" in result
        assert "-7-en-" in result
        assert "3,6-diol" in result

    def test_codeine_decoration(self):
        """Codeine: 3-methoxy instead of 3-OH, otherwise like morphine."""
        result = name_compound(
            "COc1ccc2C[C@H]3[C@@H]4C=C[C@H](O)[C@@H]5Oc1c2[C@]45CCN3C"
        )
        assert "4,5-epoxy" in result
        assert "3-methoxy" in result
        assert "17-methyl" in result
        assert "-7-en-" in result
        assert "6-ol" in result
        # Should NOT have 3-hydroxy (codeine has methoxy at 3)
        assert "3-hydroxy" not in result
        assert "3,6-diol" not in result

    def test_thebaine_decoration(self):
        """Thebaine: 3,6-dimethoxy, no hydroxyls."""
        result = name_compound(
            "COc1ccc2c3c1O[C@H]1[C@@H](OC)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"
        )
        assert "4,5-epoxy" in result
        assert "3,6-dimethoxy" in result
        assert "17-methyl" in result
        assert "morphin-7-ene" in result
        # No hydroxyls in thebaine
        assert "ol" not in result.split("morphin")[1]  # No -ol suffix


# ============================================================================
# Negative / guard tests
# ============================================================================


class TestAlkaloidNegative:
    """Tests that non-alkaloid compounds are unaffected."""

    def test_bare_morphinan_no_decorations(self):
        """Morphinan skeleton without decorations returns bare name."""
        result = name_compound("c1ccc2c(c1)CC1NCCC23CCCCC13")
        assert result == "morphinan"

    def test_cholesterol_unchanged(self):
        """Cholesterol should still be named correctly."""
        result = name_compound(
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4"
            "C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        assert result == "cholesterol"

    def test_steroid_decoration_unchanged(self):
        """Steroid with decorations should still produce correct names."""
        # Testosterone: androst-4-en-17-ol-3-one
        result = name_compound(
            "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2O"
        )
        assert "androst" in result or "androstan" in result


# ============================================================================
# Specific decoration detection tests
# ============================================================================


class TestDecorationDetection:
    """Tests for individual decoration types."""

    def test_epoxy_bridge_detected(self):
        """Epoxy bridge correctly detected as 4,5-epoxy."""
        result = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        assert "4,5-epoxy" in result

    def test_n_methyl_detected(self):
        """N-methyl at position 17 correctly detected."""
        result = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        assert "17-methyl" in result

    def test_no_duplicate_methyl(self):
        """N-methyl should not appear twice (no C-methyl + N-methyl duplication)."""
        result = name_compound(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        # Count occurrences of "methyl" in the name
        methyl_count = result.lower().count("methyl")
        assert methyl_count == 1, f"Expected 1 'methyl', found {methyl_count} in '{result}'"


# ============================================================================
# CI regression guards
# ============================================================================


class TestAlkaloidCIRegression:
    """Guard tests for compounds that must not regress."""

    def test_ci008_epoxycholestane(self):
        """ci-008: 16,22-epoxy bridge detected in cholestane derivative."""
        result = name_compound(
            "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)"
            "[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C"
        )
        assert "16,22-epoxy" in result
        assert "cholestane" in result

    def test_steroid_ester_not_methoxy(self):
        """Methyl ester on steroid side chain should NOT be detected as methoxy."""
        # m16_HA30: cholane with methyl ester at C-24
        result = name_compound(
            "COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3"
            "[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C"
        )
        assert "methoxy" not in result
        assert "trihydroxy" in result
