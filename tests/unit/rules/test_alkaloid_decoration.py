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
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.natural_products import name_natural_product


def _p101(smiles):
    """The name of the natural-product producer. The engine itself names the
    4,5-epoxymorphinans by their bridged fused PIN (slice S4, the Blue Book),
    so the decoration enumeration is tested on the producer, which the natural-product
    route still uses wherever no systematic PIN is built."""
    return name_natural_product(Chem.MolFromSmiles(smiles))


# ============================================================================
# E2E: SMILES -> decorated morphinan name
# ============================================================================


class TestMorphinanE2E:
    """End-to-end tests: SMILES -> decorated morphinan IUPAC name."""

    def test_morphine_natural_exact_lookup(self):
        """Natural (-)-morphine hits exact derivative lookup in the producer; the engine
        names it by its bridged fused PIN (slice S4)."""
        smiles = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        assert _p101(smiles) == "morphine"
        assert name_compound(smiles) == "(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano[1]benzofuro[3,2-e]isoquinoline-7,9-diol"

    def test_morphine_enantiomer_scaffold_decoration(self):
        """(+)-morphine hits scaffold detection and gets full decorations."""
        result = _p101(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        # Expect: epoxy, N-methyl, unsaturation, hydroxyls
        assert "4,5-epoxy" in result
        assert "17-methyl" in result
        assert "morphin" in result
        # fix a performance pass (wp6-tests), change-asserted-value (was '-7-en-'): the 'e'
        # of 'ene' is elided only before a vowel, (a) (the Blue Book),
        # and '-diol' begins with 'd': 'undeca-2,9-diene-4,8-diol (PIN)' (:3403). Full
        # name '(5R,6S,9S,13S,14S)-4,5-epoxy-17-methylmorphin-7-ene-3,6-diol', OPSIN
        # 2.9.0 full-InChIKey exact (an InChIKey).
        assert "-7-ene-3,6-diol" in result
        assert "3,6-diol" in result

    def test_codeine_decoration(self):
        """Codeine: 3-methoxy instead of 3-OH, otherwise like morphine."""
        result = _p101(
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
        result = _p101(
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
        result = _p101(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        assert "4,5-epoxy" in result

    def test_n_methyl_detected(self):
        """N-methyl at position 17 correctly detected."""
        result = _p101(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"
        )
        assert "17-methyl" in result

    def test_no_duplicate_methyl(self):
        """N-methyl should not appear twice (no C-methyl + N-methyl duplication)."""
        result = _p101(
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
        """ci-008: the 16,22-epoxy bridge makes the cholestane skeleton furostan.

         fix a performance pass (wp6-tests), change-asserted-value (was '16,22-epoxy' +
        'cholestane' in the name): the ring-E ether is the Table 10.1 stereoparent
        'furostan', (the Blue Book "Semisystematic names of
        recommended parent structures are listed in Table 10.1"; furostan at:51409),
        kept at the PIN tier by the controller ruling on stereoparents
        (TRIAGE.md 'Controller rulings'); OPSIN 2.9.0 full-InChIKey exact
        (an InChIKey). Open (recorded in internal notes
        wp6): the input leaves C-5 undefined, and (:51047) says "with a
        steroid the stereochemistry at 'C-5', when relevant, is indicated by α, β or
        ξ", i.e. '5ξ-furostan'; OPSIN 2.9.0 reads the bare name with C-5 undefined, so
        the round trip cannot tell the two apart."""
        result = name_compound(
            "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)"
            "[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C"
        )
        assert result == "furostan", result

    @pytest.mark.opsin_gate
    def test_steroid_ester_not_methoxy(self):
        """Methyl ester on steroid side chain should NOT be detected as methoxy.

        Task 4 continuation (2026-09-25): the PIN tier used to pass this with the
        NP name '(3R,...,20R)-3,7,15-trihydroxycholan-24-one', which dropped the
        ester's O-methyl (OPSIN: C24H40O4; the input is C25H42O5).
        (the Blue Book): every substituent is cited. The NP producer now
        declines, and no NP-parent name is a PIN,:50943), so the PIN tier
        fails closed (it already did in production). The tier contract holds
        with the gate ON, and the test's intent is checked on the best-effort
        name, which cites the ester as 'methyl...pentanoate', never 'methoxy'.
        """
        from tests.support.rt_assert import assert_tier_contract
        # m16_HA30: cholane with methyl ester at C-24
        _pin, result = assert_tier_contract(
            "COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3"
            "[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C"
        )
        assert "methoxy" not in result
        assert "trihydroxy" in result
        assert result.startswith("methyl ")
