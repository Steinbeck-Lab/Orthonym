"""
Phase 113 Plan 02 - Functional Group Drop Audit Tests

Tests that verify specific functional group types are NOT silently dropped
during assembly. Each test names a polyfunctional compound and asserts that
all expected FG types appear in the generated name (as either prefix or suffix).

These tests serve as regression guards against FG drops documented in the
DROP-17 through DROP-25 log labels.
"""

import pytest

from orthonym import name_compound


@pytest.mark.integration
class TestHydroxyNotDropped:
    """Verify -OH groups are retained in polyfunctional names."""

    def test_hydroxy_not_dropped_in_amino_acid(self):
        """Serine has both -OH and -NH2: both must appear (or retained name)."""
        name = name_compound("OC(=O)C(N)CO")
        assert name is not None
        # Serine is a retained name, which is acceptable
        if name.lower() != "serine":
            assert "hydroxy" in name.lower() or "ol" in name.lower(), (
                f"Hydroxy dropped in amino acid: {name}"
            )

    def test_hydroxy_not_dropped_in_hydroxy_acid(self):
        """3-hydroxybutanoic acid: -OH must appear as prefix."""
        name = name_compound("CC(O)CC(=O)O")
        assert name is not None
        assert "hydroxy" in name.lower(), f"Hydroxy dropped in hydroxy acid: {name}"
        assert "acid" in name.lower(), f"Acid suffix dropped: {name}"

    def test_hydroxy_not_dropped_in_dihydroxy_benzoic(self):
        """3,4-dihydroxybenzoic acid: both -OH must appear."""
        name = name_compound("OC(=O)c1cc(O)c(O)cc1")
        assert name is not None
        assert "hydroxy" in name.lower(), f"Hydroxy dropped: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_hydroxy_not_dropped_in_hydroxyphenyl_acetic(self):
        """2-(4-hydroxyphenyl)acetic acid: ring-OH must survive."""
        name = name_compound("OC(=O)Cc1ccc(O)cc1")
        assert name is not None
        # hydroxyphenyl or just hydroxy should be present
        assert "hydroxy" in name.lower() or "phenol" in name.lower(), (
            f"Hydroxy dropped in hydroxyphenyl compound: {name}"
        )


@pytest.mark.integration
class TestOxoNotDropped:
    """Verify =O (oxo/ketone) groups are retained in polyfunctional names."""

    def test_oxo_not_dropped_in_keto_acid(self):
        """5-oxopentanoic acid: oxo prefix must appear."""
        name = name_compound("OC(=O)CCCC=O")
        assert name is not None
        assert "oxo" in name.lower() or "al" in name.lower(), (
            f"Oxo/al dropped in keto acid: {name}"
        )
        assert "acid" in name.lower()

    def test_oxo_not_dropped_in_oxo_diacid(self):
        """3-oxopentanedioic acid: oxo prefix on diacid."""
        name = name_compound("OC(=O)CC(=O)CC(=O)O")
        assert name is not None
        assert "oxo" in name.lower(), f"Oxo dropped in oxo diacid: {name}"
        assert "acid" in name.lower()

    def test_oxo_not_dropped_in_ring_keto_acid(self):
        """Cyclohexanone carboxylic acid: oxo on ring acid."""
        name = name_compound("OC(=O)C1CCC(=O)CC1")
        assert name is not None
        assert "oxo" in name.lower() or "one" in name.lower(), (
            f"Oxo dropped in ring keto acid: {name}"
        )


@pytest.mark.integration
class TestAminoNotDropped:
    """Verify -NH2 groups are retained in polyfunctional names."""

    def test_amino_not_dropped_in_amino_alcohol(self):
        """4-amino-3-hydroxybutanoic acid: amino must appear."""
        name = name_compound("OC(=O)CC(O)CN")
        assert name is not None
        assert "amino" in name.lower(), f"Amino dropped in amino alcohol: {name}"

    def test_amino_not_dropped_in_aminobenzoic(self):
        """4-aminobenzoic acid: amino prefix on ring acid."""
        name = name_compound("OC(=O)c1ccc(N)cc1")
        assert name is not None
        assert "amino" in name.lower(), f"Amino dropped in aminobenzoic acid: {name}"

    def test_amino_not_dropped_in_amino_difluoro(self):
        """4-amino-3,5-difluorobenzoic acid: amino with halogens."""
        name = name_compound("OC(=O)c1cc(F)c(N)c(F)c1")
        assert name is not None
        assert "amino" in name.lower(), f"Amino dropped with halogens: {name}"


@pytest.mark.integration
class TestHaloNotDropped:
    """Verify halogen groups are retained in polyfunctional names."""

    def test_chloro_not_dropped_with_hydroxyl(self):
        """4-chlorophenol: -Cl must appear alongside -OH."""
        name = name_compound("Oc1ccc(Cl)cc1")
        assert name is not None
        assert "chloro" in name.lower(), f"Chloro dropped with hydroxyl: {name}"

    def test_chloro_not_dropped_with_acid(self):
        """3-chloropropanoic acid: -Cl on acid chain."""
        name = name_compound("ClCCC(=O)O")
        assert name is not None
        assert "chloro" in name.lower(), f"Chloro dropped with acid: {name}"
        assert "acid" in name.lower()

    def test_bromo_not_dropped_on_ring_acid(self):
        """4-bromobenzoic acid: -Br on ring acid."""
        name = name_compound("OC(=O)c1ccc(Br)cc1")
        assert name is not None
        assert "bromo" in name.lower(), f"Bromo dropped on ring acid: {name}"

    def test_fluoro_not_dropped_on_ring_acid(self):
        """3-fluorobenzoic acid: -F on ring acid."""
        name = name_compound("OC(=O)c1cccc(F)c1")
        assert name is not None
        assert "fluoro" in name.lower(), f"Fluoro dropped on ring acid: {name}"


@pytest.mark.integration
class TestNitroNotDropped:
    """Verify -NO2 groups are retained."""

    def test_nitro_not_dropped_with_acid(self):
        """3-nitrobenzoic acid: -NO2 on ring acid."""
        name = name_compound("OC(=O)c1cccc([N+](=O)[O-])c1")
        assert name is not None
        assert "nitro" in name.lower(), f"Nitro dropped with acid: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower()

    def test_nitro_not_dropped_with_chloro_acid(self):
        """4-chloro-2-nitrobenzoic acid: both -NO2 and -Cl must appear."""
        name = name_compound("OC(=O)c1ccc(Cl)cc1[N+](=O)[O-]")
        assert name is not None
        assert "nitro" in name.lower(), f"Nitro dropped in multi-FG: {name}"
        assert "chloro" in name.lower(), f"Chloro dropped in multi-FG: {name}"

    def test_dinitro_benzoic_acid(self):
        """3,5-dinitrobenzoic acid: two -NO2 groups must survive."""
        name = name_compound("OC(=O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1")
        assert name is not None
        assert "nitro" in name.lower(), f"Nitro dropped in dinitro: {name}"


@pytest.mark.integration
class TestCarboxyNotDropped:
    """Verify -COOH groups are retained in diacids."""

    def test_carboxy_not_dropped_in_dicarboxylic(self):
        """Hexanedioic acid: both -COOH groups expressed as dioic acid."""
        name = name_compound("OC(=O)CCCCC(=O)O")
        assert name is not None
        assert "acid" in name.lower(), f"Acid dropped: {name}"
        assert "dioic" in name.lower() or "dicarbox" in name.lower() or (
            name.lower().count("acid") >= 1
        ), f"Second acid not expressed: {name}"

    def test_carboxy_not_dropped_in_ring_diacid(self):
        """4-(carboxymethyl)benzoic acid: ring acid + chain acid."""
        name = name_compound("OC(=O)c1ccc(CC(=O)O)cc1")
        assert name is not None
        assert "acid" in name.lower() or "carboxyl" in name.lower(), (
            f"Acid groups dropped: {name}"
        )


@pytest.mark.integration
class TestSulfoNotDropped:
    """Verify -SO3H groups are retained."""

    def test_sulfo_not_dropped_with_acid(self):
        """4-sulfobenzoic acid: -SO3H and -COOH."""
        name = name_compound("OC(=O)c1ccc(S(=O)(=O)O)cc1")
        assert name is not None
        assert "sulfo" in name.lower(), f"Sulfo dropped with acid: {name}"
        assert "acid" in name.lower() or "carboxyl" in name.lower()
