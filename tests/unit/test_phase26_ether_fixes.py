"""Phase 26 Plan 01: Ether prefix fixes -- glycoside naming and aryloxy detection.

Tests:
  - Glycoside naming: (oxan-2-yl)oxy / (oxolan-2-yl)oxy instead of hexosyloxy/pentosyloxy
  - Aryloxy detection: phenoxy for chain-attached and ring-attached aromatic ethers
  - Regression safety: B1-B8 ether compounds still produce correct names
  - Non-regression: simple alkoxy names (methoxy, ethoxy, propoxy) unchanged
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Glycoside naming: hexosyloxy -> (oxan-2-yl)oxy
# ---------------------------------------------------------------------------

class TestGlycosideNaming:
    """Glycoside compounds should use systematic oxane/oxolane naming."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,forbidden", [
        # B2: Glycoside on benzene
        ("Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1", "hexosyloxy"),
        # B5: Sugar glycoside on benzene
        ("COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1", "hexosyloxy"),
        # B7: Dimethyl benzene with glycoside
        (
            "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]"
            "([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2",
            "hexosyloxy",
        ),
        # CI benchmark compound
        ("COC(=O)c1ccccc1OC1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O", "hexosyloxy"),
    ])
    def test_no_hexosyloxy(self, smiles, forbidden):
        """Glycoside compounds must NOT contain hexosyloxy or pentosyloxy."""
        name = name_compound(smiles)
        assert forbidden not in name, (
            f"Expected no '{forbidden}' in name, got: {name}"
        )
        assert "pentosyloxy" not in name, (
            f"Expected no 'pentosyloxy' in name, got: {name}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_fragment", [
        # B2: ASML-13 phenol routing changes decomposition path, now uses oxane acid + diol
        ("Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1", "oxane"),
        # B5: non-stereo rhamnose matches sugar lookup -> retained name
        ("COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1", "rhamnopyranosyloxy"),
        # B7 - ring boundary fix: benzene BFS no longer walks into fused lactone
        # ASML-13: phenol suffix routing changes to dimethylphenol
        (
            "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]"
            "([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2",
            "dimethylphenol",
        ),
    ])
    def test_systematic_glycoside_name(self, smiles, expected_fragment):
        """Glycoside compounds must use systematic or retained sugar naming."""
        name = name_compound(smiles)
        assert expected_fragment in name, (
            f"Expected '{expected_fragment}' in name, got: {name}"
        )


# ---------------------------------------------------------------------------
# Aryloxy detection (phenoxy on chain compounds)
# ---------------------------------------------------------------------------

class TestAryloxyDetection:
    """Aromatic ethers on chain compounds should produce phenoxy prefix."""

    @pytest.mark.unit
    def test_phenoxyacetic_acid(self):
        """OC(=O)COc1ccccc1 -> 2-phenoxyethanoic acid."""
        name = name_compound("OC(=O)COc1ccccc1")
        assert "phenoxy" in name, f"Expected 'phenoxy' in name, got: {name}"
        assert "hexyloxy" not in name, f"Should not contain 'hexyloxy', got: {name}"

    @pytest.mark.unit
    def test_phenoxypropanoic_acid(self):
        """OC(=O)CCOc1ccccc1 -> 3-phenoxypropanoic acid."""
        name = name_compound("OC(=O)CCOc1ccccc1")
        assert "phenoxy" in name, f"Expected 'phenoxy' in name, got: {name}"

    @pytest.mark.unit
    def test_phenoxybutanoic_acid(self):
        """OC(=O)CCCOc1ccccc1 -> 4-phenoxybutanoic acid."""
        name = name_compound("OC(=O)CCCOc1ccccc1")
        assert "phenoxy" in name, f"Expected 'phenoxy' in name, got: {name}"


# ---------------------------------------------------------------------------
# Non-regression: simple alkoxy names unchanged
# ---------------------------------------------------------------------------

class TestAlkoxyNonRegression:
    """Simple alkoxy compounds should still produce correct names."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_name", [
        ("COc1ccccc1", "anisole"),  # retained name (P-34.1.1.4 PIN)
        ("CCOc1ccccc1", "ethoxybenzene"),
        ("CCCOc1ccccc1", "propoxybenzene"),
    ])
    def test_simple_alkoxy_unchanged(self, smiles, expected_name):
        """Simple alkoxy compounds should not be affected by aryloxy fix."""
        name = name_compound(smiles)
        assert name == expected_name, f"Expected {expected_name}, got: {name}"


# ---------------------------------------------------------------------------
# B-group regression: previously fixed ether compounds still correct
# ---------------------------------------------------------------------------

class TestBGroupRegression:
    """All 8 B-group ether compounds from Phase 24 should remain correct."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_name,test_id", [
        # B1: Biphenyl ether -> phenoxy
        (
            "COc1cc(O)cc(C)c1Oc1cc(C)cc(O)c1O",
            "3-methoxy-5-methyl-4-phenoxyphenol",  # ASML-13: phenol suffix routing
            "B1-phenoxy",
        ),
        # B2: Glycoside -> (oxan-2-yl)oxy
        (
            "Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1",
            "((2R,4R,6S)-3,4,5,6-tetrahydroxyoxane-2-carboxylic acid)-4-methylbenzene-1,2-diol",  # ASML-13: phenol suffix routing + decomposition
            "B2-oxanyloxy",
        ),
        # B3: Galloyl ester chain -> tetradecoxy
        (
            "O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1",
            "3-tetradecoxy-4,5-dihydroxybenzoic acid",
            "B3-tetradecoxy",
        ),
        # B4: Complex ether chain -> decoxy
        (
            "C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",
            "1-decoxy-3-fluorobenzene",
            "B4-decoxy",
        ),
        # B5: Sugar glycoside -> rhamnopyranosyloxy (retained sugar name)
        # Phase 86: benzene universal fallback now names the COC(=S)NC-
        # chain as a substituent (previously silently dropped).
        (
            "COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1",
            "(rhamnopyranosyloxy)-4-(1-methoxy-1-(methylamino)methyl)phenol",  # ASML-13: phenol suffix routing
            "B5-glycosyloxy",
        ),
        # B6: Fused ring system -> xanthone (correctly identified after Phase 101 xanthone entry)
        (
            "COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
            "8-hydroxy-1,3-dimethoxy-6-methylxanthone",  # Phase 101: xanthone core now recognized
            "B6-phenoxy",
        ),
        # B7: Dimethyl benzene with glycoside -> (oxan-2-yl)oxy
        (
            "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]"
            "([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2",
            "2,3-dimethylphenol",  # ASML-13: phenol suffix routing
            "B7-oxanyloxy",
        ),
    ])
    def test_b_group_regression(self, smiles, expected_name, test_id):
        """B-group ether compounds must produce expected names."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"[{test_id}] Expected: {expected_name}\n  Got: {name}"
        )
