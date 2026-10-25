"""a phase Plan 01: Ether prefix fixes -- glycoside naming and aryloxy detection.

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
        # B2: phenol routing changes decomposition path, now uses oxane acid + diol
        ("Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1", "oxane"),
        # B7 moved to test_b7_tier_contract below (Task 12 fix a performance pass, wp6-tests):
        # its 'dimethylphenol' fragment named a small piece of the C29 input.
    ])
    def test_systematic_glycoside_name(self, smiles, expected_fragment):
        """Glycoside compounds must use systematic or retained sugar naming."""
        name = name_compound(smiles)
        assert expected_fragment in name, (
            f"Expected '{expected_fragment}' in name, got: {name}"
        )

    # Task 12 fix a performance pass (wp6-tests; t12-research rest-of-suite item 34), D-abstain:
    # B7's old expectations ('dimethylphenol' in the name; '2,3-dimethylphenol' in
    # test_b_group_regression) named a C8 fragment of the C29 input -- a different
    # molecule, never to be restored. Raw and PIN tier now abstain (the PIN is not
    # built), best-effort names it RT-exact, so both rows assert the tier contract,
    # gate ON (tests/support/rt_assert.py): best-effort
    # '(3S)-3-{(2R)-2-[(4R,6R,7S)-6,7-dimethylbicyclo[4.4.0]dec-1-en-4-yl]-1-
    # oxacyclopropan-2-yl}-9-hydroxy-7,8-dimethyl-5-oxo-2,4-dioxabicyclo[4.4.0]deca-
    # 1(6),7,9-triene', OPSIN 2.9.0 full-InChIKey exact.
    @pytest.mark.opsin_gate
    def test_b7_tier_contract(self):
        from tests.support.rt_assert import assert_tier_contract
        assert_tier_contract(
            "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]"
            "([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2")

    # B3 (trigalloyl ester) and B5 (thiocarbamate + rhamnose glycoside) are
    # complex molecules the best-effort decomposition can only name WRONGLY
    # (both were RT-WRONG snapshots gate-off; the pre-existing gate is
    # what makes them safe). In PRODUCTION (gate ON) both abstain — the 0-wrong
    # pin. This is unchanged by F-spell-oxy (the gate-off raw string moved, the
    # production behaviour did not); asserted gate-on so it is meaningful.
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smiles,test_id", [
        ("COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1", "B5-glycosyloxy"),
    ])
    def test_complex_ether_abstains_in_production(self, smiles, test_id):
        """Gate-ON: suppresses the RT-WRONG best-effort name."""
        assert name_compound(smiles, on_failure="label", emit_tier="pin") == "unknown organic compound", test_id

    # Task 12 fix a performance pass (wp6-tests; t12-research rest-of-suite item 35), D-b: B3
    # (trigalloyl ester) moved out of the abstain list -- the gate-on PIN tier now ships
    # an RT-exact name at pin_verified where it used to abstain (TRIAGE.md 'Exit
    # criteria': a decline that becomes an RT-exact name -> the test asserts the name).
    # The acid is the principal characteristic group and the esters are acyloxy
    # prefixes 'Esters cited as prefixes', the Blue Book); the
    # enclosing marks nest ({}) per (:7446). OPSIN 2.9.0 full-InChIKey
    # exact.
    @pytest.mark.opsin_gate
    def test_trigalloyl_ester_named_in_production(self):
        from tests.support.jars import jar_or_skip
        from tests.support.rt_assert import assert_full_rt
        smiles = "O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1"
        name = name_compound(smiles)
        assert name == ("3-({3,4-dihydroxy-5-[(3,4,5-trihydroxybenzoyl)oxy]benzoyl}oxy)-"
                        "4,5-dihydroxybenzoic acid"), name
        jar_or_skip()
        assert_full_rt(name, smiles)


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
        ("CCCOc1ccccc1", "propoxybenzene"),
    ])
    def test_simple_alkoxy_unchanged(self, smiles, expected_name):
        """Simple alkoxy compounds should not be affected by aryloxy fix."""
        name = name_compound(smiles)
        assert name == expected_name, f"Expected {expected_name}, got: {name}"


# ---------------------------------------------------------------------------
# B-group regression: previously fixed ether compounds still correct
# ---------------------------------------------------------------------------

