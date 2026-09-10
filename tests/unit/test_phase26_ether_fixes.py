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
        # B7 - ring boundary fix: benzene BFS no longer walks into fused lactone
        #: phenol suffix routing changes to dimethylphenol
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

    # B3 (trigalloyl ester) and B5 (thiocarbamate + rhamnose glycoside) are
    # complex molecules the best-effort decomposition can only name WRONGLY
    # (both were RT-WRONG snapshots gate-off; the pre-existing gate is
    # what makes them safe). In PRODUCTION (gate ON) both abstain — the 0-wrong
    # pin. This is unchanged by F-spell-oxy (the gate-off raw string moved, the
    # production behaviour did not); asserted gate-on so it is meaningful.
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smiles,test_id", [
        ("O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1", "B3-tetradecoxy"),
        ("COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1", "B5-glycosyloxy"),
    ])
    def test_complex_ether_abstains_in_production(self, smiles, test_id):
        """Gate-ON: suppresses the RT-WRONG best-effort name."""
        assert name_compound(smiles) == "unknown organic compound", test_id


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
        ("COc1ccccc1", "anisole"),  # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book)
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
    """All 8 B-group ether compounds from a phase should remain correct."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_name,test_id", [
        # B1: Biphenyl ether -> DECORATED aryloxy (Wave-2 C2: the old bare
        # 'phenoxy' pin DROPPED the second ring's 2 OH + methyl — a
        # structure-losing name; the decorated recognizer now emits the
        # complete OPSIN-RT-verified form).
        (
            "COc1cc(O)cc(C)c1Oc1cc(C)cc(O)c1O",
            "4-(2,3-dihydroxy-5-methylphenoxy)-3-methoxy-5-methylphenol",
            "B1-phenoxy",
        ),
        # B2: aryloxy-on-oxane. F-spell-oxy (2026-08-08): the aryloxy linkage
        # now names in FULL. The old snapshot was RT-WRONG ('6-phenyl' dropped
        # the ring's 2-OH + 4-methyl — "known emitter guard-out"); routing the
        # alkoxy morphology through composed_alkoxy_prefix now emits the complete
        # '6-(2-hydroxy-4-methylphenoxy)' — VERIFIED RT-EXACT (was RT-WRONG).
        (
            "Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1",
            "(2R,4R,6R)-3,4,5-trihydroxy-6-(2-hydroxy-4-methylphenoxy)oxane-2-carboxylic acid",
            "B2-oxanyloxy",
        ),
        # B4: benzophenone with an aminohexyloxy chain. F-spell-oxy: the correct
        # 'hexyloxy' morphology lets the WHOLE structure name and pass —
        # abstain -> VERIFIED RT-EXACT (breadth gain, 0-wrong preserved).
        (
            "C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",
            "(4-bromophenyl)({2-fluoro-4-[6-(methyl(prop-2-en-1-yl)amino)hexyloxy]phenyl})methanone",
            "B4-decoxy",
        ),
        # B6: substituted xanthone. F-spell-oxy: now emits the PIN
        # '9H-xanthen-9-one' core (matching gold rings_numbering.json:1146 where
        # de-headlined the retained 'xanthone'); the old snapshot was the
        # non-PIN retained name. Both RT-EXACT; new form is the conformant PIN.
        (
            "COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
            "8-hydroxy-1,3-dimethoxy-6-methyl-9H-xanthen-9-one",
            "B6-phenoxy",
        ),
        # B7: Dimethyl benzene with glycoside -> (oxan-2-yl)oxy
        (
            "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]"
            "([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2",
            "2,3-dimethylphenol",  #: phenol suffix routing
            "B7-oxanyloxy",
        ),
    ])
    def test_b_group_regression(self, smiles, expected_name, test_id):
        """B-group ether compounds must produce expected names."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"[{test_id}] Expected: {expected_name}\n  Got: {name}"
        )
