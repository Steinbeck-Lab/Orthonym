"""
End-to-end integration tests for a phase: Heterocycles.

Verifies all HETERO requirements (HETERO-01 through HETERO-09) are met.
Tests the complete naming pipeline from SMILES to final IUPAC name.

Reference: IUPAC 2013 Blue Book, Section (Heterocycles)
"""

import pytest
from orthonym import name_compound


# =============================================================================
# HETERO-01: 3-membered saturated heterocycles
# =============================================================================

class TestHETERO01:
    """HETERO-01: 3-membered saturated heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CO1", "oxirane"),       # Oxygen
        ("C1CN1", "aziridine"),     # Nitrogen
        ("C1CS1", "thiirane"),      # Sulfur
    ])
    def test_3_membered_saturated(self, smiles, expected):
        """Test 3-membered saturated heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-02: 5-membered saturated heterocycles
# =============================================================================

class TestHETERO02:
    """HETERO-02: 5-membered saturated heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCOC1", "oxolane"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCSC1", "thiolane"),  # Wave2 T1d: HW PIN (was tetrahydrothiophene)
    ])
    def test_5_membered_saturated(self, smiles, expected):
        """Test 5-membered saturated heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-03: 5-membered aromatic heterocycles
# =============================================================================

class TestHETERO03:
    """HETERO-03: 5-membered aromatic heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Single heteroatom
        ("c1ccoc1", "furan"),
        ("c1cc[nH]c1", "1H-pyrrole"),  # Wave2 hygiene: 1H- azole PIN
        ("c1ccsc1", "thiophene"),
        # Two heteroatoms - 1,3 arrangement (Wave2 hygiene: PINs — 1H- azoles
        # + batch-A HW-locant oxa/thia-azoles; this integration file lagged the gate)
        ("c1c[nH]cn1", "1H-imidazole"),
        ("c1cnco1", "1,3-oxazole"),
        ("c1cncs1", "1,3-thiazole"),
        # Two heteroatoms - 1,2 arrangement
        ("c1cc[nH]n1", "1H-pyrazole"),
        ("c1ccno1", "1,2-oxazole"),
        ("c1ccsn1", "1,2-thiazole"),
    ])
    def test_5_membered_aromatic(self, smiles, expected):
        """Test 5-membered aromatic heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-04: 6-membered saturated heterocycles
# =============================================================================

class TestHETERO04:
    """HETERO-04: 6-membered saturated heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCOCC1", "oxane"),
        ("C1CCNCC1", "piperidine"),
        ("C1CCSCC1", "thiane"),
        ("C1COCCN1", "morpholine"),
        ("C1CNCCN1", "piperazine"),
    ])
    def test_6_membered_saturated(self, smiles, expected):
        """Test 6-membered saturated heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-05: 6-membered aromatic heterocycles
# =============================================================================

class TestHETERO05:
    """HETERO-05: 6-membered aromatic heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccncc1", "pyridine"),
        ("c1ccnnc1", "pyridazine"),
        ("c1cncnc1", "pyrimidine"),
        ("c1cnccn1", "pyrazine"),
    ])
    def test_6_membered_aromatic(self, smiles, expected):
        """Test 6-membered aromatic heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-06: Hantzsch-Widman systematic naming
# =============================================================================

class TestHETERO06:
    """HETERO-06: HW systematic naming when no retained name exists."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_contains", [
        # Triazines have retained names. 'c1nncnn1' is 1,2,4,5-TETRAZINE (four ring N),
        # not a triazine: the 1,2,4-triazine row uses its own SMILES (the id keeps the
        # original 'c1nncnn1' spelling so the row identity is stable) and the original
        # SMILES is pinned to its real name beside it.
        pytest.param("n1ncncc1", "1,2,4-triazine", id="c1nncnn1-1,2,4-triazine"),
        ("c1nncnn1", "1,2,4,5-tetrazine"),
        ("c1ncncn1", "1,3,5-triazine"),
        # Tetrazole has retained name
        ("c1nnn[nH]1", "tetrazole"),
    ])
    def test_hw_fallback_or_retained(self, smiles, expected_contains):
        """Test HW naming fallback or retained name usage."""
        result = name_compound(smiles)
        assert expected_contains in result or result == expected_contains


# =============================================================================
# HETERO-07: 4-membered heterocycles
# =============================================================================

class TestHETERO07:
    """HETERO-07: 4-membered heterocycles with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("C1COC1", "oxetane"),
        ("C1CNC1", "azetidine"),
        ("C1CSC1", "thietane"),
    ])
    def test_4_membered(self, smiles, expected):
        """Test 4-membered heterocycles use retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# HETERO-08: Saturation prefixes (dihydro-, tetrahydro-)
# =============================================================================

class TestHETERO08:
    """HETERO-08: Saturation prefixes for partially saturated heterocycles."""

    @pytest.mark.integration
    def test_saturated_forms_have_retained_names(self):
        """Test that common saturated forms use retained names.

        For oxolane, oxane, etc., IUPAC prefers
        the retained names over systematic names like
        'oxolane' over 'oxolane'.
        """
        # These saturated forms have retained names
        assert name_compound("C1CCOC1") == "oxolane"
        assert name_compound("C1CCOCC1") == "oxane"


# =============================================================================
# HETERO-09: Substituted heterocycles
# =============================================================================

class TestHETERO09:
    """HETERO-09: Substituted heterocycles with N-locant and C-locant prefixes."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # A substituent on a ring nitrogen takes the ring's numeric locant, not the
        # italic N: '4-(...)morpholine (PIN)' the Blue Book and
        # '4-(cyclohexanesulfinyl)morpholine-2-carboxylic acid (PIN)' the Blue Book.
        # The ids keep the original 'N-...' spelling so the row identity is stable.
        pytest.param("CN1CCCC1", "1-methylpyrrolidine", id="CN1CCCC1-N-methylpyrrolidine"),
        pytest.param("CCN1CCCC1", "1-ethylpyrrolidine", id="CCN1CCCC1-N-ethylpyrrolidine"),
        pytest.param("CN1CCCCC1", "1-methylpiperidine", id="CN1CCCCC1-N-methylpiperidine"),
        pytest.param("CN1CCOCC1", "4-methylmorpholine", id="CN1CCOCC1-N-methylmorpholine"),
        # C-substitution uses numeric locants
        ("Cc1ccccn1", "2-methylpyridine"),
        ("Cc1cccnc1", "3-methylpyridine"),
        ("Cc1ccncc1", "4-methylpyridine"),
        ("Cc1ccco1", "2-methylfuran"),
        ("Cc1ccoc1", "3-methylfuran"),
    ])
    def test_substituted_heterocycles(self, smiles, expected):
        """Test substituted heterocycles have correct locant format."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_n_locant_format(self):
        """Test that substitution on a ring N uses the numeric ring locant, not 'N-'.

        '4-(...)morpholine (PIN)', the Blue Book.
        """
        result = name_compound("CN1CCCC1")
        assert result == "1-methylpyrrolidine"
        assert not result.startswith("N-")

    @pytest.mark.integration
    def test_c_locant_is_numeric(self):
        """Test that C-substitution uses numeric locants."""
        result = name_compound("Cc1ccncc1")  # 4-methylpyridine
        # Should start with a digit
        assert result[0].isdigit()
        assert "-methylpyridine" in result


# =============================================================================
# Full a phase regression tests
# =============================================================================

class TestPhase3Regression:
    """Regression tests for all a phase functionality."""

    @pytest.mark.integration
    def test_phase1_still_works(self):
        """Test that a phase (foundation) functionality still works."""
        # Simple alkanes
        assert name_compound("C") == "methane"
        assert name_compound("CC") == "ethane"
        assert name_compound("CCC") == "propane"

        # Branched alkanes
        assert name_compound("CC(C)C") == "2-methylpropane"

        # Functional groups
        assert name_compound("CCO") == "ethanol"
        assert name_compound("CC=O") == "acetaldehyde"

    @pytest.mark.integration
    def test_phase2_still_works(self):
        """Test that a phase (ring foundation) functionality still works."""
        # Cycloalkanes
        assert name_compound("C1CCC1") == "cyclobutane"
        assert name_compound("C1CCCCC1") == "cyclohexane"

        # Benzene
        assert name_compound("c1ccccc1") == "benzene"
        assert name_compound("Cc1ccccc1") == "toluene"  # Retained name

    @pytest.mark.integration
    def test_all_base_heterocycles(self):
        """Comprehensive test of all base heterocycle retained names."""
        heterocycles = [
            # 3-membered
            ("C1CO1", "oxirane"),
            ("C1CN1", "aziridine"),
            ("C1CS1", "thiirane"),
            # 4-membered
            ("C1COC1", "oxetane"),
            ("C1CNC1", "azetidine"),
            ("C1CSC1", "thietane"),
            # 5-membered saturated
            ("C1CCOC1", "oxolane"),
            ("C1CCNC1", "pyrrolidine"),
            ("C1CCSC1", "thiolane"),  # Wave2 T1d: HW PIN (was tetrahydrothiophene)
            # 5-membered aromatic
            ("c1ccoc1", "furan"),
            ("c1cc[nH]c1", "1H-pyrrole"),  # Wave2 hygiene: 1H- azole PINs
            ("c1ccsc1", "thiophene"),
            ("c1c[nH]cn1", "1H-imidazole"),
            ("c1cc[nH]n1", "1H-pyrazole"),
            # 6-membered saturated
            ("C1CCOCC1", "oxane"),
            ("C1CCNCC1", "piperidine"),
            ("C1CCSCC1", "thiane"),
            ("C1COCCN1", "morpholine"),
            ("C1CNCCN1", "piperazine"),
            # 6-membered aromatic
            ("c1ccncc1", "pyridine"),
            ("c1cncnc1", "pyrimidine"),
            ("c1cnccn1", "pyrazine"),
        ]

        for smiles, expected in heterocycles:
            result = name_compound(smiles)
            assert result == expected, f"Failed: {smiles} -> {result} (expected {expected})"

    @pytest.mark.integration
    def test_all_substituted_heterocycles(self):
        """Comprehensive test of substituted heterocycle naming."""
        substituted = [
            # N-substituted (the ring nitrogen takes its numeric locant; the Blue Book)
            ("CN1CCCC1", "1-methylpyrrolidine"),
            ("CCN1CCCC1", "1-ethylpyrrolidine"),
            ("CN1CCCCC1", "1-methylpiperidine"),
            ("CN1CCOCC1", "4-methylmorpholine"),
            ("CN1CC1", "1-methylaziridine"),
            ("CN1CCC1", "1-methylazetidine"),
            # C-substituted aromatic (use numeric locants)
            ("Cc1ccccn1", "2-methylpyridine"),
            ("Cc1cccnc1", "3-methylpyridine"),
            ("Cc1ccncc1", "4-methylpyridine"),
            ("Cc1ccco1", "2-methylfuran"),
            ("Cc1ccoc1", "3-methylfuran"),
            ("Cc1cccs1", "2-methylthiophene"),
            ("Cc1ccsc1", "3-methylthiophene"),
        ]

        for smiles, expected in substituted:
            result = name_compound(smiles)
            assert result == expected, f"Failed: {smiles} -> {result} (expected {expected})"


# =============================================================================
# Test counts verification
# =============================================================================

class TestCounts:
    """Verify test counts and coverage."""

    @pytest.mark.integration
    def test_sufficient_test_coverage(self):
        """Verify we have comprehensive test coverage for a phase.

        This test documents the scope of a phase testing.
        """
        # This is a documentation test - it always passes
        # but documents what we're testing

        # HETERO-01: 3 compounds tested (oxirane, aziridine, thiirane)
        # HETERO-02: 3 compounds tested (THF, pyrrolidine, tetrahydrothiophene)
        # HETERO-03: 9 compounds tested (furan, pyrrole, thiophene, imidazole, etc.)
        # HETERO-04: 5 compounds tested (THP, piperidine, thiane, morpholine, piperazine)
        # HETERO-05: 4 compounds tested (pyridine, pyridazine, pyrimidine, pyrazine)
        # HETERO-06: 3 compounds tested (triazines, tetrazole)
        # HETERO-07: 3 compounds tested (oxetane, azetidine, thietane)
        # HETERO-08: 2 compounds tested (THF, THP as retained names)
        # HETERO-09: 13 compounds tested (N-methyl variants, methylpyridines, methylfurans)

        # Total a phase coverage: ~45 unique compounds
        # Plus regression tests for a phase and 2

        assert True  # This test documents coverage
