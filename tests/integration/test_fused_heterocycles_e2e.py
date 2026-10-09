"""
Integration tests for a phase: Fused Heterocycle Enhancement.

Tests the complete pipeline from SMILES to IUPAC name for:
- Purine derivatives (adenine, purine, xanthine core)
- Substituted indoles with correct locants
- Substituted quinolines
- N-substituted fused heterocycles
- Oxo and amino group handling

Validates a phase requirements (FUSED-01 through FUSED-07).

Reference: IUPAC 2013 Blue Book, Section (Fused Ring Systems)
"""

import pytest
from rdkit import Chem
from orthonym import name_compound
from orthonym.rules.fused_rings import name_fused_heterocycle as _name_fused_heterocycle_raw


def name_fused_heterocycle(mol):
    """Wrapper that extracts just the name string from the tuple result."""
    result = _name_fused_heterocycle_raw(mol)
    if result is None:
        return None
    return result[0]


# =============================================================================
# Test Purine Derivatives
# =============================================================================

class TestPurineDerivatives:
    """Test naming of purine-based compounds."""

    @pytest.mark.integration
    def test_purine_base(self):
        """Unsubstituted purine should be 9H-purine."""
        result = name_compound('c1ncc2nc[nH]c2n1')
        assert result == '9H-purine', f"Got: {result}"

    @pytest.mark.integration
    def test_adenine(self):
        """Adenine: IUPAC 2013 prefers retained name 'adenine'.

        Systematic name '9H-purin-6-amine' is also valid.
        """
        result = name_compound('Nc1ncnc2nc[nH]c12')
        # IUPAC 2013: retained name "adenine" is preferred
        # Systematic "9H-purin-6-amine" is also valid
        assert result == "adenine" or ('purin' in result.lower() and 'amin' in result.lower()), f"Got: {result}"

    @pytest.mark.integration
    def test_hypoxanthine_like(self):
        """Test oxo-purine naming (hypoxanthine pattern).

        Note: Full hypoxanthine has tautomeric forms that complicate naming.
        This tests the basic oxo-purine pattern.
        IUPAC 2013 prefers retained name 'hypoxanthine' when available.
        """
        # Simple oxo-purine test
        result = name_compound('O=c1nc[nH]c2nc[nH]c12')
        # IUPAC 2013: retained name "hypoxanthine" is preferred
        # Systematic name containing "purin" is also valid
        assert result == "hypoxanthine" or 'purin' in result.lower(), f"Got: {result}"

    @pytest.mark.integration
    def test_caffeine_structure(self):
        """Caffeine: 1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione.

        Caffeine naming uses numeric locants for N-substitution positions
        (1,3,7-trimethyl) rather than N-methyl format, following IUPAC
        rules for xanthine derivatives.
        """
        result = name_compound('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        # Full IUPAC name with dihydro prefix
        expected = '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
        assert result == expected, f"Expected '{expected}', got '{result}'"


# =============================================================================
# Test Substituted Indoles
# =============================================================================

class TestSubstitutedIndoles:
    """Test correct locant assignment for substituted indoles."""

    @pytest.mark.integration
    def test_unsubstituted_indole(self):
        """Indole should be 1H-indole."""
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert result == '1H-indole', f"Got: {result}"

    @pytest.mark.integration
    def test_5_methylindole(self):
        """5-methylindole: methyl at position 5."""
        result = name_compound('Cc1ccc2[nH]ccc2c1')
        assert result == '5-methyl-1H-indole', f"Got: {result}"

    @pytest.mark.integration
    def test_n_methylindole(self):
        """N-methylindole: a ring nitrogen takes its numeric locant, '1-methyl-1H-indole'.

        A substituent on a ring heteroatom is cited with the ring locant, not the
        italic 'N': '4-(...)morpholine (PIN)' the Blue Book and
        '1-hydroxy-1H-pyrrole-2,5-dione (PIN)' the Blue Book.
        """
        result = name_compound('Cn1ccc2ccccc12')
        assert result == '1-methyl-1H-indole', f"Expected 1-methyl-1H-indole, got: {result}"

    @pytest.mark.integration
    def test_2_methylindole(self):
        """2-methylindole: methyl at position 2."""
        result = name_compound('Cc1cc2ccccc2[nH]1')
        assert '2-methyl' in result, f"Expected '2-methyl' in name, got: {result}"
        assert 'indole' in result.lower(), f"Got: {result}"

    @pytest.mark.integration
    def test_3_methylindole_skatole(self):
        """3-methylindole (skatole): methyl at position 3."""
        result = name_compound('Cc1c[nH]c2ccccc12')
        assert '3-methyl' in result, f"Expected '3-methyl' in name, got: {result}"
        assert 'indole' in result.lower(), f"Got: {result}"

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_position", [
        ('Fc1ccc2[nH]ccc2c1', '5-fluoro'),  # 5-fluoroindole
        ('Clc1ccc2[nH]ccc2c1', '5-chloro'),  # 5-chloroindole
    ])
    def test_halogenated_indoles(self, smiles, expected_position):
        """Test halogenated indoles have correct locants."""
        result = name_compound(smiles)
        assert expected_position in result, f"Expected '{expected_position}' in name, got: {result}"
        assert 'indole' in result.lower(), f"Got: {result}"


# =============================================================================
# Test Substituted Quinolines
# =============================================================================

class TestSubstitutedQuinolines:
    """Test correct locant assignment for substituted quinolines."""

    @pytest.mark.integration
    def test_unsubstituted_quinoline(self):
        """Quinoline should be quinoline (no tautomer locant needed)."""
        result = name_compound('c1ccc2ncccc2c1')
        assert result == 'quinoline', f"Got: {result}"

    @pytest.mark.integration
    def test_isoquinoline(self):
        """Isoquinoline naming."""
        result = name_compound('c1ccc2cnccc2c1')
        assert result == 'isoquinoline', f"Got: {result}"

    @pytest.mark.integration
    def test_2_methylquinoline(self):
        """2-methylquinoline (quinaldine): methyl at position 2."""
        result = name_compound('Cc1ccc2ccccc2n1')
        assert '2-methyl' in result, f"Expected '2-methyl' in name, got: {result}"
        assert 'quinoline' in result.lower(), f"Got: {result}"


# =============================================================================
# Test a phase Requirements (FUSED-01 through FUSED-07)
# =============================================================================

class TestPhase7Requirements:
    """Test a phase requirements from ROADMAP.md."""

    @pytest.mark.integration
    def test_fused_01_correct_numbering(self):
        """FUSED-01: Correct peripheral numbering for substituted fused heterocycles."""
        # 5-methylindole should have methyl at position 5, not 4 or 6
        result = name_compound('Cc1ccc2[nH]ccc2c1')
        assert '5-methyl' in result, f"Expected '5-methyl' for correct numbering, got: {result}"

    @pytest.mark.integration
    def test_fused_02_purine_skeleton(self):
        """FUSED-02: Recognize purine skeleton."""
        result = name_compound('c1ncc2nc[nH]c2n1')
        assert 'purin' in result.lower(), f"Expected 'purine' in name, got: {result}"

    @pytest.mark.integration
    def test_fused_03_adenine_naming(self):
        """FUSED-03: Adenine naming - IUPAC 2013 prefers retained name.

        Retained name 'adenine' or systematic '9H-purin-6-amine' both valid.
        """
        result = name_compound('Nc1ncnc2nc[nH]c12')
        # IUPAC 2013: retained name "adenine" is preferred
        assert result == "adenine" or '6-amine' in result, f"Expected 'adenine' or '6-amine' suffix, got: {result}"

    @pytest.mark.integration
    def test_fused_04_oxo_groups(self):
        """FUSED-04: Oxo groups use suffix naming (-one).

        IUPAC 2013 prefers retained names (hypoxanthine, xanthine) when available.
        """
        # This requires molecules with =O on fused heterocycle
        # Test with xanthine-like structure
        result = name_compound('O=c1nc[nH]c2nc[nH]c12')
        # IUPAC 2013: retained name "hypoxanthine" is preferred
        # Systematic name containing "one" or "oxo" is also valid
        assert result == "hypoxanthine" or 'one' in result.lower() or 'oxo' in result.lower(), f"Expected retained name or oxo/one handling, got: {result}"

    @pytest.mark.integration
    def test_fused_05_n_substitution(self):
        """FUSED-05: a ring N substituent is cited with the numeric ring locant.

        '1-hydroxy-1H-pyrrole-2,5-dione (PIN)', the Blue Book; the italic 'N'
        locant is for N atoms outside the ring.
        """
        result = name_compound('Cn1ccc2ccccc12')
        assert result == '1-methyl-1H-indole', f"Expected '1-methyl-1H-indole', got: {result}"

    @pytest.mark.integration
    def test_fused_06_indicated_hydrogen(self):
        """FUSED-06: Indicated hydrogen descriptors (1H-, 9H-, etc.)."""
        # Indole needs 1H-
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-' in result, f"Expected '1H-' indicated hydrogen, got: {result}"

        # Purine needs 9H-
        result = name_compound('c1ncc2nc[nH]c2n1')
        assert '9H-' in result, f"Expected '9H-' indicated hydrogen, got: {result}"

    @pytest.mark.integration
    def test_fused_07_locant_mapping(self):
        """FUSED-07: Correct IUPAC locant mappings for fused heterocycles."""
        # Test that locant mappings are consistent
        # 5-methylindole verifies benzene ring positions map correctly
        result = name_compound('Cc1ccc2[nH]ccc2c1')
        # Position 5 is on the benzene ring opposite to nitrogen
        assert '5-methyl' in result, f"Locant mapping error, got: {result}"

        # 6-aminopurine (adenine) verifies purine positions map correctly
        # IUPAC 2013: retained name "adenine" is preferred
        result = name_compound('Nc1ncnc2nc[nH]c12')
        assert result == "adenine" or '6-amine' in result, f"Purine locant mapping error, got: {result}"


# =============================================================================
# Test Data Coverage
# =============================================================================

class TestDataCoverage:
    """Test that data module has expected coverage."""

    @pytest.mark.integration
    def test_minimum_entry_count(self):
        """Should have at least 35 fused heterocycle entries (a phase target: 38)."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        assert len(FUSED_HETEROCYCLE_DATA) >= 35, f"Only {len(FUSED_HETEROCYCLE_DATA)} entries"

    @pytest.mark.integration
    def test_all_entries_have_locants(self):
        """All entries should have iupac_locants field."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        missing = []
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            if 'iupac_locants' not in data:
                missing.append(data.get('name', smiles))
        assert not missing, f"Missing iupac_locants for: {missing}"

    @pytest.mark.integration
    def test_key_fused_heterocycles_present(self):
        """Verify key fused heterocycles are in the data."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        expected_names = [
            '1H-indole',
            '1H-1,3-benzimidazole',
            'quinoline',
            'isoquinoline',
            '9H-purine',
            '1-benzofuran',
            '1,3-benzothiazole',
            'indolizine',
            '9H-carbazole',
            'acridine',
        ]

        present_names = [data['name'] for data in FUSED_HETEROCYCLE_DATA.values()]
        for expected in expected_names:
            assert expected in present_names, f"Missing expected entry: {expected}"


# =============================================================================
# Test Complete Naming Pipeline (E2E)
# =============================================================================

class TestFusedHeterocycleE2E:
    """End-to-end tests through the complete naming pipeline."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Unsubstituted fused heterocycles
        ('c1ccc2[nH]ccc2c1', '1H-indole'),
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2cnccc2c1', 'isoquinoline'),
        ('c1ccc2[nH]cnc2c1', '1H-1,3-benzimidazole'),
        #: the PIN carries the heteroatom locant,
        # the Blue Book "1-benzofuran (PIN) benzofuran";:13443 for
        # 1-benzothiophene). RETAINED_NAMES was corrected to the '1-' PIN form.
        ('c1ccc2occc2c1', '1-benzofuran'),
        ('c1ccc2sccc2c1', '1-benzothiophene'),
        ('c1ncc2nc[nH]c2n1', '9H-purine'),
    ])
    def test_unsubstituted_fused_heterocycles(self, smiles, expected):
        """Test unsubstituted fused heterocycle naming through pipeline."""
        result = name_compound(smiles)
        assert result == expected, f"SMILES {smiles}: expected '{expected}', got '{result}'"

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_contains,alt_result", [
        # Substituted indoles
        ('Cc1ccc2[nH]ccc2c1', '5-methyl', None),  # 5-methylindole
        # N-methylindole: ring N takes the numeric locant (the Blue Book,. The id
        # keeps the original 'N-methyl' spelling so the row identity is stable.
        pytest.param('Cn1ccc2ccccc12', '1-methyl-1H-indole', None,
                     id='Cn1ccc2ccccc12-N-methyl-None'),
        # Substituted quinolines
        ('Cc1ccc2ccccc2n1', '2-methyl', None),    # 2-methylquinoline
        # Amino-substituted purines - IUPAC 2013 prefers retained name "adenine"
        ('Nc1ncnc2nc[nH]c12', '6-amine', 'adenine'),   # Adenine
    ])
    def test_substituted_fused_heterocycles(self, smiles, expected_contains, alt_result):
        """Test substituted fused heterocycle naming through pipeline.

        Some compounds have IUPAC 2013 preferred retained names.
        """
        result = name_compound(smiles)
        if alt_result:
            assert expected_contains in result or result == alt_result, f"SMILES {smiles}: expected '{expected_contains}' or '{alt_result}' in '{result}'"
        else:
            assert expected_contains in result, f"SMILES {smiles}: expected '{expected_contains}' in '{result}'"


# =============================================================================
# Regression Tests (Ensure a phase still works)
# =============================================================================

class TestPhase6Regression:
    """Regression tests for a phase fused system naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # a phase fused heterocycle tests that should still pass
        ('c1ccc2[nH]ccc2c1', '1H-indole'),
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2[nH]cnc2c1', '1H-1,3-benzimidazole'),
    ])
    def test_phase6_fused_heterocycles_unchanged(self, smiles, expected):
        """a phase fused heterocycle naming should be unchanged."""
        result = name_compound(smiles)
        assert result == expected, f"Regression: {smiles} changed from '{expected}' to '{result}'"

    @pytest.mark.integration
    def test_simple_heterocycles_unchanged(self):
        """Simple (non-fused) heterocycle naming should be unchanged."""
        # These are a phase heterocycles, not fused
        assert name_compound('c1ccncc1') == 'pyridine'
        assert name_compound('c1ccoc1') == 'furan'
        # (the Blue Book): the PIN cites the indicated hydrogen, '1H-pyrrole' (the Blue Book).
        assert name_compound('c1cc[nH]c1') == '1H-pyrrole'
        assert name_compound('c1ccsc1') == 'thiophene'


# =============================================================================
# Test fused_rings Module Functions
# =============================================================================

class TestFusedRingsModule:
    """Test the fused_rings module functions directly."""

    @pytest.mark.integration
    def test_name_fused_heterocycle_indole(self):
        """Test name_fused_heterocycle function with indole."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        result = name_fused_heterocycle(mol)
        assert result == '1H-indole', f"Got: {result}"

    @pytest.mark.integration
    def test_name_fused_heterocycle_substituted(self):
        """Test name_fused_heterocycle with substituted indole."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')
        result = name_fused_heterocycle(mol)
        assert '5-methyl' in result, f"Got: {result}"
        assert 'indole' in result.lower(), f"Got: {result}"

    @pytest.mark.integration
    def test_name_fused_heterocycle_none_for_non_fused(self):
        """Test name_fused_heterocycle returns None for non-fused heterocycle."""
        mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine - not fused
        result = name_fused_heterocycle(mol)
        assert result is None, f"Expected None for non-fused, got: {result}"


# =============================================================================
# Test Summary and Coverage Documentation
# =============================================================================

class TestPhaseSummaryCounts:
    """Document test coverage summary for a phase."""

    @pytest.mark.integration
    def test_phase7_e2e_coverage(self):
        """Document comprehensive a phase E2E coverage.

        TestPurineDerivatives: 4 tests
        - purine_base: 1
        - adenine: 1
        - hypoxanthine_like: 1
        - caffeine_structure: 1 (resolved in a phase plan 04)

        TestSubstitutedIndoles: 7 tests
        - unsubstituted_indole: 1
        - 5_methylindole: 1
        - n_methylindole: 1
        - 2_methylindole: 1
        - 3_methylindole: 1
        - halogenated_indoles: 2 parametrized

        TestSubstitutedQuinolines: 3 tests
        - unsubstituted_quinoline: 1
        - isoquinoline: 1
        - 2_methylquinoline: 1

        TestPhase7Requirements: 7 tests (FUSED-01 through FUSED-07)

        TestDataCoverage: 3 tests
        - minimum_entry_count: 1
        - all_entries_have_locants: 1
        - key_fused_heterocycles_present: 1

        TestFusedHeterocycleE2E: 11 tests
        - unsubstituted: 7 parametrized
        - substituted: 4 parametrized

        TestPhase6Regression: 7 tests
        - fused_heterocycles_unchanged: 3 parametrized
        - simple_heterocycles_unchanged: 4

        TestFusedRingsModule: 3 tests

        TOTAL: ~45 unique test cases
        """
        assert True  # Documentation test
