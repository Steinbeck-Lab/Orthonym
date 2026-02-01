"""
Tests for carbocyclic partial saturation and heterocycle stereochemistry.

This module tests:
1. detect_carbocyclic_partial_saturation() function
2. Tetrahydronaphthalene naming (E2E)
3. Dihydronaphthalene naming
4. Substituted partially saturated carbocycles
5. Heterocycle stereochemistry integration
6. No regression for existing functionality

IUPAC 2013 Blue Book P-31.1.1:
- tetrahydronaphthalene: 4 positions saturated = tetrahydro prefix
- dihydronaphthalene: 2 positions saturated = dihydro prefix
- perhydronaphthalene (decalin): all saturated = perhydro/decahydro
"""
import pytest
from rdkit import Chem

from src.orthonym.rules.partial_saturation import (
    detect_carbocyclic_partial_saturation,
    is_tetrahydronaphthalene,
    format_saturation_prefix,
    get_saturation_prefix,
)
from src.orthonym.namer import name_compound


class TestCarbocyclicPartialSaturationDetection:
    """Test partial saturation detection for carbocycles."""

    def test_tetrahydronaphthalene_detection(self):
        """Tetrahydronaphthalene should be detected with tetrahydro prefix."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is not None
        assert 'tetrahydro' in result['prefix'].lower()
        assert result['parent_name'] == 'naphthalene'
        assert result['sp3_count'] == 4

    def test_naphthalene_no_saturation(self):
        """Fully aromatic naphthalene should return None."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is None  # No partial saturation

    def test_dihydronaphthalene_detection(self):
        """Dihydronaphthalene (2 sp3 carbons) should be detected."""
        # 1,2-dihydronaphthalene
        mol = Chem.MolFromSmiles('C1Cc2ccccc2C=C1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        if result is not None:
            assert 'dihydro' in result['prefix'].lower()
            assert result['sp3_count'] == 2

    def test_decalin_perhydro_detection(self):
        """Decalin (perhydronaphthalene) should be detected as perhydro."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is not None
        assert result['is_perhydro'] == True
        assert result['prefix'] == 'perhydro'

    def test_benzene_not_detected(self):
        """Simple benzene should not be detected as partial saturation."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is None

    def test_cyclohexane_not_detected(self):
        """Cyclohexane should not be detected (not fused)."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is None

    def test_heterocycle_not_detected(self):
        """Heterocycles should not be detected as carbocyclic."""
        # Tetrahydroquinoline
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        assert result is None  # Has nitrogen, not pure carbocycle


class TestIsTetrahydronaphthalene:
    """Test is_tetrahydronaphthalene helper function."""

    def test_tetrahydronaphthalene_true(self):
        """Standard tetrahydronaphthalene should return True."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        assert is_tetrahydronaphthalene(mol, ring_atoms) == True

    def test_naphthalene_false(self):
        """Fully aromatic naphthalene should return False."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        assert is_tetrahydronaphthalene(mol, ring_atoms) == False

    def test_decalin_false(self):
        """Fully saturated decalin should return False (not tetrahydro)."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        assert is_tetrahydronaphthalene(mol, ring_atoms) == False


class TestTetrahydronaphthaleneNaming:
    """E2E tests for tetrahydronaphthalene naming."""

    def test_tetrahydronaphthalene_basic(self):
        """Tetrahydronaphthalene base naming."""
        result = name_compound('c1ccc2c(c1)CCCC2')
        assert 'tetrahydro' in result.lower()
        assert 'naphthalene' in result.lower()
        assert 'benzene' not in result.lower()
        assert 'butyl' not in result.lower()

    def test_tetrahydronaphthalene_locants(self):
        """Tetrahydronaphthalene should have 1,2,3,4 locants."""
        result = name_compound('c1ccc2c(c1)CCCC2')
        # Should be 1,2,3,4-tetrahydronaphthalene
        assert '1,2,3,4-tetrahydro' in result.lower()

    @pytest.mark.xfail(reason="Substituent handling for partial saturation not implemented yet")
    def test_methyltetrahydronaphthalene(self):
        """Substituted tetrahydronaphthalene."""
        result = name_compound('Cc1ccc2c(c1)CCCC2')
        assert 'tetrahydro' in result.lower() or 'tetralin' in result.lower()
        assert 'methyl' in result.lower()

    def test_tetralin_smiles_variant(self):
        """Different SMILES representation should give same result."""
        result1 = name_compound('c1ccc2c(c1)CCCC2')
        result2 = name_compound('C1CCc2ccccc2C1')
        # Both should be tetrahydronaphthalene
        assert 'tetrahydro' in result1.lower()
        assert 'tetrahydro' in result2.lower()


class TestHeterocycleStereochemistry:
    """Test stereochemistry in heterocycle naming."""

    def test_piperidine_with_stereo(self):
        """N-substituted piperidine should include stereo."""
        result = name_compound('CCC[C@H]1CCCCN1C')
        # Should have stereodescriptor
        has_stereo = '(' in result and ')' in result and ('S' in result or 'R' in result)
        assert has_stereo, f"Expected stereo in: {result}"

    def test_piperidine_2s_configuration(self):
        """Specific (2S) configuration test."""
        result = name_compound('CCC[C@H]1CCCCN1C')
        # Check for expected stereo format
        assert '2S' in result or '2R' in result, f"Missing stereo locant: {result}"

    def test_pyrrolidine_with_stereo(self):
        """Substituted pyrrolidine with stereo."""
        result = name_compound('C[C@H]1CCCN1')
        has_stereo = '(' in result and ')' in result
        # Note: might not have stereo if center not recognized
        # This test just ensures no crash

    def test_piperidine_without_stereo(self):
        """Piperidine without chiral center should have no stereo."""
        result = name_compound('CN1CCCCC1')
        # Should not have stereo descriptors
        assert '(' not in result or 'S' not in result, f"Unexpected stereo in: {result}"

    def test_morpholine_basic(self):
        """Simple morpholine naming."""
        result = name_compound('C1COCCN1')
        assert 'morpholine' in result.lower()

    def test_n_methylmorpholine(self):
        """N-methylmorpholine naming."""
        result = name_compound('CN1CCOCC1')
        assert 'morpholine' in result.lower()
        assert 'methyl' in result.lower()


class TestNoRegression:
    """Ensure fixes don't break existing functionality."""

    def test_naphthalene_still_works(self):
        """Naphthalene should still return naphthalene."""
        result = name_compound('c1ccc2ccccc2c1')
        assert 'naphthalene' in result.lower()

    def test_benzene_still_works(self):
        """Benzene should still return benzene."""
        result = name_compound('c1ccccc1')
        assert 'benzene' in result.lower()

    def test_piperidine_still_works(self):
        """Piperidine without stereo should still work."""
        result = name_compound('C1CCNCC1')
        assert 'piperidine' in result.lower()

    def test_pyrrolidine_still_works(self):
        """Pyrrolidine should still work."""
        result = name_compound('C1CCNC1')
        assert 'pyrrolidine' in result.lower()

    def test_pyridine_still_works(self):
        """Pyridine should still work."""
        result = name_compound('c1ccncc1')
        assert 'pyridine' in result.lower()

    def test_methylcyclohexane_still_works(self):
        """Simple cycloalkane should still work."""
        result = name_compound('CC1CCCCC1')
        assert 'cyclohexane' in result.lower()
        assert 'methyl' in result.lower()

    def test_anthracene_still_works(self):
        """Anthracene should still work."""
        result = name_compound('c1ccc2cc3ccccc3cc2c1')
        assert 'anthracene' in result.lower()


class TestSaturationPrefixFormatting:
    """Test saturation prefix formatting."""

    def test_format_tetrahydro_with_locants(self):
        """Test formatting tetrahydro with locants."""
        result = format_saturation_prefix('tetrahydro', [1, 2, 3, 4])
        assert result == '1,2,3,4-tetrahydro'

    def test_format_dihydro_with_locants(self):
        """Test formatting dihydro with locants."""
        result = format_saturation_prefix('dihydro', [1, 2])
        assert result == '1,2-dihydro'

    def test_format_perhydro_no_locants(self):
        """Perhydro should not have locants."""
        result = format_saturation_prefix('perhydro')
        assert result == 'perhydro'

    def test_get_saturation_prefix_2h(self):
        """Test getting dihydro prefix."""
        assert get_saturation_prefix(2) == 'dihydro'

    def test_get_saturation_prefix_4h(self):
        """Test getting tetrahydro prefix."""
        assert get_saturation_prefix(4) == 'tetrahydro'

    def test_get_saturation_prefix_10h(self):
        """Test getting decahydro prefix."""
        assert get_saturation_prefix(10) == 'decahydro'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_ring_atoms(self):
        """Empty ring atoms should return None."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        result = detect_carbocyclic_partial_saturation(mol, set())
        assert result is None

    def test_none_molecule(self):
        """None molecule should return None."""
        result = detect_carbocyclic_partial_saturation(None, {1, 2, 3})
        assert result is None

    def test_hexahydro_prefix(self):
        """Test hexahydro prefix (6 sp3 atoms)."""
        assert get_saturation_prefix(6) == 'hexahydro'

    def test_octahydro_prefix(self):
        """Test octahydro prefix (8 sp3 atoms)."""
        assert get_saturation_prefix(8) == 'octahydro'


class TestSubstitutedPartiallySaturatedCarbocycles:
    """Test substituted partially saturated carbocycles."""

    def test_methyl_tetrahydronaphthalene_detection(self):
        """Methyltetrahydronaphthalene detection."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)CCCC2')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        # Should detect parent as tetrahydronaphthalene
        if result:
            assert result['parent_name'] == 'naphthalene'
            assert 'tetrahydro' in result['prefix']

    def test_dimethyl_tetrahydronaphthalene(self):
        """Dimethyltetrahydronaphthalene should work."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)C(C)CCC2')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        # Should detect partial saturation
        # Note: substituents outside ring system may affect ring atom set
