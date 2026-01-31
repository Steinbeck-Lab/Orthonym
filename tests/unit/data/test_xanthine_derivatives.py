"""
Unit tests for xanthine derivatives data module.

Tests:
- Data completeness for xanthine family compounds
- Individual entry verification for caffeine, theophylline, theobromine
- Identification function accuracy
- Naming function correctness
- Edge cases for non-xanthine compounds
"""

import pytest
from rdkit import Chem
from src.orthonym.data.xanthine_derivatives import (
    XANTHINE_DERIVATIVES,
    identify_xanthine,
    get_xanthine_name,
    get_xanthine_info,
    get_xanthine_n_positions,
    is_xanthine_derivative,
)


class TestXanthineData:
    """Test completeness and correctness of xanthine data entries."""

    @pytest.mark.unit
    def test_data_completeness(self):
        """Verify all expected xanthine derivatives are present."""
        expected_compounds = [
            'xanthine',
            'caffeine',
            'theophylline',
            'theobromine',
            'paraxanthine',
            '7-methylxanthine',
            '1-methylxanthine',
            '3-methylxanthine',
        ]

        common_names = [data['common_name'] for data in XANTHINE_DERIVATIVES.values()]
        for compound in expected_compounds:
            assert compound in common_names, f"Missing {compound} in XANTHINE_DERIVATIVES"

    @pytest.mark.unit
    def test_minimum_entries(self):
        """Verify at least 8 xanthine derivative entries exist."""
        assert len(XANTHINE_DERIVATIVES) >= 8

    @pytest.mark.unit
    def test_all_entries_have_required_fields(self):
        """Verify all entries have the required data fields."""
        required_fields = [
            'systematic_name',
            'common_name',
            'n_positions',
            'n_substituents',
            'indicated_h',
            'saturation',
            'parent',
        ]

        for smiles, data in XANTHINE_DERIVATIVES.items():
            for field in required_fields:
                assert field in data, f"Missing '{field}' in entry for {smiles}"

    @pytest.mark.unit
    def test_caffeine_entry(self):
        """Verify caffeine data structure is correct."""
        caffeine_data = None
        for data in XANTHINE_DERIVATIVES.values():
            if data['common_name'] == 'caffeine':
                caffeine_data = data
                break

        assert caffeine_data is not None, "Caffeine entry not found"
        assert caffeine_data['systematic_name'] == '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
        assert caffeine_data['n_positions'] == [1, 3, 7]
        assert len(caffeine_data['n_substituents']) == 3
        assert caffeine_data['indicated_h'] == '1H'
        assert caffeine_data['saturation'] == '3,7-dihydro'
        assert caffeine_data['parent'] == 'purine-2,6-dione'

    @pytest.mark.unit
    def test_theophylline_entry(self):
        """Verify theophylline data structure is correct."""
        theophylline_data = None
        for data in XANTHINE_DERIVATIVES.values():
            if data['common_name'] == 'theophylline':
                theophylline_data = data
                break

        assert theophylline_data is not None, "Theophylline entry not found"
        assert theophylline_data['systematic_name'] == '1,3-dimethyl-7H-purine-2,6-dione'
        assert theophylline_data['n_positions'] == [1, 3]
        assert theophylline_data['indicated_h'] == '7H'
        assert theophylline_data['saturation'] is None  # No dihydro prefix

    @pytest.mark.unit
    def test_theobromine_entry(self):
        """Verify theobromine data structure is correct."""
        theobromine_data = None
        for data in XANTHINE_DERIVATIVES.values():
            if data['common_name'] == 'theobromine':
                theobromine_data = data
                break

        assert theobromine_data is not None, "Theobromine entry not found"
        assert theobromine_data['systematic_name'] == '3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione'
        assert theobromine_data['n_positions'] == [3, 7]
        assert theobromine_data['indicated_h'] == '1H'
        assert theobromine_data['saturation'] == '3,7-dihydro'


class TestXanthineIdentification:
    """Test xanthine identification function accuracy."""

    @pytest.mark.unit
    def test_identify_caffeine(self):
        """Caffeine should be identified correctly."""
        # Test with multiple SMILES representations
        caffeine_smiles_variants = [
            'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Different notation
        ]
        for smiles in caffeine_smiles_variants:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Failed to parse {smiles}"
            result = identify_xanthine(mol)
            assert result is not None, f"Failed to identify caffeine from {smiles}"

    @pytest.mark.unit
    def test_identify_theophylline(self):
        """Theophylline should be identified correctly."""
        smiles = 'Cn1c2c(c(=O)n(c1=O)C)[nH]cn2'
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        result = identify_xanthine(mol)
        assert result is not None, "Failed to identify theophylline"

    @pytest.mark.unit
    def test_identify_theobromine(self):
        """Theobromine should be identified correctly."""
        smiles = 'Cn1cnc2c1c(=O)[nH]c(=O)n2C'
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        result = identify_xanthine(mol)
        assert result is not None, "Failed to identify theobromine"

    @pytest.mark.unit
    def test_identify_non_xanthine_benzene(self):
        """Benzene should not be identified as xanthine."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        result = identify_xanthine(mol)
        assert result is None, "Benzene incorrectly identified as xanthine"

    @pytest.mark.unit
    def test_identify_non_xanthine_purine(self):
        """Purine (without oxo groups) should not match xanthine patterns."""
        mol = Chem.MolFromSmiles('c1ncc2nc[nH]c2n1')  # 9H-purine
        result = identify_xanthine(mol)
        assert result is None, "Purine incorrectly identified as xanthine derivative"

    @pytest.mark.unit
    def test_identify_non_xanthine_adenine(self):
        """Adenine should not match xanthine patterns."""
        mol = Chem.MolFromSmiles('Nc1ncnc2nc[nH]c12')
        result = identify_xanthine(mol)
        assert result is None, "Adenine incorrectly identified as xanthine derivative"

    @pytest.mark.unit
    def test_identify_none_input(self):
        """None input should return None."""
        result = identify_xanthine(None)
        assert result is None


class TestXanthineNaming:
    """Test xanthine naming function correctness."""

    @pytest.mark.unit
    def test_caffeine_systematic_name(self):
        """Caffeine systematic name should be correct."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        result = get_xanthine_name(mol)
        assert result == '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'

    @pytest.mark.unit
    def test_caffeine_common_name(self):
        """Caffeine common name should be returned when requested."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        result = get_xanthine_name(mol, use_common=True)
        assert result == 'caffeine'

    @pytest.mark.unit
    def test_theophylline_systematic_name(self):
        """Theophylline systematic name should be correct."""
        mol = Chem.MolFromSmiles('Cn1c2c(c(=O)n(c1=O)C)[nH]cn2')
        result = get_xanthine_name(mol)
        assert result == '1,3-dimethyl-7H-purine-2,6-dione'

    @pytest.mark.unit
    def test_theobromine_systematic_name(self):
        """Theobromine systematic name should be correct."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)[nH]c(=O)n2C')
        result = get_xanthine_name(mol)
        assert result == '3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione'

    @pytest.mark.unit
    def test_non_xanthine_returns_none(self):
        """Non-xanthine molecules should return None."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        result = get_xanthine_name(mol)
        assert result is None


class TestXanthineInfoFunctions:
    """Test helper functions for xanthine info extraction."""

    @pytest.mark.unit
    def test_get_xanthine_info_caffeine(self):
        """get_xanthine_info should return complete data for caffeine."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        info = get_xanthine_info(mol)

        assert info is not None
        assert info['common_name'] == 'caffeine'
        assert info['n_positions'] == [1, 3, 7]
        assert info['indicated_h'] == '1H'
        assert info['saturation'] == '3,7-dihydro'

    @pytest.mark.unit
    def test_get_xanthine_n_positions_caffeine(self):
        """Caffeine should have N-positions [1, 3, 7]."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        positions = get_xanthine_n_positions(mol)
        assert positions == [1, 3, 7]

    @pytest.mark.unit
    def test_get_xanthine_n_positions_theophylline(self):
        """Theophylline should have N-positions [1, 3]."""
        mol = Chem.MolFromSmiles('Cn1c2c(c(=O)n(c1=O)C)[nH]cn2')
        positions = get_xanthine_n_positions(mol)
        assert positions == [1, 3]

    @pytest.mark.unit
    def test_is_xanthine_derivative_caffeine(self):
        """Caffeine should be recognized as xanthine derivative."""
        mol = Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(c(=O)n2C)C')
        assert is_xanthine_derivative(mol) is True

    @pytest.mark.unit
    def test_is_xanthine_derivative_benzene(self):
        """Benzene should not be recognized as xanthine derivative."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert is_xanthine_derivative(mol) is False

    @pytest.mark.unit
    def test_is_xanthine_derivative_none(self):
        """None input should return False."""
        assert is_xanthine_derivative(None) is False
