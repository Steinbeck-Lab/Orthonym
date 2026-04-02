"""
Integration tests for caffeine family (xanthine derivatives) naming.

Tests end-to-end naming through the full Orthonym pipeline for:
- Caffeine: 1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione
- Theophylline: 1,3-dimethyl-7H-purine-2,6-dione
- Theobromine: 3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione
- Related xanthine derivatives

These compounds require specific N-position numbering:
- Uses numeric locants (1,3,7-trimethyl) not N-methyl format
- Indicated hydrogen specifies tautomer (1H, 7H)
- Dihydro prefix for partial saturation (3,7-dihydro)
"""

import pytest
from orthonym import name_compound


class TestCaffeineNaming:
    """Test caffeine naming through full pipeline."""

    @pytest.mark.integration
    def test_caffeine_systematic(self):
        """Caffeine should return full systematic IUPAC name."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)
        expected = '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.integration
    def test_caffeine_n_positions(self):
        """Caffeine name should have correct 1,3,7 positions."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)
        assert '1,3,7-trimethyl' in result, f"Expected '1,3,7-trimethyl' in name, got '{result}'"

    @pytest.mark.integration
    def test_caffeine_saturation(self):
        """Caffeine should have 3,7-dihydro prefix."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)
        assert '3,7-dihydro' in result, f"Expected '3,7-dihydro' in name, got '{result}'"

    @pytest.mark.integration
    def test_caffeine_indicated_h(self):
        """Caffeine should have 1H indicated hydrogen."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)
        assert '1H-purine' in result, f"Expected '1H-purine' in name, got '{result}'"

    @pytest.mark.integration
    def test_caffeine_parent_suffix(self):
        """Caffeine should have purine-2,6-dione parent."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)
        assert 'purine-2,6-dione' in result, f"Expected 'purine-2,6-dione' in name, got '{result}'"


class TestTheophyllineNaming:
    """Test theophylline naming through full pipeline."""

    @pytest.mark.integration
    def test_theophylline_systematic(self):
        """Theophylline should return full systematic IUPAC name."""
        smiles = 'Cn1c2c(c(=O)n(c1=O)C)[nH]cn2'
        result = name_compound(smiles)
        expected = '1,3-dimethyl-7H-purine-2,6-dione'
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.integration
    def test_theophylline_no_saturation(self):
        """Theophylline should NOT have dihydro prefix."""
        smiles = 'Cn1c2c(c(=O)n(c1=O)C)[nH]cn2'
        result = name_compound(smiles)
        assert 'dihydro' not in result, f"Unexpected 'dihydro' in name, got '{result}'"

    @pytest.mark.integration
    def test_theophylline_indicated_h(self):
        """Theophylline should have 7H indicated hydrogen."""
        smiles = 'Cn1c2c(c(=O)n(c1=O)C)[nH]cn2'
        result = name_compound(smiles)
        assert '7H-purine' in result, f"Expected '7H-purine' in name, got '{result}'"

    @pytest.mark.integration
    def test_theophylline_dimethyl(self):
        """Theophylline should have 1,3-dimethyl prefix."""
        smiles = 'Cn1c2c(c(=O)n(c1=O)C)[nH]cn2'
        result = name_compound(smiles)
        assert '1,3-dimethyl' in result, f"Expected '1,3-dimethyl' in name, got '{result}'"


class TestTheobromineName:
    """Test theobromine naming through full pipeline."""

    @pytest.mark.integration
    def test_theobromine_systematic(self):
        """Theobromine should return full systematic IUPAC name."""
        smiles = 'Cn1cnc2c1c(=O)[nH]c(=O)n2C'
        result = name_compound(smiles)
        expected = '3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione'
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.integration
    def test_theobromine_positions(self):
        """Theobromine should have 3,7-dimethyl prefix."""
        smiles = 'Cn1cnc2c1c(=O)[nH]c(=O)n2C'
        result = name_compound(smiles)
        assert '3,7-dimethyl' in result, f"Expected '3,7-dimethyl' in name, got '{result}'"

    @pytest.mark.integration
    def test_theobromine_saturation(self):
        """Theobromine should have 3,7-dihydro prefix."""
        smiles = 'Cn1cnc2c1c(=O)[nH]c(=O)n2C'
        result = name_compound(smiles)
        assert '3,7-dihydro' in result, f"Expected '3,7-dihydro' in name, got '{result}'"


class TestXanthineFamily:
    """Parametrized tests for the xanthine family."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name,common_name", [
        # Caffeine
        ('Cn1cnc2c1c(=O)n(c(=O)n2C)C', '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione', 'caffeine'),
        # Theophylline
        ('Cn1c2c(c(=O)n(c1=O)C)[nH]cn2', '1,3-dimethyl-7H-purine-2,6-dione', 'theophylline'),
        # Theobromine
        ('Cn1cnc2c1c(=O)[nH]c(=O)n2C', '3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione', 'theobromine'),
        # Paraxanthine
        ('Cn1c(=O)[nH]c(=O)c2ncn(C)c12', '1,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione', 'paraxanthine'),
        # Xanthine (parent)
        ('O=c1[nH]c(=O)c2[nH]cnc2[nH]1', '3,7-dihydro-1H-purine-2,6-dione', 'xanthine'),
    ])
    def test_xanthine_derivatives(self, smiles, expected_name, common_name):
        """Test all xanthine derivatives are named correctly."""
        result = name_compound(smiles)
        assert result == expected_name, f"{common_name}: Expected '{expected_name}', got '{result}'"


class TestXanthineNamingFormat:
    """Test naming format rules for xanthines."""

    @pytest.mark.integration
    def test_numeric_locants_not_n_prefix(self):
        """Xanthines should use numeric locants (1,3,7) not N-methyl format."""
        # Caffeine has 3 N-methyl groups - should be "1,3,7-trimethyl"
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)

        # Should NOT have N-methyl format
        assert 'N-methyl' not in result, f"Found 'N-methyl' in name: {result}"
        assert 'N,N' not in result, f"Found 'N,N' in name: {result}"

        # Should have numeric locants
        assert '1,3,7-trimethyl' in result, f"Missing '1,3,7-trimethyl' in name: {result}"

    @pytest.mark.integration
    def test_locants_ordered_ascending(self):
        """N-position locants should be in ascending order."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)

        # Should be 1,3,7 not 7,3,1
        assert '1,3,7-' in result, f"Locants not in ascending order: {result}"

    @pytest.mark.integration
    def test_correct_prefix_ordering(self):
        """Name format: [substituents]-[saturation]-[indicated H]-[parent]."""
        smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
        result = name_compound(smiles)

        # Full name should follow this order
        expected_order = [
            '1,3,7-trimethyl',
            '3,7-dihydro',
            '1H-purine',
            '-2,6-dione'
        ]

        last_pos = -1
        for part in expected_order:
            pos = result.find(part)
            assert pos > last_pos, f"'{part}' not in correct position in: {result}"
            last_pos = pos


class TestXanthineEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_alternate_caffeine_smiles(self):
        """Different SMILES notations for caffeine should give same name."""
        caffeine_variants = [
            'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'Cn1c(=O)c2c(ncn2C)n(C)c1=O',  # Another canonical form
        ]
        expected = '1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'

        for smiles in caffeine_variants:
            result = name_compound(smiles)
            assert result == expected, f"SMILES '{smiles}' gave '{result}' instead of '{expected}'"

    @pytest.mark.integration
    def test_monomethyl_xanthines(self):
        """Test monomethyl xanthine derivatives."""
        # 7-methylxanthine
        smiles = 'Cn1cnc2c1c(=O)[nH]c(=O)[nH]2'
        result = name_compound(smiles)
        expected = '7-methyl-3,7-dihydro-1H-purine-2,6-dione'
        assert result == expected, f"7-methylxanthine: got '{result}'"

    @pytest.mark.integration
    def test_non_xanthine_purine_derivative(self):
        """Non-xanthine purine derivatives should use standard naming."""
        # Adenine (6-aminopurine) - should use retained name, not xanthine path
        smiles = 'Nc1ncnc2nc[nH]c12'
        result = name_compound(smiles)
        assert 'xanthine' not in result.lower(), f"Adenine incorrectly named as xanthine: {result}"
        # Should be 'adenine' (retained name)
        assert 'adenine' in result.lower(), f"Adenine should have retained name: {result}"
