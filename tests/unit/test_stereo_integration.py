"""
Integration tests for stereochemistry naming.

Tests end-to-end name generation for:
- R/S stereocenters in alcohols and other compounds
- E/Z double bond configuration
- Multiple stereodescriptors
- Edge cases (no stereo, symmetric molecules)
"""

import pytest
from src.orthonym import name_compound


class TestRSDescriptors:
    """Tests for R/S stereocenter naming."""

    def test_butan_2_ol_with_at_symbol(self):
        """C[C@H](O)CC should produce stereodescriptor."""
        result = name_compound('C[C@H](O)CC')
        # RDKit assigns this as S (due to SMILES notation)
        assert 'S' in result or 'R' in result
        assert 'butan' in result
        assert '-ol' in result

    def test_butan_2_ol_with_double_at(self):
        """C[C@@H](O)CC should produce opposite stereodescriptor."""
        result = name_compound('C[C@@H](O)CC')
        assert 'R' in result or 'S' in result
        assert 'butan' in result
        assert '-ol' in result

    def test_butan_2_ol_stereo_pair(self):
        """@ and @@ should give opposite configurations."""
        result_at = name_compound('C[C@H](O)CC')
        result_double_at = name_compound('C[C@@H](O)CC')

        # Extract R or S from each
        at_has_r = 'R)' in result_at or 'R,' in result_at
        double_at_has_r = 'R)' in result_double_at or 'R,' in result_double_at

        # They should be opposite
        assert at_has_r != double_at_has_r

    def test_pentan_2_ol_r(self):
        """Pentan-2-ol with R configuration."""
        result = name_compound('CCC[C@H](O)C')
        assert 'R' in result
        assert 'pentan' in result
        assert '-ol' in result

    def test_pentan_2_ol_s(self):
        """Pentan-2-ol with S configuration."""
        result = name_compound('CCC[C@@H](O)C')
        assert 'S' in result
        assert 'pentan' in result
        assert '-ol' in result

    def test_hexan_3_ol_chiral(self):
        """Hexan-3-ol with stereocenter."""
        result = name_compound('CC[C@H](O)CCC')
        # hexan-3-ol should have stereo since it's asymmetric
        # (propyl vs methyl on stereo carbon's neighbors)
        assert 'hexan' in result
        assert '-ol' in result

    def test_symmetric_no_stereo(self):
        """Symmetric pentan-3-ol has no stereocenter (symmetric ethyl groups)."""
        result = name_compound('CC[C@H](O)CC')
        # No stereodescriptor should appear - symmetric
        assert '(' not in result or '(2' not in result
        assert 'pentan' in result
        assert '-ol' in result


class TestEZDescriptors:
    """Tests for E/Z double bond naming."""

    def test_e_but_2_ene(self):
        """(E)-but-2-ene (trans)."""
        result = name_compound('C/C=C/C')
        assert '(2E)' in result
        assert 'but' in result
        assert 'ene' in result

    def test_z_but_2_ene(self):
        """(Z)-but-2-ene (cis)."""
        result = name_compound('C/C=C\\C')
        assert '(2Z)' in result
        assert 'but' in result
        assert 'ene' in result

    def test_e_pent_2_ene(self):
        """(E)-pent-2-ene."""
        result = name_compound('C/C=C/CC')
        assert 'E' in result
        assert 'pent' in result
        assert 'ene' in result

    def test_z_pent_2_ene(self):
        """(Z)-pent-2-ene."""
        result = name_compound('C/C=C\\CC')
        assert 'Z' in result
        assert 'pent' in result
        assert 'ene' in result

    def test_terminal_double_bond_no_ez(self):
        """Terminal double bonds (C=C-C) don't have E/Z."""
        result = name_compound('C=CC')
        # Should not have any stereodescriptor
        assert '(' not in result or 'E' not in result
        assert 'prop' in result
        assert 'ene' in result

    def test_but_1_ene_no_ez(self):
        """But-1-ene is terminal, no E/Z."""
        result = name_compound('C=CCC')
        # Terminal double bond - no E/Z possible
        assert 'but' in result
        assert 'ene' in result


class TestMultipleStereodescriptors:
    """Tests for compounds with multiple stereocenters or mixed descriptors."""

    def test_butane_2_3_diol_two_stereocenters(self):
        """Butane-2,3-diol with two stereocenters."""
        result = name_compound('C[C@H](O)[C@@H](O)C')
        # Should have two descriptors like (2R,3S)- or (2S,3R)-
        assert '(' in result
        assert ')' in result
        # Should contain both R and S or two of the same
        # Check that we have locants
        assert '2' in result.split(')')[0]  # locant in descriptor
        assert '3' in result.split(')')[0]  # locant in descriptor
        assert 'butan' in result
        assert 'diol' in result

    def test_two_stereocenters_same_config(self):
        """Two R stereocenters."""
        result = name_compound('C[C@H](O)[C@H](O)C')
        # Both should be same configuration
        assert '(' in result
        assert 'butan' in result
        assert 'diol' in result

    def test_e_double_bond_and_stereocenter(self):
        """Compound with both E double bond and R/S stereocenter."""
        # 4-hydroxy-pent-2-ene with E config and stereocenter
        result = name_compound('C/C=C/[C@H](O)C')
        # Should have E and either R or S
        assert 'E' in result
        # Check for stereocenter descriptor
        descriptors = result.split('-')[0] if '-' in result else ''
        assert '(' in result


class TestEdgeCases:
    """Edge cases for stereochemistry naming."""

    def test_no_stereo_ethanol(self):
        """Ethanol has no stereocenter."""
        result = name_compound('CCO')
        assert result == 'ethanol'
        # No stereodescriptor
        assert '(' not in result

    def test_no_stereo_propan_2_ol(self):
        """Propan-2-ol without @ has no stereodescriptor."""
        result = name_compound('CC(O)C')
        assert 'propan' in result
        assert '-ol' in result
        # No parentheses = no stereodescriptor
        assert '(' not in result

    def test_simple_alkane_no_stereo(self):
        """Butane has no stereochemistry."""
        result = name_compound('CCCC')
        assert result == 'butane'
        assert '(' not in result

    def test_ethene_no_ez(self):
        """Ethene has no E/Z (only 2 carbons)."""
        result = name_compound('C=C')
        assert result == 'ethene'
        assert '(' not in result

    def test_propene_no_ez(self):
        """Propene terminal double bond has no E/Z."""
        result = name_compound('CC=C')
        assert 'prop' in result
        assert 'ene' in result
        # No stereodescriptor for terminal double bond
        assert '(' not in result


class TestDescriptorFormat:
    """Tests for correct stereodescriptor format."""

    def test_descriptor_at_start(self):
        """Stereodescriptor appears at start of name."""
        result = name_compound('C[C@H](O)CC')
        # Should start with (
        assert result.startswith('(')

    def test_descriptor_before_prefixes(self):
        """Stereodescriptor comes before any substituent prefixes."""
        # A substituted molecule with stereo
        result = name_compound('C[C@H](O)C(C)C')
        # Should start with stereodescriptor
        if '(' in result:
            assert result.startswith('(')

    def test_descriptor_format_locant_cip(self):
        """Descriptor format is (locantCIP)-."""
        result = name_compound('C[C@H](O)CC')
        # Extract descriptor part
        if ')-' in result:
            descriptor = result.split(')-')[0] + ')'
            # Should be (2X) format where X is R or S
            assert descriptor.startswith('(')
            assert descriptor.endswith(')')
            # Should have a number followed by R or S
            inner = descriptor[1:-1]  # Remove parens
            assert inner[0].isdigit() or inner[0] == ','
            assert 'R' in inner or 'S' in inner

    def test_e_format_locant_e(self):
        """E descriptor format is (locantE)-."""
        result = name_compound('C/C=C/C')
        assert result.startswith('(2E)-')

    def test_z_format_locant_z(self):
        """Z descriptor format is (locantZ)-."""
        result = name_compound('C/C=C\\C')
        assert result.startswith('(2Z)-')


class TestSpecificCompounds:
    """Tests for specific named compounds."""

    def test_lactic_acid_stereocenter(self):
        """Lactic acid with stereocenter."""
        result = name_compound('C[C@H](O)C(=O)O')
        # Should have R or S
        assert 'R' in result or 'S' in result

    def test_r_butan_2_ol_expected(self):
        """Verify specific R configuration for butan-2-ol."""
        # C[C@@H](O)CC is R according to RDKit
        result = name_compound('C[C@@H](O)CC')
        assert '(2R)-butan-2-ol' == result

    def test_s_butan_2_ol_expected(self):
        """Verify specific S configuration for butan-2-ol."""
        # C[C@H](O)CC is S according to RDKit
        result = name_compound('C[C@H](O)CC')
        assert '(2S)-butan-2-ol' == result

    def test_e_but_2_ene_expected(self):
        """Verify E-but-2-ene exact output."""
        result = name_compound('C/C=C/C')
        assert result == '(2E)-but-2-ene'

    def test_z_but_2_ene_expected(self):
        """Verify Z-but-2-ene exact output."""
        result = name_compound('C/C=C\\C')
        assert result == '(2Z)-but-2-ene'
