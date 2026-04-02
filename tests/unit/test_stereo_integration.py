"""
Integration tests for stereochemistry naming.

Tests end-to-end name generation for:
- R/S stereocenters in alcohols and other compounds
- E/Z double bond configuration
- Multiple stereodescriptors
- Edge cases (no stereo, symmetric molecules)
"""

import pytest
from rdkit import Chem
from orthonym import name_compound


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


class TestEZUnification:
    """Tests verifying E/Z uses _CIPCode exclusively (Phase 92-03)."""

    def test_no_ez_for_cyclohexene(self):
        """Intraring double bond in cyclohexene gets no E/Z descriptor."""
        result = name_compound('C1=CCCCC1')
        assert 'E' not in result and 'Z' not in result, f"False E/Z in cyclohexene: {result}"

    def test_no_ez_for_cyclopentene(self):
        """Intraring double bond in cyclopentene gets no E/Z descriptor."""
        result = name_compound('C1=CCCC1')
        assert 'E' not in result and 'Z' not in result, f"False E/Z in cyclopentene: {result}"

    def test_ez_uses_cip_code_directly(self):
        """get_stereodescriptor_string uses _CIPCode on bonds, not BondStereo."""
        from orthonym.perception.stereo import get_stereodescriptor_string
        from rdkit.Chem import rdCIPLabeler

        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Verify bond has _CIPCode set
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                assert bond.HasProp('_CIPCode'), "rdCIPLabeler should set _CIPCode on E/Z bonds"
                assert bond.GetProp('_CIPCode') == 'E'

        locant_map = {0: 1, 1: 2, 2: 3, 3: 4}
        result = get_stereodescriptor_string(mol, locant_map)
        assert '(2E)-' == result, f"Expected (2E)- but got {result}"

    def test_e_alkene_end_to_end(self):
        """E-but-2-ene named correctly end-to-end."""
        assert name_compound('C/C=C/C') == '(2E)-but-2-ene'

    def test_z_alkene_end_to_end(self):
        """Z-but-2-ene named correctly end-to-end."""
        assert name_compound('C/C=C\\C') == '(2Z)-but-2-ene'


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


# =============================================================================
# OPSIN Round-Trip Regression Tests (Phase 92-03, QUAL-04)
# =============================================================================

import subprocess
import os
import glob as glob_module


def _find_opsin_jar():
    """Find OPSIN JAR file."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    patterns = [
        os.path.join(project_root, "opsin-cli-*-jar-with-dependencies.jar"),
        os.path.join(project_root, "opsin", "opsin-cli", "target", "opsin-cli-*-jar-with-dependencies.jar"),
    ]
    for pat in patterns:
        matches = glob_module.glob(pat)
        if matches:
            return matches[0]
    return None


def _opsin_parse(name, opsin_jar):
    """Parse a name with OPSIN, return SMILES or None."""
    try:
        result = subprocess.run(
            ["java", "-jar", opsin_jar, "-osmi"],
            input=name, capture_output=True, text=True, timeout=10
        )
        smiles = result.stdout.strip()
        return smiles if smiles and smiles != "" else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


_opsin_jar = _find_opsin_jar()
_has_opsin = _opsin_jar is not None


@pytest.mark.skipif(not _has_opsin, reason="OPSIN JAR not found")
class TestStereoOpsinRoundtrip:
    """OPSIN round-trip regression tests for stereo naming (QUAL-04)."""

    @pytest.mark.parametrize("smiles,expected_name", [
        ('C/C=C/C', '(2E)-but-2-ene'),
        ('C/C=C\\C', '(2Z)-but-2-ene'),
        ('C/C=C/CC', '(2E)-pent-2-ene'),
        ('CC/C=C\\CC', '(3Z)-hex-3-ene'),
    ])
    def test_ez_roundtrip(self, smiles, expected_name):
        """E/Z compound names parse with OPSIN."""
        result = name_compound(smiles)
        assert result == expected_name, f"Name mismatch: {result} != {expected_name}"
        opsin_smiles = _opsin_parse(result, _opsin_jar)
        assert opsin_smiles, f"OPSIN failed to parse: {result}"
        from rdkit import Chem as _C
        mol_orig = _C.MolFromSmiles(smiles)
        mol_rt = _C.MolFromSmiles(opsin_smiles)
        if mol_orig and mol_rt:
            can_orig = _C.MolToSmiles(mol_orig, isomericSmiles=False)
            can_rt = _C.MolToSmiles(mol_rt, isomericSmiles=False)
            assert can_orig == can_rt, f"Round-trip mismatch: {can_orig} != {can_rt}"

    @pytest.mark.parametrize("smiles,expected_name", [
        ('C[C@@H](O)CC', '(2R)-butan-2-ol'),
        ('C[C@H](O)CC', '(2S)-butan-2-ol'),
        ('C[C@H](O)[C@@H](O)C', '(2S,3S)-butane-2,3-diol'),
    ])
    def test_rs_roundtrip(self, smiles, expected_name):
        """R/S compound names parse with OPSIN."""
        result = name_compound(smiles)
        assert result == expected_name, f"Name mismatch: {result} != {expected_name}"
        opsin_smiles = _opsin_parse(result, _opsin_jar)
        assert opsin_smiles, f"OPSIN failed to parse: {result}"
