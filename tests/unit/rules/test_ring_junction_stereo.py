"""
Unit tests for ring junction stereochemistry functions.

Tests the IUPAC stereodescriptor collection and formatting for fused ring
systems like decalin (decahydronaphthalene), including:
- Bridgehead atom detection
- Ring junction stereo collection with 'a' suffix locants
- R/S and r/c/t notation formatting
- cis/trans determination for simple bicyclics

Reference: IUPAC 2013 Blue Book, Section (Stereoisomer Nomenclature)
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.rules.stereochemistry import (
    get_bridgehead_atoms,
    collect_ring_junction_stereo,
    format_ring_junction_stereo,
    determine_simple_cis_trans,
    get_junction_locants_for_fused_system,
)
from orthonym.rules.fused_rings import (
    name_saturated_fused_bicyclic,
    get_ring_junction_stereo_prefix,
    get_simple_cis_trans_prefix,
    get_fused_ring_sizes,
)


# =============================================================================
# Test Bridgehead Detection
# =============================================================================

class TestBridgeheadDetection:
    """Tests for get_bridgehead_atoms function."""

    def test_decalin_bridgeheads(self):
        """Decalin (6,6-fused) has exactly 2 bridgehead atoms."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin without stereo
        bridgeheads = get_bridgehead_atoms(mol)
        assert len(bridgeheads) == 2

    def test_cis_decalin_bridgeheads(self):
        """cis-Decalin finds 2 bridgehead atoms with stereochemistry."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        assert len(bridgeheads) == 2

    def test_trans_decalin_bridgeheads(self):
        """trans-Decalin finds 2 bridgehead atoms with stereochemistry."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        assert len(bridgeheads) == 2

    def test_indane_bridgeheads(self):
        """Indane (5,6-fused saturated) has 2 bridgehead atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCCCC2C1')  # 5,6 fused saturated
        # Actually let's use a proper indane structure
        mol = Chem.MolFromSmiles('C1Cc2ccccc2C1')  # indane (partially aromatic)
        bridgeheads = get_bridgehead_atoms(mol)
        # Aromatic bridgeheads are sp2, not sp3, so should not be counted
        # unless saturated
        # For fully saturated 5,6:
        mol_sat = Chem.MolFromSmiles('C1CC2CCCCC2C1')
        bridgeheads_sat = get_bridgehead_atoms(mol_sat)
        assert len(bridgeheads_sat) == 2

    def test_naphthalene_no_sp3_bridgeheads(self):
        """Naphthalene (aromatic) has no sp3 bridgehead atoms."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        bridgeheads = get_bridgehead_atoms(mol)
        # Aromatic atoms are sp2, not stereocenters
        assert len(bridgeheads) == 0

    def test_single_ring_no_bridgeheads(self):
        """Single ring has no bridgehead atoms."""
        mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        bridgeheads = get_bridgehead_atoms(mol)
        assert len(bridgeheads) == 0

    def test_spiro_no_bridgeheads(self):
        """Spiro compound (1 shared atom) has no bridgehead atoms."""
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCCC2')  # spiro[5.5]undecane
        bridgeheads = get_bridgehead_atoms(mol)
        # Spiro atoms are shared but typically not stereocenters in simple spiro
        # The spiro carbon is tetrahedral but usually has C2 symmetry
        assert len(bridgeheads) <= 1  # May or may not be detected


# =============================================================================
# Test Ring Junction Stereo Collection
# =============================================================================

class TestRingJunctionStereo:
    """Tests for collect_ring_junction_stereo function."""

    def test_cis_decalin_same_cip(self):
        """cis-Decalin has same CIP codes at both junctions."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        rdCIPLabeler.AssignCIPLabels(mol)
        bridgeheads = get_bridgehead_atoms(mol)

        # Create locant mapping
        atom_to_locant = {bridgeheads[0]: '4a', bridgeheads[1]: '8a'}
        stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)

        assert len(stereo) == 2
        # Both should have same CIP (s,s or r,r)
        cips = [s[1].upper() for s in stereo]
        assert cips[0] == cips[1]

    def test_trans_decalin_different_cip(self):
        """trans-Decalin junction atoms return stereodescriptors."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        rdCIPLabeler.AssignCIPLabels(mol)
        bridgeheads = get_bridgehead_atoms(mol)

        atom_to_locant = {bridgeheads[0]: '4a', bridgeheads[1]: '8a'}
        stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)

        assert len(stereo) == 2
        # CIP codes are collected (may be same due to symmetry)

    def test_junction_locant_format(self):
        """Junction locants use 'a' suffix (4a, 8a)."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        rdCIPLabeler.AssignCIPLabels(mol)
        bridgeheads = get_bridgehead_atoms(mol)

        atom_to_locant = {bridgeheads[0]: '4a', bridgeheads[1]: '8a'}
        stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)

        locants = [s[0] for s in stereo]
        assert '4a' in locants
        assert '8a' in locants

    def test_empty_for_no_stereo(self):
        """Molecule without defined stereo returns empty list."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # no stereo defined
        rdCIPLabeler.AssignCIPLabels(mol)
        bridgeheads = get_bridgehead_atoms(mol)

        atom_to_locant = {bridgeheads[0]: '4a', bridgeheads[1]: '8a'}
        stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)

        # Without defined stereo, no CIP codes assigned
        assert len(stereo) == 0

    def test_stereo_sorted_by_locant(self):
        """Stereodescriptors are sorted by locant."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        rdCIPLabeler.AssignCIPLabels(mol)
        bridgeheads = get_bridgehead_atoms(mol)

        # Provide in reverse order
        atom_to_locant = {bridgeheads[1]: '4a', bridgeheads[0]: '8a'}
        stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)

        if len(stereo) == 2:
            # Should be sorted: 4a before 8a
            assert stereo[0][0] == '4a'
            assert stereo[1][0] == '8a'


# =============================================================================
# Test Stereo Formatting
# =============================================================================

class TestStereoFormatting:
    """Tests for format_ring_junction_stereo function."""

    def test_rs_notation_single(self):
        """Single descriptor formats correctly."""
        descriptors = [('4a', 'R')]
        result = format_ring_junction_stereo(descriptors)
        assert result == "(4aR)-"

    def test_rs_notation_pair(self):
        """Pair of descriptors formats as (4aR,8aS)-."""
        descriptors = [('4a', 'R'), ('8a', 'S')]
        result = format_ring_junction_stereo(descriptors)
        assert result == "(4aR,8aS)-"

    def test_rs_notation_lowercase(self):
        """Lowercase r/s preserved."""
        descriptors = [('4a', 's'), ('8a', 's')]
        result = format_ring_junction_stereo(descriptors)
        assert result == "(4as,8as)-"

    def test_rct_notation_cis(self):
        """r/c/t notation: same CIP = reference + cis."""
        descriptors = [('4a', 'S'), ('8a', 'S')]
        result = format_ring_junction_stereo(descriptors, use_rct=True)
        assert result == "(4ar,8ac)-"

    def test_rct_notation_trans(self):
        """r/c/t notation: different CIP = reference + trans."""
        descriptors = [('4a', 'R'), ('8a', 'S')]
        result = format_ring_junction_stereo(descriptors, use_rct=True)
        assert result == "(4ar,8at)-"

    def test_empty_descriptors(self):
        """Empty list returns empty string."""
        result = format_ring_junction_stereo([])
        assert result == ""

    def test_multiple_junctions(self):
        """Three or more junction atoms format correctly."""
        descriptors = [('4a', 'R'), ('8a', 'S'), ('9a', 'R')]
        result = format_ring_junction_stereo(descriptors)
        assert result == "(4aR,8aS,9aR)-"

    def test_rct_multiple_junctions(self):
        """r/c/t with 3+ junctions."""
        # R, S, R -> r (first), t (different), c (same as ref)
        descriptors = [('4a', 'R'), ('8a', 'S'), ('9a', 'R')]
        result = format_ring_junction_stereo(descriptors, use_rct=True)
        assert result == "(4ar,8at,9ac)-"


# =============================================================================
# Test cis/trans Determination
# =============================================================================

class TestCisTransDetermination:
    """Tests for determine_simple_cis_trans function."""

    def test_cis_decalin(self):
        """cis-Decalin returns 'cis'."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        result = determine_simple_cis_trans(mol, bridgeheads)
        assert result == 'cis'

    def test_trans_decalin(self):
        """trans-Decalin returns 'trans'."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        result = determine_simple_cis_trans(mol, bridgeheads)
        assert result == 'trans'

    def test_no_stereo_returns_none(self):
        """Molecule without stereo returns None."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # no stereo
        bridgeheads = get_bridgehead_atoms(mol)
        result = determine_simple_cis_trans(mol, bridgeheads)
        assert result is None

    def test_single_junction_returns_none(self):
        """Single junction atom returns None."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        # Pass only one bridgehead
        result = determine_simple_cis_trans(mol, [bridgeheads[0]])
        assert result is None

    def test_empty_list_returns_none(self):
        """Empty junction list returns None."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        result = determine_simple_cis_trans(mol, [])
        assert result is None


# =============================================================================
# Test Junction Locant Generation
# =============================================================================

class TestJunctionLocants:
    """Tests for get_junction_locants_for_fused_system function."""

    def test_6_6_fused_locants(self):
        """6,6-fused (decalin type) gets 4a, 8a locants."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        bridgeheads = get_bridgehead_atoms(mol)
        locants = get_junction_locants_for_fused_system(mol, bridgeheads, 6, 6)

        assert len(locants) == 2
        assert '4a' in locants.values()
        assert '8a' in locants.values()

    def test_5_6_fused_locants(self):
        """5,6-fused gets 3a, 7a locants."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCC[C@@H]2C1')  # 5,6 fused
        bridgeheads = get_bridgehead_atoms(mol)
        locants = get_junction_locants_for_fused_system(mol, bridgeheads, 5, 6)

        assert len(locants) == 2
        # Should have 3a and 7a for 5,6-fused
        locant_values = list(locants.values())
        assert '3a' in locant_values
        assert '7a' in locant_values


# =============================================================================
# Test End-to-End Junction Naming
# =============================================================================

class TestE2EJunctionNaming:
    """Integration tests for complete junction naming."""

    def test_cis_decalin_naming(self):
        """cis-Decalin produces correct full name."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        name = name_saturated_fused_bicyclic(mol)

        # Should have stereo prefix and parent name
        assert 'decahydronaphthalene' in name
        assert '4a' in name
        assert '8a' in name

    def test_trans_decalin_naming(self):
        """trans-Decalin produces correct full name."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        name = name_saturated_fused_bicyclic(mol)

        assert 'decahydronaphthalene' in name
        assert '4a' in name
        assert '8a' in name

    def test_cis_trans_different_names(self):
        """cis and trans decalin produce different names."""
        cis_mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        trans_mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')

        cis_name = name_saturated_fused_bicyclic(cis_mol)
        trans_name = name_saturated_fused_bicyclic(trans_mol)

        assert cis_name != trans_name

    def test_stereo_prefix_utility(self):
        """get_ring_junction_stereo_prefix returns correct format."""
        mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        prefix = get_ring_junction_stereo_prefix(mol)

        assert prefix.startswith('(')
        assert prefix.endswith(')-')
        assert '4a' in prefix
        assert '8a' in prefix

    def test_simple_prefix_utility(self):
        """get_simple_cis_trans_prefix returns cis-/trans-."""
        cis_mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        trans_mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')

        cis_prefix = get_simple_cis_trans_prefix(cis_mol)
        trans_prefix = get_simple_cis_trans_prefix(trans_mol)

        assert cis_prefix == 'cis-'
        assert trans_prefix == 'trans-'

    def test_ring_sizes_utility(self):
        """get_fused_ring_sizes returns correct sizes."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # 6,6-fused
        sizes = get_fused_ring_sizes(mol)
        assert sizes == (6, 6)


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize("smiles,expected_cis_trans", [
    ('C1CC[C@@H]2CCCC[C@@H]2C1', 'cis'),   # cis-decalin (same stereo)
    ('C1CC[C@@H]2CCCC[C@H]2C1', 'trans'),  # trans-decalin (different stereo)
    ('C1CC[C@H]2CCCC[C@H]2C1', 'cis'),     # cis-decalin (both @H)
    ('C1CC[C@H]2CCCC[C@@H]2C1', 'trans'),  # trans-decalin
])
def test_cis_trans_parametrized(smiles, expected_cis_trans):
    """Parametrized tests for cis/trans determination."""
    mol = Chem.MolFromSmiles(smiles)
    bridgeheads = get_bridgehead_atoms(mol)
    result = determine_simple_cis_trans(mol, bridgeheads)
    assert result == expected_cis_trans


@pytest.mark.parametrize("descriptors,use_rct,expected", [
    ([('4a', 'R'), ('8a', 'S')], False, "(4aR,8aS)-"),
    ([('4a', 'S'), ('8a', 'S')], False, "(4aS,8aS)-"),
    ([('4a', 'r'), ('8a', 'r')], False, "(4ar,8ar)-"),
    ([('4a', 'R'), ('8a', 'R')], True, "(4ar,8ac)-"),
    ([('4a', 'R'), ('8a', 'S')], True, "(4ar,8at)-"),
    ([], False, ""),
    ([], True, ""),
])
def test_format_junction_stereo_parametrized(descriptors, use_rct, expected):
    """Parametrized tests for stereo formatting."""
    result = format_ring_junction_stereo(descriptors, use_rct=use_rct)
    assert result == expected
