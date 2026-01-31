"""
Unit tests for benzene functionalized chain detection.

Tests the _identify_functionalized_chain() and _detect_chain_functional_group()
functions that detect chains with functional groups attached to benzene rings.
"""

import pytest
from rdkit import Chem
from orthonym.rules.benzene import (
    get_benzene_substituents,
    is_benzene_ring,
    _identify_functionalized_chain,
    _detect_chain_functional_group,
)


def _get_benzene_ring_and_chain_start(smiles):
    """Helper to get benzene ring atoms and the first chain atom attached to the ring."""
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = None
    for ring in mol.GetRingInfo().AtomRings():
        if is_benzene_ring(mol, ring):
            ring_atoms = set(ring)
            break

    # Find the chain start - a carbon attached to the ring but not in it
    chain_start = None
    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in ring_atoms and neighbor.GetSymbol() == 'C':
                chain_start = nbr_idx
                break
        if chain_start is not None:
            break

    return mol, ring_atoms, chain_start


class TestIdentifyFunctionalizedChain:
    """Test _identify_functionalized_chain() function."""

    def test_carboxylic_acid_chain_detected(self):
        """Detect butanoic acid chain on benzene."""
        mol, ring_atoms, chain_start = _get_benzene_ring_and_chain_start('c1ccc(CCCC(=O)O)cc1')

        result = _identify_functionalized_chain(mol, chain_start, ring_atoms)

        assert result is not None
        assert result['name'] == 'functionalized_chain'
        assert result['functional_group'] == 'carboxylic_acid'
        assert result['chain_length'] == 4  # 4 carbons in butanoic acid chain

    def test_alcohol_chain_detected(self):
        """Detect propanol chain on benzene."""
        mol, ring_atoms, chain_start = _get_benzene_ring_and_chain_start('c1ccc(CCCO)cc1')

        result = _identify_functionalized_chain(mol, chain_start, ring_atoms)

        assert result is not None
        assert result['name'] == 'functionalized_chain'
        assert result['functional_group'] == 'alcohol'
        assert result['chain_length'] == 3  # 3 carbons

    def test_aldehyde_chain_detected(self):
        """Detect acetaldehyde chain on benzene (phenylacetaldehyde)."""
        mol, ring_atoms, chain_start = _get_benzene_ring_and_chain_start('c1ccc(CC=O)cc1')

        result = _identify_functionalized_chain(mol, chain_start, ring_atoms)

        assert result is not None
        assert result['name'] == 'functionalized_chain'
        assert result['functional_group'] == 'aldehyde'
        assert result['chain_length'] == 2  # 2 carbons

    def test_pure_alkyl_returns_none(self):
        """Pure alkyl chain should return None (handled by alkyl detection)."""
        mol, ring_atoms, chain_start = _get_benzene_ring_and_chain_start('c1ccc(CCCC)cc1')

        result = _identify_functionalized_chain(mol, chain_start, ring_atoms)

        assert result is None  # No heteroatoms = not a functionalized chain

    def test_single_carbon_carboxylic_acid(self):
        """Detect phenylacetic acid chain."""
        mol, ring_atoms, chain_start = _get_benzene_ring_and_chain_start('c1ccc(CC(=O)O)cc1')

        result = _identify_functionalized_chain(mol, chain_start, ring_atoms)

        assert result is not None
        assert result['functional_group'] == 'carboxylic_acid'
        assert result['chain_length'] == 2


class TestDetectChainFunctionalGroup:
    """Test _detect_chain_functional_group() function."""

    def test_detect_carboxylic_acid(self):
        """Carboxylic acid C(=O)O pattern detected."""
        mol = Chem.MolFromSmiles('CCCC(=O)O')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'carboxylic_acid'

    def test_detect_alcohol(self):
        """Alcohol -OH pattern detected."""
        mol = Chem.MolFromSmiles('CCCO')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'alcohol'

    def test_detect_aldehyde(self):
        """Aldehyde C=O with H pattern detected."""
        mol = Chem.MolFromSmiles('CCC=O')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'aldehyde'

    def test_pure_hydrocarbon_returns_none(self):
        """Pure hydrocarbon should return None."""
        mol = Chem.MolFromSmiles('CCCC')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result is None

    def test_formic_acid_pattern(self):
        """Formic acid (simplest carboxylic acid)."""
        mol = Chem.MolFromSmiles('C(=O)O')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'carboxylic_acid'

    def test_methanol_pattern(self):
        """Methanol (simplest alcohol)."""
        mol = Chem.MolFromSmiles('CO')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'alcohol'

    def test_formaldehyde_pattern(self):
        """Formaldehyde (simplest aldehyde)."""
        mol = Chem.MolFromSmiles('C=O')
        chain_atoms = list(range(mol.GetNumAtoms()))

        result = _detect_chain_functional_group(mol, chain_atoms)

        assert result == 'aldehyde'


class TestBenzeneSubstituentsWithFunctionalizedChain:
    """Integration tests for get_benzene_substituents with functionalized chains."""

    def test_phenylbutanoic_acid_substituent_detected(self):
        """Benzene with butanoic acid chain has substituent detected."""
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        assert ring_atoms is not None
        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0, 'Should detect substituent'
        # Get the first (and only) substituent
        sub_list = list(subs.values())[0]
        sub = sub_list[0]
        assert sub['name'] == 'functionalized_chain'
        assert sub['functional_group'] == 'carboxylic_acid'

    def test_phenylpropanol_substituent_detected(self):
        """Benzene with propanol chain has substituent detected."""
        mol = Chem.MolFromSmiles('c1ccc(CCCO)cc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0
        sub = list(subs.values())[0][0]
        assert sub['functional_group'] == 'alcohol'

    def test_phenylacetaldehyde_substituent_detected(self):
        """Benzene with acetaldehyde chain has substituent detected."""
        mol = Chem.MolFromSmiles('c1ccc(CC=O)cc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0
        sub = list(subs.values())[0][0]
        assert sub['functional_group'] == 'aldehyde'

    def test_toluene_still_detected(self):
        """Existing detection: toluene methyl group still works."""
        mol = Chem.MolFromSmiles('Cc1ccccc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0
        # Check any substituent has name 'methyl'
        all_subs = [s for sub_list in subs.values() for s in sub_list]
        assert any(sub.get('name') == 'methyl' for sub in all_subs)

    def test_chlorobenzene_still_detected(self):
        """Existing detection: chlorobenzene chloro group still works."""
        mol = Chem.MolFromSmiles('Clc1ccccc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0
        all_subs = [s for sub_list in subs.values() for s in sub_list]
        assert any(sub.get('name') == 'chloro' for sub in all_subs)

    def test_ethylbenzene_still_detected(self):
        """Existing detection: ethylbenzene ethyl group still works."""
        mol = Chem.MolFromSmiles('CCc1ccccc1')
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        assert len(subs) > 0
        all_subs = [s for sub_list in subs.values() for s in sub_list]
        assert any(sub.get('name') == 'ethyl' for sub in all_subs)

    def test_multiple_functionalized_chains_not_supported_yet(self):
        """Document that multiple functionalized chains are detected independently."""
        # This test documents current behavior - each substituent is independent
        mol = Chem.MolFromSmiles('c1ccc(CCO)cc1')  # Just one chain for now
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if is_benzene_ring(mol, ring):
                ring_atoms = ring
                break

        subs = get_benzene_substituents(mol, ring_atoms)

        # Should work with single functionalized chain
        assert len(subs) == 1
