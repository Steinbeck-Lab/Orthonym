"""
Unit tests for functional group detection.
"""

import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import (
    detect_functional_groups,
    has_functional_group,
    count_functional_groups,
)


@pytest.mark.unit
class TestFunctionalGroupDetection:
    """Tests for SMARTS-based functional group detection."""
    
    def test_carboxylic_acid_detection(self, mol_from_smiles):
        """Test detection of carboxylic acid."""
        mol = mol_from_smiles("CC(=O)O")  # acetic acid
        groups = detect_functional_groups(mol)
        
        assert "carboxylic_acid" in groups
        assert len(groups["carboxylic_acid"]) == 1
    
    def test_alcohol_detection(self, mol_from_smiles):
        """Test detection of alcohol."""
        mol = mol_from_smiles("CCO")  # ethanol
        groups = detect_functional_groups(mol)
        
        assert "primary_alcohol" in groups
        assert len(groups["primary_alcohol"]) == 1
    
    def test_ketone_detection(self, mol_from_smiles):
        """Test detection of ketone."""
        mol = mol_from_smiles("CC(=O)C")  # acetone
        groups = detect_functional_groups(mol)
        
        assert "ketone" in groups
        assert len(groups["ketone"]) == 1
    
    def test_aldehyde_detection(self, mol_from_smiles):
        """Test detection of aldehyde."""
        mol = mol_from_smiles("CC=O")  # acetaldehyde
        groups = detect_functional_groups(mol)
        
        assert "aldehyde" in groups
        assert len(groups["aldehyde"]) == 1
    
    def test_amine_detection(self, mol_from_smiles):
        """Test detection of amine."""
        mol = mol_from_smiles("CCN")  # ethylamine
        groups = detect_functional_groups(mol)
        
        assert "primary_amine" in groups
        assert len(groups["primary_amine"]) == 1
    
    def test_multiple_groups(self, mol_from_smiles):
        """Test detection of multiple functional groups."""
        mol = mol_from_smiles("OCC(=O)O")  # glycolic acid
        groups = detect_functional_groups(mol)
        
        assert "carboxylic_acid" in groups
        # Should also detect the alcohol
        alcohol_groups = [g for g in groups if "alcohol" in g]
        assert len(alcohol_groups) > 0
    
    def test_nitrile_detection(self, mol_from_smiles):
        """Test detection of nitrile."""
        mol = mol_from_smiles("CC#N")  # acetonitrile
        groups = detect_functional_groups(mol)
        
        assert "nitrile" in groups
        assert len(groups["nitrile"]) == 1
    
    def test_ester_detection(self, mol_from_smiles):
        """Test detection of ester."""
        mol = mol_from_smiles("CC(=O)OC")  # methyl acetate
        groups = detect_functional_groups(mol)
        
        assert "ester" in groups
        assert len(groups["ester"]) == 1
    
    def test_amide_detection(self, mol_from_smiles):
        """Test detection of amide."""
        mol = mol_from_smiles("CC(=O)N")  # acetamide
        groups = detect_functional_groups(mol)
        
        assert "primary_amide" in groups
        assert len(groups["primary_amide"]) == 1
    
    def test_alkene_detection(self, mol_from_smiles):
        """Test detection of alkene."""
        mol = mol_from_smiles("C=C")  # ethene
        groups = detect_functional_groups(mol)
        
        assert "alkene" in groups
        assert len(groups["alkene"]) == 1
    
    def test_alkyne_detection(self, mol_from_smiles):
        """Test detection of alkyne."""
        mol = mol_from_smiles("C#C")  # ethyne
        groups = detect_functional_groups(mol)
        
        assert "alkyne" in groups
        assert len(groups["alkyne"]) == 1
    
    def test_halogen_detection(self, mol_from_smiles):
        """Test detection of halogens."""
        mol = mol_from_smiles("CCCl")  # chloroethane
        groups = detect_functional_groups(mol)
        
        assert "chloro" in groups
        assert len(groups["chloro"]) == 1
    
    def test_no_groups_in_alkane(self, mol_from_smiles):
        """Test that simple alkane has no principal functional groups."""
        mol = mol_from_smiles("CCCC")  # butane
        groups = detect_functional_groups(mol)
        
        # Should not have any of the main functional groups
        assert "carboxylic_acid" not in groups
        assert "aldehyde" not in groups
        assert "ketone" not in groups


@pytest.mark.unit
class TestFunctionalGroupHelpers:
    """Tests for helper functions."""
    
    def test_has_functional_group(self, mol_from_smiles):
        mol = mol_from_smiles("CCO")
        
        assert has_functional_group(mol, "primary_alcohol")
        assert not has_functional_group(mol, "ketone")
    
    def test_count_functional_groups(self, mol_from_smiles):
        mol = mol_from_smiles("OCCO")  # ethylene glycol
        counts = count_functional_groups(mol)
        
        # Should count 2 alcohols
        total_alcohols = sum(
            counts.get(k, 0) for k in counts
            if "alcohol" in k
        )
        assert total_alcohols == 2
