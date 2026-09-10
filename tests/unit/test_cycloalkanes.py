"""
Unit tests for cycloalkane naming and ring classification.

Tests cover:
- Ring classification (cycloalkane, cycloalkene, aromatic, heterocyclic)
- Simple cycloalkane naming (3-10 ring sizes)
- Edge cases for larger rings (11-12+)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import classify_ring


# ============================================================================
# Ring Classification Tests
# ============================================================================

class TestRingClassification:
    """Tests for classify_ring function."""

    @pytest.mark.unit
    def test_classify_cyclopropane(self):
        """C1CC1 is a saturated carbocyclic ring -> cycloalkane."""
        mol = Chem.MolFromSmiles("C1CC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkane"

    @pytest.mark.unit
    def test_classify_cyclohexane(self):
        """C1CCCCC1 is a saturated carbocyclic ring -> cycloalkane."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkane"

    @pytest.mark.unit
    def test_classify_cyclopentane(self):
        """C1CCCC1 is a saturated carbocyclic ring -> cycloalkane."""
        mol = Chem.MolFromSmiles("C1CCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkane"

    @pytest.mark.unit
    def test_classify_cyclohexene(self):
        """C1=CCCCC1 has a double bond -> cycloalkene."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkene"

    @pytest.mark.unit
    def test_classify_cyclopentene(self):
        """C1=CCCC1 has a double bond -> cycloalkene."""
        mol = Chem.MolFromSmiles("C1=CCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkene"

    @pytest.mark.unit
    def test_classify_cyclopentadiene(self):
        """C1=CC=CC1 has two double bonds but not aromatic -> cycloalkene."""
        mol = Chem.MolFromSmiles("C1=CC=CC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkene"

    @pytest.mark.unit
    def test_classify_benzene(self):
        """c1ccccc1 is aromatic -> aromatic."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "aromatic"

    @pytest.mark.unit
    def test_classify_oxolane(self):
        """C1CCOC1 contains oxygen -> heterocyclic_saturated."""
        mol = Chem.MolFromSmiles("C1CCOC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = classify_ring(mol, ring)
        assert result == "heterocyclic_saturated"
        assert result.startswith("heterocyclic")

    @pytest.mark.unit
    def test_classify_pyridine(self):
        """c1ccncc1 contains nitrogen (heterocyclic takes priority over aromatic)."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = classify_ring(mol, ring)
        assert result == "heterocyclic_aromatic"
        assert result.startswith("heterocyclic")

    @pytest.mark.unit
    def test_classify_piperidine(self):
        """C1CCNCC1 is saturated heterocyclic -> heterocyclic_saturated."""
        mol = Chem.MolFromSmiles("C1CCNCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = classify_ring(mol, ring)
        assert result == "heterocyclic_saturated"
        assert result.startswith("heterocyclic")

    @pytest.mark.unit
    def test_classify_furan(self):
        """c1ccoc1 is aromatic heterocyclic -> heterocyclic_aromatic."""
        mol = Chem.MolFromSmiles("c1ccoc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = classify_ring(mol, ring)
        assert result == "heterocyclic_aromatic"
        assert result.startswith("heterocyclic")

    @pytest.mark.unit
    def test_classify_thiophene(self):
        """c1ccsc1 is aromatic heterocyclic -> heterocyclic_aromatic."""
        mol = Chem.MolFromSmiles("c1ccsc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = classify_ring(mol, ring)
        assert result == "heterocyclic_aromatic"
        assert result.startswith("heterocyclic")


# ============================================================================
# Simple Cycloalkane Naming Tests
# ============================================================================

class TestSimpleCycloalkanes:
    """Tests for basic cycloalkane naming (ring sizes 3-10)."""

    @pytest.mark.unit
    def test_cyclopropane(self):
        """C1CC1 -> cyclopropane (3-membered ring)."""
        assert name_compound("C1CC1") == "cyclopropane"

    @pytest.mark.unit
    def test_cyclobutane(self):
        """C1CCC1 -> cyclobutane (4-membered ring)."""
        assert name_compound("C1CCC1") == "cyclobutane"

    @pytest.mark.unit
    def test_cyclopentane(self):
        """C1CCCC1 -> cyclopentane (5-membered ring)."""
        assert name_compound("C1CCCC1") == "cyclopentane"

    @pytest.mark.unit
    def test_cyclohexane(self):
        """C1CCCCC1 -> cyclohexane (6-membered ring)."""
        assert name_compound("C1CCCCC1") == "cyclohexane"

    @pytest.mark.unit
    def test_cycloheptane(self):
        """C1CCCCCC1 -> cycloheptane (7-membered ring)."""
        assert name_compound("C1CCCCCC1") == "cycloheptane"

    @pytest.mark.unit
    def test_cyclooctane(self):
        """C1CCCCCCC1 -> cyclooctane (8-membered ring)."""
        assert name_compound("C1CCCCCCC1") == "cyclooctane"

    @pytest.mark.unit
    def test_cyclononane(self):
        """C1CCCCCCCC1 -> cyclononane (9-membered ring)."""
        assert name_compound("C1CCCCCCCC1") == "cyclononane"

    @pytest.mark.unit
    def test_cyclodecane(self):
        """C1CCCCCCCCC1 -> cyclodecane (10-membered ring)."""
        assert name_compound("C1CCCCCCCCC1") == "cyclodecane"


# ============================================================================
# Edge Cases - Large Rings
# ============================================================================

class TestCycloalkaneEdgeCases:
    """Tests for cycloalkane naming edge cases (large rings)."""

    @pytest.mark.unit
    def test_large_ring_11(self):
        """C1CCCCCCCCCC1 -> cycloundecane (11-membered ring)."""
        assert name_compound("C1CCCCCCCCCC1") == "cycloundecane"

    @pytest.mark.unit
    def test_large_ring_12(self):
        """C1CCCCCCCCCCC1 -> cyclododecane (12-membered ring)."""
        assert name_compound("C1CCCCCCCCCCC1") == "cyclododecane"

    @pytest.mark.unit
    def test_large_ring_13(self):
        """C1CCCCCCCCCCCC1 -> cyclotridecane (13-membered ring)."""
        assert name_compound("C1CCCCCCCCCCCC1") == "cyclotridecane"

    @pytest.mark.unit
    def test_large_ring_14(self):
        """C1CCCCCCCCCCCCC1 -> cyclotetradecane (14-membered ring)."""
        assert name_compound("C1CCCCCCCCCCCCC1") == "cyclotetradecane"

    @pytest.mark.unit
    def test_large_ring_15(self):
        """C1CCCCCCCCCCCCCC1 -> cyclopentadecane (15-membered ring)."""
        assert name_compound("C1CCCCCCCCCCCCCC1") == "cyclopentadecane"


# ============================================================================
# Alternative SMILES Notation Tests
# ============================================================================

class TestCycloalkaneAlternativeSMILES:
    """Tests ensuring different valid SMILES give the same name."""

    @pytest.mark.unit
    def test_cyclohexane_different_start(self):
        """Different ring numbering in SMILES should give same result."""
        # These are all cyclohexane with different starting points
        assert name_compound("C1CCCCC1") == "cyclohexane"
        # Note: C(C1CCCC1) is methylcyclopentane, NOT cyclohexane!

    @pytest.mark.unit
    def test_cyclopentane_different_notation(self):
        """Different SMILES notations for cyclopentane."""
        assert name_compound("C1CCCC1") == "cyclopentane"

    @pytest.mark.unit
    def test_cyclobutane_different_notation(self):
        """Different SMILES notations for cyclobutane."""
        assert name_compound("C1CCC1") == "cyclobutane"
