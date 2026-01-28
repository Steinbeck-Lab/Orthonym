"""
Unit tests for cycloalkene naming.

Tests cover:
- Simple cycloalkenes (3-8 membered rings)
- Cycloalkadienes with locants
- Locant omission for mono-cycloalkenes
- Cycloalkene numbering rules
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import classify_ring, get_ring_double_bond_atoms
from orthonym.rules.cycloalkanes import orient_cycloalkene, get_ring_double_bonds


# ============================================================================
# Simple Cycloalkene Tests (Sizes 3-8)
# ============================================================================

class TestSimpleCycloalkenes:
    """Tests for basic cycloalkene naming (ring sizes 3-8)."""

    @pytest.mark.unit
    def test_cyclopropene(self):
        """C1=CC1 -> cyclopropene (3-membered ring with double bond)."""
        assert name_compound("C1=CC1") == "cyclopropene"

    @pytest.mark.unit
    def test_cyclobutene(self):
        """C1=CCC1 -> cyclobutene (4-membered ring with double bond)."""
        assert name_compound("C1=CCC1") == "cyclobutene"

    @pytest.mark.unit
    def test_cyclopentene(self):
        """C1=CCCC1 -> cyclopentene (5-membered ring with double bond)."""
        assert name_compound("C1=CCCC1") == "cyclopentene"

    @pytest.mark.unit
    def test_cyclohexene(self):
        """C1=CCCCC1 -> cyclohexene (6-membered ring with double bond)."""
        assert name_compound("C1=CCCCC1") == "cyclohexene"

    @pytest.mark.unit
    def test_cycloheptene(self):
        """C1=CCCCCC1 -> cycloheptene (7-membered ring with double bond)."""
        assert name_compound("C1=CCCCCC1") == "cycloheptene"

    @pytest.mark.unit
    def test_cyclooctene(self):
        """C1=CCCCCCC1 -> cyclooctene (8-membered ring with double bond)."""
        assert name_compound("C1=CCCCCCC1") == "cyclooctene"


# ============================================================================
# Cycloalkadiene Tests (With Locants)
# ============================================================================

class TestCycloalkadienesWithLocants:
    """Tests for cycloalkadienes - require locants for multiple double bonds."""

    @pytest.mark.unit
    def test_cyclopentadiene(self):
        """C1=CC=CC1 -> cyclopenta-1,3-diene (locants required for dienes)."""
        assert name_compound("C1=CC=CC1") == "cyclopenta-1,3-diene"

    @pytest.mark.unit
    def test_cyclohexadiene_1_3(self):
        """C1=CC=CCC1 -> cyclohexa-1,3-diene (conjugated diene)."""
        assert name_compound("C1=CC=CCC1") == "cyclohexa-1,3-diene"

    @pytest.mark.unit
    def test_cyclohexadiene_1_4(self):
        """C1=CCC=CC1 -> cyclohexa-1,4-diene (non-conjugated diene)."""
        assert name_compound("C1=CCC=CC1") == "cyclohexa-1,4-diene"

    @pytest.mark.unit
    def test_cyclohepta_1_3_diene(self):
        """C1=CC=CCCC1 -> cyclohepta-1,3-diene."""
        assert name_compound("C1=CC=CCCC1") == "cyclohepta-1,3-diene"


# ============================================================================
# Locant Omission Tests
# ============================================================================

class TestCycloalkeneNumberingOmission:
    """Tests verifying that mono-cycloalkenes do NOT include locants."""

    @pytest.mark.unit
    def test_no_locant_cyclohexene(self):
        """Result should be 'cyclohexene' not 'cyclohex-1-ene'."""
        result = name_compound("C1=CCCCC1")
        assert result == "cyclohexene"
        assert "-1-" not in result

    @pytest.mark.unit
    def test_no_locant_cyclopentene(self):
        """Result should be 'cyclopentene' not 'cyclopent-1-ene'."""
        result = name_compound("C1=CCCC1")
        assert result == "cyclopentene"
        assert "-1-" not in result

    @pytest.mark.unit
    def test_no_locant_cyclobutene(self):
        """Result should be 'cyclobutene' not 'cyclobut-1-ene'."""
        result = name_compound("C1=CCC1")
        assert result == "cyclobutene"
        assert "-1-" not in result

    @pytest.mark.unit
    def test_dienes_require_locants(self):
        """Cycloalkadienes MUST include locants."""
        result = name_compound("C1=CC=CCC1")
        assert "1,3" in result or "1,4" in result


# ============================================================================
# Ring Classification Tests for Cycloalkenes
# ============================================================================

class TestCycloalkeneClassification:
    """Tests for correct classification of cycloalkenes."""

    @pytest.mark.unit
    def test_classify_simple_cycloalkene(self):
        """Single double bond in ring -> cycloalkene."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkene"

    @pytest.mark.unit
    def test_classify_diene_as_cycloalkene(self):
        """Two double bonds in ring (non-aromatic) -> cycloalkene."""
        mol = Chem.MolFromSmiles("C1=CC=CCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert classify_ring(mol, ring) == "cycloalkene"


# ============================================================================
# Ring Double Bond Detection Tests
# ============================================================================

class TestRingDoubleBondDetection:
    """Tests for get_ring_double_bonds and get_ring_double_bond_atoms."""

    @pytest.mark.unit
    def test_single_double_bond_detection(self):
        """Detect single double bond in cyclohexene."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        double_bonds = get_ring_double_bonds(mol, ring)
        assert len(double_bonds) == 1

    @pytest.mark.unit
    def test_two_double_bonds_detection(self):
        """Detect two double bonds in cyclohexadiene."""
        mol = Chem.MolFromSmiles("C1=CC=CCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        double_bonds = get_ring_double_bonds(mol, ring)
        assert len(double_bonds) == 2

    @pytest.mark.unit
    def test_double_bond_atoms_function(self):
        """get_ring_double_bond_atoms returns same result."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        double_bonds = get_ring_double_bond_atoms(mol, ring)
        assert len(double_bonds) == 1
        # Each double bond is a tuple of two atom indices
        a1, a2 = double_bonds[0]
        assert isinstance(a1, int)
        assert isinstance(a2, int)


# ============================================================================
# Cycloalkene Orientation Tests
# ============================================================================

class TestCycloalkeneOrientation:
    """Tests for orient_cycloalkene function."""

    @pytest.mark.unit
    def test_double_bond_at_position_1_2(self):
        """Double bond should be at positions 1-2 in oriented ring."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        double_bonds = get_ring_double_bonds(mol, ring)
        oriented = orient_cycloalkene(mol, ring, double_bonds)

        # The first two atoms should be connected by a double bond
        bond = mol.GetBondBetweenAtoms(oriented[0], oriented[1])
        assert bond is not None
        assert bond.GetBondType() == Chem.BondType.DOUBLE

    @pytest.mark.unit
    def test_orientation_returns_same_atoms(self):
        """Oriented ring should contain same atoms as original."""
        mol = Chem.MolFromSmiles("C1=CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        double_bonds = get_ring_double_bonds(mol, ring)
        oriented = orient_cycloalkene(mol, ring, double_bonds)

        assert set(oriented) == set(ring)


# ============================================================================
# Alternative SMILES Notation Tests
# ============================================================================

class TestCycloalkeneAlternativeSMILES:
    """Tests ensuring different valid SMILES give the same name."""

    @pytest.mark.unit
    def test_cyclohexene_different_notation(self):
        """Different SMILES notations for cyclohexene."""
        # Different ways to write cyclohexene
        assert name_compound("C1=CCCCC1") == "cyclohexene"
        assert name_compound("C=1CCCCC1") == "cyclohexene"

    @pytest.mark.unit
    def test_cyclopentene_different_notation(self):
        """Different SMILES notations for cyclopentene."""
        assert name_compound("C1=CCCC1") == "cyclopentene"
