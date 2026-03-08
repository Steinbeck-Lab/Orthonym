"""
Unit tests for ring assembly detection and naming (IUPAC P-28).

Tests cover:
  - Detection of identical ring systems connected by single bonds
  - Naming with primed locant notation (1,1'-biphenyl format)
  - Substituted ring assemblies (4-chloro-1,1'-biphenyl)
  - Negative cases: fused rings, different ring types
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.ring_assemblies import detect_ring_assembly, name_ring_assembly


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetection:
    """Tests for detect_ring_assembly()."""

    @pytest.mark.unit
    def test_biphenyl_detected(self):
        """Biphenyl (two identical benzene rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "carbocyclic"

    @pytest.mark.unit
    def test_bipyridine_detected(self):
        """Bipyridine (two identical pyridine rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccncc1-c2ccncc2")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"

    @pytest.mark.unit
    def test_bithiophene_detected(self):
        """Bithiophene (two identical thiophene rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccsc1-c1ccsc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"

    @pytest.mark.unit
    def test_naphthalene_not_assembly(self):
        """Naphthalene (fused rings, 1 ring system) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenylpyridine_not_assembly(self):
        """Phenylpyridine (different ring types) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccn2)cc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_biphenylene_not_assembly(self):
        """Biphenylene (fused, shared atoms) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2c(c1)-c1ccccc12")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenylcyclohexane_not_assembly(self):
        """Phenylcyclohexane (different aromaticity) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("C1CCCCC1c1ccccc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenanthrene_not_assembly(self):
        """Phenanthrene (fused tricyclic) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccccc12")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_single_ring_not_assembly(self):
        """A single ring (benzene) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_bifuran_detected(self):
        """Bifuran (two identical furan rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccoc1-c1ccoc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"


# ---------------------------------------------------------------------------
# Naming tests (E2E through name_compound)
# ---------------------------------------------------------------------------

class TestNaming:
    """Tests for ring assembly naming via name_compound()."""

    @pytest.mark.unit
    def test_biphenyl_name(self):
        """Biphenyl SMILES produces 'biphenyl' (retained name per P-31.1.2.4)."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert name == "biphenyl"

    @pytest.mark.unit
    def test_bipyridine_contains_bipyridine(self):
        """Bipyridine SMILES produces a name containing 'bipyridine'."""
        name = name_compound("c1ccncc1-c2ccncc2")
        assert "bipyridine" in name

    @pytest.mark.unit
    def test_22_bipyridine(self):
        """2,2'-bipyridine SMILES produces '2,2'-bipyridine'."""
        name = name_compound("c1ccc(-c2ccccn2)nc1")
        assert name == "2,2'-bipyridine"

    @pytest.mark.unit
    def test_bithiophene_contains_bithiophene(self):
        """Bithiophene SMILES produces a name containing 'bithiophene'."""
        name = name_compound("c1ccsc1-c1ccsc1")
        assert "bithiophene" in name

    @pytest.mark.unit
    def test_4_chlorobiphenyl(self):
        """4-chlorobiphenyl SMILES produces '4-chloro-1,1'-biphenyl'."""
        name = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert name == "4-chloro-1,1'-biphenyl"

    @pytest.mark.unit
    def test_biphenyl_uses_phenyl_not_benzene(self):
        """Biphenyl assembly name uses 'phenyl', not 'benzene'."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert "phenyl" in name
        assert "benzene" not in name

    @pytest.mark.unit
    def test_primed_locant_format(self):
        """Primed locants use ASCII apostrophe (U+0027).

        Uses substituted biphenyl since unsubstituted biphenyl is a retained name.
        """
        name = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert "'" in name  # ASCII apostrophe

    @pytest.mark.unit
    def test_bifuran_contains_bifuran(self):
        """Bifuran SMILES produces a name containing 'bifuran'."""
        name = name_compound("c1ccoc1-c1ccoc1")
        assert "bifuran" in name


# ---------------------------------------------------------------------------
# Boundary / negative naming tests
# ---------------------------------------------------------------------------

class TestNamingNegatives:
    """Tests that non-assemblies are NOT routed to assembly naming."""

    @pytest.mark.unit
    def test_phenanthrene_not_assembly_name(self):
        """Phenanthrene retains its PAH name, not misrouted to assembly."""
        name = name_compound("c1ccc2c(c1)ccc1ccccc12")
        assert name == "phenanthrene"

    @pytest.mark.unit
    def test_naphthalene_not_assembly_name(self):
        """Naphthalene retains its PAH name."""
        name = name_compound("c1ccc2ccccc2c1")
        assert name == "naphthalene"

    @pytest.mark.unit
    def test_phenylcyclohexane_unchanged(self):
        """Phenylcyclohexane is NOT named as assembly (different ring types)."""
        name = name_compound("C1CCCCC1c1ccccc1")
        # Should not contain 'bi' prefix for assembly naming
        assert "biphenyl" not in name
        assert "bicyclo" not in name.lower() or "cyclohex" in name
