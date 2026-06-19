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
        """Biphenyl SMILES produces the PIN '1,1'-biphenyl' (F-T9/DD6 RET-01: bare
        'biphenyl' is general-only; the ring-assembly PIN is 1,1'-biphenyl, P-28.2.1)."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert name == "1,1'-biphenyl"

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


# ---------------------------------------------------------------------------
# RING-01: Higher multiplier tests (quinque through deci)
# ---------------------------------------------------------------------------

class TestHigherMultipliers:
    """Tests for ASSEMBLY_MULTIPLIERS extension to 5-10 (IUPAC P-28.2)."""

    @pytest.mark.unit
    def test_quinque_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[5] == 'quinque'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[5] == "quinque"

    @pytest.mark.unit
    def test_sexi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[6] == 'sexi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[6] == "sexi"

    @pytest.mark.unit
    def test_septi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[7] == 'septi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[7] == "septi"

    @pytest.mark.unit
    def test_octi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[8] == 'octi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[8] == "octi"

    @pytest.mark.unit
    def test_novi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[9] == 'novi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[9] == "novi"

    @pytest.mark.unit
    def test_deci_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[10] == 'deci'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[10] == "deci"

    @pytest.mark.unit
    def test_all_nine_entries(self):
        """ASSEMBLY_MULTIPLIERS has exactly 9 entries (2 through 10)."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert len(ASSEMBLY_MULTIPLIERS) == 9
        for k in range(2, 11):
            assert k in ASSEMBLY_MULTIPLIERS

    @pytest.mark.unit
    def test_quinquepyridine_name(self):
        """name_ring_assembly() with 5 identical pyridine rings returns name
        containing 'quinquepyridine'."""
        # 5 pyridines linked: c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1
        smiles = "c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect 5-ring pyridine assembly"
        assert info["count"] == 5
        result = name_ring_assembly(mol, info, None)
        assert result is not None
        assert "quinquepyridine" in result

    @pytest.mark.unit
    def test_quinquephenyl_prefix(self):
        """name_ring_assembly() with 5 identical benzene rings returns name
        containing 'quinquephenyl'."""
        from orthonym.rules.ring_assemblies import name_ring_assembly_prefix
        smiles = "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccccc5)cc4)cc3)cc2)cc1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect 5-ring benzene assembly"
        assert info["count"] == 5
        result = name_ring_assembly(mol, info, None)
        assert result is not None
        assert "quinquephenyl" in result

    @pytest.mark.unit
    def test_count_11_unsupported(self):
        """ASSEMBLY_MULTIPLIERS does not contain count=11."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert 11 not in ASSEMBLY_MULTIPLIERS

    @pytest.mark.unit
    def test_regression_biphenyl(self):
        """Regression: biphenyl detection + naming produces the PIN 1,1'-biphenyl
        (F-T9/DD6 RET-01)."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert name == "1,1'-biphenyl"

    @pytest.mark.unit
    def test_regression_bipyridine(self):
        """Regression: bipyridine detection and naming still works."""
        name = name_compound("c1ccncc1-c2ccncc2")
        assert "bipyridine" in name


# ---------------------------------------------------------------------------
# RING-02: Fused ring assembly unit tests
# ---------------------------------------------------------------------------

class TestFusedRingAssemblyUnits:
    """Tests for fused ring systems as ring assembly units (IUPAC P-28.1)."""

    @pytest.mark.unit
    def test_get_ring_parent_name_naphthalene(self):
        """_get_ring_parent_name() with naphthalene system atoms returns 'naphthalene'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        # naphthalene is a single ring system; all atoms form the system
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for naphthalene"
        assert name == "naphthalene"

    @pytest.mark.unit
    def test_get_ring_parent_name_quinoline(self):
        """_get_ring_parent_name() with quinoline system atoms returns 'quinoline'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for quinoline"
        assert name == "quinoline"

    @pytest.mark.unit
    def test_get_ring_parent_name_indole(self):
        """_get_ring_parent_name() with indole system atoms returns a name containing 'indole'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for indole"
        assert "indole" in name

    @pytest.mark.unit
    def test_binaphthalene_detection(self):
        """detect_ring_assembly() on two identical naphthalene systems returns count==2."""
        # 1,1'-binaphthalene
        smiles = "c1ccc2ccccc2c1-c1ccc2ccccc2c1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect binaphthalene as 2-ring assembly"
        assert info["count"] == 2

    @pytest.mark.unit
    def test_binaphthalene_name(self):
        """name_compound() on binaphthalene SMILES produces name containing 'binaphthalene'."""
        smiles = "c1ccc2ccccc2c1-c1ccc2ccccc2c1"
        name = name_compound(smiles)
        assert name is not None
        assert "binaphthalene" in name or "binaphthalen" in name, \
            f"Expected 'binaphthalene' in name, got: {name}"

    @pytest.mark.unit
    def test_biquinoline_detection(self):
        """detect_ring_assembly() on two identical quinoline systems returns count==2."""
        smiles = "c1ccc2ncccc2c1-c1ccc2ncccc2c1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect biquinoline as 2-ring assembly"
        assert info["count"] == 2
