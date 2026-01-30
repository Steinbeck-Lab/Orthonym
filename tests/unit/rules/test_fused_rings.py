"""
Unit tests for fused ring system classification and naming.

Tests cover:
- classify_fused_system() for ortho-fused, ortho-peri-fused detection
- is_fused_bicyclic() for 2-ring systems
- name_fused_heterocycle() with retained names and substituents
- N-substitution handling
- get_fused_heterocycle_substituents() for substituent detection
- name_ortho_fused_bicyclic() for systematic naming fallback
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.fused_rings import (
    classify_fused_system,
    is_fused_bicyclic,
    name_fused_heterocycle,
    get_fused_heterocycle_substituents,
    get_shared_atoms,
    name_ortho_fused_bicyclic,
    is_fused_aromatic_system,
    is_fused_heterocyclic_system,
)
from src.orthonym.data.fused_heterocycles import match_fused_heterocycle_core


# ============================================================================
# Test classify_fused_system()
# ============================================================================

class TestClassifyFusedSystem:
    """Tests for classify_fused_system function."""

    @pytest.mark.unit
    def test_indole_ortho_fused(self):
        """Indole (2 rings, 1 shared edge) should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        assert classify_fused_system(mol) == 'ortho-fused'

    @pytest.mark.unit
    def test_naphthalene_ortho_fused(self):
        """Naphthalene (2 rings, 1 shared edge) should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        assert classify_fused_system(mol) == 'ortho-fused'

    @pytest.mark.unit
    def test_quinoline_ortho_fused(self):
        """Quinoline should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        assert classify_fused_system(mol) == 'ortho-fused'

    @pytest.mark.unit
    def test_carbazole_ortho_fused(self):
        """Carbazole (3 rings, each pair shares 1 edge) should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')  # carbazole
        assert classify_fused_system(mol) == 'ortho-fused'

    @pytest.mark.unit
    def test_pyrene_ortho_peri_fused(self):
        """Pyrene (ring shares atoms with 3+ other rings) should be ortho-peri-fused."""
        mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')  # pyrene
        assert classify_fused_system(mol) == 'ortho-peri-fused'

    @pytest.mark.unit
    def test_cyclohexane_not_fused(self):
        """Single ring should not be fused."""
        mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        assert classify_fused_system(mol) == 'not-fused'

    @pytest.mark.unit
    def test_benzene_not_fused(self):
        """Benzene (single ring) should not be fused."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        assert classify_fused_system(mol) == 'not-fused'

    @pytest.mark.unit
    def test_biphenyl_not_fused(self):
        """Biphenyl (two rings connected by single bond) should not be fused."""
        mol = Chem.MolFromSmiles('c1ccc(cc1)c1ccccc1')  # biphenyl
        assert classify_fused_system(mol) == 'not-fused'

    @pytest.mark.unit
    def test_anthracene_ortho_fused(self):
        """Anthracene (linear 3-ring system) should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')  # anthracene
        assert classify_fused_system(mol) == 'ortho-fused'

    @pytest.mark.unit
    def test_phenanthrene_ortho_fused(self):
        """Phenanthrene (angular 3-ring system) should be ortho-fused."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)ccc1ccccc12')  # phenanthrene
        assert classify_fused_system(mol) == 'ortho-fused'


# ============================================================================
# Test is_fused_bicyclic()
# ============================================================================

class TestIsFusedBicyclic:
    """Tests for is_fused_bicyclic function."""

    @pytest.mark.unit
    def test_indole_is_bicyclic(self):
        """Indole should be detected as fused bicyclic."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        assert is_fused_bicyclic(mol) is True

    @pytest.mark.unit
    def test_quinoline_is_bicyclic(self):
        """Quinoline should be detected as fused bicyclic."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        assert is_fused_bicyclic(mol) is True

    @pytest.mark.unit
    def test_naphthalene_is_bicyclic(self):
        """Naphthalene should be detected as fused bicyclic."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert is_fused_bicyclic(mol) is True

    @pytest.mark.unit
    def test_carbazole_not_bicyclic(self):
        """Carbazole (tricyclic) should not be detected as bicyclic."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        assert is_fused_bicyclic(mol) is False

    @pytest.mark.unit
    def test_benzene_not_bicyclic(self):
        """Benzene (monocyclic) should not be detected as bicyclic."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert is_fused_bicyclic(mol) is False

    @pytest.mark.unit
    def test_anthracene_not_bicyclic(self):
        """Anthracene (tricyclic) should not be detected as bicyclic."""
        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        assert is_fused_bicyclic(mol) is False


# ============================================================================
# Test name_fused_heterocycle()
# ============================================================================

class TestNameFusedHeterocycle:
    """Tests for name_fused_heterocycle function."""

    @pytest.mark.unit
    def test_indole_name(self):
        """Indole should be named '1H-indole'."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        assert name_fused_heterocycle(mol) == '1H-indole'

    @pytest.mark.unit
    def test_quinoline_name(self):
        """Quinoline should be named 'quinoline'."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        assert name_fused_heterocycle(mol) == 'quinoline'

    @pytest.mark.unit
    def test_isoquinoline_name(self):
        """Isoquinoline should be named 'isoquinoline'."""
        mol = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        assert name_fused_heterocycle(mol) == 'isoquinoline'

    @pytest.mark.unit
    def test_carbazole_name(self):
        """Carbazole should be named '9H-carbazole'."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        assert name_fused_heterocycle(mol) == '9H-carbazole'

    @pytest.mark.unit
    def test_benzofuran_name(self):
        """Benzofuran should be named '1-benzofuran'."""
        mol = Chem.MolFromSmiles('c1ccc2occc2c1')
        assert name_fused_heterocycle(mol) == '1-benzofuran'

    @pytest.mark.unit
    def test_benzothiophene_name(self):
        """Benzothiophene should be named '1-benzothiophene'."""
        mol = Chem.MolFromSmiles('c1ccc2sccc2c1')
        assert name_fused_heterocycle(mol) == '1-benzothiophene'

    @pytest.mark.unit
    def test_benzimidazole_name(self):
        """Benzimidazole should be named '1H-benzimidazole'."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]cnc2c1')
        assert name_fused_heterocycle(mol) == '1H-benzimidazole'

    @pytest.mark.unit
    def test_purine_name(self):
        """Purine should be named '9H-purine'."""
        mol = Chem.MolFromSmiles('c1ncc2nc[nH]c2n1')
        assert name_fused_heterocycle(mol) == '9H-purine'

    @pytest.mark.unit
    def test_naphthalene_returns_none(self):
        """Naphthalene (carbocyclic) should return None for heterocycle naming."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert name_fused_heterocycle(mol) is None

    @pytest.mark.unit
    def test_benzene_returns_none(self):
        """Benzene should return None for heterocycle naming."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert name_fused_heterocycle(mol) is None


# ============================================================================
# Test substituted fused heterocycles
# ============================================================================

class TestSubstitutedFusedHeterocycles:
    """Tests for naming substituted fused heterocycles."""

    @pytest.mark.unit
    def test_5_methylindole(self):
        """5-methylindole should include locant and substituent."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        name = name_fused_heterocycle(mol)
        # Should contain 'methyl' and '1H-indole'
        assert name is not None
        assert 'methyl' in name
        assert '1H-indole' in name

    @pytest.mark.unit
    def test_3_methylindole(self):
        """3-methylindole (skatole) should be named correctly."""
        mol = Chem.MolFromSmiles('Cc1c[nH]c2ccccc12')  # 3-methylindole
        name = name_fused_heterocycle(mol)
        assert name is not None
        assert 'methyl' in name
        assert '1H-indole' in name

    @pytest.mark.unit
    def test_2_methylquinoline(self):
        """2-methylquinoline should be named correctly."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2n1')  # 2-methylquinoline
        name = name_fused_heterocycle(mol)
        assert name is not None
        assert 'methyl' in name
        assert 'quinoline' in name


# ============================================================================
# Test N-substitution handling
# ============================================================================

class TestNSubstitution:
    """Tests for N-substitution handling in fused heterocycles."""

    @pytest.mark.unit
    def test_n_methylindole(self):
        """N-methylindole should use N-locant format."""
        mol = Chem.MolFromSmiles('Cn1ccc2ccccc12')  # N-methylindole
        name = name_fused_heterocycle(mol)
        assert name is not None
        # Should have N-methyl format
        assert 'N-methyl' in name or '1-methyl' in name
        # The core name should be present
        assert 'indole' in name

    @pytest.mark.unit
    def test_n_methylcarbazole(self):
        """N-methylcarbazole should use N-locant format."""
        mol = Chem.MolFromSmiles('Cn1c2ccccc2c2ccccc12')  # N-methylcarbazole
        name = name_fused_heterocycle(mol)
        assert name is not None
        assert 'methyl' in name
        assert 'carbazole' in name


# ============================================================================
# Test get_fused_heterocycle_substituents()
# ============================================================================

class TestGetFusedHeterocycleSubstituents:
    """Tests for get_fused_heterocycle_substituents function."""

    @pytest.mark.unit
    def test_methylindole_substituent_detection(self):
        """Should detect methyl substituent on indole."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        core_name, atom_mapping, _core_smiles = result

        subs = get_fused_heterocycle_substituents(mol, atom_mapping)
        assert 'c_substituents' in subs
        # Should have methyl as a C-substituent
        assert 'methyl' in subs['c_substituents'] or len(subs['c_substituents']) > 0

    @pytest.mark.unit
    def test_n_methylindole_substituent_detection(self):
        """Should detect N-methyl substituent on indole."""
        mol = Chem.MolFromSmiles('Cn1ccc2ccccc12')  # N-methylindole
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        core_name, atom_mapping, _core_smiles = result

        subs = get_fused_heterocycle_substituents(mol, atom_mapping)
        assert 'n_substituents' in subs
        # Should have methyl as an N-substituent
        assert 'methyl' in subs['n_substituents'] or len(subs['n_substituents']) > 0


# ============================================================================
# Test get_shared_atoms()
# ============================================================================

class TestGetSharedAtoms:
    """Tests for get_shared_atoms helper function."""

    @pytest.mark.unit
    def test_naphthalene_shared_atoms(self):
        """Naphthalene rings should share exactly 2 atoms."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        assert len(rings) == 2

        shared = get_shared_atoms(mol, rings[0], rings[1])
        assert len(shared) == 2

    @pytest.mark.unit
    def test_indole_shared_atoms(self):
        """Indole rings should share exactly 2 atoms."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        assert len(rings) == 2

        shared = get_shared_atoms(mol, rings[0], rings[1])
        assert len(shared) == 2

    @pytest.mark.unit
    def test_biphenyl_no_shared_atoms(self):
        """Biphenyl rings should share 0 atoms."""
        mol = Chem.MolFromSmiles('c1ccc(cc1)c1ccccc1')
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        assert len(rings) == 2

        shared = get_shared_atoms(mol, rings[0], rings[1])
        assert len(shared) == 0


# ============================================================================
# Test name_ortho_fused_bicyclic()
# ============================================================================

class TestNameOrthoFusedBicyclic:
    """Tests for name_ortho_fused_bicyclic function."""

    @pytest.mark.unit
    def test_known_pah_defers(self):
        """Known PAHs should be handled by polycyclic naming, returning None."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
        # This function is a fallback - should return None for known PAHs
        # since they're handled by the main polycyclics module
        result = name_ortho_fused_bicyclic(mol)
        # Should either return None or be handled by fused_heterocycle
        assert result is None

    @pytest.mark.unit
    def test_fused_heterocycle_named(self):
        """Fused heterocycles should get names from retained name lookup."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        result = name_ortho_fused_bicyclic(mol)
        # Should delegate to name_fused_heterocycle
        assert result == '1H-indole'


# ============================================================================
# Test is_fused_aromatic_system() and is_fused_heterocyclic_system()
# ============================================================================

class TestFusedSystemDetection:
    """Tests for fused system detection functions."""

    @pytest.mark.unit
    def test_naphthalene_is_fused_aromatic(self):
        """Naphthalene should be detected as fused aromatic."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert is_fused_aromatic_system(mol) is True

    @pytest.mark.unit
    def test_indole_is_fused_aromatic(self):
        """Indole should be detected as fused aromatic."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        assert is_fused_aromatic_system(mol) is True

    @pytest.mark.unit
    def test_benzene_not_fused_aromatic(self):
        """Benzene should not be detected as fused aromatic."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert is_fused_aromatic_system(mol) is False

    @pytest.mark.unit
    def test_indole_is_fused_heterocyclic(self):
        """Indole should be detected as fused heterocyclic."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        assert is_fused_heterocyclic_system(mol) is True

    @pytest.mark.unit
    def test_naphthalene_not_fused_heterocyclic(self):
        """Naphthalene should not be detected as fused heterocyclic."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert is_fused_heterocyclic_system(mol) is False

    @pytest.mark.unit
    def test_quinoline_is_fused_heterocyclic(self):
        """Quinoline should be detected as fused heterocyclic."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        assert is_fused_heterocyclic_system(mol) is True


# ============================================================================
# Additional edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_none_molecule(self):
        """None molecule should be handled gracefully."""
        assert name_fused_heterocycle(None) is None

    @pytest.mark.unit
    def test_acyclic_molecule(self):
        """Acyclic molecule should not be fused."""
        mol = Chem.MolFromSmiles('CCCCCC')  # hexane
        assert classify_fused_system(mol) == 'not-fused'
        assert is_fused_bicyclic(mol) is False

    @pytest.mark.unit
    def test_indoline_saturated(self):
        """Indoline (saturated indole) should be recognized."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')  # indoline
        name = name_fused_heterocycle(mol)
        assert name == 'indoline'

    @pytest.mark.unit
    def test_tetrahydroquinoline(self):
        """Tetrahydroquinoline should be recognized."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # 1,2,3,4-tetrahydroquinoline
        name = name_fused_heterocycle(mol)
        assert name == '1,2,3,4-tetrahydroquinoline'

    @pytest.mark.unit
    def test_acridine_tricyclic(self):
        """Acridine should be recognized as tricyclic fused heterocycle."""
        mol = Chem.MolFromSmiles('c1ccc2nc3ccccc3cc2c1')  # acridine
        name = name_fused_heterocycle(mol)
        assert name == 'acridine'

    @pytest.mark.unit
    def test_phenazine_tricyclic(self):
        """Phenazine should be recognized as tricyclic fused heterocycle."""
        mol = Chem.MolFromSmiles('c1ccc2nc3ccccc3nc2c1')  # phenazine
        name = name_fused_heterocycle(mol)
        assert name == 'phenazine'
