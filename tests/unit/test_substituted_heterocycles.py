"""
Tests for substituted heterocycle naming (Plan 03-04).

Tests N-substitution (HETERO-09) and C-substitution for heterocyclic compounds:
- N-substitution uses N-locant format (N-methyl, N,N-dimethyl)
- C-substitution uses numeric locants (2-methyl, 3-ethyl)
- Saturation prefixes (dihydro-, tetrahydro-) with explicit locants

Reference: IUPAC 2013 Blue Book, Section P-22 (Heterocycles)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.heterocycles import (
    get_heterocycle_substituents,
    name_substituted_heterocycle,
    orient_heterocycle_with_substituents,
    name_heterocycle,
)


# =============================================================================
# HETERO-09: N-substitution tests
# =============================================================================

class TestNSubstitution:
    """Tests for N-substituted heterocycles (HETERO-09)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # N-methylpyrrolidine
        ("CN1CCCC1", "N-methylpyrrolidine"),
        # N-ethylpyrrolidine
        ("CCN1CCCC1", "N-ethylpyrrolidine"),
        # N-methylpiperidine
        ("CN1CCCCC1", "N-methylpiperidine"),
        # N-ethylpiperidine
        ("CCN1CCCCC1", "N-ethylpiperidine"),
        # N-methylmorpholine (N is at position 4 in morpholine, but uses N-locant)
        ("CN1CCOCC1", "N-methylmorpholine"),
        # N-methylazetidine
        ("CN1CCC1", "N-methylazetidine"),
        # N-methylaziridine
        ("CN1CC1", "N-methylaziridine"),
    ])
    def test_n_substituted_saturated(self, smiles, expected):
        """Test N-substituted saturated heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_n_substituent_detection(self):
        """Test that N-substituents are correctly identified."""
        mol = Chem.MolFromSmiles("CN1CCCC1")  # N-methylpyrrolidine
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        # Find substituent positions
        sub_positions = set()
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    sub_positions.add(idx)
                    break

        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )
        subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_locant)

        # Should have one substituent with is_on_nitrogen=True
        assert len(subs) == 1
        for locant, sub_list in subs.items():
            for sub_info in sub_list:
                assert sub_info['is_on_nitrogen'] is True
                assert sub_info['carbon_count'] == 1


# =============================================================================
# C-substitution tests
# =============================================================================

class TestCSubstitution:
    """Tests for C-substituted heterocycles."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Methylpyridine isomers
        ("Cc1ccccn1", "2-methylpyridine"),
        ("Cc1cccnc1", "3-methylpyridine"),
        ("Cc1ccncc1", "4-methylpyridine"),
        # Methylfuran isomers
        ("Cc1ccco1", "2-methylfuran"),
        ("Cc1ccoc1", "3-methylfuran"),
        # Methylthiophene isomers
        ("Cc1cccs1", "2-methylthiophene"),
        ("Cc1ccsc1", "3-methylthiophene"),
    ])
    def test_c_substituted_aromatic(self, smiles, expected):
        """Test C-substituted aromatic heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_c_substituent_detection(self):
        """Test that C-substituents are correctly identified."""
        mol = Chem.MolFromSmiles("Cc1ccccn1")  # 2-methylpyridine
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        sub_positions = set()
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    sub_positions.add(idx)
                    break

        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )
        subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_locant)

        # Should have one substituent with is_on_nitrogen=False
        assert len(subs) == 1
        for locant, sub_list in subs.items():
            for sub_info in sub_list:
                assert sub_info['is_on_nitrogen'] is False
                assert sub_info['carbon_count'] == 1


# =============================================================================
# Locant optimization tests
# =============================================================================

class TestLocantOptimization:
    """Tests for correct locant assignment with substituents."""

    @pytest.mark.unit
    def test_methylpyridine_locant_optimization(self):
        """Test that methylpyridine gets lowest possible locant."""
        # 2-methylpyridine should be locant 2, not 6
        mol = Chem.MolFromSmiles("Cc1ccccn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        sub_positions = set()
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    sub_positions.add(idx)
                    break

        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )
        subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_locant)

        # Methyl should be at locant 2
        locants = list(subs.keys())
        assert 2 in locants

    @pytest.mark.unit
    def test_unsubstituted_still_works(self):
        """Test that unsubstituted heterocycles still name correctly."""
        # Unsubstituted should use original orient_heterocycle behavior
        assert name_compound("c1ccncc1") == "pyridine"
        assert name_compound("C1CCNC1") == "pyrrolidine"
        assert name_compound("c1ccoc1") == "furan"


# =============================================================================
# Direct function tests
# =============================================================================

class TestSubstituentFunctions:
    """Tests for substituent handling functions."""

    @pytest.mark.unit
    def test_name_substituted_heterocycle_n_single(self):
        """Test name_substituted_heterocycle with single N-substituent."""
        mol = Chem.MolFromSmiles("CN1CCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        sub_positions = {idx for idx in ring
                        if any(n.GetIdx() not in ring_set
                               for n in mol.GetAtomWithIdx(idx).GetNeighbors())}

        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )
        subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_locant)
        parent = name_heterocycle(mol, ring)

        result = name_substituted_heterocycle(
            mol, ring, parent, subs, atom_to_locant
        )
        assert result == "N-methylpyrrolidine"

    @pytest.mark.unit
    def test_name_substituted_heterocycle_c_single(self):
        """Test name_substituted_heterocycle with single C-substituent."""
        mol = Chem.MolFromSmiles("Cc1ccccn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        sub_positions = {idx for idx in ring
                        if any(n.GetIdx() not in ring_set
                               for n in mol.GetAtomWithIdx(idx).GetNeighbors())}

        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )
        subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_locant)
        parent = name_heterocycle(mol, ring)

        result = name_substituted_heterocycle(
            mol, ring, parent, subs, atom_to_locant
        )
        assert result == "2-methylpyridine"

    @pytest.mark.unit
    def test_orient_with_substituents_vs_without(self):
        """Test that orient_with_substituents gives lower locants than basic orient."""
        mol = Chem.MolFromSmiles("Cc1ccccn1")  # 2-methylpyridine
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_set = set(ring)

        sub_positions = {idx for idx in ring
                        if any(n.GetIdx() not in ring_set
                               for n in mol.GetAtomWithIdx(idx).GetNeighbors())}

        # With substituent consideration
        oriented_with, atl_with = orient_heterocycle_with_substituents(
            mol, ring, sub_positions
        )

        # Get the methyl's locant
        subs = get_heterocycle_substituents(mol, ring, oriented_with, atl_with)
        methyl_locant = list(subs.keys())[0]

        # Should be 2 (lowest possible), not 6
        assert methyl_locant == 2


# =============================================================================
# Ethyl and larger substituents
# =============================================================================

class TestLargerSubstituents:
    """Tests for ethyl and larger substituents on heterocycles."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Ethyl substituents
        ("CCc1ccncc1", "4-ethylpyridine"),
        ("CCc1cccnc1", "3-ethylpyridine"),
        ("CCc1ccccn1", "2-ethylpyridine"),
        # Propyl substituents
        ("CCCc1ccncc1", "4-propylpyridine"),
    ])
    def test_larger_alkyl_substituents(self, smiles, expected):
        """Test ethyl and larger alkyl substituents."""
        assert name_compound(smiles) == expected


# =============================================================================
# Regression tests
# =============================================================================

class TestSubstitutedHeterocycleRegression:
    """Regression tests to ensure substitution doesn't break existing naming."""

    @pytest.mark.unit
    def test_unsubstituted_heterocycles_unchanged(self):
        """Test that adding substituent support doesn't break unsubstituted naming."""
        # 5-membered
        assert name_compound("c1ccoc1") == "furan"
        assert name_compound("c1cc[nH]c1") == "1H-pyrrole"  # v23 IH-01: leading indicated-H
        assert name_compound("c1ccsc1") == "thiophene"
        assert name_compound("C1CCOC1") == "oxolane"
        assert name_compound("C1CCNC1") == "pyrrolidine"

        # 6-membered
        assert name_compound("c1ccncc1") == "pyridine"
        assert name_compound("C1CCNCC1") == "piperidine"
        assert name_compound("C1CCOCC1") == "oxane"
        assert name_compound("C1COCCN1") == "morpholine"

        # 3 and 4 membered
        assert name_compound("C1CO1") == "oxirane"
        assert name_compound("C1COC1") == "oxetane"

    @pytest.mark.unit
    def test_non_heterocycles_unchanged(self):
        """Test that heterocycle changes don't break non-heterocycles."""
        # Alkanes
        assert name_compound("CC") == "ethane"
        assert name_compound("CCC") == "propane"

        # Cycloalkanes
        assert name_compound("C1CCCCC1") == "cyclohexane"

        # Benzene
        assert name_compound("c1ccccc1") == "benzene"


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for substituted heterocycles."""

    @pytest.mark.unit
    def test_empty_substituents(self):
        """Test that empty substituent dict returns parent name."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, atom_to_locant = orient_heterocycle_with_substituents(mol, ring, set())
        parent = name_heterocycle(mol, ring)

        result = name_substituted_heterocycle(
            mol, ring, parent, {}, atom_to_locant
        )
        assert result == "pyridine"

    @pytest.mark.unit
    def test_substituent_on_multiple_rings_not_yet_supported(self):
        """Test behavior with potentially complex multi-ring systems."""
        # For now, just test single ring case works
        # Multi-ring heterocycles are Phase 6+
        mol = Chem.MolFromSmiles("CN1CCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert len(ring) == 5  # Single 5-membered ring
