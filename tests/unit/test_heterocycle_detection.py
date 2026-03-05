"""
Unit tests for heterocycle detection, classification, and ring numbering.

Tests the heterocycles.py module which provides:
- classify_heterocycle: Ring analysis (size, aromaticity, heteroatoms)
- number_heterocycle_ring: IUPAC ring numbering
- orient_heterocycle: Combined classification + numbering
- get_heteroatom_locants: Locant extraction for heteroatoms

IUPAC 2013 heterocycle numbering rules:
- Position 1 goes to highest-priority heteroatom
- Priority order: O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B
- Direction chosen to minimize locants for other heteroatoms
- First-point-of-difference comparison (not sum of locants)
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.heterocycles import (
    classify_heterocycle,
    number_heterocycle_ring,
    orient_heterocycle,
    get_heteroatom_locants,
)


# =============================================================================
# CLASSIFICATION TESTS
# =============================================================================


class TestClassifyHeterocycle:
    """Tests for classify_heterocycle function."""

    # -------------------------------------------------------------------------
    # 3-Membered Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_oxirane_classification(self):
        """Oxirane (ethylene oxide) - 3-membered, saturated, 1 oxygen."""
        mol = Chem.MolFromSmiles("C1CO1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 3
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "O"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_aziridine_classification(self):
        """Aziridine - 3-membered, saturated, 1 nitrogen."""
        mol = Chem.MolFromSmiles("C1CN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 3
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_thiirane_classification(self):
        """Thiirane (ethylene sulfide) - 3-membered, saturated, 1 sulfur."""
        mol = Chem.MolFromSmiles("C1CS1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 3
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "S"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    # -------------------------------------------------------------------------
    # 5-Membered Aromatic Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_furan_classification(self):
        """Furan - 5-membered aromatic, 1 oxygen."""
        mol = Chem.MolFromSmiles("c1ccoc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "O"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_pyrrole_classification(self):
        """Pyrrole - 5-membered aromatic, 1 nitrogen."""
        mol = Chem.MolFromSmiles("c1cc[nH]c1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_thiophene_classification(self):
        """Thiophene - 5-membered aromatic, 1 sulfur."""
        mol = Chem.MolFromSmiles("c1ccsc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "S"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_imidazole_classification(self):
        """Imidazole - 5-membered aromatic, 2 nitrogens."""
        mol = Chem.MolFromSmiles("c1cnc[nH]1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_oxazole_classification(self):
        """Oxazole - 5-membered aromatic, 1 oxygen + 1 nitrogen."""
        mol = Chem.MolFromSmiles("c1cnco1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "O"  # O > N
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_thiazole_classification(self):
        """Thiazole - 5-membered aromatic, 1 sulfur + 1 nitrogen."""
        mol = Chem.MolFromSmiles("c1cncs1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "S"  # S > N
        assert info["is_aromatic"] is True

    # -------------------------------------------------------------------------
    # 5-Membered Saturated Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_oxolane_classification(self):
        """Tetrahydrofuran (THF) - 5-membered saturated, 1 oxygen."""
        mol = Chem.MolFromSmiles("C1CCOC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "O"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_pyrrolidine_classification(self):
        """Pyrrolidine - 5-membered saturated, 1 nitrogen."""
        mol = Chem.MolFromSmiles("C1CCNC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_tetrahydrothiophene_classification(self):
        """Tetrahydrothiophene (thiolane) - 5-membered saturated, 1 sulfur."""
        mol = Chem.MolFromSmiles("C1CCSC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 5
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "S"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    # -------------------------------------------------------------------------
    # 6-Membered Aromatic Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_pyridine_classification(self):
        """Pyridine - 6-membered aromatic, 1 nitrogen."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_pyrimidine_classification(self):
        """Pyrimidine - 6-membered aromatic, 2 nitrogens at 1,3."""
        mol = Chem.MolFromSmiles("c1ncncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_pyrazine_classification(self):
        """Pyrazine - 6-membered aromatic, 2 nitrogens at 1,4."""
        mol = Chem.MolFromSmiles("c1cnccn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_pyridazine_classification(self):
        """Pyridazine - 6-membered aromatic, 2 nitrogens at 1,2."""
        mol = Chem.MolFromSmiles("c1ccnnc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    @pytest.mark.unit
    def test_triazine_1_3_5_classification(self):
        """1,3,5-Triazine - 6-membered aromatic, 3 nitrogens."""
        mol = Chem.MolFromSmiles("c1ncncn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 3
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is True

    # -------------------------------------------------------------------------
    # 6-Membered Saturated Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_piperidine_classification(self):
        """Piperidine - 6-membered saturated, 1 nitrogen."""
        mol = Chem.MolFromSmiles("C1CCNCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_oxane_classification(self):
        """Tetrahydropyran - 6-membered saturated, 1 oxygen."""
        mol = Chem.MolFromSmiles("C1CCOCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "O"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_morpholine_classification(self):
        """Morpholine - 6-membered saturated, 1 nitrogen + 1 oxygen."""
        mol = Chem.MolFromSmiles("C1COCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "O"  # O > N
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_piperazine_classification(self):
        """Piperazine - 6-membered saturated, 2 nitrogens."""
        mol = Chem.MolFromSmiles("C1CNCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "N"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_dioxane_classification(self):
        """1,4-Dioxane - 6-membered saturated, 2 oxygens."""
        mol = Chem.MolFromSmiles("C1COCCO1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 6
        assert info["num_heteroatoms"] == 2
        assert info["dominant_heteroatom"] == "O"
        assert info["is_aromatic"] is False
        assert info["is_saturated"] is True

    # -------------------------------------------------------------------------
    # 4-Membered Rings
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_oxetane_classification(self):
        """Oxetane - 4-membered saturated, 1 oxygen."""
        mol = Chem.MolFromSmiles("C1COC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 4
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "O"
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_azetidine_classification(self):
        """Azetidine - 4-membered saturated, 1 nitrogen."""
        mol = Chem.MolFromSmiles("C1CNC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 4
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"
        assert info["is_saturated"] is True

    @pytest.mark.unit
    def test_thietane_classification(self):
        """Thietane - 4-membered saturated, 1 sulfur."""
        mol = Chem.MolFromSmiles("C1CSC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 4
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "S"
        assert info["is_saturated"] is True


# =============================================================================
# NUMBERING TESTS
# =============================================================================


class TestNumberHeterocycleRing:
    """Tests for number_heterocycle_ring and orient_heterocycle functions."""

    # -------------------------------------------------------------------------
    # Single Heteroatom Numbering
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_pyridine_numbering_n_at_position_1(self):
        """Pyridine: N should be at position 1."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented = number_heterocycle_ring(mol, ring)

        # Position 1 (index 0) should be N
        assert mol.GetAtomWithIdx(oriented[0]).GetSymbol() == "N"

    @pytest.mark.unit
    def test_furan_numbering_o_at_position_1(self):
        """Furan: O should be at position 1."""
        mol = Chem.MolFromSmiles("c1ccoc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented = number_heterocycle_ring(mol, ring)

        assert mol.GetAtomWithIdx(oriented[0]).GetSymbol() == "O"

    @pytest.mark.unit
    def test_thiophene_numbering_s_at_position_1(self):
        """Thiophene: S should be at position 1."""
        mol = Chem.MolFromSmiles("c1ccsc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented = number_heterocycle_ring(mol, ring)

        assert mol.GetAtomWithIdx(oriented[0]).GetSymbol() == "S"

    @pytest.mark.unit
    def test_piperidine_numbering_n_at_position_1(self):
        """Piperidine: N should be at position 1."""
        mol = Chem.MolFromSmiles("C1CCNCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented = number_heterocycle_ring(mol, ring)

        assert mol.GetAtomWithIdx(oriented[0]).GetSymbol() == "N"

    # -------------------------------------------------------------------------
    # Multiple Identical Heteroatoms - Locant Optimization
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_pyrimidine_numbering_locants_1_3(self):
        """Pyrimidine: N atoms should get locants [1, 3]."""
        mol = Chem.MolFromSmiles("c1ncncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        # Should be [1, 3], not [1, 5]
        assert [loc for loc, _ in locants] == [1, 3]

    @pytest.mark.unit
    def test_pyrazine_numbering_locants_1_4(self):
        """Pyrazine: N atoms should get locants [1, 4]."""
        mol = Chem.MolFromSmiles("c1cnccn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 4]

    @pytest.mark.unit
    def test_pyridazine_numbering_locants_1_2(self):
        """Pyridazine: N atoms should get locants [1, 2]."""
        mol = Chem.MolFromSmiles("c1ccnnc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 2]

    @pytest.mark.unit
    def test_piperazine_numbering_locants_1_4(self):
        """Piperazine: N atoms should get locants [1, 4]."""
        mol = Chem.MolFromSmiles("C1CNCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 4]

    @pytest.mark.unit
    def test_dioxane_numbering_locants_1_4(self):
        """1,4-Dioxane: O atoms should get locants [1, 4]."""
        mol = Chem.MolFromSmiles("C1COCCO1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 4]

    @pytest.mark.unit
    def test_imidazole_numbering_locants_1_3(self):
        """Imidazole: N atoms should get locants [1, 3]."""
        mol = Chem.MolFromSmiles("c1cnc[nH]1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 3]

    @pytest.mark.unit
    def test_triazine_135_numbering_locants_1_3_5(self):
        """1,3,5-Triazine: N atoms should get locants [1, 3, 5]."""
        mol = Chem.MolFromSmiles("c1ncncn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert [loc for loc, _ in locants] == [1, 3, 5]

    # -------------------------------------------------------------------------
    # Different Heteroatoms - Priority Rules
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_morpholine_numbering_o_at_1_n_at_4(self):
        """Morpholine: O at position 1 (higher priority), N at position 4."""
        mol = Chem.MolFromSmiles("C1COCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)

        # Position 1 should be O
        assert mol.GetAtomWithIdx(oriented[0]).GetSymbol() == "O"

        # Get all heteroatom locants
        locants = get_heteroatom_locants(oriented, mol)
        locant_dict = {elem: loc for loc, elem in locants}

        assert locant_dict["O"] == 1
        assert locant_dict["N"] == 4

    @pytest.mark.unit
    def test_oxazole_numbering_o_at_1_n_at_3(self):
        """Oxazole: O at position 1, N at position 3."""
        mol = Chem.MolFromSmiles("c1cnco1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        locant_dict = {elem: loc for loc, elem in locants}

        assert locant_dict["O"] == 1
        assert locant_dict["N"] == 3

    @pytest.mark.unit
    def test_thiazole_numbering_s_at_1_n_at_3(self):
        """Thiazole: S at position 1, N at position 3."""
        mol = Chem.MolFromSmiles("c1cncs1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        locant_dict = {elem: loc for loc, elem in locants}

        assert locant_dict["S"] == 1
        assert locant_dict["N"] == 3

    @pytest.mark.unit
    def test_isoxazole_numbering_o_at_1_n_at_2(self):
        """Isoxazole: O at position 1, N at position 2."""
        mol = Chem.MolFromSmiles("c1ccno1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        locant_dict = {elem: loc for loc, elem in locants}

        assert locant_dict["O"] == 1
        assert locant_dict["N"] == 2

    @pytest.mark.unit
    def test_isothiazole_numbering_s_at_1_n_at_2(self):
        """Isothiazole: S at position 1, N at position 2."""
        mol = Chem.MolFromSmiles("c1ccns1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        locant_dict = {elem: loc for loc, elem in locants}

        assert locant_dict["S"] == 1
        assert locant_dict["N"] == 2


# =============================================================================
# ORIENT HETEROCYCLE TESTS
# =============================================================================


class TestOrientHeterocycle:
    """Tests for orient_heterocycle convenience function."""

    @pytest.mark.unit
    def test_orient_returns_tuple(self):
        """orient_heterocycle returns (oriented_ring, atom_to_locant) tuple."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        result = orient_heterocycle(mol, ring)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], dict)

    @pytest.mark.unit
    def test_orient_mapping_is_complete(self):
        """atom_to_locant mapping covers all ring atoms."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)

        # All ring atoms should be in mapping
        for atom_idx in ring:
            assert atom_idx in mapping

        # Locants should be 1 through ring_size
        assert set(mapping.values()) == set(range(1, len(ring) + 1))

    @pytest.mark.unit
    def test_orient_mapping_matches_oriented_ring(self):
        """atom_to_locant mapping is consistent with oriented_ring order."""
        mol = Chem.MolFromSmiles("C1COCCN1")  # morpholine
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, mapping = orient_heterocycle(mol, ring)

        # Check each position
        for locant, atom_idx in enumerate(oriented, 1):
            assert mapping[atom_idx] == locant


# =============================================================================
# GET HETEROATOM LOCANTS TESTS
# =============================================================================


class TestGetHeteroatomLocants:
    """Tests for get_heteroatom_locants function."""

    @pytest.mark.unit
    def test_single_heteroatom_returns_one_tuple(self):
        """Single heteroatom returns list with one (locant, element) tuple."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert len(locants) == 1
        assert locants[0] == (1, "N")

    @pytest.mark.unit
    def test_multiple_heteroatoms_returns_sorted_tuples(self):
        """Multiple heteroatoms return sorted list of tuples."""
        mol = Chem.MolFromSmiles("C1COCCN1")  # morpholine
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        assert len(locants) == 2
        # Should be sorted by locant
        assert locants[0][0] < locants[1][0]

    @pytest.mark.unit
    def test_locants_are_sorted_ascending(self):
        """Heteroatom locants are always in ascending order."""
        mol = Chem.MolFromSmiles("c1ncncc1")  # pyrimidine
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        locs = [loc for loc, _ in locants]
        assert locs == sorted(locs)


# =============================================================================
# DIRECTION SELECTION TESTS
# =============================================================================


class TestDirectionSelection:
    """Tests for clockwise vs counterclockwise direction selection."""

    @pytest.mark.unit
    def test_direction_minimizes_other_heteroatom_locants(self):
        """Direction chosen to give lowest locants to non-position-1 heteroatoms."""
        # Pyrimidine: if N at 1, other N could be at 3 or 5 depending on direction
        # Should choose [1, 3] over [1, 5]
        mol = Chem.MolFromSmiles("c1ncncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        locs = [loc for loc, _ in locants]
        assert locs == [1, 3]

    @pytest.mark.unit
    def test_first_point_of_difference_comparison(self):
        """Uses first-point-of-difference, not sum-of-locants."""
        # 1,3,5-Triazine: [1, 3, 5] vs any other would lose
        mol = Chem.MolFromSmiles("c1ncncn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)

        locs = [loc for loc, _ in locants]
        assert locs == [1, 3, 5]

    @pytest.mark.unit
    def test_symmetric_ring_gives_consistent_result(self):
        """Symmetric rings give deterministic, consistent numbering."""
        # Pyrazine is symmetric - should still give consistent result
        mol = Chem.MolFromSmiles("c1cnccn1")
        ring = mol.GetRingInfo().AtomRings()[0]

        # Run multiple times - should be consistent
        results = []
        for _ in range(5):
            oriented, _ = orient_heterocycle(mol, ring)
            locants = get_heteroatom_locants(oriented, mol)
            results.append([loc for loc, _ in locants])

        # All results should be identical
        assert all(r == results[0] for r in results)
        assert results[0] == [1, 4]


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    @pytest.mark.unit
    def test_empty_ring_returns_empty(self):
        """Empty ring input returns empty list."""
        mol = Chem.MolFromSmiles("C")
        oriented = number_heterocycle_ring(mol, [], [])
        assert oriented == []

    @pytest.mark.unit
    def test_no_heteroatoms_returns_original(self):
        """Ring with no heteroatoms returns original order."""
        # This shouldn't happen in practice, but test the fallback
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented = number_heterocycle_ring(mol, ring)

        # Should return same atoms (though order may be preserved)
        assert set(oriented) == set(ring)

    @pytest.mark.unit
    def test_7_membered_ring(self):
        """7-membered heterocycle is classified correctly."""
        # Azepane (7-membered saturated with N)
        mol = Chem.MolFromSmiles("C1CCCNCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        assert info["ring_size"] == 7
        assert info["num_heteroatoms"] == 1
        assert info["dominant_heteroatom"] == "N"

    @pytest.mark.unit
    def test_classification_heteroatoms_use_atom_indices(self):
        """classify_heterocycle returns atom indices, not ring positions."""
        mol = Chem.MolFromSmiles("C1COCCN1")  # morpholine
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)

        # heteroatoms should be (atom_idx, element) tuples
        for atom_idx, elem in info["heteroatoms"]:
            atom = mol.GetAtomWithIdx(atom_idx)
            assert atom.GetSymbol() == elem


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_dominant,expected_ring_size",
    [
        ("C1CO1", "O", 3),  # oxirane
        ("C1CN1", "N", 3),  # aziridine
        ("C1CS1", "S", 3),  # thiirane
        ("C1COC1", "O", 4),  # oxetane
        ("C1CNC1", "N", 4),  # azetidine
        ("C1CSC1", "S", 4),  # thietane
        ("c1ccoc1", "O", 5),  # furan
        ("c1cc[nH]c1", "N", 5),  # pyrrole
        ("c1ccsc1", "S", 5),  # thiophene
        ("C1CCOC1", "O", 5),  # oxolane
        ("C1CCNC1", "N", 5),  # pyrrolidine
        ("c1ccncc1", "N", 6),  # pyridine
        ("c1ncncc1", "N", 6),  # pyrimidine
        ("C1CCOCC1", "O", 6),  # oxane
        ("C1CCNCC1", "N", 6),  # piperidine
        ("C1COCCN1", "O", 6),  # morpholine (O > N)
        ("C1CNCCN1", "N", 6),  # piperazine
        ("c1cnco1", "O", 5),  # oxazole (O > N)
        ("c1cncs1", "S", 5),  # thiazole (S > N)
    ],
)
def test_parametrized_classification(smiles, expected_dominant, expected_ring_size):
    """Parametrized test for heterocycle classification."""
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    info = classify_heterocycle(mol, ring)

    assert info["ring_size"] == expected_ring_size
    assert info["dominant_heteroatom"] == expected_dominant


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_locants",
    [
        ("c1ccncc1", [1]),  # pyridine: N at 1
        ("c1ncncc1", [1, 3]),  # pyrimidine: N at 1, 3
        ("c1cnccn1", [1, 4]),  # pyrazine: N at 1, 4
        ("c1ccnnc1", [1, 2]),  # pyridazine: N at 1, 2
        ("c1ncncn1", [1, 3, 5]),  # 1,3,5-triazine
        ("C1CNCCN1", [1, 4]),  # piperazine: N at 1, 4
        ("C1COCCO1", [1, 4]),  # 1,4-dioxane: O at 1, 4
        ("c1cnc[nH]1", [1, 3]),  # imidazole: N at 1, 3
    ],
)
def test_parametrized_numbering(smiles, expected_locants):
    """Parametrized test for heterocycle numbering."""
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    oriented, _ = orient_heterocycle(mol, ring)
    locants = get_heteroatom_locants(oriented, mol)

    actual_locants = [loc for loc, _ in locants]
    assert actual_locants == expected_locants


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_o_locant,expected_n_locant",
    [
        ("C1COCCN1", 1, 4),  # morpholine
        ("c1cnco1", 1, 3),  # oxazole
        ("c1ccno1", 1, 2),  # isoxazole
    ],
)
def test_parametrized_o_n_priority(smiles, expected_o_locant, expected_n_locant):
    """Parametrized test for O vs N priority in numbering."""
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    oriented, _ = orient_heterocycle(mol, ring)
    locants = get_heteroatom_locants(oriented, mol)
    locant_dict = {elem: loc for loc, elem in locants}

    assert locant_dict["O"] == expected_o_locant
    assert locant_dict["N"] == expected_n_locant


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected_s_locant,expected_n_locant",
    [
        ("c1cncs1", 1, 3),  # thiazole
        ("c1ccns1", 1, 2),  # isothiazole
    ],
)
def test_parametrized_s_n_priority(smiles, expected_s_locant, expected_n_locant):
    """Parametrized test for S vs N priority in numbering."""
    mol = Chem.MolFromSmiles(smiles)
    ring = mol.GetRingInfo().AtomRings()[0]
    oriented, _ = orient_heterocycle(mol, ring)
    locants = get_heteroatom_locants(oriented, mol)
    locant_dict = {elem: loc for loc, elem in locants}

    assert locant_dict["S"] == expected_s_locant
    assert locant_dict["N"] == expected_n_locant
