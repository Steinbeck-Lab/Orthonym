# tests/unit/rules/test_vonbaeyer_universal.py
"""v25 G2: universal von-Baeyer cage analysis (aromatic via kekulized enes)."""
import pytest
from rdkit import Chem

from orthonym.rules.vonbaeyer_universal import analyze_cage_universal

pytestmark = pytest.mark.unit


class TestSaturatedCages:
    def test_norbornane(self):
        cage = analyze_cage_universal(Chem.MolFromSmiles("C1CC2CCC1C2"))
        assert cage is not None
        assert cage.descriptor == "bicyclo[2.2.1]"
        assert cage.total_atoms == 7
        assert cage.unsaturation["double_bonds"] == []

    def test_norbornadiene_two_enes(self):
        cage = analyze_cage_universal(Chem.MolFromSmiles("C1=CC2CC1C=C2"))
        assert cage is not None
        assert cage.descriptor == "bicyclo[2.2.1]"
        assert len(cage.unsaturation["double_bonds"]) == 2


class TestAromaticCages:
    def test_naphthalene_five_enes(self):
        cage = analyze_cage_universal(Chem.MolFromSmiles("c1ccc2ccccc2c1"))
        assert cage is not None
        assert cage.descriptor == "bicyclo[4.4.0]"
        assert cage.total_atoms == 10
        # kekulized naphthalene has exactly 5 ring double bonds
        assert len(cage.unsaturation["double_bonds"]) == 5

    def test_benzimidazole_diaza(self):
        cage = analyze_cage_universal(Chem.MolFromSmiles("c1ccc2[nH]cnc2c1"))
        assert cage is not None
        assert cage.descriptor == "bicyclo[4.3.0]"
        assert "diaza" in cage.hetero_prefix
        # kekulized benzimidazole has 4 ring double bonds
        assert len(cage.unsaturation["double_bonds"]) == 4

    def test_indole_maps_to_original_indices(self):
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        cage = analyze_cage_universal(mol)
        assert cage is not None
        assert set(cage.cage_atoms) == set(range(mol.GetNumHeavyAtoms()))
        assert set(cage.atom_to_locant) == set(cage.cage_atoms)
        assert sorted(cage.atom_to_locant.values()) == list(
            range(1, len(cage.cage_atoms) + 1))


class TestDeterminism:
    @pytest.mark.parametrize("spellings", [
        ["c1ccc2ccccc2c1", "c1ccc2c(c1)cccc2", "C1=CC2=CC=CC=C2C=C1"],
        ["c1ccc2[nH]cnc2c1", "c1nc2ccccc2[nH]1"],
    ])
    def test_same_pieces_for_any_spelling(self, spellings):
        results = []
        for smi in spellings:
            cage = analyze_cage_universal(Chem.MolFromSmiles(smi))
            assert cage is not None
            results.append((cage.descriptor, cage.hetero_prefix,
                            tuple(cage.unsaturation["double_bonds"])))
        assert len(set(results)) == 1


class TestRefusals:
    def test_monocycle_refused(self):
        assert analyze_cage_universal(Chem.MolFromSmiles("c1ccccc1")) is None

    def test_spiro_refused(self):
        # spiro[4.4]nonane: shared atom is not 2 bridgeheads
        assert analyze_cage_universal(
            Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")) is None

    def test_acyclic_refused(self):
        assert analyze_cage_universal(Chem.MolFromSmiles("CCCC")) is None
