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


class TestMancudeRefused:
    # v25 G5-A: von-Baeyer is non-PIN for mancude/aromatic systems; the engine
    # fail-closes on them (their PIN is a fused/retained parent + added/indicated
    # H, P-25/P-58.2.2, a future G5-B build). Emitting von-Baeyer polyene cages
    # here ships non-PIN strings and — with oxo — invalid names (the caffeine
    # oxo/ene valence-clash class).
    @pytest.mark.parametrize("smi", [
        "c1ccc2ccccc2c1",       # naphthalene
        "c1ccc2[nH]cnc2c1",     # benzimidazole
        "c1ccc2[nH]ccc2c1",     # indole
        "C1CCc2ccccc2C1",       # tetralin (has an aromatic ring)
        "O=c1cc2ccccc2[nH]1",   # a quinolinone-ish mancude oxo system
    ])
    def test_aromatic_cage_refused(self, smi):
        assert analyze_cage_universal(Chem.MolFromSmiles(smi)) is None


class TestSaturatedCageMappingDeterminism:
    # index-mapping + determinism, exercised on a SATURATED cage (decalin) so it
    # survives the G5-A mancude refusal above.
    def test_decalin_maps_to_original_indices(self):
        mol = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
        cage = analyze_cage_universal(mol)
        assert cage is not None and cage.is_mancude is False
        assert set(cage.cage_atoms) == set(range(mol.GetNumHeavyAtoms()))
        assert set(cage.atom_to_locant) == set(cage.cage_atoms)
        assert sorted(cage.atom_to_locant.values()) == list(
            range(1, len(cage.cage_atoms) + 1))

    @pytest.mark.parametrize("spellings", [
        ["C1CCC2CCCCC2C1", "C2CCC1CCCCC1C2", "C1CCC2CCCCC2C1"],
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
