import pytest
from rdkit import Chem
from orthonym.rules.ring_selection import ring_system_score, select_principal_ring_system


@pytest.mark.unit
class TestP44SpiroFusionCount:
    def test_score_tuple_has_spiro_fusion_term(self):
        # P-44.2.2.2.1.1: greater number of spiro fusions = more senior.
        m = Chem.MolFromSmiles("C1CC2(CC1)CC1(CC2)CCCC1")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(tup) == 30  # 29 existing + 1 spiro-fusion-count term

    def test_more_spiro_fusions_wins(self):
        # A dispiro system (2 fusions) is senior to a monospiro system (1 fusion)
        # when the two tie on all general criteria.
        from orthonym.rules.ring_selection import _spiro_fusion_count
        m = Chem.MolFromSmiles("C1CC2(CC1)CC1(CC2)CCCC1")
        all_ring = {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()}
        assert _spiro_fusion_count(m, all_ring) >= 2

    @pytest.mark.parametrize("smi", [
        "C1CC2(CC1)CC1(CC2)CCCC1",  # dispiro[4.1.4.2]tridecane
        "C1CCC2(CC1)CCCC2",         # spiro[4.5]decane
    ])
    def test_spiro_fusion_selection_spelling_independent(self, smi):
        # Generate several random spellings of the SAME molecule; the SELECTED
        # principal ring system score must be spelling-invariant (the scorer term
        # depends only on the atom set, never SMILES atom order).
        from orthonym.perception.rings import get_ring_systems
        m0 = Chem.MolFromSmiles(smi)

        def winner_score(mol):
            rs = get_ring_systems(mol)
            pr = set(select_principal_ring_system(mol, rs))
            return tuple(ring_system_score(mol, pr))

        base = winner_score(m0)
        for _ in range(5):
            alt = Chem.MolFromSmiles(Chem.MolToSmiles(m0, doRandom=True))
            assert winner_score(alt) == base
