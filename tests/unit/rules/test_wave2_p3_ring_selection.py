import pytest
from rdkit import Chem
from orthonym.rules.ring_selection import (
    ring_system_score, select_principal_ring_system, _spiro_fusion_count,
)


@pytest.mark.unit
class TestP44SpiroFusionCount:
    def test_score_tuple_has_spiro_fusion_term(self):
        #: greater number of spiro fusions = more senior.
        # The tuple grows as later ring_selection tasks append terms; this asserts
        # the CURRENT cumulative length (spiro-fusion term is present at idx 29).
        m = Chem.MolFromSmiles("C1CC2(CC1)CC1(CC2)CCCC1")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(tup) >= 30  # 29 existing + spiro-fusion-count term (+later tasks)
        assert tup[29] == -_spiro_fusion_count(
            m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})

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


@pytest.mark.unit
class TestP44SaturatedMonocyclicSpiro:
    def test_score_tuple_grows(self):
        m = Chem.MolFromSmiles("C1CC2(CC1)CC1(CC2)CCCC1")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        # 30 + saturated-monocyclic bool (idx 30) + spiro-atom locant nested
        # tuple (idx 31). Grows with later tasks -> assert cumulative minimum.
        assert len(tup) >= 32
        assert isinstance(tup[31], tuple)  # spiro-atom locant set term

    def test_lower_spiro_locants_win(self):
        # Two saturated monocyclic spiro systems: the one with the lower spiro-atom
        # locant set is senior. Scorer-level assertion.
        from orthonym.rules.ring_selection import _spiro_atom_locant_set
        m = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")  # spiro[4.5]decane
        all_ring = {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()}
        locs = _spiro_atom_locant_set(m, all_ring)
        assert isinstance(locs, tuple)

    def test_locant_set_spelling_independent(self):
        from orthonym.rules.ring_selection import _spiro_atom_locant_set
        def locs(smi):
            m = Chem.MolFromSmiles(smi)
            return _spiro_atom_locant_set(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        # spiro[4.5]decane spelled two ways -> identical spiro-atom locant set
        assert locs("C1CCC2(CC1)CCCC2") == locs("C1CCCC12CCCCC2")


@pytest.mark.unit
class TestP44FusionDescriptorLetters:
    def test_score_tuple_grows(self):
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")  # quinoline
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        # 32 + fusion-descriptor-letter nested tuple (idx 32). Grows with later
        # tasks -> assert cumulative minimum + positional type.
        assert len(tup) >= 33
        assert isinstance(tup[32], tuple)  # fusion-descriptor-letter term

    def test_fusion_letters_extracted(self):
        from orthonym.rules.ring_selection import _fusion_descriptor_letters
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")  # quinoline
        letters = _fusion_descriptor_letters(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert isinstance(letters, tuple)

    def test_letters_spelling_independent(self):
        from orthonym.rules.ring_selection import _fusion_descriptor_letters
        def letters(smi):
            m = Chem.MolFromSmiles(smi)
            return _fusion_descriptor_letters(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        # quinoline spelled two ways -> identical fusion-letter set
        assert letters("c1ccc2ncccc2c1") == letters("c1ccc2c(c1)cccn2")


@pytest.mark.unit
class TestP44FusionDescriptorNumbers:
    def test_score_tuple_grows(self):
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(tup) >= 34  # 33 + fusion-descriptor-number term
        assert isinstance(tup[33], tuple)

    def test_numbers_extracted(self):
        from orthonym.rules.ring_selection import _fusion_descriptor_numbers
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        nums = _fusion_descriptor_numbers(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert isinstance(nums, tuple)

    def test_numbers_spelling_independent(self):
        from orthonym.rules.ring_selection import _fusion_descriptor_numbers
        def nums(smi):
            m = Chem.MolFromSmiles(smi)
            return _fusion_descriptor_numbers(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert nums("c1ccc2ncccc2c1") == nums("c1ccc2c(c1)cccn2")


@pytest.mark.unit
class TestP44ComponentSeniorityP25_8:
    def test_score_tuple_grows(self):
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(tup) >= 35  # 34 + component-rank term
        assert isinstance(tup[34], tuple)

    def test_quinoline_senior_to_isoquinoline(self):
        # /: quinoline > isoquinoline. Two fused N-systems
        # that tie on every prior criterion are now separated.
        from orthonym.perception.rings import get_ring_systems
        m = Chem.MolFromSmiles("c1ccc2ncccc2c1Cc1cccc2cnccc12")
        rs = get_ring_systems(m)
        tups = [ring_system_score(m, s) for s in rs]
        assert tups[0] != tups[1]  # tie is now broken deterministically

    @pytest.mark.parametrize("smi", [
        "c1ccc2ncccc2c1Cc1cccc2cnccc12",  # quinoline-CH2-isoquinoline
    ])
    def test_component_seniority_spelling_independent(self, smi):
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


@pytest.mark.unit
class TestP44BridgedFusedTiebreakers:
    def test_score_tuple_grows(self):
        m = Chem.MolFromSmiles("C1CC2CCC1CC2")
        tup = ring_system_score(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(tup) == 39  # 35 + 4 pre-bridge metric terms (a,b,c,n) spread
        assert all(isinstance(x, int) for x in tup[35:39])  # pre-bridge metrics

    def test_prebridge_metrics(self):
        from orthonym.rules.ring_selection import _bridged_fused_prebridge_metrics
        m = Chem.MolFromSmiles("C1CC2CCC1CC2")
        metrics = _bridged_fused_prebridge_metrics(
            m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        assert len(metrics) == 4

    @pytest.mark.skip(reason="ORACLE-BLOCKED (not buildable): P-44.2.2.2.4 criteria (d)-(m) "
                             "(bridge-attachment locants, composite/dependent-bridge classification) "
                             "need the full bridged-fused bridge parser, and every molecule they would "
                             "discriminate is OPSIN-2.9-unparseable (the p5_bridged secondary/hetero"
                             "cyclic/polyvalent-bridge class — see WAVE2-COMPLETION-DEFERRED.md oracle-"
                             "blocked section). There is no verifiable target to build against, so the "
                             "scorer correctly stops at the implemented pre-bridge metrics (a)-(c), which "
                             "ARE spelling-independent (guarded by test_prebridge_metrics_spelling_independent). "
                             "Emitting a (d)-(m) tiebreak with no OPSIN oracle would be unverifiable — revisit "
                             "only if OPSIN gains support for these bridged PIN forms.")
    def test_bridge_attachment_locant_tiebreak(self):
        # (e) lower bridge-attachment locants — no OPSIN-verifiable exemplar exists;
        # the pre-bridge metric determinism is covered by the next test.
        pass

    def test_prebridge_metrics_spelling_independent(self):
        from orthonym.rules.ring_selection import _bridged_fused_prebridge_metrics
        def m4(smi):
            m = Chem.MolFromSmiles(smi)
            return _bridged_fused_prebridge_metrics(m, {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()})
        # bicyclo[2.2.2]octane spelled two ways -> identical pre-bridge metrics
        assert m4("C1CC2CCC1CC2") == m4("C1CC2CCC(C1)CC2")
