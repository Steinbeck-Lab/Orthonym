"""Tests for ring substituent detection and classification.

This module tests the ability to correctly identify when a substituent is a ring
(phenyl, cyclohexyl, piperidinyl) versus an alkyl chain (pentyl, hexyl).

The core bug this addresses:
- Before: Ring substituents were named by counting carbons (piperidine -> pentyl)
- After: Ring substituents are correctly identified and named (piperidine -> piperidinyl)
"""
import pytest
from rdkit import Chem
from orthonym.perception.chains import (
    classify_substituent,
    is_ring_substituent,
)
from orthonym.namer import name_compound
from tests.support.hydro_wiring import needs_wiring
from tests.support.rt_assert import assert_full_rt


class TestIsRingSubstituent:
    """Test ring detection in substituent atoms."""

    def test_phenyl_ring_is_detected(self):
        """Phenyl ring should be detected as ring substituent."""
        mol = Chem.MolFromSmiles('c1ccccc1C')  # toluene
        # In toluene, benzene ring atoms are 0-5, methyl carbon is 6
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_atoms = {6}  # methyl is parent (simplified test)
        assert is_ring_substituent(mol, ring_atoms, parent_atoms)

    def test_piperidine_ring_is_detected(self):
        """Piperidine should be detected as ring substituent."""
        mol = Chem.MolFromSmiles('C1CCNCC1C')  # methylpiperidine
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        # Find the methyl carbon (not in ring)
        all_ring = set(ring_atoms)
        methyl_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in all_ring:
                methyl_idx = atom.GetIdx()
                break
        assert is_ring_substituent(mol, ring_atoms, {methyl_idx})

    def test_cyclohexane_ring_is_detected(self):
        """Cyclohexane should be detected as ring substituent."""
        mol = Chem.MolFromSmiles('C1CCCCC1C')  # methylcyclohexane
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        all_ring = set(ring_atoms)
        methyl_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in all_ring:
                methyl_idx = atom.GetIdx()
                break
        assert is_ring_substituent(mol, ring_atoms, {methyl_idx})

    def test_pyrrolidine_ring_is_detected(self):
        """Pyrrolidine (5-membered N heterocycle) should be detected."""
        mol = Chem.MolFromSmiles('C1CCNC1C')  # methylpyrrolidine
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        all_ring = set(ring_atoms)
        methyl_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in all_ring:
                methyl_idx = atom.GetIdx()
                break
        assert is_ring_substituent(mol, ring_atoms, {methyl_idx})

    def test_alkyl_chain_not_ring(self):
        """Alkyl chains should NOT be detected as ring substituents."""
        mol = Chem.MolFromSmiles('CCCCC')  # pentane
        assert not is_ring_substituent(mol, [0, 1, 2], set())

    def test_partial_ring_not_ring_substituent(self):
        """Partial ring atoms should NOT be detected as ring substituent."""
        mol = Chem.MolFromSmiles('C1CCCCC1C')  # methylcyclohexane
        # Only include some ring atoms - not a complete ring
        partial_ring = [0, 1, 2]
        assert not is_ring_substituent(mol, partial_ring, set())

    def test_empty_substituent(self):
        """Empty substituent should return False."""
        mol = Chem.MolFromSmiles('C')  # methane
        assert not is_ring_substituent(mol, [], set())


class TestClassifySubstituent:
    """Test substituent classification as ring vs alkyl."""

    def test_phenyl_classified_as_ring(self):
        """Phenyl substituent should be classified as ring type."""
        mol = Chem.MolFromSmiles('c1ccccc1CCC(=O)O')  # phenylpropanoic acid
        # Benzene ring is atoms 0-5
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        # Parent atoms are the chain: 6, 7, 8, 9, 10
        parent_atoms = set(range(6, mol.GetNumAtoms()))
        result = classify_substituent(mol, ring_atoms, parent_atoms)
        assert result['type'] == 'ring'
        assert 'phenyl' in result['name'].lower()

    def test_cyclohexyl_classified_as_ring(self):
        """Cyclohexyl substituent should be classified as ring type."""
        mol = Chem.MolFromSmiles('C1CCCCC1CCC(=O)O')  # cyclohexylpropanoic acid
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_atoms = set(range(6, mol.GetNumAtoms()))
        result = classify_substituent(mol, ring_atoms, parent_atoms)
        assert result['type'] == 'ring'
        assert 'cyclohexyl' in result['name'].lower()

    def test_piperidine_classified_as_ring(self):
        """Piperidine substituent should be classified as ring type."""
        mol = Chem.MolFromSmiles('C1CCNCC1CCC(=O)O')  # piperidinylpropanoic acid
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_atoms = set(range(6, mol.GetNumAtoms()))
        result = classify_substituent(mol, ring_atoms, parent_atoms)
        assert result['type'] == 'ring'
        # STALE EXPECTATION CORRECTED 2026-07-30: was `'piperidinyl' in...lower`.
        # Production emits the PIN form `piperidin-3-yl` -- requires the free
        # valence to carry its locant, and `ring_substituents.py:508` builds it
        # correctly. Measured: `(piperidin-4-yl)acetic acid`,
        # `3-(piperidin-3-yl)propanoic acid` etc. all OPSIN-clean, so ZERO molecules
        # were mis-named -- only this assertion was.
        # Asserted EXACTLY, not by substring: the loose `'piperid' in...` shape used
        # at line ~178 is why this went unnoticed, since it passes for both spellings.
        assert result['name'] == 'piperidin-3-yl'

    def test_methyl_classified_as_alkyl(self):
        """Methyl substituent should be classified as alkyl."""
        mol = Chem.MolFromSmiles('CC(C)C')  # isobutane
        # Test a single carbon as substituent
        result = classify_substituent(mol, [2], {0, 1, 3})
        assert result['type'] == 'alkyl'
        assert 'methyl' in result['name'].lower()

    def test_ethyl_classified_as_alkyl(self):
        """Ethyl substituent should be classified as alkyl."""
        mol = Chem.MolFromSmiles('CCCC')  # butane
        # atoms 0,1 as substituent, 2,3 as parent
        result = classify_substituent(mol, [0, 1], {2, 3})
        assert result['type'] == 'alkyl'
        assert 'ethyl' in result['name'].lower()

    def test_pentyl_classified_as_alkyl(self):
        """Pentyl chain (not ring) should be classified as alkyl."""
        mol = Chem.MolFromSmiles('CCCCCC')  # hexane
        # atoms 0-4 as substituent (5 carbons)
        result = classify_substituent(mol, [0, 1, 2, 3, 4], {5})
        assert result['type'] == 'alkyl'
        assert 'pentyl' in result['name'].lower()


class TestEndToEndRingSubstituents:
    """E2E tests for correct ring substituent naming."""

    def test_phenylacetic_acid(self):
        """Phenylacetic acid should have phenyl, not hexyl."""
        result = name_compound('c1ccccc1CC(=O)O')
        assert 'hexyl' not in result.lower(), f'Should not contain hexyl, got {result}'
        assert 'phenyl' in result.lower(), f'Should contain phenyl, got {result}'

    def test_phenylpropanoic_acid(self):
        """3-phenylpropanoic acid should have phenyl."""
        result = name_compound('c1ccccc1CCC(=O)O')
        assert 'phenyl' in result.lower(), f'Expected phenyl, got {result}'
        assert 'octyl' not in result.lower(), f'Should not be octyl, got {result}'

    def test_cyclohexylacetic_acid(self):
        """Cyclohexylacetic acid should have cyclohexyl."""
        result = name_compound('C1CCCCC1CC(=O)O')
        assert 'cyclohexyl' in result.lower(), f'Expected cyclohexyl, got {result}'
        # Note: 'hexyl' appears in 'cyclohexyl' so we check for standalone 'hexyl'
        parts = result.lower().split('-')
        for part in parts:
            if 'hexyl' in part and 'cyclohexyl' not in part:
                pytest.fail(f'Should not have standalone hexyl, got {result}')

    @needs_wiring
    def test_piperidinyl_substituent_on_pyridine(self):
        """A piperidine ring on a pyridine ring is detected as a ring, not as pentyl.

        Leads R2A: the PIN is not the substituent name '3-(piperidin-2-yl)pyridine' but the hydro-modified
        ring assembly. (the Blue Book 'UNSATURATION IN RING ASSEMBLIES COMPOSED OF MONOCYCLIC
        MANCUDE AND SATURATED RINGS'),:24153 "When assemblies of otherwise identical rings contain both
        mancude and saturated rings, the use of hydro prefixes is preferred";:24159
        '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)' (not '2-(piperidin-2-yl)pyridine'). The junction
        locants are the lowest, (2,3') before (3,2'),:15567 "Lowest possible locants must be
        used to denote the positions of attachment")."""
        result = name_compound('c1cncc(C2CCCCN2)c1')
        assert 'pentyl' not in result.lower(), f'Should not be pentyl, got {result}'
        assert result == "1,2,3,4,5,6-hexahydro-2,3'-bipyridine", result
        assert_full_rt(result, 'c1cncc(C2CCCCN2)c1')

    def test_cyclopentyl_substituent(self):
        """Cyclopentyl substituent should be named correctly."""
        result = name_compound('C1CCCC1CC(=O)O')  # cyclopentylacetic acid
        assert 'cyclopentyl' in result.lower(), f'Expected cyclopentyl, got {result}'

    def test_pyrrolidinyl_substituent(self):
        """Pyrrolidinyl (5-membered N heterocycle) should be detected."""
        result = name_compound('c1ccc(C2CCCN2)cc1')  # phenyl with pyrrolidine
        # Should contain pyrrolidin or related name, not butyl
        assert 'butyl' not in result.lower(), f'Should not be butyl, got {result}'

    def test_morpholinyl_substituent(self):
        """Morpholinyl substituent should be detected.

        a phase cleanup: removed stale @pytest.mark.xfail. a phase
        retained-name expansion + a phase OPSIN XML import shipped
        morpholine retained-name lookup; the test passes cleanly.
        """
        result = name_compound('c1ccc(C2COCCN2)cc1')  # phenyl with morpholine
        # Should contain morpholin, not be named as alkyl
        assert 'morpholin' in result.lower() or 'oxazin' in result.lower(), \
            f'Expected morpholinyl or similar, got {result}'


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_fused_ring_substituent(self):
        """Test that fused ring systems are handled.

        a phase cleanup: removed stale @pytest.mark.xfail. a phase
        ring-as-substituent naming (naphthalene → naphthalen-1-yl)
        shipped; the test passes cleanly.
        """
        # Naphthalene as substituent
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1CC(=O)O')  # naphthylacetic acid
        result = name_compound('c1ccc2ccccc2c1CC(=O)O')
        # Should recognize naphthyl, not decyl (10 carbons)
        assert 'decyl' not in result.lower(), f'Should not be decyl, got {result}'

    def test_multiple_ring_substituents(self):
        """Test multiple ring substituents on same parent.

        a phase cleanup: removed stale @pytest.mark.xfail. a phase
        IUPAC cascade with locants ships ring-ring parent
        selection; the test passes cleanly.
        """
        result = name_compound('c1ccc(C2CCCCC2)cc1')  # cyclohexylbenzene
        assert 'cyclohexyl' in result.lower() or 'phenyl' in result.lower(), \
            f'Expected ring names, got {result}'

    def test_spiro_like_connection(self):
        """Test ring connected via single atom (not truly spiro)."""
        # This tests a ring substituent connected to another ring
        mol = Chem.MolFromSmiles('C1CCCCC1C2CCCCC2')  # dicyclohexylmethane
        # Each ring should be recognized as ring, not as chain
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        assert len(rings) == 2, "Should have two rings"

    def test_substituent_atoms_must_match_ring(self):
        """Test that all substituent atoms must form the ring."""
        mol = Chem.MolFromSmiles('c1ccccc1CC')  # ethylbenzene
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        # Include ethyl carbon with ring - should still be ring type
        # because ring is complete within atoms
        extended_atoms = ring_atoms + [6]  # add first ethyl carbon
        result = classify_substituent(mol, extended_atoms, {7})
        # Should recognize the ring even with extra atoms
        assert result['type'] == 'ring'


class TestRingTypeIdentification:
    """Test that different ring types are correctly identified."""

    def test_aromatic_carbocycle_benzene(self):
        """Benzene ring gives phenyl."""
        mol = Chem.MolFromSmiles('c1ccccc1C')  # toluene
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        result = classify_substituent(mol, ring_atoms, {6})
        assert result['name'] == 'phenyl'

    def test_saturated_carbocycle_cyclohexane(self):
        """Cyclohexane ring gives cyclohexyl."""
        mol = Chem.MolFromSmiles('C1CCCCC1C')  # methylcyclohexane
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in set(ring_atoms):
                parent_idx = atom.GetIdx()
                break
        result = classify_substituent(mol, ring_atoms, {parent_idx})
        assert result['name'] == 'cyclohexyl'

    def test_saturated_heterocycle_piperidine(self):
        """Piperidine ring gives piperidinyl."""
        mol = Chem.MolFromSmiles('C1CCNCC1C')  # methylpiperidine
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in set(ring_atoms):
                parent_idx = atom.GetIdx()
                break
        result = classify_substituent(mol, ring_atoms, {parent_idx})
        # STALE EXPECTATION CORRECTED 2026-07-30 -- see the note at the piperidine
        # test above. `piperidin-3-yl` is the PIN; 0 molecules were mis-named.
        assert result['name'] == 'piperidin-3-yl'

    def test_aromatic_heterocycle_pyridine(self):
        """Pyridine ring gives the PIN substituent prefix `pyridin-3-yl`.

        Docstring corrected 2026-07-30: it said "gives pyridyl". `pyridyl` is the
        retained/CAS form; requires the free valence to carry its locant, so
        the PIN is `pyridin-3-yl` -- which is what production emits.
        """
        mol = Chem.MolFromSmiles('c1ccncc1C')  # methylpyridine
        ri = mol.GetRingInfo()
        ring_atoms = list(ri.AtomRings()[0])
        parent_idx = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in set(ring_atoms):
                parent_idx = atom.GetIdx()
                break
        result = classify_substituent(mol, ring_atoms, {parent_idx})
        # STALE EXPECTATION CORRECTED 2026-07-30: was
        # `'pyridyl' in name or 'pyridinyl' in name`. Both disjuncts are
        # locant-less and neither is the PIN. Asserted exactly -- a two-way
        # substring disjunct cannot distinguish the PIN from either superseded form.
        assert result['name'] == 'pyridin-3-yl'
